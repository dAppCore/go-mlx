// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// The vision-chat lane (#98): image-bearing chat turns served through the
// existing decode loop. The ONLY multimodal step is the prefill — projected
// vision features are injected at placeholder positions in one forward —
// after which decode is ordinary text generation, so the lane reuses
// generateTokensFrom (sampler, stops, budget, metrics) with a multimodal
// promptPreparer instead of the text preparePrompt.
//
// Deliberate non-features of this first lane, all perf not correctness:
// no prompt-cache participation (the cache is keyed on token IDs alone —
// identical placeholders with DIFFERENT image bytes would collide), no
// chunked prefill (a placeholder run split across cache-only chunks would
// embed as plain tokens and skip injection), no session routing. A vision
// request pays one full prefill; the APC/vision-LRU follow-ups (#98 list)
// buy the caching back.

package metal

import (
	"context"
	"iter"
	"time"

	core "dappco.re/go"
)

// VisionLanguageModel is the optional capability a family model implements to
// serve image chat turns. gemma4 satisfies it via the SigLIP tower lane.
// Encoding is two-phase — pixels (cheap, yields the soft-token count the
// prompt needs) then projection (the tower forward, the expensive half) —
// so the engine can cache projected features by image content (#99).
type VisionLanguageModel interface {
	InternalModel
	// EncodeImagePixels decodes encoded image bytes (PNG/JPEG) into
	// vision-tower pixels plus the soft-token count the image occupies.
	EncodeImagePixels(data []byte) (*Array, int, error)
	// ProjectImageFeatures runs one image's pixels through the tower +
	// projector; the caller owns the returned features.
	ProjectImageFeatures(pixels *Array) (*Array, error)
	// ImagePlaceholderBlock renders the prompt block that tokenizes to
	// exactly softTokens placeholder IDs.
	ImagePlaceholderBlock(softTokens int) string
	// ImagePlaceholderTokenID is the ID the placeholder block expands to.
	ImagePlaceholderTokenID() int32
	// ForwardImageFeatures is the prefill forward over pre-projected
	// features, BORROWED in placeholder order.
	ForwardImageFeatures(tokens *Array, features []*Array, caches []Cache) *Array
	// AcceptsImageInput reports whether THIS checkpoint shipped the tower
	// (the family supporting vision does not mean the snapshot does).
	AcceptsImageInput() bool
}

// AcceptsImages reports whether the loaded checkpoint can serve image chat
// turns — the serve layer's capability probe.
func (m *Model) AcceptsImages() bool {
	if m == nil || m.model == nil {
		return false
	}
	vlm, ok := m.model.(VisionLanguageModel)
	return ok && vlm.AcceptsImageInput()
}

func chatMessagesCarryImages(messages []ChatMessage) bool {
	for i := range messages {
		for _, img := range messages[i].Images {
			if len(img) > 0 {
				return true
			}
		}
	}
	return false
}

// chatVisionImage is one image's journey through the lane: a cache hit
// arrives with features (request-owned clone); a miss carries pixels until
// the prefill projects them.
type chatVisionImage struct {
	key        visionFeatureKey
	features   *Array
	pixels     *Array
	softTokens int
}

// chatVision serves an image-bearing chat: probe the feature cache / decode
// every image in turn order, splice placeholder blocks ahead of each turn's
// text (the HF processor convention), format with the model's own template,
// verify the tokenizer produced exactly the placeholder count promised, then
// run the standard generation loop over a multimodal prefill. The vision
// tower runs only for cache misses (#99) — a replayed conversation pays the
// SigLIP forward once per unique image, not once per turn.
func (m *Model) chatVision(ctx context.Context, messages []ChatMessage, cfg GenerateConfig) iter.Seq[Token] {
	fail := func(err error) iter.Seq[Token] {
		return func(yield func(Token) bool) {
			m.lastErr = err
			m.lastMetrics = Metrics{}
		}
	}
	vlm, ok := m.model.(VisionLanguageModel)
	if !ok || !vlm.AcceptsImageInput() {
		return fail(core.NewError("mlx: model does not accept image input"))
	}

	cache := m.visionFeatureCacheLazy()
	var images []*chatVisionImage
	freeImages := func() {
		for _, img := range images {
			Free(img.features, img.pixels)
			img.features, img.pixels = nil, nil
		}
	}
	cacheHits, cacheMisses := 0, 0
	totalSoftTokens := 0
	spliced := make([]ChatMessage, len(messages))
	for i, msg := range messages {
		spliced[i] = ChatMessage{Role: msg.Role, Content: msg.Content}
		if len(msg.Images) == 0 {
			continue
		}
		var blocks core.Builder
		for _, data := range msg.Images {
			if len(data) == 0 {
				continue
			}
			img := &chatVisionImage{key: visionFeatureCacheKey(data)}
			if features, softTokens, ok := cache.get(img.key); ok {
				img.features = features
				img.softTokens = softTokens
				cacheHits++
			} else {
				pix, softTokens, err := vlm.EncodeImagePixels(data)
				if err != nil {
					freeImages()
					return fail(core.E("Model.Chat", "encode image", err))
				}
				img.pixels = pix
				img.softTokens = softTokens
				cacheMisses++
			}
			images = append(images, img)
			totalSoftTokens += img.softTokens
			blocks.WriteString(vlm.ImagePlaceholderBlock(img.softTokens))
			blocks.WriteString("\n")
		}
		spliced[i].Content = blocks.String() + msg.Content
	}
	if len(images) == 0 {
		return fail(core.NewError("mlx: image chat carried no decodable images"))
	}

	prompt := m.formatChat(spliced, cfg)
	tokens := m.tokenizer.Encode(prompt)
	placeholderID := vlm.ImagePlaceholderTokenID()
	placeholders := 0
	for _, id := range tokens {
		if id == placeholderID {
			placeholders++
		}
	}
	if placeholders != totalSoftTokens {
		freeImages()
		return fail(core.NewError(core.Sprintf(
			"mlx: tokenizer produced %d image placeholders, want %d — tokenizer and processor config disagree",
			placeholders, totalSoftTokens)))
	}

	prepare := func(ctx context.Context, tokens []int32, cfg GenerateConfig) (PromptPreparation, error) {
		start := time.Now()
		select {
		case <-ctx.Done():
			return PromptPreparation{}, ctx.Err()
		default:
		}
		// Project the misses under the generation stream; hits already
		// hold their features. The cache takes its own Clone — handles
		// are independent, eviction can never pull a live request's
		// buffer.
		features := make([]*Array, len(images))
		for i, img := range images {
			if img.features == nil {
				projected, err := vlm.ProjectImageFeatures(img.pixels)
				if err != nil {
					return PromptPreparation{}, core.E("Model.Chat", "project image features", err)
				}
				Free(img.pixels)
				img.pixels = nil
				img.features = projected
				cache.put(img.key, projected.Clone(), img.softTokens)
			}
			features[i] = img.features
		}
		caches := m.newCachesWithRequestFixedSize(m.generationFixedSlidingCacheSize(len(tokens), cfg.MaxTokens))
		vTokens := FromValues(tokens, len(tokens))
		input := Reshape2(vTokens, 1, int32(len(tokens)))
		logits := vlm.ForwardImageFeatures(input, features, caches)
		Free(vTokens, input)
		if logits == nil || !logits.Valid() {
			_ = LastError()
			Free(logits)
			FreeCaches(caches)
			return PromptPreparation{}, core.NewError("mlx: multimodal prefill returned no logits")
		}
		lastLogits, err := materializeLastTokenLogits(logits)
		if err != nil {
			FreeCaches(caches)
			return PromptPreparation{}, core.E("Model.Chat", "multimodal prefill", err)
		}
		if err := evalCachesBeforeDetach(caches); err != nil {
			Free(lastLogits)
			FreeCaches(caches)
			return PromptPreparation{}, core.E("Model.Chat", "multimodal prefill cache state", err)
		}
		DetachCaches(caches)
		return PromptPreparation{
			Caches:          caches,
			Logits:          lastLogits,
			Duration:        time.Since(start),
			CacheMissTokens: len(tokens),
		}, nil
	}

	return func(yield func(Token) bool) {
		defer freeImages()
		m.lastErr = nil
		m.lastMetrics = Metrics{}
		release, err := m.acquireSlot(ctx)
		if err != nil {
			m.lastErr = err
			return
		}
		defer release()
		if err := m.withDevice(func() {
			if streamErr := m.withGenerationStream(func() {
				if seedErr := applyGenerationSeed(cfg); seedErr != nil {
					m.lastErr = seedErr
					return
				}
				m.generateTokensFrom(ctx, tokens, cfg, prepare)(yield)
			}); streamErr != nil {
				m.lastErr = streamErr
			}
		}); err != nil {
			m.lastErr = err
		}
		// The generation defer wrote lastMetrics; annotate the vision
		// cache outcome on top — the live proof the LRU is working.
		m.lastMetrics.VisionFeatureCacheHits = cacheHits
		m.lastMetrics.VisionFeatureCacheMisses = cacheMisses
	}
}
