// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"encoding/binary"
	"iter"
	"math"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/scheduler"
	infspine "dappco.re/go/inference/spine"
	session "dappco.re/go/inference/state/session"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	_ "dappco.re/go/mlx/pkg/model/gemma4/chat"
	// Register the native model loaders the reactive LoadTokenModelDir dispatches to — the deleted
	// per-arch loaders used to pull these in transitively; the serve layer now imports them explicitly
	// (pkg/native itself stays arch-free).
	"dappco.re/go/mlx/pkg/model/gemma4"
	_ "dappco.re/go/mlx/pkg/model/mistral"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// nativeTextModel exposes the no-cgo native token-loop contract (a model.TokenModel
// + tokenizer) as an inference.TextModel — the sibling of metaladapter (which wraps
// the cgo metal.Model). The OpenAI/Anthropic/Ollama serve handlers drive it with
// ZERO cgo: model.Generate over the contract under the hood (incrementally, since
// NativeTokenModel is a SessionModel — each call opens a persistent-cache session
// and Close frees it). The straight contract path: no prompt cache / MTP / batching
// (those are pkg/metal engine features), so it is the simplest correct serve, the
// proof that the no-cgo stack serves real tokens through the unified contract.
//
// Greedy requests stream from the native session as each token is decoded; sampling
// uses the native sampled session path when available, falling back to the shared
// model sampler loop. Close is a no-op: the resident weights live for the process
// (a single served model), matching the load-once serve shape.
type nativeTextModel struct {
	tm              model.TokenModel
	tok             *tokenizer.Tokenizer
	modelType       string
	modelPath       string
	info            inference.ModelInfo
	maxLen          int
	cacheMode       string
	kvStorageDType  string
	pagedKVPageSize int
	pagedKVPrealloc bool
	adapter         inference.AdapterIdentity
	adapterPack     string
	imageFeatures   *native.VisionImageFeatureConfig
	audioFeatures   *native.AudioFeatureExtractor
	visionCache     *nativeVisionFeatureCache

	mu          sync.Mutex
	lastErr     error
	lastMetrics inference.GenerateMetrics
	cacheSess   nativeTextPromptCacheSession
	cacheBlocks []inference.CacheBlockRef

	schedulerMu sync.Mutex
	scheduler   *scheduler.Model
	probeSink   inference.ProbeSink
	continuity  *ConversationContinuity
}

var _ inference.TextModel = (*nativeTextModel)(nil)

type nativeTextPromptCacheSession interface {
	model.DecodeStepper
	WarmPromptCache([]int32) error
	GenerateCached([]int32, int, int) ([]int32, error)
	ClearPromptCache()
}

type nativeTextPromptCacheSizer interface {
	CachedPrefixLen() int
}

type nativeTextPromptCacheExactSizer interface {
	CachedPrefixLen([]int32) int
}

type nativeTextGreedyStreamSession interface {
	GenerateEach([]int32, int, int, func(int32) bool) ([]int32, error)
}

type nativeTextGreedyTransformStreamSession interface {
	GenerateEachTransformed([]int32, int, int, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextGreedySuppressStreamSession interface {
	GenerateEachWithSuppression([]int32, int, int, []int32, func(int32) bool) ([]int32, error)
}

type nativeTextGreedySuppressTransformStreamSession interface {
	GenerateEachWithSuppressionAndTransform([]int32, int, int, []int32, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextSampledStreamSession interface {
	GenerateSampledEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextSampledOneShotStreamSession interface {
	GenerateSampledOneShotEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextMTPDecodeFunc func(target, draft model.DecodeStepper, prompt []int32, maxNew, eos, k int, yield func(int32) bool) (*native.MTPResult, bool, error)

type nativeTextMTPSampledDecodeFunc func(target, draft model.DecodeStepper, prompt []int32, maxNew int, stopTokens []int32, targetSampler, draftSampler *model.Sampler, params model.SampleParams, k int, yield func(int32) bool) (*native.MTPResult, bool, error)

type nativeTextAssistantDecodeFunc func(pair *native.AssistantPair, target model.DecodeStepper, prompt []int32, maxNew, eos, draftTokens int, suppress []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error)

type nativeTextAssistantSampledDecodeFunc func(pair *native.AssistantPair, target model.DecodeStepper, prompt []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int, yield func(int32) bool) (native.AssistantGenerateResult, bool, error)

var (
	nativeTextMTPDecode nativeTextMTPDecodeFunc = func(target, draft model.DecodeStepper, prompt []int32, maxNew, eos, k int, yield func(int32) bool) (*native.MTPResult, bool, error) {
		targetArch, ok := target.(*native.ArchSession)
		if !ok {
			return nil, false, nil
		}
		draftArch, ok := draft.(*native.ArchSession)
		if !ok {
			return nil, false, nil
		}
		res, err := native.MTPDecodeEach(targetArch, draftArch, prompt, maxNew, eos, k, yield)
		return res, true, err
	}
	nativeTextMTPSampledDecode nativeTextMTPSampledDecodeFunc = func(target, draft model.DecodeStepper, prompt []int32, maxNew int, stopTokens []int32, targetSampler, draftSampler *model.Sampler, params model.SampleParams, k int, yield func(int32) bool) (*native.MTPResult, bool, error) {
		targetArch, ok := target.(*native.ArchSession)
		if !ok {
			return nil, false, nil
		}
		draftArch, ok := draft.(*native.ArchSession)
		if !ok {
			return nil, false, nil
		}
		res, err := native.MTPDecodeSampledEach(targetArch, draftArch, prompt, maxNew, stopTokens, targetSampler, draftSampler, params, k, yield)
		return res, true, err
	}
	nativeTextAssistantDecode nativeTextAssistantDecodeFunc = func(pair *native.AssistantPair, target model.DecodeStepper, prompt []int32, maxNew, eos, draftTokens int, suppress []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		targetArch, ok := target.(*native.ArchSession)
		if !ok {
			return native.AssistantGenerateResult{}, false, nil
		}
		res, err := pair.GenerateFromSessionEach(targetArch, prompt, maxNew, eos, draftTokens, suppress, yield)
		return res, true, err
	}
	nativeTextAssistantSampledDecode nativeTextAssistantSampledDecodeFunc = func(pair *native.AssistantPair, target model.DecodeStepper, prompt []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		targetArch, ok := target.(*native.ArchSession)
		if !ok {
			return native.AssistantGenerateResult{}, false, nil
		}
		res, err := pair.GenerateSampledFromSessionEach(targetArch, prompt, maxNew, stopTokens, sampler, params, draftTokens, yield)
		return res, true, err
	}
)

type nativeTextPromptCacheGreedyStreamSession interface {
	GenerateCachedEach([]int32, int, int, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedyTransformStreamSession interface {
	GenerateCachedEachTransformed([]int32, int, int, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedySuppressStreamSession interface {
	GenerateCachedEachWithSuppression([]int32, int, int, []int32, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedySuppressTransformStreamSession interface {
	GenerateCachedEachWithSuppressionAndTransform([]int32, int, int, []int32, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheSampledStreamSession interface {
	GenerateCachedSampledEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextRootSession interface {
	PrefillTokens([]int32) error
	PrefillTokenEmbeddings([]int32, [][]byte) error
	AppendTokens([]int32) error
	BoundaryLogits() ([]byte, error)
	GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error)
	GenerateFromCacheEachWithSuppressionAndTransform(int, int, []int32, native.TokenTransform, func(int32) bool) ([]int32, error)
	GenerateSampledFromCacheEach(int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
	StateBlockSource(int) (native.SessionStateBlockSource, error)
	StateBlockSourceFrom(int, int) (native.SessionStateBlockSource, error)
	RestoreStateBlocks(native.SessionStateBlockSource) error
	Pos() int
	Close() error
}

type nativeTextSession struct {
	mu              sync.Mutex
	model           *nativeTextModel
	sess            nativeTextRootSession
	tokens          []int32
	generated       []int32
	logits          []byte
	err             error
	prefillDuration time.Duration
	tokenPhases     []metal.TokenPhaseTrace
	closed          bool
}

var _ inference.SessionHandle = (*nativeTextSession)(nil)

type nativeTextImageChatModel interface {
	AcceptsImageInput() bool
	ImagePlaceholderBlock(int) string
	ImagePlaceholderTokenID() int32
	ProjectImageFeatures([]byte) ([]byte, error)
}

type nativeTextImagePixelProjector interface {
	ProjectImagePixels([]float32, int, int) ([]byte, error)
}

type nativeTextAudioChatModel interface {
	AcceptsAudioInput() bool
	AudioPlaceholderBlock(int) string
	AudioPlaceholderTokenID() int32
	ProjectAudioFeatures([]byte, int, int) ([]byte, error)
}

// LoadNativeTextModel loads a gemma4 checkpoint directory as an inference.TextModel
// served entirely without cgo: the no-cgo native contract stack
// (native.LoadTokenModelDir — the reactive registry: dense / MoE / E2B-E4B PLE, 4-bit or bf16) plus
// the tokenizer, behind the standard serve handlers. WithContextLength sizes the KV
// cache (default 4096). The metallib loads at runtime (MLX_METALLIB_PATH or the
// embedded metallib), so the standard lthn-mlx binary serves it — no cgo, no Python.
func LoadNativeTextModel(modelPath string, opts ...LoadOption) (inference.TextModel, error) {
	loadCfg, err := normalizeLoadConfig(applyLoadOptions(opts))
	if err != nil {
		return nil, err
	}
	maxLen := loadCfg.ContextLength
	if maxLen <= 0 {
		maxLen = 4096
	}
	tm, err := native.LoadTokenModelDirWithConfig(modelPath, maxLen, native.TokenModelLoadConfig{
		PagedKVPageSize: loadCfg.PagedKVPageSize,
		PagedKVPrealloc: loadCfg.PagedKVPrealloc,
	})
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, core.E("mlx.LoadNativeTextModel", "load tokenizer", err)
	}
	imageFeatures, err := loadNativeImageFeatureConfig(modelPath)
	if err != nil {
		return nil, core.E("mlx.LoadNativeTextModel", "load image processor", err)
	}
	audioFeatures, err := loadNativeAudioFeatureExtractor(modelPath)
	if err != nil {
		return nil, core.E("mlx.LoadNativeTextModel", "load audio processor", err)
	}
	return &nativeTextModel{
		tm: tm, tok: tok, maxLen: maxLen, cacheMode: nativeTextRuntimeCacheMode(loadCfg), kvStorageDType: loadCfg.KVCacheStorageDType,
		pagedKVPageSize: loadCfg.PagedKVPageSize, pagedKVPrealloc: loadCfg.PagedKVPrealloc,
		modelType: "gemma4", modelPath: modelPath, imageFeatures: imageFeatures, audioFeatures: audioFeatures,
		info: inference.ModelInfo{Architecture: "gemma4", VocabSize: tm.Vocab()},
	}, nil
}

func nativeTextRuntimeCacheMode(cfg LoadConfig) string {
	return "paged"
}

func loadNativeImageFeatureConfig(modelPath string) (*native.VisionImageFeatureConfig, error) {
	imageCfg, _, err := gemma4.LoadGemma4ImageFeatureConfigs(modelPath)
	if err != nil || imageCfg == nil {
		return nil, err
	}
	return &native.VisionImageFeatureConfig{
		PatchSize:         imageCfg.PatchSize,
		MaxSoftTokens:     imageCfg.MaxSoftTokens,
		PoolingKernelSize: imageCfg.PoolingKernelSize,
		RescaleFactor:     imageCfg.RescaleFactor,
		DoResize:          imageCfg.DoResize,
		DoConvertRGB:      imageCfg.DoConvertRGB,
	}, nil
}

func loadNativeAudioFeatureExtractor(modelPath string) (*native.AudioFeatureExtractor, error) {
	audioCfg, err := native.LoadAudioFeatureConfig(modelPath)
	if err != nil || audioCfg == nil {
		return nil, err
	}
	return native.NewAudioFeatureExtractor(audioCfg)
}

// Generate streams tokens for a raw prompt (no chat template — Chat applies that).
func (m *nativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream(ctx, m.tok.Encode(prompt), inference.ApplyGenerateOpts(opts))
}

// Chat streams tokens from a multi-turn conversation rendered with the gemma turn
// template (user/model turns, a trailing model turn to complete).
func (m *nativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	if m != nil && m.continuity != nil {
		if seq, ok := m.continuity.Chat(ctx, messages, opts...); ok {
			return seq
		}
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if inferenceMessagesCarryImages(messages) {
		return m.chatVision(ctx, messages, cfg)
	}
	return m.stream(ctx, m.tok.Encode(m.formatChat(messages, cfg)), cfg)
}

func (m *nativeTextModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if m == nil {
		return "", errMLXModelNil
	}
	return m.formatChat(messages, inference.GenerateConfig{}), nil
}

// FormatChatPrompt renders a conversation opening in the model's chat
// template, including the generation header — the native no-cgo lane's
// counterpart to mlx.Model.FormatChatPrompt (chat_config.go). The -state CLI
// turn loop (cmd/mlx generate.go) drives whichever lane loaded the model
// through this shared method name.
func (m *nativeTextModel) FormatChatPrompt(messages []inference.Message) string {
	return m.formatChatTurns(messages, inference.GenerateConfig{}, false)
}

// FormatChatContinuation renders messages as an append to a session whose
// retained state ends inside an open model turn: the family template closes
// that turn, renders only the new turns, and reopens the generation header.
// The native no-cgo lane's counterpart to mlx.Model.FormatChatContinuation.
func (m *nativeTextModel) FormatChatContinuation(messages []inference.Message) string {
	return m.formatChatTurns(messages, inference.GenerateConfig{}, true)
}

// FormatChatPromptThinking is FormatChatPrompt with an explicit thinking
// override (nil = model default) — the native counterpart to
// mlx.Model.FormatChatPromptThinking, so the -state CLI's -think flag
// reaches the template on this lane too.
func (m *nativeTextModel) FormatChatPromptThinking(messages []inference.Message, thinking *bool) string {
	return m.formatChatTurns(messages, inference.GenerateConfig{EnableThinking: thinking}, false)
}

// FormatChatContinuationThinking is FormatChatContinuation with an explicit
// thinking override (nil = model default).
func (m *nativeTextModel) FormatChatContinuationThinking(messages []inference.Message, thinking *bool) string {
	return m.formatChatTurns(messages, inference.GenerateConfig{EnableThinking: thinking}, true)
}

func (m *nativeTextModel) Encode(text string) []int32 {
	if m == nil || m.tok == nil {
		return nil
	}
	return m.tok.Encode(text)
}

func (m *nativeTextModel) Decode(ids []int32) string {
	if m == nil || m.tok == nil {
		return ""
	}
	return m.tok.Decode(ids)
}

func (m *nativeTextModel) AcceptsImages() bool {
	_, ok := m.imageChatModel()
	return ok && m.imageFeatures != nil
}

func (m *nativeTextModel) AcceptsAudioInput() bool {
	_, ok := m.audioChatModel()
	return ok && m.audioFeatures != nil
}

func (m *nativeTextModel) errorStream(err error) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {
		m.setErr(err)
	}
}

func (m *nativeTextModel) chatVision(ctx context.Context, messages []inference.Message, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		if err := ctx.Err(); err != nil {
			m.setErr(err)
			return
		}
		if m == nil || m.tok == nil || m.tm == nil {
			m.setErr(core.NewError("mlx.nativeTextModel.Chat: model is not initialised"))
			return
		}
		imageModel, ok := m.imageChatModel()
		if !ok {
			m.setErr(core.NewError("mlx.nativeTextModel.Chat: model does not accept image input"))
			return
		}
		if m.imageFeatures == nil {
			m.setErr(core.NewError("mlx.nativeTextModel.Chat: image feature config is nil"))
			return
		}
		start := time.Now()
		spliced := make([]inference.Message, len(messages))
		var features [][]byte
		totalSoftTokens := 0
		cache := m.nativeVisionFeatureCacheLazy()
		for i, msg := range messages {
			spliced[i] = inference.Message{Role: msg.Role, Content: msg.Content}
			if len(msg.Images) == 0 {
				continue
			}
			var blocks core.Builder
			for _, data := range msg.Images {
				if len(data) == 0 {
					continue
				}
				key := nativeVisionFeatureCacheKey(data)
				projected, softTokens, ok := cache.get(key)
				if !ok {
					if pixelModel, ok := imageModel.(nativeTextImagePixelProjector); ok {
						pixels, h, w, st, err := native.VisionImagePixels(data, m.imageFeatures)
						if err != nil {
							m.setErr(core.E("mlx.nativeTextModel.Chat", "encode image", err))
							return
						}
						projected, err = pixelModel.ProjectImagePixels(pixels, int(h), int(w))
						if err != nil {
							m.setErr(core.E("mlx.nativeTextModel.Chat", "project image pixels", err))
							return
						}
						softTokens = st
					} else {
						patches, st, err := native.VisionImagePatches(data, m.imageFeatures)
						if err != nil {
							m.setErr(core.E("mlx.nativeTextModel.Chat", "encode image", err))
							return
						}
						projected, err = imageModel.ProjectImageFeatures(patches)
						if err != nil {
							m.setErr(core.E("mlx.nativeTextModel.Chat", "project image features", err))
							return
						}
						softTokens = st
					}
					cache.put(key, projected, softTokens)
				}
				features = append(features, projected)
				totalSoftTokens += softTokens
				blocks.WriteString(imageModel.ImagePlaceholderBlock(softTokens))
				blocks.WriteString("\n")
			}
			spliced[i].Content = blocks.String() + msg.Content
		}
		if totalSoftTokens == 0 {
			m.setErr(core.NewError("mlx.nativeTextModel.Chat: image chat carried no decodable images"))
			return
		}
		ids := m.tok.Encode(m.formatChat(spliced, cfg))
		placeholderID := imageModel.ImagePlaceholderTokenID()
		placeholders := 0
		for _, id := range ids {
			if id == placeholderID {
				placeholders++
			}
		}
		if placeholders != totalSoftTokens {
			m.setErr(core.NewError(core.Sprintf(
				"mlx.nativeTextModel.Chat: tokenizer produced %d image placeholders, want %d",
				placeholders, totalSoftTokens)))
			return
		}
		embeddings, err := m.nativeImagePromptEmbeddings(ids, placeholderID, features)
		if err != nil {
			m.setErr(err)
			return
		}
		sess, err := m.openNativeRootSession()
		if err != nil {
			m.setErr(err)
			return
		}
		defer func() { _ = sess.Close() }()
		if err := sess.PrefillTokenEmbeddings(ids, embeddings); err != nil {
			m.setErr(err)
			return
		}
		prefill := time.Since(start)
		decodeStart := time.Now()
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || len(ids)+maxNew > m.maxLen {
			maxNew = m.maxLen - len(ids)
		}
		if maxNew <= 0 {
			m.setErr(core.NewError("mlx.nativeTextModel.Chat: prompt fills the context window, no room to generate"))
			return
		}
		stopTokens := m.stopTokens(cfg)
		eos := singleStopToken(stopTokens)
		emit := func(id int32) bool {
			if err := ctx.Err(); err != nil {
				return false
			}
			if yield != nil && !yield(inference.Token{ID: id, Text: m.tok.DecodeToken(id)}) {
				return false
			}
			return !tokenInSet(id, stopTokens)
		}
		var out []int32
		sampled := cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1
		minStopActive := cfg.MinTokensBeforeStop > 0 && len(stopTokens) > 0
		switch {
		case sampled:
			params := model.SampleParams{
				Temperature:         cfg.Temperature,
				TopK:                cfg.TopK,
				TopP:                cfg.TopP,
				MinP:                cfg.MinP,
				SuppressTokens:      cfg.SuppressTokens,
				MinTokensBeforeStop: cfg.MinTokensBeforeStop,
				RepeatPenalty:       cfg.RepeatPenalty,
			}
			out, err = sess.GenerateSampledFromCacheEach(maxNew, stopTokens, model.NewSampler(nativeSamplerSeed(cfg)), params, nil, emit)
		case minStopActive:
			prefix := cfg.MinTokensBeforeStop
			if prefix > maxNew {
				prefix = maxNew
			}
			prefixSuppress := nativeTextMergeSuppressTokens(cfg.SuppressTokens, stopTokens)
			out, err = sess.GenerateFromCacheEachWithSuppressionAndTransform(prefix, eos, prefixSuppress, nil, emit)
			if err == nil && len(out) == prefix && len(out) < maxNew {
				var more []int32
				if len(cfg.SuppressTokens) > 0 {
					more, err = sess.GenerateFromCacheEachWithSuppressionAndTransform(maxNew-len(out), eos, cfg.SuppressTokens, nil, emit)
				} else {
					more, err = sess.GenerateFromCacheEach(maxNew-len(out), eos, emit)
				}
				out = append(out, more...)
			}
		case len(cfg.SuppressTokens) > 0:
			out, err = sess.GenerateFromCacheEachWithSuppressionAndTransform(maxNew, eos, cfg.SuppressTokens, nil, emit)
		default:
			out, err = sess.GenerateFromCacheEach(maxNew, eos, emit)
		}
		if err != nil {
			m.setErr(err)
			return
		}
		if err := ctx.Err(); err != nil {
			m.setErr(err)
			return
		}
		m.setSessionMetrics(len(ids), len(out), prefill, time.Since(decodeStart))
	}
}

func (m *nativeTextModel) nativeImagePromptEmbeddings(ids []int32, placeholderID int32, features [][]byte) ([][]byte, error) {
	return m.nativeModalityPromptEmbeddings(ids, placeholderID, features, "image")
}

func (m *nativeTextModel) nativeAudioPromptEmbeddings(ids []int32, placeholderID int32, features [][]byte) ([][]byte, error) {
	return m.nativeModalityPromptEmbeddings(ids, placeholderID, features, "audio")
}

type nativeTextPromptEmbedInto interface {
	EmbeddingBytes() int
	EmbedInto([]byte, int32) ([]byte, error)
}

func nativeTextPromptEmbeddingRows(source any, ids []int32, embed func(int32) ([]byte, error), scope string) ([][]byte, error) {
	if fast, ok := source.(nativeTextPromptEmbedInto); ok {
		rowBytes := fast.EmbeddingBytes()
		if rowBytes > 0 {
			rows := make([][]byte, len(ids))
			storage := make([]byte, len(ids)*rowBytes)
			for i, id := range ids {
				row := storage[i*rowBytes : (i+1)*rowBytes]
				emb, err := fast.EmbedInto(row, id)
				if err != nil {
					return nil, err
				}
				if len(emb) != rowBytes {
					return nil, core.NewError(scope + ": token embedding is empty")
				}
				rows[i] = emb
			}
			return rows, nil
		}
	}
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := embed(id)
		if err != nil {
			return nil, err
		}
		if len(emb) == 0 {
			return nil, core.NewError(scope + ": token embedding is empty")
		}
		rows[i] = emb
	}
	return rows, nil
}

func (m *nativeTextModel) nativeModalityPromptEmbeddings(ids []int32, placeholderID int32, features [][]byte, label string) ([][]byte, error) {
	embeddings, err := nativeTextPromptEmbeddingRows(m.tm, ids, m.tm.Embed, "mlx.nativeTextModel.Chat")
	if err != nil {
		return nil, err
	}
	featureIdx, featureOff, used := 0, 0, 0
	for i, id := range ids {
		if id != placeholderID {
			continue
		}
		rowBytes := len(embeddings[i])
		for featureIdx < len(features) && featureOff >= len(features[featureIdx]) {
			featureIdx++
			featureOff = 0
		}
		if featureIdx >= len(features) || featureOff+rowBytes > len(features[featureIdx]) {
			return nil, core.NewError("mlx.nativeTextModel.Chat: " + label + " feature rows do not match placeholder embeddings")
		}
		embeddings[i] = features[featureIdx][featureOff : featureOff+rowBytes]
		featureOff += rowBytes
		used++
	}
	if used == 0 {
		return nil, core.NewError("mlx.nativeTextModel.Chat: no " + label + " placeholders found")
	}
	for featureIdx < len(features) {
		if featureOff < len(features[featureIdx]) {
			return nil, core.NewError("mlx.nativeTextModel.Chat: unused " + label + " feature rows")
		}
		featureIdx++
		featureOff = 0
	}
	return embeddings, nil
}

func (m *nativeTextModel) nativeVisionFeatureCacheLazy() *nativeVisionFeatureCache {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.visionCache == nil {
		m.visionCache = newNativeVisionFeatureCache(0)
	}
	return m.visionCache
}

func (m *nativeTextModel) imageChatModel() (nativeTextImageChatModel, bool) {
	if m == nil || m.tm == nil {
		return nil, false
	}
	imageModel, ok := m.tm.(nativeTextImageChatModel)
	return imageModel, ok && imageModel.AcceptsImageInput()
}

func (m *nativeTextModel) audioChatModel() (nativeTextAudioChatModel, bool) {
	if m == nil || m.tm == nil {
		return nil, false
	}
	audioModel, ok := m.tm.(nativeTextAudioChatModel)
	return audioModel, ok && audioModel.AcceptsAudioInput()
}

// GenerateChunks streams tokens for a prompt supplied as bounded text chunks.
// Each chunk is tokenized independently and appended into one logical prompt
// stream, matching the metal chunk path without first joining the full prompt.
func (m *nativeTextModel) GenerateChunks(ctx context.Context, chunks iter.Seq[string], opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		ids, err := m.encodePromptChunks(ctx, chunks, "mlx.nativeTextModel.GenerateChunks")
		if err != nil {
			m.setErr(err)
			return
		}
		for tok := range m.stream(ctx, ids, cfg) {
			if !yield(tok) {
				return
			}
		}
	}
}

// ChatChunks formats a chat prompt and feeds it through the chunked native
// generation path, preserving the same concatenated prompt as Chat.
func (m *nativeTextModel) ChatChunks(ctx context.Context, messages []inference.Message, chunkBytes int, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return m.GenerateChunks(ctx, nativeTextPromptChunks(m.formatChat(messages, cfg), chunkBytes), opts...)
}

func (m *nativeTextModel) formatChat(messages []inference.Message, cfg inference.GenerateConfig) string {
	return m.formatChatTurns(messages, cfg, false)
}

func (m *nativeTextModel) formatChatTurns(messages []inference.Message, cfg inference.GenerateConfig, continuation bool) string {
	chatCfg := m.chatConfig(cfg)
	chatCfg.Continuation = continuation
	return chat.Format(messages, chatCfg)
}

func (m *nativeTextModel) chatConfig(cfg inference.GenerateConfig) chat.Config {
	info := m.Info()
	architecture := info.Architecture
	if architecture == "" {
		architecture = m.modelType
	}
	if architecture == "" {
		architecture = "gemma4"
	}
	numHeads := 0
	if m != nil && m.tm != nil {
		if reporter, ok := m.tm.(interface{ NumQueryHeads() int }); ok {
			numHeads = reporter.NumQueryHeads()
		}
	}
	out := chat.ConfigForArchitecture(architecture, numHeads)
	if cfg.EnableThinking != nil {
		out.EnableThinking = *cfg.EnableThinking
	}
	if m != nil && m.tm != nil {
		if suppressor, ok := m.tm.(interface{ NeedsThoughtChannelSuppressor() bool }); ok {
			out.LargeVariant = suppressor.NeedsThoughtChannelSuppressor()
		}
	}
	return out
}

// formatGemmaChat renders messages in the gemma turn format. gemma has no system
// role, so system/user fold to "user" and assistant to "model"; a trailing model
// turn opens the completion.
func formatGemmaChat(messages []inference.Message) string {
	out := "<bos>"
	for _, msg := range messages {
		role := "user"
		if msg.Role == "assistant" {
			role = "model"
		}
		out += core.Sprintf("<start_of_turn>%s\n%s<end_of_turn>\n", role, msg.Content)
	}
	return out + "<start_of_turn>model\n"
}

func nativeGemmaChatChunks(messages []inference.Message, chunkBytes int) iter.Seq[string] {
	return nativeTextPromptChunks(formatGemmaChat(messages), chunkBytes)
}

func nativeTextPromptChunks(prompt string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if chunkBytes <= 0 {
			yield(prompt)
			return
		}
		for i := 0; i < len(prompt); i += chunkBytes {
			end := i + chunkBytes
			if end > len(prompt) {
				end = len(prompt)
			}
			if !yield(prompt[i:end]) {
				return
			}
		}
	}
}

func (m *nativeTextModel) stream(ctx context.Context, ids []int32, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		if m.BlockDiffusionCapable() {
			if m.streamBlockDiffusion(ctx, ids, cfg, start, yield) {
				return
			}
			m.setErr(core.NewError("mlx.nativeTextModel: block-diffusion native model requires the block-diffusion sampler; refusing autoregressive fallback"))
			return
		}
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || len(ids)+maxNew > m.maxLen {
			maxNew = m.maxLen - len(ids)
		}
		if maxNew <= 0 {
			m.setErr(core.NewError("mlx.nativeTextModel: prompt fills the context window, no room to generate"))
			return
		}
		stopTokens := m.stopTokens(cfg)
		eos := singleStopToken(stopTokens)
		suppressTokens := cfg.SuppressTokens
		needsSuppress := len(suppressTokens) > 0
		budgetTracker := newNativeThinkingBudgetTracker(m.tok, cfg)
		var nativeTransform native.TokenTransform
		var modelTransform model.TokenTransform
		if budgetTracker != nil {
			nativeTransform = native.TokenTransform(budgetTracker.observe)
			modelTransform = model.TokenTransform(budgetTracker.observe)
		}
		budgetForced := func() bool {
			return budgetTracker != nil && budgetTracker.forcedClose()
		}
		emit := func(id int32) bool {
			if ctx.Err() != nil {
				return false
			}
			if !yield(inference.Token{ID: id, Text: m.tok.DecodeToken(id)}) {
				return false
			}
			return !tokenInSet(id, stopTokens)
		}
		var (
			out      []int32
			err      error
			streamed bool
		)
		repeatPenaltyActive := cfg.RepeatPenalty > 1
		minStopActive := cfg.MinTokensBeforeStop > 0 && len(stopTokens) > 0
		sampled := cfg.Temperature > 0 || repeatPenaltyActive
		if !sampled {
			m.mu.Lock()
			cacheSess := m.cacheSess
			if cacheSess != nil {
				if minStopActive {
					if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedySuppressTransformStreamSession); ok {
						streamed = true
						out, err = nativeTextGenerateCachedGreedyMinStop(streamSess, ids, maxNew, eos, cfg.MinTokensBeforeStop, suppressTokens, stopTokens, nativeTransform, emit)
					} else {
						cacheSess = nil
					}
				} else if needsSuppress || nativeTransform != nil {
					if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedySuppressTransformStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEachWithSuppressionAndTransform(ids, maxNew, eos, suppressTokens, nativeTransform, emit)
					} else if needsSuppress && nativeTransform == nil {
						if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedySuppressStreamSession); ok {
							streamed = true
							out, err = streamSess.GenerateCachedEachWithSuppression(ids, maxNew, eos, suppressTokens, emit)
						} else {
							cacheSess = nil
						}
					} else if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedyTransformStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEachTransformed(ids, maxNew, eos, nativeTransform, emit)
					} else {
						cacheSess = nil
					}
				} else {
					if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedyStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEach(ids, maxNew, eos, emit)
					} else {
						out, err = cacheSess.GenerateCached(ids, maxNew, eos)
					}
				}
			}
			m.mu.Unlock()
			if cacheSess != nil {
				if err != nil {
					m.setErr(err)
					return
				}
				if !streamed {
					emitted := 0
					for _, id := range out {
						emitted++
						if !emit(id) {
							m.setMetricsWithThinkingBudget(len(ids), emitted, time.Since(start), budgetForced())
							return
						}
					}
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
		}
		if sampled || minStopActive { // stochastic, or greedy with logits-side policy
			sampler := model.NewSampler(nativeSamplerSeed(cfg))
			sampleParams := model.SampleParams{Temperature: cfg.Temperature, TopK: cfg.TopK, TopP: cfg.TopP, MinP: cfg.MinP, SuppressTokens: suppressTokens, MinTokensBeforeStop: cfg.MinTokensBeforeStop, RepeatPenalty: cfg.RepeatPenalty}
			m.mu.Lock()
			cacheSess := m.cacheSess
			if cacheSess != nil {
				if streamSess, ok := cacheSess.(nativeTextPromptCacheSampledStreamSession); ok {
					streamed = true
					out, err = streamSess.GenerateCachedSampledEach(ids, maxNew, stopTokens, sampler, sampleParams, modelTransform, emit)
				} else {
					cacheSess = nil
				}
			}
			m.mu.Unlock()
			if cacheSess != nil {
				if err != nil {
					m.setErr(err)
					return
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
			if out, err, streamed = m.streamSampledSession(ids, maxNew, stopTokens, sampler, sampleParams, modelTransform, emit); streamed {
				if err != nil {
					m.setErr(err)
					return
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
			streamed = true // the sampled path streams per token (parity with greedy + pkg/metal)
			out, err = model.GenerateSampledWithStopTokensTransformEach(m.tm, sampler, sampleParams, ids, maxNew, stopTokens, modelTransform, emit)
		} else if out, err, streamed = m.streamGreedySession(ids, maxNew, eos, suppressTokens, nativeTransform, emit); streamed {
			if err != nil {
				m.setErr(err)
				return
			}
			m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
			return
		} else if needsSuppress || modelTransform != nil {
			sampler := model.NewSampler(nativeSamplerSeed(cfg))
			streamed = true // greedy-with-suppression/transform also streams per token
			out, err = model.GenerateSampledWithStopTokensTransformEach(m.tm, sampler, model.SampleParams{MinP: cfg.MinP, SuppressTokens: suppressTokens, MinTokensBeforeStop: cfg.MinTokensBeforeStop}, ids, maxNew, stopTokens, modelTransform, emit)
		} else {
			out, err = model.Generate(m.tm, ids, maxNew, eos)
		}
		if err != nil {
			m.setErr(err)
			return
		}
		if !streamed { // batch paths emit here; the streaming paths already emitted via emit
			emitted := 0
			for _, id := range out {
				emitted++
				if !emit(id) {
					m.setMetricsWithThinkingBudget(len(ids), emitted, time.Since(start), budgetForced())
					return
				}
			}
		}
		m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
	}
}

func (m *nativeTextModel) streamBlockDiffusion(ctx context.Context, ids []int32, cfg inference.GenerateConfig, start time.Time, yield func(inference.Token) bool) bool {
	bd, ok := m.tm.(native.BlockDiffusionTokenGenerator)
	if !ok {
		return false
	}
	if ctx == nil {
		ctx = context.Background()
	}
	maxNew := cfg.MaxTokens
	if maxNew <= 0 || len(ids)+maxNew > m.maxLen {
		maxNew = m.maxLen - len(ids)
	}
	if maxNew <= 0 {
		m.setErr(core.NewError("mlx.nativeTextModel: prompt fills the context window, no room to generate"))
		return true
	}
	stopTokens := m.stopTokens(cfg)
	emitted := 0
	opts := native.BlockDiffusionOptions{
		MaxTokens:   maxNew,
		Temperature: cfg.Temperature,
		Seed:        cfg.Seed,
		SeedSet:     cfg.SeedSet,
		StopTokens:  append([]int32(nil), stopTokens...),
	}
	metrics, err := bd.GenerateBlockDiffusionTokens(ctx, ids, opts, func(id int32) bool {
		if ctx.Err() != nil || emitted >= maxNew {
			return false
		}
		emitted++
		if yield != nil && !yield(inference.Token{ID: id, Text: m.tok.DecodeToken(id)}) {
			return false
		}
		if tokenInSet(id, stopTokens) {
			return false
		}
		return emitted < maxNew
	})
	if err != nil {
		m.setErr(err)
		return true
	}
	if err := ctx.Err(); err != nil {
		m.setErr(err)
		return true
	}
	promptTokens := metrics.PrefillTokens
	if promptTokens <= 0 {
		promptTokens = len(ids)
	}
	decodeDur := metrics.DenoiseDur + metrics.CommitDur
	if decodeDur <= 0 && metrics.TotalDur > metrics.PrefillDur {
		decodeDur = metrics.TotalDur - metrics.PrefillDur
	}
	if metrics.PrefillDur == 0 && decodeDur == 0 {
		m.setMetrics(promptTokens, emitted, time.Since(start))
		return true
	}
	m.setSessionMetrics(promptTokens, emitted, metrics.PrefillDur, decodeDur)
	return true
}

func (m *nativeTextModel) stopTokens(cfg inference.GenerateConfig) []int32 {
	if len(cfg.StopTokens) > 0 {
		return cfg.StopTokens
	}
	if m != nil && m.tok != nil && m.tok.HasEOSToken() {
		return []int32{m.tok.EOSToken()}
	}
	return nil
}

func singleStopToken(tokens []int32) int {
	if len(tokens) == 1 {
		return int(tokens[0])
	}
	return -1
}

func tokenInSet(id int32, tokens []int32) bool {
	for _, token := range tokens {
		if id == token {
			return true
		}
	}
	return false
}

func nativeTextMergeSuppressTokens(base, extra []int32) []int32 {
	if len(extra) == 0 {
		return base
	}
	if len(base) == 0 {
		return extra
	}
	merged := append([]int32(nil), base...)
	for _, token := range extra {
		if !tokenInSet(token, merged) {
			merged = append(merged, token)
		}
	}
	return merged
}

func nativeTextPromptWithGenerated(prompt, generated []int32) []int32 {
	if len(generated) == 0 {
		return prompt
	}
	out := make([]int32, 0, len(prompt)+len(generated))
	out = append(out, prompt...)
	out = append(out, generated...)
	return out
}

func nativeTextGenerateCachedGreedyMinStop(streamSess nativeTextPromptCacheGreedySuppressTransformStreamSession, ids []int32, maxNew, eos, minTokens int, suppressTokens, stopTokens []int32, transform native.TokenTransform, emit func(int32) bool) ([]int32, error) {
	prefix := minTokens
	if prefix > maxNew {
		prefix = maxNew
	}
	keepGoing := true
	stageEmit := func(id int32) bool {
		if !keepGoing {
			return false
		}
		if emit == nil {
			return true
		}
		keepGoing = emit(id)
		return keepGoing
	}
	prefixSuppress := nativeTextMergeSuppressTokens(suppressTokens, stopTokens)
	out, err := streamSess.GenerateCachedEachWithSuppressionAndTransform(ids, prefix, eos, prefixSuppress, transform, stageEmit)
	if err != nil || !keepGoing || len(out) < prefix || len(out) >= maxNew {
		return out, err
	}
	nextIDs := nativeTextPromptWithGenerated(ids, out)
	var more []int32
	if len(suppressTokens) == 0 && transform == nil {
		if plainSess, ok := streamSess.(nativeTextPromptCacheGreedyStreamSession); ok {
			more, err = plainSess.GenerateCachedEach(nextIDs, maxNew-len(out), eos, stageEmit)
		} else {
			more, err = streamSess.GenerateCachedEachWithSuppressionAndTransform(nextIDs, maxNew-len(out), eos, nil, nil, stageEmit)
		}
	} else {
		more, err = streamSess.GenerateCachedEachWithSuppressionAndTransform(nextIDs, maxNew-len(out), eos, suppressTokens, transform, stageEmit)
	}
	out = append(out, more...)
	return out, err
}

func nativeSamplerSeed(cfg inference.GenerateConfig) uint64 {
	if cfg.SeedSet {
		return cfg.Seed
	}
	return uint64(time.Now().UnixNano())
}

func (m *nativeTextModel) streamGreedySession(ids []int32, maxNew, eos int, suppress []int32, transform native.TokenTransform, yield func(int32) bool) ([]int32, error, bool) {
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, nil, false
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err, true
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	needsSuppress := len(suppress) > 0
	if needsSuppress || transform != nil {
		if streamSess, ok := sess.(nativeTextGreedySuppressTransformStreamSession); ok {
			out, err := streamSess.GenerateEachWithSuppressionAndTransform(ids, maxNew, eos, suppress, transform, yield)
			return out, err, true
		}
		if needsSuppress && transform == nil {
			if streamSess, ok := sess.(nativeTextGreedySuppressStreamSession); ok {
				out, err := streamSess.GenerateEachWithSuppression(ids, maxNew, eos, suppress, yield)
				return out, err, true
			}
			return nil, nil, false
		}
		if streamSess, ok := sess.(nativeTextGreedyTransformStreamSession); ok {
			out, err := streamSess.GenerateEachTransformed(ids, maxNew, eos, transform, yield)
			return out, err, true
		}
		return nil, nil, false
	}
	streamSess, ok := sess.(nativeTextGreedyStreamSession)
	if !ok {
		return nil, nil, false
	}
	out, err := streamSess.GenerateEach(ids, maxNew, eos, yield)
	return out, err, true
}

func (m *nativeTextModel) streamSampledSession(ids []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error, bool) {
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, nil, false
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err, true
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	if streamSess, ok := sess.(nativeTextSampledOneShotStreamSession); ok {
		out, err := streamSess.GenerateSampledOneShotEach(ids, maxNew, stopTokens, sampler, params, transform, yield)
		return out, err, true
	}
	streamSess, ok := sess.(nativeTextSampledStreamSession)
	if !ok {
		return nil, nil, false
	}
	out, err := streamSess.GenerateSampledEach(ids, maxNew, stopTokens, sampler, params, transform, yield)
	return out, err, true
}

func (m *nativeTextModel) GenerateNativeSpeculative(ctx context.Context, draftModel *nativeTextModel, prompt string, cfg SpeculativeDecodeConfig) (SpeculativeDecodeResult, bool, error) {
	return m.GenerateNativeSpeculativeEach(ctx, draftModel, prompt, cfg, nil)
}

func (m *nativeTextModel) GenerateNativeAssistant(ctx context.Context, pair *native.AssistantPair, prompt string, cfg GenerateConfig, draftTokens int) (SpeculativeDecodeResult, bool, error) {
	return m.GenerateNativeAssistantEach(ctx, pair, prompt, cfg, draftTokens, nil)
}

func (m *nativeTextModel) GenerateNativeAssistantEach(ctx context.Context, pair *native.AssistantPair, prompt string, cfg GenerateConfig, draftTokens int, yield func(decode.Token) bool) (SpeculativeDecodeResult, bool, error) {
	if m == nil || m.tm == nil || m.tok == nil {
		return SpeculativeDecodeResult{}, false, nil
	}
	if pair == nil {
		return SpeculativeDecodeResult{}, true, core.NewError("mlx.nativeTextModel.GenerateNativeAssistant: assistant pair is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SpeculativeDecodeResult{}, true, err
	}
	generateCfg := cfg
	if generateCfg.MaxTokens == 0 {
		generateCfg = DefaultGenerateConfig()
	}
	maxNew := generateCfg.MaxTokens
	if maxNew <= 0 {
		return SpeculativeDecodeResult{}, false, nil
	}
	if draftTokens <= 0 {
		draftTokens = MTPDefaultDraftTokens
	}
	promptIDs := m.tok.Encode(prompt)
	if len(promptIDs) == 0 {
		return SpeculativeDecodeResult{}, true, core.NewError("mlx.nativeTextModel.GenerateNativeAssistant: empty prompt after tokenisation")
	}
	targetSess, closeTarget, promptCacheTarget, err := m.nativeTextAssistantTargetSession(promptIDs)
	if err != nil {
		return SpeculativeDecodeResult{}, true, err
	}
	if targetSess == nil {
		return SpeculativeDecodeResult{}, false, nil
	}
	if closeTarget != nil {
		defer closeTarget()
	}

	stopTokens := generateCfg.StopTokens
	if len(stopTokens) == 0 && m.tok.HasEOSToken() {
		stopTokens = []int32{m.tok.EOSToken()}
	}
	start := time.Now()
	var yieldID func(int32) bool
	if yield != nil {
		yieldID = func(id int32) bool {
			return yield(decode.Token{ID: id, Text: m.tok.DecodeToken(id)})
		}
	}
	var (
		res     native.AssistantGenerateResult
		handled bool
	)
	if nativeTextMTPPlainGreedy(generateCfg, stopTokens) {
		res, handled, err = nativeTextAssistantDecode(pair, targetSess, promptIDs, maxNew, singleStopToken(stopTokens), draftTokens, generateCfg.SuppressTokens, yieldID)
	} else {
		params := model.SampleParams{
			Temperature:         generateCfg.Temperature,
			TopK:                generateCfg.TopK,
			TopP:                generateCfg.TopP,
			MinP:                generateCfg.MinP,
			SuppressTokens:      generateCfg.SuppressTokens,
			MinTokensBeforeStop: generateCfg.MinTokensBeforeStop,
			RepeatPenalty:       generateCfg.RepeatPenalty,
		}
		res, handled, err = nativeTextAssistantSampledDecode(pair, targetSess, promptIDs, maxNew, stopTokens, model.NewSampler(nativeTextSpeculativeSamplerSeed(generateCfg)), params, draftTokens, yieldID)
	}
	if err != nil || !handled {
		return SpeculativeDecodeResult{}, handled, err
	}
	if promptCacheTarget {
		m.rememberNativePromptCacheBlock(promptIDs)
	}
	return m.nativeTextAssistantResult(prompt, res, time.Since(start)), true, nil
}

func (m *nativeTextModel) nativeTextAssistantTargetSession(promptIDs []int32) (model.DecodeStepper, func(), bool, error) {
	m.mu.Lock()
	if m.cacheSess != nil {
		if sizer, ok := m.cacheSess.(nativeTextPromptCacheExactSizer); ok && sizer.CachedPrefixLen(promptIDs) > 0 {
			sess := m.cacheSess
			m.mu.Unlock()
			return sess, nil, true, nil
		}
	}
	m.mu.Unlock()

	targetSM, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, nil, false, nil
	}
	targetSess, err := targetSM.OpenSession()
	if err != nil {
		return nil, nil, false, err
	}
	var closeTarget func()
	if c, ok := targetSess.(interface{ Close() error }); ok {
		closeTarget = func() { _ = c.Close() }
	}
	return targetSess, closeTarget, false, nil
}

func (m *nativeTextModel) rememberNativePromptCacheBlock(promptIDs []int32) {
	if len(promptIDs) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	var labels map[string]string
	if len(m.cacheBlocks) > 0 {
		labels = nativeCacheLabels(m.cacheBlocks[0].Labels)
	}
	m.cacheBlocks = nativePromptCacheBlocks(promptIDs, labels)
}

func (m *nativeTextModel) GenerateNativeSpeculativeEach(ctx context.Context, draftModel *nativeTextModel, prompt string, cfg SpeculativeDecodeConfig, yield func(decode.Token) bool) (SpeculativeDecodeResult, bool, error) {
	if m == nil || m.tm == nil || m.tok == nil {
		return SpeculativeDecodeResult{}, false, nil
	}
	if draftModel == nil || draftModel.tm == nil {
		return SpeculativeDecodeResult{}, false, nil
	}
	targetSM, ok := m.tm.(model.SessionModel)
	if !ok {
		return SpeculativeDecodeResult{}, false, nil
	}
	draftSM, ok := draftModel.tm.(model.SessionModel)
	if !ok {
		return SpeculativeDecodeResult{}, false, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SpeculativeDecodeResult{}, true, err
	}
	generateCfg := cfg.GenerateConfig
	if generateCfg.MaxTokens == 0 {
		generateCfg = DefaultGenerateConfig()
	}
	maxNew := cfg.MaxTokens
	if maxNew <= 0 {
		maxNew = generateCfg.MaxTokens
	}
	if maxNew <= 0 {
		return SpeculativeDecodeResult{}, false, nil
	}
	k := cfg.DraftTokens
	if k <= 0 {
		k = MTPDefaultDraftTokens
	}
	promptIDs := m.tok.Encode(prompt)
	if len(promptIDs) == 0 {
		return SpeculativeDecodeResult{}, true, core.NewError("mlx.nativeTextModel.GenerateNativeSpeculative: empty prompt after tokenisation")
	}
	targetSess, err := targetSM.OpenSession()
	if err != nil {
		return SpeculativeDecodeResult{}, true, err
	}
	if c, ok := targetSess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	draftSess, err := draftSM.OpenSession()
	if err != nil {
		return SpeculativeDecodeResult{}, true, err
	}
	if c, ok := draftSess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}

	stopTokens := generateCfg.StopTokens
	if len(stopTokens) == 0 && m.tok.HasEOSToken() {
		stopTokens = []int32{m.tok.EOSToken()}
	}
	start := time.Now()
	var (
		res     *native.MTPResult
		handled bool
	)
	var yieldID func(int32) bool
	if yield != nil {
		yieldID = func(id int32) bool {
			return yield(decode.Token{ID: id, Text: m.tok.DecodeToken(id)})
		}
	}
	if nativeTextMTPPlainGreedy(generateCfg, stopTokens) {
		res, handled, err = nativeTextMTPDecode(targetSess, draftSess, promptIDs, maxNew, singleStopToken(stopTokens), k, yieldID)
	} else {
		seed := nativeTextSpeculativeSamplerSeed(generateCfg)
		params := model.SampleParams{
			Temperature:         generateCfg.Temperature,
			TopK:                generateCfg.TopK,
			TopP:                generateCfg.TopP,
			MinP:                generateCfg.MinP,
			SuppressTokens:      generateCfg.SuppressTokens,
			MinTokensBeforeStop: generateCfg.MinTokensBeforeStop,
			RepeatPenalty:       generateCfg.RepeatPenalty,
		}
		res, handled, err = nativeTextMTPSampledDecode(targetSess, draftSess, promptIDs, maxNew, stopTokens, model.NewSampler(seed), model.NewSampler(seed+0x9e3779b97f4a7c15), params, k, yieldID)
	}
	if err != nil || !handled {
		return SpeculativeDecodeResult{}, handled, err
	}
	return m.nativeTextMTPResult(prompt, res, time.Since(start)), true, nil
}

func nativeTextMTPPlainGreedy(cfg GenerateConfig, stopTokens []int32) bool {
	return cfg.Temperature <= 0 &&
		cfg.TopK <= 0 &&
		cfg.TopP <= 0 &&
		cfg.MinP <= 0 &&
		len(cfg.SuppressTokens) == 0 &&
		cfg.MinTokensBeforeStop <= 0 &&
		cfg.RepeatPenalty <= 1 &&
		len(stopTokens) <= 1
}

func nativeTextSpeculativeSamplerSeed(cfg GenerateConfig) uint64 {
	if cfg.SeedSet {
		return cfg.Seed
	}
	return uint64(time.Now().UnixNano())
}

func (m *nativeTextModel) nativeTextMTPResult(prompt string, res *native.MTPResult, duration time.Duration) SpeculativeDecodeResult {
	if res == nil {
		return SpeculativeDecodeResult{Mode: SpeculativeDecodeModeMTP, Prompt: prompt}
	}
	tokens := make([]decode.Token, len(res.Tokens))
	var text core.Builder
	for i, id := range res.Tokens {
		piece := m.tok.DecodeToken(id)
		tokens[i].ID = id
		tokens[i].Text = piece
		text.WriteString(piece)
	}
	rejected := res.Drafted - res.Accepted
	if rejected < 0 {
		rejected = 0
	}
	var acceptanceRate float64
	if res.Drafted > 0 {
		acceptanceRate = float64(res.Accepted) / float64(res.Drafted)
	}
	return SpeculativeDecodeResult{
		Mode:   SpeculativeDecodeModeMTP,
		Prompt: prompt,
		Text:   text.String(),
		Tokens: tokens,
		Metrics: decode.Metrics{
			TargetTokens:   len(res.Tokens),
			DraftTokens:    res.Drafted,
			AcceptedTokens: res.Accepted,
			RejectedTokens: rejected,
			EmittedTokens:  len(res.Tokens),
			AcceptanceRate: acceptanceRate,
			TargetCalls:    res.Rounds,
			DraftCalls:     res.Rounds,
			Duration:       duration,
		},
	}
}

func (m *nativeTextModel) nativeTextAssistantResult(prompt string, res native.AssistantGenerateResult, duration time.Duration) SpeculativeDecodeResult {
	tokens := make([]decode.Token, len(res.Tokens))
	var text core.Builder
	for i, id := range res.Tokens {
		piece := m.tok.DecodeToken(id)
		tokens[i].ID = id
		tokens[i].Text = piece
		text.WriteString(piece)
	}
	var acceptanceRate float64
	if res.DraftTokens > 0 {
		acceptanceRate = float64(res.AcceptedTokens) / float64(res.DraftTokens)
	}
	return SpeculativeDecodeResult{
		Mode:   SpeculativeDecodeModeMTP,
		Prompt: prompt,
		Text:   text.String(),
		Tokens: tokens,
		Metrics: decode.Metrics{
			TargetTokens:   res.TargetTokens,
			DraftTokens:    res.DraftTokens,
			AcceptedTokens: res.AcceptedTokens,
			RejectedTokens: res.RejectedTokens,
			EmittedTokens:  len(res.Tokens),
			AcceptanceRate: acceptanceRate,
			TargetCalls:    res.TargetCalls,
			DraftCalls:     res.DraftCalls,
			Duration:       duration,
		},
	}
}

func (m *nativeTextModel) WarmPromptCache(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ids := m.tok.Encode(prompt)
	m.mu.Lock()
	defer m.mu.Unlock()
	cacheSess, err := m.promptCacheSessionLocked()
	if err != nil {
		m.lastErr = err
		return err
	}
	if err := cacheSess.WarmPromptCache(ids); err != nil {
		m.lastErr = err
		return err
	}
	m.cacheBlocks = nativePromptCacheBlocks(ids, nil)
	m.lastErr = nil
	return ctx.Err()
}

func (m *nativeTextModel) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheStats{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cacheStatsLocked(nil), ctx.Err()
}

func (m *nativeTextModel) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheWarmResult{}, err
	}
	ids := append([]int32(nil), req.Tokens...)
	if len(ids) == 0 && req.Prompt != "" {
		ids = m.tok.Encode(req.Prompt)
	}
	labels := nativeCacheLabels(req.Labels)

	m.mu.Lock()
	defer m.mu.Unlock()
	if len(ids) > 0 {
		cacheSess, err := m.promptCacheSessionLocked()
		if err != nil {
			m.lastErr = err
			return inference.CacheWarmResult{}, err
		}
		if err := cacheSess.WarmPromptCache(ids); err != nil {
			m.lastErr = err
			return inference.CacheWarmResult{}, err
		}
	}
	if len(ids) > 0 {
		m.cacheBlocks = nativePromptCacheBlocks(ids, labels)
	}
	m.lastErr = nil
	stats := m.cacheStatsLocked(labels)
	result := inference.CacheWarmResult{
		Stats:  stats,
		Labels: labels,
	}
	if len(ids) > 0 {
		result.Blocks = nativeCloneCacheBlocks(m.cacheBlocks)
	}
	return result, ctx.Err()
}

func (m *nativeTextModel) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	entries := m.cacheEntriesLocked(labels)
	return entries, ctx.Err()
}

func (m *nativeTextModel) WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ids, err := m.encodePromptChunks(ctx, chunks, "mlx.nativeTextModel.WarmPromptCacheChunks")
	if err != nil {
		m.setErr(err)
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	cacheSess, err := m.promptCacheSessionLocked()
	if err != nil {
		m.lastErr = err
		return err
	}
	if err := cacheSess.WarmPromptCache(ids); err != nil {
		m.lastErr = err
		return err
	}
	m.cacheBlocks = nativePromptCacheBlocks(ids, nil)
	m.lastErr = nil
	return ctx.Err()
}

func (m *nativeTextModel) ClearPromptCache() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cacheSess != nil {
		m.cacheSess.ClearPromptCache()
	}
	m.cacheBlocks = nil
}

func (m *nativeTextModel) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheStats{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(labels) > 0 && !m.cacheHasEntryMatchingLabelsLocked(labels) {
		return m.cacheStatsLocked(labels), ctx.Err()
	}
	if m.cacheSess != nil {
		m.cacheSess.ClearPromptCache()
	}
	m.cacheBlocks = nil
	return m.cacheStatsLocked(labels), ctx.Err()
}

func (m *nativeTextModel) CaptureKV(ctx context.Context, prompt string) (*metal.KVSnapshot, error) {
	return m.CaptureKVWithOptions(ctx, prompt, metal.KVSnapshotCaptureOptions{})
}

func (m *nativeTextModel) CaptureKVWithOptions(ctx context.Context, prompt string, opts metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	handle, err := m.newOneShotNativeSession("mlx.nativeTextModel.CaptureKV")
	if err != nil {
		return nil, err
	}
	defer func() { _ = handle.Close() }()
	if err := handle.Prefill(ctx, prompt); err != nil {
		return nil, err
	}
	snapshot, err := handle.CaptureKVWithOptions(ctx, kv.CaptureOptions{RawKVOnly: opts.RawKVOnly, BlockStartToken: opts.BlockStartToken})
	if err != nil {
		return nil, err
	}
	return kvconv.ToMetalKVSnapshot(snapshot), nil
}

func (m *nativeTextModel) CaptureKVChunks(ctx context.Context, chunks iter.Seq[string]) (*metal.KVSnapshot, error) {
	return m.CaptureKVChunksWithOptions(ctx, chunks, metal.KVSnapshotCaptureOptions{})
}

func (m *nativeTextModel) CaptureKVChunksWithOptions(ctx context.Context, chunks iter.Seq[string], opts metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	handle, err := m.newOneShotNativeSession("mlx.nativeTextModel.CaptureKVChunks")
	if err != nil {
		return nil, err
	}
	defer func() { _ = handle.Close() }()
	if err := handle.PrefillChunks(ctx, chunks); err != nil {
		return nil, err
	}
	snapshot, err := handle.CaptureKVWithOptions(ctx, kv.CaptureOptions{RawKVOnly: opts.RawKVOnly, BlockStartToken: opts.BlockStartToken})
	if err != nil {
		return nil, err
	}
	return kvconv.ToMetalKVSnapshot(snapshot), nil
}

func (m *nativeTextModel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	snapshot, err := m.CaptureKVWithOptions(ctx, prompt, metal.KVSnapshotCaptureOptions{RawKVOnly: true})
	if err != nil {
		return nil, err
	}
	return nativeTextAttentionSnapshotFromKV(snapshot)
}

func nativeTextAttentionSnapshotFromKV(snapshot *metal.KVSnapshot) (*inference.AttentionSnapshot, error) {
	if snapshot == nil {
		return nil, nil
	}
	keys := make([][][]float32, len(snapshot.Layers))
	for i, layer := range snapshot.Layers {
		stateLayer, err := nativeTextStateLayerFromMetal(layer, snapshot.SeqLen)
		if err != nil {
			return nil, err
		}
		if len(stateLayer.KeyBytes) == 0 {
			continue
		}
		keyHeads, err := nativeTextAttentionKeysFromLayer(stateLayer, snapshot.SeqLen)
		if err != nil {
			return nil, err
		}
		keys[i] = keyHeads
	}
	return &inference.AttentionSnapshot{
		NumLayers:     snapshot.NumLayers,
		NumHeads:      snapshot.NumHeads,
		SeqLen:        snapshot.SeqLen,
		HeadDim:       snapshot.HeadDim,
		NumQueryHeads: snapshot.NumQueryHeads,
		Keys:          keys,
		Architecture:  snapshot.Architecture,
	}, nil
}

func nativeTextAttentionKeysFromLayer(layer native.SessionStateLayerBlock, tokenCount int) ([][]float32, error) {
	if tokenCount < 0 || layer.KVHeads <= 0 || layer.HeadDim <= 0 {
		return nil, core.NewError("mlx.nativeTextModel.InspectAttention: invalid native KV geometry")
	}
	wantBytes := tokenCount * layer.KVHeads * layer.HeadDim * 2
	if len(layer.KeyBytes) != wantBytes {
		return nil, core.NewError("mlx.nativeTextModel.InspectAttention: native key slab size mismatch")
	}
	keys := make([][]float32, layer.KVHeads)
	for head := 0; head < layer.KVHeads; head++ {
		key := make([]float32, tokenCount*layer.HeadDim)
		for token := 0; token < tokenCount; token++ {
			srcBase := (token*layer.KVHeads + head) * layer.HeadDim * 2
			dstBase := token * layer.HeadDim
			for dim := 0; dim < layer.HeadDim; dim++ {
				off := srcBase + dim*2
				key[dstBase+dim] = math.Float32frombits(uint32(uint16(layer.KeyBytes[off])|uint16(layer.KeyBytes[off+1])<<8) << 16)
			}
		}
		keys[head] = key
	}
	return keys, nil
}

func (m *nativeTextModel) RestorePromptCacheFromKV(ctx context.Context, snapshot *metal.KVSnapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	source, _, _, _, err := nativeTextStateSourceFromSnapshot(snapshot)
	if err != nil {
		m.setErr(err)
		return err
	}
	return m.restorePromptCacheStateSource(ctx, source, "mlx.nativeTextModel.RestorePromptCacheFromKV")
}

func (m *nativeTextModel) RestorePromptCacheFromKVBlocks(ctx context.Context, source metal.KVSnapshotBlockSource) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	restored, _, _, err := nativeTextStateSourceFromBlockSource(ctx, source, nil)
	if err != nil {
		m.setErr(err)
		return err
	}
	return m.restorePromptCacheStateSource(ctx, restored, "mlx.nativeTextModel.RestorePromptCacheFromKVBlocks")
}

func (m *nativeTextModel) restorePromptCacheStateSource(ctx context.Context, source native.SessionStateBlockSource, scope string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	cacheSess, err := m.promptCacheSessionLocked()
	if err != nil {
		m.lastErr = err
		return err
	}
	restorer, ok := cacheSess.(interface {
		RestoreStateBlocks(native.SessionStateBlockSource) error
	})
	if !ok {
		err := core.NewError(scope + ": native session cannot restore KV prompt cache")
		m.lastErr = err
		return err
	}
	if err := restorer.RestoreStateBlocks(source); err != nil {
		m.lastErr = err
		return err
	}
	m.lastErr = nil
	return ctx.Err()
}

func (m *nativeTextModel) newOneShotNativeSession(scope string) (*nativeTextSession, error) {
	sess, err := m.openNativeRootSession()
	if err != nil {
		if scope != "" {
			return nil, core.E(scope, "open session", err)
		}
		return nil, err
	}
	return &nativeTextSession{model: m, sess: sess}, nil
}

func (m *nativeTextModel) cacheStatsLocked(labels map[string]string) inference.CacheStats {
	stats := inference.CacheStats{
		CacheMode: "native-prompt",
		Labels:    nativeCacheLabels(labels),
	}
	if len(m.cacheBlocks) > 0 {
		stats.Blocks = len(m.cacheBlocks)
		return stats
	}
	if m.cacheSess == nil {
		return stats
	}
	if sizer, ok := m.cacheSess.(nativeTextPromptCacheSizer); ok {
		if sizer.CachedPrefixLen() > 0 {
			stats.Blocks = 1
		}
		return stats
	}
	stats.Blocks = 1
	return stats
}

func (m *nativeTextModel) cacheEntriesLocked(labels map[string]string) []inference.CacheBlockRef {
	entries := m.cacheBlocks
	if len(entries) == 0 {
		if prefixLen := m.cachedPromptPrefixLenLocked(); prefixLen > 0 {
			entries = nativePromptCacheBlocks(make([]int32, prefixLen), nil)
		}
	}
	if len(entries) == 0 {
		return nil
	}
	out := make([]inference.CacheBlockRef, 0, len(entries))
	for _, entry := range entries {
		if len(labels) > 0 && !nativeCacheBlockMatchesLabels(entry, labels) {
			continue
		}
		out = append(out, nativeCloneCacheBlock(entry))
	}
	return out
}

func (m *nativeTextModel) cacheHasEntryMatchingLabelsLocked(labels map[string]string) bool {
	if len(labels) == 0 {
		return true
	}
	for _, entry := range m.cacheEntriesLocked(nil) {
		if nativeCacheBlockMatchesLabels(entry, labels) {
			return true
		}
	}
	return false
}

func (m *nativeTextModel) cachedPromptPrefixLenLocked() int {
	if m.cacheSess == nil {
		return 0
	}
	if sizer, ok := m.cacheSess.(nativeTextPromptCacheSizer); ok {
		return sizer.CachedPrefixLen()
	}
	return 0
}

func nativePromptCacheBlocks(ids []int32, labels map[string]string) []inference.CacheBlockRef {
	if len(ids) == 0 {
		return nil
	}
	return []inference.CacheBlockRef{{
		ID:         "native-prompt",
		Kind:       "prompt",
		TokenStart: 0,
		TokenCount: len(ids),
		Labels:     nativeCacheLabels(labels),
	}}
}

func nativeCloneCacheBlocks(blocks []inference.CacheBlockRef) []inference.CacheBlockRef {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]inference.CacheBlockRef, len(blocks))
	for i, block := range blocks {
		out[i] = nativeCloneCacheBlock(block)
	}
	return out
}

func nativeCloneCacheBlock(block inference.CacheBlockRef) inference.CacheBlockRef {
	block.Labels = nativeCacheLabels(block.Labels)
	return block
}

func nativeCacheBlockMatchesLabels(block inference.CacheBlockRef, labels map[string]string) bool {
	for key, want := range labels {
		switch key {
		case "model_hash":
			if block.ModelHash != want {
				return false
			}
		case "adapter_hash":
			if block.AdapterHash != want {
				return false
			}
		case "tokenizer_hash":
			if block.TokenizerHash != want {
				return false
			}
		default:
			if block.Labels[key] != want {
				return false
			}
		}
	}
	return true
}

func nativeCacheLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}

func (m *nativeTextModel) encodePromptChunks(ctx context.Context, chunks iter.Seq[string], scope string) ([]int32, error) {
	return m.encodePromptChunksWithPrefix(ctx, chunks, scope, false)
}

func (m *nativeTextModel) encodePromptChunksWithPrefix(ctx context.Context, chunks iter.Seq[string], scope string, seenContent bool) ([]int32, error) {
	if m == nil || m.tok == nil {
		return nil, core.NewError("mlx.nativeTextModel: tokenizer is nil")
	}
	if chunks == nil {
		return nil, core.NewError("mlx.nativeTextModel: prompt chunks are nil")
	}
	if scope == "" {
		scope = "mlx.nativeTextModel.GenerateChunks"
	}
	tokens := make([]int32, 0, 256)
	for chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if chunk == "" {
			continue
		}
		ids := m.tok.Encode(chunk)
		if seenContent {
			ids = stripNativeImplicitChunkBOS(m.tok, ids)
		}
		if len(ids) == 0 {
			continue
		}
		tokens = append(tokens, ids...)
		seenContent = true
	}
	if len(tokens) == 0 {
		return nil, core.NewError(scope + ": empty prompt after tokenisation")
	}
	return tokens, nil
}

func stripNativeImplicitChunkBOS(tok *tokenizer.Tokenizer, ids []int32) []int32 {
	if tok == nil || !tok.HasBOSToken() || len(ids) == 0 {
		return ids
	}
	if ids[0] != tok.BOSToken() {
		return ids
	}
	return ids[1:]
}

func (m *nativeTextModel) promptCacheSessionLocked() (nativeTextPromptCacheSession, error) {
	if m.cacheSess != nil {
		return m.cacheSess, nil
	}
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, core.NewError("mlx.nativeTextModel: prompt cache requires a session model")
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err
	}
	cacheSess, ok := sess.(nativeTextPromptCacheSession)
	if !ok {
		if c, closeOK := sess.(interface{ Close() error }); closeOK {
			_ = c.Close()
		}
		return nil, core.NewError("mlx.nativeTextModel: native session does not support prompt cache")
	}
	m.cacheSess = cacheSess
	return cacheSess, nil
}

func (m *nativeTextModel) NewSession() inference.SessionHandle {
	sess, err := m.openNativeRootSession()
	if err != nil {
		return nil
	}
	return &nativeTextSession{model: m, sess: sess}
}

func (m *nativeTextModel) enableConversationContinuity(opts ConversationContinuityOptions) (*ConversationContinuity, error) {
	if m == nil {
		return nil, core.E("mlx.EnableConversationContinuity", "model is nil", nil)
	}
	continuity, err := newConversationContinuityForTarget("mlx.EnableConversationContinuity", nativeTextContinuityTarget{model: m}, opts)
	if err != nil {
		return nil, err
	}
	m.continuity = continuity
	return continuity, nil
}

type nativeTextContinuityTarget struct {
	model *nativeTextModel
}

func (t nativeTextContinuityTarget) NewContinuitySession() (*ModelSession, error) {
	if t.model == nil {
		return nil, errMLXModelNil
	}
	handle := t.model.NewSession()
	if handle == nil {
		return nil, errNativeNilSession
	}
	return session.New(handle, inferenceSpineModelInfo(nativeTextContinuityModelInfo(t.model)), infspine.NewTokenizer(t.model.tok)), nil
}

func (t nativeTextContinuityTarget) FormatContinuityChat(messages []inference.Message, thinking *bool, continuation bool) string {
	if t.model == nil {
		return ""
	}
	return t.model.formatChatTurns(messages, inference.GenerateConfig{EnableThinking: thinking}, continuation)
}

func (t nativeTextContinuityTarget) ContinuityNative() any {
	return t.model
}

func nativeTextContinuityModelInfo(m *nativeTextModel) ModelInfo {
	if m == nil {
		return ModelInfo{}
	}
	info := m.Info()
	return ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: m.maxLen,
	}
}

func (m *nativeTextModel) openNativeRootSession() (nativeTextRootSession, error) {
	if m == nil || m.tm == nil {
		return nil, core.NewError("mlx.nativeTextModel.NewSession: model is not initialised")
	}
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, core.NewError("mlx.nativeTextModel.NewSession: token model does not support sessions")
	}
	stepper, err := sm.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(nativeTextRootSession)
	if !ok {
		if c, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = c.Close()
		}
		return nil, core.NewError("mlx.nativeTextModel.NewSession: native session does not support state blocks")
	}
	return sess, nil
}

func (s *nativeTextSession) Prefill(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.Prefill"); err != nil {
		s.err = err
		return err
	}
	if s.model.tok == nil {
		err := core.NewError("mlx.nativeTextSession.Prefill: tokenizer is nil")
		s.err = err
		return err
	}
	ids := s.model.tok.Encode(prompt)
	return s.prefillTokensLocked(ctx, ids)
}

func (s *nativeTextSession) PrefillChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.PrefillChunks"); err != nil {
		s.err = err
		return err
	}
	ids, err := s.model.encodePromptChunks(ctx, chunks, "mlx.nativeTextSession.PrefillChunks")
	if err != nil {
		s.err = err
		return err
	}
	return s.prefillTokensLocked(ctx, ids)
}

func (s *nativeTextSession) PrefillTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.PrefillTokens"); err != nil {
		s.err = err
		return err
	}
	return s.prefillTokensLocked(ctx, tokens)
}

func (s *nativeTextSession) prefillTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.PrefillTokens: empty prompt tokens")
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	start := time.Now()
	ids := append([]int32(nil), tokens...)
	if err := s.sess.PrefillTokens(ids); err != nil {
		s.err = err
		return err
	}
	logits, err := s.sess.BoundaryLogits()
	if err != nil {
		s.err = err
		return err
	}
	s.tokens = ids
	s.generated = nil
	s.logits = append(s.logits[:0], logits...)
	s.prefillDuration = time.Since(start)
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) AppendPrompt(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.AppendPrompt"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendPrompt: no retained prefix")
		s.err = err
		return err
	}
	if s.model.tok == nil {
		err := core.NewError("mlx.nativeTextSession.AppendPrompt: tokenizer is nil")
		s.err = err
		return err
	}
	ids := stripNativeImplicitChunkBOS(s.model.tok, s.model.tok.Encode(prompt))
	return s.appendTokensLocked(ctx, ids)
}

func (s *nativeTextSession) AppendPromptChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.AppendPromptChunks"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendPromptChunks: no retained prefix")
		s.err = err
		return err
	}
	ids, err := s.model.encodePromptChunksWithPrefix(ctx, chunks, "mlx.nativeTextSession.AppendPromptChunks", true)
	if err != nil {
		s.err = err
		return err
	}
	return s.appendTokensLocked(ctx, ids)
}

func (s *nativeTextSession) AppendTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.AppendTokens"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendTokens: no retained prefix")
		s.err = err
		return err
	}
	return s.appendTokensLocked(ctx, stripNativeImplicitChunkBOS(s.model.tok, tokens))
}

func (s *nativeTextSession) appendTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendTokens: empty prompt tokens")
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	start := time.Now()
	ids := append([]int32(nil), tokens...)
	if err := s.sess.AppendTokens(ids); err != nil {
		s.err = err
		return err
	}
	logits, err := s.sess.BoundaryLogits()
	if err != nil {
		s.err = err
		return err
	}
	s.tokens = append(s.tokens, ids...)
	s.generated = nil
	s.logits = append(s.logits[:0], logits...)
	s.prefillDuration += time.Since(start)
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) Generate(ctx context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		totalStart := time.Now()
		if ctx == nil {
			ctx = context.Background()
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if err := s.readyForGenerateLocked("mlx.nativeTextSession.Generate"); err != nil {
			s.err = err
			return
		}
		s.tokenPhases = s.tokenPhases[:0]
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || s.sess.Pos()+maxNew > s.model.maxLen {
			maxNew = s.model.maxLen - s.sess.Pos()
		}
		if maxNew <= 0 {
			s.err = core.NewError("mlx.nativeTextSession.Generate: no room to generate")
			return
		}
		promptLen := len(s.tokens)
		stopTokens := s.stopTokens(cfg)
		eos := singleStopToken(stopTokens)
		tracePhases := cfg.TraceTokenPhases
		var tokenPhases []metal.TokenPhaseTrace
		if tracePhases {
			tokenPhases = make([]metal.TokenPhaseTrace, 0, maxNew)
		}
		emit := func(id int32) bool {
			if err := ctx.Err(); err != nil {
				s.err = err
				return false
			}
			if yield != nil && !yield(inference.Token{ID: id, Text: s.decodeToken(id)}) {
				return false
			}
			return !tokenInSet(id, stopTokens)
		}
		if tracePhases {
			phaseStart := time.Now()
			emit = func(id int32) bool {
				if err := ctx.Err(); err != nil {
					s.err = err
					return false
				}
				ready := time.Now()
				textStart := time.Now()
				text := s.decodeToken(id)
				textDuration := time.Since(textStart)
				yieldStart := time.Now()
				yieldOK := true
				if yield != nil && !yield(inference.Token{ID: id, Text: text}) {
					yieldOK = false
				}
				yieldDuration := time.Since(yieldStart)
				stop := tokenInSet(id, stopTokens)
				forwardDuration := ready.Sub(phaseStart)
				totalDuration := time.Since(phaseStart)
				if forwardDuration <= 0 {
					forwardDuration = time.Nanosecond
				}
				if totalDuration <= 0 {
					totalDuration = time.Nanosecond
				}
				phase := metal.TokenPhaseTrace{
					Step:               len(tokenPhases),
					TokenID:            id,
					FinalToken:         stop || !yieldOK,
					TotalDuration:      totalDuration,
					ForwardDuration:    forwardDuration,
					DecodeTextDuration: textDuration,
					YieldDuration:      yieldDuration,
				}
				if cfg.TraceTokenText {
					phase.TokenText = text
				}
				tokenPhases = append(tokenPhases, phase)
				phaseStart = time.Now()
				return yieldOK && !stop
			}
		}
		var (
			out []int32
			err error
		)
		sampled := cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1
		if sampled {
			params := model.SampleParams{
				Temperature:         cfg.Temperature,
				TopK:                cfg.TopK,
				TopP:                cfg.TopP,
				MinP:                cfg.MinP,
				SuppressTokens:      cfg.SuppressTokens,
				MinTokensBeforeStop: cfg.MinTokensBeforeStop,
				RepeatPenalty:       cfg.RepeatPenalty,
			}
			out, err = s.sess.GenerateSampledFromCacheEach(maxNew, stopTokens, model.NewSampler(nativeSamplerSeed(cfg)), params, nil, emit)
		} else if cfg.MinTokensBeforeStop > 0 && len(stopTokens) > 0 {
			prefix := cfg.MinTokensBeforeStop
			if prefix > maxNew {
				prefix = maxNew
			}
			prefixSuppress := nativeTextMergeSuppressTokens(cfg.SuppressTokens, stopTokens)
			out, err = s.sess.GenerateFromCacheEachWithSuppressionAndTransform(prefix, eos, prefixSuppress, nil, emit)
			if err == nil && len(out) == prefix && len(out) < maxNew {
				var more []int32
				if len(cfg.SuppressTokens) > 0 {
					more, err = s.sess.GenerateFromCacheEachWithSuppressionAndTransform(maxNew-len(out), eos, cfg.SuppressTokens, nil, emit)
				} else {
					more, err = s.sess.GenerateFromCacheEach(maxNew-len(out), eos, emit)
				}
				out = append(out, more...)
			}
		} else if len(cfg.SuppressTokens) > 0 {
			out, err = s.sess.GenerateFromCacheEachWithSuppressionAndTransform(maxNew, eos, cfg.SuppressTokens, nil, emit)
		} else {
			out, err = s.sess.GenerateFromCacheEach(maxNew, eos, emit)
		}
		if err != nil {
			s.err = err
			return
		}
		s.tokens = append(s.tokens, out...)
		s.generated = append(s.generated, out...)
		logits, err := s.sess.BoundaryLogits()
		if err != nil {
			s.logits = nil
			s.err = err
			return
		}
		s.logits = append(s.logits[:0], logits...)
		s.err = ctx.Err()
		if tracePhases {
			if len(tokenPhases) > 0 {
				tokenPhases[len(tokenPhases)-1].FinalToken = true
			}
			s.tokenPhases = append(s.tokenPhases[:0], tokenPhases...)
		}
		s.model.setSessionMetrics(promptLen, len(out), s.prefillDuration, time.Since(totalStart))
	}
}

func (s *nativeTextSession) LastTokenPhases() []metal.TokenPhaseTrace {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.tokenPhases) == 0 {
		return nil
	}
	return append([]metal.TokenPhaseTrace(nil), s.tokenPhases...)
}

func (s *nativeTextSession) CaptureKV(ctx context.Context) (*kv.Snapshot, error) {
	return s.CaptureKVWithOptions(ctx, kv.CaptureOptions{})
}

func (s *nativeTextSession) CaptureKVWithOptions(ctx context.Context, opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("mlx.nativeTextSession.CaptureKV"); err != nil {
		s.err = err
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return nil, err
	}
	source, err := s.sess.StateBlockSource(max(1, s.sess.Pos()))
	if err != nil {
		s.err = err
		return nil, err
	}
	if source.BlockCount != 1 {
		err := core.NewError("mlx.nativeTextSession.CaptureKV: unexpected state block count")
		s.err = err
		return nil, err
	}
	block, err := source.Load(0)
	if err != nil {
		s.err = err
		return nil, err
	}
	snapshot := s.snapshotFromNativeBlock(source, block, true, true)
	if opts.RawKVOnly {
		kv.DropFloat32(snapshot)
	}
	s.err = nil
	return snapshot, ctx.Err()
}

func (s *nativeTextSession) RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if yield == nil {
		return core.NewError("mlx.nativeTextSession.RangeKVBlocks: nil yield")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("mlx.nativeTextSession.RangeKVBlocks"); err != nil {
		s.err = err
		return err
	}
	source, err := s.sess.StateBlockSourceFrom(opts.BlockStartToken, blockSize)
	if err != nil {
		s.err = err
		return err
	}
	for i := 0; i < source.BlockCount; i++ {
		if err := ctx.Err(); err != nil {
			s.err = err
			return err
		}
		block, err := source.Load(i)
		if err != nil {
			s.err = err
			return err
		}
		final := block.TokenStart+block.TokenCount == source.Position
		snapshot := s.snapshotFromNativeBlock(source, block, final, false)
		if opts.RawKVOnly {
			kv.DropFloat32(snapshot)
		}
		ok, err := yield(kv.Block{
			Index:      block.Index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   snapshot,
		})
		if err != nil || !ok {
			s.err = err
			return err
		}
	}
	s.err = nil
	return nil
}

func (s *nativeTextSession) RestoreKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if snapshot == nil {
		return core.NewError("mlx.nativeTextSession.RestoreKV: nil snapshot")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.RestoreKV"); err != nil {
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	source, tokens, generated, logits, err := nativeTextStateSourceFromSnapshot(kvconv.ToMetalKVSnapshot(snapshot))
	if err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RestoreStateBlocks(source); err != nil {
		s.err = err
		return err
	}
	s.tokens = tokens
	s.generated = generated
	s.logits = logits
	s.prefillDuration = 0
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) RestoreKVBlocks(ctx context.Context, source kv.BlockSource) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.RestoreKVBlocks"); err != nil {
		s.err = err
		return err
	}
	restored, tokens, logits, err := nativeTextStateSourceFromBlockSource(ctx, metalKVBlockSourceFromRoot(source), s.tokens)
	if err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RestoreStateBlocks(restored); err != nil {
		s.err = err
		return err
	}
	s.tokens = tokens
	s.generated = nil
	s.logits = logits
	s.prefillDuration = 0
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) Fork(ctx context.Context) (inference.SessionHandle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := s.CaptureKV(ctx)
	if err != nil {
		return nil, err
	}
	fork := s.model.NewSession()
	if fork == nil {
		return nil, core.NewError("mlx.nativeTextSession.Fork: native model returned nil session")
	}
	restorer, ok := fork.(interface {
		RestoreKV(context.Context, *kv.Snapshot) error
	})
	if !ok {
		_ = fork.Close()
		return nil, core.NewError("mlx.nativeTextSession.Fork: forked session cannot restore KV")
	}
	if err := restorer.RestoreKV(ctx, snapshot); err != nil {
		_ = fork.Close()
		return nil, err
	}
	return fork, nil
}

func (s *nativeTextSession) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model == nil || s.closed {
		s.tokens = nil
		s.generated = nil
		s.logits = nil
		s.prefillDuration = 0
		return
	}
	next, err := s.model.openNativeRootSession()
	if err != nil {
		s.err = err
		return
	}
	old := s.sess
	s.sess = next
	s.tokens = nil
	s.generated = nil
	s.logits = nil
	s.prefillDuration = 0
	s.err = nil
	if old != nil {
		_ = old.Close()
	}
}

func (s *nativeTextSession) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.err
	}
	s.closed = true
	s.tokens = nil
	s.generated = nil
	s.logits = nil
	s.prefillDuration = 0
	if s.sess == nil {
		return s.err
	}
	err := s.sess.Close()
	if err != nil {
		s.err = err
	}
	s.sess = nil
	return err
}

func (s *nativeTextSession) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *nativeTextSession) readyLocked(scope string) error {
	if s == nil || s.model == nil || s.sess == nil {
		return core.NewError(scope + ": nil session")
	}
	if s.closed {
		return core.NewError(scope + ": session is closed")
	}
	return nil
}

func (s *nativeTextSession) readyForGenerateLocked(scope string) error {
	if err := s.readyLocked(scope); err != nil {
		return err
	}
	if len(s.tokens) == 0 || s.sess.Pos() <= 0 {
		return core.NewError(scope + ": no retained prompt state")
	}
	return nil
}

func (s *nativeTextSession) stopTokens(cfg inference.GenerateConfig) []int32 {
	if len(cfg.StopTokens) > 0 {
		return cfg.StopTokens
	}
	if s != nil && s.model != nil && s.model.tok != nil && s.model.tok.HasEOSToken() {
		return []int32{s.model.tok.EOSToken()}
	}
	return nil
}

func (s *nativeTextSession) decodeToken(id int32) string {
	if s == nil || s.model == nil || s.model.tok == nil {
		return ""
	}
	return s.model.tok.DecodeToken(id)
}

func (s *nativeTextSession) snapshotFromNativeBlock(source native.SessionStateBlockSource, block native.SessionStateBlock, final, includeGenerated bool) *kv.Snapshot {
	layers := make([]kv.LayerSnapshot, len(block.Layers))
	numHeads, headDim := 0, 0
	for i, layer := range block.Layers {
		kvHeads := nativeTextSnapshotLayerKVHeads(layer)
		if numHeads == 0 {
			numHeads = kvHeads
		}
		if headDim == 0 {
			headDim = layer.HeadDim
		}
		cacheMode := string(metal.KVCacheModeFixed)
		if layer.CacheMode != "" {
			cacheMode = layer.CacheMode
		}
		layers[i] = kv.LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			CacheMode:  cacheMode,
			MaxSize:    layer.MaxSize,
			KeyDType:   kvconv.RootKVHeadDType(metal.DTypeBFloat16, layer.KeyBytes),
			KeyBytes:   append([]byte(nil), layer.KeyBytes...),
			KeyShape:   []int32{int32(block.TokenCount), int32(kvHeads), int32(layer.HeadDim)},
			ValueDType: kvconv.RootKVHeadDType(metal.DTypeBFloat16, layer.ValueBytes),
			ValueBytes: append([]byte(nil), layer.ValueBytes...),
			ValueShape: []int32{int32(block.TokenCount), int32(kvHeads), int32(layer.HeadDim)},
		}
	}
	tokens := s.snapshotTokens(source, block.TokenStart, block.TokenCount)
	var generated []int32
	if includeGenerated {
		generated = append([]int32(nil), s.generated...)
	}
	var (
		logitShape []int32
		logits     []float32
	)
	if final && len(s.logits) > 0 {
		logitShape = []int32{1, 1, int32(s.modelVocab())}
		logits = nativeTextBF16ToF32(s.logits)
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  s.modelArchitecture(),
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   block.TokenStart + block.TokenCount,
		NumLayers:     nativeTextSnapshotLayerCount(layers, s.modelNumLayers()),
		NumHeads:      numHeads,
		SeqLen:        block.TokenCount,
		HeadDim:       headDim,
		NumQueryHeads: s.modelNumQueryHeads(numHeads),
		LogitShape:    logitShape,
		Logits:        logits,
		Layers:        layers,
	}
}

func nativeTextSnapshotLayerKVHeads(layer native.SessionStateLayerBlock) int {
	if layer.RowBytes > 0 && layer.HeadDim > 0 {
		rowUnit := layer.HeadDim * 2
		if rowUnit > 0 && layer.RowBytes%rowUnit == 0 {
			if heads := layer.RowBytes / rowUnit; heads > 0 {
				return heads
			}
		}
	}
	return layer.KVHeads
}

func (s *nativeTextSession) snapshotTokens(source native.SessionStateBlockSource, start, count int) []int32 {
	tokens := s.tokens
	if len(source.CachedIDs) >= source.Position {
		tokens = source.CachedIDs
	}
	if start < 0 || count <= 0 || start+count > len(tokens) {
		return nil
	}
	return append([]int32(nil), tokens[start:start+count]...)
}

func (s *nativeTextSession) modelArchitecture() string {
	if s == nil || s.model == nil {
		return ""
	}
	if s.model.info.Architecture != "" {
		return s.model.info.Architecture
	}
	return s.model.modelType
}

func (s *nativeTextSession) modelNumLayers() int {
	if s == nil || s.model == nil {
		return 0
	}
	return s.model.info.NumLayers
}

func (s *nativeTextSession) modelNumQueryHeads(fallback int) int {
	if s == nil || s.model == nil || s.model.tm == nil {
		return fallback
	}
	if reporter, ok := s.model.tm.(interface{ NumQueryHeads() int }); ok {
		if n := reporter.NumQueryHeads(); n > 0 {
			return n
		}
	}
	return fallback
}

func (s *nativeTextSession) modelVocab() int {
	if s == nil || s.model == nil {
		return 0
	}
	if s.model.info.VocabSize > 0 {
		return s.model.info.VocabSize
	}
	if s.model.tm != nil {
		return s.model.tm.Vocab()
	}
	return 0
}

func nativeTextSnapshotLayerCount(layers []kv.LayerSnapshot, fallback int) int {
	if fallback > 0 {
		return fallback
	}
	n := 0
	for _, layer := range layers {
		if layer.Layer >= n {
			n = layer.Layer + 1
		}
	}
	return n
}

func nativeTextStateSourceFromSnapshot(snapshot *metal.KVSnapshot) (native.SessionStateBlockSource, []int32, []int32, []byte, error) {
	if snapshot == nil {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: nil snapshot")
	}
	visibleTokens := snapshot.SeqLen
	if visibleTokens <= 0 {
		visibleTokens = len(snapshot.Tokens)
	}
	position := visibleTokens
	if snapshot.TokenOffset > position {
		position = snapshot.TokenOffset
	}
	if visibleTokens <= 0 || position <= 0 {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: empty snapshot")
	}
	if len(snapshot.Tokens) > visibleTokens {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: token state exceeds sequence length")
	}
	layers, err := nativeTextStateLayersFromSnapshot(snapshot, visibleTokens)
	if err != nil {
		return native.SessionStateBlockSource{}, nil, nil, nil, err
	}
	if len(layers) == 0 {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: snapshot has no native KV slabs")
	}
	blocks, err := nativeTextStateBlocksFromSnapshot(layers, position, visibleTokens)
	if err != nil {
		return native.SessionStateBlockSource{}, nil, nil, nil, err
	}
	tokens := append([]int32(nil), snapshot.Tokens...)
	logits := nativeTextF32ToBF16(snapshot.Logits)
	source := native.SessionStateBlockSource{
		Position:           position,
		CachedIDs:          append([]int32(nil), tokens...),
		CachedPromptIDs:    append([]int32(nil), tokens...),
		CachedPromptLogits: append([]byte(nil), logits...),
		RetainedLogits:     append([]byte(nil), logits...),
		BlockCount:         len(blocks),
	}
	source.Load = func(index int) (native.SessionStateBlock, error) {
		if index < 0 || index >= len(blocks) {
			return native.SessionStateBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: block index out of range")
		}
		return blocks[index], nil
	}
	return source, tokens, append([]int32(nil), snapshot.Generated...), logits, nil
}

func nativeTextStateBlocksFromSnapshot(layers []native.SessionStateLayerBlock, position, visibleTokens int) ([]native.SessionStateBlock, error) {
	if visibleTokens <= 0 || position < visibleTokens {
		return nil, core.NewError("mlx.nativeTextSession.RestoreKV: invalid snapshot token timeline")
	}
	if position == visibleTokens {
		return []native.SessionStateBlock{{
			Index:      0,
			TokenStart: 0,
			TokenCount: visibleTokens,
			Layers:     layers,
		}}, nil
	}
	prefixTokens := position - visibleTokens
	prefixLayers := make([]native.SessionStateLayerBlock, len(layers))
	tailLayers := make([]native.SessionStateLayerBlock, len(layers))
	for i, layer := range layers {
		if layer.MaxSize <= 0 {
			return nil, core.NewError("mlx.nativeTextSession.RestoreKV: token offset exceeds unbounded KV snapshot length")
		}
		if visibleTokens < layer.MaxSize {
			return nil, core.NewError("mlx.nativeTextSession.RestoreKV: token offset snapshot is shorter than cache window")
		}
		prefixLayers[i] = layer
		prefixLayers[i].KeyBytes = nil
		prefixLayers[i].ValueBytes = nil
		tailLayers[i] = layer
	}
	return []native.SessionStateBlock{
		{
			Index:      0,
			TokenStart: 0,
			TokenCount: prefixTokens,
			Layers:     prefixLayers,
		},
		{
			Index:      1,
			TokenStart: prefixTokens,
			TokenCount: visibleTokens,
			Layers:     tailLayers,
		},
	}, nil
}

func nativeTextStateSourceFromBlockSource(ctx context.Context, source metal.KVSnapshotBlockSource, residentTokens []int32) (native.SessionStateBlockSource, []int32, []byte, error) {
	if source.BlockCount <= 0 || source.Load == nil {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: invalid block source")
	}
	position := source.PrefixTokens
	if position <= 0 || position > source.TokenCount {
		position = source.TokenCount
	}
	if position <= 0 {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: empty block source")
	}
	blocks := make([]native.SessionStateBlock, 0, source.BlockCount)
	tokens := make([]int32, position)
	filledUntil := 0
	trustedPrefix, trustedFirstBlock := 0, 0
	compactTokenStart := 0
	normaliseBlockIndexes := false
	var logits []byte
	for i := 0; i < source.BlockCount && filledUntil < position; i++ {
		if err := ctx.Err(); err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		block, err := source.Load(ctx, i)
		if err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		if block.Snapshot == nil {
			return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: nil block snapshot")
		}
		if block.TokenStart < 0 || block.TokenCount <= 0 {
			continue
		}
		take := block.TokenCount
		if block.TokenStart+take > position {
			take = position - block.TokenStart
		}
		if take <= 0 {
			continue
		}
		if len(block.Snapshot.Tokens) < take {
			return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
		}
		layers, err := nativeTextStateLayersFromBlock(block.Snapshot, take)
		if err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		if block.TokenStart > filledUntil {
			if filledUntil != 0 || block.TokenStart > len(residentTokens) {
				if filledUntil != 0 || !nativeTextStateLayersCanRestoreExpiredPrefix(layers) {
					return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
				}
				blocks = append(blocks, native.SessionStateBlock{
					Index:      0,
					TokenStart: 0,
					TokenCount: block.TokenStart,
					Layers:     nativeTextStateExpiredPrefixLayers(layers),
				})
				filledUntil = block.TokenStart
				compactTokenStart = block.TokenStart
				normaliseBlockIndexes = true
			} else {
				copy(tokens[:block.TokenStart], residentTokens[:block.TokenStart])
				filledUntil = block.TokenStart
				if block.Index > 0 {
					trustedPrefix = block.TokenStart
					trustedFirstBlock = block.Index
				}
			}
		}
		if block.TokenStart != filledUntil {
			return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
		}
		blockIndex := block.Index
		if normaliseBlockIndexes {
			blockIndex = len(blocks)
		}
		blocks = append(blocks, native.SessionStateBlock{
			Index:      blockIndex,
			TokenStart: block.TokenStart,
			TokenCount: take,
			Layers:     layers,
		})
		copy(tokens[block.TokenStart:block.TokenStart+take], block.Snapshot.Tokens[:take])
		filledUntil += take
		if block.TokenStart+take == position && len(block.Snapshot.Logits) > 0 {
			logits = nativeTextF32ToBF16(block.Snapshot.Logits)
		}
	}
	if filledUntil != position {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
	}
	restoredTokens := tokens
	if compactTokenStart > 0 {
		restoredTokens = append([]int32(nil), tokens[compactTokenStart:position]...)
	}
	restored := native.SessionStateBlockSource{
		Position:           position,
		CachedIDs:          append([]int32(nil), restoredTokens...),
		CachedPromptIDs:    append([]int32(nil), restoredTokens...),
		CachedPromptLogits: append([]byte(nil), logits...),
		RetainedLogits:     append([]byte(nil), logits...),
		BlockCount:         len(blocks),
	}
	restored.Load = func(index int) (native.SessionStateBlock, error) {
		if index < 0 || index >= len(blocks) {
			return native.SessionStateBlock{}, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block index out of range")
		}
		return blocks[index], nil
	}
	if trustedPrefix > 0 && trustedFirstBlock > 0 {
		if err := restored.TrustPrefixTokens(trustedPrefix, trustedFirstBlock); err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
	}
	return restored, restoredTokens, logits, nil
}

func nativeTextStateLayersCanRestoreExpiredPrefix(layers []native.SessionStateLayerBlock) bool {
	if len(layers) == 0 {
		return false
	}
	for _, layer := range layers {
		if layer.MaxSize <= 0 || layer.RowBytes <= 0 || layer.KVHeads <= 0 || layer.HeadDim <= 0 {
			return false
		}
	}
	return true
}

func nativeTextStateExpiredPrefixLayers(layers []native.SessionStateLayerBlock) []native.SessionStateLayerBlock {
	prefix := make([]native.SessionStateLayerBlock, len(layers))
	copy(prefix, layers)
	for i := range prefix {
		prefix[i].KeyBytes = nil
		prefix[i].ValueBytes = nil
	}
	return prefix
}

func nativeTextStateLayersFromSnapshot(snapshot *metal.KVSnapshot, tokenCount int) ([]native.SessionStateLayerBlock, error) {
	layers := make([]native.SessionStateLayerBlock, 0, len(snapshot.Layers))
	for _, layer := range snapshot.Layers {
		stateLayer, err := nativeTextStateLayerFromMetal(layer, tokenCount)
		if err != nil {
			return nil, err
		}
		if len(stateLayer.KeyBytes) == 0 && len(stateLayer.ValueBytes) == 0 {
			continue
		}
		layers = append(layers, stateLayer)
	}
	return layers, nil
}

func nativeTextStateLayersFromBlock(snapshot *metal.KVSnapshot, tokenCount int) ([]native.SessionStateLayerBlock, error) {
	layers := make([]native.SessionStateLayerBlock, 0, len(snapshot.Layers))
	for _, layer := range snapshot.Layers {
		stateLayer, err := nativeTextStateLayerFromMetal(layer, tokenCount)
		if err != nil {
			return nil, err
		}
		if len(stateLayer.KeyBytes) == 0 && len(stateLayer.ValueBytes) == 0 {
			continue
		}
		n := tokenCount * stateLayer.RowBytes
		if n > len(stateLayer.KeyBytes) || n > len(stateLayer.ValueBytes) {
			return nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block payload shorter than token count")
		}
		stateLayer.KeyBytes = stateLayer.KeyBytes[:n]
		stateLayer.ValueBytes = stateLayer.ValueBytes[:n]
		layers = append(layers, stateLayer)
	}
	return layers, nil
}

func nativeTextStateLayerFromMetal(layer metal.KVLayerSnapshot, tokenCount int) (native.SessionStateLayerBlock, error) {
	if len(layer.KeyBytes) == 0 && len(layer.ValueBytes) == 0 {
		if len(layer.Heads) == 0 {
			return native.SessionStateLayerBlock{}, nil
		}
		return nativeTextStateLayerFromMetalHeads(layer, tokenCount)
	}
	kvHeads, headDim, rowBytes, err := nativeTextLayerGeometry(layer, tokenCount)
	if err != nil {
		return native.SessionStateLayerBlock{}, err
	}
	keyBytes, err := nativeTextLayerSlabToBF16(layer.KeyBytes, layer.KeyDType, kvHeads, headDim, tokenCount)
	if err != nil {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: unsupported key dtype")
	}
	valueBytes, err := nativeTextLayerSlabToBF16(layer.ValueBytes, layer.ValueDType, kvHeads, headDim, tokenCount)
	if err != nil {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: unsupported value dtype")
	}
	n := tokenCount * rowBytes
	if nativeTextLayerSlabIsHeadMajor(layer.KeyShape, tokenCount) {
		keyBytes = nativeTextLayerSlabToTokenMajor(keyBytes, kvHeads, headDim, tokenCount)
	}
	if nativeTextLayerSlabIsHeadMajor(layer.ValueShape, tokenCount) {
		valueBytes = nativeTextLayerSlabToTokenMajor(valueBytes, kvHeads, headDim, tokenCount)
	}
	if len(keyBytes) != n || len(valueBytes) != n {
		if inferredHeads := nativeTextInferBF16SlabKVHeads(keyBytes, valueBytes, tokenCount, headDim); inferredHeads > 0 {
			kvHeads = inferredHeads
			rowBytes = kvHeads * headDim * 2
			n = tokenCount * rowBytes
		}
	}
	if len(keyBytes) != n || len(valueBytes) != n {
		return native.SessionStateLayerBlock{}, core.NewError(core.Sprintf(
			"mlx.nativeTextSession.RestoreKV: native KV slab size mismatch key=%d value=%d want=%d key_shape=%v value_shape=%v token_count=%d",
			len(keyBytes), len(valueBytes), n, layer.KeyShape, layer.ValueShape, tokenCount))
	}
	return native.SessionStateLayerBlock{
		Layer:      layer.Layer,
		CacheIndex: layer.CacheIndex,
		CacheMode:  string(layer.CacheMode),
		MaxSize:    layer.MaxSize,
		KVHeads:    kvHeads,
		HeadDim:    headDim,
		RowBytes:   rowBytes,
		KeyBytes:   keyBytes,
		ValueBytes: valueBytes,
	}, nil
}

func nativeTextInferBF16SlabKVHeads(keyBytes, valueBytes []byte, tokenCount, headDim int) int {
	if len(keyBytes) == 0 || len(keyBytes) != len(valueBytes) || tokenCount <= 0 || headDim <= 0 {
		return 0
	}
	rowUnit := tokenCount * headDim * 2
	if rowUnit <= 0 || len(keyBytes)%rowUnit != 0 {
		return 0
	}
	return len(keyBytes) / rowUnit
}

func nativeTextLayerSlabToBF16(raw []byte, dtype metal.DType, kvHeads, headDim, tokenCount int) ([]byte, error) {
	if dtype == 0 || dtype == metal.DTypeBFloat16 {
		return raw, nil
	}
	if dtype != metal.DTypeFloat32 {
		return nil, core.NewError("unsupported dtype")
	}
	values := kvHeads * headDim * tokenCount
	if values <= 0 || len(raw) != values*4 {
		return nil, core.NewError("float32 slab size mismatch")
	}
	out := make([]byte, values*2)
	for i := 0; i < values; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(bits>>16))
	}
	return out, nil
}

func nativeTextStateLayerFromMetalHeads(layer metal.KVLayerSnapshot, tokenCount int) (native.SessionStateLayerBlock, error) {
	if tokenCount <= 0 || len(layer.Heads) == 0 {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: missing native KV geometry")
	}
	headDim, err := nativeTextHeadDim(layer.Heads[0], tokenCount, true)
	if err != nil {
		return native.SessionStateLayerBlock{}, err
	}
	valueHeadDim, err := nativeTextHeadDim(layer.Heads[0], tokenCount, false)
	if err != nil {
		return native.SessionStateLayerBlock{}, err
	}
	if headDim <= 0 || valueHeadDim != headDim {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: missing native KV geometry")
	}
	kvHeads := len(layer.Heads)
	rowBytes := kvHeads * headDim * 2
	keyBytes := make([]byte, tokenCount*rowBytes)
	valueBytes := make([]byte, tokenCount*rowBytes)
	for hi, head := range layer.Heads {
		keyHeadDim, err := nativeTextHeadDim(head, tokenCount, true)
		if err != nil {
			return native.SessionStateLayerBlock{}, err
		}
		valueHeadDim, err := nativeTextHeadDim(head, tokenCount, false)
		if err != nil {
			return native.SessionStateLayerBlock{}, err
		}
		if keyHeadDim != headDim || valueHeadDim != headDim {
			return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: native KV head dims differ")
		}
		if err := nativeTextCopyHeadTensorBF16(keyBytes, hi, kvHeads, headDim, tokenCount, head, true); err != nil {
			return native.SessionStateLayerBlock{}, err
		}
		if err := nativeTextCopyHeadTensorBF16(valueBytes, hi, kvHeads, headDim, tokenCount, head, false); err != nil {
			return native.SessionStateLayerBlock{}, err
		}
	}
	return native.SessionStateLayerBlock{
		Layer:      layer.Layer,
		CacheIndex: layer.CacheIndex,
		CacheMode:  string(layer.CacheMode),
		MaxSize:    layer.MaxSize,
		KVHeads:    kvHeads,
		HeadDim:    headDim,
		RowBytes:   rowBytes,
		KeyBytes:   keyBytes,
		ValueBytes: valueBytes,
	}, nil
}

func nativeTextHeadDim(head metal.KVHeadSnapshot, tokenCount int, key bool) (int, error) {
	values := head.Value
	raw, dtype := head.ValueBytes, head.ValueDType
	if key {
		values = head.Key
		raw, dtype = head.KeyBytes, head.KeyDType
	}
	if len(values) > 0 {
		if tokenCount <= 0 || len(values)%tokenCount != 0 {
			return 0, core.NewError("mlx.nativeTextSession.RestoreKV: native KV head size mismatch")
		}
		return len(values) / tokenCount, nil
	}
	bytesPerValue := metal.DTypeByteSize(dtype)
	if len(raw) == 0 || bytesPerValue <= 0 || tokenCount <= 0 || len(raw)%bytesPerValue != 0 {
		return 0, core.NewError("mlx.nativeTextSession.RestoreKV: native KV head size mismatch")
	}
	elements := len(raw) / bytesPerValue
	if elements%tokenCount != 0 {
		return 0, core.NewError("mlx.nativeTextSession.RestoreKV: native KV head size mismatch")
	}
	return elements / tokenCount, nil
}

func nativeTextCopyHeadTensorBF16(dst []byte, headIndex, kvHeads, headDim, tokenCount int, head metal.KVHeadSnapshot, key bool) error {
	values := head.Value
	raw, dtype := head.ValueBytes, head.ValueDType
	if key {
		values = head.Key
		raw, dtype = head.KeyBytes, head.KeyDType
	}
	rowBytes := kvHeads * headDim * 2
	for t := 0; t < tokenCount; t++ {
		dstOff := t*rowBytes + headIndex*headDim*2
		if len(values) > 0 {
			srcOff := t * headDim
			for d := 0; d < headDim; d++ {
				bits := math.Float32bits(values[srcOff+d])
				binary.LittleEndian.PutUint16(dst[dstOff+d*2:dstOff+d*2+2], uint16(bits>>16))
			}
			continue
		}
		srcOff := t * headDim * metal.DTypeByteSize(dtype)
		switch dtype {
		case metal.DTypeBFloat16:
			copy(dst[dstOff:dstOff+headDim*2], raw[srcOff:srcOff+headDim*2])
		case metal.DTypeFloat32:
			for d := 0; d < headDim; d++ {
				bits := binary.LittleEndian.Uint32(raw[srcOff+d*4 : srcOff+d*4+4])
				binary.LittleEndian.PutUint16(dst[dstOff+d*2:dstOff+d*2+2], uint16(bits>>16))
			}
		default:
			return core.NewError("mlx.nativeTextSession.RestoreKV: unsupported native KV head dtype")
		}
	}
	return nil
}

func nativeTextLayerGeometry(layer metal.KVLayerSnapshot, tokenCount int) (int, int, int, error) {
	kvHeads, headDim := 0, 0
	if len(layer.KeyShape) == 3 && int(layer.KeyShape[0]) == tokenCount {
		kvHeads, headDim = int(layer.KeyShape[1]), int(layer.KeyShape[2])
	} else if len(layer.KeyShape) == 4 && int(layer.KeyShape[2]) == tokenCount {
		kvHeads, headDim = int(layer.KeyShape[1]), int(layer.KeyShape[3])
	}
	if kvHeads <= 0 || headDim <= 0 {
		return 0, 0, 0, core.NewError("mlx.nativeTextSession.RestoreKV: missing native KV geometry")
	}
	rowBytes := kvHeads * headDim * 2
	return kvHeads, headDim, rowBytes, nil
}

func nativeTextLayerSlabIsHeadMajor(shape []int32, tokenCount int) bool {
	return len(shape) == 4 && shape[0] == 1 && int(shape[2]) == tokenCount
}

func nativeTextLayerSlabToTokenMajor(raw []byte, kvHeads, headDim, tokenCount int) []byte {
	if kvHeads <= 0 || headDim <= 0 || tokenCount <= 0 {
		return raw
	}
	headBytes := headDim * 2
	rowBytes := kvHeads * headBytes
	out := make([]byte, len(raw))
	for t := 0; t < tokenCount; t++ {
		for h := 0; h < kvHeads; h++ {
			src := (h*tokenCount + t) * headBytes
			dst := t*rowBytes + h*headBytes
			copy(out[dst:dst+headBytes], raw[src:src+headBytes])
		}
	}
	return out
}

func nativeTextBF16ToF32(raw []byte) []float32 {
	if len(raw) == 0 {
		return nil
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		off := i * 2
		out[i] = math.Float32frombits(uint32(uint16(raw[off])|uint16(raw[off+1])<<8) << 16)
	}
	return out
}

func nativeTextF32ToBF16(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	out := make([]byte, len(values)*2)
	for i, v := range values {
		bits := math.Float32bits(v)
		h := uint16(bits >> 16)
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

func (m *nativeTextModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

func (m *nativeTextModel) setMetrics(promptTokens, genTokens int, total time.Duration) {
	m.setMetricsWithThinkingBudget(promptTokens, genTokens, total, false)
}

func (m *nativeTextModel) setMetricsWithThinkingBudget(promptTokens, genTokens int, total time.Duration, thinkingBudgetForced bool) {
	tps := 0.0
	if total > 0 {
		tps = float64(genTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:         promptTokens,
		GeneratedTokens:      genTokens,
		TotalDuration:        total,
		DecodeDuration:       total,
		DecodeTokensPerSec:   tps,
		ThinkingBudgetForced: thinkingBudgetForced,
	}
	m.mu.Unlock()
}

func (m *nativeTextModel) setSessionMetrics(promptTokens, genTokens int, prefill, decode time.Duration) {
	prefillTPS := 0.0
	if prefill > 0 {
		prefillTPS = float64(promptTokens) / prefill.Seconds()
	}
	decodeTPS := 0.0
	if decode > 0 {
		decodeTPS = float64(genTokens) / decode.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:        promptTokens,
		GeneratedTokens:     genTokens,
		PrefillDuration:     prefill,
		DecodeDuration:      decode,
		TotalDuration:       prefill + decode,
		PrefillTokensPerSec: prefillTPS,
		DecodeTokensPerSec:  decodeTPS,
	}
	m.mu.Unlock()
}

// Classify samples one token per prompt (greedy) — the prefill-only fast path
// approximated over the contract (the contract has no batched prefill; one short
// Generate per prompt).
func (m *nativeTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(prompts) == 0 {
		return core.Ok([]inference.ClassifyResult{})
	}
	if m == nil || m.tm == nil || m.tok == nil {
		return core.Fail(core.NewError("mlx.nativeTextModel.Classify: model is not initialised"))
	}
	start := time.Now()
	cfg := inference.ApplyGenerateOpts(opts)
	encoded := make([][]int32, len(prompts))
	totalPromptTokens := 0
	for i, p := range prompts {
		encoded[i] = m.tok.Encode(p)
		totalPromptTokens += len(encoded[i])
	}
	results := make([]inference.ClassifyResult, len(prompts))
	sampler := model.NewSampler(nativeSamplerSeed(cfg))
	for i := range encoded {
		if err := ctx.Err(); err != nil {
			return core.Fail(err)
		}
		logits, err := nativePromptLogits(m.tm, encoded[i])
		if err != nil {
			return core.Fail(err)
		}
		id, err := sampler.Sample(logits, m.tm.Vocab(), model.SampleParams{Temperature: cfg.Temperature, TopK: cfg.TopK, TopP: cfg.TopP, MinP: cfg.MinP, SuppressTokens: cfg.SuppressTokens})
		if err != nil {
			return core.Fail(err)
		}
		results[i] = inference.ClassifyResult{Token: inference.Token{ID: id, Text: m.tok.DecodeToken(id)}}
		if cfg.ReturnLogits {
			results[i].Logits, err = nativeBF16LogitsToF32(logits, m.tm.Vocab())
			if err != nil {
				return core.Fail(err)
			}
		}
	}
	m.setClassifyMetrics(totalPromptTokens, len(results), time.Since(start))
	return core.Ok(results)
}

func nativePromptLogits(tm model.TokenModel, ids []int32) ([]byte, error) {
	if tm == nil {
		return nil, core.NewError("mlx.nativePromptLogits: nil model")
	}
	if len(ids) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: empty prompt")
	}
	if sm, ok := tm.(model.SessionModel); ok {
		return nativePromptLogitsStepwise(sm, ids)
	}
	return nativePromptLogitsWholeSeq(tm, ids)
}

func nativePromptLogitsStepwise(tm model.SessionModel, ids []int32) ([]byte, error) {
	sess, err := tm.OpenSession()
	if err != nil {
		return nil, err
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	stepID, idAware := sess.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	})
	var hidden []byte
	for _, id := range ids {
		emb, err := tm.Embed(id)
		if err != nil {
			return nil, err
		}
		if idAware {
			hidden, err = stepID.StepWithID(id, emb)
		} else {
			hidden, err = sess.Step(emb)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(hidden) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: session returned no hidden state")
	}
	return tm.Head(hidden)
}

func nativePromptLogitsWholeSeq(tm model.TokenModel, ids []int32) ([]byte, error) {
	seq := make([][]byte, 0, len(ids))
	for _, id := range ids {
		emb, err := tm.Embed(id)
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}
	hidden, err := tm.DecodeForward(seq)
	if err != nil {
		return nil, err
	}
	if len(hidden) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: backend returned no hidden states")
	}
	return tm.Head(hidden[len(hidden)-1])
}

func nativeBF16LogitsToF32(logits []byte, vocab int) ([]float32, error) {
	if len(logits) != vocab*2 {
		return nil, core.NewError("mlx.nativeBF16LogitsToF32: logits must be vocab bf16 bytes")
	}
	out := make([]float32, vocab)
	for i := 0; i < vocab; i++ {
		off := i * 2
		out[i] = math.Float32frombits(uint32(uint16(logits[off])|uint16(logits[off+1])<<8) << 16)
	}
	return out, nil
}

func (m *nativeTextModel) setClassifyMetrics(promptTokens, generatedTokens int, total time.Duration) {
	tps := 0.0
	if total > 0 {
		tps = float64(promptTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:        promptTokens,
		GeneratedTokens:     generatedTokens,
		PrefillDuration:     total,
		TotalDuration:       total,
		PrefillTokensPerSec: tps,
	}
	m.mu.Unlock()
}

func (m *nativeTextModel) setBatchGenerateMetrics(promptTokens, generatedTokens int, total time.Duration, err error) {
	tps := 0.0
	if total > 0 {
		tps = float64(generatedTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = err
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:       promptTokens,
		GeneratedTokens:    generatedTokens,
		DecodeDuration:     total,
		TotalDuration:      total,
		DecodeTokensPerSec: tps,
	}
	m.mu.Unlock()
}

// BatchGenerate runs one Generate per prompt (the contract is single-sequence; no
// true batching — that is a pkg/metal scheduler feature).
func (m *nativeTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	totalStart := time.Now()
	totalPromptTokens := 0
	totalGenerated := 0
	var batchErr error
	for i, p := range prompts {
		ids := m.tok.Encode(p)
		totalPromptTokens += len(ids)
		var toks []inference.Token
		for tok := range m.stream(ctx, ids, cfg) {
			toks = append(toks, tok)
		}
		totalGenerated += len(toks)
		err := resultError(m.Err())
		if batchErr == nil {
			batchErr = err
		}
		results[i] = inference.BatchResult{Tokens: toks, Err: err}
	}
	if len(prompts) > 0 {
		m.setBatchGenerateMetrics(totalPromptTokens, totalGenerated, time.Since(totalStart), batchErr)
	}
	return core.Ok(results)
}

func (m *nativeTextModel) ModelType() string { return m.modelType }

func (m *nativeTextModel) NumLayers() int { return m.Info().NumLayers }

func (m *nativeTextModel) BlockDiffusionCapable() bool {
	if m == nil || m.tm == nil {
		return false
	}
	bd, ok := m.tm.(interface{ BlockDiffusionCapable() bool })
	return ok && bd.BlockDiffusionCapable()
}

func (m *nativeTextModel) Capabilities() inference.CapabilityReport {
	if m == nil {
		return inference.TextModelCapabilities(inference.RuntimeIdentity{
			Backend:       "native",
			CacheMode:     "paged",
			NativeRuntime: true,
		}, nil)
	}
	cacheMode := m.cacheMode
	if cacheMode == "" {
		cacheMode = "paged"
	}
	report := inference.TextModelCapabilities(inference.RuntimeIdentity{
		Backend:       "native",
		CacheMode:     cacheMode,
		NativeRuntime: true,
	}, m)
	nativeTextReportSupportedCapability(&report, inference.CapabilityKVSnapshot, inference.CapabilityGroupRuntime)
	nativeTextReportSupportedCapability(&report, inference.CapabilityPromptCache, inference.CapabilityGroupRuntime)
	report.Model.ContextLength = m.maxLen
	report.CacheModes = []string{"paged"}
	if report.Labels == nil {
		report.Labels = make(map[string]string, 3)
	}
	if m.pagedKVPageSize > 0 {
		report.Labels["paged_kv_page_size"] = core.Sprintf("%d", m.pagedKVPageSize)
	}
	if m.pagedKVPrealloc {
		report.Labels["paged_kv_prealloc"] = "true"
	}
	if m.kvStorageDType != "" {
		report.Labels["kv_storage_dtype"] = m.kvStorageDType
	}
	if len(report.Labels) == 0 {
		report.Labels = nil
	}
	return report
}

func nativeTextReportSupportedCapability(report *inference.CapabilityReport, id inference.CapabilityID, group inference.CapabilityGroup) {
	if report == nil {
		return
	}
	for _, capability := range report.Capabilities {
		if capability.ID == id {
			return
		}
	}
	report.Capabilities = append(report.Capabilities, inference.SupportedCapability(id, group))
}

func (m *nativeTextModel) Info() inference.ModelInfo {
	if m == nil {
		return inference.ModelInfo{}
	}
	info := m.info
	if info.Architecture == "" {
		info.Architecture = m.modelType
	}
	if m.tm == nil {
		return info
	}
	if info.VocabSize == 0 {
		info.VocabSize = m.tm.Vocab()
	}
	if info.NumLayers == 0 {
		if reporter, ok := m.tm.(interface{ NumLayers() int }); ok {
			info.NumLayers = reporter.NumLayers()
		}
	}
	if info.HiddenSize == 0 {
		if reporter, ok := m.tm.(interface{ HiddenSize() int }); ok {
			info.HiddenSize = reporter.HiddenSize()
		}
	}
	if info.QuantBits == 0 {
		if reporter, ok := m.tm.(interface{ QuantBits() int }); ok {
			info.QuantBits = reporter.QuantBits()
		}
	}
	if info.QuantGroup == 0 {
		if reporter, ok := m.tm.(interface{ QuantGroup() int }); ok {
			info.QuantGroup = reporter.QuantGroup()
		}
	}
	return info
}

func (m *nativeTextModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

func (m *nativeTextModel) Err() core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	return core.ResultOf(nil, m.lastErr)
}

// Close releases any retained prompt-cache session. The resident weights live
// for the process in the serve shape; a warmed cache session is explicit mutable
// state and should be dropped on teardown or hot-swap.
func (m *nativeTextModel) Close() core.Result {
	if m == nil {
		return core.Ok(nil)
	}
	m.mu.Lock()
	m.visionCache = nil
	m.clearNativePromptCacheLocked()
	tm, adapterPack := m.tm, m.adapterPack
	m.tm = nil
	m.adapter = inference.AdapterIdentity{}
	m.adapterPack = ""
	m.mu.Unlock()
	if err := closeNativeTokenModel(tm); err != nil {
		_ = nativeRemoveAll(adapterPack)
		return core.Fail(err)
	}
	return core.ResultOf(nil, nativeRemoveAll(adapterPack))
}
