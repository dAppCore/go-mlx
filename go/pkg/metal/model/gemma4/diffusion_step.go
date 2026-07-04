// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"sort"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// The block-diffusion denoising step (docs/RFC.diffusion-gemma.md, Unit B).
//
// One step = one bidirectional forward of the whole canvas against the
// read-only prompt cache, then confidence-based acceptance. The caller owns
// the cache transaction via TruncateTo: the forward advances the caches by
// the canvas length (the multi-token Update path commits directly), and
// TruncateTo(prefix) rolls the offset back after sampling — the masked-write
// transaction in reverse, the same lever MTP verify uses for rejected
// drafts. Canvases live pre-cap by construction (prompt + canvas within the
// fixed size), so the rollback never declines. The accepted-canvas COMMIT
// is a plain causal forward with no truncate — Unit C wires that loop.

// DiffusionStepConfig carries the reference sampler defaults
// (google-deepmind/gemma gemma/diffusion/_sampler.py).
type DiffusionStepConfig struct {
	// EntropyBound is the confidence budget per step: tokens are accepted in
	// ascending-entropy order while the accumulated entropy stays within it.
	EntropyBound float32
	// MaxTemperature..MinTemperature anneal as noise clears:
	// t = min + (max-min) * (1 - (1-noise)^Exponent).
	MaxTemperature float32
	MinTemperature float32
	Exponent       float32
	// TextVocabSize bounds the uniform renoise draw (the tokenizer's text
	// vocabulary, not the padded embedding rows).
	TextVocabSize int32
	// Seed roots the per-step PRNG key chain. Draws across denoise steps run
	// in separate graphs, where the default key repeats — every step derives
	// explicit keys from Seed and the step index (the reference's
	// jax.random.split chain, flattened).
	Seed uint64
}

// DefaultDiffusionStepConfig is the measured decode profile: the reference
// ships EntropyBound 0.1, but 0.3 converges in fewer steps with no quality
// loss on the 4bit checkpoint (wave-2 sweep, docs/RFC.diffusion-gemma.md —
// and 0.5+ backfires: greedy early-locking destabilises the canvas). The
// temperature anneal stays the reference schedule.
func DefaultDiffusionStepConfig(textVocabSize int32) DiffusionStepConfig {
	return DiffusionStepConfig{
		EntropyBound:   0.3,
		MaxTemperature: 0.8,
		MinTemperature: 0.4,
		Exponent:       1.0,
		TextVocabSize:  textVocabSize,
	}
}

// DenoiseForward runs one bidirectional canvas forward: embed + the
// self-conditioning block, the trunk layers under the diffusion masks, final
// norm and the tied output head. canvas is [1, L] token ids; scEmb is the
// previous step's self-conditioning embedding [1, L, D] (nil on step 0).
// Returns full-canvas logits [1, L, V]. The forward advances every cache by
// L — the caller rolls back with TruncateTo after sampling (step-local K/V).
func (m *DiffusionGemmaModel) DenoiseForward(canvas *metal.Array, scEmb *metal.Array, caches []metal.Cache) *metal.Array {
	var shapeBuf [metal.MaxTensorRank]int32
	shape := canvas.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	offset := int32(0)
	if len(caches) > 0 && caches[0] != nil {
		offset = int32(caches[0].Offset())
	}
	keyLen := offset + L

	globalMask := diffusionGlobalCanvasMask(B, L, keyLen)
	localMask := diffusionBlockLocalCanvasMask(B, L, keyLen, offset, m.Cfg.SlidingWindow)
	defer metal.Free(globalMask, localMask)
	return m.DenoiseForwardWithMasks(canvas, scEmb, caches, globalMask, localMask)
}

// DenoiseForwardWithMasks is DenoiseForward with caller-owned masks — the
// canvas position is fixed for a whole denoising loop, so the loop builds
// the two masks once and reuses them across every step.
func (m *DiffusionGemmaModel) DenoiseForwardWithMasks(canvas, scEmb *metal.Array, caches []metal.Cache, globalMask, localMask *metal.Array) *metal.Array {
	ov := &gemma4ForwardOverrides{
		fullMask:    globalMask,
		slidingMask: localMask,
		embedHook: func(h *metal.Array) *metal.Array {
			return m.selfConditionForward(h, scEmb)
		},
	}
	h, _, _ := m.forwardHiddenOverride(canvas, nil, caches, ov)
	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out
}

// selfConditionForward applies the reference SelfConditioning block:
// RMSNorm_noscale( canvas_embeddings + FFW(RMSNorm_scaled(sc_signal)) ).
// On step 0 (scEmb nil) the FFW arm contributes nothing but the scale-free
// post-norm still applies — matching the reference exactly.
func (m *DiffusionGemmaModel) selfConditionForward(h *metal.Array, scEmb *metal.Array) *metal.Array {
	combined := h
	if scEmb != nil && scEmb.Valid() {
		normed := metal.RMSNorm(scEmb, m.SelfCondPreNorm.Weight, m.Cfg.RMSNormEps)
		ffw := m.SelfCondMLP.Forward(normed)
		metal.Free(normed)
		combined = metal.Add(h, ffw)
		metal.Free(h, ffw)
	}
	out := metal.RMSNormNoScale(combined, m.Cfg.RMSNormEps)
	metal.Free(combined)
	return out
}

// EncodeLogits converts shaped logits into the next step's self-conditioning
// signal: softmax(logits) @ embedding_table * sqrt(d) — the expected
// embedding under the predicted distribution (Embedder.encode_logits).
func (m *DiffusionGemmaModel) EncodeLogits(shapedLogits *metal.Array) *metal.Array {
	probs := metal.Softmax(shapedLogits)
	embed := m.EmbedTokens
	var encoded *metal.Array
	if embed.Scales != nil && embed.Scales.Valid() {
		encoded = metal.QuantizedMatmul(probs, embed.Weight, embed.Scales, embed.Biases, false, embed.GroupSize, embed.Bits)
	} else {
		encoded = metal.Matmul(probs, embed.Weight)
	}
	metal.Free(probs)
	scaled := metal.MulScalar(encoded, float32(math.Sqrt(float64(m.Cfg.HiddenSize))))
	metal.Free(encoded)
	return scaled
}

// DiffusionStepResult is one denoising step's outcome.
type DiffusionStepResult struct {
	Canvas      []int32      // next canvas: accepted predictions + renoised rest
	Greedy      []int32      // argmax of the shaped logits — the convergence signal
	SCEmb       *metal.Array // self-conditioning signal for the next step [1,L,D]
	Accepted    int          // tokens locked in this step
	Changed     int          // positions that differ from the previous canvas
	MeanEntropy float32      // mean per-token entropy — the confidence signal
	// ForwardDur/SampleDur split the step cost (set by the generation loop):
	// the graph-build of the canvas forward vs the sampler chain (which
	// includes the eval that drains BOTH — MLX is lazy, so the forward
	// "build" is host graph time and the GPU work lands in the sample eval).
	ForwardDur time.Duration
	SampleDur  time.Duration
	// EncodeDur/GpuDur split the forward's eval: host command-encode (EvalAsync)
	// vs GPU-wait (Synchronize) — the lever diagnostic (host-encode = replay/fuse
	// headroom; GPU-bound = no host lever, just a faster GPU).
	EncodeDur time.Duration
	GpuDur    time.Duration
}

// SampleDenoiseStep applies the reference sampler to one step's logits:
// annealing temperature, categorical draw, entropy-sorted acceptance within
// the entropy budget, uniform renoise of every non-accepted position. The
// canvas/entropy materialisations are [L]-sized — the [L,V] logits never
// leave the GPU; the self-conditioning encode reuses the shaped logits.
func (m *DiffusionGemmaModel) SampleDenoiseStep(logits *metal.Array, canvas []int32, step int, noiseProportion float32, cfg DiffusionStepConfig) (DiffusionStepResult, error) {
	const op = "gemma4.SampleDenoiseStep"
	L := len(canvas)
	categoricalKey := metal.RandomKey(cfg.Seed ^ (uint64(step)*2 + 1))
	renoiseKey := metal.RandomKey(cfg.Seed ^ (uint64(step)*2 + 2))
	defer metal.Free(categoricalKey, renoiseKey)

	// Annealing temperature: t = min + (max-min) * (1 - (1-noise)^exp).
	frac := 1.0 - float32(math.Pow(float64(1.0-noiseProportion), float64(cfg.Exponent)))
	temp := cfg.MinTemperature + frac*(cfg.MaxTemperature-cfg.MinTemperature)
	if temp <= 0 {
		temp = 1e-6
	}
	shaped := metal.MulScalar(logits, 1.0/temp)

	// Categorical draw + per-token entropy, all on-GPU:
	// H = logsumexp(z) - sum(softmax(z) * z) per position.
	sampled := metal.RandomCategoricalWithKey(shaped, categoricalKey)
	greedy := metal.Argmax(shaped, -1, false)
	lse := metal.LogSumExp(shaped, -1, false)
	probs := metal.Softmax(shaped)
	pz := metal.Mul(probs, shaped)
	sumPZ := metal.Sum(pz, -1, false)
	entropy := metal.Subtract(lse, sumPZ)
	metal.Free(probs, pz, lse, sumPZ)

	// The next self-conditioning signal encodes the SHAPED logits; renoise
	// tokens draw from the same GPU RNG domain. Everything evaluates in ONE
	// batch — only [L]-sized arrays ever reach the host.
	scEmb := m.EncodeLogits(shaped)
	metal.Free(shaped)
	renoiseF := metal.RandomUniformWithKey(0, float32(cfg.TextVocabSize), []int32{int32(L)}, metal.DTypeFloat32, renoiseKey)

	if err := metal.Eval(sampled, greedy, entropy, scEmb, renoiseF); err != nil {
		metal.Free(sampled, greedy, entropy, scEmb, renoiseF)
		return DiffusionStepResult{}, core.E(op, "eval step outputs", err)
	}
	sampledIDs := append([]int32(nil), sampled.DataInt32()...)
	greedyIDs := append([]int32(nil), greedy.DataInt32()...)
	entropies := append([]float32(nil), entropy.Floats()...)
	renoiseVals := append([]float32(nil), renoiseF.Floats()...)
	metal.Free(sampled, greedy, entropy, renoiseF)
	if len(sampledIDs) < L || len(entropies) < L || len(renoiseVals) < L {
		metal.Free(scEmb)
		return DiffusionStepResult{}, core.E(op, core.Sprintf("step outputs short: %d/%d/%d of %d", len(sampledIDs), len(entropies), len(renoiseVals), L), nil)
	}
	renoise := make([]int32, L)
	for i, v := range renoiseVals[:L] {
		id := int32(v)
		if id >= cfg.TextVocabSize {
			id = cfg.TextVocabSize - 1
		}
		renoise[i] = id
	}

	var entropySum float32
	for _, h := range entropies[:L] {
		entropySum += h
	}

	// Entropy-sorted acceptance: take the most confident positions while the
	// accumulated entropy (excluding each candidate's own) fits the budget.
	order := make([]int, L)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return entropies[order[a]] < entropies[order[b]] })
	accept := make([]bool, L)
	accepted := 0
	var accumulated float32
	for _, idx := range order {
		if accumulated > cfg.EntropyBound {
			break
		}
		accept[idx] = true
		accepted++
		accumulated += entropies[idx]
	}

	next := make([]int32, L)
	changed := 0
	for i := range next {
		if accept[i] {
			next[i] = sampledIDs[i]
		} else {
			next[i] = renoise[i]
		}
		if next[i] != canvas[i] {
			changed++
		}
	}

	return DiffusionStepResult{
		Canvas:      next,
		Greedy:      greedyIDs[:L],
		SCEmb:       scEmb,
		Accepted:    accepted,
		Changed:     changed,
		MeanEntropy: entropySum / float32(L),
	}, nil
}

// diffusionGlobalCanvasMask: every canvas token attends to the whole valid
// cache prefix AND to every other canvas token (full bidirectional canvas
// self-attention). Additive convention: 0 = attend, -inf = blocked.
func diffusionGlobalCanvasMask(B, L, keyLen int32) *metal.Array {
	data := make([]float32, int(B)*int(L)*int(keyLen))
	return metal.FromValues(data, int(B), 1, int(L), int(keyLen))
}

// diffusionBlockLocalCanvasMask: block-local attention for sliding layers —
// a context window [offset-window, offset) SHARED by every canvas token,
// plus full canvas self-attention over [offset, offset+L).
func diffusionBlockLocalCanvasMask(B, L, keyLen, offset, window int32) *metal.Array {
	negInf := float32(math.Inf(-1))
	contextStart := offset - window
	if contextStart < 0 {
		contextStart = 0
	}
	data := make([]float32, int(B)*int(L)*int(keyLen))
	for b := int32(0); b < B; b++ {
		base := int(b) * int(L) * int(keyLen)
		for i := int32(0); i < L; i++ {
			row := base + int(i)*int(keyLen)
			for j := int32(0); j < keyLen; j++ {
				inContext := j >= contextStart && j < offset
				inCanvas := j >= offset && j < offset+L
				if !inContext && !inCanvas {
					data[row+int(j)] = negInf
				}
			}
		}
	}
	return metal.FromValues(data, int(B), 1, int(L), int(keyLen))
}
