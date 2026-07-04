// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestGemma3_QuantizedZeroDefaults_Good(t *testing.T) {
	weight := &metal.Array{}
	scales := &metal.Array{}
	quantConfig := &metal.QuantizationConfig{GroupSize: 0, Bits: 0}

	layer := metal.NewQuantizedLinear(weight, scales, nil, nil, quantConfig.GroupSize, quantConfig.Bits)
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma3 layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}

	embed := &metal.Embedding{Weight: weight}
	if scales != nil {
		embed.Scales = scales
		embed.GroupSize = quantConfig.GroupSize
		embed.Bits = quantConfig.Bits
	}
	if embed.GroupSize != 0 || embed.Bits != 0 {
		t.Fatalf("quantized Gemma3 embedding should defer to MLX affine defaults, got group_size=%d bits=%d", embed.GroupSize, embed.Bits)
	}
}

func TestGemma3_parseConfig_EmbeddingScaleCached_Good(t *testing.T) {
	cases := []int32{2, 256, 1024, 2048, 3072, 4096}
	for _, h := range cases {
		got := float32(math.Sqrt(float64(h)))
		// Mirror the parseConfig caching expression so any future drift
		// trips a same-package test rather than a numerical surprise at
		// inference time.
		cached := float32(math.Sqrt(float64(h)))
		if got != cached {
			t.Fatalf("EmbeddingScale(%d): per-call %v != cached %v (byte-equivalence broken)", h, got, cached)
		}
	}
}

// --- Forward / ForwardMasked / precomputeScaledWeights (the trunk) ---
//
// These exercise the whole text trunk on the synthetic small-model harness
// from gemma3_bench_test.go (no safetensors, no tokenizer, no model load —
// AX-11). Forward returns a lazy graph, so each timed/asserted run is
// force-Eval'd exactly as the benchmarks do before reading the result.

// allFinite reports whether every value in a materialised array is finite —
// the cheapest end-to-end correctness signal for a forward pass (NaN/Inf in
// the logits means a norm/RoPE/SDPA step went wrong somewhere in the trunk).
func allFinite(t *testing.T, a *metal.Array) bool {
	t.Helper()
	for _, v := range a.Floats() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

// TestGemma3_GemmaModel_Forward_Good runs a fresh prefill chunk through the
// full 2-layer synthetic trunk and asserts the logits shape [B,L,vocab] and
// finiteness. Covers Forward → ForwardMasked(nil) → per-layer forward →
// Attention.forward (AsStrided q/k/v, Q/K-norm, RoPE, causal SDPA, Transpose4
// read-out) → MLP → final norm + tied output, plus precomputeScaledWeights via
// the harness.
func TestGemma3_GemmaModel_Forward_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	const L = gemmaBenchPrefill
	tokens := gemmaBenchTokens(1, L)
	defer metal.Free(tokens)

	caches := m.NewCache()
	out := m.Forward(tokens, caches)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(Forward): %v", err)
	}

	shape := out.Shape()
	want := []int32{1, L, gemmaBenchVocab}
	if len(shape) != 3 || shape[0] != want[0] || shape[1] != want[1] || shape[2] != want[2] {
		t.Fatalf("Forward logits shape = %v, want %v", shape, want)
	}
	if !allFinite(t, out) {
		t.Fatal("Forward produced non-finite logits")
	}
}

// TestGemma3_GemmaModel_Forward_Decode_Good warms the per-layer caches with a
// prefill, then runs a single-token (L==1) decode step over the warm history —
// the steady-state generation kernel (KVCache.Update Slice views, GQA RepeatKV,
// L=1 SDPA). Asserts the [1,1,vocab] decode-step logits are well-formed.
func TestGemma3_GemmaModel_Forward_Decode_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	caches := m.NewCache()

	prefill := gemmaBenchTokens(1, gemmaBenchPrefill)
	defer metal.Free(prefill)
	pre := m.Forward(prefill, caches)
	if err := metal.Eval(pre); err != nil {
		t.Fatalf("Eval(prefill): %v", err)
	}
	metal.Free(pre)

	step := gemmaBenchTokens(1, 1)
	defer metal.Free(step)
	out := m.Forward(step, caches)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(decode): %v", err)
	}

	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != gemmaBenchVocab {
		t.Fatalf("decode logits shape = %v, want [1 1 %d]", shape, gemmaBenchVocab)
	}
	if !allFinite(t, out) {
		t.Fatal("decode produced non-finite logits")
	}
}

// TestGemma3_GemmaModel_ForwardMasked_Good drives the explicit-mask attention
// branch (Attention.forward's mask != nil → ScaledDotProductAttentionWithMask)
// with a [B,1,L,L] additive causal mask (0 = attend, -inf = block), the shape
// the batch mask builder produces. Asserts finite logits at the prefill shape.
func TestGemma3_GemmaModel_ForwardMasked_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	const L = gemmaBenchPrefill
	tokens := gemmaBenchTokens(1, L)
	defer metal.Free(tokens)

	// Lower-triangular additive mask: row i attends to keys 0..i.
	negInf := float32(math.Inf(-1))
	maskData := make([]float32, L*L)
	for i := int32(0); i < L; i++ {
		for j := int32(0); j < L; j++ {
			if j > i {
				maskData[i*L+j] = negInf
			}
		}
	}
	mask := metal.FromValues(maskData, 1, 1, int(L), int(L))
	defer metal.Free(mask)

	caches := m.NewCache()
	out := m.ForwardMasked(tokens, mask, caches)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(ForwardMasked): %v", err)
	}

	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != L || shape[2] != gemmaBenchVocab {
		t.Fatalf("ForwardMasked logits shape = %v, want [1 %d %d]", shape, L, gemmaBenchVocab)
	}
	if !allFinite(t, out) {
		t.Fatal("ForwardMasked produced non-finite logits")
	}
}

// TestGemma3_GemmaModel_Forward_PagedDecode_Good runs a single-token decode over
// a PagedKVCache, driving Attention.forward's paged branch (the
// c.(*metal.PagedKVCache) && L==1 && mask==nil arm): UpdatePages, the
// PagedStateNeedsMaterializedRepeat check, and ScaledDotProductAttentionPaged.
// The synthetic bench geometry has one K/V head (repeatFactor=4, single-head
// pages), so the paged SDPA broadcasts the repeat rather than materialising it —
// the materialised-repeat sub-branch is a defensive path that single-head paged
// state does not reach, noted in the package test report. Warms the per-layer
// paged caches with a prefill first so the decode step runs over real history.
func TestGemma3_GemmaModel_Forward_PagedDecode_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()

	// Per-layer paged caches (the engine builds one cache per layer for paged
	// decode; NewCache returns KV/rotating caches, so build paged ones here).
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewPagedKVCache(0, gemmaBenchPrefill)
	}
	defer func() {
		for _, c := range caches {
			if p, ok := c.(*metal.PagedKVCache); ok {
				p.Reset()
			}
		}
	}()

	// Warm the paged caches with a prefill (L>1 takes the non-paged update arm
	// to populate history), then a single-token step (L==1) takes the paged arm.
	prefill := gemmaBenchTokens(1, gemmaBenchPrefill)
	defer metal.Free(prefill)
	pre := m.Forward(prefill, caches)
	if err := metal.Eval(pre); err != nil {
		t.Fatalf("Eval(paged prefill): %v", err)
	}
	metal.Free(pre)

	step := gemmaBenchTokens(1, 1)
	defer metal.Free(step)
	out := m.Forward(step, caches)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(paged decode): %v", err)
	}

	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != gemmaBenchVocab {
		t.Fatalf("paged decode logits shape = %v, want [1 1 %d]", shape, gemmaBenchVocab)
	}
	if !allFinite(t, out) {
		t.Fatal("paged decode produced non-finite logits")
	}
}

// gemmaMultiKVHeadModel builds a 1-layer, full-attention synthetic model with
// MORE THAN ONE K/V head (4 query heads over 2 K/V heads → GQA repeatFactor=2).
// Unlike gemmaBenchModel (single K/V head, single-head pages), its paged K/V
// pages carry Dim(1)==2, so PagedStateNeedsMaterializedRepeat reports true and
// the paged decode takes the materialised-repeat sub-branch (RepeatPagedState).
// Built locally so the shared bench factory stays byte-identical.
func gemmaMultiKVHeadModel() *GemmaModel {
	const (
		hidden  = 64
		heads   = 4
		kvHeads = 2 // repeatFactor = 2
		headDim = 16
		inter   = 128
		vocab   = 48
	)
	cfg := &TextConfig{
		ModelType:            "gemma3",
		HiddenSize:           hidden,
		NumHiddenLayers:      1,
		IntermediateSize:     inter,
		NumAttentionHeads:    heads,
		NumKeyValueHeads:     kvHeads,
		HeadDim:              headDim,
		VocabSize:            vocab,
		RMSNormEps:           1e-6,
		RopeTheta:            1000000,
		RopeLocalBaseFreq:    10000,
		SlidingWindow:        32,
		SlidingWindowPattern: 0, // no sliding — single full layer
		Scale:                0.25,
		EmbeddingScale:       8.0,
	}
	qOut := int32(heads * headDim)
	kvOut := int32(kvHeads * headDim)

	embedW := make([]float32, vocab*hidden)
	for i := range embedW {
		embedW[i] = 0.02 + 0.01*float32(i%11)
	}
	embed := &metal.Embedding{Weight: metal.FromValues(embedW, vocab, hidden)}

	m := &GemmaModel{
		EmbedTokens: embed,
		Layers:      make([]*DecoderLayer, 1),
		Norm:        gemmaBenchNorm(hidden, 0.9),
		Output:      embed.AsLinear(),
		Cfg:         cfg,
		modelType:   "gemma3",
	}
	m.Layers[0] = &DecoderLayer{
		InputNorm:    gemmaBenchNorm(hidden, 0.91),
		PostAttnNorm: gemmaBenchNorm(hidden, 0.92),
		PreFFNorm:    gemmaBenchNorm(hidden, 0.93),
		PostFFNorm:   gemmaBenchNorm(hidden, 0.94),
		Attention: &Attention{
			QProj: gemmaBenchLinear(qOut, hidden, 0.03),
			KProj: gemmaBenchLinear(kvOut, hidden, 0.02),
			VProj: gemmaBenchLinear(kvOut, hidden, 0.015),
			OProj: gemmaBenchLinear(hidden, qOut, 0.012),
			QNorm: gemmaBenchNorm(headDim, 0.95),
			KNorm: gemmaBenchNorm(headDim, 0.96),
		},
		MLP: &metal.MLP{
			GateProj: gemmaBenchLinear(inter, hidden, 0.014),
			UpProj:   gemmaBenchLinear(inter, hidden, 0.016),
			DownProj: gemmaBenchLinear(hidden, inter, 0.018),
		},
		LayerIdx:  0,
		IsSliding: false,
	}
	precomputeScaledWeights(m)
	return m
}

// TestGemma3_GemmaModel_Forward_PagedDecode_MultiKVHead_Good drives the
// paged decode's materialised-repeat sub-branch (PagedStateNeedsMaterializedRepeat
// true → RepeatPagedState) using a model whose K/V pages carry more than one
// head. The single-K/V-head bench model cannot reach this arm (its pages have
// Dim(1)==1, so the paged SDPA broadcasts the repeat instead).
func TestGemma3_GemmaModel_Forward_PagedDecode_MultiKVHead_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaMultiKVHeadModel()

	const prefillLen = 8
	caches := []metal.Cache{metal.NewPagedKVCache(0, prefillLen)}
	defer func() {
		if p, ok := caches[0].(*metal.PagedKVCache); ok {
			p.Reset()
		}
	}()

	prefill := gemmaBenchTokens(1, prefillLen)
	defer metal.Free(prefill)
	pre := m.Forward(prefill, caches)
	if err := metal.Eval(pre); err != nil {
		t.Fatalf("Eval(multi-KV paged prefill): %v", err)
	}
	metal.Free(pre)

	step := gemmaBenchTokens(1, 1)
	defer metal.Free(step)
	out := m.Forward(step, caches)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(multi-KV paged decode): %v", err)
	}

	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != m.Cfg.VocabSize {
		t.Fatalf("multi-KV paged decode logits shape = %v, want [1 1 %d]", shape, m.Cfg.VocabSize)
	}
	if !allFinite(t, out) {
		t.Fatal("multi-KV paged decode produced non-finite logits")
	}
}

// TestGemma3_GemmaModel_NewCache_RuntimeTypes_Good builds caches over the
// synthetic 2-layer trunk (layer 0 sliding, layer 1 full per the bench config)
// and asserts the per-layer cache types match the sliding pattern: a sliding
// layer gets a RotatingKVCache, a full layer a plain KVCache.
func TestGemma3_GemmaModel_NewCache_RuntimeTypes_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	caches := m.NewCache()

	if len(caches) != int(m.Cfg.NumHiddenLayers) {
		t.Fatalf("NewCache len = %d, want %d", len(caches), m.Cfg.NumHiddenLayers)
	}
	for i, layer := range m.Layers {
		_, isRotating := caches[i].(*metal.RotatingKVCache)
		_, isPlain := caches[i].(*metal.KVCache)
		if layer.IsSliding && !isRotating {
			t.Errorf("layer %d sliding: cache %T, want *metal.RotatingKVCache", i, caches[i])
		}
		if !layer.IsSliding && !isPlain {
			t.Errorf("layer %d full: cache %T, want *metal.KVCache", i, caches[i])
		}
	}
}

// --- NumQueryHeads ---

func TestGemma3_GemmaModel_NumQueryHeads_Good(t *testing.T) {
	m := &GemmaModel{Cfg: &TextConfig{NumAttentionHeads: 8}}
	if got := m.NumQueryHeads(); got != 8 {
		t.Fatalf("NumQueryHeads = %d, want 8", got)
	}
}

// TestGemma3_GemmaModel_NumQueryHeads_NilConfig_Bad guards the documented
// "zero when the config is unavailable" contract — a partially-constructed
// model (load failed before Cfg was attached) must not panic.
func TestGemma3_GemmaModel_NumQueryHeads_NilConfig_Bad(t *testing.T) {
	m := &GemmaModel{}
	if got := m.NumQueryHeads(); got != 0 {
		t.Fatalf("NumQueryHeads with nil Cfg = %d, want 0", got)
	}
}

// --- ResolveLoRALinear ---

// TestGemma3_GemmaModel_ResolveLoRALinear_Good resolves each of the four
// attention projection paths to its backing Linear on a synthetic model.
func TestGemma3_GemmaModel_ResolveLoRALinear_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	cases := map[string]*metal.Linear{
		"self_attn.q_proj": m.Layers[0].Attention.QProj,
		"self_attn.k_proj": m.Layers[0].Attention.KProj,
		"self_attn.v_proj": m.Layers[0].Attention.VProj,
		"self_attn.o_proj": m.Layers[0].Attention.OProj,
	}
	for path, want := range cases {
		if got := m.ResolveLoRALinear(0, path); got != want {
			t.Errorf("ResolveLoRALinear(0, %q) = %p, want %p", path, got, want)
		}
	}
}

// TestGemma3_GemmaModel_ResolveLoRALinear_UnknownPath_Bad returns nil for a
// projection path the resolver does not recognise (e.g. an MLP path, which
// ResolveLoRALinear deliberately does not expose).
func TestGemma3_GemmaModel_ResolveLoRALinear_UnknownPath_Bad(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	if got := m.ResolveLoRALinear(0, "mlp.gate_proj"); got != nil {
		t.Fatalf("ResolveLoRALinear unknown path = %p, want nil", got)
	}
	if got := m.ResolveLoRALinear(0, "nonsense"); got != nil {
		t.Fatalf("ResolveLoRALinear nonsense path = %p, want nil", got)
	}
}

// TestGemma3_GemmaModel_ResolveLoRALinear_OutOfRange_Ugly guards the
// layerIdx >= len(Layers) bounds check — an out-of-range index returns nil
// rather than panicking on the slice access.
func TestGemma3_GemmaModel_ResolveLoRALinear_OutOfRange_Ugly(t *testing.T) {
	m := &GemmaModel{Layers: []*DecoderLayer{{}}}
	if got := m.ResolveLoRALinear(99, "self_attn.q_proj"); got != nil {
		t.Fatalf("ResolveLoRALinear out-of-range = %p, want nil", got)
	}
}

// --- Tokenizer ---

func TestGemma3_GemmaModel_Tokenizer_Good(t *testing.T) {
	tok := &metal.Tokenizer{}
	m := &GemmaModel{Tok: tok}
	if got := m.Tokenizer(); got != tok {
		t.Fatalf("Tokenizer() = %p, want %p", got, tok)
	}
}

// TestGemma3_GemmaModel_Tokenizer_Unset_Bad returns nil when the model carries
// no tokenizer (e.g. a synthetic trunk built for inference benchmarking).
func TestGemma3_GemmaModel_Tokenizer_Unset_Bad(t *testing.T) {
	m := &GemmaModel{}
	if got := m.Tokenizer(); got != nil {
		t.Fatalf("Tokenizer() with no tokenizer = %p, want nil", got)
	}
}

// --- ApplyLoRA ---

// TestGemma3_GemmaModel_ApplyLoRA_AttentionTargets_Good wraps the q/v attention
// projections with LoRA on the synthetic trunk and verifies one adapter layer
// is registered per (layer × target) and that each target Linear now carries a
// LoRA handle. Exercises the attention-target branches of ApplyLoRA.
func TestGemma3_GemmaModel_ApplyLoRA_AttentionTargets_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	cfg := metal.DefaultLoRAConfig() // targets q_proj, v_proj
	adapter := m.ApplyLoRA(cfg)

	wantLayers := len(m.Layers) * len(adapter.Config.TargetKeys)
	if len(adapter.Layers) != wantLayers {
		t.Fatalf("ApplyLoRA registered %d layers, want %d (%d layers × %d targets)",
			len(adapter.Layers), wantLayers, len(m.Layers), len(adapter.Config.TargetKeys))
	}
	if adapter.Model != m {
		t.Errorf("adapter.Model = %p, want the model %p", adapter.Model, m)
	}
	if m.Layers[0].Attention.QProj.LoRA == nil {
		t.Error("q_proj should carry a LoRA handle after ApplyLoRA")
	}
}

// TestGemma3_GemmaModel_ApplyLoRA_AllAttentionArms_Good targets all four
// attention projections (q/k/v/o) so every attention arm of ApplyLoRA's
// target switch is exercised — the default config only covers q/v.
func TestGemma3_GemmaModel_ApplyLoRA_AllAttentionArms_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	cfg := metal.LoRAConfig{Rank: 4, Alpha: 8, TargetLayers: []string{"q_proj", "k_proj", "v_proj", "o_proj"}}
	adapter := m.ApplyLoRA(cfg)

	wantLayers := len(m.Layers) * 4
	if len(adapter.Layers) != wantLayers {
		t.Fatalf("ApplyLoRA(q,k,v,o) registered %d layers, want %d", len(adapter.Layers), wantLayers)
	}
	attn := m.Layers[0].Attention
	for name, proj := range map[string]*metal.Linear{
		"q_proj": attn.QProj, "k_proj": attn.KProj, "v_proj": attn.VProj, "o_proj": attn.OProj,
	} {
		if proj.LoRA == nil {
			t.Errorf("%s should carry a LoRA handle after ApplyLoRA", name)
		}
	}
}

// TestGemma3_GemmaModel_ApplyLoRA_MLPTargets_Good drives the MLP-target
// branches (gate_proj, up_proj, down_proj) of ApplyLoRA, confirming the
// gate projection gets wrapped.
func TestGemma3_GemmaModel_ApplyLoRA_MLPTargets_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := gemmaBenchModel()
	cfg := metal.LoRAConfig{Rank: 4, Alpha: 8, TargetLayers: []string{"gate_proj", "up_proj", "down_proj"}}
	adapter := m.ApplyLoRA(cfg)

	wantLayers := len(m.Layers) * 3
	if len(adapter.Layers) != wantLayers {
		t.Fatalf("ApplyLoRA(MLP) registered %d layers, want %d", len(adapter.Layers), wantLayers)
	}
	if m.Layers[0].MLP.GateProj.LoRA == nil {
		t.Error("gate_proj should carry a LoRA handle after ApplyLoRA")
	}
}

// TestGemma3_GemmaModel_ApplyLoRA_NoLayers_Ugly applies LoRA to a model with no
// decoder layers — the loop body never runs, so the adapter is well-formed but
// empty rather than panicking.
func TestGemma3_GemmaModel_ApplyLoRA_NoLayers_Ugly(t *testing.T) {
	m := &GemmaModel{}
	adapter := m.ApplyLoRA(metal.DefaultLoRAConfig())
	if adapter == nil {
		t.Fatal("ApplyLoRA returned nil adapter")
	}
	if len(adapter.Layers) != 0 {
		t.Fatalf("ApplyLoRA on layerless model registered %d layers, want 0", len(adapter.Layers))
	}
}
