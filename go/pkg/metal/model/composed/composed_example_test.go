// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package composed

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// composed_example_test.go is the usage-in-situ companion to composed_test.go:
// runnable Example functions that double as documentation and coverage for the
// model's public surface. Each one composes the model the way a caller does and
// asserts a deterministic result via // Output. No checkpoint is loaded (AX-11);
// the in-memory ComposedModel is either a bare struct (for the pure accessors)
// or the same synthetic fixture buildComposed assembles from metal.FromValues
// weights that composed_test.go uses — never a model off disk.

// ExampleLoadComposed shows the disk entry point. It reads config.json,
// tokenizer.json and the safetensors weights under modelPath, so it cannot run
// without a checkpoint (AX-11: no model loads) — the body is compiled but never
// executed (the qwen3 ExampleLoad pattern). The disk-free composition core it
// wraps, buildComposed, is exercised directly by composed_test.go.
func ExampleLoadComposed() {
	model, err := LoadComposed("/path/to/a/composed-or-hybrid/checkpoint")
	_, _ = model, err
}

// ExampleComposedModel_Forward composes the 2-layer full-attention synthetic
// model the unit tests use and runs a prefill forward over a [B=1, L=2] token
// batch. ForwardMasked returns a lazy graph, so Materialize forces the actual
// compute before the shape is read; the result is logits [B, L, vocab]. This is
// the usage-in-situ for the whole orchestration — embedding → per-layer
// composedLayer.forward (norm → mixer → add, norm → MLP → add) → final norm →
// output projection — driven exactly as the generate loop drives it.
func ExampleComposedModel_Forward() {
	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	model, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		core.Println(err)
		return
	}
	defer model.CloseModel()

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	tokens := metal.FromValues([]int32{1, 2}, 1, 2)
	defer metal.Free(tokens)

	logits := model.Forward(tokens, caches)
	defer metal.Free(logits)
	metal.Materialize(logits)

	core.Println(logits.Shape())
	// Output: [1 2 6]
}

// ExampleComposedModel_ForwardMasked drives the same synthetic model with an
// explicit causal mask instead of the nil-mask default. The output shape is
// identical to the unmasked prefill — [B, L, vocab] — because the mask shapes
// attention scores, not the logits tensor. Passing nil here is the standard
// causal path Forward delegates to.
func ExampleComposedModel_ForwardMasked() {
	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	model, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		core.Println(err)
		return
	}
	defer model.CloseModel()

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	tokens := metal.FromValues([]int32{1, 2, 3}, 1, 3)
	defer metal.Free(tokens)

	logits := model.ForwardMasked(tokens, nil, caches) // nil mask = standard causal
	defer metal.Free(logits)
	metal.Materialize(logits)

	core.Println(logits.Shape())
	// Output: [1 3 6]
}

// ExampleComposedModel_NewCache shows the cache layout NewCache builds: one
// entry per layer, typed by each layer mixer's declared state. Every layer of
// the synthetic model is full-attention, so every cache is the growing KV cache
// (*metal.KVCache), 1:1 with the layers Forward will index.
func ExampleComposedModel_NewCache() {
	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	model, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		core.Println(err)
		return
	}
	defer model.CloseModel()

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.KVCache
}

// ExampleComposedModel_NewCache_nilLayer shows the defensive fallback: a layer
// that is nil (or carries no mixer) gets a plain KV cache rather than a panic,
// so NewCache always returns a usable cache per layer slot. This is the floor
// the generate loop relies on — caches stay 1:1 with layers even on a
// half-built model.
func ExampleComposedModel_NewCache_nilLayer() {
	model := &ComposedModel{Layers: []*composedLayer{nil, nil}}

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	core.Println(len(caches), core.Sprintf("%T", caches[0]))
	// Output: 2 *metal.KVCache
}

// ExampleComposedModel_NumLayers reports the transformer depth — the number of
// composedLayer blocks the model holds.
func ExampleComposedModel_NumLayers() {
	model := &ComposedModel{Layers: []*composedLayer{nil, nil, nil}}

	core.Println(model.NumLayers())
	// Output: 3
}

// ExampleComposedModel_NumQueryHeads reports the attention query-head count read
// from the config (the QueryHeadCounter the engine dispatches on). A model with
// no config reports 0 rather than panicking.
func ExampleComposedModel_NumQueryHeads() {
	model := &ComposedModel{Cfg: &metal.DenseConfig{}}
	model.Cfg.NumAttentionHeads = 16

	bare := &ComposedModel{} // no config → 0, not a nil-deref
	core.Println(model.NumQueryHeads(), bare.NumQueryHeads())
	// Output: 16 0
}

// ExampleComposedModel_ModelType returns the architecture id the config
// declared — what the registry resolved the model under ("composed", "hybrid",
// or a concrete family id).
func ExampleComposedModel_ModelType() {
	model := &ComposedModel{modelType: "hybrid"}

	core.Println(model.ModelType())
	// Output: hybrid
}

// ExampleComposedModel_Tokenizer returns the model's tokenizer. On a bare
// struct (no checkpoint loaded) it is nil — the tokenizer is populated by
// LoadComposed off disk — so a caller guards before use. Driving it on an empty
// model documents that nil-before-load reality without a model load (AX-11).
func ExampleComposedModel_Tokenizer() {
	model := &ComposedModel{} // no checkpoint loaded → tokenizer is nil
	tok := model.Tokenizer()

	core.Println(tok == nil)
	// Output: true
}

// ExampleComposedModel_FillModelInfo fills architecture metadata from the config
// (the ModelInfoReporter the engine queries) — vocab, hidden size and context
// length are copied straight from the dense config, and the quantization bits /
// group size are copied when the config declares quantization.
func ExampleComposedModel_FillModelInfo() {
	model := &ComposedModel{Cfg: &metal.DenseConfig{
		Quantization: &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	model.Cfg.VocabSize = 32000
	model.Cfg.HiddenSize = 512
	model.Cfg.MaxPositionEmbeddings = 8192

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)

	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength, info.QuantBits, info.QuantGroup)
	// Output: 32000 512 8192 4 64
}

// ExampleComposedModel_ApplyLoRA wraps the SwiGLU MLP projections the composed
// model owns directly. NormalizeLoRAConfig fills the canonical fields: an
// unset Alpha derives from Scale×Rank, TargetLayers seed TargetKeys. The
// adapter binds one LoRALinear per (layer, target) where the projection exists
// — here one synthetic layer with all three MLP projections and all three
// targets, so three adapter entries. The mixer is opaque, so only the MLP path
// is wrapped (attention-target LoRA on a composed mixer is a later refinement).
func ExampleComposedModel_ApplyLoRA() {
	model := &ComposedModel{
		Layers: []*composedLayer{
			{MLP: &metal.SiLUMLP{
				GateProj: metal.NewLinear(ramp(tInter, tHidden), nil),
				UpProj:   metal.NewLinear(ramp(tInter, tHidden), nil),
				DownProj: metal.NewLinear(ramp(tHidden, tInter), nil),
			}},
		},
	}

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         4,
		Scale:        2,
		TargetLayers: []string{"gate_proj", "up_proj", "down_proj"},
	})

	core.Println(adapter.Config.TargetKeys, adapter.Config.Rank, adapter.Config.Alpha, len(adapter.Layers))
	// Output: [gate_proj up_proj down_proj] 4 8 3
}

// ExampleComposedModel_CloseModel releases the model's Metal arrays. It is
// nil-safe (a nil model is a no-op) and skips an output projection that is the
// tied embedding (so the shared weight is freed once, via the embedding). Here
// a synthetic model is composed and closed — the usage-in-situ for the
// ModelCloser the engine calls at teardown.
func ExampleComposedModel_CloseModel() {
	cfg := composedTestConfig([]string{"full_attention", "full_attention"})
	model, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		core.Println(err)
		return
	}

	model.CloseModel()

	var nilModel *ComposedModel
	nilModel.CloseModel() // nil-safe: no panic

	core.Println("closed")
	// Output: closed
}

// ExampleComposedModel_heterogeneous composes a hybrid trunk — one softmax
// layer and one recurrent layer — through the registry, and shows NewCache
// typing each layer by its mixer's declared state: a KV cache for the softmax
// layer, the recurrent holder for the recurrent one. This is the composition
// the package exists for; the recurrent mixer here is a minimal stand-in that
// declares StateRecurrent.
func ExampleComposedModel_heterogeneous() {
	metal.RegisterMixerLoader("example_recurrent", func(metal.MixerBuildCtx) (metal.MixerCompute, error) {
		return exampleRecurrentMixer{}, nil
	})

	cfg := composedTestConfig([]string{"full_attention", "example_recurrent"})
	model, err := buildComposed(cfg, fullAttentionWeights(tLayers), nil)
	if err != nil {
		core.Println(err)
		return
	}
	defer model.CloseModel()

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	_, layer0Recurrent := caches[0].(metal.RecurrentCache)
	_, layer1Recurrent := caches[1].(metal.RecurrentCache)
	core.Println(model.Layers[0].Kind, layer0Recurrent, model.Layers[1].Kind, layer1Recurrent)
	// Output: full_attention false example_recurrent true
}

// exampleRecurrentMixer is a minimal StateRecurrent mixer for the heterogeneous
// example — it declares the recurrent state kind so NewCache gives its layer the
// recurrent holder. Forward is never exercised (the example asserts cache typing).
type exampleRecurrentMixer struct{}

func (exampleRecurrentMixer) Kind() string            { return "example_recurrent" }
func (exampleRecurrentMixer) State() scheme.StateKind { return scheme.StateRecurrent }
func (exampleRecurrentMixer) Forward(x *metal.Array, _ *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	return x, metal.SharedKV{}
}
