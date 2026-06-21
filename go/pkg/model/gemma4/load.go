// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// load.go is the backend-agnostic, quant-agnostic gemma4 weight assembler — the loading that
// pkg/model was missing (the reference is pkg/metal/model/gemma4's loader, READ not edited). It
// walks the Arch and turns each weight into a model.Linear (bf16 or quant decided PER WEIGHT by
// .scales, geometry read from the shapes) or raw norm bytes, viewing the source mmap. A backend
// (pkg/native, future go-rocm) consumes the model.LoadedModel + uploads the byte views to its device;
// it never re-parses config, re-derives the arch, or hand-codes a per-weight quant decision.
//
// This is why e2b / e4b — 4-bit, qat-4-bit (a quantised per_layer_model_projection where e2b keeps
// it bf16), MatFormer per-layer FFN, KV-shared layers — all assemble through ONE path: nothing here
// assumes a weight's format or a uniform FFN width.

// The neutral loaded-weights types (model.LoadedModel / model.LoadedLayer / model.LoadedMoE) + Tied / ValidateRequired
// live at the pkg/model root; this file is just gemma4's mapping from its tensor names onto them.

// Load reads a gemma4 checkpoint directory — config.json → Arch, the safetensors shards
// (zero-copy mmap) → the weight set — and returns the model.LoadedModel plus the DirMapping whose mmap
// the model's byte views reference (the caller MUST Close it after binding GPU buffers / when
// done). The single on-disk entry both backends share.
func Load(dir string) (*model.LoadedModel, *safetensors.DirMapping, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, nil, core.E("gemma4.Load", "read config.json", err)
	}
	cfg, err := parseGemma4Config([]byte(cfgStr)) // the faithful parse: wrapper-merge + validation + don't-guess
	if err != nil {
		return nil, nil, err
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, nil, err
	}
	// Resolve the dims metal's loader reads from the weight SHAPES (buildGemma4FromWeights) — read,
	// never guess: head_dim from the q_proj rows, vocab from the embedding rows, PLE size from the
	// per-layer projection, with the hidden/heads fallback only as a last resort.
	t := model.NormalizeWrapperNames(dm.Tensors)
	if inferred := inferGemma4HeadDim(t, cfg.LayerTypes, int(cfg.NumAttentionHeads), "sliding_attention"); inferred > 0 {
		cfg.HeadDim = int32(inferred)
	}
	if inferred := inferGemma4HeadDim(t, cfg.LayerTypes, int(cfg.NumAttentionHeads), "full_attention"); inferred > 0 {
		cfg.GlobalHeadDim = int32(inferred)
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.VocabSize == 0 {
		if w, ok := model.WeightAny(t, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(w.Shape) > 0 && w.Shape[0] > 0 {
			cfg.VocabSize = int32(w.Shape[0])
		}
	}
	if cfg.VocabSizePerLayerInput == 0 {
		cfg.VocabSizePerLayerInput = cfg.VocabSize
	}
	if inferred := inferGemma4PerLayerInputSize(t, int(cfg.NumHiddenLayers)); inferred > 0 {
		cfg.HiddenSizePerLayerInput = int32(inferred)
	}
	if cfg.HiddenSizePerLayerInput > 0 {
		_, e1 := model.WeightAny(t, "model.embed_tokens_per_layer.weight")
		_, e2 := model.WeightAny(t, "model.per_layer_model_projection.weight")
		_, e3 := model.WeightAny(t, "model.per_layer_projection_norm.weight")
		if !e1 || !e2 || !e3 {
			cfg.HiddenSizePerLayerInput = 0
		}
	}
	gemma4FinaliseEmbeddingScales(cfg) // re-cache against the resolved dims (matches metal load.go)
	arch, err := cfg.Arch()
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	m, err := Assemble(dm.Tensors, arch)
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	return m, dm, nil
}

// Assemble builds the model.LoadedModel from a safetensors tensor set + the derived Arch. The quant
// decision is per-weight (model.LoadLinear keys on .scales and reads the affine geometry from the
// shapes), so no quantization block is needed here — bf16 / 4 / 5 / 6 / 8-bit and mixed all work.
// gemma4 packs are affine; the kind is fixed to "affine" (the registered native/metal QuantMatVec).
func Assemble(tensors map[string]safetensors.Tensor, arch model.Arch) (*model.LoadedModel, error) {
	const kind = "affine"
	t := model.NormalizeWrapperNames(tensors)
	d := arch.Hidden
	lin := func(prefix string, inDim int) *model.Linear { return model.LoadLinear(t, prefix, inDim, kind) }
	norm := func(name string) []byte {
		if x, ok := t[name]; ok {
			return x.Data
		}
		return nil
	}

	m := &model.LoadedModel{Arch: arch, FinalNorm: norm("model.norm.weight")}
	m.Embed = lin("model.embed_tokens", d)
	if m.Embed == nil {
		return nil, core.NewError("gemma4.Assemble: model.embed_tokens.weight absent")
	}
	m.LMHead = lin("lm_head", d) // nil ⇒ tied to Embed

	if arch.PerLayerInputHidden > 0 {
		plDim := len(arch.Layer) * arch.PerLayerInputHidden
		m.EmbedPerLayer = lin("model.embed_tokens_per_layer", plDim)
		m.PerLayerModelProj = lin("model.per_layer_model_projection", d) // bf16 (e2b) or 4-bit (e4b) — same path
		m.PerLayerProjNorm = norm("model.per_layer_projection_norm.weight")
	}

	m.Layers = make([]model.LoadedLayer, len(arch.Layer))
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		spec := arch.Layer[i]
		qDim := arch.Heads * spec.HeadDim // o_proj input width (per-layer: global layers have a larger head_dim)
		L := &m.Layers[i]
		L.AttnNorm = norm(p + ".input_layernorm.weight")
		L.PostAttnNorm = norm(p + ".post_attention_layernorm.weight")
		L.QNorm = norm(p + ".self_attn.q_norm.weight")
		L.KNorm = norm(p + ".self_attn.k_norm.weight")
		L.LayerScalar = norm(p + ".layer_scalar")
		L.Q = lin(p+".self_attn.q_proj", d)
		// KV-shared layers carry no own k/v_proj (they read the owner's cache). v_proj is also
		// absent on K==V layers — LoadLinear returns nil for either, no special case needed.
		if spec.OwnsCache() {
			L.K = lin(p+".self_attn.k_proj", d)
			L.V = lin(p+".self_attn.v_proj", d)
		}
		L.O = lin(p+".self_attn.o_proj", qDim)

		if spec.MoE {
			L.MoE = assembleMoE(t, p, arch, lin, norm)
		} else {
			L.MLPNorm = norm(p + ".pre_feedforward_layernorm.weight")
			L.Gate = lin(p+".mlp.gate_proj", d)
			L.Up = lin(p+".mlp.up_proj", d)
			ff := arch.FF
			if L.Gate != nil { // per-layer FFN width (MatFormer): read from the gate's output rows
				ff = L.Gate.OutDim
			}
			L.Down = lin(p+".mlp.down_proj", ff)
			L.PostFFNorm = norm(p + ".post_feedforward_layernorm.weight")
		}

		if arch.PerLayerInputHidden > 0 {
			L.PerLayerGate = lin(p+".per_layer_input_gate", d)
			L.PerLayerProjection = lin(p+".per_layer_projection", arch.PerLayerInputHidden)
			L.PostPerLayerInputNorm = norm(p + ".post_per_layer_input_norm.weight")
		}
	}
	if err := m.ValidateRequired(arch); err != nil {
		return nil, err
	}
	return m, nil
}

// assembleMoE builds a gemma4 MoE layer's dual-branch FFN (local dense MLP + sparse experts).
func assembleMoE(t map[string]safetensors.Tensor, p string, arch model.Arch, lin func(string, int) *model.Linear, norm func(string) []byte) *model.LoadedMoE {
	d := arch.Hidden
	return &model.LoadedMoE{
		PreFFNorm:      norm(p + ".pre_feedforward_layernorm.weight"),
		PreFFNorm2:     norm(p + ".pre_feedforward_layernorm_2.weight"),
		PostFFNorm1:    norm(p + ".post_feedforward_layernorm_1.weight"),
		PostFFNorm2:    norm(p + ".post_feedforward_layernorm_2.weight"),
		PostFFNorm:     norm(p + ".post_feedforward_layernorm.weight"),
		RouterScale:    norm(p + ".router.scale"),
		PerExpertScale: norm(p + ".router.per_expert_scale"),
		LocalGate:      lin(p+".mlp.gate_proj", d),
		LocalUp:        lin(p+".mlp.up_proj", d),
		LocalDown:      lin(p+".mlp.down_proj", arch.ExpertFF),
		Router:         lin(p+".router.proj", d),
		ExpGate:        lin(p+".experts.switch_glu.gate_proj", d),
		ExpUp:          lin(p+".experts.switch_glu.up_proj", d),
		ExpDown:        lin(p+".experts.switch_glu.down_proj", arch.ExpertFF),
	}
}
