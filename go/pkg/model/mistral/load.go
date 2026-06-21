// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// load.go is the backend-agnostic Mistral weight assembler — the sibling of gemma4's, producing the
// SAME neutral model.LoadedModel a backend consumes. Ministral-3 is a standard Mistral transformer (a
// gemma4 SUBSET): two norms per layer and none of gemma4's extras (QK-norm, post-attention / post-FF
// norms, layer-scalar, PLE, MoE). The one mapping difference is the pre-MLP norm — Mistral names it
// post_attention_layernorm (gemma4 names it pre_feedforward_layernorm) — so it lands in MLPNorm and
// the gemma4-only fields stay nil. Real packs are the Mistral3ForConditionalGeneration wrapper (text
// weights under language_model.*); model.NormalizeWrapperNames strips it. Dense bf16 scope today (FP8
// + vision are later slices); the per-weight LoadLinear .scales decision keeps it quant-agnostic.

// Load reads a Mistral checkpoint directory — config.json → Arch, safetensors shards (zero-copy mmap)
// → the weight set — returning the LoadedModel + the DirMapping the byte views reference (the caller
// Closes it after binding device buffers). Registered for the Mistral model_type ids (register.go).
func Load(dir string) (*model.LoadedModel, *safetensors.DirMapping, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, nil, core.E("mistral.Load", "read config.json", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		return nil, nil, core.NewError("mistral.Load: config.json parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, nil, err
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, nil, err
	}
	m, err := Assemble(dm.Tensors, arch)
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	return m, dm, nil
}

// Assemble maps a Mistral tensor set onto the neutral LoadedModel per the Arch's dims. The pre-MLP
// norm comes from post_attention_layernorm (Mistral's name for it); gemma4's extras have no Mistral
// tensors and stay nil — the arch executor skips an absent norm, so the assembled model decodes the
// faithful Mistral layer: rms(input) → attn → +residual → rms(post_attn) → SwiGLU → +residual.
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
		return nil, core.NewError("mistral.Assemble: model.embed_tokens.weight absent")
	}
	m.LMHead = lin("lm_head", d) // nil ⇒ tied to Embed

	m.Layers = make([]model.LoadedLayer, len(arch.Layer))
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		spec := arch.Layer[i]
		qDim := arch.Heads * spec.HeadDim // o_proj input width
		L := &m.Layers[i]
		L.AttnNorm = norm(p + ".input_layernorm.weight")
		L.Q = lin(p+".self_attn.q_proj", d)
		if spec.OwnsCache() {
			L.K = lin(p+".self_attn.k_proj", d)
			L.V = lin(p+".self_attn.v_proj", d)
		}
		L.O = lin(p+".self_attn.o_proj", qDim)
		L.MLPNorm = norm(p + ".post_attention_layernorm.weight") // Mistral's pre-MLP norm
		L.Gate = lin(p+".mlp.gate_proj", d)
		L.Up = lin(p+".mlp.up_proj", d)
		ff := arch.FF
		if L.Gate != nil { // (uniform for Mistral; read from shapes for parity with gemma4 MatFormer)
			ff = L.Gate.OutDim
		}
		L.Down = lin(p+".mlp.down_proj", ff)
	}
	if err := m.ValidateRequired(arch); err != nil {
		return nil, err
	}
	return m, nil
}
