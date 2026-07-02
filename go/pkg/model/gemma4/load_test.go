// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// minimalGemma4Tensors builds a complete dense bf16 gemma4 tensor set for arch — just the required
// weights at the right shapes, distinct fills not needed (the validation only checks presence).
func minimalGemma4Tensors(arch model.Arch) map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{}
	bf := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
	}
	d := arch.Hidden
	ts["model.embed_tokens.weight"] = mat(arch.Vocab, d)
	ts["model.norm.weight"] = bf(d)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		ts[p+".input_layernorm.weight"] = bf(d)
		ts[p+".self_attn.q_proj.weight"] = mat(arch.Heads*arch.HeadDim, d)
		ts[p+".self_attn.k_proj.weight"] = mat(arch.KVHeads*arch.HeadDim, d)
		ts[p+".self_attn.v_proj.weight"] = mat(arch.KVHeads*arch.HeadDim, d)
		ts[p+".self_attn.o_proj.weight"] = mat(d, arch.Heads*arch.HeadDim)
		ts[p+".pre_feedforward_layernorm.weight"] = bf(d)
		ts[p+".mlp.gate_proj.weight"] = mat(arch.FF, d)
		ts[p+".mlp.up_proj.weight"] = mat(arch.FF, d)
		ts[p+".mlp.down_proj.weight"] = mat(d, arch.FF)
		ts[p+".post_feedforward_layernorm.weight"] = bf(d)
	}
	return ts
}

// TestAssembleValidatesRequired gates the presence validation: a complete set assembles, and a set
// missing a required weight (q_proj) is rejected with a clean error rather than a nil-deref later.
func TestAssembleValidatesRequired(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)
	if _, err := gemma4Assemble(ts, arch); err != nil {
		t.Fatalf("Assemble of a complete set: %v", err)
	}
	delete(ts, "model.layers.0.self_attn.q_proj.weight")
	if _, err := gemma4Assemble(ts, arch); err == nil {
		t.Fatal("expected an error on a missing required q_proj")
	}
}

func TestLoadDiffusionGemmaDecoderTrunk_Good(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, GlobalHeadDim: 16,
		VocabSize: 32, RMSNormEps: 1e-6, SlidingWindow: 4,
		LayerTypes: []string{"sliding_attention", "full_attention"},
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	decoder := make(map[string]safetensors.Tensor)
	for name, tensor := range minimalGemma4Tensors(arch) {
		decoder["model.decoder."+name] = tensor
	}
	bf := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
	}
	decoder["self_conditioning.pre_norm.weight"] = bf(64)
	decoder["self_conditioning.gate_proj.weight"] = mat(128, 64)
	decoder["self_conditioning.up_proj.weight"] = mat(128, 64)
	decoder["self_conditioning.down_proj.weight"] = mat(64, 128)
	decoder["model.encoder.language_model.layers.0.layer_scalar"] = bf(1)
	decoder["model.encoder.language_model.layers.1.layer_scalar"] = bf(1)

	dir := t.TempDir()
	configJSON := `{
		"model_type": "diffusion_gemma",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 16,
		"global_head_dim": 16,
		"vocab_size": 32,
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"canvas_length": 4,
		"eos_token_id": [1, 2],
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), configJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(decoder)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}

	loaded, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load(diffusion_gemma trunk): %v", err)
	}
	defer func() { _ = dm.Close() }()
	if loaded.Embed == nil || loaded.FinalNorm == nil || len(loaded.Layers) != 2 {
		t.Fatalf("loaded diffusion trunk incomplete: embed=%v finalNorm=%v layers=%d", loaded.Embed != nil, loaded.FinalNorm != nil, len(loaded.Layers))
	}
	if loaded.Layers[0].Q == nil || loaded.Layers[1].O == nil {
		t.Fatalf("loaded diffusion trunk missing attention projections: layer0.Q=%v layer1.O=%v", loaded.Layers[0].Q != nil, loaded.Layers[1].O != nil)
	}
	if loaded.Diffusion == nil {
		t.Fatalf("loaded diffusion trunk missing neutral diffusion extras")
	}
	if loaded.Diffusion.CanvasLength != 4 {
		t.Fatalf("diffusion canvas length = %d, want 4", loaded.Diffusion.CanvasLength)
	}
	if len(loaded.Diffusion.EOSTokens) != 2 || loaded.Diffusion.EOSTokens[0] != 1 || loaded.Diffusion.EOSTokens[1] != 2 {
		t.Fatalf("diffusion eos tokens = %v, want [1 2]", loaded.Diffusion.EOSTokens)
	}
	if len(loaded.Diffusion.EncoderLayerScalars) != 2 {
		t.Fatalf("diffusion encoder scalars = %d, want 2", len(loaded.Diffusion.EncoderLayerScalars))
	}
	if loaded.Diffusion.SelfCondPreNorm == nil || loaded.Diffusion.SelfCondGate == nil || loaded.Diffusion.SelfCondUp == nil || loaded.Diffusion.SelfCondDown == nil {
		t.Fatalf("loaded diffusion self-conditioning block incomplete: %+v", loaded.Diffusion)
	}
}

// gemma4Snapshot resolves an HF-cache snapshot dir for repo, or "" when not cached.
func gemma4Snapshot(repo string) string {
	base := filepath.Join(os.Getenv("HOME"), ".cache/huggingface/hub", repo, "snapshots")
	ents, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, e := range ents {
		if e.IsDir() {
			d := filepath.Join(base, e.Name())
			if _, err := os.Stat(filepath.Join(d, "config.json")); err == nil {
				return d
			}
		}
	}
	return ""
}

// TestLoad_EFamily_QuantAgnostic loads e2b (4-bit) and e4b (qat-4-bit) through the SINGLE shared
// assembler and asserts the things native used to re-bug per model: KV-shared layers carry no own
// K, the MatFormer per-layer FFN width is read from the gate shape, and — the headline — e4b's
// per_layer_model_projection is seen as quantised while e2b's is bf16, with NO per-weight branch.
// AX-11: mmap metadata only, no compute / no GPU.
func TestLoad_EFamily_QuantAgnostic(t *testing.T) {
	cases := []struct {
		key, repo     string
		wantProjQuant bool // per_layer_model_projection: e2b bf16, e4b 4-bit (the bug case)
	}{
		{"e2b", "models--mlx-community--gemma-4-E2B-it-4bit", false},
		{"e4b", "models--mlx-community--gemma-4-E4B-it-qat-4bit", true},
	}
	for _, c := range cases {
		t.Run(c.key, func(t *testing.T) {
			dir := gemma4Snapshot(c.repo)
			if dir == "" {
				t.Skipf("%s not cached", c.key)
			}
			m, dm, err := model.Load(dir)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer dm.Close()

			if len(m.Layers) != len(m.Arch.Layer) {
				t.Fatalf("layers %d != arch %d", len(m.Layers), len(m.Arch.Layer))
			}
			if m.Embed == nil || m.FinalNorm == nil {
				t.Fatal("embed or final norm missing")
			}
			owners, ffs := 0, map[int]int{}
			for i, L := range m.Layers {
				spec := m.Arch.Layer[i]
				if L.Q == nil || L.AttnNorm == nil {
					t.Fatalf("layer %d missing Q / AttnNorm", i)
				}
				if spec.OwnsCache() {
					owners++
					if L.K == nil {
						t.Fatalf("cache-owner layer %d missing K", i)
					}
				} else if L.K != nil {
					t.Fatalf("KV-shared layer %d has its own K — KV-share broken", i)
				}
				if L.MoE == nil { // dense MLP
					if L.Gate == nil || L.Gate.OutDim <= 0 {
						t.Fatalf("layer %d gate FFN width not read from shape", i)
					}
					ffs[L.Gate.OutDim]++
				}
			}
			if m.PerLayerModelProj == nil {
				t.Fatal("PLE per_layer_model_projection missing")
			}
			if got := m.PerLayerModelProj.Quantised(); got != c.wantProjQuant {
				t.Fatalf("per_layer_model_projection Quantised()=%v, want %v", got, c.wantProjQuant)
			}
			t.Logf("%s: %d layers · %d cache owners (%d shared) · FFN widths %v · PLE-proj quantised=%v",
				c.key, len(m.Layers), owners, len(m.Layers)-owners, ffs, m.PerLayerModelProj.Quantised())
		})
	}
}
