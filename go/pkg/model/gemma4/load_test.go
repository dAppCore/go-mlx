// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
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
	if _, err := Assemble(ts, arch); err != nil {
		t.Fatalf("Assemble of a complete set: %v", err)
	}
	delete(ts, "model.layers.0.self_attn.q_proj.weight")
	if _, err := Assemble(ts, arch); err == nil {
		t.Fatal("expected an error on a missing required q_proj")
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
			m, dm, err := Load(dir)
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
