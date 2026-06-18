// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"path/filepath"
	"testing"
)

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
