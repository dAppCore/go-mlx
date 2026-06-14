// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Tiny hand-traceable fuse oracle (rank 1, out 2, in 3, scale 2):
//
//	W     = [[1,2,3],[4,5,6]]
//	A     = [[0.1,0.2,0.3]]            (rank 1 × in 3)
//	B     = [[2],[3]]                  (out 2 × rank 1)
//	B·A   = [[0.2,0.4,0.6],[0.3,0.6,0.9]]
//	delta = (B·A)·2 = [[0.4,0.8,1.2],[0.6,1.2,1.8]]
//	W+delta        = [[1.4,2.8,4.2],[4.6,6.2,7.8]]
//
// The expected values are computed independently of FuseLoRAIntoWeights, so the
// test is a real oracle (a wrong matmul orientation or a dropped scale fails it).
const fuseLayer = "model.layers.0.self_attn.q_proj"

func fuseBaseAdapter() (base, adapter map[string]*Array) {
	base = map[string]*Array{
		fuseLayer + ".weight": FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		"model.norm.weight":   FromValues([]float32{7, 8, 9}, 3), // untargeted carry-through
	}
	adapter = map[string]*Array{
		fuseLayer + ".lora_a": FromValues([]float32{0.1, 0.2, 0.3}, 1, 3),
		fuseLayer + ".lora_b": FromValues([]float32{2, 3}, 2, 1),
	}
	return base, adapter
}

// TestFuseLoRA_Merge_Good is the numeric oracle: the targeted weight comes back
// as W + (B·A)·scale, the fused-layer list is reported, and an untargeted tensor
// passes through untouched.
func TestFuseLoRA_Merge_Good(t *testing.T) {
	base, adapter := fuseBaseAdapter()
	merged, fused, err := FuseLoRAIntoWeights(base, adapter, 2.0)
	if err != nil {
		t.Fatalf("FuseLoRAIntoWeights: %v", err)
	}

	if len(fused) != 1 || fused[0] != fuseLayer {
		t.Fatalf("fused layers = %v, want [%s]", fused, fuseLayer)
	}

	got := merged[fuseLayer+".weight"]
	Materialize(got)
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i, v := range got.Floats() {
		if math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("merged[%d] = %.4f, want %.4f", i, v, want[i])
		}
	}

	// Untargeted tensor carried through unchanged (same handle).
	if merged["model.norm.weight"] != base["model.norm.weight"] {
		t.Error("untargeted model.norm.weight was not carried through")
	}
}

// TestFuseLoRA_MissingBase_Bad: an adapter delta with no base weight is a loud
// error, not a silent skip.
func TestFuseLoRA_MissingBase_Bad(t *testing.T) {
	_, adapter := fuseBaseAdapter()
	base := map[string]*Array{"model.norm.weight": FromValues([]float32{1, 2, 3}, 3)}
	if _, _, err := FuseLoRAIntoWeights(base, adapter, 2.0); err == nil {
		t.Error("expected error for adapter delta with no base weight, got nil")
	}
}

// TestFuseLoRA_QuantizedBase_Bad: a quantized base is refused (dense-only cut).
func TestFuseLoRA_QuantizedBase_Bad(t *testing.T) {
	base, adapter := fuseBaseAdapter()
	base[fuseLayer+".scales"] = FromValues([]float32{0.1}, 1) // marks the base quantized
	if _, _, err := FuseLoRAIntoWeights(base, adapter, 2.0); err == nil {
		t.Error("expected error fusing into a quantized base, got nil")
	}
}

// TestFuseLoRA_ShapeMismatch_Bad: a delta whose shape does not match the base
// weight is a loud error, not a silent wrong-shape merge.
func TestFuseLoRA_ShapeMismatch_Bad(t *testing.T) {
	base, adapter := fuseBaseAdapter()
	// B·A becomes [3,3] (out 3) while the base weight is [2,3].
	adapter[fuseLayer+".lora_b"] = FromValues([]float32{1, 2, 3}, 3, 1)
	if _, _, err := FuseLoRAIntoWeights(base, adapter, 2.0); err == nil {
		t.Error("expected error for base/delta shape mismatch, got nil")
	}
}

// TestFuseModelDir_RoundTrip_Good is the end-to-end on-disk fuse: a synthetic
// base model dir + an engine-format adapter dir fuse to a new dir whose
// model.safetensors reloads with W+(B·A)·scale (scale = alpha/rank = 2), and the
// base's config.json is carried across.
func TestFuseModelDir_RoundTrip_Good(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	outDir := core.JoinPath(t.TempDir(), "fused") // FuseModelDir creates it

	// Base model.safetensors: the targeted weight + an untargeted one.
	W := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	norm := FromValues([]float32{7, 8, 9}, 3)
	Materialize(W, norm)
	if err := SaveSafetensors(core.JoinPath(baseDir, "model.safetensors"), map[string]*Array{
		fuseLayer + ".weight": W,
		"model.norm.weight":   norm,
	}); err != nil {
		t.Fatalf("write base: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(baseDir, "config.json"), `{"model_type":"composed"}`); err != nil {
		t.Fatalf("write base config: %v", err)
	}

	// Adapter dir: engine format — adapter.safetensors {lora_a,lora_b} + config.
	a := FromValues([]float32{0.1, 0.2, 0.3}, 1, 3)
	b := FromValues([]float32{2, 3}, 2, 1)
	Materialize(a, b)
	if err := SaveSafetensors(core.JoinPath(adapterDir, "adapter.safetensors"), map[string]*Array{
		fuseLayer + ".lora_a": a,
		fuseLayer + ".lora_b": b,
	}); err != nil {
		t.Fatalf("write adapter: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(adapterDir, "adapter_config.json"), `{"rank":1,"alpha":2,"lora_layers":["self_attn.q_proj"]}`); err != nil {
		t.Fatalf("write adapter config: %v", err)
	}

	fused, err := FuseModelDir(baseDir, adapterDir, outDir)
	if err != nil {
		t.Fatalf("FuseModelDir: %v", err)
	}
	if len(fused) != 1 || fused[0] != fuseLayer {
		t.Fatalf("fused = %v, want [%s]", fused, fuseLayer)
	}

	// Reload the fused dir and assert the merged weight survived disk.
	out, err := LoadModelWeights(outDir)
	if err != nil {
		t.Fatalf("reload fused dir: %v", err)
	}
	got := ResolveWeight(out, fuseLayer+".weight")
	if got == nil {
		t.Fatal("fused dir has no merged weight")
	}
	Materialize(got)
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i, v := range got.Floats() {
		if math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("reloaded merged[%d] = %.4f, want %.4f", i, v, want[i])
		}
	}

	// config.json carried across.
	if info := core.Stat(core.JoinPath(outDir, "config.json")); !info.OK {
		t.Error("base config.json was not carried into the fused dir")
	}
}

// TestFuseModelDir_NoMatch_Bad: an adapter whose targets match no base weight is
// a loud error, not an empty (silently base-identical) output.
func TestFuseModelDir_NoMatch_Bad(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	W := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	Materialize(W)
	if err := SaveSafetensors(core.JoinPath(baseDir, "model.safetensors"), map[string]*Array{
		"model.layers.0.mlp.gate_proj.weight": W, // not what the adapter targets
	}); err != nil {
		t.Fatalf("write base: %v", err)
	}
	a := FromValues([]float32{0.1, 0.2, 0.3}, 1, 3)
	b := FromValues([]float32{2, 3}, 2, 1)
	Materialize(a, b)
	if err := SaveSafetensors(core.JoinPath(adapterDir, "adapter.safetensors"), map[string]*Array{
		fuseLayer + ".lora_a": a,
		fuseLayer + ".lora_b": b,
	}); err != nil {
		t.Fatalf("write adapter: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(adapterDir, "adapter_config.json"), `{"rank":1,"alpha":2,"lora_layers":["self_attn.q_proj"]}`); err != nil {
		t.Fatalf("write adapter config: %v", err)
	}
	if _, err := FuseModelDir(baseDir, adapterDir, core.JoinPath(t.TempDir(), "fused")); err == nil {
		t.Error("expected error when adapter targets no base weight, got nil")
	}
}
