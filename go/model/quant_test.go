// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// TestDeriveAffineBits_Good derives the bit-width of an MLX affine-quantised
// linear from the only thing that can't lie: the packed-weight and scales
// last-dims plus the group size. The identity is
//
//	packed weight last-dim = logical_in * bits / 32
//	scales      last-dim   = logical_in / group_size
//	⇒ bits = 32 * weightLast / (scalesLast * group_size)
//
// The cases are real shapes pulled from gemma-4 packs (q4 g64 is E2B's q_proj),
// so a green test means the engine reads bits from the bytes, not the filename.
func TestDeriveAffineBits_Good(t *testing.T) {
	cases := []struct {
		name                       string
		weightLast, scalesLast, gs int
		want                       int
	}{
		{"q4 g64 (E2B q_proj)", 192, 24, 64, 4},
		{"q4 g64 (256-wide)", 256, 32, 64, 4},
		{"q8 g64", 512, 32, 64, 8},
		{"q6 g64", 384, 32, 64, 6},
		{"q4 g32", 96, 24, 32, 4},
	}
	for _, c := range cases {
		got, ok := deriveAffineBits(c.weightLast, c.scalesLast, c.gs)
		if !ok || got != c.want {
			t.Errorf("%s: deriveAffineBits(%d,%d,%d) = (%d,%v), want (%d,true)",
				c.name, c.weightLast, c.scalesLast, c.gs, got, ok, c.want)
		}
	}
}

// TestDeriveAffineBits_Bad rejects shapes that don't describe a clean affine
// pack — zero/negative dims, and ratios that don't yield an integer 1..8 bit
// width. Garbage in must fail loud (ok=false), never return a plausible-looking
// wrong bit-width.
func TestDeriveAffineBits_Bad(t *testing.T) {
	cases := []struct {
		name                       string
		weightLast, scalesLast, gs int
	}{
		{"zero group", 192, 24, 0},
		{"zero scales", 192, 0, 64},
		{"zero weight", 0, 24, 64},
		{"negative", -192, 24, 64},
		{"non-integer bits", 100, 24, 64},
		{"bits over 8", 4096, 24, 64},
	}
	for _, c := range cases {
		if got, ok := deriveAffineBits(c.weightLast, c.scalesLast, c.gs); ok {
			t.Errorf("%s: deriveAffineBits(%d,%d,%d) = (%d,true), want ok=false",
				c.name, c.weightLast, c.scalesLast, c.gs, got)
		}
	}
}

// TestResolveQuant_Good resolves real gemma-4 packs from the HF cache: q4 and q6
// E2B. It proves the read comes from the model's own bytes — bits derived from
// the tensor geometry, cross-checked against the declared group — across two
// bit-widths. Skips when the packs aren't cached so CI stays green without them.
func TestResolveQuant_Good(t *testing.T) {
	cases := []struct {
		repo string
		bits int
	}{
		{"models--mlx-community--gemma-4-e2b-it-4bit", 4},
		{"models--mlx-community--gemma-4-e2b-it-6bit", 6},
	}
	for _, c := range cases {
		dir := hfSnapshotOrSkip(t, c.repo)
		spec, err := ResolveQuant(dir)
		if err != nil {
			t.Fatalf("%s: ResolveQuant: %v", c.repo, err)
		}
		if spec.Format != QuantAffine || spec.Bits != c.bits || spec.GroupSize != 64 {
			t.Fatalf("%s: got %+v, want {affine %d 64}", c.repo, spec, c.bits)
		}
	}
}

// TestResolveQuant_FullPrecision resolves a bf16 pack to QuantNone — a model
// with no quantization block carries no bits.
func TestResolveQuant_FullPrecision(t *testing.T) {
	dir := hfSnapshotOrSkip(t, "models--mlx-community--gemma-4-E2B-it-bf16")
	spec, err := ResolveQuant(dir)
	if err != nil {
		t.Fatalf("ResolveQuant: %v", err)
	}
	if spec.Format != QuantNone || spec.Bits != 0 {
		t.Fatalf("got %+v, want {none 0}", spec)
	}
}

// writeSyntheticQuantPack writes the minimum a model directory needs for
// ResolveQuant to read a quantisation spec from the bytes: a config.json
// carrying the declared quantization block, and a single .safetensors shard
// whose header alone (no tensor payload) pins the packed geometry. ResolveQuant
// only reads the safetensors *header* index — never the weight bytes — so a
// header with zero-length data_offsets is a complete, load-free fixture. The
// q_proj weight/scales last-dims encode the bit-width via
// bits = 32*weightLast/(scalesLast*group); the caller picks the shape.
func writeSyntheticQuantPack(t *testing.T, dir, configJSON string, weightLast, scalesLast int64) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(configJSON), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	header := map[string]safetensors.HeaderEntry{
		"model.layers.0.self_attn.q_proj.weight": {DType: "U32", Shape: []int64{256, weightLast}, DataOffsets: []int64{0, 0}},
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []int64{256, scalesLast}, DataOffsets: []int64{0, 0}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), out, 0o644); !result.OK {
		t.Fatalf("write model.safetensors: %v", result.Value)
	}
}

// TestResolveQuant_Synthetic resolves an affine pack from synthetic bytes — no
// HF cache required, so it runs everywhere. The config declares 4-bit / group
// 64; the q_proj geometry (weight last-dim 192, scales last-dim 24) derives the
// same 4 bits via 32*192/(24*64), the two cross-check, and ResolveQuant returns
// {affine 4 64}. This drives the whole read chain — readModelConfig +
// safetensorsIndex + deriveQuantFromIndex — that the cache-gated _Good case
// skips on a machine without the packs.
func TestResolveQuant_Synthetic(t *testing.T) {
	dir := t.TempDir()
	writeSyntheticQuantPack(t, dir,
		`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":64}}`, 192, 24)

	spec, err := ResolveQuant(dir)
	if err != nil {
		t.Fatalf("ResolveQuant: %v", err)
	}
	if spec.Format != QuantAffine || spec.Bits != 4 || spec.GroupSize != 64 {
		t.Fatalf("got %+v, want {affine 4 64}", spec)
	}
	if len(spec.Exclude) != 0 {
		t.Fatalf("Exclude = %v, want empty for a uniform pack", spec.Exclude)
	}
}

// TestResolveQuant_FullPrecisionSynthetic resolves a config with no quantization
// block to QuantNone — the model ships full precision, so no safetensors read is
// even attempted. Covers the early QuantNone return without the HF cache.
func TestResolveQuant_FullPrecisionSynthetic(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","hidden_size":2048}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	spec, err := ResolveQuant(dir)
	if err != nil {
		t.Fatalf("ResolveQuant: %v", err)
	}
	if spec.Format != QuantNone || spec.Bits != 0 || spec.GroupSize != 0 {
		t.Fatalf("got %+v, want {none 0 0}", spec)
	}
}

// TestResolveQuant_BitMismatch_Bad fails loud when config.json declares one
// bit-width but the packed tensor geometry derives another — the model is the
// truth and a declared/derived disagreement must error, never silently trust the
// declaration. Config says 8-bit; the q_proj geometry (192/24/64) derives 4-bit.
func TestResolveQuant_BitMismatch_Bad(t *testing.T) {
	dir := t.TempDir()
	writeSyntheticQuantPack(t, dir,
		`{"model_type":"qwen3","quantization_config":{"bits":8,"group_size":64}}`, 192, 24)

	spec, err := ResolveQuant(dir)
	if err == nil {
		t.Fatalf("ResolveQuant = %+v, want error on declared/derived bit mismatch", spec)
	}
}

// TestResolveQuant_DeclaredBitsNoGroup_Bad fails loud when the quantization block
// declares bits but omits group_size — the group is required to derive the pack
// geometry, so an incomplete block errors rather than guessing.
func TestResolveQuant_DeclaredBitsNoGroup_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","quantization_config":{"bits":4}}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	spec, err := ResolveQuant(dir)
	if err == nil {
		t.Fatalf("ResolveQuant = %+v, want error when bits declared without group_size", spec)
	}
}

// TestResolveQuant_NoSafetensors_Bad fails loud when the config declares a quant
// block but the directory carries no .safetensors shard to confirm the geometry
// against — safetensorsIndex must surface the missing-pack error.
func TestResolveQuant_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":64}}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	spec, err := ResolveQuant(dir)
	if err == nil {
		t.Fatalf("ResolveQuant = %+v, want error when no .safetensors shard present", spec)
	}
}

// TestResolveQuant_MissingDir_Bad fails loud when the model directory does not
// exist — readModelConfig must surface the error, not return a zero spec.
func TestResolveQuant_MissingDir_Bad(t *testing.T) {
	spec, err := ResolveQuant(core.PathJoin(t.TempDir(), "does-not-exist"))
	if err == nil {
		t.Fatalf("ResolveQuant = %+v, want error for a missing model directory", spec)
	}
}

// hfSnapshotOrSkip resolves a HuggingFace cache snapshot directory
// (~/.cache/huggingface/hub/<repo>/snapshots/<hash>), skipping the test when the
// pack isn't present so a machine without the cache stays green.
func hfSnapshotOrSkip(t *testing.T, repo string) string {
	t.Helper()
	home := core.UserHomeDir()
	if !home.OK {
		t.Skip("no home dir")
	}
	snapshots := core.PathJoin(home.Value.(string), ".cache", "huggingface", "hub", repo, "snapshots")
	listed := core.ReadDir(core.DirFS(snapshots), ".")
	if !listed.OK {
		t.Skipf("not cached: %s", repo)
	}
	for _, entry := range listed.Value.([]core.FsDirEntry) {
		return core.PathJoin(snapshots, entry.Name())
	}
	t.Skipf("no snapshot under %s", snapshots)
	return ""
}
