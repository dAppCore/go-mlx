// SPDX-Licence-Identifier: EUPL-1.2

// Branch coverage for the residual arms of quant.go that the happy-path
// and _Bad suites in quant_test.go don't reach: the clean-integer-but-
// out-of-range bit-width rejection in deriveAffineBits, the
// missing-weight / zero-shape continue in deriveQuantFromIndex, the
// could-not-derive error return in ResolveQuant, and safetensorsIndex's
// directory-listing failure (reached directly — ResolveQuant fails at
// readModelConfig first on a bad dir, so the ReadDir error is unreachable
// through the public entry point).

package model

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// TestDeriveAffineBits_CleanButOutOfRange pins the `bits < 1 || bits > 8`
// rejection (line ~95) with shapes that DO reduce to a whole number — the
// _Bad suite's "bits over 8" case is caught earlier by the non-integer
// guard, so the integer-but-too-wide arm needs its own fixtures.
func TestDeriveAffineBits_CleanButOutOfRange(t *testing.T) {
	cases := []struct {
		name                       string
		weightLast, scalesLast, gs int
	}{
		// 32*576/(24*32) = 18432/768 = 24 — a clean integer, but 24 > 8.
		{"clean 24 bits", 576, 24, 32},
		// 32*9/(24*64) ... pick values reducing to exactly 0 is impossible
		// (numerator>0), so the <1 side is unreachable with positive dims;
		// the >8 side is the live branch. A second over-range integer:
		// 32*480/(24*32) = 15360/768 = 20.
		{"clean 20 bits", 480, 24, 32},
	}
	for _, c := range cases {
		if got, ok := deriveAffineBits(c.weightLast, c.scalesLast, c.gs); ok {
			t.Errorf("%s: deriveAffineBits(%d,%d,%d) = (%d,true), want ok=false (out of 1..8)",
				c.name, c.weightLast, c.scalesLast, c.gs, got)
		}
	}
}

// TestDeriveQuantFromIndex_MissingWeight drives the continue arm at line
// ~113: a ".scales" tensor whose paired ".weight" is absent from the
// index. With no other derivable tensor the whole walk falls through to
// the `return 0, false` at line ~122.
func TestDeriveQuantFromIndex_MissingWeight(t *testing.T) {
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{
		// scales present, but no matching .weight — skip and fall through.
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []uint64{256, 24}},
		// a plain weight with no scales companion — also skipped (no suffix).
		"model.layers.0.mlp.gate_proj.weight": {DType: "U32", Shape: []uint64{256, 192}},
	}}
	if bits, ok := deriveQuantFromIndex(index, 64); ok {
		t.Fatalf("deriveQuantFromIndex with no paired weight = (%d,true), want ok=false", bits)
	}
}

// TestDeriveQuantFromIndex_ZeroShape drives the `len(...Shape) == 0`
// half of the continue guard at line ~113 — a paired weight/scales pair
// where one carries an empty shape slice (scalar header), which can't
// pin a bit-width.
func TestDeriveQuantFromIndex_ZeroShape(t *testing.T) {
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []uint64{}},
		"model.layers.0.self_attn.q_proj.weight": {DType: "U32", Shape: []uint64{256, 192}},
	}}
	if bits, ok := deriveQuantFromIndex(index, 64); ok {
		t.Fatalf("deriveQuantFromIndex with zero-shape scales = (%d,true), want ok=false", bits)
	}
}

// TestResolveQuant_UnderivableGeometry_Bad drives ResolveQuant's
// could-not-derive error return (line ~62): a config that declares bits +
// group_size, paired with a safetensors header whose q_proj geometry
// reduces to an out-of-range (24-bit) width, so deriveQuantFromIndex
// returns ok=false and ResolveQuant must fail loud rather than guess.
func TestResolveQuant_UnderivableGeometry_Bad(t *testing.T) {
	dir := t.TempDir()
	// weightLast=576, scalesLast=24, group=32 → 32*576/(24*32)=24 bits → reject.
	writeSyntheticQuantPack(t, dir,
		`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":32}}`, 576, 24)
	spec, err := ResolveQuant(dir)
	if err == nil {
		t.Fatalf("ResolveQuant = %+v, want error when no clean bit-width derives", spec)
	}
}

// TestSafetensorsIndex_ReadDirError drives safetensorsIndex's
// `!listed.OK` arm (line ~129) directly: ResolveQuant can't reach it
// (readModelConfig fails first on a missing directory), so the ReadDir
// failure is only observable by calling the unexported helper against a
// path that does not exist.
func TestSafetensorsIndex_ReadDirError(t *testing.T) {
	if _, err := safetensorsIndex(core.PathJoin(t.TempDir(), "no-such-dir")); err == nil {
		t.Fatalf("safetensorsIndex of a missing directory: want error, got nil")
	}
}

// TestSafetensorsIndex_NoShards drives the `len(paths) == 0` arm: a real
// directory that lists clean but holds no .safetensors shard. (Belt-and-
// braces alongside the ResolveQuant_NoSafetensors_Bad case — this pins
// the helper boundary directly.)
func TestSafetensorsIndex_NoShards(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	if _, err := safetensorsIndex(dir); err == nil {
		t.Fatalf("safetensorsIndex of a shard-less directory: want error, got nil")
	}
}

// TestSafetensorsIndex_GoodHeader confirms the success path returns a
// populated index — pins the tail of safetensorsIndex (the IndexFiles
// hand-off) on a synthetic single-shard header with no payload.
func TestSafetensorsIndex_GoodHeader(t *testing.T) {
	dir := t.TempDir()
	header := map[string]safetensors.HeaderEntry{
		"model.embed_tokens.weight": {DType: "BF16", Shape: []int64{32, 8}, DataOffsets: []int64{0, 0}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), out, 0o644); !result.OK {
		t.Fatalf("write shard: %v", result.Value)
	}
	index, err := safetensorsIndex(dir)
	if err != nil {
		t.Fatalf("safetensorsIndex: %v", err)
	}
	if _, ok := index.Tensors["model.embed_tokens.weight"]; !ok {
		t.Fatalf("index missing the embed_tokens entry: %+v", index.Tensors)
	}
}
