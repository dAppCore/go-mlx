// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	core "dappco.re/go"
)

// TestQuantizeModelPack_ByteStable pins the sha256 of the model.gguf that
// QuantizeModelPack emits for a fixed multi-tensor safetensors source. The
// existing writer byte-identity test guards the streaming writer against the
// buffered reference; this test guards the WHOLE OUTER PIPELINE
// (orchestration: index build, descriptor construction, metadata assembly,
// header layout, tensor offsets, padding) against any orchestration-level
// change — in particular the AX-11 alloc-reduction edits, which must not
// perturb a single output byte.
//
// The golden is computed once on the unmodified pipeline. If an
// optimisation changes it, the change is NOT byte-identical and must be
// reverted. The fixture deliberately mixes tensor sizes (a 2-D weight that
// quantises, plus a 1-D norm the quantiser passes through) and includes
// metadata-bearing config so the header path is non-trivial.
func TestQuantizeModelPack_ByteStable(t *testing.T) {
	const goldenSHA256 = "0772fc845df3212318fdbf7be38dd658ef1f869522577792ce06e07b3c00c7e2"

	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 4}, Data: ascendingFloat32s(128)},
		{Name: "model.layers.1.self_attn.k_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
		{Name: "model.layers.2.mlp.down_proj.weight", Shape: []int{32, 3}, Data: ascendingFloat32s(96)},
		{Name: "model.norm.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	output := core.PathJoin(t.TempDir(), "out-bytestable")

	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	}); err != nil {
		t.Fatalf("QuantizeModelPack() error = %v", err)
	}

	gguf := readFileOrFatal(t, core.PathJoin(output, "model.gguf"))
	sum := sha256.Sum256(gguf)
	got := hex.EncodeToString(sum[:])
	if got != goldenSHA256 {
		t.Fatalf("model.gguf sha256 = %s (%d bytes), want %s — pipeline output is not byte-identical",
			got, len(gguf), goldenSHA256)
	}
}
