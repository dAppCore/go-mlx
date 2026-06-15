// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	"encoding/binary"
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/safetensors"
)

// ExampleResolveQuant detects a model's quantisation from its own bytes — the
// declared config.json block supplies the group size, the packed tensor
// geometry confirms the bit-width, and the two are cross-checked. The model is
// the truth; the filename is never consulted. The synthetic pack here declares
// 4-bit / group 64 and ships a q_proj whose U32 weight (last-dim 192) and F16
// scales (last-dim 24) derive the same 4 bits via 32*192/(24*64), so the spec
// resolves to {affine, 4, 64}.
func ExampleResolveQuant() {
	dir := exampleWriteAffinePack()
	defer core.RemoveAll(dir)

	spec, err := model.ResolveQuant(dir)
	if err != nil {
		fmt.Println("resolve error:", err)
		return
	}

	fmt.Println("Format:   ", spec.Format)
	fmt.Println("Bits:     ", spec.Bits)
	fmt.Println("GroupSize:", spec.GroupSize)
	// Output:
	// Format:    affine
	// Bits:      4
	// GroupSize: 64
}

// ExampleResolveQuant_fullPrecision resolves a config with no quantisation block
// to QuantNone — the model ships full precision, so no safetensors read is even
// attempted and the spec carries zero bits. This is the early-return contract a
// caller relies on to tell "unquantised" apart from "quantised, N-bit".
func ExampleResolveQuant_fullPrecision() {
	made := core.MkdirTemp("", "model-fp-example-*")
	if !made.OK {
		panic(made.Value)
	}
	dir := made.Value.(string)
	defer core.RemoveAll(dir)
	if r := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","hidden_size":2048}`), 0o644); !r.OK {
		panic(r.Value)
	}

	spec, err := model.ResolveQuant(dir)
	if err != nil {
		fmt.Println("resolve error:", err)
		return
	}

	fmt.Printf("%q %d %d\n", spec.Format, spec.Bits, spec.GroupSize)
	// Output:
	// "" 0 0
}

// exampleWriteAffinePack synthesises a minimal MLX affine-quantised safetensors
// pack: a config declaring 4-bit / group 64 and a single q_proj weight/scales
// pair whose last-dims pin the bit-width. The safetensors body is just the
// header (a uint64 length prefix + the JSON tensor map) — ResolveQuant reads
// only the header geometry, never the tensor data, so the offsets can be zero.
// Setup failures panic: an example that cannot stage its fixture has nothing to
// demonstrate.
func exampleWriteAffinePack() string {
	made := core.MkdirTemp("", "model-quant-example-*")
	if !made.OK {
		panic(made.Value)
	}
	dir := made.Value.(string)

	if r := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":64}}`), 0o644); !r.OK {
		panic(r.Value)
	}

	header := map[string]safetensors.HeaderEntry{
		"model.layers.0.self_attn.q_proj.weight": {DType: "U32", Shape: []int64{256, 192}, DataOffsets: []int64{0, 0}},
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []int64{256, 24}, DataOffsets: []int64{0, 0}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		panic(encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if r := core.WriteFile(core.PathJoin(dir, "model.safetensors"), out, 0o644); !r.OK {
		panic(r.Value)
	}
	return dir
}
