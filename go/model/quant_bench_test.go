// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the quant-resolution path in quant.go. Per AX-11 —
// ResolveQuant fires once per model directory before any backend touches
// the weights: it reads config.json, lists the .safetensors shards, builds
// the header index and walks it to confirm the bit-width from the packed
// tensor geometry. The bench-coverage audit flagged ResolveQuant at 0%
// bench coverage; this file closes that gap and pins the alloc floor of
// the two helpers ResolveQuant owns directly (safetensorsIndex's directory
// listing + the deriveQuantFromIndex map walk), separately from the
// already-optimised readModelConfig + safetensors.IndexFiles it delegates
// to.
//
// Run:    go test -bench=BenchmarkQuant -benchmem -run='^$' ./go/model

package model

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

// Sinks defeat compiler DCE.
var (
	quantSinkSpec  QuantSpec
	quantSinkErr   error
	quantSinkIndex safetensors.Index
	quantSinkBits  int
	quantSinkOK    bool
)

// writeBenchQuantPack writes the minimum a model directory needs for
// ResolveQuant to read a spec from the bytes — a config.json with the
// declared quantization block and a single .safetensors shard whose header
// alone (zero-length payload) pins the packed geometry. Mirrors the
// *testing.T helper writeSyntheticQuantPack in quant_test.go but takes
// testing.TB so it serves benches too. The q_proj weight/scales last-dims
// encode the bit-width via bits = 32*weightLast/(scalesLast*group).
func writeBenchQuantPack(tb testing.TB, dir, configJSON string, weightLast, scalesLast int64) {
	tb.Helper()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(configJSON), 0o644); !result.OK {
		tb.Fatalf("write config.json: %v", result.Value)
	}
	header := map[string]safetensors.HeaderEntry{
		"model.layers.0.self_attn.q_proj.weight": {DType: "U32", Shape: []int64{256, weightLast}, DataOffsets: []int64{0, 0}},
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []int64{256, scalesLast}, DataOffsets: []int64{0, 0}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		tb.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), out, 0o644); !result.OK {
		tb.Fatalf("write model.safetensors: %v", result.Value)
	}
}

// BenchmarkQuant_ResolveQuant_Affine drives the whole quant-resolution
// chain end to end against a synthetic 4-bit affine pack: readModelConfig
// (disk read + hand-rolled probe walk), safetensorsIndex (ReadDir + shard
// path build + IndexFiles header parse) and deriveQuantFromIndex (map walk
// + geometry derive). This is the per-model-load cost the audit flagged.
func BenchmarkQuant_ResolveQuant_Affine(b *testing.B) {
	dir := b.TempDir()
	writeBenchQuantPack(b, dir,
		`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":64}}`, 192, 24)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantSinkSpec, quantSinkErr = ResolveQuant(dir)
	}
}

// BenchmarkQuant_ResolveQuant_FullPrecision drives the early QuantNone
// return — a config with no quantization block returns before any
// safetensors read. Pins the config-read floor of the path on its own.
func BenchmarkQuant_ResolveQuant_FullPrecision(b *testing.B) {
	dir := b.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"),
		[]byte(`{"model_type":"qwen3","hidden_size":2048}`), 0o644); !result.OK {
		b.Fatalf("write config.json: %v", result.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantSinkSpec, quantSinkErr = ResolveQuant(dir)
	}
}

// BenchmarkQuant_SafetensorsIndex isolates the directory-listing + shard
// path build that ResolveQuant owns directly (the IndexFiles hand-off is
// already benched in the safetensors package). Surfaces the cost of
// ReadDir + the paths slice + PathJoin per shard, away from the JSON
// config read.
func BenchmarkQuant_SafetensorsIndex(b *testing.B) {
	dir := b.TempDir()
	writeBenchQuantPack(b, dir,
		`{"model_type":"qwen3","quantization_config":{"bits":4,"group_size":64}}`, 192, 24)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantSinkIndex, quantSinkErr = safetensorsIndex(dir)
	}
}

// BenchmarkQuant_DeriveQuantFromIndex isolates the in-memory map walk —
// no disk, no JSON. This is the hot loop that builds a ".weight" lookup
// key per ".scales" tensor and derives the bit-width. A real model index
// carries hundreds of tensors, so the per-scales key-build cost compounds;
// this fixture keeps a representative q_proj/scales pair plus decoy
// tensors so the suffix scan and lookup both run.
func BenchmarkQuant_DeriveQuantFromIndex(b *testing.B) {
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{
		"model.layers.0.self_attn.q_proj.weight": {DType: "U32", Shape: []uint64{256, 192}},
		"model.layers.0.self_attn.q_proj.scales": {DType: "F16", Shape: []uint64{256, 24}},
		"model.layers.0.mlp.gate_proj.weight":    {DType: "U32", Shape: []uint64{256, 192}},
		"model.embed_tokens.weight":              {DType: "BF16", Shape: []uint64{262144, 2304}},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantSinkBits, quantSinkOK = deriveQuantFromIndex(index, 64)
	}
}
