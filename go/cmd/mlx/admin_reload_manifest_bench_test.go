// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
)

// AX-11 micro-benchmarks for the .sha256 manifest read/write path
// (readModelManifest / writeModelManifest). These run model-free —
// pure file I/O over a synthetic manifest — so they measure the
// parse/serialise allocation shape, not the model.
//
//	go test -tags metal_runtime -run '^$' -bench 'ModelManifest' -benchmem ./cmd/mlx/
//
// A real model directory carries one .sha256 line per artefact
// (weights shards, config, tokenizer, special-tokens, …); 64 entries
// is a comfortably large upper bound that makes the per-line cost of
// the parse/serialise loop visible above the fixed file-I/O floor.

// benchManifestDigests builds a deterministic {filename → sha256}
// map of n entries for the manifest benchmarks.
func benchManifestDigests(n int) map[string]string {
	digests := make(map[string]string, n)
	for i := 0; i < n; i++ {
		name := core.Sprintf("model-%03d.safetensors", i)
		// 64-hex sha256 with an uppercase nibble so readModelManifest's
		// core.Lower call has real work (the production path lower-cases
		// the hash) — keeps the benchmark honest about that cost.
		digests[name] = core.Sprintf("ABCDEF%058d", i)
	}
	return digests
}

func BenchmarkWriteModelManifest_64(b *testing.B) {
	digests := benchManifestDigests(64)
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := writeModelManifest(dir, digests); err != nil {
			b.Fatalf("writeModelManifest: %v", err)
		}
	}
}

var benchManifestSink map[string]string

func BenchmarkReadModelManifest_64(b *testing.B) {
	digests := benchManifestDigests(64)
	dir := b.TempDir()
	if err := writeModelManifest(dir, digests); err != nil {
		b.Fatalf("seed writeModelManifest: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := readModelManifest(dir)
		if err != nil {
			b.Fatalf("readModelManifest: %v", err)
		}
		benchManifestSink = m
	}
}
