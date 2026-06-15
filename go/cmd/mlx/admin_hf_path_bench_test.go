// SPDX-Licence-Identifier: EUPL-1.2

package main

import "testing"

// AX-11 micro-benchmark for isSafeHFEntryPath — the per-file path
// guard run once for every entry in an HF tree-API response. Model-
// free pure-string validation.
//
//	go test -tags metal_runtime -run '^$' -bench 'SafeHFEntryPath' -benchmem ./cmd/mlx/
//
// A model repo lists tens-to-hundreds of files; the guard runs once
// per entry, so a per-call slice allocation multiplies across the
// whole tree. The cases cover the common accept path (a nested
// safetensors shard) and the dotfile-segment reject path.

var benchSafeHFSink bool

func BenchmarkIsSafeHFEntryPath(b *testing.B) {
	paths := []string{
		"model-00001-of-00004.safetensors",
		"onnx/decoder_model_merged.onnx",
		"subdir/nested/deeper/weights.bin",
		".git/config",            // dotfile segment → reject
		"ok/.hidden/thing",       // mid-path dotfile → reject
		"config.json",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range paths {
			benchSafeHFSink = isSafeHFEntryPath(p)
		}
	}
}
