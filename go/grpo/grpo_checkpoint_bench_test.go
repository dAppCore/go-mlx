// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for grpo_checkpoint.go — experimental GRPO checkpoint
// metadata. Per AX-11 — LoadGRPOCheckpointMetadata is the resume read
// path (core.ReadFile the sidecar then core.JSONUnmarshal it); it runs
// once per resumed run. Its allocs are stdlib json + the file-read
// buffer + the returned *meta escape, none package-reducible. Pinning
// that floor here is the load-bearing AX commitment of this file — a
// regression that slipped an un-presized decode helper into the loader
// body would show up against it.
//
// Run:    go test -bench='BenchmarkGRPO_LoadCheckpointMetadata' -benchmem -run='^$' ./go

package grpo

import (
	"testing"
)

var grpoBenchSinkMeta *GRPOCheckpointMetadata

// BenchmarkGRPO_LoadCheckpointMetadata — the resume read path. The
// sidecar is written once into a temp dir from a realistic metadata
// value (a populated update + policy identity); the loop measures the
// steady-state read+decode.
func BenchmarkGRPO_LoadCheckpointMetadata(b *testing.B) {
	cfg := GRPOConfig{
		KLCoefficient: 0.1,
		LearningRate:  1e-5,
		GroupSize:     4,
		ResumePath:    "/tmp/resume",
	}
	result := &GRPOResult{Policy: ModelInfo{Architecture: "qwen3", VocabSize: 32000}}
	result.Metrics.Samples = 800
	result.Metrics.Rollouts = 3200
	update := GRPOUpdate{Step: 100, Epoch: 1, RewardMean: 0.5, RewardStd: 0.2, KLMean: 0.01, Loss: 1.25}
	dir := b.TempDir()
	meta := NewGRPOCheckpointMetadata(dir, cfg, result, update)
	if err := SaveGRPOCheckpointMetadata(dir, meta); err != nil {
		b.Fatalf("seed SaveGRPOCheckpointMetadata: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loaded, err := LoadGRPOCheckpointMetadata(dir)
		if err != nil {
			b.Fatalf("LoadGRPOCheckpointMetadata: %v", err)
		}
		grpoBenchSinkMeta = loaded
	}
}
