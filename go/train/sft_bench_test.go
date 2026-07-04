// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for sft.go — supervised LoRA fine-tuning pipeline.
// Per AX-11 — probe meta builds per gradient step (hundreds/thousands per
// training run); SFTLoRAMetadata clone fires per checkpoint + per final
// adapter save; sftBatchFromExamples runs once per accumulated batch
// (one per BatchSize samples). Pinning the alloc shape of these hot
// paths is the load-bearing AX commitment of this file.
//
// Run:    go test -bench='BenchmarkSFT' -benchmem -run='^$' ./go

package train

import (
	"strconv"
	"testing"

	"dappco.re/go/mlx/spine"
)

// sftBenchTokenizer is the minimal TokenizerImpl the tokenise bench needs
// (mirrors the root sft_test fixture, which stayed with the Model tests).
type sftBenchTokenizer struct {
	encoded map[string][]int32
	eos     int32
}

func (f sftBenchTokenizer) Encode(text string) []int32 { return f.encoded[text] }

func (f sftBenchTokenizer) Decode([]int32) string { return "" }

func (f sftBenchTokenizer) DecodeOne(int32) string { return "" }

func (f sftBenchTokenizer) TokenID(string) (int32, bool) { return 0, false }

func (f sftBenchTokenizer) IDToken(int32) string { return "" }

func (f sftBenchTokenizer) BOS() int32 { return 0 }

func (f sftBenchTokenizer) EOS() int32 { return f.eos }

func (f sftBenchTokenizer) HasBOSToken() bool { return false }

var (
	sftBenchSinkMap      map[string]string
	sftBenchSinkLoRA     SFTLoRAMetadata
	sftBenchSinkBatch    SFTBatch
	sftBenchSinkExample  sftExample
	sftBenchSinkStepName string
	sftBenchSinkInt      int
)

// BenchmarkSFT_EffectiveBatchSize — called inline by the probe meta
// builder (once per gradient step) and by SFTResult.Metrics. Tracks
// whether the helper stays tight or starts paying for unrelated
// normalisation work like LoRA TargetKeys backfills.
func BenchmarkSFT_EffectiveBatchSize(b *testing.B) {
	cfg := SFTConfig{
		BatchSize:                 4,
		GradientAccumulationSteps: 2,
		LoRA: spine.LoRAConfig{
			Rank:         8,
			TargetKeys:   []string{"q_proj", "v_proj"},
			TargetLayers: []string{"layer.0", "layer.1"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkInt = SFTEffectiveBatchSize(cfg)
	}
}

// sftBenchBuildProbeMeta isolates the meta map shape used in the probe
// emission so the bench tracks the same alloc shape as the production
// path without spinning up an entire SFT run.
func sftBenchBuildProbeMeta(cfg SFTConfig, optimizerSteps int) map[string]string {
	meta := make(map[string]string, 6)
	meta["batch_size"] = sftBenchFormatInt(cfg.BatchSize)
	meta["effective_batch_size"] = sftBenchFormatInt(SFTEffectiveBatchSize(cfg))
	meta["gradient_accumulation_steps"] = sftBenchFormatInt(cfg.GradientAccumulationSteps)
	meta["sequence_packing"] = sftBenchFormatBool(cfg.SequencePacking)
	meta["optimizer_step"] = sftBenchFormatInt(optimizerSteps)
	meta["sft_checkpoint_metadata_ver"] = sftBenchFormatInt(SFTCheckpointMetadataVersion)
	return meta
}

func sftBenchFormatInt(i int) string {
	// Mirrors the production formatter at the bench-call site.
	return strconv.Itoa(i)
}

func sftBenchFormatBool(v bool) string {
	return strconv.FormatBool(v)
}
