// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for sft.go — supervised LoRA fine-tuning pipeline.
// Per AX-11 — probe meta builds per gradient step (hundreds/thousands per
// training run); SFTLoRAMetadata clone fires per checkpoint + per final
// adapter save; sftBatchFromExamples runs once per accumulated batch
// (one per BatchSize samples); int32ToIntSlice runs twice per example.
// Pinning the alloc shape of these hot paths is the load-bearing AX
// commitment of this file.
//
// Run:    go test -bench='BenchmarkSFT' -benchmem -run='^$' ./go

package mlx

import (
	"strconv"
	"testing"

	"dappco.re/go/mlx/dataset"
)

var (
	sftBenchSinkMap      map[string]string
	sftBenchSinkLoRA     SFTLoRAMetadata
	sftBenchSinkBatch    SFTBatch
	sftBenchSinkInts     []int
	sftBenchSinkExample  sftExample
	sftBenchSinkStepName string
)

// BenchmarkSFT_RunProbeMeta mirrors the runSFTBatchGroup probe.Event.Meta
// construction (6 string fields, all int-formatted today via Sprintf).
// Fires once per gradient step when a probe sink is attached.
func BenchmarkSFT_RunProbeMeta(b *testing.B) {
	cfg := SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, SequencePacking: true}
	cfg = normalizeSFTConfig(cfg)
	optimizerSteps := 1234
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkMap = sftBenchBuildProbeMeta(cfg, optimizerSteps)
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

// BenchmarkSFT_LoRAMetadata measures the per-checkpoint clone of
// TargetKeys/TargetLayers when persisting metadata.
func BenchmarkSFT_LoRAMetadata(b *testing.B) {
	cfg := LoRAConfig{
		Rank:         8,
		Alpha:        16,
		TargetKeys:   []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		TargetLayers: []string{"layer.0", "layer.1", "layer.2", "layer.3"},
		DType:        DTypeFloat32,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkLoRA = sftLoRAMetadata(cfg)
	}
}

// BenchmarkSFT_BatchFromExamples mirrors sftBatchFromExamples — runs
// once per gradient accumulation flush (BatchSize examples).
func BenchmarkSFT_BatchFromExamples(b *testing.B) {
	examples := make([]sftExample, 8)
	for i := range examples {
		examples[i] = sftExample{
			inputs:  []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			targets: []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			mask:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkBatch = sftBatchFromExamples(examples)
	}
}

// BenchmarkSFT_Int32ToIntSlice measures the per-example token width
// conversion used twice on every buildSFTExample call.
func BenchmarkSFT_Int32ToIntSlice(b *testing.B) {
	values := make([]int32, 256)
	for i := range values {
		values[i] = int32(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkInts = int32ToIntSlice(values)
	}
}

// BenchmarkSFT_HasTrainingTarget exercises the mask scan executed once
// per buildSFTExample.
func BenchmarkSFT_HasTrainingTarget(b *testing.B) {
	mask := make([]float32, 256)
	mask[200] = 1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = hasTrainingTarget(mask)
	}
}

// BenchmarkSFT_StreamingPacker — exercise the per-sample packer add
// + final flush path. maxSeqLen=64, 8 samples of length 6 (no trim,
// no mid-add flush) → tests the pre-sized accumulator growth.
func BenchmarkSFT_StreamingPacker(b *testing.B) {
	ex := sftExample{
		inputs:  []int{1, 2, 3, 4, 5, 6},
		targets: []int{2, 3, 4, 5, 6, 7},
		mask:    []float32{0, 0, 0, 1, 1, 1},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packer := newSFTStreamingPacker(64, func(sftExample) error { return nil })
		for j := 0; j < 8; j++ {
			_ = packer.add(ex)
		}
		_ = packer.finish()
	}
}

// BenchmarkSFT_StepName tracks the checkpoint directory-name builder
// — runs every CheckpointEvery steps during long training runs.
func BenchmarkSFT_StepName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkStepName = sftStepName(12345)
	}
}

// BenchmarkSFT_HasTrainingTarget_AllZero — worst case (full scan).
func BenchmarkSFT_HasTrainingTarget_AllZero(b *testing.B) {
	mask := make([]float32, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = hasTrainingTarget(mask)
	}
}

// BenchmarkSFT_BuildExample exercises buildSFTExample end-to-end with
// a fake tokenizer — the per-sample hot path of every SFT run.
func BenchmarkSFT_BuildExample(b *testing.B) {
	tok := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {10, 11, 12, 13},
			"response": {20, 21, 22, 23, 24, 25, 26, 27},
		},
		eos: 2,
	}}
	sample := dataset.Sample{Prompt: "prompt", Response: "response"}
	cfg := SFTConfig{BatchSize: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkExample, _, _ = buildSFTExample(tok, sample, cfg)
	}
}

// BenchmarkSFT_BatchBuilderFinish mirrors the final batch flush + clone.
func BenchmarkSFT_BatchBuilderFinish(b *testing.B) {
	example := sftExample{
		inputs:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		targets: []int{2, 3, 4, 5, 6, 7, 8, 9},
		mask:    []float32{0, 0, 1, 1, 1, 1, 1, 1},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := newSFTBatchBuilder(2)
		for j := 0; j < 8; j++ {
			builder.add(example)
		}
		_ = builder.finish()
	}
}
