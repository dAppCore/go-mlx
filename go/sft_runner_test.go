// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/dataset"
	"testing"

	core "dappco.re/go"
)

func TestBuildSFTTrainingBatches_UsesAccumulationAsEffectiveBatch_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	}}
	dataset := dataset.NewJSONL([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildSFTTrainingBatches(tokenizer, dataset, SFTConfig{
		BatchSize:                 1,
		GradientAccumulationSteps: 2,
	})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want one effective optimizer batch", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 2 {
		t.Fatalf("batch sequences = %d, want 2 micro-batches", len(batches[0].Batch.Tokens))
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1}) ||
		!equalFloat32Slices(batches[0].Batch.LossMask[1], []float32{1, 1}) {
		t.Fatalf("loss masks = %v, want response-only masks preserved", batches[0].Batch.LossMask)
	}
}

func TestBuildSFTTrainingBatches_NilDataset_Bad(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{eos: 9}}
	_, err := BuildSFTTrainingBatches(tokenizer, nil, SFTConfig{})
	if err == nil {
		t.Fatal("expected nil dataset error")
	}
}

func TestBuildSFTTrainingBatches_PackedDataset_Ugly(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	}}
	dataset := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildSFTTrainingBatches(tokenizer, dataset, SFTConfig{
		BatchSize:       1,
		MaxSeqLen:       8,
		SequencePacking: true,
	})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 || len(batches[0].Batch.Tokens) != 1 {
		t.Fatalf("batches = %+v, want one packed sequence", batches)
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{1, 2, 3, 4}) {
		t.Fatalf("packed inputs = %v, want [1 2 3 4]", batches[0].Batch.Tokens[0])
	}
}

func TestSFTCheckpointMetadata_RoundTrip_Good(t *testing.T) {
	dir := t.TempDir()
	meta := SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      dir,
		AdapterPath:               core.PathJoin(dir, "adapter.safetensors"),
		Step:                      7,
		OptimizerStep:             7,
		Epoch:                     2,
		Samples:                   13,
		Loss:                      0.125,
		LearningRate:              2e-4,
		BatchSize:                 2,
		GradientAccumulationSteps: 4,
		SequencePacking:           true,
		Model:                     "qwen3",
		LoRA: SFTLoRAMetadata{
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "v_proj"},
		},
	}

	if err := SaveSFTCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	got, err := LoadSFTCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if got.Step != 7 || got.Epoch != 2 || got.GradientAccumulationSteps != 4 || got.LoRA.Rank != 16 {
		t.Fatalf("metadata = %+v, want round-tripped training state", got)
	}
}

func TestLoadSFTCheckpointMetadata_Missing_Bad(t *testing.T) {
	_, err := LoadSFTCheckpointMetadata(core.PathJoin(t.TempDir(), "missing"))
	if err == nil {
		t.Fatal("expected missing metadata error")
	}
}

func TestLoadSFTResumeMetadata_LoadsAdjacentMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	meta := SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      dir,
		Step:                      11,
		OptimizerStep:             11,
		Epoch:                     3,
		Samples:                   21,
		Loss:                      0.5,
		GradientAccumulationSteps: 2,
	}
	if err := SaveSFTCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: dir}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata() error = %v", err)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 11 || result.ResumePath != dir {
		t.Fatalf("resume result = %+v, want metadata attached", result)
	}
}

func TestSFTAdapterArtifactMetadata_Good(t *testing.T) {
	result := &SFTResult{Steps: 3, Samples: 5, LastLoss: 0.25}
	cfg := normalizeSFTConfig(SFTConfig{
		SavePath:                  core.PathJoin(t.TempDir(), "adapter"),
		BatchSize:                 2,
		GradientAccumulationSteps: 4,
		LearningRate:              1e-4,
		LoRA:                      LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj"}},
	})

	meta := NewSFTArtifactMetadata(cfg.SavePath, "gemma4", cfg, result)
	if meta.Path != cfg.SavePath || meta.Step != 3 || meta.Samples != 5 {
		t.Fatalf("artifact metadata = %+v, want final adapter state", meta)
	}
	if meta.GradientAccumulationSteps != 4 || meta.LoRA.Rank != 8 || meta.Model != "gemma4" {
		t.Fatalf("artifact metadata = %+v, want config attached", meta)
	}
}

func TestSFTResult_Metrics_Good(t *testing.T) {
	result := &SFTResult{
		Steps:       4,
		Epochs:      2,
		Samples:     9,
		LastLoss:    0.75,
		Checkpoints: []string{"a", "b"},
		Evaluations: []SFTEvalResult{{Step: 2}, {Step: 4}},
	}

	metrics := result.Metrics(SFTConfig{
		BatchSize:                 2,
		GradientAccumulationSteps: 3,
		LearningRate:              2e-4,
	})
	if metrics.OptimizerSteps != 4 || metrics.EffectiveBatchSize != 6 || metrics.CheckpointCount != 2 || metrics.EvaluationCount != 2 {
		t.Fatalf("metrics = %+v, want SFT counters", metrics)
	}
}

func TestSFTAdamWConfig_UsesExplicitOptimizer_Bad(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{
		AdamW: AdamWConfig{
			LearningRate:   3e-4,
			Beta1:          0.85,
			Beta2:          0.98,
			WeightDecay:    0,
			WeightDecaySet: true,
			PackedState:    false,
			PackedStateSet: true,
		},
	})

	adam := sftAdamWConfig(cfg)
	if adam.LearningRate != 3e-4 || adam.Beta1 != 0.85 || adam.Beta2 != 0.98 || adam.WeightDecay != 0 || adam.PackedState {
		t.Fatalf("adam = %+v, want explicit optimizer config", adam)
	}
	meta := sftAdamWMetadata(adam)
	if meta.PackedState {
		t.Fatalf("adam metadata = %+v, want explicit packed-state setting", meta)
	}
}

func TestNormalizeSFTConfig_DefaultsLoRA_Ugly(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{})
	meta := sftLoRAMetadata(cfg.LoRA)
	if meta.Rank != 8 || meta.Alpha != 16 || !equalStringSlices(meta.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("lora metadata = %+v, want default adapter identity", meta)
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
