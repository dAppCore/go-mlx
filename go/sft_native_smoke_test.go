// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

func TestSFTNativeSmoke_OneLoRAStep_Good(t *testing.T) {
	modelPath := core.Trim(core.Env("GO_MLX_SFT_SMOKE_MODEL"))
	if modelPath == "" {
		t.Skip("set GO_MLX_SFT_SMOKE_MODEL to run the local native SFT smoke")
	}

	model, err := LoadModel(
		modelPath,
		WithContextLength(1024),
		WithBatchSize(128),
		WithPrefillChunkSize(128),
		WithGemma4SlidingWindow(512),
		WithPromptCache(false),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	defer func() {
		if err := model.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	}()

	result, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{
		Prompt:   "What should a retained State runner preserve?",
		Response: "It should preserve the useful KV state without replaying unchanged context.",
	}}), SFTConfig{
		LoRA: LoRAConfig{
			Rank:       2,
			Alpha:      4,
			TargetKeys: []string{"q_proj"},
		},
		BatchSize:       1,
		Epochs:          1,
		LearningRate:    1e-5,
		MaxSeqLen:       64,
		SequencePacking: false,
		NoEOS:           true,
	})
	if err != nil {
		t.Fatalf("TrainSFT() error = %v", err)
	}
	if result == nil {
		t.Fatal("TrainSFT() result is nil")
	}
	if result.Steps != 1 {
		t.Fatalf("Steps = %d, want 1", result.Steps)
	}
	if result.Adapter == nil {
		t.Fatal("Adapter is nil")
	}
	if math.IsNaN(result.LastLoss) || math.IsInf(result.LastLoss, 0) {
		t.Fatalf("LastLoss = %v, want finite", result.LastLoss)
	}
}
