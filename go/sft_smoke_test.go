// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/internal/metaltest"
	"math"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/dataset"
)

const gemma4NativeSFTSmokeMaxSeqLen = 256

func requireLocalGemma4E2BQ6SFTModel(t *testing.T) string {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable local Gemma-4 native SFT smoke")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	return metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")
}

func TestSFTNativeSmoke_Gemma4Q6SavesReloadableAdapter_Good(t *testing.T) {
	modelPath := requireLocalGemma4E2BQ6SFTModel(t)

	model, err := LoadModel(
		modelPath,
		WithExpectedQuantization(6),
		WithPromptCache(false),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	t.Cleanup(func() { _ = model.Close() })

	datasetCfg := DatasetConfigForModel(model.Info())
	evalDataset := func() dataset.Dataset {
		return gemma4SFTJSONLDataset(t, datasetCfg)
	}
	baseReport, err := RunModelEval(context.Background(), model, evalDataset(), eval.Config{
		Batch: dataset.BatchConfig{BatchSize: 1, MaxSeqLen: gemma4NativeSFTSmokeMaxSeqLen},
	})
	if err != nil {
		t.Fatalf("base RunModelEval() error = %v", err)
	}
	if baseReport.Metrics.Tokens == 0 || !finitePositive(baseReport.Metrics.Perplexity) {
		t.Fatalf("base eval metrics = %+v, want token coverage and finite perplexity", baseReport.Metrics)
	}

	adapterPath := core.PathJoin(t.TempDir(), "gemma4-e2b-q6-sft-adapter")
	result, err := model.TrainSFT(context.Background(), evalDataset(), SFTConfig{
		LoRA: LoRAConfig{
			Rank:       2,
			Alpha:      4,
			TargetKeys: []string{"q_proj"},
		},
		BatchSize:       1,
		Epochs:          3,
		LearningRate:    5e-3,
		MaxSeqLen:       gemma4NativeSFTSmokeMaxSeqLen,
		SequencePacking: false,
		NoEOS:           true,
		SavePath:        adapterPath,
	})
	if err != nil {
		t.Fatalf("TrainSFT() error = %v", err)
	}
	if result == nil {
		t.Fatal("TrainSFT() result is nil")
	}
	if result.Steps != 3 {
		t.Fatalf("Steps = %d, want 3", result.Steps)
	}
	if result.Adapter == nil {
		t.Fatal("Adapter is nil")
	}
	if !finitePositive(result.LastLoss) {
		t.Fatalf("LastLoss = %v, want finite", result.LastLoss)
	}
	if result.AdapterPath != adapterPath || result.AdapterMetadata == nil {
		t.Fatalf("adapter path=%q metadata=%+v, want saved adapter metadata", result.AdapterPath, result.AdapterMetadata)
	}
	if result.AdapterMetadata.LoRA.Rank != 2 || result.AdapterMetadata.LoRA.Alpha != 4 {
		t.Fatalf("adapter metadata LoRA = %+v, want rank=2 alpha=4", result.AdapterMetadata.LoRA)
	}
	if _, err := LoadSFTCheckpointMetadata(adapterPath); err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	for _, path := range []string{
		core.PathJoin(adapterPath, "adapter_config.json"),
		core.PathJoin(adapterPath, "adapter.safetensors"),
		core.PathJoin(adapterPath, "sft_checkpoint.json"),
	} {
		if stat := core.Stat(path); !stat.OK {
			t.Fatalf("saved adapter missing %s: %v", path, stat.Value)
		}
	}

	if err := model.Close(); err != nil {
		t.Fatalf("Close() trained model error = %v", err)
	}
	ClearCache()

	adapted, err := LoadModel(
		modelPath,
		WithExpectedQuantization(6),
		WithPromptCache(false),
		WithAdapterPath(adapterPath),
	)
	if err != nil {
		t.Fatalf("LoadModel(... WithAdapterPath) error = %v", err)
	}
	t.Cleanup(func() { _ = adapted.Close() })

	info := adapted.Info()
	if info.Adapter.Path != adapterPath || info.Adapter.Rank != 2 || info.Adapter.Alpha != 4 {
		t.Fatalf("adapted model adapter = %+v, want saved rank-2 adapter identity", info.Adapter)
	}
	adapterReport, err := RunModelEval(context.Background(), adapted, evalDataset(), eval.Config{
		Batch: dataset.BatchConfig{BatchSize: 1, MaxSeqLen: gemma4NativeSFTSmokeMaxSeqLen},
	})
	if err != nil {
		t.Fatalf("adapter RunModelEval() error = %v", err)
	}
	if adapterReport.Adapter.Path != adapterPath || adapterReport.ModelInfo.Adapter.Path != adapterPath {
		t.Fatalf("eval adapter=%+v model adapter=%+v, want saved adapter path", adapterReport.Adapter, adapterReport.ModelInfo.Adapter)
	}
	if adapterReport.Metrics.Tokens != baseReport.Metrics.Tokens || !finitePositive(adapterReport.Metrics.Perplexity) {
		t.Fatalf("adapter eval metrics = %+v, base metrics = %+v", adapterReport.Metrics, baseReport.Metrics)
	}
	if adapterReport.Metrics.Loss == baseReport.Metrics.Loss {
		t.Fatalf("adapter eval loss = base loss %f, want saved LoRA to change eval output", adapterReport.Metrics.Loss)
	}
	t.Logf(
		"Gemma-4 q6 JSONL SFT eval adapter path=%s rank=%d alpha=%.1f loss %.6f -> %.6f perplexity %.6f -> %.6f",
		adapterReport.Adapter.Path,
		adapterReport.Adapter.Rank,
		adapterReport.Adapter.Alpha,
		baseReport.Metrics.Loss,
		adapterReport.Metrics.Loss,
		baseReport.Metrics.Perplexity,
		adapterReport.Metrics.Perplexity,
	)
}

func gemma4SFTJSONLDataset(t *testing.T, cfg dataset.Config) dataset.Dataset {
	t.Helper()
	input := core.Join("\n",
		`{"messages":[{"role":"user","content":"What should a retained State runner preserve?"},{"role":"assistant","content":"It should preserve the useful KV state without replaying unchanged context."}]}`,
	)
	ds, err := dataset.LoadJSONL(strings.NewReader(input), cfg)
	if err != nil {
		t.Fatalf("dataset.LoadJSONL() error = %v", err)
	}
	return ds
}

func finitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}
