// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// liveDatasetStream is a tiny resettable inference.DatasetStream feeding a fixed
// set of message samples — enough for Evaluate/TrainSFT to run a forward (and a
// LoRA step) on the live model without touching disk.
type liveDatasetStream struct {
	samples []inference.DatasetSample
	pos     int
}

func newLiveDatasetStream() *liveDatasetStream {
	// inferenceDataset.Next maps Prompt/Response/Text (not Messages) into the
	// tokenizable dataset.Sample, so the rows are stated as prompt+response.
	return &liveDatasetStream{samples: []inference.DatasetSample{
		{
			Prompt:   "What carries forward across a retained conversation?",
			Response: "The useful KV state, without replaying unchanged context.",
		},
		{
			Prompt:   "What should a State runner avoid?",
			Response: "Replaying unchanged context that the KV cache already holds.",
		},
	}}
}

func (s *liveDatasetStream) Next() (inference.DatasetSample, bool, error) {
	if s.pos >= len(s.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := s.samples[s.pos]
	s.pos++
	return sample, true, nil
}

func (s *liveDatasetStream) Reset() error {
	s.pos = 0
	return nil
}

// TestMetalAdapter_ContractTokenizerLiveModel drives the tokenizer/structure
// contract methods whose bodies dereference the real *metal.Model:
// Encode/Decode round-trip, NumLayers, InternalModel, ApplyChatTemplate,
// Capabilities (loaded arm), and ActiveAdapter on a clean model.
func TestMetalAdapter_ContractTokenizerLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()

	ids := adapter.Encode("hello world")
	if len(ids) == 0 {
		t.Fatal("Encode returned no tokens")
	}
	round := adapter.Decode(ids)
	if round == "" {
		t.Fatal("Decode returned empty string")
	}
	t.Logf("Encode(%q) -> %d tokens -> Decode %q", "hello world", len(ids), round)

	if n := adapter.NumLayers(); n == 0 {
		t.Errorf("NumLayers returned 0 on a loaded model")
	}

	// InternalModel exposes the underlying handle — non-nil on a loaded model.
	internal := adapter.InternalModel()
	if internal == nil {
		t.Errorf("InternalModel returned nil on a loaded model")
	}

	// ApplyChatTemplate over the loaded arch's chat config.
	formatted, err := adapter.ApplyChatTemplate([]inference.Message{
		{Role: "user", Content: "hi"},
	})
	if err != nil {
		t.Fatalf("ApplyChatTemplate: %v", err)
	}
	if formatted == "" {
		t.Fatal("ApplyChatTemplate returned empty prompt")
	}

	// Capabilities — the loaded (model != nil) arm.
	report := adapter.Capabilities()
	if !report.Available {
		t.Errorf("Capabilities.Available=false on a loaded model: %+v", report)
	}
	if report.Model.Architecture == "" {
		t.Errorf("Capabilities missing model architecture on loaded arm: %+v", report.Model)
	}

	// ActiveAdapter on a clean model — empty identity, loaded arm.
	_ = adapter.ActiveAdapter()
}

// TestMetalAdapter_ApplyLoRALiveModel drives metaladapter.ApplyLoRA — building a
// fresh LoRA adapter on the live model and translating the config.
func TestMetalAdapter_ApplyLoRALiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()

	loraAdapter := adapter.ApplyLoRA(inference.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"q_proj"},
		BFloat16:   true,
	})
	if loraAdapter == nil {
		t.Fatal("ApplyLoRA returned nil adapter")
	}
	// ActiveAdapter should now reflect a rank-2 adapter applied in-place.
	active := adapter.ActiveAdapter()
	t.Logf("ApplyLoRA -> active adapter rank=%d alpha=%.1f", active.Rank, active.Alpha)
}

// TestMetalAdapter_EvaluateLiveModel drives metaladapter.Evaluate — the real
// dataset-forward eval body past the nil-model guard.
func TestMetalAdapter_EvaluateLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()

	report, err := adapter.Evaluate(context.Background(), newLiveDatasetStream(), inference.EvalConfig{
		BatchSize: 1,
		MaxSeqLen: 128,
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if report == nil {
		t.Fatal("Evaluate returned nil report")
	}
	if report.Metrics.Tokens == 0 {
		t.Errorf("Evaluate report has no token coverage: %+v", report.Metrics)
	}
	t.Logf("Evaluate: samples=%d tokens=%d loss=%.4f ppl=%.4f",
		report.Metrics.Samples, report.Metrics.Tokens, report.Metrics.Loss, report.Metrics.Perplexity)
}

// TestMetalAdapter_TrainSFTLiveModel drives metaladapter.TrainSFT — a short
// in-memory LoRA SFT run on the live model, the heaviest contract body.
func TestMetalAdapter_TrainSFTLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()

	result, err := adapter.TrainSFT(context.Background(), newLiveDatasetStream(), inference.TrainingConfig{
		Epochs:       2,
		BatchSize:    1,
		LearningRate: 5e-3,
		LoRA: inference.LoRAConfig{
			Rank:       2,
			Alpha:      4,
			TargetKeys: []string{"q_proj"},
		},
	})
	if err != nil {
		t.Fatalf("TrainSFT: %v", err)
	}
	if result == nil {
		t.Fatal("TrainSFT returned nil result")
	}
	t.Logf("TrainSFT: epoch=%d step=%d loss=%.4f",
		result.Metrics.Epoch, result.Metrics.Step, result.Metrics.Loss)
}
