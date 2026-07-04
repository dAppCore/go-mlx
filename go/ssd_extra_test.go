// SPDX-Licence-Identifier: EUPL-1.2

// Extra coverage for ssd.go: the option-translation helper (ssdOptions) and
// the *Model nil guard on RunSSD. Neither loads a model.

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/mlx/spine"
)

func TestSSDOptions_ForwardsSetSamplerFields_Good(t *testing.T) {
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   1.2,
		TopK:          20,
		TopP:          0.8,
		MinP:          0.05,
		RepeatPenalty: 1.1,
	}
	applied := spine.ApplyGenerateOptions(ssdOptions(cfg))
	if applied.MaxTokens != 128 {
		t.Fatalf("MaxTokens = %d, want 128", applied.MaxTokens)
	}
	if applied.Temperature != 1.2 {
		t.Fatalf("Temperature = %v, want 1.2", applied.Temperature)
	}
	if applied.TopK != 20 || applied.TopP != 0.8 || applied.MinP != 0.05 {
		t.Fatalf("sampler fields = topK %d topP %v minP %v, want 20/0.8/0.05", applied.TopK, applied.TopP, applied.MinP)
	}
	if applied.RepeatPenalty != 1.1 {
		t.Fatalf("RepeatPenalty = %v, want 1.1", applied.RepeatPenalty)
	}
}

func TestSSDOptions_OmitsUnsetSamplerFields_Ugly(t *testing.T) {
	// Zero-valued optional fields must NOT emit their WithX option, so the
	// applied config keeps the defaults rather than forcing zeros.
	cfg := GenerateConfig{MaxTokens: 16, Temperature: 0.7}
	applied := spine.ApplyGenerateOptions(ssdOptions(cfg))
	if applied.MaxTokens != 16 || applied.Temperature != 0.7 {
		t.Fatalf("base fields = maxTokens %d temp %v, want 16/0.7", applied.MaxTokens, applied.Temperature)
	}
	if applied.TopK != 0 || applied.TopP != 0 || applied.MinP != 0 || applied.RepeatPenalty != 0 {
		t.Fatalf("unset sampler fields leaked: topK %d topP %v minP %v rp %v", applied.TopK, applied.TopP, applied.MinP, applied.RepeatPenalty)
	}
}

func TestModelRunSSD_NilModel_Bad(t *testing.T) {
	var m *Model
	_, err := m.RunSSD(context.Background(), nil, DefaultSSDConfig())
	if err == nil {
		t.Fatal("(*Model)(nil).RunSSD error = nil, want model-nil error")
	}
}
