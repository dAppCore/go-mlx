// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// audioFeatureTestConfig mirrors the E2B processor_config.json
// feature_extractor section (the config truth the loader reads).
func audioFeatureTestConfig() *Gemma4AudioFeatureConfig {
	return &Gemma4AudioFeatureConfig{
		FeatureSize:      128,
		SamplingRate:     16000,
		FrameLength:      320,
		HopLength:        160,
		FFTLength:        512,
		MinFrequency:     0,
		MaxFrequency:     8000,
		MelFloor:         1e-3,
		InputScaleFactor: 1,
		PreemphasisHTK:   true,
	}
}

// The goldens in audio_features_golden_test.go are the actual outputs of the
// HF transformers Gemma4AudioFeatureExtractor on the embedded waveforms —
// this is reference parity, not self-consistency. The tolerance covers
// float32-vs-float64 multiply ordering between numpy and Go; observed
// divergence is ~1e-6 on log-mel values spanning roughly [-7, +5].
func TestGemma4_AudioFeatures_GoldenParity_Good(t *testing.T) {
	extractor, err := NewGemma4AudioFeatureExtractor(audioFeatureTestConfig())
	if err != nil {
		t.Fatalf("NewGemma4AudioFeatureExtractor: %v", err)
	}
	const tolerance = 1e-4
	for _, golden := range audioFeatureGoldens {
		features, mask, frames, err := extractor.Extract(golden.samples)
		if err != nil {
			t.Fatalf("%s: Extract: %v", golden.name, err)
		}
		if frames != golden.frames {
			t.Fatalf("%s: frames = %d, want %d", golden.name, frames, golden.frames)
		}
		if len(mask) != len(golden.mask) {
			t.Fatalf("%s: mask length = %d, want %d", golden.name, len(mask), len(golden.mask))
		}
		for i := range mask {
			if mask[i] != golden.mask[i] {
				t.Fatalf("%s: mask[%d] = %v, want %v", golden.name, i, mask[i], golden.mask[i])
			}
		}
		if len(features) != len(golden.features) {
			t.Fatalf("%s: features length = %d, want %d", golden.name, len(features), len(golden.features))
		}
		maxDiff := 0.0
		maxAt := 0
		for i := range features {
			diff := math.Abs(float64(features[i]) - float64(golden.features[i]))
			if diff > maxDiff {
				maxDiff, maxAt = diff, i
			}
		}
		t.Logf("%s: %d frames, max |Δ| vs HF reference = %.3g (at flat index %d)", golden.name, frames, maxDiff, maxAt)
		if maxDiff > tolerance {
			t.Fatalf("%s: max |Δ| = %v exceeds %v (frame %d, mel bin %d: got %v want %v)",
				golden.name, maxDiff, tolerance, maxAt/128, maxAt%128, features[maxAt], golden.features[maxAt])
		}
	}
}

func TestGemma4_AudioFeatures_LoadConfig_Good(t *testing.T) {
	dir := t.TempDir()
	payload := []byte(`{
		"audio_ms_per_token": 40,
		"audio_seq_length": 750,
		"feature_extractor": {
			"feature_size": 128, "sampling_rate": 16000,
			"frame_length": 320, "hop_length": 160, "fft_length": 512,
			"min_frequency": 0.0, "max_frequency": 8000.0,
			"mel_floor": 0.001, "input_scale_factor": 1.0,
			"preemphasis": 0.0, "preemphasis_htk_flavor": true
		}
	}`)
	if r := core.WriteFile(core.PathJoin(dir, "processor_config.json"), payload, 0o600); !r.OK {
		t.Fatal("write processor_config.json failed")
	}
	cfg, err := LoadGemma4AudioFeatureConfig(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AudioFeatureConfig: %v", err)
	}
	if cfg == nil || cfg.FeatureSize != 128 || cfg.FrameLength != 320 || cfg.HopLength != 160 ||
		cfg.FFTLength != 512 || cfg.MaxFrequency != 8000 || cfg.MelFloor != 0.001 {
		t.Fatalf("loaded config = %+v, want the declared feature_extractor section", cfg)
	}
	if _, err := NewGemma4AudioFeatureExtractor(cfg); err != nil {
		t.Fatalf("extractor from loaded config: %v", err)
	}
}

func TestGemma4_AudioFeatures_NoProcessorConfig_Good(t *testing.T) {
	cfg, err := LoadGemma4AudioFeatureConfig(t.TempDir())
	if err != nil || cfg != nil {
		t.Fatalf("absent processor_config.json gave (%+v, %v), want (nil, nil)", cfg, err)
	}
}

// Converted snapshots ship partial feature_extractor sections — the
// mlx-community shape carries only sampling_rate / num_mel_filters /
// fft_length / hop_length. Absent fields resolve to the HF constructor
// defaults exactly as transformers does.
func TestGemma4_AudioFeatures_PartialConfigDefaults_Good(t *testing.T) {
	extractor, err := NewGemma4AudioFeatureExtractor(&Gemma4AudioFeatureConfig{
		SamplingRate:  16000,
		NumMelFilters: 128,
		FFTLength:     512,
		HopLength:     160,
	})
	if err != nil {
		t.Fatalf("partial config failed: %v", err)
	}
	cfg := extractor.cfg
	if cfg.FeatureSize != 128 || cfg.FrameLength != 320 || cfg.HopLength != 160 ||
		cfg.MaxFrequency != 8000 || cfg.MelFloor != 1e-3 {
		t.Fatalf("resolved config = %+v, want HF constructor defaults", cfg)
	}
	samples := make([]float32, 1600)
	if _, _, frames, err := extractor.Extract(samples); err != nil || frames != 10 {
		t.Fatalf("partial-config extract frames=%d err=%v, want 10", frames, err)
	}
}

func TestGemma4_AudioFeatures_FailLoud_Bad(t *testing.T) {
	if _, err := NewGemma4AudioFeatureExtractor(nil); err == nil {
		t.Fatal("nil config built an extractor")
	}
	bad := audioFeatureTestConfig()
	bad.FFTLength = 300 // not a power of two
	if _, err := NewGemma4AudioFeatureExtractor(bad); err == nil {
		t.Fatal("non-power-of-two FFT built an extractor")
	}
	band := audioFeatureTestConfig()
	band.MinFrequency = 9000 // above max: contradictory, not absent
	if _, err := NewGemma4AudioFeatureExtractor(band); err == nil {
		t.Fatal("empty mel band built an extractor")
	}

	extractor, err := NewGemma4AudioFeatureExtractor(audioFeatureTestConfig())
	if err != nil {
		t.Fatalf("NewGemma4AudioFeatureExtractor: %v", err)
	}
	if _, _, _, err := extractor.Extract(nil); err == nil {
		t.Fatal("empty waveform extracted")
	}
}
