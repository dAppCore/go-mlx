// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for pure-Go validators / converters reachable with no model:
// normalizeLoadConfig's negative-field and device-normalisation arms,
// normalizeKVCacheStorageDType, PlanMemory's non-M2 pack arm, and the
// Model.Adapter resolution arms (cached vs info-derived) via fakeNativeModel.

package mlx

import (
	"testing"

	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/pkg/metal"
)

func TestNormalizeLoadConfig_NegativeFieldArms(t *testing.T) {
	bad := []struct {
		name string
		cfg  LoadConfig
	}{
		{"context", LoadConfig{ContextLength: -1}},
		{"parallel", LoadConfig{ParallelSlots: -1}},
		{"promptcachemin", LoadConfig{PromptCacheMinTokens: -1}},
		{"quantization", LoadConfig{Quantization: -1}},
		{"batchsize", LoadConfig{BatchSize: -1}},
		{"prefillchunk", LoadConfig{PrefillChunkSize: -1}},
		{"expectedquant", LoadConfig{ExpectedQuantization: -1}},
		{"pagedkvpage", LoadConfig{PagedKVPageSize: -1}},
		{"fixedsliding", LoadConfig{FixedSlidingCacheSize: -1}},
	}
	for _, tc := range bad {
		if _, err := normalizeLoadConfig(tc.cfg); err == nil {
			t.Errorf("%s: negative field should error", tc.name)
		}
	}

	// PromptCache with zero min-tokens → default applied.
	cfg, err := normalizeLoadConfig(LoadConfig{PromptCache: true})
	if err != nil {
		t.Fatalf("valid prompt-cache config errored: %v", err)
	}
	if cfg.PromptCacheMinTokens != DefaultPromptCacheMinTokens {
		t.Fatalf("PromptCacheMinTokens = %d, want default %d", cfg.PromptCacheMinTokens, DefaultPromptCacheMinTokens)
	}

	// Unknown KV cache mode rejected.
	if _, err := normalizeLoadConfig(LoadConfig{CacheMode: memory.KVCacheMode("nonsense")}); err == nil {
		t.Fatalf("unknown cache mode should error")
	}
}

func TestNormalizeLoadConfig_DeviceArms(t *testing.T) {
	// Canonical fast paths.
	for in, want := range map[string]string{"": "gpu", "gpu": "gpu", "cpu": "cpu"} {
		cfg, err := normalizeLoadConfig(LoadConfig{Device: in})
		if err != nil {
			t.Fatalf("device %q errored: %v", in, err)
		}
		if cfg.Device != want {
			t.Fatalf("device %q -> %q, want %q", in, cfg.Device, want)
		}
	}

	// Slow path: mixed-case canonical device normalises.
	cfg, err := normalizeLoadConfig(LoadConfig{Device: "  GPU  "})
	if err != nil {
		t.Fatalf("'  GPU  ' errored: %v", err)
	}
	if cfg.Device != "gpu" {
		t.Fatalf("'  GPU  ' -> %q, want gpu", cfg.Device)
	}

	// Slow path: whitespace-only device → gpu default.
	cfg, err = normalizeLoadConfig(LoadConfig{Device: "   "})
	if err != nil {
		t.Fatalf("whitespace device errored: %v", err)
	}
	if cfg.Device != "gpu" {
		t.Fatalf("whitespace device -> %q, want gpu", cfg.Device)
	}

	// Slow path: unknown device rejected.
	if _, err := normalizeLoadConfig(LoadConfig{Device: "tpu"}); err == nil {
		t.Fatalf("unknown device should error")
	}
}

func TestNormalizeKVCacheStorageDType_AllArms(t *testing.T) {
	cases := map[string]string{
		"":           "",
		"native":     "",
		"default":    "",
		"fp16":       "fp16",
		"float16":    "fp16",
		"f16":        "fp16",
		"bf16":       "bf16",
		"bfloat16":   "bf16",
		"  FP16  ":   "fp16", // trim + lower
		"int4-weird": "unsupported",
	}
	for in, want := range cases {
		if got := normalizeKVCacheStorageDType(in); got != want {
			t.Errorf("normalizeKVCacheStorageDType(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPlanMemory_NonM2Pack(t *testing.T) {
	// Non-nil pack without MiniMax M2 metadata → returns the base plan
	// (exercises the pack-present-but-not-M2 early return).
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{Architecture: "apple-m3", MemorySize: 32 << 30},
		Pack:   &mp.ModelPack{Architecture: "gemma4_text"},
	})
	if plan.ModelForwardSkeletonValidated {
		t.Fatalf("non-M2 pack should not validate an M2 skeleton")
	}
}

func TestModelAdapter_ResolutionArms(t *testing.T) {
	// Nil model.
	if got := (*Model)(nil).Adapter(); !got.IsEmpty() {
		t.Fatalf("nil model Adapter should be empty, got %+v", got)
	}

	// Cached adapterInfo wins.
	cached := &Model{adapterInfo: lora.AdapterInfo{Name: "cached-lora", Hash: "h"}}
	if got := cached.Adapter(); got.Name != "cached-lora" {
		t.Fatalf("cached adapter not returned: %+v", got)
	}

	// No cache, model present → derive from Info().Adapter.
	fromInfo := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Adapter: metal.AdapterInfo{Name: "info-lora"}}}}
	if got := fromInfo.Adapter(); got.Name != "info-lora" {
		t.Fatalf("info-derived adapter not returned: %+v", got)
	}

	// No cache, no model → empty.
	if got := (&Model{}).Adapter(); !got.IsEmpty() {
		t.Fatalf("empty model Adapter should be empty, got %+v", got)
	}
}
