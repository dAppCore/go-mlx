// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"testing"

	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
)

func TestLoadModel_ExposesAdapterIdentityInInfoAndMetrics_Good(t *testing.T) {
	adapterDir := writeTestLoRAAdapter(t, `{"rank":8,"alpha":16,"lora_layers":["q_proj","v_proj"]}`)
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if cfg.AdapterPath != adapterDir {
			t.Fatalf("AdapterPath = %q, want %q", cfg.AdapterPath, adapterDir)
		}
		return &fakeNativeModel{
			info:    metal.ModelInfo{Architecture: "qwen3", NumLayers: 2},
			metrics: metal.Metrics{PromptTokens: 4},
		}, nil
	}

	model, err := LoadModel("/models/qwen3", WithAdapterPath(adapterDir))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	info := model.Info()
	metrics := model.Metrics()
	if info.Adapter.Path != adapterDir || info.Adapter.Rank != 8 || info.Adapter.Hash == "" {
		t.Fatalf("Info().Adapter = %+v, want loaded identity", info.Adapter)
	}
	if metrics.Adapter.Hash != info.Adapter.Hash || metrics.Adapter.Path != adapterDir {
		t.Fatalf("Metrics().Adapter = %+v, want same identity as Info", metrics.Adapter)
	}
}

func TestModelSwapLoRA_UpdatesAdapterIdentity_Good(t *testing.T) {
	first := writeTestLoRAAdapter(t, `{"rank":4,"alpha":8,"lora_layers":["q_proj"]}`)
	second := writeTestLoRAAdapter(t, `{"rank":16,"alpha":32,"lora_layers":["v_proj"]}`)
	native := &fakeNativeModel{loadedLoRAAdapter: &metal.LoRAAdapter{}}
	model := &Model{model: native}

	if _, err := model.LoadLoRA(first); err != nil {
		t.Fatalf("LoadLoRA() error = %v", err)
	}
	if model.Adapter().Path != first || model.Adapter().Rank != 4 {
		t.Fatalf("adapter after load = %+v, want first adapter", model.Adapter())
	}
	if _, err := model.SwapLoRA(second); err != nil {
		t.Fatalf("SwapLoRA() error = %v", err)
	}
	if model.Adapter().Path != second || model.Adapter().Rank != 16 {
		t.Fatalf("adapter after swap = %+v, want second adapter", model.Adapter())
	}
	if native.unloadLoRACalls != 1 {
		t.Fatalf("unload calls = %d, want 1", native.unloadLoRACalls)
	}
}

func TestModelNewSessionFromBundle_RejectsAdapterMismatch_Bad(t *testing.T) {
	session := &fakeNativeSession{}
	model := &Model{
		model:       &fakeNativeModel{session: session, info: metal.ModelInfo{Architecture: "qwen3", NumLayers: 1}},
		adapterInfo: lora.AdapterInfo{Path: "/adapters/live", Hash: "sha256:live", Rank: 8},
	}
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		Model:   StateBundleModel{Architecture: "qwen3", NumLayers: 1},
		Adapter: StateBundleAdapter{Path: "/adapters/other", Hash: "sha256:other", Rank: 8},
		KV:      stateBundleTestSnapshot(),
	}

	restored, err := model.NewSessionFromBundle(bundle)
	if err == nil {
		t.Fatal("expected adapter mismatch error")
	}
	if restored != nil {
		t.Fatalf("session = %v, want nil", restored)
	}
	if session.restoredKV != nil {
		t.Fatalf("session restored KV despite mismatch: %+v", session.restoredKV)
	}
}
