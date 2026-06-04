// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"testing"

	core "dappco.re/go"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

func TestInspectLoRAAdapter_ReadsMetadataAndHashes_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"rank":16,"alpha":32,"lora_layers":["self_attn.q_proj","self_attn.v_proj"]}`)

	info, err := lora.InspectAdapter(dir)
	if err != nil {
		t.Fatalf("lora.InspectAdapter() error = %v", err)
	}
	if info.Name != core.PathBase(dir) || info.Path != dir {
		t.Fatalf("adapter identity = %+v, want name/path", info)
	}
	if info.Rank != 16 || info.Alpha != 32 || info.Hash == "" {
		t.Fatalf("adapter metadata = %+v, want rank/alpha/hash", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"self_attn.q_proj", "self_attn.v_proj"}) {
		t.Fatalf("adapter targets = %v, want q/v", info.TargetKeys)
	}
}

func TestInspectLoRAAdapter_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := lora.InspectAdapter(dir)
	if err == nil {
		t.Fatal("expected missing adapter_config.json error")
	}
}

func TestInspectLoRAAdapter_SafetensorsPath_Ugly(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"r":4,"lora_alpha":8,"target_modules":["q_proj"]}`)
	path := core.PathJoin(dir, "adapter.safetensors")

	info, err := lora.InspectAdapter(path)
	if err != nil {
		t.Fatalf("lora.InspectAdapter(.safetensors) error = %v", err)
	}
	if info.Path != path || info.Name != "adapter.safetensors" || info.Rank != 4 || info.Alpha != 8 {
		t.Fatalf("adapter info = %+v, want safetensors path metadata", info)
	}
}

func TestStateBundleCompatibility_MatchingAdapter_Good(t *testing.T) {
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "qwen3", NumLayers: 1},
		Adapter: mlxbundle.Adapter{Path: "/adapters/a", Hash: "sha256:a", Rank: 8},
		KV:      stateBundleTestSnapshot(),
	}

	err := mlxbundle.CheckCompatibility(modelInfoToBundle(ModelInfo{
		Architecture: "qwen3",
		NumLayers:    1,
		Adapter:      lora.AdapterInfo{Path: "/adapters/a", Hash: "sha256:a", Rank: 8},
	}), b)
	if err != nil {
		t.Fatalf("CheckStateBundleCompatibility() error = %v", err)
	}
}

func TestStateBundleCompatibility_RejectsAdapterMismatch_Bad(t *testing.T) {
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "qwen3", NumLayers: 1},
		Adapter: mlxbundle.Adapter{Path: "/adapters/a", Hash: "sha256:a", Rank: 8},
		KV:      stateBundleTestSnapshot(),
	}

	err := mlxbundle.CheckCompatibility(modelInfoToBundle(ModelInfo{
		Architecture: "qwen3",
		NumLayers:    1,
		Adapter:      lora.AdapterInfo{Path: "/adapters/b", Hash: "sha256:b", Rank: 8},
	}), b)
	if err == nil {
		t.Fatal("expected adapter mismatch error")
	}
}

func TestStateBundleCompatibility_RejectsMissingAdapter_Ugly(t *testing.T) {
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		Adapter: mlxbundle.Adapter{Path: "/adapters/domain", Hash: "sha256:domain", Rank: 16},
		KV:      stateBundleTestSnapshot(),
	}

	err := mlxbundle.CheckCompatibility(modelInfoToBundle(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}), b)
	if err == nil {
		t.Fatal("expected missing active adapter error")
	}
}

func writeTestLoRAAdapter(t *testing.T, config string) string {
	t.Helper()
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(config), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter_config: %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub-weights"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
	}
	return dir
}

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
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "qwen3", NumLayers: 1},
		Adapter: mlxbundle.Adapter{Path: "/adapters/other", Hash: "sha256:other", Rank: 8},
		KV:      stateBundleTestSnapshot(),
	}

	restored, err := model.NewSessionFromBundle(b)
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
func TestNewLoRA_ForwardsRFCCompatibilityFields_Good(t *testing.T) {
	wantAdapter := &metal.LoRAAdapter{}
	native := &fakeNativeModel{loraAdapter: wantAdapter}
	model := &Model{model: native}

	got := NewLoRA(model, &LoRAConfig{
		Rank:         4,
		Scale:        1.5,
		TargetLayers: []string{"q_proj", "v_proj"},
		Lambda:       0.01,
		DType:        metal.DTypeBFloat16,
	})

	if got != wantAdapter {
		t.Fatalf("NewLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.lastLoRAConfig.Rank != 4 {
		t.Fatalf("Rank = %d, want 4", native.lastLoRAConfig.Rank)
	}
	if native.lastLoRAConfig.Scale != 1.5 {
		t.Fatalf("Scale = %f, want 1.5", native.lastLoRAConfig.Scale)
	}
	if native.lastLoRAConfig.Lambda != 0.01 {
		t.Fatalf("Lambda = %f, want 0.01", native.lastLoRAConfig.Lambda)
	}
	if native.lastLoRAConfig.DType != metal.DTypeBFloat16 {
		t.Fatalf("DType = %v, want %v", native.lastLoRAConfig.DType, metal.DTypeBFloat16)
	}
	if !reflect.DeepEqual(native.lastLoRAConfig.TargetLayers, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetLayers = %v, want [q_proj v_proj]", native.lastLoRAConfig.TargetLayers)
	}
	if len(native.lastLoRAConfig.TargetKeys) != 0 {
		t.Fatalf("TargetKeys = %v, want nil for RFC alias path", native.lastLoRAConfig.TargetKeys)
	}
}

func TestNewLoRA_ForwardsProbeSink_Good(t *testing.T) {
	recorder := probe.NewRecorder()
	wantAdapter := &metal.LoRAAdapter{}
	native := &fakeNativeModel{loraAdapter: wantAdapter}
	model := &Model{model: native}

	got := NewLoRA(model, &LoRAConfig{ProbeSink: recorder})

	if got != wantAdapter {
		t.Fatalf("NewLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.lastLoRAConfig.ProbeSink == nil {
		t.Fatal("native LoRA probe.Sink = nil, want configured")
	}
	native.lastLoRAConfig.ProbeSink.EmitProbe(metal.ProbeEvent{
		Kind:  metal.ProbeEventTraining,
		Phase: metal.ProbePhaseTraining,
		Training: &metal.ProbeTraining{
			Step: 3,
			Loss: 0.25,
		},
	})
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("probe events len = %d, want 1", len(events))
	}
	if events[0].Training == nil || events[0].Training.Step != 3 || events[0].Training.Loss != 0.25 {
		t.Fatalf("probe training event = %+v", events[0])
	}
}
