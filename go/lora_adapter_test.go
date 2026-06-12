// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/internal/sessionfake"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
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

func TestInspectLoRAAdapter_UsesSharedConfigPrecedence_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{
		"rank": 4,
		"scale": 2,
		"target_keys": ["explicit"],
		"target_modules": ["peft"],
		"lora_layers": ["mlx-lm"]
	}`)

	info, err := lora.InspectAdapter(dir)
	if err != nil {
		t.Fatalf("lora.InspectAdapter() error = %v", err)
	}
	if info.Rank != 4 || info.Alpha != 8 || info.Scale != 2 {
		t.Fatalf("adapter metadata = %+v, want scale-derived alpha", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"explicit"}) {
		t.Fatalf("adapter targets = %v, want shared explicit target_keys precedence", info.TargetKeys)
	}
}

func TestInspectLoRAAdapter_PreservesMissingRank_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"target_modules":["q_proj"]}`)

	info, err := lora.InspectAdapter(dir)
	if err != nil {
		t.Fatalf("lora.InspectAdapter() error = %v", err)
	}
	if info.Rank != 0 || info.Alpha != 0 || info.Scale != 0 {
		t.Fatalf("adapter metadata = %+v, want missing rank/alpha/scale preserved", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("adapter targets = %v, want target_modules alias", info.TargetKeys)
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

	err := mlxbundle.CheckCompatibility(spine.ModelInfoToBundle(ModelInfo{
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

	err := mlxbundle.CheckCompatibility(spine.ModelInfoToBundle(ModelInfo{
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

	err := mlxbundle.CheckCompatibility(spine.ModelInfoToBundle(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}), b)
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
	adapterDir := writeTestLoRAAdapter(t, `{"r":8,"lora_alpha":16,"target_modules":["q_proj","v_proj"]}`)
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (NativeModel, error) {
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
	if !equalStringSlices(info.Adapter.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("Info().Adapter.TargetKeys = %v, want PEFT target_modules", info.Adapter.TargetKeys)
	}
	if metrics.Adapter.Hash != info.Adapter.Hash || metrics.Adapter.Path != adapterDir {
		t.Fatalf("Metrics().Adapter = %+v, want same identity as Info", metrics.Adapter)
	}
}

func TestLoadModel_MergesNativeAdapterDefaultsIntoIdentity_Good(t *testing.T) {
	adapterDir := writeTestLoRAAdapter(t, `{"target_modules":["q_proj"]}`)
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (NativeModel, error) {
		if cfg.AdapterPath != adapterDir {
			t.Fatalf("AdapterPath = %q, want %q", cfg.AdapterPath, adapterDir)
		}
		return &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "qwen3",
				NumLayers:    2,
				Adapter: metal.AdapterInfo{
					Rank:       8,
					Alpha:      16,
					Scale:      2,
					TargetKeys: []string{"q_proj"},
				},
			},
		}, nil
	}

	model, err := LoadModel("/models/qwen3", WithAdapterPath(adapterDir))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	info := model.Info()
	if info.Adapter.Path != adapterDir || info.Adapter.Hash == "" {
		t.Fatalf("Info().Adapter identity = %+v, want inspected path/hash", info.Adapter)
	}
	if info.Adapter.Rank != 8 || info.Adapter.Alpha != 16 || info.Adapter.Scale != 2 {
		t.Fatalf("Info().Adapter = %+v, want native-normalised rank/alpha/scale", info.Adapter)
	}
	if !equalStringSlices(info.Adapter.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("Info().Adapter.TargetKeys = %v, want native-normalised targets", info.Adapter.TargetKeys)
	}
}

func TestModelLoadLoRA_MergesLoadedAdapterDefaultsIntoIdentity_Good(t *testing.T) {
	adapterDir := writeTestLoRAAdapter(t, `{"target_modules":["q_proj"]}`)
	native := &fakeNativeModel{
		loadedLoRAAdapter: &metal.LoRAAdapter{
			Config: metal.LoRAConfig{
				Rank:       8,
				Alpha:      16,
				Scale:      2,
				TargetKeys: []string{"q_proj"},
			},
		},
	}
	model := &Model{model: native}

	if _, err := model.LoadLoRA(adapterDir); err != nil {
		t.Fatalf("LoadLoRA() error = %v", err)
	}
	info := model.Adapter()
	if info.Path != adapterDir || info.Hash == "" {
		t.Fatalf("Adapter() identity = %+v, want inspected path/hash", info)
	}
	if info.Rank != 8 || info.Alpha != 16 || info.Scale != 2 {
		t.Fatalf("Adapter() = %+v, want loaded adapter defaults", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("Adapter().TargetKeys = %v, want loaded adapter targets", info.TargetKeys)
	}
	metrics := model.Metrics()
	if metrics.Adapter.Rank != 8 || metrics.Adapter.Path != adapterDir {
		t.Fatalf("Metrics().Adapter = %+v, want merged loaded identity", metrics.Adapter)
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
	session := &sessionfake.Handle{}
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
	if session.RestoredKV != nil {
		t.Fatalf("session restored KV despite mismatch: %+v", session.RestoredKV)
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

func TestNewLoRA_LeavesNilConfigToNativeNormaliser_Good(t *testing.T) {
	wantAdapter := &metal.LoRAAdapter{}
	native := &fakeNativeModel{loraAdapter: wantAdapter}
	model := &Model{model: native}

	got := NewLoRA(model, nil)

	if got != wantAdapter {
		t.Fatalf("NewLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.lastLoRAConfig.Rank != 0 || native.lastLoRAConfig.Alpha != 0 || native.lastLoRAConfig.Scale != 0 || native.lastLoRAConfig.DType != 0 {
		t.Fatalf("last LoRA config = %+v, want zero scalar overrides", native.lastLoRAConfig)
	}
	if len(native.lastLoRAConfig.TargetKeys) != 0 || len(native.lastLoRAConfig.TargetLayers) != 0 {
		t.Fatalf("last LoRA targets = %v/%v, want native defaults", native.lastLoRAConfig.TargetKeys, native.lastLoRAConfig.TargetLayers)
	}
}

func TestNewLoRA_ForwardsExplicitDefaults_Good(t *testing.T) {
	wantAdapter := &metal.LoRAAdapter{}
	native := &fakeNativeModel{loraAdapter: wantAdapter}
	model := &Model{model: native}
	cfg := DefaultLoRAConfig()

	got := NewLoRA(model, &cfg)

	if got != wantAdapter {
		t.Fatalf("NewLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.lastLoRAConfig.Rank != 8 || native.lastLoRAConfig.Alpha != 16 || native.lastLoRAConfig.Scale != 2 {
		t.Fatalf("rank/alpha/scale = %d/%f/%f, want generic defaults", native.lastLoRAConfig.Rank, native.lastLoRAConfig.Alpha, native.lastLoRAConfig.Scale)
	}
	if !equalStringSlices(native.lastLoRAConfig.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetKeys = %v, want explicit generic defaults", native.lastLoRAConfig.TargetKeys)
	}
	cfg.TargetKeys[0] = "mutated"
	if native.lastLoRAConfig.TargetKeys[0] == "mutated" {
		t.Fatalf("TargetKeys aliases caller slice: %v", native.lastLoRAConfig.TargetKeys)
	}
}

func TestInferenceLoRAConfig_LeavesDefaultsToNativeNormaliser_Good(t *testing.T) {
	cfg := toMetalInferenceLoRAConfig(inference.LoRAConfig{})
	if cfg.Rank != 0 || cfg.Alpha != 0 || cfg.Scale != 0 || cfg.DType != 0 || len(cfg.TargetKeys) != 0 || len(cfg.TargetLayers) != 0 {
		t.Fatalf("toMetalInferenceLoRAConfig(empty) = %+v, want no root-side defaults", cfg)
	}
}

func TestInferenceLoRAConfig_ForwardsExplicitDefaults_Good(t *testing.T) {
	src := inference.DefaultLoRAConfig()
	cfg := toMetalInferenceLoRAConfig(src)
	if cfg.Rank != 8 || cfg.Alpha != 16 {
		t.Fatalf("rank/alpha = %d/%f, want inference defaults", cfg.Rank, cfg.Alpha)
	}
	if !equalStringSlices(cfg.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetKeys = %v, want explicit inference defaults", cfg.TargetKeys)
	}
	src.TargetKeys[0] = "mutated"
	if cfg.TargetKeys[0] == "mutated" {
		t.Fatalf("TargetKeys aliases caller slice: %v", cfg.TargetKeys)
	}
}

func TestInferenceLoRAConfig_ForwardsBFloat16_Good(t *testing.T) {
	cfg := toMetalInferenceLoRAConfig(inference.LoRAConfig{BFloat16: true})
	if cfg.DType != metal.DTypeBFloat16 {
		t.Fatalf("DType = %v, want BFloat16", cfg.DType)
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
