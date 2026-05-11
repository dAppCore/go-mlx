// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/lora"
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
