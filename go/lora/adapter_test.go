// SPDX-Licence-Identifier: EUPL-1.2

// Tests for adapter.go — InspectAdapter metadata/hash extraction. Moved
// from the root lora_adapter_test.go in the orphan sweep: the symbol
// lives here, so its tests do too.

package lora

import (
	"testing"

	core "dappco.re/go"
)

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestInspectLoRAAdapter_ReadsMetadataAndHashes_Good(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"rank":16,"alpha":32,"lora_layers":["self_attn.q_proj","self_attn.v_proj"]}`)

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
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

	_, err := InspectAdapter(dir)
	if err == nil {
		t.Fatal("expected missing adapter_config.json error")
	}
}

func TestInspectLoRAAdapter_SafetensorsPath_Ugly(t *testing.T) {
	dir := writeTestLoRAAdapter(t, `{"r":4,"lora_alpha":8,"target_modules":["q_proj"]}`)
	path := core.PathJoin(dir, "adapter.safetensors")

	info, err := InspectAdapter(path)
	if err != nil {
		t.Fatalf("InspectAdapter(.safetensors) error = %v", err)
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

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
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

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter() error = %v", err)
	}
	if info.Rank != 0 || info.Alpha != 0 || info.Scale != 0 {
		t.Fatalf("adapter metadata = %+v, want missing rank/alpha/scale preserved", info)
	}
	if !equalStringSlices(info.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("adapter targets = %v, want target_modules alias", info.TargetKeys)
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
