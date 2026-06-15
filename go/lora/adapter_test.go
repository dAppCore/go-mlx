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

func TestAdapter_IsEmpty_Good(t *testing.T) {
	// Good: the zero-value AdapterInfo (no inference adapter attached) is
	// the canonical "empty" case IsEmpty exists to recognise.
	var info AdapterInfo
	if !info.IsEmpty() {
		t.Fatalf("AdapterInfo{}.IsEmpty() = false, want true for zero value")
	}
}

func TestAdapter_IsEmpty_Bad(t *testing.T) {
	// Bad (for the empty predicate): a fully populated adapter identity is
	// emphatically not empty.
	info := AdapterInfo{
		Name:       "my-lora",
		Path:       "/models/my-lora",
		Hash:       "deadbeef",
		Rank:       16,
		Alpha:      32,
		Scale:      2,
		TargetKeys: []string{"self_attn.q_proj"},
	}
	if info.IsEmpty() {
		t.Fatalf("populated AdapterInfo.IsEmpty() = true, want false: %+v", info)
	}
}

func TestAdapter_IsEmpty_Ugly(t *testing.T) {
	// Ugly: prove that setting ANY single field on an otherwise-zero
	// AdapterInfo flips IsEmpty to false. This is the assertion that earns
	// its keep — it catches a field being dropped from the AND chain. One
	// mutator per field IsEmpty inspects.
	cases := []struct {
		name string
		set  func(*AdapterInfo)
	}{
		{"Name", func(a *AdapterInfo) { a.Name = "x" }},
		{"Path", func(a *AdapterInfo) { a.Path = "/x" }},
		{"Hash", func(a *AdapterInfo) { a.Hash = "abc" }},
		{"Rank", func(a *AdapterInfo) { a.Rank = 1 }},
		{"Alpha", func(a *AdapterInfo) { a.Alpha = 0.5 }},
		{"Scale", func(a *AdapterInfo) { a.Scale = 0.5 }},
		{"TargetKeys", func(a *AdapterInfo) { a.TargetKeys = []string{"q_proj"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var info AdapterInfo
			tc.set(&info)
			if info.IsEmpty() {
				t.Fatalf("AdapterInfo with only %s set reported IsEmpty() = true, want false: %+v", tc.name, info)
			}
		})
	}
}

func TestInspectLoRAAdapter_LargeShardStreamingHash_Good(t *testing.T) {
	// Drives InspectAdapter through the large-shard streaming hash path:
	// streamHashWeightFile only fires for weight files larger than
	// streamHashMinBytes (128 KiB), so a synthetic shard above that gate
	// exercises the streaming accumulator that the small stub fixtures
	// never reach. Asserts the hash is deterministic for identical content
	// and changes when a single byte changes — proving the streamed bytes
	// actually feed the digest.
	const shardSize = streamHashMinBytes + 64*1024 // ~192 KiB, well over the 128 KiB gate

	makeAdapter := func(t *testing.T, fillByte byte) AdapterInfo {
		t.Helper()
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter_config: %s", result.Error())
		}
		weights := make([]byte, shardSize)
		for i := range weights {
			weights[i] = fillByte
		}
		if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), weights, 0o600); !result.OK {
			t.Fatalf("WriteFile large shard: %s", result.Error())
		}
		info, err := InspectAdapter(dir)
		if err != nil {
			t.Fatalf("InspectAdapter(large shard) error = %v", err)
		}
		if info.Hash == "" {
			t.Fatalf("InspectAdapter(large shard) produced empty hash: %+v", info)
		}
		return info
	}

	first := makeAdapter(t, 0xAB)
	repeat := makeAdapter(t, 0xAB)
	if first.Hash != repeat.Hash {
		t.Fatalf("streaming hash not deterministic: %q != %q", first.Hash, repeat.Hash)
	}

	different := makeAdapter(t, 0xCD)
	if first.Hash == different.Hash {
		t.Fatalf("streaming hash collided across distinct shard content: %q", first.Hash)
	}
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
