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
	// Drives InspectAdapter (delegated to inference/lora) through its
	// large-shard streaming hash path: inference/lora streams any weight
	// file above its own internal size gate rather than reading it whole,
	// so a synthetic shard comfortably above any sane gate exercises the
	// streaming accumulator that the small stub fixtures never reach.
	// Asserts the hash is deterministic for identical content and changes
	// when a single byte changes — proving the streamed bytes actually
	// feed the digest. The exact gate is now an inference/lora
	// implementation detail, so size generously rather than referencing it.
	const shardSize = 512 * 1024 // 512 KiB, well over any sane streaming gate

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

func TestInspectLoRAAdapter_MalformedConfig_Bad(t *testing.T) {
	// Bad: adapter_config.json exists and reads OK, but the bytes are not
	// valid JSON. ParseConfig fails, so Inspect must surface a parse error
	// (the L71-73 wrap) rather than silently returning a zero AdapterInfo.
	// This is reachable without fault injection — a corrupt config on disk
	// is an ordinary user-facing failure, not a simulated IO fault.
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte("{not valid json"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter_config: %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("stub"), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
	}

	_, err := InspectAdapter(dir)
	if err == nil {
		t.Fatal("InspectAdapter(malformed config) error = nil, want parse error")
	}
	if !core.Contains(err.Error(), "parse adapter_config.json") {
		t.Fatalf("error = %v, want parse-config context", err)
	}
}

func TestInspectLoRAAdapter_EmptyPath_Bad(t *testing.T) {
	// Bad: an empty adapter path is the guard at the top of Inspect — both
	// entry points delegate to inference/lora, which returns its own
	// unexported sentinel before touching the filesystem. We can no longer
	// name that sentinel from this package, so assert on the stable message
	// content instead. InspectAdapter forwards path as both arguments, so
	// the public entry point exercises the same guard.
	if _, err := InspectAdapter(""); err == nil || !core.Contains(err.Error(), "adapter path is required") {
		t.Fatalf("InspectAdapter(\"\") error = %v, want an adapter-path-required error", err)
	}
	if _, err := Inspect("", "/some/identity"); err == nil || !core.Contains(err.Error(), "adapter path is required") {
		t.Fatalf("Inspect(\"\", identity) error = %v, want an adapter-path-required error", err)
	}
}

func TestInspectLoRAAdapter_UnreadableShardSkipped_Ugly(t *testing.T) {
	// Ugly: hashAdapter globs *.safetensors and hashes each match, skipping
	// any entry it cannot read (the L208-209 !ok branch in hashWeightFile).
	// A *directory* whose name ends in .safetensors is matched by the glob
	// but fails ReadFile — so the skip fires without any permission games.
	// The resulting hash must still be deterministic and must equal the hash
	// of an adapter that never had the unreadable entry, proving the skipped
	// directory contributed nothing to the digest.
	makeAdapter := func(t *testing.T, withDir bool) AdapterInfo {
		t.Helper()
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter_config: %s", result.Error())
		}
		if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("real-weights"), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter.safetensors: %s", result.Error())
		}
		if withDir {
			// A directory matched by the *.safetensors glob — ReadFile on a
			// directory fails, so hashWeightFile returns ok=false and the
			// cursor does not advance for it.
			if result := core.MkdirAll(core.PathJoin(dir, "extra.safetensors"), 0o755); !result.OK {
				t.Fatalf("MkdirAll extra.safetensors: %s", result.Error())
			}
		}
		info, err := InspectAdapter(dir)
		if err != nil {
			t.Fatalf("InspectAdapter error = %v", err)
		}
		if info.Hash == "" {
			t.Fatalf("InspectAdapter produced empty hash: %+v", info)
		}
		return info
	}

	withSkip := makeAdapter(t, true)
	withoutSkip := makeAdapter(t, false)
	if withSkip.Hash != withoutSkip.Hash {
		t.Fatalf("unreadable .safetensors directory altered the digest: %q != %q", withSkip.Hash, withoutSkip.Hash)
	}
}

func TestJoinDirChildPattern_Branches_Ugly(t *testing.T) {
	// Ugly: joinDirChildPattern is the filepath.Clean-skipping join shared by
	// the hash and fuse paths. It has three branches the hot tests never hit
	// individually — empty dir (relative passthrough), dir already ending in
	// '/' (collapse the duplicate separator), and the plain insert-separator
	// case. One assertion per branch.
	cases := []struct {
		name  string
		dir   string
		child string
		want  string
	}{
		{"emptyDir", "", "*.safetensors", "*.safetensors"},
		{"trailingSlash", "/models/lora/", "*.safetensors", "/models/lora/*.safetensors"},
		{"plainJoin", "/models/lora", "*.safetensors", "/models/lora/*.safetensors"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := joinDirChildPattern(tc.dir, tc.child); got != tc.want {
				t.Fatalf("joinDirChildPattern(%q, %q) = %q, want %q", tc.dir, tc.child, got, tc.want)
			}
		})
	}
}

func TestResultError_Branches_GoodBadUgly(t *testing.T) {
	// resultError unwraps a core.Result into a plain error for core.E
	// chaining. Three branches, all reachable with synthetic Results — no
	// fault injection needed:
	//   Good: an OK result has no error.
	//   Bad:  a failed result carrying an error returns that error.
	//   Ugly: a failed result whose Value is NOT an error falls back to the
	//         errResultFailed sentinel (the defensive L289 path).
	if err := resultError(core.Ok([]byte("data"))); err != nil {
		t.Fatalf("resultError(Ok) = %v, want nil", err)
	}
	sentinel := core.NewError("boom")
	if err := resultError(core.Fail(sentinel)); err != sentinel {
		t.Fatalf("resultError(Fail(err)) = %v, want the wrapped error", err)
	}
	nonError := core.Result{Value: "a bare string, not an error", OK: false}
	if err := resultError(nonError); err != errResultFailed {
		t.Fatalf("resultError(non-error failure) = %v, want errResultFailed fallback", err)
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
