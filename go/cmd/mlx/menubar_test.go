// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"strings"
	"testing"

	core "dappco.re/go"
)

// TestMenubar_PrefsPath_Good — the prefs path lands under the macOS
// Application Support convention, rooted at $HOME.
func TestMenubar_PrefsPath_Good(t *testing.T) {
	t.Setenv("HOME", "/Users/test")
	got := prefsPath()
	want := core.PathJoin("/Users/test", "Library", "Application Support", "lthn-mlx", "preferences.json")
	if got != want {
		t.Fatalf("prefsPath = %q, want %q", got, want)
	}
}

// TestMenubar_SaveLoadPrefs_Good — round-trip: savePrefs then loadPrefs
// returns the stored model, written under a temp HOME.
func TestMenubar_SaveLoadPrefs_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	savePrefs(menubarPrefs{Model: "/models/lemer-lite"})
	got := loadPrefs()
	if got.Model != "/models/lemer-lite" {
		t.Fatalf("loaded model = %q, want the saved value", got.Model)
	}
}

// TestMenubar_LoadPrefs_Bad — when no preferences file exists yet, load
// returns the zero-value prefs rather than erroring.
func TestMenubar_LoadPrefs_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // empty dir, no preferences.json
	got := loadPrefs()
	if got.Model != "" {
		t.Fatalf("loaded model = %q, want empty for missing file", got.Model)
	}
}

// TestMenubar_LoadPrefs_Ugly — a corrupt (non-JSON) preferences file
// collapses to zero-value prefs instead of propagating a parse error
// into the boot path.
func TestMenubar_LoadPrefs_Ugly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	path := prefsPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(path, []byte("{not valid json"), 0o644); !r.OK {
		t.Fatalf("write corrupt prefs: %v", r.Value)
	}
	got := loadPrefs()
	if got.Model != "" {
		t.Fatalf("loaded model = %q, want empty for corrupt file", got.Model)
	}
}

// TestMenubar_PickModelPath_Good — a saved prefs model wins over both the
// env var and the built-in default.
func TestMenubar_PickModelPath_Good(t *testing.T) {
	t.Setenv("LTHN_MLX_MODEL", "/env/model")
	got := pickModelPath(menubarPrefs{Model: "/prefs/model"})
	if got != "/prefs/model" {
		t.Fatalf("pickModelPath = %q, want the prefs model", got)
	}
}

// TestMenubar_PickModelPath_Bad — with no saved prefs, the LTHN_MLX_MODEL
// env var is the next choice.
func TestMenubar_PickModelPath_Bad(t *testing.T) {
	t.Setenv("LTHN_MLX_MODEL", "/env/model")
	got := pickModelPath(menubarPrefs{})
	if got != "/env/model" {
		t.Fatalf("pickModelPath = %q, want the env model", got)
	}
}

// TestMenubar_PickModelPath_Ugly — neither prefs nor env set; the default
// lemer-lite snapshot path under ~/.cache/huggingface is returned.
func TestMenubar_PickModelPath_Ugly(t *testing.T) {
	t.Setenv("LTHN_MLX_MODEL", "")
	t.Setenv("HOME", "/Users/test")
	got := pickModelPath(menubarPrefs{})
	if !strings.Contains(got, "models--lthn--lemer-lite") {
		t.Fatalf("pickModelPath fallback = %q, want lemer-lite default", got)
	}
}

// TestMenubar_ShortPath_Good — a path under $HOME collapses the home
// prefix to ~ for a compact menu label.
func TestMenubar_ShortPath_Good(t *testing.T) {
	t.Setenv("HOME", "/Users/test")
	if got := shortPath("/Users/test/models/x"); got != "~/models/x" {
		t.Fatalf("shortPath = %q, want ~/models/x", got)
	}
}

// TestMenubar_ShortPath_Bad — a path outside $HOME is returned unchanged.
func TestMenubar_ShortPath_Bad(t *testing.T) {
	t.Setenv("HOME", "/Users/test")
	if got := shortPath("/opt/models/x"); got != "/opt/models/x" {
		t.Fatalf("shortPath = %q, want unchanged", got)
	}
}

// TestMenubar_ShortPath_Ugly — when HOME is unset the path is returned
// verbatim (no spurious "~" prefix on an empty-home match).
func TestMenubar_ShortPath_Ugly(t *testing.T) {
	t.Setenv("HOME", "")
	if got := shortPath("/Users/test/x"); got != "/Users/test/x" {
		t.Fatalf("shortPath (no HOME) = %q, want unchanged", got)
	}
}
