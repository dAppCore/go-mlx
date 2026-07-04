// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	"dappco.re/go/inference/memory"
)

// TestCacheMode_ParseRuntimeCacheMode_Good — a recognised mode parses,
// trims surrounding whitespace, and reports present=true so the caller
// knows to apply the override.
func TestCacheMode_ParseRuntimeCacheMode_Good(t *testing.T) {
	mode, present := parseRuntimeCacheMode("  fp16 ")
	if !present {
		t.Fatal("present = false, want true for a non-empty value")
	}
	if mode != memory.KVCacheModeFP16 {
		t.Fatalf("mode = %q, want fp16", mode)
	}
}

// TestCacheMode_ParseRuntimeCacheMode_Bad — an empty / whitespace-only
// flag is the "no override" signal: present=false, mode=default.
func TestCacheMode_ParseRuntimeCacheMode_Bad(t *testing.T) {
	for _, raw := range []string{"", "   ", "\t"} {
		mode, present := parseRuntimeCacheMode(raw)
		if present {
			t.Fatalf("parseRuntimeCacheMode(%q) present = true, want false", raw)
		}
		if mode != memory.KVCacheModeDefault {
			t.Fatalf("parseRuntimeCacheMode(%q) mode = %q, want default", raw, mode)
		}
	}
}

// TestCacheMode_ParseRuntimeCacheMode_Ugly — parse does not validate; an
// unknown-but-non-empty token is returned verbatim with present=true. It
// is isRuntimeCacheMode's job (not parse's) to reject it downstream.
func TestCacheMode_ParseRuntimeCacheMode_Ugly(t *testing.T) {
	mode, present := parseRuntimeCacheMode("not-a-real-mode")
	if !present {
		t.Fatal("present = false, want true (parse is non-validating)")
	}
	if mode != memory.KVCacheMode("not-a-real-mode") {
		t.Fatalf("mode = %q, want the verbatim token", mode)
	}
}

// TestCacheMode_IsRuntimeCacheMode_Good — every known non-default mode is
// accepted as a runtime override.
func TestCacheMode_IsRuntimeCacheMode_Good(t *testing.T) {
	for _, mode := range []memory.KVCacheMode{
		memory.KVCacheModeFP16,
		memory.KVCacheModeQ8,
		memory.KVCacheModeKQ8VQ4,
		memory.KVCacheModePaged,
		memory.KVCacheModeTurboQuant,
	} {
		if !isRuntimeCacheMode(mode) {
			t.Fatalf("isRuntimeCacheMode(%q) = false, want true", mode)
		}
	}
}

// TestCacheMode_IsRuntimeCacheMode_Bad — the default (empty) mode is not
// a runtime override; it means "leave the engine default in place".
func TestCacheMode_IsRuntimeCacheMode_Bad(t *testing.T) {
	if isRuntimeCacheMode(memory.KVCacheModeDefault) {
		t.Fatal("isRuntimeCacheMode(default) = true, want false")
	}
}

// TestCacheMode_IsRuntimeCacheMode_Ugly — a non-empty but unrecognised
// mode is rejected so a typo'd --cache-mode flag never silently engages.
func TestCacheMode_IsRuntimeCacheMode_Ugly(t *testing.T) {
	if isRuntimeCacheMode(memory.KVCacheMode("fp17")) {
		t.Fatal("isRuntimeCacheMode(fp17) = true, want false for an unknown mode")
	}
}
