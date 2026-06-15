// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
	"time"

	core "dappco.re/go"
)

func TestTrace_NativePhaseTraceEvents_Good(t *testing.T) {
	resetNativePhaseTraceEvents()

	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: "gemma4.layer.00.attention", Duration: time.Millisecond, Pages: 8, Tokens: 8192})
	events := takeNativePhaseTraceEvents()

	if len(events) != 1 || events[0].Name != "gemma4.layer.00.attention" || events[0].Duration != time.Millisecond || events[0].Pages != 8 || events[0].Tokens != 8192 {
		t.Fatalf("events = %+v, want one attention event", events)
	}
	if again := takeNativePhaseTraceEvents(); len(again) != 0 {
		t.Fatalf("events after take = %+v, want empty", again)
	}
}

func TestTrace_NativePhaseTraceEvents_Bad(t *testing.T) {
	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: "disabled", Duration: time.Millisecond})

	if events := takeNativePhaseTraceEvents(); len(events) != 0 || NativePhaseTraceArmed() {
		t.Fatalf("events = %+v armed=%v, want unarmed trace to stay empty", events, NativePhaseTraceArmed())
	}
}

func TestTrace_NativePhaseTraceEvents_Ugly(t *testing.T) {
	resetNativePhaseTraceEvents()

	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: core.Trim("  ffn  "), Error: "boom"})
	events := takeNativePhaseTraceEvents()

	if len(events) != 1 || events[0].Name != "ffn" || events[0].Error != "boom" {
		t.Fatalf("events = %+v, want error event preserved", events)
	}
}

func TestTrace_NativePhaseTraceSkip_Good(t *testing.T) {
	resetNativePhaseTraceEvents()

	TraceNativeSkip("gemma4.layer.00.native_layer.skip", "unsupported quantization")
	events := takeNativePhaseTraceEvents()

	if len(events) != 1 || events[0].Name != "gemma4.layer.00.native_layer.skip" || events[0].Error != "unsupported quantization" {
		t.Fatalf("events = %+v, want skip reason event", events)
	}
}

func TestTrace_NativePhaseTraceSkip_Bad(t *testing.T) {
	// Unarmed trace, or an empty name/reason, both drop the skip event.
	if NativePhaseTraceArmed() {
		takeNativePhaseTraceEvents() // disarm without leaving residue
	}
	TraceNativeSkip("unarmed.skip", "reason")
	if events := takeNativePhaseTraceEvents(); len(events) != 0 {
		t.Fatalf("unarmed skip events = %+v, want empty", events)
	}

	resetNativePhaseTraceEvents()
	TraceNativeSkip("", "reason") // empty name → dropped
	TraceNativeSkip("named", "")  // empty reason → dropped
	if events := takeNativePhaseTraceEvents(); len(events) != 0 {
		t.Fatalf("empty-field skip events = %+v, want empty", events)
	}
}

func TestTrace_NativePhaseValueHashCapture_Good(t *testing.T) {
	// The phase value-hash capture toggle is a pure-Go diagnostic switch — set
	// it, observe it reads back, clear it. Save/restore so the global state is
	// untouched for concurrent sibling tests.
	prev := NativePhaseValueHashEnabled()
	defer SetNativePhaseValueHashCapture(prev)

	SetNativePhaseValueHashCapture(true)
	if !NativePhaseValueHashEnabled() {
		t.Fatal("after SetNativePhaseValueHashCapture(true), NativePhaseValueHashEnabled() = false")
	}
	SetNativePhaseValueHashCapture(false)
	if NativePhaseValueHashEnabled() {
		t.Fatal("after SetNativePhaseValueHashCapture(false), NativePhaseValueHashEnabled() = true")
	}
}

func TestTrace_TakeNativePhaseValueHashes_Bad(t *testing.T) {
	// With no captured hashes, Take returns an empty (clearing) result, never
	// nil-deref or stale data — and a second take is still empty.
	first := TakeNativePhaseValueHashes()
	if len(first) != 0 {
		t.Fatalf("TakeNativePhaseValueHashes (none captured) = %+v, want empty", first)
	}
	if again := TakeNativePhaseValueHashes(); len(again) != 0 {
		t.Fatalf("second take = %+v, want empty", again)
	}
}

func TestTrace_NativePhaseMaterializeTraceEnabled_Good(t *testing.T) {
	// The materialize-trace steering flag is off by default (never ambient env);
	// it is a code-only diagnostic. Assert the documented default.
	if NativePhaseMaterializeTraceEnabled() {
		t.Fatal("NativePhaseMaterializeTraceEnabled() = true at rest, want false (off by default)")
	}
}
