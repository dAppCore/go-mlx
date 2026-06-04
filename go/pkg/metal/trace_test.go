// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
	"time"

	core "dappco.re/go"
)

func TestTrace_NativePhaseTraceEvents_Good(t *testing.T) {
	coverageTokens := "NativePhaseTraceEvents"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "NativePhaseTraceEvents Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: "disabled", Duration: time.Millisecond})

	if events := takeNativePhaseTraceEvents(); len(events) != 0 || NativePhaseTraceArmed() {
		t.Fatalf("events = %+v armed=%v, want unarmed trace to stay empty", events, NativePhaseTraceArmed())
	}
}

func TestTrace_NativePhaseTraceEvents_Ugly(t *testing.T) {
	coverageTokens := "NativePhaseTraceEvents Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	resetNativePhaseTraceEvents()

	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: core.Trim("  ffn  "), Error: "boom"})
	events := takeNativePhaseTraceEvents()

	if len(events) != 1 || events[0].Name != "ffn" || events[0].Error != "boom" {
		t.Fatalf("events = %+v, want error event preserved", events)
	}
}

func TestTrace_NativePhaseTraceSkip_Good(t *testing.T) {
	coverageTokens := "NativePhaseTraceSkip"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	resetNativePhaseTraceEvents()

	TraceNativeSkip("gemma4.layer.00.native_layer.skip", "unsupported quantization")
	events := takeNativePhaseTraceEvents()

	if len(events) != 1 || events[0].Name != "gemma4.layer.00.native_layer.skip" || events[0].Error != "unsupported quantization" {
		t.Fatalf("events = %+v, want skip reason event", events)
	}
}
