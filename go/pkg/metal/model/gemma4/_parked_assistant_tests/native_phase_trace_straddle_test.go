// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// PARKED — ESCALATED. This test (relocated from package metal's decode_test.go
// with the gemma4-internal nativeGemma4FixedGreedyToken kernel) inspects the
// native-phase trace event stream to prove the model-greedy path SKIPS when the
// MoE native layer is disabled. The phase-trace primitives are split:
//
//   - public (gemma4 production already depends on these):
//       metal.NativePhaseTraceArmed, metal.AppendNativePhaseTraceEvent,
//       metal.NativePhaseTrace (type), metal.TraceNativeMaterialize/Skip
//   - UNEXPORTED test seam (no public wrapper):
//       resetNativePhaseTraceEvents, takeNativePhaseTraceEvents (metal/trace.go)
//
// From package gemma4 there is no way to arm + drain the trace buffer, so the
// test cannot compile here, and it cannot stay in package metal (the kernel it
// exercises, nativeGemma4FixedGreedyToken, moved to package gemma4). It is NOT
// gutted — it is parked verbatim pending a decision on the minimal sanctioned
// seam: export ResetNativePhaseTraceEvents() + TakeNativePhaseTraceEvents()
// []NativePhaseTrace in metal/trace.go (thin wrappers completing the already-
// public NativePhaseTrace API), then move this into decode_kernels_test.go.
// This directory is Go-ignored so it does not block the build. See the report.
package gemma4

import (
	"testing"

	. "dappco.re/go/mlx/pkg/metal"
)

func TestDecode_nativeGemma4FixedGreedyToken_MoEGateSkip_Ugly(t *testing.T) {
	target := "nativeGemma4FixedGreedyToken MoEGateSkip"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "0"))
	t.Setenv("GO_MLX_TRACE_FORWARD_EVAL", "1")
	requireMetalRuntime(t)

	cfg := testGemma4NativeLayerConfig()
	cfg.NumHiddenLayers = 1
	layer := testGemma4NativeMoELayer()
	model := &Gemma4Model{
		Cfg:               cfg,
		Layers:            []*Gemma4DecoderLayer{layer},
		PreviousKVs:       []int32{0},
		CacheIndexByLayer: []int32{0},
		NormScaled:        FromValues([]float32{1, 1}, 2),
		Output: NewLinear(FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, 3, 2), nil),
	}
	defer closeGemma4(model)

	hidden := FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	cache := NewFixedKVCache(4)
	masks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	defer Free(hidden, perLayer)
	defer cache.Reset()
	defer masks.Free()

	resetNativePhaseTraceEvents()
	got, ok, err := nativeGemma4FixedGreedyToken(hidden, []*Array{perLayer}, []Cache{cache}, model, masks)
	if err != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken() error = %v", err)
	}
	if ok || got != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken() = ok %v token %v, want skip", ok, got)
	}
	events := takeNativePhaseTraceEvents()
	if len(events) != 1 || events[0].Name != "gemma4.model.greedy_token.skip" || events[0].Error != "layer 00: moe native layer is disabled" {
		t.Fatalf("events = %+v, want model Greedy MoE gate skip", events)
	}
}
