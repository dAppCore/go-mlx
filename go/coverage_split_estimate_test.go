// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for the split-executor memory-estimate facade and the native
// split-runtime entry guard — synthetic where possible (no real model), so
// these run under -tags metal_runtime against a constructed CPU-FFN pack.

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestSplitExecutor_CPUSplitFFNMemoryEstimate drives the memory-estimate facade
// across its three arms: a real CPU-FFN executor (the estimator path), a nil
// receiver, and an executor whose FFN does not implement the estimator.
func TestSplitExecutor_CPUSplitFFNMemoryEstimate(t *testing.T) {
	source := writeCPUSplitFFNTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	executor, err := LoadSplitExecutor(context.Background(), slicePath, WithCPUSplitFFNExecutor())
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	// Estimator arm — the CPU FFN executor reports a residency estimate.
	report, err := executor.CPUSplitFFNMemoryEstimate(context.Background())
	if err != nil {
		t.Fatalf("CPUSplitFFNMemoryEstimate: %v", err)
	}
	if report == nil {
		t.Fatal("CPUSplitFFNMemoryEstimate returned nil report on a CPU-FFN executor")
	}
	t.Logf("CPU FFN memory estimate: %+v", *report)

	// Nil-receiver arm — returns (nil, nil) without panicking.
	var nilExecutor *SplitExecutor
	if r, err := nilExecutor.CPUSplitFFNMemoryEstimate(context.Background()); r != nil || err != nil {
		t.Errorf("nil executor estimate = (%v, %v), want (nil, nil)", r, err)
	}
}

// TestSplitExecutor_CPUSplitFFNMemoryEstimate_NonEstimatorFFN drives the
// not-an-estimator arm: an executor whose FFN is a plain stub returns
// (nil, nil) because the FFN does not implement splitFFNMemoryEstimator.
func TestSplitExecutor_CPUSplitFFNMemoryEstimate_NonEstimatorFFN(t *testing.T) {
	source := writeCPUSplitFFNTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice-stub")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	// splitExecutorTestFFN (defined in split_executor_test.go) is a placement
	// stub that does NOT implement the memory estimator.
	executor, err := LoadSplitExecutor(context.Background(), slicePath, WithSplitFFNExecutor(splitExecutorTestFFN{}))
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}
	report, err := executor.CPUSplitFFNMemoryEstimate(context.Background())
	if report != nil || err != nil {
		t.Errorf("non-estimator FFN estimate = (%v, %v), want (nil, nil)", report, err)
	}
}

// TestNativeSplitLocalRuntime_ForwardAttentionNotReady covers the entry guard of
// ForwardAttention: a zero-value runtime (no slice path) is not ready, so the
// method returns the readiness error before any model work. The post-guard
// device forward remains real-model/native-split bound.
func TestNativeSplitLocalRuntime_ForwardAttentionNotReady(t *testing.T) {
	runtime := &NativeSplitLocalRuntime{}
	_, err := runtime.ForwardAttention(context.Background(), SplitAttentionRequest{Layer: 0})
	if err == nil {
		t.Fatal("ForwardAttention on a not-ready runtime should error")
	}
}
