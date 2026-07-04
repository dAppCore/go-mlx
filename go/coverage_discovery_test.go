// SPDX-Licence-Identifier: EUPL-1.2

// Final clean-reachable batch: selectModelSliceTensorRefs ExtractLevelAll arm
// (synthetic safetensors index, no file content), DiscoverLocalRuntime's
// real-model-dir candidate path (via the existing slice test pack, which is a
// valid discoverable safetensors dir — no model is loaded), and the
// safeRuntimeDeviceInfo GPU-probe gate.

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/inference/safetensors"
)

func TestSelectModelSliceTensorRefs_ExtractLevelAll(t *testing.T) {
	index := safetensors.Index{
		Names: []string{"a.weight", "b.weight"},
		Tensors: map[string]safetensors.TensorRef{
			"a.weight": {Name: "a.weight", Elements: 4, ByteLen: 16},
			"b.weight": {Name: "b.weight", Elements: 4, ByteLen: 16},
		},
	}

	// ExtractLevelAll copies every tensor regardless of name.
	refsAll, namesAll := selectModelSliceTensorRefs(&inference.ModelSlicePlan{ExtractLevel: inference.ModelExtractLevelAll}, index)
	if len(refsAll) != 2 || len(namesAll) != 2 {
		t.Fatalf("ExtractLevelAll should select all tensors, got %d refs / %d names", len(refsAll), len(namesAll))
	}

	// A component-masked plan keeps only the matching tensors. "a.weight"
	// has no attention marker, so the attention-only plan selects nothing.
	refsMasked, _ := selectModelSliceTensorRefs(&inference.ModelSlicePlan{
		Components: []inference.ModelComponent{inference.ModelComponentAttention},
	}, index)
	if len(refsMasked) != 0 {
		t.Fatalf("attention-only plan should select no plain-weight tensors, got %d", len(refsMasked))
	}
}

func TestDiscoverLocalRuntime_WithDiscoverableModelDir(t *testing.T) {
	// writeModelSliceTestPack builds a valid safetensors model directory
	// (config.json + tokenizer + model.safetensors). DiscoverLocalRuntime
	// only inspects metadata + plans tuning candidates — it never loads the
	// model for inference — so the IncludeModels + IncludeCandidates path is
	// reachable here.
	modelDir := writeModelSliceTestPack(t)
	parent := parentDirOf(modelDir)

	report, err := DiscoverLocalRuntime(context.Background(), LocalDiscoveryConfig{
		IncludeModels:     true,
		IncludeCandidates: true,
		ModelDirs:         []string{parent},
		MaxModels:         4,
	})
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime: %v", err)
	}
	// At least the one model we wrote should be discovered, and candidate
	// planning should have produced entries for it.
	if len(report.Models) == 0 {
		t.Fatalf("expected at least one discovered model under %s", parent)
	}
	if len(report.Candidates) == 0 {
		t.Fatalf("IncludeCandidates should produce tuning candidates")
	}
}

func TestSafeRuntimeDeviceInfo_ProbeGate(t *testing.T) {
	// Default (gate off) → host-reported device info, matching
	// metal.HostDeviceInfo() exactly (no native MLX device query).
	host := safeRuntimeDeviceInfo()
	if want := metal.HostDeviceInfo(); host != want {
		t.Fatalf("safeRuntimeDeviceInfo() (gate off) = %+v, want host-reported %+v", host, want)
	}

	// Flip the in-code diagnostic gate to exercise the native-probe arm,
	// restoring it afterwards so no global state leaks.
	prev := reportDeviceInfoGate
	reportDeviceInfoGate = true
	t.Cleanup(func() { reportDeviceInfoGate = prev })
	probed := safeRuntimeDeviceInfo()
	if want := GetDeviceInfo(); probed != want {
		t.Fatalf("safeRuntimeDeviceInfo() (gate on) = %+v, want native probe %+v", probed, want)
	}
}

// parentDirOf returns the parent directory of dir using the same path helpers
// the package uses, so DiscoverLocalRuntime scans the directory that contains
// the model rather than the model dir itself.
func parentDirOf(dir string) string {
	for i := len(dir) - 1; i >= 0; i-- {
		if dir[i] == '/' {
			return dir[:i]
		}
	}
	return dir
}
