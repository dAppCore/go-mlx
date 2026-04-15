//go:build darwin && arm64 && !nomlx

package mlx

import (
	"testing"

	"dappco.re/go/core/inference"
	"dappco.re/go/mlx/internal/metal"
)

func TestMetalBackendLoadModel_ForwardsCPUDeviceWhenGPULayersZero_Good(t *testing.T) {
	original := loadBackendModel
	t.Cleanup(func() { loadBackendModel = original })

	var got metal.LoadConfig
	loadBackendModel = func(_ string, cfg metal.LoadConfig) (*metal.Model, error) {
		got = cfg
		return &metal.Model{}, nil
	}

	backend := &metalBackend{}
	if _, err := backend.LoadModel("/tmp/model", inference.WithGPULayers(0)); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got.Device != metal.DeviceCPU {
		t.Fatalf("device = %q, want %q", got.Device, metal.DeviceCPU)
	}
}
