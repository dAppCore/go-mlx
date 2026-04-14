package mlx

import "testing"

func TestBackendDeviceForGPULayers_Good(t *testing.T) {
	tests := []struct {
		name                   string
		gpuLayers              int
		wantDevice             string
		wantPartialOffloadWarn bool
	}{
		{name: "default", gpuLayers: -1, wantDevice: "gpu"},
		{name: "cpu_only", gpuLayers: 0, wantDevice: "cpu"},
		{name: "partial_gpu_offload", gpuLayers: 12, wantDevice: "gpu", wantPartialOffloadWarn: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDevice, gotWarn := backendDeviceForGPULayers(tt.gpuLayers)
			if gotDevice != tt.wantDevice {
				t.Fatalf("device = %q, want %q", gotDevice, tt.wantDevice)
			}
			if gotWarn != tt.wantPartialOffloadWarn {
				t.Fatalf("partialOffloadUnsupported = %t, want %t", gotWarn, tt.wantPartialOffloadWarn)
			}
		})
	}
}
