// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestGeluKernelCapabilityReflectsLoadedFlag(t *testing.T) {
	old := customLibraryLoaded
	defer func() { customLibraryLoaded = old }()

	customLibraryLoaded = false
	if gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel true when custom library flag is false")
	}
	customLibraryLoaded = true
	if !gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel false when custom library flag is true")
	}
}

func TestMulScalarBF16MatchesBroadcastMultiply(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes([]float32{-2, -0.5, 0, 0.25, 1.5, 3})
	scalar := toBF16Bytes([]float32{0.375})
	got, err := MulScalarBF16(in, scalar)
	if err != nil {
		t.Fatalf("MulScalarBF16: %v", err)
	}
	want, err := MulBF16(in, scalarFillBF16(scalar, len(in)/bf16Size))
	if err != nil {
		t.Fatalf("broadcast MulBF16: %v", err)
	}
	eqBytes(t, "MulScalarBF16", got, want)
}
