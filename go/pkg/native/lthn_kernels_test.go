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
