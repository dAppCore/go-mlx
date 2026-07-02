// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64)

package safetensors

import sharedsafetensors "dappco.re/go/inference/safetensors"

// float16SliceToFloat32 converts n half-precision values from src into the
// first n elements of dst using the scalar dappco.re/go/inference/safetensors
// Float16ToFloat32 path (this package's own Float16ToFloat32 is a thin
// wrapper over the same function — called directly here to skip that
// indirection). Used on every non-(darwin && arm64) build. The NEON FCVTL
// path in float16_neon_darwin_arm64.go produces bit-identical output — see
// TestFloat16ToFloat32_NEONParity_BitExact for the cross-architecture
// invariant.
func float16SliceToFloat32(src []uint16, dst []float32, n int) {
	for i := 0; i < n; i++ {
		dst[i] = sharedsafetensors.Float16ToFloat32(src[i])
	}
}
