// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestMatVecBF16BufMatchesHost isolates MatVecBF16Buf from the shard/offset/decode machinery: it binds a
// FRESH buffer holding identical bytes at offset 0 — the exact thing MatVecBF16 does internally — so the two
// must be byte-for-byte equal. If this fails, MatVecBF16Buf's own dispatch differs from MatVecBF16; if it
// passes, any decode divergence is the shard binding (offset/alignment), not the kernel call.
func TestMatVecBF16BufMatchesHost(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	outDim, inDim := 96, 256
	mat := make([]byte, outDim*inDim*bf16Size)
	vec := make([]byte, inDim*bf16Size)
	for i := range mat {
		mat[i] = byte((i*131 + 17) & 0x3f) // small deterministic bf16 bit patterns (no NaN/Inf)
	}
	for i := range vec {
		vec[i] = byte((i*97 + 5) & 0x3f)
	}
	host, err := MatVecBF16(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16: %v", err)
	}
	var res []byte
	withAutoreleasePool(func() {
		b := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&mat[0]), uint(len(mat)), metal.MTLResourceStorageModeShared)
		res, err = MatVecBF16Buf(bufView{buf: b, off: 0}, vec, outDim, inDim)
	})
	if err != nil {
		t.Fatalf("MatVecBF16Buf: %v", err)
	}
	for i := range host {
		if host[i] != res[i] {
			t.Fatalf("byte %d: host %d != resident %d — MatVecBF16Buf differs from MatVecBF16 with an identical buffer at offset 0", i, host[i], res[i])
		}
	}
	t.Logf("✓ MatVecBF16Buf == MatVecBF16 (%d out bytes, fresh buffer off 0)", len(host))
}
