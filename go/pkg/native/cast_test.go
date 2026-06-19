// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

// TestBF16F32CastRoundtrip verifies the copy_v cast kernel ABI: widening bf16->fp32 is lossless and
// narrowing a value that was already bf16 back to bf16 is exact, so bf16 -> fp32 -> bf16 must be the
// identity. A non-identity result means the cast wrapper's buffer/dispatch ABI is wrong.
func TestBF16F32CastRoundtrip(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	n := 1024
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(i-512) * 0.137 // spread of finite values, both signs
	}
	bf := toBF16Bytes(f) // the bf16 storage values
	var back []byte
	withAutoreleasePool(func() {
		bfBuf := sharedBytes(bf)
		f32 := scratch(n)     // fp32 intermediate
		bf2 := scratchBF16(n) // bf16 output
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		_ = encWidenBF16ToF32(enc, bfBuf, f32, n)
		_ = encNarrowF32ToBF16(enc, f32, bf2, n)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		back = make([]byte, n*bf16Size)
		copy(back, unsafe.Slice((*byte)(bf2.Contents()), n*bf16Size))
	})
	if !bytes.Equal(back, bf) {
		diff := 0
		for i := 0; i+1 < len(bf); i += 2 {
			if bf[i] != back[i] || bf[i+1] != back[i+1] {
				diff++
			}
		}
		t.Errorf("bf16->f32->bf16 not identity: %d/%d elements differ — cast ABI wrong", diff, n)
	}
}
