// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func TestPinnedNoCopyBytesGPURead(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Skipf("custom scalar kernel unavailable: %v", err)
	}
	in := toBF16Bytes([]float32{1, 2, 3, 4})
	scalar := bf16ScalarBytes(2)
	want := toBF16Bytes([]float32{2, 4, 6, 8})
	var got []byte
	withAutoreleasePool(func() {
		if err := withPinnedNoCopyBytes(in, func(inBuf metal.MTLBuffer) error {
			out := scratchBF16(len(in) / bf16Size)
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if err := encMulScalarBF16(enc, inBuf, sharedBytes(scalar[:]), out, 0, len(in)/bf16Size); err != nil {
				enc.EndEncoding()
				return err
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			got = append([]byte(nil), unsafe.Slice((*byte)(out.Contents()), len(in))...)
			return nil
		}); err != nil {
			t.Fatalf("withPinnedNoCopyBytes: %v", err)
		}
	})
	if !bytes.Equal(got, want) {
		t.Fatalf("pinned no-copy GPU read = %v, want %v", got, want)
	}
}

func TestResidentBytesNoCopyReflectsHostMutation(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Skipf("custom scalar kernel unavailable: %v", err)
	}
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	in := toBF16Bytes([]float32{1, 2, 3, 4})
	buf := residentBytes(in)
	copy(in, toBF16Bytes([]float32{5, 6, 7, 8}))

	scalar := bf16ScalarBytes(1)
	want := toBF16Bytes([]float32{5, 6, 7, 8})
	var got []byte
	withAutoreleasePool(func() {
		out := scratchBF16(len(in) / bf16Size)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if err := encMulScalarBF16(enc, buf, sharedBytes(scalar[:]), out, 0, len(in)/bf16Size); err != nil {
			enc.EndEncoding()
			t.Fatalf("encMulScalarBF16: %v", err)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		got = append([]byte(nil), unsafe.Slice((*byte)(out.Contents()), len(in))...)
	})
	if !bytes.Equal(got, want) {
		t.Fatalf("resident no-copy GPU read = %v, want %v", got, want)
	}
}
