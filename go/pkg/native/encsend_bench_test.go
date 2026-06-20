// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"testing"
	"unsafe"

	basepurego "github.com/ebitengine/purego"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// Typed, non-variadic objc_msgSend stubs registered ONCE — the zero-alloc ceiling
// the setPSO/setBuf seam could reach. This is the exact pattern tmc/apple uses for
// its own (unexported) msgSend0..8, replicated here only to MEASURE the ceiling:
// no []any, no per-arg boxing, no make([]uintptr) — so escape analysis can keep
// every argument on the stack. If this drops to ~0 alloc/send, the residual ~10
// alloc/send the current fix still pays is the objc.Send variadic boxing, and the
// real zero-alloc win is to move setPSO/setBuf onto these stubs (an FFI-init step,
// flagged for Snider — NOT done in production here).
var (
	stubMsgSend1 func(id, sel, a1 uintptr) uintptr
	stubMsgSend3 func(id, sel, a1, a2, a3 uintptr) uintptr
	stubOnce     sync.Once
)

func initMsgSendStubs() {
	h, err := basepurego.Dlopen("/usr/lib/libobjc.A.dylib", basepurego.RTLD_LAZY|basepurego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	addr, err := basepurego.Dlsym(h, "objc_msgSend")
	if err != nil {
		return
	}
	basepurego.RegisterFunc(&stubMsgSend1, addr)
	basepurego.RegisterFunc(&stubMsgSend3, addr)
}

func setPSOStub(enc metal.MTLComputeCommandEncoder, pso metal.MTLComputePipelineState) {
	stubMsgSend1(uintptr(enc.GetID()), uintptr(selSetComputePipelineState), uintptr(pso.GetID()))
}

func setBufStub(enc metal.MTLComputeCommandEncoder, buf metal.MTLBuffer, off, idx uint) {
	stubMsgSend3(uintptr(enc.GetID()), uintptr(selSetBufferOffsetAtIndex), uintptr(buf.GetID()), uintptr(off), uintptr(idx))
}

var _ = objc.Sel // keep objc import referenced if selectors move

// AX-11 encode-only micro-benches isolating the per-token Metal-send allocation
// cost the no-cgo decode path pays on every encoder setup. No model load: a
// fresh command buffer + compute encoder, the three SetBuffer + one SetPSO calls
// that every kernel dispatch makes, then EndEncoding (no Commit — we measure the
// host-side encode, not GPU work). The two benches differ only in HOW the buffer
// and pipeline bindings are issued:
//
//   - …WrapperSend uses tmc/apple's generated interface wrappers
//     (enc.SetComputePipelineState / enc.SetBufferWithOffsetAtIndex), which box
//     the MTLComputePipelineState / MTLBuffer interfaces into objc.Send's slow
//     path → purego.RegisterFunc → reflect.MakeFunc per call.
//   - …FastSend uses setPSO / setBuf (encsend.go), which extract the raw objc.ID
//     and hit tmc/apple's pre-registered msgSendN fast path — no reflect.
//
// allocs/op is the figure of merit: the delta is the per-encode reflect-trampoline
// cost removed, multiplied across every binding of every kernel of every token.
func benchEncodeSetup(b *testing.B, fast bool) {
	requireNativeRuntime(b)
	pso, err := pipelineFor("rmsfloat32")
	if err != nil {
		b.Fatalf("pipelineFor: %v", err)
	}
	const n = 1024
	x := syntheticFloat32(n, 3)
	w := syntheticFloat32(n, 5)
	var xBuf, wBuf, outBuf metal.MTLBuffer
	withAutoreleasePool(func() {
		xBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		wBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&w[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		outBuf = device.NewBufferWithLengthOptions(uint(n*4), metal.MTLResourceStorageModeShared)
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		withAutoreleasePool(func() {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if fast {
				setPSO(enc, pso)
				setBuf(enc, xBuf, 0, 0)
				setBuf(enc, wBuf, 0, 1)
				setBuf(enc, outBuf, 0, 2)
			} else {
				enc.SetComputePipelineState(pso)
				enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
				enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
				enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
			}
			enc.EndEncoding()
		})
	}
}

// BenchmarkEncodeSetupWrapperSend is the baseline: interface-wrapper sends (slow path).
func BenchmarkEncodeSetupWrapperSend(b *testing.B) { benchEncodeSetup(b, false) }

// BenchmarkEncodeSetupFastSend is the fast path: raw-ID sends via setPSO/setBuf.
func BenchmarkEncodeSetupFastSend(b *testing.B) { benchEncodeSetup(b, true) }

// benchBindOnly isolates the per-send allocation cost: a single reused encoder,
// looping ONLY the four bindings (1 PSO + 3 buffers) per op. Encoder/command-buffer
// creation is hoisted out, so allocs/op is purely the send cost (÷4 = per-send).
// This is the figure that multiplies across every kernel of every decoded token.
type sendMode int

const (
	sendWrapper sendMode = iota // tmc/apple interface wrappers (slow path, reflect trampoline)
	sendFast                    // setPSO/setBuf: objc.Send fast path (no trampoline, still []any boxing)
	sendStub                    // typed non-variadic msgSend stub: the zero-alloc ceiling
)

func benchBindOnly(b *testing.B, mode sendMode) {
	requireNativeRuntime(b)
	if mode == sendStub {
		stubOnce.Do(initMsgSendStubs)
		if stubMsgSend3 == nil {
			b.Skip("objc_msgSend stub unavailable")
		}
	}
	pso, err := pipelineFor("rmsfloat32")
	if err != nil {
		b.Fatalf("pipelineFor: %v", err)
	}
	const n = 1024
	x := syntheticFloat32(n, 3)
	w := syntheticFloat32(n, 5)
	withAutoreleasePool(func() {
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&w[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(n*4), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			switch mode {
			case sendWrapper:
				enc.SetComputePipelineState(pso)
				enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
				enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
				enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
			case sendFast:
				setPSO(enc, pso)
				setBuf(enc, xBuf, 0, 0)
				setBuf(enc, wBuf, 0, 1)
				setBuf(enc, outBuf, 0, 2)
			case sendStub:
				setPSOStub(enc, pso)
				setBufStub(enc, xBuf, 0, 0)
				setBufStub(enc, wBuf, 0, 1)
				setBufStub(enc, outBuf, 0, 2)
			}
		}
		b.StopTimer()
		enc.EndEncoding()
	})
}

// BenchmarkBindOnly* isolate the 4-binding send cost (÷4 = per-send):
//   - WrapperSend: baseline (interface wrappers → reflect trampoline per call)
//   - FastSend:    the shipped seam (setPSO/setBuf → objc.Send fast path)
//   - TypedStub:   the zero-alloc ceiling (typed non-variadic stub; not in production)
func BenchmarkBindOnlyWrapperSend(b *testing.B) { benchBindOnly(b, sendWrapper) }
func BenchmarkBindOnlyFastSend(b *testing.B)    { benchBindOnly(b, sendFast) }
func BenchmarkBindOnlyTypedStub(b *testing.B)   { benchBindOnly(b, sendStub) }
