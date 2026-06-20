// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// Fast-path encoder setters for the two hottest per-token Metal sends.
//
// WHY: tmc/apple's objc.Send has a zero-allocation fast path (msgSend0..8,
// pre-registered once at init) that triggers only when the return type is
// struct{}/uintptr AND every argument is uintptr-castable (ID, SEL, uint, …).
// Its generated MTLComputeCommandEncoder wrappers, however, pass the *interface*
// values MTLComputePipelineState and MTLBuffer straight through to Send. An
// interface value is not one of the tryFastArgs cases, so Send takes the slow
// path: purego/objc.Send, which re-declares a variadic func and calls
// purego.RegisterFunc (reflect.MakeFunc) on EVERY call. That reflect trampoline
// is the dominant per-token heap allocator on the no-cgo decode path (AX-11).
//
// These helpers do exactly what the slow path does — extract the raw objc.ID
// from the interface (the same GetID() the wrapper's slow path reaches) and
// issue the same objc_msgSend with the same arguments — but pass IDs and uints,
// so tryFastArgs matches and the pre-registered typed stub is used instead of a
// freshly-reflected trampoline. Byte-identical by construction: same selector,
// same receiver, same argument bits; only the dispatch mechanism differs.
//
// Scope is deliberately the two interface-arg selectors that dominate encoder
// setup. setBytes:length:atIndex: ([]byte slice arg) and the dispatch*/MTLSize
// struct-arg selectors keep the standard wrappers — their arg shapes cannot use
// this fast path without separate, riskier handling.

var (
	selSetComputePipelineState = objc.Sel("setComputePipelineState:")
	selSetBufferOffsetAtIndex  = objc.Sel("setBuffer:offset:atIndex:")
)

// setPSO binds a compute pipeline state on enc via the zero-alloc fast send.
// Equivalent to enc.SetComputePipelineState(pso).
func setPSO(enc metal.MTLComputeCommandEncoder, pso metal.MTLComputePipelineState) {
	objc.Send[struct{}](enc.GetID(), selSetComputePipelineState, pso.GetID())
}

// setBuf binds buf at (offset, index) on enc via the zero-alloc fast send.
// Equivalent to enc.SetBufferWithOffsetAtIndex(buf, offset, index).
func setBuf(enc metal.MTLComputeCommandEncoder, buf metal.MTLBuffer, offset, index uint) {
	objc.Send[struct{}](enc.GetID(), selSetBufferOffsetAtIndex, buf.GetID(), offset, index)
}
