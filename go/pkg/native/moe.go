// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// scalarFillBF16 returns an n-element bf16 buffer with every element set to the
// single bf16 value in val (2 bytes) — used to broadcast a router weight across a
// column for the weighted expert combine.
func scalarFillBF16(val []byte, n int) []byte {
	out := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		out[i*bf16Size] = val[0]
		out[i*bf16Size+1] = val[1]
	}
	return out
}

// encGeluGateMul encodes the tanh-approx SwiGLU activation gelu(gate)·up into enc —
// the same inline chain as encMLPHalfBF16, factored so the MoE experts reuse it.
// Reads gate/up, writes out; sc supplies the gelu scratch + constant buffers.
func encGeluGateMul(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, sc mlpScratch, dFF int) {
	_ = encMulBF16(enc, gate, gate, sc.x2, dFF)
	_ = encMulBF16(enc, sc.x2, gate, sc.x3, dFF)
	_ = encMulBF16(enc, sc.x3, sc.c044, sc.x3s, dFF)
	_ = encAddBF16(enc, gate, sc.x3s, sc.inner, dFF)
	_ = encMulBF16(enc, sc.inner, sc.c079, sc.scaled, dFF)
	_ = encTanhBF16(enc, sc.scaled, sc.tnh, dFF)
	_ = encAddBF16(enc, sc.tnh, sc.c1, sc.onePlus, dFF)
	_ = encMulBF16(enc, gate, sc.c05, sc.halfG, dFF)
	_ = encMulBF16(enc, sc.halfG, sc.onePlus, sc.gelu, dFF)
	_ = encMulBF16(enc, sc.gelu, up, out, dFF)
}

// MoEExperts runs the expert branch of a gemma4 MoE layer: for each of the topK
// selected experts (idx) it runs that expert's SwiGLU MLP on x and accumulates the
// router-weighted result —  out = Σ_i weights[i] · Wdown_e( gelu(Wgate_e·x)·(Wup_e·x) ).
// Given the routing decision (idx, weights from the router); the routing itself is a
// separate sub-slice. Correctness-first: one gemv per selected expert from its
// weight slice (no gather kernel; big-model bind-by-offset is a later optimisation).
// gateW/upW are [numExperts × dFF × dModel] row-major bf16, downW is
// [numExperts × dModel × dFF]; x is dModel bf16, idx topK int32, weights topK bf16.
// Byte-for-byte against a composed reference of the parity-proven ops in the tests.
func MoEExperts(x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	gateSz, downSz := dFF*dModel*bf16Size, dModel*dFF*bf16Size
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExperts: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExperts: idx/weights length must equal topK")
	}
	if len(gateW) != numExperts*gateSz || len(upW) != numExperts*gateSz || len(downW) != numExperts*downSz {
		return nil, core.NewError("native.MoEExperts: expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExperts: expert index out of range")
		}
	}
	if topK == 0 {
		return make([]byte, dModel*bf16Size), nil
	}

	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		msc := newMLPScratch(dModel, dFF)
		downE, scaled, acc := scratchBF16(dModel), scratchBF16(dModel), scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for i := 0; i < topK; i++ {
			e := int(idx[i])
			gE := sharedBytes(gateW[e*gateSz : (e+1)*gateSz])
			uE := sharedBytes(upW[e*gateSz : (e+1)*gateSz])
			dE := sharedBytes(downW[e*downSz : (e+1)*downSz])
			if encErr = encGemvBF16(enc, gE, xBuf, msc.gate, dFF, dModel); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encGemvBF16(enc, uE, xBuf, msc.up, dFF, dModel)
			encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF)
			_ = encGemvBF16(enc, dE, msc.gated, downE, dModel, dFF)
			wBuf := sharedBytes(scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel))
			if i == 0 {
				_ = encMulBF16(enc, downE, wBuf, acc, dModel) // acc = w0·down0
			} else {
				_ = encMulBF16(enc, downE, wBuf, scaled, dModel)
				_ = encAddBF16(enc, acc, scaled, acc, dModel) // acc += wi·downi
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(acc.Contents()), len(out)))
	})
	return out, encErr
}
