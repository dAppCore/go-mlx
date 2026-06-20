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
func encGeluGateMul(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, sc mlpScratch, dFF int) error {
	if gpuHasGeluKernel() { // fused kernel (1 dispatch, fp32-internal) when loaded, composed bf16 chain otherwise
		return encGeluGateMulFused(enc, gate, up, out, dFF)
	}
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
	return nil
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
			// resident (cached by address): the same expert weight is reused every token, so it must
			// NOT be re-uploaded per token (the retained-buffer leak that OOM'd the MoE decode).
			gE := residentBytes(gateW[e*gateSz : (e+1)*gateSz])
			uE := residentBytes(upW[e*gateSz : (e+1)*gateSz])
			dE := residentBytes(downW[e*downSz : (e+1)*downSz])
			if encErr = encGemvBF16(enc, gE, xBuf, msc.gate, dFF, dModel); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encGemvBF16(enc, uE, xBuf, msc.up, dFF, dModel)
			if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
				enc.EndEncoding()
				return
			}
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

// MoEExpertsQuant is MoEExperts for 4-bit experts: the gemma4 26B-A4B SwitchGLU stores all
// experts batched (experts.switch_glu.{gate,up,down}_proj as [numExperts × out × in] affine-
// quant tensors), so gate/up/down are QuantWeights whose Packed/Scales/Biases hold every
// expert's slice. For each of the topK selected experts it runs the SwiGLU via QMVBF16
// (gate/up: dModel→dFF, down: dFF→dModel) and accumulates weights[i]·downᵢ — the quant sibling
// of MoEExperts, encQMVBF16 in place of encGemvBF16. groupSize/bits are the checkpoint's quant.
func MoEExpertsQuant(x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, numExperts, topK, dModel, dFF, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuant: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuant: idx/weights length must equal topK")
	}
	if dModel%groupSize != 0 || dFF%groupSize != 0 {
		return nil, core.NewError("native.MoEExpertsQuant: dModel and dFF must be multiples of groupSize")
	}
	gatePacked, gateScale := dFF*dModel*bits/8, dFF*(dModel/groupSize)*bf16Size // per expert (gate, up)
	downPacked, downScale := dModel*dFF*bits/8, dModel*(dFF/groupSize)*bf16Size // per expert (down)
	if len(gate.Packed) != numExperts*gatePacked || len(up.Packed) != numExperts*gatePacked || len(down.Packed) != numExperts*downPacked ||
		len(gate.Scales) != numExperts*gateScale || len(up.Scales) != numExperts*gateScale || len(down.Scales) != numExperts*downScale ||
		len(gate.Biases) != numExperts*gateScale || len(up.Biases) != numExperts*gateScale || len(down.Biases) != numExperts*downScale {
		return nil, core.NewError("native.MoEExpertsQuant: batched expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExpertsQuant: expert index out of range")
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
		// the e-th expert's slice of a batched [numExperts × …] quant tensor — resident (cached by
		// address): the same expert weight is reused every token, so it must NOT be re-uploaded per
		// token (the retained-buffer leak that OOM'd 26B-A4B). xBuf/scratch/wBuf stay transient.
		slice := func(b []byte, e, sz int) metal.MTLBuffer { return residentBytes(b[e*sz : (e+1)*sz]) }

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for i := 0; i < topK; i++ {
			e := int(idx[i])
			if encErr = encQMVBF16(enc, slice(gate.Packed, e, gatePacked), slice(gate.Scales, e, gateScale), slice(gate.Biases, e, gateScale), xBuf, msc.gate, 0, 0, 0, 0, dFF, dModel, groupSize, bits); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encQMVBF16(enc, slice(up.Packed, e, gatePacked), slice(up.Scales, e, gateScale), slice(up.Biases, e, gateScale), xBuf, msc.up, 0, 0, 0, 0, dFF, dModel, groupSize, bits)
			if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encQMVBF16(enc, slice(down.Packed, e, downPacked), slice(down.Scales, e, downScale), slice(down.Biases, e, downScale), msc.gated, downE, 0, 0, 0, 0, dModel, dFF, groupSize, bits)
			wBuf := sharedBytes(scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel))
			if i == 0 {
				_ = encMulBF16(enc, downE, wBuf, acc, dModel)
			} else {
				_ = encMulBF16(enc, downE, wBuf, scaled, dModel)
				_ = encAddBF16(enc, acc, scaled, acc, dModel)
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(acc.Contents()), len(out)))
	})
	return out, encErr
}
