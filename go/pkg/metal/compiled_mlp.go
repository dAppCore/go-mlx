// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"runtime/debug"
	"sync"

	"dappco.re/go"
)

// Compiled decode MLP — the gated feed-forward block as ONE mlx_compile'd
// closure. MLX traces the gate/up/GELU/down graph once per quantisation
// config and replays it for every layer of every token, replacing the
// per-call op-by-op graph build + schedule with a cached-graph replay. The
// layer weights enter as closure INPUTS (not captured constants), so a single
// trace serves all layers that share a config.
//
// Decode regime only: callers guard to single-token inputs host-side, so the
// frozen trace shapes are the steady decode shapes. Prefill keeps the
// uncompiled paths. Gated by GateCompiledMLPDecode; any panic from the
// compile/replay path poisons the feature for the process and every later
// call falls through to the uncompiled paths.

type compiledMLPKey struct {
	bits      int
	groupSize int
}

var (
	compiledMLPFns    sync.Map // compiledMLPKey -> *CompiledFunc
	compiledMLPPoison sync.Map // compiledMLPKey -> true (config failed; stay off)
)

// compiledMLPDecodeForward runs the MLP through the compiled closure when the
// gate is on, the input is decode-shaped, and the projections share a
// compile-eligible quantisation config. ok=false means the caller runs its
// normal paths.
func compiledMLPDecodeForward(x *Array, m *MLP) (out *Array, ok bool) {
	if !compiledMLPDecodeRuntimeEnabled() || x == nil || !x.Valid() || m == nil {
		return nil, false
	}
	key, eligible := compiledMLPConfig(m)
	if !eligible {
		return nil, false
	}
	if _, poisoned := compiledMLPPoison.Load(key); poisoned {
		return nil, false
	}
	if !decodeShaped(x) {
		return nil, false
	}

	defer func() {
		if recovered := recover(); recovered != nil {
			core.Error("mlx: compiled decode MLP failed; falling back to uncompiled paths",
				"error", recovered, "stack", string(debug.Stack()))
			compiledMLPPoison.Store(key, true)
			out, ok = nil, false
		}
	}()

	fn := compiledMLPFn(key)
	outs := fn.Call(x,
		m.GateProj.Weight, m.GateProj.Scales, m.GateProj.Biases,
		m.UpProj.Weight, m.UpProj.Scales, m.UpProj.Biases,
		m.DownProj.Weight, m.DownProj.Scales, m.DownProj.Biases,
	)
	if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
		Free(outs...)
		compiledMLPPoison.Store(key, true)
		return nil, false
	}
	return outs[0], true
}

// compiledMLPConfig reports the shared quantisation config of the three
// projections and whether the block is compile-eligible: affine quantisation
// on the gemm-preferring path, no LoRA, no bias, uniform bits + group size.
func compiledMLPConfig(m *MLP) (compiledMLPKey, bool) {
	var key compiledMLPKey
	for _, linear := range []*Linear{m.GateProj, m.UpProj, m.DownProj} {
		if linear == nil || linear.LoRA != nil {
			return key, false
		}
		if linear.Bias != nil && linear.Bias.Valid() {
			return key, false
		}
		if linear.Weight == nil || !linear.Weight.Valid() || linear.Scales == nil || !linear.Scales.Valid() || linear.Biases == nil || !linear.Biases.Valid() {
			return key, false
		}
		if !IsAffineQuantizationMode(linear.QuantizationMode) || !AffineQuantPrefersGemm(linear) {
			return key, false
		}
		if key.bits == 0 {
			key.bits = linear.Bits
			key.groupSize = linear.GroupSize
			continue
		}
		if linear.Bits != key.bits || linear.GroupSize != key.groupSize {
			return key, false
		}
	}
	return key, key.bits != 0
}

// decodeShaped reports whether x is a single-token decode activation — the
// regime the compiled trace is shaped for.
func decodeShaped(x *Array) bool {
	var shapeBuf [MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	if len(shape) != 3 {
		return false
	}
	return shape[0] == 1 && shape[1] == 1
}

// compiledMLPFn returns (building on first use) the compiled closure for a
// quantisation config. Inputs: [x, gateW, gateScales, gateBiases, upW, upS,
// upB, downW, downS, downB] -> [down(gelu(gate(x)) * up(x))].
//
// The body prefers the fused native kernels — the same custom Metal kernels
// the uncompiled fast path dispatches; they are MLX fast-kernel primitives,
// so the trace replays them with the per-call graph build and schedule
// removed. Configs the fused kernels decline trace the gemm graph instead.
func compiledMLPFn(key compiledMLPKey) *CompiledFunc {
	if cached, found := compiledMLPFns.Load(key); found {
		return cached.(*CompiledFunc)
	}
	fn := CompileShapeless(func(in []*Array) []*Array {
		x := in[0]
		gate := &Linear{Weight: in[1], Scales: in[2], Biases: in[3], QuantizationMode: "affine", GroupSize: key.groupSize, Bits: key.bits}
		up := &Linear{Weight: in[4], Scales: in[5], Biases: in[6], QuantizationMode: "affine", GroupSize: key.groupSize, Bits: key.bits}
		down := &Linear{Weight: in[7], Scales: in[8], Biases: in[9], QuantizationMode: "affine", GroupSize: key.groupSize, Bits: key.bits}
		if activated, ok, err := quantizedDenseGELUSplitGateUpMatVec(x, gate, up); ok && err == nil {
			if out, okDown, errDown := QuantizedDenseMatVec(activated, down); okDown && errDown == nil {
				Free(activated)
				return []*Array{out}
			}
			Free(activated)
		}
		gateOut := quantizedMatmulMode(x, in[1], in[2], in[3], true, key.groupSize, key.bits, "affine")
		upOut := quantizedMatmulMode(x, in[4], in[5], in[6], true, key.groupSize, key.bits, "affine")
		activated := GeluGateMul(gateOut, upOut)
		Free(gateOut, upOut)
		out := quantizedMatmulMode(activated, in[7], in[8], in[9], true, key.groupSize, key.bits, "affine")
		Free(activated)
		return []*Array{out}
	}, true)
	cached, _ := compiledMLPFns.LoadOrStore(key, fn)
	return cached.(*CompiledFunc)
}
