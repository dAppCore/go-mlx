// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/metal"
)

// encAttnHalfShared is the KV-SHARING attention half: a layer that shares another
// layer's KV cache projects ONLY its query (from its own input) and attends over
// the owner's cache — no K/V projection, no K-RoPE, no cache write. attendK/attendV
// are the owner's seq-major caches; the window N=pos+1 is the owner's live length
// (the owner wrote row pos earlier this token). Writes x + Wo·attn -> h.
func encAttnHalfShared(
	enc metal.MTLComputeCommandEncoder,
	x, attnNormW, attendK, attendV, offBuf, h, postAttnNorm, qNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW int, base, scale, eps float32,
) error {
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16(enc, x, attnNormW, sc.normed, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if qNorm != nil { // gemma4 per-head QK-norm before RoPE (sharers project only Q)
		if err := encRMSNormRowsBF16(enc, sc.q, qNorm, sc.q, nHeads, headDim, eps); err != nil {
			return err
		}
	}
	if err := encRoPEBF16(enc, sc.q, sc.qr, offBuf, nHeads, headDim, base, scale); err != nil {
		return err
	}
	// attend the OWNER's cache, windowed (global: all; sliding: last slideW), no write
	start, n := slideWindow(pos, slideW)
	if err := encSDPAStrided(enc, sc.qr, attendK, attendV, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, uint(start*kvDim*bf16Size)); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}

// archLayerBufs holds one layer's resident buffers for runArchDecode: bf16 norms +
// the (bf16 or 4-bit qmv) projector + the growing KV caches. kCache/vCache are nil for
// sharer layers (they attend the owner's); mnw and the projector's MLP weights are
// unbound for MoE layers (MoEBlockBF16 owns that FFN).
type archLayerBufs struct {
	anw, mnw                 metal.MTLBuffer
	postAttnNorm, postFFNorm metal.MTLBuffer // gemma4 post-attn/post-FF norms (nil = skip)
	qNorm, kNorm             metal.MTLBuffer // gemma4 per-head QK-norm (nil = skip)
	kCache, vCache           metal.MTLBuffer
	proj                     projector
}

// archDecodeState holds the resident buffers of an arch decode — the per-layer weights/
// caches (lb), shared scratch, and the position buffer — so a single token can be stepped
// repeatedly over a PERSISTENT, growing KV cache. Both the whole-sequence runArchDecode and
// the incremental generation loop build one (inside a withAutoreleasePool) and call
// stepToken per token; the caches in lb persist across calls within that pool, which is
// what turns the O(N²) re-decode into O(1)/token incremental decode.
type archDecodeState struct {
	specs        []g4.LayerSpec
	lb           []archLayerBufs
	moeWeights   []*MoELayerWeights
	asc          attnScratch
	msc          mlpScratch
	hBuf, xA, xB metal.MTLBuffer
	offBuf       metal.MTLBuffer

	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int
	base, scale, eps                                      float32
}

// newArchDecodeState builds the shared scratch + position buffer over the caller's
// per-layer buffers. MUST be called inside a withAutoreleasePool.
func newArchDecodeState(specs []g4.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int, base, scale, eps float32) archDecodeState {
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	off := int32(0)
	return archDecodeState{
		specs: specs, lb: lb, moeWeights: moeWeights,
		asc: newAttnScratch(dModel, qDim, kvDim), msc: newMLPScratch(dModel, dFF),
		hBuf: scratchBF16(dModel), xA: scratchBF16(dModel), xB: scratchBF16(dModel),
		offBuf:        device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared),
		dModel:        dModel,
		nHeads:        nHeads,
		nKVHeads:      nKVHeads,
		headDim:       headDim,
		dFF:           dFF,
		slidingWindow: slidingWindow,
		base:          base, scale: scale, eps: eps,
	}
}

// stepToken decodes ONE token (its embedding) at sequence position pos, writing this
// token's K/V into the growing cache, and returns its output hidden state. The projector
// seam keeps it weight-representation-agnostic (bf16 / 4-bit qmv); it honours owner/sharer
// KV-sharing, sliding-window, the gemma4 norms, and MoE (the mid-token command-buffer flush
// because the router does host top-k). The caches persist across calls, so successive
// positions extend the same sequence. MUST be called inside a withAutoreleasePool.
func (s *archDecodeState) stepToken(inputEmb []byte, pos int) ([]byte, error) {
	*(*int32)(s.offBuf.Contents()) = int32(pos)
	copy(unsafe.Slice((*byte)(s.xA.Contents()), s.dModel*bf16Size), inputEmb)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	in, out := s.xA, s.xB
	for li := 0; li < len(s.specs); li++ {
		slideW := 0
		if s.specs[li].Attention == g4.SlidingAttention {
			slideW = s.slidingWindow
		}
		if s.specs[li].OwnsCache() {
			if err := encAttnHalfKV(enc, in, s.lb[li].anw, s.lb[li].kCache, s.lb[li].vCache, s.offBuf, s.hBuf, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, s.nKVHeads, s.headDim, pos, slideW, s.base, s.scale, s.eps); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		} else {
			own := s.specs[li].KVShareFrom
			if err := encAttnHalfShared(enc, in, s.lb[li].anw, s.lb[own].kCache, s.lb[own].vCache, s.offBuf, s.hBuf, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, s.nKVHeads, s.headDim, pos, slideW, s.base, s.scale, s.eps); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		}
		if moeW := s.moeWeights[li]; moeW != nil {
			// the MoE FFN needs h on the host (the router does host top-k): flush the
			// attention half, run the dual-branch block host-side, resume a fresh encoder.
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			hostH := make([]byte, s.dModel*bf16Size)
			copy(hostH, unsafe.Slice((*byte)(s.hBuf.Contents()), s.dModel*bf16Size))
			res, err := MoEBlockBF16(hostH, *moeW, s.dModel, s.dFF, s.eps)
			if err != nil {
				return nil, err
			}
			copy(unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size), res)
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
		} else if err := encMLPHalfBF16(enc, s.hBuf, s.lb[li].mnw, out, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, s.dFF, s.eps); err != nil {
			enc.EndEncoding()
			return nil, err
		}
		in, out = out, in
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	res := make([]byte, s.dModel*bf16Size)
	copy(res, unsafe.Slice((*byte)(in.Contents()), s.dModel*bf16Size))
	return res, nil
}

// runArchDecode is the whole-sequence arch decode: it builds a state and steps each input
// token at its position over a fresh growing cache. See archDecodeState/stepToken — the
// bf16 (DecodeForwardArch) and 4-bit qmv (DecodeForwardArchQuant) forwards share this. MUST
// be called inside a withAutoreleasePool.
func runArchDecode(
	inputs [][]byte, specs []g4.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights,
	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int, base, scale, eps float32,
) ([][]byte, error) {
	s := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, base, scale, eps)
	outputs := make([][]byte, len(inputs))
	for t := range inputs {
		out, err := s.stepToken(inputs[t], t)
		if err != nil {
			return nil, err
		}
		outputs[t] = out
	}
	return outputs, nil
}

// DecodeForwardArch is the bf16 arch-driven decode forward: it runs a decode DRIVEN by
// a declared gemma4 arch (specs, one LayerSpec per layer) rather than treating every
// layer uniformly. It honours the full cache-topology (owner/sharer KV), the per-layer
// attention type (sliding window), and MoE layers (the dual-branch MoEBlockBF16). With
// an all-owner, all-global, dense arch it equals DecodeForward byte-for-byte (gated).
// bf16 re-encode path (one commit+wait/token; MoE layers flush mid-token). The 4-bit
// variant DecodeForwardArchQuant shares the loop (runArchDecode) via the projector seam.
func DecodeForwardArch(
	inputs [][]byte, layers []DecodeLayerWeights, specs []g4.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArch: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArch: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArch: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArch: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArch: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE != (layers[li].MoE != nil) {
			return nil, core.NewError("native.DecodeForwardArch: spec.MoE must match the presence of layer MoE weights")
		}
	}

	var outputs [][]byte
	var err error
	withAutoreleasePool(func() {
		lb, moeWeights := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen)
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, base, scale, eps)
	})
	return outputs, err
}

// buildBF16ArchLayerBufs builds the per-layer resident buffers for a bf16 arch decode:
// bf16 norms + the bf16 projector + the growing KV caches (owner layers only), and the
// per-layer MoE weights (moeWeights[li] != nil ⟺ a MoE layer, whose dense MLP norm +
// gate/up/down stay unbound — MoEBlockBF16 owns that FFN). Shared by the whole-sequence
// forward and the incremental generation loop. MUST be called inside a withAutoreleasePool.
func buildBF16ArchLayerBufs(layers []DecodeLayerWeights, specs []g4.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen int) ([]archLayerBufs, []*MoELayerWeights) {
	nLayers := len(layers)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	lb := make([]archLayerBufs, nLayers)
	moeWeights := make([]*MoELayerWeights, nLayers)
	cacheBytes := uint(maxLen * kvDim * bf16Size)
	for li := range layers {
		w := layers[li]
		lb[li].anw = sharedBytes(w.AttnNormW)
		lb[li].postAttnNorm = sharedOrNil(w.PostAttnNormW)
		lb[li].postFFNorm = sharedOrNil(w.PostFFNormW)
		lb[li].qNorm = sharedOrNil(w.QNormW)
		lb[li].kNorm = sharedOrNil(w.KNormW)
		if specs[li].OwnsCache() {
			lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		}
		p := bf16Projector{
			wQ: sharedBytes(w.WQ), wK: sharedBytes(w.WK), wV: sharedBytes(w.WV), wO: sharedBytes(w.WO),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		if layers[li].MoE == nil {
			lb[li].mnw = sharedBytes(w.MLPNormW)
			p.wGate = sharedBytes(w.WGate)
			p.wUp = sharedBytes(w.WUp)
			p.wDown = sharedBytes(w.WDown)
		} else {
			moeWeights[li] = layers[li].MoE
		}
		lb[li].proj = p
	}
	return lb, moeWeights
}
