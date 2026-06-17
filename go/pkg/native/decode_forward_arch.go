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
	x, attnNormW, attendK, attendV, offBuf, h, postAttnNorm metal.MTLBuffer,
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
	kCache, vCache           metal.MTLBuffer
	proj                     projector
}

// runArchDecode runs the arch-driven per-token decode loop over pre-built per-layer
// buffers. The projector seam makes it weight-representation-agnostic: the bf16
// (DecodeForwardArch) and 4-bit qmv (DecodeForwardArchQuant) forwards share this exact
// loop, differing only in the projector each archLayerBufs carries. It honours the
// cache-topology (owner projects+writes+attends its own cache; sharer attends the
// owner's, Q-only), the per-layer attention type (sliding window), and MoE (the
// dual-branch MoEBlockBF16, which flushes the command buffer mid-token because its
// router does host top-k). moeWeights[li] != nil ⟺ layer li is MoE. MUST be called
// inside a withAutoreleasePool (it allocates scratch + per-token command buffers).
func runArchDecode(
	inputs [][]byte, specs []g4.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights,
	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int, base, scale, eps float32,
) ([][]byte, error) {
	nLayers, T := len(lb), len(inputs)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}

	asc := newAttnScratch(dModel, qDim, kvDim)
	msc := newMLPScratch(dModel, dFF)
	hBuf := scratchBF16(dModel)
	xA, xB := scratchBF16(dModel), scratchBF16(dModel)
	off := int32(0)
	offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)

	for t := 0; t < T; t++ {
		*(*int32)(offBuf.Contents()) = int32(t)
		copy(unsafe.Slice((*byte)(xA.Contents()), dModel*bf16Size), inputs[t])
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		in, out := xA, xB
		for li := 0; li < nLayers; li++ {
			slideW := 0
			if specs[li].Attention == g4.SlidingAttention {
				slideW = slidingWindow
			}
			if specs[li].OwnsCache() {
				if err := encAttnHalfKV(enc, in, lb[li].anw, lb[li].kCache, lb[li].vCache, offBuf, hBuf, lb[li].postAttnNorm, asc, lb[li].proj, dModel, nHeads, nKVHeads, headDim, t, slideW, base, scale, eps); err != nil {
					enc.EndEncoding()
					return nil, err
				}
			} else {
				own := specs[li].KVShareFrom
				if err := encAttnHalfShared(enc, in, lb[li].anw, lb[own].kCache, lb[own].vCache, offBuf, hBuf, lb[li].postAttnNorm, asc, lb[li].proj, dModel, nHeads, nKVHeads, headDim, t, slideW, base, scale, eps); err != nil {
					enc.EndEncoding()
					return nil, err
				}
			}
			if moeW := moeWeights[li]; moeW != nil {
				// the MoE FFN needs h on the host (the router does host top-k), so
				// flush the attention half — committing this token's prior layers and
				// the cache writes — run the dual-branch block host-side, write its
				// result into out, then resume a fresh encoder for the rest.
				enc.EndEncoding()
				cb.Commit()
				cb.WaitUntilCompleted()
				hostH := make([]byte, dModel*bf16Size)
				copy(hostH, unsafe.Slice((*byte)(hBuf.Contents()), dModel*bf16Size))
				res, err := MoEBlockBF16(hostH, *moeW, dModel, dFF, eps)
				if err != nil {
					return nil, err
				}
				copy(unsafe.Slice((*byte)(out.Contents()), dModel*bf16Size), res)
				cb = queue.CommandBuffer()
				enc = cb.ComputeCommandEncoder()
			} else if err := encMLPHalfBF16(enc, hBuf, lb[li].mnw, out, lb[li].postFFNorm, msc, lb[li].proj, dModel, dFF, eps); err != nil {
				enc.EndEncoding()
				return nil, err
			}
			in, out = out, in
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(outputs[t], unsafe.Slice((*byte)(in.Contents()), dModel*bf16Size))
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
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
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
		// per-layer resident buffers: bf16 norms + the bf16 projector + caches (owners
		// only). A MoE layer carries its weights in moeWeights; its dense MLP norm +
		// gate/up/down stay unbound (MoEBlockBF16 owns that FFN, and they may be nil —
		// sharedBytes must not be handed a nil slice).
		lb := make([]archLayerBufs, nLayers)
		moeWeights := make([]*MoELayerWeights, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		for li := range layers {
			w := layers[li]
			lb[li].anw = sharedBytes(w.AttnNormW)
			lb[li].postAttnNorm = sharedOrNil(w.PostAttnNormW)
			lb[li].postFFNorm = sharedOrNil(w.PostFFNormW)
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
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, base, scale, eps)
	})
	return outputs, err
}
