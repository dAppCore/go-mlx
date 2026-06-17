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
	x, attnNormW, attendK, attendV, offBuf, h metal.MTLBuffer,
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
	return encAddBF16(enc, x, sc.attnOut, h, dModel)
}

// DecodeForwardArch is the first executor slice: a decode forward DRIVEN by a
// declared gemma4 arch (specs, one LayerSpec per layer) instead of treating every
// layer uniformly. It honours the cache-topology — owner layers (spec.OwnsCache)
// project Q/K/V, write+attend their own growing seq-major cache; sharer layers
// (spec.KVShareFrom != self) skip K/V projection entirely and attend the owner's
// cache. With an all-owner arch it equals DecodeForward byte-for-byte (gated). bf16
// re-encode path (one commit+wait/token); the ICB/quant arch-forwards follow.
// Sliding-window and MoE are later slices; this honours attention type only insofar
// as it routes KV-sharing (the cache-topology). All raw bf16.
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
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	var encErr error
	withAutoreleasePool(func() {
		// per-layer weights + projectors; caches only for OWNER layers
		type lb struct {
			anw, mnw       metal.MTLBuffer
			kCache, vCache metal.MTLBuffer // nil for sharers
		}
		l := make([]lb, nLayers)
		projs := make([]bf16Projector, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		for li := range layers {
			w := layers[li]
			l[li].anw = sharedBytes(w.AttnNormW)
			l[li].mnw = sharedBytes(w.MLPNormW)
			if specs[li].OwnsCache() {
				l[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				l[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			projs[li] = bf16Projector{
				wQ: sharedBytes(w.WQ), wK: sharedBytes(w.WK), wV: sharedBytes(w.WV), wO: sharedBytes(w.WO),
				wGate: sharedBytes(w.WGate), wUp: sharedBytes(w.WUp), wDown: sharedBytes(w.WDown),
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
			}
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
					if encErr = encAttnHalfKV(enc, in, l[li].anw, l[li].kCache, l[li].vCache, offBuf, hBuf, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, slideW, base, scale, eps); encErr != nil {
						enc.EndEncoding()
						return
					}
				} else {
					own := specs[li].KVShareFrom
					if encErr = encAttnHalfShared(enc, in, l[li].anw, l[own].kCache, l[own].vCache, offBuf, hBuf, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, slideW, base, scale, eps); encErr != nil {
						enc.EndEncoding()
						return
					}
				}
				if encErr = encMLPHalfBF16(enc, hBuf, l[li].mnw, out, msc, projs[li], dModel, dFF, eps); encErr != nil {
					enc.EndEncoding()
					return
				}
				in, out = out, in
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			copy(outputs[t], unsafe.Slice((*byte)(in.Contents()), dModel*bf16Size))
		}
	})
	return outputs, encErr
}
