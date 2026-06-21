// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// vision.go ports the gemma4 SigLIP vision tower forward to the no-cgo native path. The decode path
// holds byte-parity with mlx-c because error drift compounds over thousands of autoregressive
// tokens; the vision tower instead runs ONCE per image at prefill and emits soft-token rows into the
// text stream, so it is COMPOSED from native's existing bit-exact kernels (looped gemv, rmsnorm,
// rope, gelu, add) rather than wrapping mlx's fused steel GEMM / steel attention. The output is
// numerically EQUIVALENT to pkg/metal's tower within a measured tolerance (vision_test.go pins it),
// not bit-identical — a deliberate trade: there is no autoregressive feedback at prefill for the
// small fp difference to compound, AX-11 says the tower is not a perf target, and the composition is
// driver-agnostic, so go-rocm can share it verbatim (a gemv-loop + softmax are not Metal-specific
// the way the steel kernels are). Engine-neutral: this file names no model; the SigLIP geometry
// arrives as a VisionConfig.

// VisionConfig is the engine-neutral SigLIP tower geometry the forward reads — dimensions only, the
// loader fills it from the checkpoint's own declared dims (the vision-side sibling of model.Arch).
// No model name lives here: the same fields describe any patch-embedded vision transformer.
type VisionConfig struct {
	Hidden      int     // encoder width (gemma4-E4B: 768)
	PatchDim    int     // channels·patch·patch — the flattened patch-projection input (3·16·16 = 768)
	NumLayers   int     // encoder layer count
	NumHeads    int     // attention query heads
	NumKVHeads  int     // attention kv heads (GQA; == NumHeads for SigLIP)
	HeadDim     int     // per-head width (Hidden/NumHeads = 64)
	GridH       int     // patch grid rows (for 2-D rope + spatial pooling)
	GridW       int     // patch grid cols
	RopeBase    float32 // 2-D rope theta
	RMSNormEps  float32
	PoolKernel  int  // spatial pooling kernel (gemma4 default 3)
	Standardize bool // post-pool (x-bias)·scale
	// EmbeddingScale is √Hidden, multiplied into the pooled rows (cached to skip a per-pass sqrt).
	EmbeddingScale float32
}

// encGemvRowsBF16 projects L contiguous bf16 row-vectors through one bf16 weight in a single command
// encoder: for each row r, out[r] = W · in[r], driving native's bit-exact single-row gemv per row.
// It is the composition stand-in for a fused multi-row GEMM — correct and prefill-cheap (one commit,
// L serial dispatches; Metal's default hazard tracking serialises the non-overlapping writes to the
// shared out buffer). W is row-major [outDim, inDim] bf16 at byte offset wOff; in is the [L, inDim]
// activations and out the [L, outDim] result, both contiguous bf16. The per-row vec/out bindings
// carry the row's byte offset — encGemvBF16To binds vec at 0, so the rows loop lives here.
func encGemvRowsBF16(enc metal.MTLComputeCommandEncoder, w, in, out metal.MTLBuffer, wOff uint, L, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	nOutPerTgp := bm * sm * tm
	nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
	for r := 0; r < L; r++ {
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(w, wOff, 0)
		enc.SetBufferWithOffsetAtIndex(in, uint(r*inDim*bf16Size), 1)
		enc.SetBufferWithOffsetAtIndex(out, uint(r*outDim*bf16Size), 3)
		setEncInt32(enc, int32(inDim), 4)
		setEncInt32(enc, int32(outDim), 5)
		setEncInt32(enc, int32(inDim), 6)
		setEncInt32(enc, 1, 9)
		setEncInt32(enc, 1, 10)
		setEncInt64(enc, 0, 11)
		setEncInt64(enc, 0, 12)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
		)
	}
	return nil
}

// MatRowsBF16 is the multi-row sibling of MatVecBF16: out[L, outDim] = in[L, inDim] @ Wᵀ where W is
// row-major (outDim × inDim) bf16, all raw bf16 bytes. It composes the result by looping native's
// bit-exact gemv over the L rows in one command buffer — the projection primitive every vision/audio
// prefill matmul is built from (patch embed, Q/K/V/O, gate/up/down, projector). Unlike MatVecBF16
// (byte-parity with one mlx gemv), this is numerically EQUIVALENT to pkg/metal.Matmul — which
// dispatches the fused steel GEMM — within a small tolerance, NOT bit-identical: the per-row gemv
// reduces over K in a different order than the tiled GEMM. The trade is deliberate (see the file
// header); vision_test.go measures the deviation.
//
//	out, err := native.MatRowsBF16(weightBytes, inBytes, L, outDim, inDim)
func MatRowsBF16(w, in []byte, L, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(w) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.MatRowsBF16: len(w) must equal outDim*inDim*2 bytes")
	}
	if len(in) != L*inDim*bf16Size {
		return nil, core.NewError("native.MatRowsBF16: len(in) must equal L*inDim*2 bytes")
	}
	if L == 0 || outDim == 0 || inDim == 0 {
		return make([]byte, L*outDim*bf16Size), nil
	}

	outLen := L * outDim * bf16Size
	out := make([]byte, outLen)
	var encErr error
	withAutoreleasePool(func() {
		wBuf := sharedBytes(w)
		inBuf := sharedBytes(in)
		outBuf := scratchBF16(L * outDim)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encGemvRowsBF16(enc, wBuf, inBuf, outBuf, 0, L, outDim, inDim); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, encErr
}

// scaleSiglipPatchesBF16 applies the SigLIP input normalisation (x-0.5)·2 to bf16 patch pixels,
// host-side. This is BYTE-IDENTICAL to metal's on-device AddScalar(-0.5)+MulScalar(2): the ×2 is an
// exact bf16 exponent bump (mantissa unchanged, no rounding), so the only rounding either way is the
// one after the subtract, and round-to-nearest-even commutes with the doubling. Host-side keeps the
// cheap per-pixel affine off the GPU; the heavy patch projection stays on-device (MatRowsBF16).
func scaleSiglipPatchesBF16(pixels []byte) []byte {
	out := make([]byte, len(pixels))
	for i := 0; i+1 < len(pixels); i += bf16Size {
		x := bf16ToF32(pixels[i], pixels[i+1])
		h := f32ToBF16((x - 0.5) * 2.0)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// VisionPatchEmbed runs the SigLIP patch embedding: scale the (pre-patchified) pixel patches by the
// SigLIP convention (x-0.5)·2, project them to the encoder width, and add the learned position
// embeddings. pixels is [L, patchDim] bf16; weight is the patch projection [hidden, patchDim] bf16 —
// a non-overlapping patch conv IS exactly this linear projection, so the conv-weight checkpoint and
// the linear-weight checkpoint feed the same matmul. posEmb is the per-patch position embedding rows
// [L, hidden] bf16 already arranged for this grid, or nil when the tower uses only 2-D rope. Returns
// the [L, hidden] bf16 patch rows that open the encoder. Composed from the proven byte-identical
// MatRowsBF16 + AddBF16, so it inherits their equivalence to pkg/metal's patch embedder.
func VisionPatchEmbed(pixels, weight, posEmb []byte, L, patchDim, hidden int) ([]byte, error) {
	if len(pixels) != L*patchDim*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(pixels) must equal L*patchDim*2 bytes")
	}
	if len(weight) != hidden*patchDim*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(weight) must equal hidden*patchDim*2 bytes")
	}
	proj, err := MatRowsBF16(weight, scaleSiglipPatchesBF16(pixels), L, hidden, patchDim)
	if err != nil {
		return nil, err
	}
	if posEmb == nil {
		return proj, nil
	}
	if len(posEmb) != L*hidden*bf16Size {
		return nil, core.NewError("native.VisionPatchEmbed: len(posEmb) must equal L*hidden*2 bytes")
	}
	return AddBF16(proj, posEmb)
}
