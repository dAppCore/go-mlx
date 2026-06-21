// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
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

// encGemvRowsF32 is the float32 sibling of encGemvRowsBF16: out[r] = W · in[r] over L rows, one
// dispatch per row into a single encoder. Used for the attention scores and the score·V product,
// which run in fp32 — the precision the fused SDPA keeps through QK^T and the softmax, so the
// decomposition tracks it instead of rounding logits to bf16 before the softmax.
func encGemvRowsF32(enc metal.MTLComputeCommandEncoder, w, in, out metal.MTLBuffer, L, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	nOutPerTgp := bm * sm * tm
	nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
	for r := 0; r < L; r++ {
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(w, 0, 0)
		enc.SetBufferWithOffsetAtIndex(in, uint(r*inDim*4), 1)
		enc.SetBufferWithOffsetAtIndex(out, uint(r*outDim*4), 3)
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

// matRowsF32 is the float32 multi-row matmul out[L,outDim] = in[L,inDim] @ Wᵀ (W row-major
// [outDim,inDim]), composed by looping the bit-exact fp32 gemv. The attention core's two products run
// through it so the scores and the softmax stay in fp32.
func matRowsF32(w, in []float32, L, outDim, inDim int) ([]float32, error) {
	if len(w) != outDim*inDim || len(in) != L*inDim {
		return nil, core.NewError("native.matRowsF32: size mismatch (w=outDim*inDim, in=L*inDim)")
	}
	if L == 0 || outDim == 0 || inDim == 0 {
		return make([]float32, L*outDim), nil
	}
	out := make([]float32, L*outDim)
	var encErr error
	withAutoreleasePool(func() {
		wBuf, inBuf := shared(w), shared(in)
		outBuf := scratch(L * outDim)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encGemvRowsF32(enc, wBuf, inBuf, outBuf, L, outDim, inDim); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), L*outDim))
	})
	return out, encErr
}

// bf16HeadF32 reads one [L,headDim] head out of a [heads,L,headDim] bf16 buffer as fp32.
func bf16HeadF32(b []byte, head, L, headDim int) []float32 {
	out := make([]float32, L*headDim)
	base := head * L * headDim * bf16Size
	for i := range out {
		o := base + i*bf16Size
		out[i] = bf16ToF32(b[o], b[o+1])
	}
	return out
}

// transposeF32 returns the [cols,rows] transpose of a row-major [rows,cols] fp32 matrix.
func transposeF32(m []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = m[r*cols+c]
		}
	}
	return out
}

// softmaxRowsF32 softmaxes each length-cols row of an [rows,cols] fp32 matrix in place, max-shifted
// for stability — the fp32 softmax the fused SDPA does internally.
func softmaxRowsF32(m []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := m[r*cols : r*cols+cols]
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for j, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			row[j] = e
			sum += e
		}
		inv := float32(1) / sum
		for j := range row {
			row[j] *= inv
		}
	}
}

// VisionSDPA computes full (non-causal, no-mask) bidirectional attention by DECOMPOSITION — the
// composition stand-in for the fused steel attention the vision tower's encoder would otherwise need
// wrapping. q is [nHeads,L,headDim] bf16, k/v are [nKVHeads,L,headDim] bf16 (B=1), out is
// [nHeads,L,headDim] bf16. Per query head: scores[L,L] = q·kᵀ·scale (fp32) → row softmax (fp32) →
// out = scores·v (fp32) → bf16. GQA maps each query head to kv head h/(nHeads/nKVHeads). Keeping the
// scores and softmax in fp32 (the precision the fused kernel keeps) bounds the deviation; the matmuls
// run on-device (matRowsF32), the softmax host-side. Numerically equivalent to
// pkg/metal.ScaledDotProductAttention within a measured tolerance (vision_test.go), not bit-identical.
func VisionSDPA(q, k, v []byte, L, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.VisionSDPA: nHeads must be a multiple of nKVHeads")
	}
	if len(q) != nHeads*L*headDim*bf16Size {
		return nil, core.NewError("native.VisionSDPA: len(q) must equal nHeads*L*headDim*2 bytes")
	}
	if len(k) != nKVHeads*L*headDim*bf16Size || len(v) != len(k) {
		return nil, core.NewError("native.VisionSDPA: len(k)/len(v) must equal nKVHeads*L*headDim*2 bytes")
	}
	grp := nHeads / nKVHeads
	out := make([]byte, nHeads*L*headDim*bf16Size)
	for h := 0; h < nHeads; h++ {
		kvh := h / grp
		qh := bf16HeadF32(q, h, L, headDim)
		kh := bf16HeadF32(k, kvh, L, headDim)
		vh := bf16HeadF32(v, kvh, L, headDim)

		// scores[i,j] = qh[i] · kh[j]  →  matRowsF32(W=kh[L,d], in=qh[L,d]) = qh @ khᵀ
		scores, err := matRowsF32(kh, qh, L, L, headDim)
		if err != nil {
			return nil, err
		}
		for i := range scores {
			scores[i] *= scale
		}
		softmaxRowsF32(scores, L, L)

		// out[i,o] = Σ_j scores[i,j]·vh[j,o]  →  matRowsF32(W=vhᵀ[d,L], in=scores[L,L])
		oh, err := matRowsF32(transposeF32(vh, L, headDim), scores, L, headDim, L)
		if err != nil {
			return nil, err
		}
		base := h * L * headDim * bf16Size
		for i, val := range oh {
			hh := f32ToBF16(val)
			out[base+i*bf16Size], out[base+i*bf16Size+1] = byte(hh), byte(hh>>8)
		}
	}
	return out, nil
}

// bf16ToF32Slice / f32ToBf16Slice convert a whole bf16 byte buffer to/from fp32 — the host-side edge
// where the per-head norms and the 2-D rope run before they hand bf16 back to the device matmuls.
func bf16ToF32Slice(b []byte) []float32 {
	out := make([]float32, len(b)/bf16Size)
	for i := range out {
		out[i] = bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
	}
	return out
}

func f32ToBf16Slice(f []float32) []byte {
	b := make([]byte, len(f)*bf16Size)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*bf16Size], b[i*bf16Size+1] = byte(h), byte(h>>8)
	}
	return b
}

// rmsNormVec RMS-normalises v in place (x·rsqrt(mean(x²)+eps)), then scales by w when non-nil — the
// plain gemma RMSNorm (no +1 bias), matching native's RMSNorm kernel and metal's RMSNormNoScale for
// the value path. Run per head over the headDim axis.
func rmsNormVec(v, w []float32, eps float32) {
	var ss float32
	for _, x := range v {
		ss += x * x
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(v))+eps)))
	for i := range v {
		v[i] *= inv
		if w != nil {
			v[i] *= w[i]
		}
	}
}

// ropePartRotate applies one rotate-half RoPE block to a length-m slice for a single grid coordinate:
// out[d] = part[d]·cos(θ_d) + rot[d]·sin(θ_d), rot = [-part[half:], part[:half]], θ_d = coord·invFreq[d%（m/2)].
// Lifted from metal's gemma4VisionRotatePart + gemma4Vision2DRoPETables (the 2-D vision RoPE).
func ropePartRotate(out, part []float32, coord float64, invFreq []float64, m int) {
	half := m / 2
	for d := 0; d < m; d++ {
		angle := coord * invFreq[d%half]
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		var rot float32
		if d < half {
			rot = -part[half+d]
		} else {
			rot = part[d-half]
		}
		out[d] = part[d]*c + rot*s
	}
}

// vision2DRoPEHeadMajor applies the gemma4 vision 2-D RoPE to x [L, N, headDim] (fp32, per-patch
// per-head, after QK-norm) and transposes to head-major [N, L, headDim]. The first rotatedPerDim =
// 2·(headDim/4) dims rotate with the patch X coordinate, the next rotatedPerDim with Y, any tail
// passes through — exactly metal's gemma4VisionApply2DRoPE. base==0 ⇒ no rotation (only the
// transpose). gridH·gridW must equal L.
func vision2DRoPEHeadMajor(x []float32, L, N, headDim, gridH, gridW int, base float32) []float32 {
	out := make([]float32, N*L*headDim)
	rotatedPerDim := 2 * (headDim / 4)
	rotatedTotal := rotatedPerDim * 2
	doRoPE := base != 0 && rotatedPerDim >= 2 && gridW > 0
	var invFreq []float64
	if doRoPE {
		half := rotatedPerDim / 2
		invFreq = make([]float64, half)
		for i := 0; i < half; i++ {
			invFreq[i] = 1.0 / math.Pow(float64(base), float64(2*i)/float64(rotatedPerDim))
		}
	}
	for pos := 0; pos < L; pos++ {
		cx, cy := float64(pos%gridW), float64(pos/gridW)
		for h := 0; h < N; h++ {
			in := x[(pos*N+h)*headDim : (pos*N+h)*headDim+headDim]
			o := out[(h*L+pos)*headDim : (h*L+pos)*headDim+headDim]
			if !doRoPE {
				copy(o, in)
				continue
			}
			ropePartRotate(o[0:rotatedPerDim], in[0:rotatedPerDim], cx, invFreq, rotatedPerDim)
			ropePartRotate(o[rotatedPerDim:rotatedTotal], in[rotatedPerDim:rotatedTotal], cy, invFreq, rotatedPerDim)
			for d := rotatedTotal; d < headDim; d++ {
				o[d] = in[d]
			}
		}
	}
	return out
}

// qkNormRoPEHeadMajor takes a [L, N·headDim] bf16 projection, applies the per-head QK-norm (RMSNorm
// with normW) then the 2-D RoPE, and returns head-major [N, L, headDim] bf16 ready for VisionSDPA.
func qkNormRoPEHeadMajor(proj, normW []byte, L, N, headDim, gridH, gridW int, base, eps float32) []byte {
	f := bf16ToF32Slice(proj) // [L, N, headDim]
	w := bf16ToF32Slice(normW)
	for i := 0; i < L*N; i++ {
		rmsNormVec(f[i*headDim:i*headDim+headDim], w, eps)
	}
	return f32ToBf16Slice(vision2DRoPEHeadMajor(f, L, N, headDim, gridH, gridW, base))
}

// vNormHeadMajor takes a [L, N·headDim] bf16 V projection, applies the no-scale per-head RMSNorm
// (metal's RMSNormNoScale), and transposes to head-major [N, L, headDim] bf16.
func vNormHeadMajor(proj []byte, L, N, headDim int, eps float32) []byte {
	f := bf16ToF32Slice(proj) // [L, N, headDim]
	out := make([]float32, N*L*headDim)
	for pos := 0; pos < L; pos++ {
		for h := 0; h < N; h++ {
			v := f[(pos*N+h)*headDim : (pos*N+h)*headDim+headDim]
			rmsNormVec(v, nil, eps)
			copy(out[(h*L+pos)*headDim:(h*L+pos)*headDim+headDim], v)
		}
	}
	return f32ToBf16Slice(out)
}

// VisionLayerWeights is one SigLIP encoder layer's weights as bf16 byte views — the native-side,
// engine-neutral mirror of gemma4.LoadedVisionLayer (an adapter fills it; native imports no model).
// The four norms are [hidden]; QNorm/KNorm are [headDim]; the projections are row-major bf16.
type VisionLayerWeights struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm []byte
	WQ, WK, WV, WO                                 []byte
	QNorm, KNorm                                   []byte
	WGate, WUp, WDown                              []byte
}

// visionAttention runs the SigLIP attention subblock on a pre-normed [L, hidden] input: Q/K/V
// projections (on-device) → per-head QK-norm + 2-D RoPE (host) → decomposed full attention
// (VisionSDPA) → output projection. Returns [L, hidden] bf16.
func visionAttention(normed []byte, w *VisionLayerWeights, cfg VisionConfig) ([]byte, error) {
	qDim, kvDim := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	qP, err := MatRowsBF16(w.WQ, normed, cfg.GridH*cfg.GridW, qDim, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	kP, err := MatRowsBF16(w.WK, normed, cfg.GridH*cfg.GridW, kvDim, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	vP, err := MatRowsBF16(w.WV, normed, cfg.GridH*cfg.GridW, kvDim, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	L := cfg.GridH * cfg.GridW
	q := qkNormRoPEHeadMajor(qP, w.QNorm, L, cfg.NumHeads, cfg.HeadDim, cfg.GridH, cfg.GridW, cfg.RopeBase, cfg.RMSNormEps)
	k := qkNormRoPEHeadMajor(kP, w.KNorm, L, cfg.NumKVHeads, cfg.HeadDim, cfg.GridH, cfg.GridW, cfg.RopeBase, cfg.RMSNormEps)
	v := vNormHeadMajor(vP, L, cfg.NumKVHeads, cfg.HeadDim, cfg.RMSNormEps)

	// The actual gemma4 vision loader (buildGemma4VisionModel) hardcodes the attention scale to 1.0
	// (Gemma4VisionAttention.Attention = 1.0) — NOT 1/√headDim. The QK-norm makes the usual scaling
	// unnecessary. Taken from the real code, not derived.
	attn, err := VisionSDPA(q, k, v, L, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, 1.0)
	if err != nil {
		return nil, err
	}

	// head-major [N, L, headDim] → token-major [L, N·headDim] for the output projection.
	af := bf16ToF32Slice(attn)
	tok := make([]float32, L*qDim)
	for h := 0; h < cfg.NumHeads; h++ {
		for pos := 0; pos < L; pos++ {
			copy(tok[(pos*cfg.NumHeads+h)*cfg.HeadDim:(pos*cfg.NumHeads+h)*cfg.HeadDim+cfg.HeadDim],
				af[(h*L+pos)*cfg.HeadDim:(h*L+pos)*cfg.HeadDim+cfg.HeadDim])
		}
	}
	return MatRowsBF16(w.WO, f32ToBf16Slice(tok), L, cfg.Hidden, qDim)
}

// visionMLP runs the gated-GeLU feed-forward on [L, hidden] bf16: gate/up projections → gelu(gate)·up
// → down projection. The gelu·gate·up runs in fp32 (gemma's tanh-approx gelu) then back to bf16.
func visionMLP(ffIn []byte, w *VisionLayerWeights, L, hidden int) ([]byte, error) {
	ffDim := len(w.WGate) / bf16Size / hidden
	gate, err := MatRowsBF16(w.WGate, ffIn, L, ffDim, hidden)
	if err != nil {
		return nil, err
	}
	up, err := MatRowsBF16(w.WUp, ffIn, L, ffDim, hidden)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMul(bf16ToF32Slice(gate), bf16ToF32Slice(up))
	if err != nil {
		return nil, err
	}
	return MatRowsBF16(w.WDown, f32ToBf16Slice(gated), L, hidden, ffDim)
}

// VisionEncoderLayer runs one pre-norm SigLIP encoder block — the faithful re-expression of metal's
// Gemma4VisionEncoderLayer.Forward composed from native's validated ops: InputNorm → attention
// subblock → PostAttnNorm → residual → PreFFNorm → gated MLP → PostFFNorm → residual. x and the
// result are [L, hidden] bf16 (L = GridH·GridW). Numerically equivalent to metal within the measured
// vision tolerance, not bit-identical (the attention softmax + the host norms/rope are fp32).
func VisionEncoderLayer(x []byte, w *VisionLayerWeights, cfg VisionConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	L := cfg.GridH * cfg.GridW
	if len(x) != L*cfg.Hidden*bf16Size {
		return nil, core.NewError("native.VisionEncoderLayer: len(x) must equal GridH*GridW*Hidden*2 bytes")
	}
	normed, err := RMSNormBF16(x, w.InputNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	attnOut, err := visionAttention(normed, w, cfg)
	if err != nil {
		return nil, err
	}
	attnNormed, err := RMSNormBF16(attnOut, w.PostAttnNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	h, err := AddBF16(x, attnNormed)
	if err != nil {
		return nil, err
	}
	ffIn, err := RMSNormBF16(h, w.PreFFNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	ff, err := visionMLP(ffIn, w, L, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	ffNormed, err := RMSNormBF16(ff, w.PostFFNorm, L, cfg.Hidden, cfg.RMSNormEps)
	if err != nil {
		return nil, err
	}
	return AddBF16(h, ffNormed)
}
