// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/scheme"
	"github.com/tmc/apple/metal"
)

// This file assembles the attention half of a decode step on-device, in bf16
// (the dtype attention actually runs in). The enc* helpers each encode one
// dispatch into a caller-supplied encoder — the bf16 siblings of chain.go's
// float32 encode helpers, with bindings copied verbatim from the parity-proven
// bf16 ops in bf16.go / sdpa.go. AttentionBlock chains them in one command
// buffer with every intermediate resident.

func sharedBytes(b []byte) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
}

// residentBufs caches the GPU buffer for a RESIDENT weight slice. The MoE expert weights are the
// SAME mmap bytes every token, but the host-orchestrated MoE compute re-uploaded (sharedBytes COPIES)
// each selected expert's weight EVERY token. Those buffers are objc-"new" RETAINED, which
// withAutoreleasePool cannot free, so a long generation leaked tens of MB/token → 26B-A4B OOM'd at
// ~70 tokens (badLayers=0 throughout — a leak, not a decode bug). residentBytes uploads each distinct
// weight slice ONCE — keyed by its start address in the stable safetensors mmap — and reuses it, the
// resident pattern the dense projector already uses. Process-lifetime: model weights live as long as
// the model (a model swap would want eviction, not a concern for a single served model). The mutex
// guards concurrent sessions; the decode itself is single-goroutine.
var (
	residentBufMu sync.Mutex
	residentBufs  = map[uintptr]residentBuf{}
)

// residentBuf pins the backing slice alongside its uploaded buffer: caching by &b[0] is only sound
// while that address stays valid, which is automatic for the safetensors mmap (never moved) but NOT
// for a Go-managed slice (GC can free it and reuse the address → a stale cache hit). Holding b keeps
// it alive, so the key can never be re-issued for different data.
type residentBuf struct {
	buf metal.MTLBuffer
	pin []byte
}

func residentBytes(b []byte) metal.MTLBuffer {
	key := uintptr(unsafe.Pointer(&b[0]))
	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	if r, ok := residentBufs[key]; ok {
		return r.buf
	}
	buf := sharedBytes(b)
	residentBufs[key] = residentBuf{buf: buf, pin: b}
	return buf
}

// sharedOrNil is sharedBytes for an optional weight: nil/empty → a nil MTLBuffer (the
// half-encoders treat a nil norm buffer as "skip"), so callers can pass an absent gemma4
// post-norm straight through without a length guard.
func sharedOrNil(b []byte) metal.MTLBuffer {
	if len(b) == 0 {
		return nil
	}
	return sharedBytes(b)
}

func scratchBF16(nElems int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(nElems*bf16Size), metal.MTLResourceStorageModeShared)
}

// scratchF32 allocates a shared float32 scratch buffer of nElems — the 2-pass SDPA
// per-block sums/maxs intermediates are float32 (the online-softmax accumulators).
func scratchF32(nElems int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(nElems*4), metal.MTLResourceStorageModeShared)
}

// encRMSNormBF16 encodes a single-row bf16 RMSNorm (axisSize ≤ 4096) into enc. wOff offsets the
// WEIGHT binding (bytes) — the zero-copy weight path binds the norm weight at its offset into the
// shared shard mmap buffer rather than uploading it; wOff=0 is the plain (copied-buffer) binding.
func encRMSNormBF16(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, wOff uint, axisSize int, eps float32) error {
	pso, err := pipelineFor(rmsKernelBF16(axisSize))
	if err != nil {
		return err
	}
	// single-row up to the limit, else the looped kernel (a max-threads threadgroup that grid-strides
	// the axis) — a single row of axis > 4096 (gemma4 31B hidden 5376) overruns the single-row cap.
	// One shared body (emitRMSNorm) records the binding ABI into the live encoder here and into the ICB
	// recorder's setRMS — the path-unifying dispatchSink (one math, two targets).
	emitRMSNorm(encSink{enc}, pso, x, w, out, wOff, axisSize, eps, rmsThreadgroup(axisSize, pso))
	return nil
}

// encRMSNormRowsBF16 RMS-norms `rows` contiguous rows of axisSize each, independently,
// with the single shared weight (axisSize) — one threadgroup per row (the grid carries
// the batch, exactly as the standalone RMSNormBF16's rows path). gemma4 QK-norm uses this
// to norm each attention head's headDim slice (rows = nHeads, axisSize = headDim) with the
// shared q_norm/k_norm weight. wOff offsets the WEIGHT binding (the zero-copy path binds it at its
// offset into the shared shard buffer; 0 is the plain binding). Safe in-place (the per-row
// reduction barriers before the write phase, and each thread writes only its own element).
func encRMSNormRowsBF16(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, rows, axisSize int, eps float32) error {
	pso, err := pipelineFor("rmsbfloat16")
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(x, xOff, 0)
	enc.SetBufferWithOffsetAtIndex(w, wOff, 1)
	enc.SetBufferWithOffsetAtIndex(out, outOff, 2)
	setEncFloat32(enc, eps, 3)
	setEncInt32(enc, int32(axisSize), 4)
	setEncInt32(enc, 1, 5)
	tg := uint(rmsSimdSize * ((((axisSize + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1},
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
	)
	return nil
}

// encGemvBF16 encodes out = mat @ vec (bf16, mat row-major outDim×inDim) into enc.
func encGemvBF16(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, outDim, inDim int) error {
	return encGemvBF16To(enc, mat, vec, out, 0, 0, outDim, inDim)
}

// encGemvBF16To is encGemvBF16 that binds the weight MATRIX at matOff BYTES and writes the result
// starting at outOff BYTES into out. matOff lets the zero-copy weight path bind the projection
// weight at its offset into the shared shard mmap buffer (vs an uploaded copy); outOff lets the
// decode KV path project K/V straight into the (seq-major) cache at the current token's row, so
// the projection IS the cache append (no copy kernel; the gemv output index is relative to the
// bound buffer offset). matOff=outOff=0 is the plain projection.
func encGemvBF16To(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, matOff, outOff uint, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(mat, matOff, 0)
	enc.SetBufferWithOffsetAtIndex(vec, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, outOff, 3)
	setEncInt32(enc, int32(inDim), 4)
	setEncInt32(enc, int32(outDim), 5)
	setEncInt32(enc, int32(inDim), 6)
	setEncInt32(enc, 1, 9)
	setEncInt32(enc, 1, 10)
	setEncInt64(enc, 0, 11)
	setEncInt64(enc, 0, 12)
	nOutPerTgp := bm * sm * tm
	nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
	)
	return nil
}

// encQMVBF16 encodes a bf16-activation 4-bit quantised matvec (out = x @ Wᵀ) into
// enc — the chained sibling of QMVBF16 for the quantised decode layer. Same kernel
// (affine_qmv[_fast]_bfloat16_t) and ABI as QMVBF16. wqOff/scalesOff/biasesOff bind the three
// quant weight tensors at their offsets into the shared shard mmap buffer(s) (the zero-copy weight
// path; each tensor can sit in a different shard, hence three offsets) — 0/0/0 is the plain
// (uploaded-copy) binding. outOff lets the projection write its result straight into a cache row
// (the V projection), exactly like encGemvBF16To. wq is packed 4-bit; scales/biases bf16.
func qmvBF16KernelName(outDim, inDim, groupSize, bits int) string {
	variant := "_qmv_"
	if outDim%8 == 0 && inDim%512 == 0 {
		variant = "_qmv_fast_"
	}
	return core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d_batch_0", variant, groupSize, bits)
}

func encQMVBF16(enc metal.MTLComputeCommandEncoder, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, outOff uint, outDim, inDim, groupSize, bits int) error {
	pso, err := pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(wq, wqOff, 0)
	enc.SetBufferWithOffsetAtIndex(scales, scalesOff, 1)
	enc.SetBufferWithOffsetAtIndex(biases, biasesOff, 2)
	enc.SetBufferWithOffsetAtIndex(x, 0, 3)
	enc.SetBufferWithOffsetAtIndex(out, outOff, 4)
	setEncInt32(enc, int32(inDim), 5)  // K
	setEncInt32(enc, int32(outDim), 6) // N
	const bn, bk = 8, 32
	nTgp := (outDim + bn - 1) / bn
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1},
		metal.MTLSize{Width: bk, Height: 2, Depth: 1},
	)
	return nil
}

// encRoPEBF16 encodes single-token bf16 RoPE over x (b=1, nHeads, 1, headDim) at
// the position in offBuf into enc. offBuf holds one int32.
func encRoPEBF16(enc metal.MTLComputeCommandEncoder, x, out, offBuf metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	return encRoPEBF16To(enc, x, out, 0, 0, offBuf, nHeads, headDim, rotaryDim, base, scale)
}

// encRoPEBF16To is encRoPEBF16 that reads from inOff and writes the rotated result starting at
// outOff BYTES — used to RoPE the new token's K in place within the (seq-major) KV cache row.
// rotaryDim rotates only the first rotaryDim of each head (gemma4 partial rotary; == headDim is
// full); the kernel writes only the rotated dims, so for partial rotary call it IN PLACE
// (in==out, inOff==outOff) so the untouched [rotaryDim:headDim] tail keeps its input value.
func encRoPEBF16To(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	pso, err := ropePipelineBF16(false)
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(x, inOff, 0)
	enc.SetBufferWithOffsetAtIndex(out, outOff, 1)
	enc.SetBufferWithOffsetAtIndex(offBuf, 0, 2)
	setEncFloat32(enc, scale, 3)
	setEncInt64(enc, int64(headDim), 4) // out_strides[0] = T*D, T==1 — FULL head stride (the tail lives here)
	setEncFloat32(enc, float32(math.Log2(float64(base))), 10)
	dim0 := uint(rd / 2) // grid.x = rotaryDim/2 → pairs i with i+rotaryDim/2, freq normalised over rotaryDim
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: dim0, Height: uint(nHeads), Depth: 1},
		metal.MTLSize{Width: dim0, Height: 1, Depth: 1},
	)
	return nil
}

// encSDPA encodes single-query bf16 attention over a HEAD-MAJOR cache into enc:
// q (1, nHeads, 1, headDim), k/v (1, nKVHeads, kvLen, headDim) → out (1, nHeads,
// 1, headDim). No mask / not causal.
func encSDPA(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, kvLen int, scale float32) error {
	// head-major: head h, seq i, dim d at (h*kvLen + i)*headDim + d
	return encSDPAStrided(enc, q, k, v, out, nHeads, nKVHeads, headDim, kvLen,
		int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale, 0)
}

// slideWindow returns the cache window the SDPA attends for a layer decoding at
// position pos: the full prefix [0..pos] (start 0, n pos+1) for a global layer
// (slideW <= 0), or the last slideW rows once the window is exceeded — the
// correctness of sliding-window attention. (The cache still stores all rows; the
// rotating W-sized buffer is a separate memory optimisation.)
func slideWindow(pos, slideW int) (start, n int) {
	if slideW > 0 && pos+1 > slideW {
		return pos + 1 - slideW, slideW
	}
	return 0, pos + 1
}

// encSDPAStrided encodes single-query bf16 attention with explicit element
// strides — the sdpa_vector kernel indexes keys as kv_head*k_head_stride +
// seq*k_seq_stride + d with headDim contiguous (innermost), so the cache layout
// is the caller's choice. The decode KV path uses a SEQ-MAJOR cache
// [seq, nKVHeads, headDim] (k_head_stride=headDim, k_seq_stride=nKVHeads*headDim)
// so appending a token is one contiguous row write; encSDPA passes the head-major
// strides. n is the live cache length (the grown window).
// kvByteOff offsets the K and V bindings (bytes) — used to attend a window of the
// cache starting at a non-zero row (sliding-window attention reads the last W rows).
func encSDPAStrided(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	pso, err := sdpaVectorPipeline(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim))
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(q, 0, 0)
	enc.SetBufferWithOffsetAtIndex(k, kvByteOff, 1)
	enc.SetBufferWithOffsetAtIndex(v, kvByteOff, 2)
	enc.SetBufferWithOffsetAtIndex(out, 0, 3)
	setEncInt32(enc, int32(nHeads/nKVHeads), 4) // gqa_factor
	setEncInt32(enc, int32(n), 5)               // N (live cache length)
	setEncInt64(enc, kHeadStride, 6)
	setEncInt64(enc, kSeqStride, 7)
	setEncInt64(enc, vHeadStride, 8)
	setEncInt64(enc, vSeqStride, 9)
	setEncFloat32(enc, scale, 10)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, // b=1
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
	return nil
}

// encSDPA2PassStrided encodes the TWO-pass long-context SDPA into enc (b=1 decode):
// pass 1 (sdpa_vector_2pass_1) fans the cache reduction over `blocks` threadgroups,
// each writing its segment's online-softmax partials (weighted-V sum + sum/max) into
// the caller's once-allocated intermediates; pass 2 (sdpa_vector_2pass_2) merges them
// into the head output. Same q/k/v/out + element strides + kvByteOff as
// encSDPAStrided (the strides describe the caller's cache layout, the offset selects a
// sliding window) — the two dispatches are serial in enc so pass 2 sees pass 1's
// writes. Token-identical to encSDPAStrided (sdpa_2pass_test.go), differing only in how
// the reduction parallelises — so it keeps scaling where the single-pass kernel stalls.
func encSDPA2PassStrided(enc metal.MTLComputeCommandEncoder, q, k, v, out, partials, sums, maxs metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	gqa := nHeads / nKVHeads
	blocks := sdpa2PassBlocks(n)
	pso1, err := sdpaVector2Pass1Pipeline(core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", headDim, headDim), blocks)
	if err != nil {
		return err
	}
	pso2, err := sdpaVector2Pass2Pipeline(core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", headDim))
	if err != nil {
		return err
	}
	// Pass 1: per-block partials. grid (nKVHeads, b=1, blocks); group (32, gqa, qseq=1).
	enc.SetComputePipelineState(pso1)
	enc.SetBufferWithOffsetAtIndex(q, 0, 0)
	enc.SetBufferWithOffsetAtIndex(k, kvByteOff, 1)
	enc.SetBufferWithOffsetAtIndex(v, kvByteOff, 2)
	enc.SetBufferWithOffsetAtIndex(partials, 0, 3)
	enc.SetBufferWithOffsetAtIndex(sums, 0, 4)
	enc.SetBufferWithOffsetAtIndex(maxs, 0, 5)
	setEncInt32(enc, int32(n), 7) // N
	setEncInt64(enc, kHeadStride, 8)
	setEncInt64(enc, kSeqStride, 9)
	setEncInt64(enc, vHeadStride, 10)
	setEncInt64(enc, vSeqStride, 11)
	setEncFloat32(enc, scale, 12)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(nKVHeads), Height: 1, Depth: uint(blocks)},
		metal.MTLSize{Width: 32, Height: uint(gqa), Depth: 1},
	)
	// Pass 2: merge per-block partials into the head output.
	enc.SetComputePipelineState(pso2)
	enc.SetBufferWithOffsetAtIndex(partials, 0, 0)
	enc.SetBufferWithOffsetAtIndex(sums, 0, 1)
	enc.SetBufferWithOffsetAtIndex(maxs, 0, 2)
	enc.SetBufferWithOffsetAtIndex(out, 0, 3)
	setEncInt32(enc, blocks, 4)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, // b=1
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
	return nil
}

// encSDPADecode routes a single-query decode SDPA to the 2-pass long-context kernels
// once the attended window n reaches the single-pass knee AND the scratch carries the
// (once-allocated) 2-pass intermediates; otherwise the proven single-pass kernel. Same
// buffers/strides/offset either way, so the choice is invisible to the caller and
// token-identical — only the cache-reduction parallelism differs. The intermediates
// live in sc so the long-context path adds NO per-token allocation.
func encSDPADecode(enc metal.MTLComputeCommandEncoder, sc attnScratch, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32, kvByteOff uint) error {
	if n >= sdpa2PassMinKV && sc.p2Partials != nil && !sdpa2PassDisabledForTest {
		return encSDPA2PassStrided(enc, q, k, v, out, sc.p2Partials, sc.p2Sums, sc.p2Maxs, nHeads, nKVHeads, headDim, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale, kvByteOff)
	}
	return encSDPAStrided(enc, q, k, v, out, nHeads, nKVHeads, headDim, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale, kvByteOff)
}

// encBinaryDT encodes the element-wise binary op (op = "Add" | "Multiply") in the
// activation dtype dt — kernel "vv_<op><dt.Name>" — over n elements into enc. The
// dtype is resolved from the registered scheme (scheme.BFloat16, scheme.Float32, …),
// so a new activation dtype is a registered scheme, not a new hardcoded encoder.
func encBinaryDT(enc metal.MTLComputeCommandEncoder, op string, dt scheme.DType, a, b, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor("vv_" + op + dt.Name())
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(a, 0, 0)
	enc.SetBufferWithOffsetAtIndex(b, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, 0, 2)
	setEncInt32(enc, int32(n), 3)
	group := uint(256)
	if uint(n) < group {
		group = uint(n)
	}
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

// encAddBF16 / encMulBF16 are the bf16-bound conveniences for gemma's MLP and
// residual paths — the registered scheme.BFloat16 dtype through encBinaryDT.
func encAddBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryDT(enc, "Add", scheme.BFloat16, a, b, out, n)
}
func encMulBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryDT(enc, "Multiply", scheme.BFloat16, a, b, out, n)
}

// encUnaryDT encodes the element-wise unary op (op = "Tanh", …) in the activation
// dtype dt — kernel "v_<op><dt.Name><dt.Name>" (the metallib repeats the dtype for
// in+out) — over n elements. The count is a uint32 at index 2 (SetBytes), matching
// TanhBF16. Dtype resolved from the registered scheme, not hardcoded.
func encUnaryDT(enc metal.MTLComputeCommandEncoder, op string, dt scheme.DType, in, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor("v_" + op + dt.Name() + dt.Name())
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(in, 0, 0)
	enc.SetBufferWithOffsetAtIndex(out, 0, 1)
	cnt := uint32(n)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&cnt)), 4), 4, 2)
	group := uint(256)
	if uint(n) < group {
		group = uint(n)
	}
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

// encTanhBF16 is the bf16-bound tanh (gemma's gelu nonlinearity) — scheme.BFloat16 through encUnaryDT.
func encTanhBF16(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, n int) error {
	return encUnaryDT(enc, "Tanh", scheme.BFloat16, in, out, n)
}

// AttentionBlock runs the attention half of a gemma decode step on-device, in
// bf16, over a given KV cache (the read path of a single new token):
//
//	normed  = rmsnorm(x, normWeight)
//	q       = wQ · normed                 (dModel → nHeads·headDim)
//	q       = rope(q, offset)             (per head, full rotary)
//	attn    = sdpa(q, kCache, vCache)     (single query over the cache)
//	attnOut = wO · attn                   (nHeads·headDim → dModel)
//	out     = x + attnOut                 (residual)
//
// Every buffer is bf16 and stays resident; the whole block is one command
// buffer, one commit. kCache/vCache are the post-RoPE cache (1, nKVHeads, kvLen,
// headDim). The cache-write half (wK/wV projections, RoPE on the new K, append)
// is a separate follow-up. All inputs/outputs are raw bf16 bytes. The result
// equals the same native bf16 ops run separately — proven in the tests.
func AttentionBlock(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(normWeight) != dModel*bf16Size {
		return nil, core.NewError("native.AttentionBlock: x/normWeight must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.AttentionBlock: wQ/wO size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.AttentionBlock: kCache/vCache size mismatch")
	}

	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		nwBuf := residentBytes(normWeight)
		wqBuf, woBuf := residentBytes(wQ), residentBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)

		normed := scratchBF16(dModel)
		q, qr := scratchBF16(qDim), scratchBF16(qDim)
		attn := scratchBF16(qDim)
		attnOut := scratchBF16(dModel)
		outBuf := scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		steps := []func() error{
			func() error { return encRMSNormBF16(enc, xBuf, nwBuf, normed, 0, dModel, eps) },
			func() error { return encGemvBF16(enc, wqBuf, normed, q, qDim, dModel) },
			func() error { return encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, headDim, base, scale) },
			func() error { return encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale) },
			func() error { return encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim) },
			func() error { return encAddBF16(enc, xBuf, attnOut, outBuf, dModel) },
		}
		for _, step := range steps {
			if encErr = step(); encErr != nil {
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	return out, encErr
}
