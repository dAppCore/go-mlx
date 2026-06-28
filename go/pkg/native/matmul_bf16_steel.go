// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// matmul_bf16_steel.go drives MLX's fused steel GEMM for bf16 — the multi-row projection that streams
// the weight ONCE for all M rows, vs MatRowsBF16's per-row gemv that re-reads the weight M times. It is
// BYTE-IDENTICAL to MatRowsBF16 (and to pkg/metal.Matmul): bf16's rounding absorbs the GEMM-vs-gemv
// accumulation-order difference completely (TestProbeBF16GemvVsMatmul measured 0/N across MTP shapes,
// TestMatMulBF16NT pins it), so unlike the f32 path no dispatch-matching is needed — any correct bf16
// GEMM tiling rounds to the same bf16 bytes. This is the MTP batched-verify decode speedup: K draft
// rows projected for ~one row's weight bandwidth.

// bf16SteelNT is the fused nt tiling (a · bᵀ, b stored [N,K]); the metallib ships this bf16 variant.
var bf16SteelNT = steelTiling{64, 32, 32, 2, 2, "steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2"}

var (
	matMulBF16SteelScratchPool   sync.Pool
	errMatMulBF16SteelScratchDim = core.NewError("native.matMulBF16SteelScratch: dimension mismatch")
)

type matMulBF16SteelScratch struct {
	M, K, N       int
	a, out        *pinnedNoCopyBytes
	params        *pinnedNoCopyBytes
	paramsFilled  bool
	outViewPtr    uintptr
	outViewLen    int
	outView       metal.MTLBuffer
	outViewPinned *pinnedNoCopyBytes
}

func newMatMulBF16SteelScratch(M, K, N int) (*matMulBF16SteelScratch, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return nil, core.NewError("native.newMatMulBF16SteelScratch: invalid dimensions")
	}
	a, err := newPinnedNoCopyBytes(M * K * bf16Size)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(M * N * bf16Size)
	if err != nil {
		a.Close()
		return nil, err
	}
	params, err := newPinnedNoCopyBytes(72)
	if err != nil {
		a.Close()
		out.Close()
		return nil, err
	}
	return &matMulBF16SteelScratch{M: M, K: K, N: N, a: a, out: out, params: params}, nil
}

func getMatMulBF16SteelScratch(M, K, N int) (*matMulBF16SteelScratch, error) {
	if v := matMulBF16SteelScratchPool.Get(); v != nil {
		s := v.(*matMulBF16SteelScratch)
		if s.M == M && s.K == K && s.N == N && s.a != nil && s.out != nil && s.params != nil {
			return s, nil
		}
		s.Close()
	}
	return newMatMulBF16SteelScratch(M, K, N)
}

func putMatMulBF16SteelScratch(s *matMulBF16SteelScratch) {
	if s != nil {
		matMulBF16SteelScratchPool.Put(s)
	}
}

func (s *matMulBF16SteelScratch) Close() {
	if s == nil {
		return
	}
	if s.a != nil {
		s.a.Close()
		s.a = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	if s.params != nil {
		s.params.Close()
		s.params = nil
	}
	s.closeOutputView()
	s.M, s.K, s.N = 0, 0, 0
	s.paramsFilled = false
}

func (s *matMulBF16SteelScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outViewPinned != nil {
		s.outViewPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outViewPinned = nil
}

func (s *matMulBF16SteelScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outViewPinned = pinned
	return buf, true
}

func (s *matMulBF16SteelScratch) buffers(a []byte, t steelTiling) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.a == nil || s.out == nil || s.params == nil {
		return nil, nil, nil, core.NewError("native.matMulBF16SteelScratch.buffers: scratch is nil")
	}
	if len(a) != s.M*s.K*bf16Size || len(s.out.bytes) != s.M*s.N*bf16Size || len(s.params.bytes) != 72 {
		return nil, nil, nil, errMatMulBF16SteelScratchDim
	}
	aBuf, err := s.a.copyBuffer(a)
	if err != nil {
		return nil, nil, nil, err
	}
	if !s.paramsFilled {
		tn, tm := (s.N+t.bn-1)/t.bn, (s.M+t.bm-1)/t.bm
		fillMatMulBF16SteelParams(s.params.bytes, s.M, s.K, s.N, tn, tm, s.K/t.bk)
		s.paramsFilled = true
	}
	return aBuf, s.out.buf, s.params.buf, nil
}

func fillMatMulBF16SteelParams(params []byte, M, K, N, tilesN, tilesM, kIterations int) {
	for i := range params {
		params[i] = 0
	}
	putI32 := func(off int, v int32) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, int32(M))
	putI32(4, int32(N))
	putI32(8, int32(K))
	putI32(12, int32(K)) // lda
	putI32(16, int32(K)) // ldb (nt)
	putI32(20, int32(N)) // ldd
	putI32(24, int32(tilesN))
	putI32(28, int32(tilesM))
	putI32(56, 0)
	putI32(60, int32(kIterations)) // gemm_k_iterations_aligned
	putI32(64, 1)                  // batch_ndim
}

// MatMulBF16NT computes out[M,N] = a[M,K] @ w[N,K]ᵀ (w row-major [N,K]) in bf16 via the fused steel
// GEMM — byte-identical to MatRowsBF16(w, a, M, N, K) (the per-row gemv) but streaming w once. All raw
// bf16 bytes. This is the projection primitive the MTP batched verify uses to project K draft rows in
// one weight pass.
func MatMulBF16NT(a, w []byte, M, K, N int) ([]byte, error) {
	return MatMulBF16NTInto(nil, a, w, M, K, N)
}

func MatMulBF16NTInto(out []byte, a, w []byte, M, K, N int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != M*K*bf16Size || len(w) != N*K*bf16Size {
		return nil, core.NewError("native.MatMulBF16NT: a must be [M,K] and w [N,K] bf16 bytes")
	}
	outLen := M * N * bf16Size
	if M == 0 || N == 0 || K == 0 {
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	t := bf16SteelNT
	alignM, alignN, alignK := M%t.bm == 0, N%t.bn == 0, K%t.bk == 0
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		pso, err := steelGemmPipeline(t.name, false, false, false, alignM, alignN, alignK)
		if err != nil {
			encErr = err
			return
		}
		tn, tm := (N+t.bn-1)/t.bn, (M+t.bm-1)/t.bm

		scratch, err := getMatMulBF16SteelScratch(M, K, N)
		if err != nil {
			encErr = err
			return
		}
		defer putMatMulBF16SteelScratch(scratch)
		aBuf, outBuf, paramsBuf, err := scratch.buffers(a, t)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		wBuf := residentBytes(w)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSteelGemm(encSink{enc}, pso, aBuf, wBuf, outBuf, paramsBuf, tn, tm, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	return out, encErr
}
