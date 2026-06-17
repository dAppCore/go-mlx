// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"
)

// toBF16Bytes converts float32 data to bf16 bytes via f32ToBF16 (round-to-even),
// for the internal measurement tests (which can't reach the native_test helper).
func toBF16Bytes(f []float32) []byte {
	b := make([]byte, len(f)*2)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*2] = byte(h)
		b[i*2+1] = byte(h >> 8)
	}
	return b
}

// TestSquareICBDebug isolates the basic ICB mechanism (one op, scalar-as-buffer,
// residency, execute) from the multi-op barrier path.
func TestSquareICBDebug(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	in := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got, err := squareICB(in)
	if err != nil {
		t.Fatalf("squareICB: %v", err)
	}
	t.Logf("squareICB = %v (want squares)", got)
	for i := range in {
		if got[i] != in[i]*in[i] {
			t.Fatalf("squareICB bad at [%d]: %v, want %v", i, got[i], in[i]*in[i])
		}
	}
}

// TestGemvICBDebug isolates gemv-in-ICB (threadgroups dispatch + 10 binds).
func TestGemvICBDebug(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const outDim, inDim = 512, 256
	mat := make([]float32, outDim*inDim)
	for i := range mat {
		mat[i] = float32((i*37)%101-50) * 0.01
	}
	vec := make([]float32, inDim)
	for i := range vec {
		vec[i] = float32((i*53)%97-48) * 0.01
	}
	got, err := gemvICB(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("gemvICB: %v", err)
	}
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	t.Logf("gemvICB[:4]=%v want[:4]=%v", got[:4], want[:4])
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gemvICB bad at [%d]: %v, want %v", i, got[i], want[i])
		}
	}
}

// TestAttentionEncodeBypass measures the host encode-bypass: re-encoding the 6-op
// attention block every rep (persistent buffers) vs replaying it from an ICB. The
// per-rep difference is the host encode the ICB skips — extrapolating to N layers
// per token. GPU + commit/wait are identical both sides, so the absolute ratio is
// diluted by GPU time; the per-rep DELTA is the honest encode-bypass number.
func TestAttentionEncodeBypass(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, kvLen = 512, 8, 8, 64, 16
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	const reps = 500
	qDim := nHeads * headDim
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := toBF16Bytes(mk(dModel, 37))
	normW := toBF16Bytes(mk(dModel, 13))
	wQ := toBF16Bytes(mk(qDim*dModel, 53))
	wO := toBF16Bytes(mk(dModel*qDim, 17))
	kCache := toBF16Bytes(mk(nKV*kvLen*headDim, 23))
	vCache := toBF16Bytes(mk(nKV*kvLen*headDim, 41))

	// warm both paths (build pipelines, etc.)
	_ = attentionReEncode(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	_, _ = AttentionBlockICB(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)

	t0 := time.Now()
	if err := attentionReEncode(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, reps); err != nil {
		t.Fatalf("attentionReEncode: %v", err)
	}
	reEnc := time.Since(t0)

	t1 := time.Now()
	if _, err := AttentionBlockICB(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, reps); err != nil {
		t.Fatalf("AttentionBlockICB: %v", err)
	}
	icb := time.Since(t1)

	reUs := float64(reEnc.Microseconds()) / reps
	icbUs := float64(icb.Microseconds()) / reps
	t.Logf("attention 6-op block, %d reps: re-encode %.1f µs/rep, ICB-replay %.1f µs/rep, host saved %.1f µs/rep (%.2fx)",
		reps, reUs, icbUs, reUs-icbUs, reUs/icbUs)
}

// TestDecodeLayerEncodeBypass measures the PER-LAYER host encode-bypass: re-encoding
// the full 21-op DecodeLayer every rep (persistent buffers) vs replaying it from an
// ICB. The per-rep difference is the host encode the ICB skips — the figure that
// scales by the model's layer count per decoded token. GPU + commit/wait are
// identical both sides, so they cancel in the delta; the per-rep DELTA is the honest
// per-layer encode-bypass number. This is the full-layer analogue of
// TestAttentionEncodeBypass.
func TestDecodeLayerEncodeBypass(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, kvLen, dFF = 512, 8, 8, 64, 16, 1024
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	const reps = 300
	qDim := nHeads * headDim
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := toBF16Bytes(mk(dModel, 37))
	attnNormW := toBF16Bytes(mk(dModel, 13))
	wQ := toBF16Bytes(mk(qDim*dModel, 53))
	wO := toBF16Bytes(mk(dModel*qDim, 17))
	kCache := toBF16Bytes(mk(nKV*kvLen*headDim, 23))
	vCache := toBF16Bytes(mk(nKV*kvLen*headDim, 41))
	mlpNormW := toBF16Bytes(mk(dModel, 19))
	wGate := toBF16Bytes(mk(dFF*dModel, 61))
	wUp := toBF16Bytes(mk(dFF*dModel, 29))
	wDown := toBF16Bytes(mk(dModel*dFF, 47))

	// warm both paths (build pipelines, etc.)
	_ = layerReEncode(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	_, _ = DecodeLayerICB(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)

	t0 := time.Now()
	if err := layerReEncode(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, reps); err != nil {
		t.Fatalf("layerReEncode: %v", err)
	}
	reEnc := time.Since(t0)

	t1 := time.Now()
	if _, err := DecodeLayerICB(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, reps); err != nil {
		t.Fatalf("DecodeLayerICB: %v", err)
	}
	icb := time.Since(t1)

	reUs := float64(reEnc.Microseconds()) / reps
	icbUs := float64(icb.Microseconds()) / reps
	t.Logf("DecodeLayer 21-op layer, %d reps: re-encode %.1f µs/rep, ICB-replay %.1f µs/rep, per-layer host saved %.1f µs/rep (%.2fx)",
		reps, reUs, icbUs, reUs-icbUs, reUs/icbUs)
}
