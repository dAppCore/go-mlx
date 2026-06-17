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

// TestICBRebindOffset proves the cache-grow lever: an ICB command recorded once
// can have only its output buffer OFFSET re-set between replays, and each replay
// writes the new row. This is the mechanism the growing KV cache needs — the
// per-token write row advances while the rest of the command stays recorded — so
// it must hold before the real cache-grow ICB is built on it.
func TestICBRebindOffset(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const outDim, inDim, nRows = 128, 64, 4
	mat := make([]float32, outDim*inDim)
	for i := range mat {
		mat[i] = float32((i*37)%101-50) * 0.01
	}
	vec := make([]float32, inDim)
	for i := range vec {
		vec[i] = float32((i*53)%97-48) * 0.01
	}
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	got, err := rebindProbeICB(mat, vec, outDim, inDim, nRows)
	if err != nil {
		t.Fatalf("rebindProbeICB: %v", err)
	}
	for r := 0; r < nRows; r++ {
		row := got[r*outDim : (r+1)*outDim]
		for i := range want {
			if row[i] != want[i] {
				t.Fatalf("rebind row %d differs at [%d]: %v vs %v (offset re-set did not take effect)", r, i, row[i], want[i])
			}
		}
	}
	t.Logf("ICB offset rebind: %d replays each wrote its own row via SetKernelBufferOffsetAtIndex — cache-grow lever holds", nRows)
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

// tokenFixture builds the synthetic bf16 weights/KV for an nLayers-deep token
// harness — one shared set, the same dims the per-layer tests use.
func tokenFixture() (x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKV, headDim, kvLen, dFF int, base, scale float32, offset int, eps float32) {
	dModel, nHeads, nKV, headDim, kvLen, dFF = 512, 8, 8, 64, 16, 1024
	base, scale, offset, eps = 10000, 0.125, 5, 1e-5
	qDim := nHeads * headDim
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x = toBF16Bytes(mk(dModel, 37))
	attnNormW = toBF16Bytes(mk(dModel, 13))
	wQ = toBF16Bytes(mk(qDim*dModel, 53))
	wO = toBF16Bytes(mk(dModel*qDim, 17))
	kCache = toBF16Bytes(mk(nKV*kvLen*headDim, 23))
	vCache = toBF16Bytes(mk(nKV*kvLen*headDim, 41))
	mlpNormW = toBF16Bytes(mk(dModel, 19))
	wGate = toBF16Bytes(mk(dFF*dModel, 61))
	wUp = toBF16Bytes(mk(dFF*dModel, 29))
	wDown = toBF16Bytes(mk(dModel*dFF, 47))
	return
}

// TestDecodeTokenParity anchors the whole per-token stack to the proven single
// layer: an nLayers-deep DecodeTokenICB (replays=1) and the chained tokenReEncode
// must each equal nLayers applications of DecodeLayer (the mlx-c-parity-gated op)
// byte-for-byte. This verifies the OUTPUT — not just the timing — so a wrong
// cross-layer barrier or ping-pong binding fails here rather than hiding behind a
// plausible number (the ICB-replay lesson: verify output, never timing alone).
func TestDecodeTokenParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps := tokenFixture()
	const nLayers = 3

	// oracle: nLayers applications of the proven DecodeLayer, output feeding input
	ref := x
	for i := 0; i < nLayers; i++ {
		var err error
		ref, err = DecodeLayer(ref, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
		if err != nil {
			t.Fatalf("DecodeLayer[%d]: %v", i, err)
		}
	}

	reEnc, err := tokenReEncode(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("tokenReEncode: %v", err)
	}
	if len(reEnc) != len(ref) {
		t.Fatalf("tokenReEncode len %d != ref %d", len(reEnc), len(ref))
	}
	for i := range ref {
		if reEnc[i] != ref[i] {
			t.Fatalf("tokenReEncode != %d×DecodeLayer at byte %d: %#x vs %#x", nLayers, i, reEnc[i], ref[i])
		}
	}

	icbOut, err := DecodeTokenICB(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeTokenICB: %v", err)
	}
	if len(icbOut) != len(ref) {
		t.Fatalf("DecodeTokenICB len %d != ref %d", len(icbOut), len(ref))
	}
	for i := range ref {
		if icbOut[i] != ref[i] {
			t.Fatalf("DecodeTokenICB != %d×DecodeLayer at byte %d: %#x vs %#x", nLayers, i, icbOut[i], ref[i])
		}
	}
	t.Logf("%d-layer token: DecodeTokenICB == tokenReEncode == %d×DecodeLayer (byte-identical)", nLayers, nLayers)
}

// TestTokenEncodeBypass is the UN-DILUTED headline: a full decode token is its
// whole layer stack submitted with ONE commit+wait, so re-encoding all
// nLayers*21 ops per token vs replaying the recorded stack isolates the per-token
// host saving without the per-layer commit+wait that diluted TestDecodeLayer-
// EncodeBypass. Sweeping depth shows the saving (and the ratio) grow with the
// stack — the host re-encode the ICB removes from a real token's critical path.
func TestTokenEncodeBypass(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps := tokenFixture()
	const reps = 100
	for _, nLayers := range []int{1, 12, 24, 48} {
		// warm both paths at this depth (build pipelines + record the ICB)
		if _, err := tokenReEncode(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
			t.Fatalf("tokenReEncode warm: %v", err)
		}
		if _, err := DecodeTokenICB(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
			t.Fatalf("DecodeTokenICB warm: %v", err)
		}

		t0 := time.Now()
		if _, err := tokenReEncode(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, reps); err != nil {
			t.Fatalf("tokenReEncode: %v", err)
		}
		reEnc := time.Since(t0)

		t1 := time.Now()
		if _, err := DecodeTokenICB(x, anw, wQ, wO, kC, vC, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, reps); err != nil {
			t.Fatalf("DecodeTokenICB: %v", err)
		}
		icb := time.Since(t1)

		reUs := float64(reEnc.Microseconds()) / reps
		icbUs := float64(icb.Microseconds()) / reps
		t.Logf("%2d-layer token (%4d ops), %d reps: re-encode %6.1f µs/tok, ICB-replay %6.1f µs/tok, host saved %6.1f µs/tok (%.2fx)",
			nLayers, nLayers*21, reps, reUs, icbUs, reUs-icbUs, reUs/icbUs)
	}
}
