// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// stepFixture builds synthetic bf16 inputs for the KV-cache decode step. GQA is
// exercised (nHeads=8, nKVHeads=4 → factor 2). The seq-major caches are filled
// with synthetic rows 0..maxLen-1; the step overwrites row `pos`.
func stepFixture(pos, maxLen int) (x, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKV, headDim, dFF int, base, scale, eps float32) {
	dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	base, scale, eps = 10000, 0.125, 1e-5
	qDim, kvDim := nHeads*headDim, nKV*headDim
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
	wK = toBF16Bytes(mk(kvDim*dModel, 71))
	wV = toBF16Bytes(mk(kvDim*dModel, 83))
	wO = toBF16Bytes(mk(dModel*qDim, 17))
	kCache = toBF16Bytes(mk(maxLen*kvDim, 23))
	vCache = toBF16Bytes(mk(maxLen*kvDim, 41))
	mlpNormW = toBF16Bytes(mk(dModel, 19))
	wGate = toBF16Bytes(mk(dFF*dModel, 61))
	wUp = toBF16Bytes(mk(dFF*dModel, 29))
	wDown = toBF16Bytes(mk(dModel*dFF, 47))
	return
}

// seqToHeadMajor re-lays a seq-major KV cache [seq, nKV, headDim] (the layout the
// decode step appends into) into head-major [nKV, L, headDim] (the layout the
// proven exported SDPA expects), over the live window L=pos+1.
func seqToHeadMajor(seqMajor []byte, nKV, headDim, L int) []byte {
	kvDim := nKV * headDim
	hm := make([]byte, nKV*L*headDim*bf16Size)
	rb := headDim * bf16Size
	for h := 0; h < nKV; h++ {
		for i := 0; i < L; i++ {
			src := (i*kvDim + h*headDim) * bf16Size
			dst := ((h*L + i) * headDim) * bf16Size
			copy(hm[dst:dst+rb], seqMajor[src:src+rb])
		}
	}
	return hm
}

// TestAttentionStepKV gates the new cache-write half against the parity-proven
// ops. It checks BOTH halves of the mechanism: (1) the grown seq-major cache rows
// equal the proven RoPE(Wk·rms(x)) / Wv·rms(x) placed at row pos, and (2) the
// attention output over that grown window equals the proven exported (head-major)
// SDPA on the same logical rows — so the seq-major append AND the seq-major stride
// path are both validated against the proven path, not just timed.
func TestAttentionStepKV(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const pos, maxLen = 5, 8
	x, anw, wQ, wK, wV, wO, kCache, vCache, _, _, _, _, dModel, nHeads, nKV, headDim, _, base, scale, eps := stepFixture(pos, maxLen)
	qDim, kvDim := nHeads*headDim, nKV*headDim
	L := pos + 1

	initK := append([]byte(nil), kCache...)
	initV := append([]byte(nil), vCache...)

	// run the step (grows kCache/vCache at row pos, returns x + Wo·attn)
	got, err := AttentionStepKV(x, anw, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps)
	if err != nil {
		t.Fatalf("AttentionStepKV: %v", err)
	}

	// reference K/V row from proven ops
	normed, err := RMSNormBF16(x, anw, 1, dModel, eps)
	if err != nil {
		t.Fatalf("rms: %v", err)
	}
	kProj, err := MatVecBF16(wK, normed, kvDim, dModel)
	if err != nil {
		t.Fatalf("wK: %v", err)
	}
	kNew, err := RoPEBF16(kProj, 1, nKV, headDim, base, scale, pos, false)
	if err != nil {
		t.Fatalf("rope k: %v", err)
	}
	vNew, err := MatVecBF16(wV, normed, kvDim, dModel)
	if err != nil {
		t.Fatalf("wV: %v", err)
	}

	// (1) the grown caches == initial with row pos replaced by kNew/vNew
	rowBytes := kvDim * bf16Size
	expK := append([]byte(nil), initK...)
	copy(expK[pos*rowBytes:(pos+1)*rowBytes], kNew)
	expV := append([]byte(nil), initV...)
	copy(expV[pos*rowBytes:(pos+1)*rowBytes], vNew)
	eqBytes(t, "kCache append", kCache, expK)
	eqBytes(t, "vCache append", vCache, expV)

	// (2) attention output == proven head-major SDPA on the same window
	q, err := MatVecBF16(wQ, normed, qDim, dModel)
	if err != nil {
		t.Fatalf("wQ: %v", err)
	}
	qr, err := RoPEBF16(q, 1, nHeads, headDim, base, scale, pos, false)
	if err != nil {
		t.Fatalf("rope q: %v", err)
	}
	kHM := seqToHeadMajor(expK, nKV, headDim, L)
	vHM := seqToHeadMajor(expV, nKV, headDim, L)
	attn, err := SDPA(qr, kHM, vHM, 1, nHeads, nKV, headDim, L, scale)
	if err != nil {
		t.Fatalf("SDPA ref: %v", err)
	}
	attnOut, err := MatVecBF16(wO, attn, dModel, qDim)
	if err != nil {
		t.Fatalf("wO: %v", err)
	}
	want, err := AddBF16(x, attnOut)
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	eqBytes(t, "AttentionStepKV out", got, want)
	t.Logf("AttentionStepKV(pos=%d, GQA %d/%d): cache append + grown-window attention byte-identical to proven ops", pos, nHeads, nKV)
}

func TestAttentionStepKVAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	kvDim := nKV * headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	kWarm := append([]byte(nil), kCache...)
	vWarm := append([]byte(nil), vCache...)
	if _, err := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kWarm, vWarm, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps); err != nil {
		t.Fatalf("AttentionStepKV warmup: %v", err)
	}

	var stepErr error
	allocs := testing.AllocsPerRun(5, func() {
		kc := append([]byte(nil), kCache...)
		vc := append([]byte(nil), vCache...)
		_, stepErr = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps)
	})
	if stepErr != nil {
		t.Fatalf("AttentionStepKV: %v", stepErr)
	}
	if allocs > 45 {
		t.Fatalf("AttentionStepKV allocations = %.0f, want <= 45", allocs)
	}
}

func TestDecodeStepAttentionScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small := getAttnScratch(96, 96, 48, 3, 6)
	putAttnScratch(small)
	large := getAttnScratch(160, 160, 80, 5, 10)
	putAttnScratch(large)

	gotSmall := getAttnScratch(96, 96, 48, 3, 6)
	defer putAttnScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("decode-step attention scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge := getAttnScratch(160, 160, 80, 5, 10)
	defer putAttnScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("decode-step attention scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestDecodeStepMLPScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small := getMLPScratch(96, 192)
	putMLPScratch(small)
	large := getMLPScratch(160, 320)
	putMLPScratch(large)

	gotSmall := getMLPScratch(96, 192)
	defer putMLPScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("decode-step MLP scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge := getMLPScratch(160, 320)
	defer putMLPScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("decode-step MLP scratch pool evicted the large scratch after reusing the small scratch")
	}
}

// TestDecodeStepKV gates the full real decode step: out == the proven MLP block
// fed the attention-half output, and the grown caches match. AttentionStepKV is
// already gated against proven ops above, MLPBlockBF16 is parity-proven, so this
// anchors the full step (attention-with-KV + MLP) to the proven path.
func TestDecodeStepKV(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const pos, maxLen = 5, 8
	x, anw, wQ, wK, wV, wO, kCache, vCache, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, dFF, base, scale, eps := stepFixture(pos, maxLen)

	// reference: attention half (on a cache copy) then the proven MLP block
	kRef := append([]byte(nil), kCache...)
	vRef := append([]byte(nil), vCache...)
	attnOut, err := AttentionStepKV(x, anw, wQ, wK, wV, wO, kRef, vRef, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps)
	if err != nil {
		t.Fatalf("AttentionStepKV ref: %v", err)
	}
	want, err := MLPBlockBF16(attnOut, mnw, wG, wU, wD, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBF16 ref: %v", err)
	}

	// the full step on a fresh cache copy
	kGot := append([]byte(nil), kCache...)
	vGot := append([]byte(nil), vCache...)
	got, err := DecodeStepKV(x, anw, wQ, wK, wV, wO, kGot, vGot, mnw, wG, wU, wD, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeStepKV: %v", err)
	}
	eqBytes(t, "DecodeStepKV out", got, want)
	eqBytes(t, "DecodeStepKV kCache", kGot, kRef)
	eqBytes(t, "DecodeStepKV vCache", vGot, vRef)
	t.Logf("DecodeStepKV(pos=%d): full real layer == AttentionStepKV ▸ proven MLPBlockBF16 (byte-identical), cache grown", pos)
}

func TestDecodeStepKVAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	kvDim := nKV * headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	kWarm := append([]byte(nil), kCache...)
	vWarm := append([]byte(nil), vCache...)
	if _, err := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kWarm, vWarm, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
		t.Fatalf("DecodeStepKV warmup: %v", err)
	}

	var stepErr error
	allocs := testing.AllocsPerRun(5, func() {
		kc := append([]byte(nil), kCache...)
		vc := append([]byte(nil), vCache...)
		_, stepErr = DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps)
	})
	if stepErr != nil {
		t.Fatalf("DecodeStepKV: %v", stepErr)
	}
	if allocs > 45 {
		t.Fatalf("DecodeStepKV allocations = %.0f, want <= 45", allocs)
	}
}
