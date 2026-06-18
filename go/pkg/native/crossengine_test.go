// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
	g4metal "dappco.re/go/mlx/pkg/metal/model/gemma4"
	"dappco.re/go/mlx/pkg/model"
)

// TestCrossEngine12BPerStep localises the native 12B decode bug against the
// TRUSTED metal engine (metal 12B is coherent). It loads the SAME checkpoint into
// both, opens an incremental decode session on each (the exact path the bug lives
// in), and steps a fixed token sequence through both — comparing, per position,
// the EMBEDDING cosine (native.Embed vs metal.Embed) and the output-HIDDEN cosine
// (native.Step vs metal.Step). Separating the two isolates the cause: if embCos≈1
// but hidCos drops, the divergence is in the decode step (attention/cache), not the
// embed. The first low-hidCos position is where native departs from metal — the
// position the #3 fix targets. Tokens are arbitrary-but-fixed: the bug is in the
// cache/attention mechanism, not token-specific. Real-model load (functional gate,
// Snider-cleared); set CROSS_12B_DIR to the 4bit snapshot.
func TestCrossEngine12BPerStep(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("CROSS_12B_DIR")
	if dir == "" {
		t.Skip("set CROSS_12B_DIR to the gemma-4-12B-it-4bit snapshot")
	}
	const maxLen = 64
	nm, err := LoadGemma4TokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("native load: %v", err)
	}
	sm, ok := nm.(model.SessionModel)
	if !ok {
		t.Fatalf("native model is not a SessionModel")
	}
	ns, err := sm.OpenSession()
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	mm, err := g4metal.LoadGemma4(dir)
	if err != nil {
		t.Fatalf("metal load: %v", err)
	}
	mb, err := g4metal.NewBackend(mm)
	if err != nil {
		t.Fatalf("metal backend: %v", err)
	}
	ms, err := mb.OpenSession()
	if err != nil {
		t.Fatalf("metal session: %v", err)
	}

	ids := make([]int32, 40)
	for i := range ids {
		ids[i] = int32(1000 + i*97) // spread across the vocab, all < 262144
	}
	const captureStep = 6 // a position with meaningful divergence (cross-engine showed hidCos ~0.96 here)
	firstEmb, firstHid := -1, -1
	for i, id := range ids {
		ne, err := nm.Embed(id)
		if err != nil {
			t.Fatalf("native embed pos %d: %v", i, err)
		}
		me, err := mb.Embed(id)
		if err != nil {
			t.Fatalf("metal embed pos %d: %v", i, err)
		}
		ec := cosineBF16(ne, me)
		if i == captureStep { // arm per-layer capture on both engines for this step
			capturedLayerHiddens = nil
			capturedAttnHiddens = nil
			captureLayerHiddens = true
			g4metal.CaptureLayerHiddens(true)
		}
		nh, err := ns.Step(ne)
		if err != nil {
			t.Fatalf("native step %d: %v", i, err)
		}
		mh, err := ms.Step(me)
		if err != nil {
			t.Fatalf("metal step %d: %v", i, err)
		}
		hc := cosineBF16(nh, mh)
		if i == captureStep { // read both engines' per-layer hiddens for this step + diff
			captureLayerHiddens = false
			nl, na := capturedLayerHiddens, capturedAttnHiddens
			ml, ma := g4metal.CapturedLayerHiddens(), g4metal.CapturedAttnHiddens()
			g4metal.CaptureLayerHiddens(false)
			diff := func(label string, nv, mv [][]byte) {
				n := len(nv)
				if len(mv) < n {
					n = len(mv)
				}
				t.Logf("--- %s cross-engine cosine at pos %d (native %d, metal %d) ---", label, i, len(nv), len(mv))
				worst, worstL := 2.0, -1
				for L := 0; L < n; L++ {
					c := cosineBF16(nv[L], mv[L])
					if c < worst {
						worst, worstL = c, L
					}
					t.Logf("  %s L%2d cosine=%.5f", label, L, c)
				}
				t.Logf("--- %s worst layer %d cosine=%.5f ---", label, worstL, worst)
			}
			diff("POST-ATTN", na, ma) // isolates attention (the global-layer suspect)
			diff("POST-LAYER", nl, ml)
		}
		if ec < 0.99 && firstEmb < 0 {
			firstEmb = i
		}
		if hc < 0.99 && firstHid < 0 {
			firstHid = i
		}
		flag := ""
		if hc < 0.99 {
			flag = "  <-- HIDDEN DIVERGES"
		}
		t.Logf("pos %2d  embCos=%.5f  hidCos=%.5f%s", i, ec, hc, flag)
	}
	t.Logf("first embed divergence pos: %d | first hidden divergence pos: %d", firstEmb, firstHid)
}

// TestRMSNormParity3840 checks native RMSNormBF16 vs metal RMSNorm at the 12B residual-stream
// width (dModel 3840) — the suspect for the distributed ~0.06%/layer residual (RMSNorm runs ~6×
// per layer). A divergence here means a bf16-vs-fp32 accumulation gap in the rms kernel. AX-11:
// no model load — synthetic vectors.
func TestRMSNormParity3840(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const d = 3840
	x := make([]byte, d*bf16Size)
	w := make([]byte, d*bf16Size)
	for i := 0; i < d; i++ {
		xv := f32ToBF16(float32((i%37)-18) * 0.4) // varied, hidden-state scale
		wv := f32ToBF16(0.4 + float32(i%11)*0.05)
		x[i*bf16Size], x[i*bf16Size+1] = byte(xv), byte(xv>>8)
		w[i*bf16Size], w[i*bf16Size+1] = byte(wv), byte(wv>>8)
	}
	nout, err := RMSNormBF16(x, w, 1, d, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	xArr := mc.FromRawBytes(x, []int{1, d}, mc.DTypeBFloat16)
	wArr := mc.FromRawBytes(w, []int{d}, mc.DTypeBFloat16)
	mArr := mc.RMSNorm(xArr, wArr, 1e-6)
	mc.Materialize(mArr)
	mout := append([]byte(nil), mArr.RawBytes()...)
	mc.Free(xArr, wArr, mArr)
	cos := cosineBF16(nout, mout)
	var maxRel float64
	for i := 0; i < d; i++ {
		nv := float64(bf16ToF32(nout[i*bf16Size], nout[i*bf16Size+1]))
		mv := float64(bf16ToF32(mout[i*bf16Size], mout[i*bf16Size+1]))
		if mv != 0 {
			if r := math.Abs(nv-mv) / math.Abs(mv); r > maxRel {
				maxRel = r
			}
		}
	}
	t.Logf("RMSNorm dModel=%d: cosine(native,metal)=%.6f maxRelDiff=%.4f%%", d, cos, maxRel*100)
}

// TestGeluGateMulParity checks native GeluGateMulBF16 (a ~10-op tanh-gelu composition) vs metal's
// fused GELUGateMul — a candidate for the MLP half of the distributed residual (native rounds each
// intermediate to bf16; metal fuses in fp32). AX-11: synthetic.
func TestGeluGateMulParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const d = 8192
	gate := make([]byte, d*bf16Size)
	up := make([]byte, d*bf16Size)
	gf := make([]float32, d)
	uf := make([]float32, d)
	for i := 0; i < d; i++ {
		gv := float32((i%53)-26) * 0.13
		uv := float32((i%31)-15) * 0.21
		gf[i], uf[i] = gv, uv
		g, u := f32ToBF16(gv), f32ToBF16(uv)
		gate[i*bf16Size], gate[i*bf16Size+1] = byte(g), byte(g>>8)
		up[i*bf16Size], up[i*bf16Size+1] = byte(u), byte(u>>8)
	}
	nout, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatal(err)
	}
	gA := mc.FromValues(gf, d)
	uA := mc.FromValues(uf, d)
	mArr := mc.GELUGateMul(gA, uA)
	mc.Materialize(mArr)
	mb := mc.AsType(mArr, mc.DTypeBFloat16)
	mc.Materialize(mb)
	mout := append([]byte(nil), mb.RawBytes()...)
	mc.Free(gA, uA, mArr, mb)
	t.Logf("GeluGateMul d=%d: cosine(native,metal)=%.6f", d, cosineBF16(nout, mout))
}

// TestSDPAParityGlobal checks native SDPA vs metal at the 12B GLOBAL-layer shape (16 query heads,
// 1 KV head = 16:1 GQA, headDim 512) — the existing parity test only covers 8:8 headDim-64, so the
// extreme-GQA + big-head global config is untested, and it's the prime remaining attn suspect for
// the residual. AX-11: synthetic.
func TestSDPAParityGlobal(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const b, nHeads, nKV, headDim, kvLen = 1, 16, 1, 512, 16
	scale := float32(1.0 / 22.627) // ~1/sqrt(512), keeps the softmax non-saturated for a real test
	mk := func(n, seed int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(((i*seed)%53)-26) * 0.02
		}
		return f
	}
	qA := mc.AsType(mc.FromValues(mk(b*nHeads*headDim, 37), b, nHeads, 1, headDim), mc.DTypeBFloat16)
	kA := mc.AsType(mc.FromValues(mk(b*nKV*kvLen*headDim, 53), b, nKV, kvLen, headDim), mc.DTypeBFloat16)
	vA := mc.AsType(mc.FromValues(mk(b*nKV*kvLen*headDim, 29), b, nKV, kvLen, headDim), mc.DTypeBFloat16)
	mc.Materialize(qA)
	mc.Materialize(kA)
	mc.Materialize(vA)
	res := mc.ScaledDotProductAttention(qA, kA, vA, scale, false)
	mc.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...)
	got, err := SDPA(qA.RawBytes(), kA.RawBytes(), vA.RawBytes(), b, nHeads, nKV, headDim, kvLen, scale)
	mc.Free(qA, kA, vA, res)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SDPA global(16h/1kv/512d): cosine(native,metal)=%.6f", cosineBF16(got, want))
}

// cosineBF16 is the cosine similarity of two equal-length bf16 byte vectors — 1.0
// when identical in direction, lower as they diverge (robust to the small
// numerical differences between metal's mlx-c ops and native's kernels, unlike a
// byte compare).
func cosineBF16(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i+1 < len(a); i += 2 {
		av := float64(bf16ToF32(a[i], a[i+1]))
		bv := float64(bf16ToF32(b[i], b[i+1]))
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
