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
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
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
	if os.Getenv("CROSS_12B_NO_ICB") != "" { // discriminator: suppress the ICB before the session builds
		prev := icbDisabledForTest
		icbDisabledForTest = true
		defer func() { icbDisabledForTest = prev }()
	}
	nm, err := LoadTokenModelDir(dir, maxLen)
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
	if os.Getenv("CROSS_12B_NO_ICB") != "" { // discriminator: force Step onto the re-encode stepToken lane
		if as, ok := ns.(*ArchSession); ok && as.state.icb != nil {
			as.state.icb = nil
			t.Log("ICB replay disabled — Step rides stepToken (re-encode)")
		}
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

// TestArchQuantSession12BFeatureMixMatchesBF16 reproduces #254 at synthetic scale: the 12B's
// quant-lane feature mix — K==V global layers (no v_proj), per-attention-type KV head counts
// AND head dims (sliding 4×32, global 1×64), the proportional global rope — stepped through
// NewArchQuantSession against the SAME weights DEQUANTISED through the bf16 session (the
// metal-parity-proven lane). Both sessions consume identical embedding bytes, so the layer
// stack is the only variable; identical weights mean a healthy quant lane tracks the bf16 lane
// to kernel-rounding (cosine ≈ 1). The 12B bug shows as cosine collapse; the ablation subtests
// name the feature that breaks it.
func TestArchQuantSession12BFeatureMixMatchesBF16(t *testing.T) {
	requireNativeRuntime(t)
	const (
		maxLen, window = 16, 4
		nLayers        = 3
		bits           = 4
		vocab          = 32
	)
	mkF := func(n, salt int) []float32 { return syntheticFloat32(n, salt) }

	run := func(t *testing.T, keqv bool, dModel, nHeads, hdSliding, kvSliding, gHD, gKV, dFF, gs int) {
		t.Helper()
		deq := func(w QuantWeight, rows, cols int) []byte {
			f, err := dequantizeAffineRowsF32(w.Packed, w.Scales, w.Biases, rows, cols, gs, bits)
			if err != nil {
				t.Fatalf("dequant: %v", err)
			}
			return toBF16Bytes(f)
		}
		cfg := g4.Config{
			HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
			NumAttentionHeads: nHeads, NumKeyValueHeads: kvSliding, HeadDim: hdSliding,
			GlobalHeadDim: gHD, NumGlobalKeyValueHeads: gKV,
			VocabSize: vocab, RMSNormEps: 1e-5,
			SlidingWindow: window,
			LayerTypes:    []string{"sliding_attention", "full_attention", "sliding_attention"},
			AttentionKEqV: keqv,
			RopeParameters: map[string]g4.RopeParam{
				"full_attention":    {RopeTheta: 1e6, PartialRotaryFactor: 0.25, RopeType: "proportional"},
				"sliding_attention": {RopeTheta: 1e4, RopeType: "default"},
			},
		}
		arch, err := cfg.Arch()
		if err != nil {
			t.Fatalf("Arch: %v", err)
		}

		qLayers := make([]QuantizedLayerWeights, nLayers)
		bLayers := make([]DecodeLayerWeights, nLayers)
		for li := range qLayers {
			spec := arch.Layer[li]
			hd, kv := spec.HeadDim, spec.KVHeads
			qDim, kvDim := nHeads*hd, kv*hd
			salt := (li + 1) * 100
			ql := QuantizedLayerWeights{
				AttnNormW: toBF16Bytes(mkF(dModel, salt+13)),
				MLPNormW:  toBF16Bytes(mkF(dModel, salt+19)),
				Q:         quantW(t, mkF(qDim*dModel, salt+53), qDim, dModel, gs, bits),
				K:         quantW(t, mkF(kvDim*dModel, salt+71), kvDim, dModel, gs, bits),
				O:         quantW(t, mkF(dModel*qDim, salt+17), dModel, qDim, gs, bits),
				Gate:      quantW(t, mkF(dFF*dModel, salt+61), dFF, dModel, gs, bits),
				Up:        quantW(t, mkF(dFF*dModel, salt+29), dFF, dModel, gs, bits),
				Down:      quantW(t, mkF(dModel*dFF, salt+47), dModel, dFF, gs, bits),
				QNormW:    toBF16Bytes(mkF(hd, salt+81)),
				KNormW:    toBF16Bytes(mkF(hd, salt+87)),
				GroupSize: gs, Bits: bits,
			}
			keqvLayer := keqv && spec.Attention == model.GlobalAttention
			if !keqvLayer {
				ql.V = quantW(t, mkF(kvDim*dModel, salt+83), kvDim, dModel, gs, bits)
			}
			qLayers[li] = ql
			bl := DecodeLayerWeights{
				AttnNormW: ql.AttnNormW, MLPNormW: ql.MLPNormW,
				WQ: deq(ql.Q, qDim, dModel), WK: deq(ql.K, kvDim, dModel),
				WO: deq(ql.O, dModel, qDim), WGate: deq(ql.Gate, dFF, dModel),
				WUp: deq(ql.Up, dFF, dModel), WDown: deq(ql.Down, dModel, dFF),
				QNormW: ql.QNormW, KNormW: ql.KNormW,
			}
			if !keqvLayer {
				bl.WV = deq(ql.V, kvDim, dModel)
			}
			bLayers[li] = bl
		}

		embedF := mkF(vocab*dModel, 21)
		qm := &QuantModel{
			Layers: qLayers, FinalNorm: toBF16Bytes(mkF(dModel, 22)), Tied: true,
			GroupSize: gs, Bits: bits,
		}
		eq := quantW(t, embedF, vocab, dModel, gs, bits)
		qm.Embed, qm.EmbedScales, qm.EmbedBiases = eq.Packed, eq.Scales, eq.Biases
		bm := &BF16Model{
			Layers: bLayers, Embed: toBF16Bytes(embedF),
			FinalNorm: qm.FinalNorm, LMHead: toBF16Bytes(embedF), Tied: true,
		}

		qs, err := NewArchQuantSession(qm, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchQuantSession: %v", err)
		}
		defer qs.Close()
		bs, err := NewArchSession(bm, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		defer bs.Close()

		worst, worstPos := 2.0, -1
		for i := 0; i < 8; i++ {
			emb := toBF16Bytes(mkF(dModel, 500+i))
			qh, err := qs.Step(emb)
			if err != nil {
				t.Fatalf("quant step %d: %v", i, err)
			}
			bh, err := bs.Step(emb)
			if err != nil {
				t.Fatalf("bf16 step %d: %v", i, err)
			}
			c := cosineBF16(qh, bh)
			t.Logf("pos %d cosine=%.5f", i, c)
			if c < worst {
				worst, worstPos = c, i
			}
		}
		if worst < 0.99 {
			t.Fatalf("quant lane diverges from the dequantised bf16 oracle: worst cosine %.5f at pos %d", worst, worstPos)
		}
	}

	t.Run("12B feature mix", func(t *testing.T) { run(t, true, 128, 4, 64, 4, 128, 1, 256, 32) })
	t.Run("no k_eq_v", func(t *testing.T) { run(t, false, 128, 4, 64, 4, 128, 1, 256, 32) })
	t.Run("uniform kv heads", func(t *testing.T) { run(t, true, 128, 4, 64, 4, 128, 4, 256, 32) })
	t.Run("uniform head dim", func(t *testing.T) { run(t, true, 128, 4, 64, 4, 64, 1, 256, 32) })
	// the REAL 12B geometry: dModel 3840 (non-fast bf16 qmv), gs 64, hd 256/512, kv 8/1, dFF 15360
	t.Run("12B real dims", func(t *testing.T) { run(t, true, 3840, 16, 256, 8, 512, 1, 15360, 64) })
}

// TestCrossEngine12BArchDump prints the arch the REGISTERED loader derives from the real 12B
// checkpoint — the config-vs-derivation audit (set CROSS_12B_DIR).
func TestCrossEngine12BArchDump(t *testing.T) {
	dir := os.Getenv("CROSS_12B_DIR")
	if dir == "" {
		t.Skip("set CROSS_12B_DIR")
	}
	nm, err := LoadTokenModelDir(dir, 64)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tm, ok := nm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("not a NativeTokenModel")
	}
	a := tm.arch
	t.Logf("Hidden=%d Heads=%d KVHeads=%d HeadDim=%d GlobalHeadDim=%d GlobalKVHeads=%d FF=%d Vocab=%d",
		a.Hidden, a.Heads, a.KVHeads, a.HeadDim, a.GlobalHeadDim, a.GlobalKVHeads, a.FF, a.Vocab)
	t.Logf("Eps=%g AttnScale=%g SoftCap=%g SlidingWindow=%d ValueNorm=%v AttentionKEqV=%v",
		a.Eps, a.AttnScale, a.SoftCap, a.SlidingWindow, a.ValueNorm, a.AttentionKEqV)
	t.Logf("RopeBase=%g RopeLocalBase=%g RopeScale=%g RotaryDim=%d RotaryDimLocal=%d",
		a.RopeBase, a.RopeLocalBase, a.RopeScale, a.RotaryDim, a.RotaryDimLocal)
	for i := 0; i < 8 && i < len(a.Layer); i++ {
		sp := a.Layer[i]
		t.Logf("L%d attn=%v headDim=%d kvHeads=%d moe=%v shareFrom=%d", i, sp.Attention, sp.HeadDim, sp.KVHeads, sp.MoE, sp.KVShareFrom)
	}
	q := tm.quant
	if q != nil && len(q.Layers) > 5 {
		l0, l5 := q.Layers[0], q.Layers[5]
		t.Logf("L0 V present=%v qnorm=%d knorm=%d scalar=%d postAttn=%d postFF=%d dff=%d",
			l0.V.Packed != nil, len(l0.QNormW), len(l0.KNormW), len(l0.LayerScalarW), len(l0.PostAttnNormW), len(l0.PostFFNormW), l0.DFF)
		t.Logf("L5 V present=%v qnorm=%d knorm=%d scalar=%d postAttn=%d postFF=%d dff=%d",
			l5.V.Packed != nil, len(l5.QNormW), len(l5.KNormW), len(l5.LayerScalarW), len(l5.PostAttnNormW), len(l5.PostFFNormW), l5.DFF)
		t.Logf("L0 Q gs=%d bits=%d packed=%d scales=%d", l0.Q.GroupSize, l0.Q.Bits, len(l0.Q.Packed), len(l0.Q.Scales))
		t.Logf("L5 Q gs=%d bits=%d packed=%d scales=%d", l5.Q.GroupSize, l5.Q.Bits, len(l5.Q.Packed), len(l5.Q.Scales))
	}
}

// TestCrossEngine12BWeightAudit compares the WEIGHT VALUES both engines hold after loading the
// same real 12B checkpoint — the load-vs-semantics discriminator for the pos-0 hidden divergence
// (hidCos −0.38 at pos 0 while embCos=1.0). At pos 0 attention output ≡ V, so the divergence is in
// projections/norms/mapping, and the first question is whether the two engines even loaded the
// same numbers into the same slots. Per audited layer: the 7 quant projections (first 4 rows,
// dequantised both sides), each native norm slot scored against ALL FOUR metal layer norms (a
// slot-swap shows as a cross-match), Q/K norms raw + (1+w)-scaled, and the layer scalar values.
// No Step involved — pure load audit. Set CROSS_12B_DIR.
func TestCrossEngine12BWeightAudit(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("CROSS_12B_DIR")
	if dir == "" {
		t.Skip("set CROSS_12B_DIR")
	}
	nm, err := LoadTokenModelDir(dir, 64)
	if err != nil {
		t.Fatalf("native load: %v", err)
	}
	tm, ok := nm.(*NativeTokenModel)
	if !ok || tm.quant == nil {
		t.Fatalf("native model is not a quant NativeTokenModel")
	}
	mm, err := g4metal.LoadGemma4(dir)
	if err != nil {
		t.Fatalf("metal load: %v", err)
	}

	cosF32 := func(a, b []float32) float64 {
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		var dot, na, nb float64
		for i := 0; i < n; i++ {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		if na == 0 || nb == 0 {
			return 0
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb))
	}
	bf16F32 := func(w []byte) []float32 {
		out := make([]float32, len(w)/2)
		for i := range out {
			out[i] = bf16ToF32(w[2*i], w[2*i+1])
		}
		return out
	}
	const auditRows = 4
	// native: dequantise the first auditRows rows of a quant tensor
	deqN := func(w QuantWeight, cols int) []float32 {
		if len(w.Packed) == 0 {
			return nil
		}
		gs, bits := w.GroupSize, w.Bits
		rowPacked := cols * bits / 8
		rowSB := (cols / gs) * 2
		f, err := dequantizeAffineRowsF32(w.Packed[:auditRows*rowPacked], w.Scales[:auditRows*rowSB], w.Biases[:auditRows*rowSB], auditRows, cols, gs, bits)
		if err != nil {
			t.Fatalf("native dequant: %v", err)
		}
		return f
	}
	// metal: slice the first auditRows rows on-device, dequantise, read back
	deqM := func(lin *mc.Linear) []float32 {
		if lin == nil || lin.Weight == nil || !lin.Weight.Valid() {
			return nil
		}
		if lin.Scales == nil || !lin.Scales.Valid() {
			return lin.Weight.Floats()[:0] // bf16 linear unexpected in a 4-bit checkpoint
		}
		w := mc.SliceAxis(lin.Weight, 0, 0, auditRows)
		s := mc.SliceAxis(lin.Scales, 0, 0, auditRows)
		b := mc.SliceAxis(lin.Biases, 0, 0, auditRows)
		d := mc.Dequantize(w, s, b, lin.GroupSize, lin.Bits)
		out := d.Floats()
		mc.Free(w, s, b, d)
		return out
	}
	cmpProj := func(li int, name string, nw QuantWeight, cols int, lin *mc.Linear) {
		nf, mf := deqN(nw, cols), deqM(lin)
		switch {
		case nf == nil && mf == nil:
			t.Logf("L%02d %-5s ABSENT both sides", li, name)
		case nf == nil || mf == nil:
			t.Logf("L%02d %-5s PRESENCE MISMATCH native=%v metal=%v  <-- SLOT BUG", li, name, nf != nil, mf != nil)
		default:
			c := cosF32(nf, mf)
			flag := ""
			if c < 0.9999 {
				flag = "  <-- VALUE MISMATCH"
			}
			t.Logf("L%02d %-5s rows0..%d cosine=%.6f (n0=%.4f m0=%.4f n1=%.4f m1=%.4f)%s", li, name, auditRows-1, c, nf[0], mf[0], nf[1], mf[1], flag)
		}
	}

	layers := []int{0, 1, 2, 3, 4, 5, 6, 47}
	for _, li := range layers {
		if li >= len(tm.quant.Layers) || li >= len(mm.Layers) {
			continue
		}
		l := tm.quant.Layers[li]
		ml := mm.Layers[li]
		sp := tm.arch.Layer[li]
		hd, kv := sp.HeadDim, sp.KVHeads
		if hd == 0 {
			hd = tm.arch.HeadDim
		}
		if kv == 0 {
			kv = tm.arch.KVHeads
		}
		dModel := tm.arch.Hidden
		dff := l.DFF
		if dff == 0 {
			dff = tm.arch.FF
		}
		t.Logf("=== L%02d attn=%v hd=%d kv=%d dff=%d (metal sliding=%v keqv=%v) ===", li, sp.Attention, hd, kv, dff, ml.IsSliding, ml.Attention.UseKEqV)
		cmpProj(li, "Q", l.Q, dModel, ml.Attention.QProj)
		cmpProj(li, "K", l.K, dModel, ml.Attention.KProj)
		cmpProj(li, "V", l.V, dModel, ml.Attention.VProj)
		cmpProj(li, "O", l.O, tm.arch.Heads*hd, ml.Attention.OProj)
		cmpProj(li, "Gate", l.Gate, dModel, ml.MLP.GateProj)
		cmpProj(li, "Up", l.Up, dModel, ml.MLP.UpProj)
		cmpProj(li, "Down", l.Down, dff, ml.MLP.DownProj)

		// norm-slot mapping audit: each native slot vs all four metal layer norms (raw weights)
		metalNorms := []struct {
			name string
			m    *mc.RMSNormModule
		}{
			{"input", ml.InputNorm}, {"postAttn", ml.PostAttnNorm}, {"preFF", ml.PreFFNorm}, {"postFF", ml.PostFFNorm},
		}
		metalF := make([][]float32, len(metalNorms))
		for i, mn := range metalNorms {
			if mn.m != nil && mn.m.Weight != nil && mn.m.Weight.Valid() {
				metalF[i] = mn.m.Weight.Floats()
			}
		}
		for _, ns := range []struct {
			name string
			w    []byte
		}{
			{"AttnNormW", l.AttnNormW}, {"PostAttnNormW", l.PostAttnNormW}, {"MLPNormW", l.MLPNormW}, {"PostFFNormW", l.PostFFNormW},
		} {
			if len(ns.w) == 0 {
				t.Logf("L%02d %-14s nil on native", li, ns.name)
				continue
			}
			nf := bf16F32(ns.w)
			best, bestName := -2.0, "?"
			for i, mf := range metalF {
				if mf == nil {
					continue
				}
				if c := cosF32(nf, mf); c > best {
					best, bestName = c, metalNorms[i].name
				}
			}
			t.Logf("L%02d %-14s best-match metal %-8s cosine=%.6f (n0=%.4f)", li, ns.name, bestName, best, nf[0])
		}
		// q/k norms: raw + scaled
		att := ml.Attention
		for _, qk := range []struct {
			name   string
			w      []byte
			raw    *mc.RMSNormModule
			scaled *mc.Array
		}{
			{"QNormW", l.QNormW, att.QNorm, att.QNormScaled}, {"KNormW", l.KNormW, att.KNorm, att.KNormScaled},
		} {
			if len(qk.w) == 0 {
				t.Logf("L%02d %-14s nil on native", li, qk.name)
				continue
			}
			nf := bf16F32(qk.w)
			rc, sc := -2.0, -2.0
			if qk.raw != nil && qk.raw.Weight != nil && qk.raw.Weight.Valid() {
				rc = cosF32(nf, qk.raw.Weight.Floats())
			}
			if qk.scaled != nil && qk.scaled.Valid() {
				sc = cosF32(nf, qk.scaled.Floats())
			}
			t.Logf("L%02d %-14s vs raw=%.6f vs scaled=%.6f (n0=%.4f)", li, qk.name, rc, sc, nf[0])
		}
		// layer scalar values
		if len(l.LayerScalarW) > 0 {
			nf := bf16F32(l.LayerScalarW)
			mv := []float32{}
			if ml.LayerScalar != nil && ml.LayerScalar.Valid() {
				mv = ml.LayerScalar.Floats()
			}
			t.Logf("L%02d LayerScalar native=%v metal(first4)=%v", li, nf, mv[:min(4, len(mv))])
		}
	}
	// model-level final norm
	if mm.Norm != nil && mm.Norm.Weight != nil && mm.Norm.Weight.Valid() && len(tm.quant.FinalNorm) > 0 {
		t.Logf("FinalNorm cosine vs metal raw=%.6f", cosF32(bf16F32(tm.quant.FinalNorm), mm.Norm.Weight.Floats()))
	}
}
