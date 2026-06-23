// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
	g4 "dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// audio_test.go validates the native Conformer audio tower for BYTE-IDENTITY to metal (Snider's bar).
// IMPORTANT: the audio tower runs in FP32 — its f32 gradient-clipping clamp + f32 ReLU zero promote
// the activation (see audio_f32.go) — so the TOWER path is the fp32 blocks, asserted against the REAL
// metal *.Forward in TestAudioLayer (eqF32). The TestAudio{FeedForward,LightConv,Subsample} below
// assert the DEPRECATED all-bf16 blocks against mbf-wrapped op chains (internally consistent, but the
// mbf wrap forces bf16 and hides the tower's fp32 promotion — they are NOT the tower path).
// TestAudioAttention covers the bf16-input attention entry (still byte-identical for bf16 input).

func mbf(a *mc.Array) *mc.Array         { return mc.AsType(a, mc.DTypeBFloat16) }
func marr(b []byte, s ...int) *mc.Array { return mc.FromRawBytes(b, s, mc.DTypeBFloat16) }

// TestAudioFeedForward asserts native.AudioFeedForward is BYTE-IDENTICAL to metal's
// Gemma4AudioFeedForward.Forward composed from metal ops: clamp → RMSNorm → FFW1 → SiLU → FFW2 →
// clamp → RMSNorm → ·residual → +x (the original). Every intermediate stays bf16.
func TestAudioFeedForward(t *testing.T) {
	requireNativeRuntime(t)
	const L, hidden, inter = 16, 128, 512
	eps, residual, gc := float32(1e-6), float32(0.5), float32(6.0)
	preN, postN := toBF16Bytes(syntheticFloat32(hidden, 1)), toBF16Bytes(syntheticFloat32(hidden, 2))
	ffw1, ffw2 := toBF16Bytes(syntheticFloat32(inter*hidden, 3)), toBF16Bytes(syntheticFloat32(hidden*inter, 4))
	x := toBF16Bytes(syntheticFloat32(L*hidden, 5))

	got, err := AudioFeedForward(x, &AudioFeedForwardWeights{PreNorm: preN, PostNorm: postN, FFW1: ffw1, FFW2: ffw2},
		AudioConfig{Hidden: hidden, FFInter: inter, Eps: eps, Act: "silu", FFResidual: residual, ClipMin: -gc, ClipMax: gc})
	if err != nil {
		t.Fatalf("AudioFeedForward: %v", err)
	}

	gcMin, gcMax := mc.FromValue(-gc), mc.FromValue(gc)
	xa := marr(x, L, hidden)
	pre := mbf(mc.RMSNorm(mbf(mc.Clip(xa, gcMin, gcMax)), marr(preN, hidden), eps))
	up := mbf(mc.Matmul(pre, mc.Transpose(marr(ffw1, inter, hidden), 1, 0)))
	act := mbf(mc.SiLU(up))
	down := mbf(mc.Matmul(act, mc.Transpose(marr(ffw2, hidden, inter), 1, 0)))
	post := mbf(mc.RMSNorm(mbf(mc.Clip(down, gcMin, gcMax)), marr(postN, hidden), eps))
	out := mbf(mc.Add(mbf(mc.MulScalar(post, residual)), xa))
	mc.Materialize(out)

	eqBytes(t, "AudioFeedForward vs metal FF", got, append([]byte(nil), out.RawBytes()...))
}

// TestAudioLightConv asserts native.AudioLightConv is BYTE-IDENTICAL to metal's
// Gemma4AudioLightConv.Forward composed from metal ops: RMSNorm → LinearStart → GLU (gate·σ(gateIn))
// → PadAxis+Conv1d (causal depthwise) → Clip → RMSNorm → SiLU → LinearEnd → +x. eqBytes, not tolerance.
func TestAudioLightConv(t *testing.T) {
	requireNativeRuntime(t)
	const L, hidden, K = 16, 128, 5
	ch := hidden
	eps, gc := float32(1e-6), float32(6.0)
	preN, convN := toBF16Bytes(syntheticFloat32(hidden, 1)), toBF16Bytes(syntheticFloat32(ch, 2))
	lstart, lend, dw := toBF16Bytes(syntheticFloat32(2*ch*hidden, 3)), toBF16Bytes(syntheticFloat32(hidden*ch, 4)), toBF16Bytes(syntheticFloat32(ch*K, 5))
	x := toBF16Bytes(syntheticFloat32(L*hidden, 6))

	got, err := AudioLightConv(x, &AudioLightConvWeights{PreNorm: preN, ConvNorm: convN, LinearStart: lstart, LinearEnd: lend, DepthwiseWeight: dw},
		AudioConfig{Hidden: hidden, Channels: ch, KernelSize: K, Eps: eps, Act: "silu", ClipMin: -gc, ClipMax: gc})
	if err != nil {
		t.Fatalf("AudioLightConv: %v", err)
	}

	gcMin, gcMax := mc.FromValue(-gc), mc.FromValue(gc)
	xa := marr(x, L, hidden)
	pre := mbf(mc.RMSNorm(xa, marr(preN, hidden), eps))
	start := mbf(mc.Matmul(pre, mc.Transpose(marr(lstart, 2*ch, hidden), 1, 0)))
	gate := mbf(mc.SliceAxis(start, -1, 0, int32(ch)))
	gateIn := mbf(mc.SliceAxis(start, -1, int32(ch), int32(2*ch)))
	glu := mbf(mc.Mul(gate, mbf(mc.Sigmoid(gateIn))))
	padded := mbf(mc.PadAxis(mc.Reshape(glu, 1, L, int32(ch)), 1, K-1, 0))
	conv := mc.Reshape(mbf(mc.Conv1d(padded, marr(dw, ch, K, 1), 1, 0, 1, ch)), L, int32(ch))
	normed := mbf(mc.RMSNorm(mbf(mc.Clip(conv, gcMin, gcMax)), marr(convN, ch), eps))
	end := mbf(mc.Matmul(mbf(mc.SiLU(normed)), mc.Transpose(marr(lend, hidden, ch), 1, 0)))
	out := mbf(mc.Add(end, xa))
	mc.Materialize(out)

	eqBytes(t, "AudioLightConv vs metal LightConv", got, append([]byte(nil), out.RawBytes()...))
}

// TestAudioSubsample asserts native.AudioSubsample is BYTE-IDENTICAL to metal's
// Gemma4AudioSubSampleConvProjection.Forward: reshape → 2×(Conv2d 3×3 s2 p1 → LayerNorm → ReLU) →
// flatten → InputProj. eqBytes, not tolerance.
func TestAudioSubsample(t *testing.T) {
	requireNativeRuntime(t)
	const fr, mel, oc0, oc1, hid = 16, 80, 8, 8, 128
	eps := float32(1e-5)
	t0, f0 := convOut(fr), convOut(mel)
	t1, f1 := convOut(t0), convOut(f0)
	K := f1 * oc1
	conv0, n0w, n0b := toBF16Bytes(syntheticFloat32(oc0*9*1, 3)), toBF16Bytes(syntheticFloat32(oc0, 5)), toBF16Bytes(syntheticFloat32(oc0, 7))
	conv1, n1w, n1b := toBF16Bytes(syntheticFloat32(oc1*9*oc0, 9)), toBF16Bytes(syntheticFloat32(oc1, 11)), toBF16Bytes(syntheticFloat32(oc1, 13))
	ip, feat := toBF16Bytes(syntheticFloat32(hid*K, 15)), toBF16Bytes(syntheticFloat32(fr*mel, 17))

	got, err := AudioSubsample(feat, &AudioSubsampleWeights{Conv0: conv0, Norm0W: n0w, Norm0B: n0b, Conv1: conv1, Norm1W: n1w, Norm1B: n1b, InputProj: ip},
		AudioSubsampleConfig{Frames: fr, MelBins: mel, OutC0: oc0, OutC1: oc1, Hidden: hid, Eps: eps})
	if err != nil {
		t.Fatalf("AudioSubsample: %v", err)
	}

	zero := mc.FromValue(float32(0))
	c0 := mbf(mc.Conv2d(marr(feat, 1, fr, mel, 1), marr(conv0, oc0, 3, 3, 1), 2, 2, 1, 1, 1, 1, 1))
	r0 := mbf(mc.Maximum(mbf(mc.LayerNorm(c0, marr(n0w, oc0), marr(n0b, oc0), eps)), zero))
	c1 := mbf(mc.Conv2d(r0, marr(conv1, oc1, 3, 3, oc0), 2, 2, 1, 1, 1, 1, 1))
	r1 := mbf(mc.Maximum(mbf(mc.LayerNorm(c1, marr(n1w, oc1), marr(n1b, oc1), eps)), zero))
	flat := mc.Reshape(r1, int32(t1), int32(K))
	out := mbf(mc.Matmul(flat, mc.Transpose(marr(ip, hid, K), 1, 0)))
	mc.Materialize(out)

	eqBytes(t, "AudioSubsample vs metal subsample", got, append([]byte(nil), out.RawBytes()...))
}

func TestAudioSubsampleF32(t *testing.T) {
	requireNativeRuntime(t)
	const fr, mel, oc0, oc1, hid = 16, 80, 8, 8, 128
	eps := float32(1e-5)
	t0, f0 := convOut(fr), convOut(mel)
	t1, f1 := convOut(t0), convOut(f0)
	K := f1 * oc1
	conv0, n0w, n0b := toBF16Bytes(syntheticFloat32(oc0*9*1, 3)), toBF16Bytes(syntheticFloat32(oc0, 5)), toBF16Bytes(syntheticFloat32(oc0, 7))
	conv1, n1w, n1b := toBF16Bytes(syntheticFloat32(oc1*9*oc0, 9)), toBF16Bytes(syntheticFloat32(oc1, 11)), toBF16Bytes(syntheticFloat32(oc1, 13))
	ip, feat := toBF16Bytes(syntheticFloat32(hid*K, 15)), toBF16Bytes(syntheticFloat32(fr*mel, 17))

	got, err := AudioSubsampleF32(feat, &AudioSubsampleWeights{Conv0: conv0, Norm0W: n0w, Norm0B: n0b, Conv1: conv1, Norm1W: n1w, Norm1B: n1b, InputProj: ip},
		AudioSubsampleConfig{Frames: fr, MelBins: mel, OutC0: oc0, OutC1: oc1, Hidden: hid, Eps: eps})
	if err != nil {
		t.Fatalf("AudioSubsampleF32: %v", err)
	}

	zero := mc.FromValue(float32(0))
	c0 := mbf(mc.Conv2d(marr(feat, 1, fr, mel, 1), marr(conv0, oc0, 3, 3, 1), 2, 2, 1, 1, 1, 1, 1))
	r0 := mc.Maximum(mbf(mc.LayerNorm(c0, marr(n0w, oc0), marr(n0b, oc0), eps)), zero)
	c1 := mc.Conv2d(r0, marr(conv1, oc1, 3, 3, oc0), 2, 2, 1, 1, 1, 1, 1)
	r1 := mc.Maximum(mc.LayerNorm(c1, marr(n1w, oc1), marr(n1b, oc1), eps), zero)
	flat := mc.Reshape(r1, int32(t1), int32(K))
	out := mc.Matmul(flat, mc.Transpose(marr(ip, hid, K), 1, 0))
	eqF32(t, "AudioSubsampleF32 vs metal subsample", got, out)
}

func TestAudioEncodeNoLayers(t *testing.T) {
	requireNativeRuntime(t)
	const fr, mel, oc0, oc1, hid, outDim = 16, 80, 8, 8, 128, 32
	eps := float32(1e-5)
	t0, f0 := convOut(fr), convOut(mel)
	_, f1 := convOut(t0), convOut(f0)
	K := f1 * oc1
	weights := &AudioSubsampleWeights{
		Conv0:     toBF16Bytes(syntheticFloat32(oc0*9*1, 3)),
		Norm0W:    toBF16Bytes(syntheticFloat32(oc0, 5)),
		Norm0B:    toBF16Bytes(syntheticFloat32(oc0, 7)),
		Conv1:     toBF16Bytes(syntheticFloat32(oc1*9*oc0, 9)),
		Norm1W:    toBF16Bytes(syntheticFloat32(oc1, 11)),
		Norm1B:    toBF16Bytes(syntheticFloat32(oc1, 13)),
		InputProj: toBF16Bytes(syntheticFloat32(hid*K, 15)),
	}
	subCfg := AudioSubsampleConfig{Frames: fr, MelBins: mel, OutC0: oc0, OutC1: oc1, Hidden: hid, Eps: eps}
	features := toBF16Bytes(syntheticFloat32(fr*mel, 17))
	proj := toBF16Bytes(syntheticFloat32(outDim*hid, 19))

	got, err := AudioEncode(features, &AudioEncoderWeights{Subsample: weights, SubsampleC: subCfg, OutputProj: proj, OutputDim: outDim}, AudioConfig{Hidden: hid})
	if err != nil {
		t.Fatalf("AudioEncode: %v", err)
	}
	sub, err := AudioSubsampleF32(features, weights, subCfg)
	if err != nil {
		t.Fatalf("AudioSubsampleF32 reference: %v", err)
	}
	want, err := matF32MixedNT(sub, proj, len(sub)/hid, outDim, hid)
	if err != nil {
		t.Fatalf("matF32MixedNT reference: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("AudioEncode len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("AudioEncode[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestAudioAttention asserts native.AudioAttention is BYTE-IDENTICAL to metal's
// Gemma4AudioAttention.Forward (the real attention object, no-clip projections): per-dim q-scale,
// blocked-context windows, the relative-key projection (split-K), Transformer-XL relShift, tanh
// soft-cap, validity mask, fp32 softmax, and the bf16 output projection. eqBytes, not tolerance.
func TestAudioAttention(t *testing.T) {
	requireNativeRuntime(t)
	const hid, H, D, chunk, past, future, T = 128, 4, 32, 4, 2, 1, 10
	hd, P := H*D, past+1
	kscale, cap, inval := float32(0.5), float32(50), float32(-1e9)
	qw := toBF16Bytes(syntheticFloat32(hd*hid, 3))
	kw := toBF16Bytes(syntheticFloat32(hd*hid, 5))
	vw := toBF16Bytes(syntheticFloat32(hd*hid, 7))
	pw := toBF16Bytes(syntheticFloat32(hid*hd, 9))
	rkw := toBF16Bytes(syntheticFloat32(hd*hid, 11))
	qscale := syntheticFloat32(D, 13)
	pos := syntheticFloat32(P*hid, 15)
	x := toBF16Bytes(syntheticFloat32(T*hid, 17))

	got, err := AudioAttention(x, &AudioAttentionWeights{QProj: qw, KProj: kw, VProj: vw, Post: pw, RelativeKProj: rkw, QScalePerDim: qscale, PosEmbed: pos, PosCount: P},
		AudioConfig{Hidden: hid, NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: kscale, LogitCap: cap, InvalidLogit: inval})
	if err != nil {
		t.Fatalf("AudioAttention: %v", err)
	}

	cl := func(b []byte, o, i int) *g4.Gemma4AudioClippableLinear {
		return &g4.Gemma4AudioClippableLinear{Linear: mc.NewLinear(marr(b, o, i), nil)}
	}
	attn := &g4.Gemma4AudioAttention{
		QProj: cl(qw, hd, hid), KProj: cl(kw, hd, hid), VProj: cl(vw, hd, hid), Post: cl(pw, hid, hd),
		RelativeKProj: mc.NewLinear(marr(rkw, hd, hid), nil),
		QScalePerDim:  mc.FromValues(qscale, 1, 1, 1, D), PosEmbed: mc.FromValues(pos, P, hid),
		NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: kscale, LogitCap: cap, InvalidLogit: inval,
	}
	r := mc.AsType(attn.Forward(marr(x, 1, T, hid)), mc.DTypeBFloat16)
	mc.Materialize(r)
	eqBytes(t, "AudioAttention vs metal attention", got, append([]byte(nil), r.RawBytes()...))
}

// bf16Scalar makes a bf16 [1] array — a model-dtype per-linear clamp scalar (input_min etc).
func bf16Scalar(v float32) *mc.Array {
	return mc.FromRawBytes(toBF16Bytes([]float32{v}), []int{1}, mc.DTypeBFloat16)
}

// TestAudioAttentionClipped asserts AudioAttention stays BYTE-IDENTICAL when per-linear activation
// clamps are present in the model dtype (bf16): metal.Clip(bf16, bf16, bf16) stays bf16, so clampBF16
// matches. (f32 clamp arrays would promote the projection to fp32 — handled at load if found.)
func TestAudioAttentionClipped(t *testing.T) {
	requireNativeRuntime(t)
	const hid, H, D, chunk, past, future, T = 128, 4, 32, 4, 2, 1, 10
	hd, P := H*D, past+1
	kscale, cap, inval := float32(0.5), float32(50), float32(-1e9)
	qw := toBF16Bytes(syntheticFloat32(hd*hid, 3))
	kw := toBF16Bytes(syntheticFloat32(hd*hid, 5))
	vw := toBF16Bytes(syntheticFloat32(hd*hid, 7))
	pw := toBF16Bytes(syntheticFloat32(hid*hd, 9))
	rkw := toBF16Bytes(syntheticFloat32(hd*hid, 11))
	qscale := syntheticFloat32(D, 13)
	pos := syntheticFloat32(P*hid, 15)
	x := toBF16Bytes(syntheticFloat32(T*hid, 17)) // 0.05-scaled in the harness; here syntheticFloat32 range exercises clamps

	nw := &AudioAttentionWeights{QProj: qw, KProj: kw, VProj: vw, Post: pw, RelativeKProj: rkw, QScalePerDim: qscale, PosEmbed: pos, PosCount: P,
		QClip: ClipPair{In: ClipBound{Min: -2, Max: 2, Present: true}, Out: ClipBound{Min: -4, Max: 4, Present: true}},
		VClip: ClipPair{Out: ClipBound{Min: -3, Max: 3, Present: true}}}
	got, err := AudioAttention(x, nw, AudioConfig{Hidden: hid, NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: kscale, LogitCap: cap, InvalidLogit: inval})
	if err != nil {
		t.Fatalf("AudioAttention(clipped): %v", err)
	}

	cl := func(b []byte, o, i int) *g4.Gemma4AudioClippableLinear {
		return &g4.Gemma4AudioClippableLinear{Linear: mc.NewLinear(marr(b, o, i), nil)}
	}
	q, v := cl(qw, hd, hid), cl(vw, hd, hid)
	q.InputMin, q.InputMax, q.OutputMin, q.OutputMax = bf16Scalar(-2), bf16Scalar(2), bf16Scalar(-4), bf16Scalar(4)
	v.OutputMin, v.OutputMax = bf16Scalar(-3), bf16Scalar(3)
	attn := &g4.Gemma4AudioAttention{
		QProj: q, KProj: cl(kw, hd, hid), VProj: v, Post: cl(pw, hid, hd),
		RelativeKProj: mc.NewLinear(marr(rkw, hd, hid), nil),
		QScalePerDim:  mc.FromValues(qscale, 1, 1, 1, D), PosEmbed: mc.FromValues(pos, P, hid),
		NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: kscale, LogitCap: cap, InvalidLogit: inval,
	}
	r := mc.AsType(attn.Forward(marr(x, 1, T, hid)), mc.DTypeBFloat16)
	mc.Materialize(r)
	eqBytes(t, "AudioAttention(clipped) vs metal", got, append([]byte(nil), r.RawBytes()...))
}

// f32Sy is a synthetic fp32 fill scaled like the harnesses (a range that exercises the GC clamp).
func f32Sy(n, salt int, sc float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*salt+7)%101-50) * sc
	}
	return v
}

func eqF32(t *testing.T, name string, got []float32, want *mc.Array) {
	t.Helper()
	mc.Materialize(want)
	w := want.Floats()
	if len(got) != len(w) {
		t.Fatalf("%s: length %d vs %d", name, len(got), len(w))
	}
	for i := range w {
		if math.Float32bits(got[i]) != math.Float32bits(w[i]) {
			t.Fatalf("%s differs at %d: %v vs %v (not byte-identical)", name, i, got[i], w[i])
		}
	}
}

// TestAudioLayer asserts the fp32 AudioLayer is BYTE-IDENTICAL to the REAL metal Gemma4AudioLayer.
// Forward (not an mbf-composed op chain) across data scales — exercising AudioFeedForwardF32,
// AudioAttentionF32, AudioLightConvF32 and the layer's clamp/RMSNorm/residual composition. The audio
// tower runs fp32 (the f32 GC clamp promotes the activation); the all-bf16 blocks only matched
// data-dependently, which composing the REAL Forward exposes.
func TestAudioLayer(t *testing.T) {
	requireNativeRuntime(t)
	const hid, inter, ch, K, H, D, chunk, past, future, T = 128, 512, 128, 5, 4, 32, 4, 2, 1, 10
	hd, P := H*D, past+1
	eps, gc, res, ksc, cap, inval := float32(1e-6), float32(6), float32(0.5), float32(0.5), float32(50), float32(-1e9)
	cfg := AudioConfig{Hidden: hid, FFInter: inter, Channels: ch, KernelSize: K, Eps: eps, Act: "silu", FFResidual: res, ClipMin: -gc, ClipMax: gc, NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: ksc, LogitCap: cap, InvalidLogit: inval}
	gcmn, gcmx := mc.FromValue(-gc), mc.FromValue(gc)
	cl := func(b []byte, o, i int) *g4.Gemma4AudioClippableLinear {
		return &g4.Gemma4AudioClippableLinear{Linear: mc.NewLinear(marr(b, o, i), nil)}
	}
	for _, sc := range []float32{0.04, 0.09, 0.2} {
		tbS := func(n, s int) []byte { return toBF16Bytes(f32Sy(n, s, sc)) }
		f1a, f1b, pn1, pn2 := tbS(inter*hid, 3), tbS(hid*inter, 4), tbS(hid, 1), tbS(hid, 2)
		f2a, f2b, pn3, pn4 := tbS(inter*hid, 13), tbS(hid*inter, 14), tbS(hid, 11), tbS(hid, 12)
		qw, kw, vw, pw, rkw := tbS(hd*hid, 21), tbS(hd*hid, 22), tbS(hd*hid, 23), tbS(hid*hd, 24), tbS(hd*hid, 25)
		qsc, pos := f32Sy(D, 26, sc), f32Sy(P*hid, 27, sc)
		ls, le, dw, lpn, lcn := tbS(2*ch*hid, 31), tbS(hid*ch, 32), tbS(ch*K, 33), tbS(hid, 34), tbS(ch, 35)
		npa, npo, no := tbS(hid, 41), tbS(hid, 42), tbS(hid, 43)
		xf := f32Sy(T*hid, 51, sc)

		nw := &AudioLayerWeights{
			FF1:         &AudioFeedForwardWeights{PreNorm: pn1, PostNorm: pn2, FFW1: f1a, FFW2: f1b},
			FF2:         &AudioFeedForwardWeights{PreNorm: pn3, PostNorm: pn4, FFW1: f2a, FFW2: f2b},
			Attn:        &AudioAttentionWeights{QProj: qw, KProj: kw, VProj: vw, Post: pw, RelativeKProj: rkw, QScalePerDim: qsc, PosEmbed: pos, PosCount: P},
			LConv:       &AudioLightConvWeights{PreNorm: lpn, ConvNorm: lcn, LinearStart: ls, LinearEnd: le, DepthwiseWeight: dw},
			NormPreAttn: npa, NormPostAttn: npo, NormOut: no,
		}
		got, err := AudioLayer(xf, nw, cfg)
		if err != nil {
			t.Fatalf("AudioLayer(scale %v): %v", sc, err)
		}
		ff := func(a, b, p1, p2 []byte) *g4.Gemma4AudioFeedForward {
			return &g4.Gemma4AudioFeedForward{FFW1: cl(a, inter, hid), FFW2: cl(b, hid, inter), PreNorm: marr(p1, hid), PostNorm: marr(p2, hid), Eps: eps, Residual: res, Act: "silu", ClipMin: gcmn, ClipMax: gcmx}
		}
		attn := &g4.Gemma4AudioAttention{QProj: cl(qw, hd, hid), KProj: cl(kw, hd, hid), VProj: cl(vw, hd, hid), Post: cl(pw, hid, hd), RelativeKProj: mc.NewLinear(marr(rkw, hd, hid), nil), QScalePerDim: mc.FromValues(qsc, 1, 1, 1, D), PosEmbed: mc.FromValues(pos, P, hid), NumHeads: H, HeadDim: D, ChunkSize: chunk, PastHorizon: past, FutureHorizon: future, KScale: ksc, LogitCap: cap, InvalidLogit: inval}
		lc := &g4.Gemma4AudioLightConv{LinearStart: cl(ls, 2*ch, hid), LinearEnd: cl(le, hid, ch), DepthwiseWeight: marr(dw, ch, K, 1), PreNorm: marr(lpn, hid), ConvNorm: marr(lcn, ch), Eps: eps, KernelSize: K, Channels: ch, Act: "silu", ClipMin: gcmn, ClipMax: gcmx}
		layer := &g4.Gemma4AudioLayer{FeedForward1: ff(f1a, f1b, pn1, pn2), FeedForward2: ff(f2a, f2b, pn3, pn4), SelfAttn: attn, LConv: lc, NormPreAttn: marr(npa, hid), NormPostAttn: marr(npo, hid), NormOut: marr(no, hid), Eps: eps, ClipMin: gcmn, ClipMax: gcmx}
		eqF32(t, "AudioLayer vs real Gemma4AudioLayer.Forward", got, layer.Forward(mc.FromValues(xf, 1, T, hid)))
	}
}
