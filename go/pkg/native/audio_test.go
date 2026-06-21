// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// audio_test.go validates the native Conformer audio tower for BYTE-IDENTITY to metal (Snider's bar).
// The FF check follows parity_test.go: compose the SAME metal OPS the block uses (bf16-chained, since
// every gemma4 op keeps bf16) and assert eqBytes — NOT a tolerance. (TestAudioLightConv below is still
// on the old self-contained tolerance ref, pending its byte-identical rebuild.)

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

func sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// TestAudioLightConv validates the Conformer GLU-conv module (RMSNorm → LinearStart → GLU → causal
// depthwise conv1d → clamp → RMSNorm → silu → LinearEnd → +x) against a pure-Go fp32 reference of
// metal's Gemma4AudioLightConv.Forward — including the left-padded causal conv.
func TestAudioLightConv(t *testing.T) {
	requireNativeRuntime(t)
	const L, hidden, K = 16, 128, 5
	ch := hidden
	eps := float32(1e-6)
	w := func(s, n int) []float32 { return bf16Round(syntheticFloat32(n, s)) }
	preN, convN, lstart, lend, dw := w(1, hidden), w(2, ch), w(3, 2*ch*hidden), w(4, hidden*ch), w(5, ch*K)
	x := w(6, L*hidden)

	aw := &AudioLightConvWeights{PreNorm: toBF16Bytes(preN), ConvNorm: toBF16Bytes(convN), LinearStart: toBF16Bytes(lstart), LinearEnd: toBF16Bytes(lend), DepthwiseWeight: toBF16Bytes(dw)}
	cfg := AudioConfig{Hidden: hidden, Channels: ch, KernelSize: K, Eps: eps, Act: "silu", ClipMin: -50, ClipMax: 50}
	got, err := AudioLightConv(toBF16Bytes(x), aw, cfg)
	if err != nil {
		t.Fatalf("AudioLightConv: %v", err)
	}

	pre := refRMSRows(x, preN, L, hidden, eps)
	start := refMatmul(pre, lstart, L, 2*ch, hidden)
	glu := make([]float32, L*ch)
	for ti := 0; ti < L; ti++ {
		for c := 0; c < ch; c++ {
			glu[ti*ch+c] = start[ti*2*ch+c] * sigmoidf(start[ti*2*ch+ch+c])
		}
	}
	conv := make([]float32, L*ch)
	for ti := 0; ti < L; ti++ {
		for c := 0; c < ch; c++ {
			var acc float32
			for k := 0; k < K; k++ {
				if src := ti - (K - 1) + k; src >= 0 {
					acc += glu[src*ch+c] * dw[c*K+k]
				}
			}
			conv[ti*ch+c] = acc
		}
	}
	normed := refRMSRows(conv, convN, L, ch, eps)
	for i := range normed {
		normed[i] = normed[i] * sigmoidf(normed[i])
	}
	end := refMatmul(normed, lend, L, hidden, ch)
	want := make([]float32, len(end))
	for i := range end {
		want[i] = end[i] + x[i]
	}

	relL2, cos := relL2Cos(bf16Floats(got), want)
	t.Logf("AudioLightConv vs fp32 reference [L=%d hidden=%d kernel=%d]: rel-L2=%.3e cosine=%.6f", L, hidden, K, relL2, cos)
	if cos < 0.999 || relL2 > 1e-2 {
		t.Fatalf("AudioLightConv rel-L2 %.3e cosine %.6f — wiring/conv bug", relL2, cos)
	}
}
