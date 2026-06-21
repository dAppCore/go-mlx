// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// audio_test.go validates the native Conformer audio tower against SELF-CONTAINED pure-Go fp32
// references (no pkg/metal) transcribing metal's actual audio_encoder.go — same discipline as
// vision_test.go. Reuses the shared refMatmul / refRMSRows / relL2Cos / bf16Round helpers.

// TestAudioFeedForward validates one Conformer FeedForward block (clamp → RMSNorm → FFW1 → silu →
// FFW2 → clamp → RMSNorm → ·residual → +x) against a pure-Go fp32 reference of metal's
// Gemma4AudioFeedForward.Forward.
func TestAudioFeedForward(t *testing.T) {
	requireNativeRuntime(t)
	const L, hidden, inter = 16, 128, 512
	eps, residual := float32(1e-6), float32(0.5)
	clipMin, clipMax := float32(-50), float32(50)
	w := func(s, n int) []float32 { return bf16Round(syntheticFloat32(n, s)) }
	preN, postN, ffw1, ffw2 := w(1, hidden), w(2, hidden), w(3, inter*hidden), w(4, hidden*inter)
	x := w(5, L*hidden)

	aw := &AudioFeedForwardWeights{PreNorm: toBF16Bytes(preN), PostNorm: toBF16Bytes(postN), FFW1: toBF16Bytes(ffw1), FFW2: toBF16Bytes(ffw2)}
	cfg := AudioConfig{Hidden: hidden, FFInter: inter, Eps: eps, Act: "silu", FFResidual: residual, ClipMin: clipMin, ClipMax: clipMax}
	got, err := AudioFeedForward(toBF16Bytes(x), aw, cfg)
	if err != nil {
		t.Fatalf("AudioFeedForward: %v", err)
	}

	clamp := func(v []float32) {
		for i := range v {
			if v[i] < clipMin {
				v[i] = clipMin
			} else if v[i] > clipMax {
				v[i] = clipMax
			}
		}
	}
	cl := append([]float32(nil), x...)
	clamp(cl)
	pre := refRMSRows(cl, preN, L, hidden, eps)
	up := refMatmul(pre, ffw1, L, inter, hidden)
	for i := range up {
		up[i] = up[i] / (1 + float32(math.Exp(float64(-up[i])))) // silu
	}
	down := refMatmul(up, ffw2, L, hidden, inter)
	clamp(down)
	post := refRMSRows(down, postN, L, hidden, eps)
	want := make([]float32, len(post))
	for i := range post {
		want[i] = post[i]*residual + x[i]
	}

	relL2, cos := relL2Cos(bf16Floats(got), want)
	t.Logf("AudioFeedForward vs fp32 reference [L=%d hidden=%d inter=%d]: rel-L2=%.3e cosine=%.6f", L, hidden, inter, relL2, cos)
	if cos < 0.999 || relL2 > 1e-2 {
		t.Fatalf("AudioFeedForward rel-L2 %.3e cosine %.6f — wiring bug", relL2, cos)
	}
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
