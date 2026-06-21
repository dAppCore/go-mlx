// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
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
