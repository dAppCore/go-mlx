// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestEmbedLMHeadQuant gates the quant decode bookends against metal as the oracle: a random
// embedding table is quantised (the checkpoint's packed form), and
//   - EmbedTokensQuant (scale=1) must equal metal.Dequantize on the gathered rows BYTE-FOR-BYTE
//     — the host affine-dequant (nibble unpack + scale·code+bias) matches the metallib exactly;
//   - the √hidden-scaled embed must equal EmbedTokensBF16 on the dequantised table (the bf16
//     embed is already gated, so this pins the scale);
//   - LMHeadQuant must agree with LMHeadBF16 on the dequantised table — same argmax + close
//     logits (QMVBF16's in-kernel f32 dequant differs from bf16-dequant-then-matvec only by
//     rounding; QMVBF16's exact byte-parity vs QuantizedMatmul is gated separately).
func TestEmbedLMHeadQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const vocab, dModel, gs, bits = 16, 64, 32, 4
	mk := func(n, salt int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*salt+11)%89-44) * 0.03
		}
		return f
	}

	// build + quantise a random bf16 embedding table → the checkpoint's (packed, scales, biases).
	arr := metal.FromRawBytes(toBF16Bytes(mk(vocab*dModel, 7)), []int{vocab, dModel}, metal.DTypeBFloat16)
	wq, scales, biases, err := metal.Quantize(arr, gs, bits, "affine")
	if err != nil {
		metal.Free(arr)
		t.Fatalf("Quantize: %v", err)
	}
	deq := metal.Dequantize(wq, scales, biases, gs, bits) // the oracle: the dequantised table
	metal.Materialize(wq, scales, biases, deq)
	packed := append([]byte(nil), wq.RawBytes()...)
	sc := append([]byte(nil), scales.RawBytes()...)
	bi := append([]byte(nil), biases.RawBytes()...)
	deqBytes := append([]byte(nil), deq.RawBytes()...)
	metal.Free(arr, wq, scales, biases, deq)

	ids := []int32{0, 3, 7, 15}

	// (1) host dequant ≡ oracle rows, byte-for-byte (the crux parity).
	embs, err := EmbedTokensQuant(packed, sc, bi, ids, vocab, dModel, gs, bits, 1.0)
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}
	for i, tok := range ids {
		want := deqBytes[int(tok)*dModel*bf16Size : (int(tok)+1)*dModel*bf16Size]
		if !bytes.Equal(embs[i], want) {
			t.Fatalf("token %d: host dequant != metal.Dequantize oracle\n got %x\nwant %x", tok, embs[i], want)
		}
	}

	// (2) √hidden-scaled ≡ the gated bf16 embed on the dequantised table.
	scale := float32(math.Sqrt(float64(dModel)))
	refScaled, err := EmbedTokensBF16(deqBytes, ids, vocab, dModel, scale)
	if err != nil {
		t.Fatalf("EmbedTokensBF16: %v", err)
	}
	qScaled, err := EmbedTokensQuant(packed, sc, bi, ids, vocab, dModel, gs, bits, scale)
	if err != nil {
		t.Fatalf("EmbedTokensQuant scaled: %v", err)
	}
	for i := range ids {
		if !bytes.Equal(qScaled[i], refScaled[i]) {
			t.Fatalf("scaled embed token %d != bf16-on-dequant", ids[i])
		}
	}

	// (3) LMHeadQuant agrees with LMHeadBF16 on the dequantised table: same argmax + close logits.
	hidden := toBF16Bytes(mk(dModel, 5))
	finalNorm := toBF16Bytes(mk(dModel, 9))
	const eps, softCap = float32(1e-6), float32(30)
	qLogits, err := LMHeadQuant(hidden, finalNorm, packed, sc, bi, dModel, vocab, gs, bits, eps, softCap)
	if err != nil {
		t.Fatalf("LMHeadQuant: %v", err)
	}
	refLogits, err := LMHeadBF16(hidden, finalNorm, deqBytes, dModel, vocab, eps, softCap)
	if err != nil {
		t.Fatalf("LMHeadBF16: %v", err)
	}
	argmax := func(b []byte) (int, float32) {
		best, bi := -1, float32(-1e30)
		for i := 0; i < vocab; i++ {
			v := bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
			if v > bi {
				best, bi = i, v
			}
		}
		return best, bi
	}
	qa, _ := argmax(qLogits)
	ra, _ := argmax(refLogits)
	if qa != ra {
		t.Fatalf("LMHeadQuant argmax %d != LMHeadBF16-on-dequant argmax %d", qa, ra)
	}
	var maxDiff float32
	for i := 0; i < vocab; i++ {
		d := bf16ToF32(qLogits[i*bf16Size], qLogits[i*bf16Size+1]) - bf16ToF32(refLogits[i*bf16Size], refLogits[i*bf16Size+1])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.5 { // bf16/f32-dequant rounding only; not a semantic gap
		t.Fatalf("LMHeadQuant logits diverge from bf16-on-dequant: max abs diff %v", maxDiff)
	}
	t.Logf("quant bookends: host dequant ≡ metal.Dequantize byte-for-byte; scaled embed ≡ bf16-on-dequant; LMHeadQuant argmax=%d ≡ ref, max logit diff %.4f", qa, maxDiff)
}

func TestLMHeadQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(t, vocab, dModel, groupSize, bits, 53)
	if _, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
		t.Fatalf("LMHeadQuant warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
			t.Fatalf("LMHeadQuant: %v", err)
		}
	})
	if allocs > 35 {
		t.Fatalf("LMHeadQuant allocations = %.0f, want <= 35", allocs)
	}
}

func TestLMHeadQuantIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(t, vocab, dModel, groupSize, bits, 53)
	want, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadQuant reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, vocab*bf16Size)

	scratch, err := getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x6a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := LMHeadQuantInto(out, hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadQuantInto: %v", err)
	}
	if len(got) != len(want) || &got[0] != &out[0] {
		t.Fatal("LMHeadQuantInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "LMHeadQuantInto", got, want)

	scratch, err = getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("LMHeadQuantInto wrote through pooled scratch output instead of caller output")
	}
}
