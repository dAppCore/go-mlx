// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// lastTokenDiffers reports whether two forwards' final-token outputs differ.
func lastTokenDiffers(a, b [][]byte) bool {
	la, lb := a[len(a)-1], b[len(b)-1]
	if len(la) != len(lb) {
		return true
	}
	for i := range la {
		if la[i] != lb[i] {
			return true
		}
	}
	return false
}

// TestDecodeForwardArchQuant gates the 4-bit arch-driven forward. (a) an all-owner,
// all-global, dense quant arch is byte-for-byte the proven DecodeForwardQuant (the arch
// executor + qmv projector ≡ the standalone quant forward when the arch routes nothing)
// — the correctness anchor. (b) a KV-share quant arch differs from the all-owner one
// (sharing genuinely reroutes layer 1's attention on the quant path). (c) a sliding
// quant arch (W=3) differs from full attention over 6 tokens (the window clips on the
// quant path).
func TestDecodeForwardArchQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}

	// (a) all-owner all-global ≡ DecodeForwardQuant byte-for-byte.
	const nL, T, maxLen = 3, 4, 8
	ql := make([]QuantizedLayerWeights, nL)
	types := make([]string, nL)
	for l := range ql {
		ql[l] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (l+1)*100)
		types[l] = "full_attention"
	}
	inputs := mkInputs(T)
	specsOwn := model.DeriveLayers(types, 0)
	gotArch, err := DecodeForwardArchQuant(inputs, ql, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant all-owner: %v", err)
	}
	ref, err := DecodeForwardQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("quant all-owner vs DecodeForwardQuant tok%d", tok), gotArch[tok], ref[tok])
	}

	// (b) KV-share reroutes attention: 2 layers, layer 1 shares layer 0's cache vs both
	// own. Different layer weights → the shared and owned results must differ.
	ql2 := []QuantizedLayerWeights{
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 100),
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 200),
	}
	in2 := mkInputs(T)
	specsShare := model.DeriveLayers([]string{"full_attention", "full_attention"}, 1)
	specsBothOwn := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	gotShare, err := DecodeForwardArchQuant(in2, ql2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant share: %v", err)
	}
	gotBothOwn, err := DecodeForwardArchQuant(in2, ql2, specsBothOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant both-own: %v", err)
	}
	if !lastTokenDiffers(gotShare, gotBothOwn) {
		t.Fatal("quant KV-share produced the same output as all-owner — sharing did not reroute attention")
	}

	// (c) sliding clips on the quant path: all-sliding W=3 over 6 tokens vs full (W=0).
	const W, T2, maxLen2 = 3, 6, 8
	slideTypes := make([]string, nL)
	for i := range slideTypes {
		slideTypes[i] = "sliding_attention"
	}
	specsSlide := model.DeriveLayers(slideTypes, 0)
	in3 := mkInputs(T2)
	gotSlide, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding: %v", err)
	}
	gotFull, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding-full: %v", err)
	}
	if !lastTokenDiffers(gotSlide, gotFull) {
		t.Fatal("quant sliding (W=3) matched full attention over 6 tokens — the window did not clip")
	}

	t.Logf("quant arch: all-owner ≡ DecodeForwardQuant byte-for-byte; KV-share reroutes; sliding (W=%d, %d toks) clips — 4-bit on the arch path", W, T2)
}

func TestDecodeForwardArchQuantMoELayer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim = 64, 1, 1, 64
	const dFF, expertDFF, numExperts, topK = 128, 96, 4, 2
	const gs, bits, maxLen, T = 32, 4, 4, 2
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	inputs := decodeInputsFixture(T, dModel)
	denseLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, gs, bits)
	moeLayer := denseLayer
	moeLayer.MLPNormW, moeLayer.Gate, moeLayer.Up, moeLayer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	moeLayer.MoE = &moeWeights

	denseSpecs := model.DeriveLayers([]string{"full_attention"}, 0)
	moeSpecs := model.DeriveLayers([]string{"full_attention"}, 0)
	moeSpecs[0].MoE = true

	gotMoE, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{moeLayer}, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant MoE: %v", err)
	}
	gotDense, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{denseLayer}, denseSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant dense: %v", err)
	}
	if len(gotMoE) != T {
		t.Fatalf("MoE outputs = %d tokens, want %d", len(gotMoE), T)
	}
	for i := range gotMoE {
		if len(gotMoE[i]) != dModel*bf16Size {
			t.Fatalf("MoE token %d has %d bytes, want %d", i, len(gotMoE[i]), dModel*bf16Size)
		}
	}
	if !lastTokenDiffers(gotMoE, gotDense) {
		t.Fatal("quant MoE arch matched dense MLP output; MoE block was not used")
	}

	t.Logf("quant MoE arch: DecodeForwardArchQuant runs the loader-shaped MoE layer through MoEBlockQuant")
}

func TestDecodeForwardArchQuantKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layers := []QuantizedLayerWeights{layer}
	specs := model.DeriveLayers([]string{"full_attention"}, 0)

	if _, err := DecodeForwardArchQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	weights := []struct {
		name string
		buf  []byte
	}{
		{"attnNorm", layer.AttnNormW},
		{"mlpNorm", layer.MLPNormW},
		{"q.packed", layer.Q.Packed}, {"q.scales", layer.Q.Scales}, {"q.biases", layer.Q.Biases},
		{"k.packed", layer.K.Packed}, {"k.scales", layer.K.Scales}, {"k.biases", layer.K.Biases},
		{"v.packed", layer.V.Packed}, {"v.scales", layer.V.Scales}, {"v.biases", layer.V.Biases},
		{"o.packed", layer.O.Packed}, {"o.scales", layer.O.Scales}, {"o.biases", layer.O.Biases},
		{"gate.packed", layer.Gate.Packed}, {"gate.scales", layer.Gate.Scales}, {"gate.biases", layer.Gate.Biases},
		{"up.packed", layer.Up.Packed}, {"up.scales", layer.Up.Scales}, {"up.biases", layer.Up.Biases},
		{"down.packed", layer.Down.Packed}, {"down.scales", layer.Down.Scales}, {"down.biases", layer.Down.Biases},
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missing := make([]string, 0)
	for _, weight := range weights {
		if _, ok := residentBufs[key(weight.buf)]; !ok {
			missing = append(missing, weight.name)
		}
	}
	residentBufMu.Unlock()

	if len(missing) != 0 {
		t.Fatalf("DecodeForwardArchQuant did not keep fixed weights resident (missing=%v resident=%d want>=%d)", missing, got, len(weights))
	}
}

func TestDecodeForwardArchQuantHonoursPerWeightGeometry(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const mlpGroupSize, mlpBits = 32, 8
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layer.Gate = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 20)
	layer.Up = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 22)
	layer.Down = quantWeightFixture(t, dModel, dFF, mlpGroupSize, mlpBits, 26)
	specs := model.DeriveLayers([]string{"full_attention"}, 0)

	got, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{layer}, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant with per-weight MLP geometry: %v", err)
	}
	ref, err := DecodeForwardQuant(inputs, []QuantizedLayerWeights{layer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant with per-weight MLP geometry: %v", err)
	}
	for tok := range got {
		eqBytes(t, core.Sprintf("mixed quant arch vs DecodeForwardQuant tok%d", tok), got[tok], ref[tok])
	}
}
