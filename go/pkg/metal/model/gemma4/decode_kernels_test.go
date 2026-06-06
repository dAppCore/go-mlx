// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

// Native + compiled Gemma 4 decode-kernel tests, relocated from package metal's
// decode_test.go with the gemma4-internal kernels they exercise
// (nativeGemma4FixedOwnerAttentionBlock/ResidualBlock, nativeGemma4DecodeLayer,
// compiledGemma4DecodeLayer, the fixed-attention mask helpers, sharedKV,
// closeGemma4, logitSoftcap). Package metal keeps the
// non-gemma4 native-kernel tests and the metal-resident Gemma 4 validators
// (ValidateGemma4LayerOutputs/Shapes, nativeGemma4LayerLinearAvailable).
//
// metal is dot-imported so this near-verbatim relocation keeps the original
// bare metal-symbol usage; the in-package gemma4 kernels resolve without a
// prefix. The two layer-native/compiled gates moved from package-metal
// unexported vars to the public SetRuntimeGate seam (decode.go sanctions
// SetRuntimeGate for explicit probes).

import (
	"testing"

	. "dappco.re/go/mlx/pkg/metal"
)

func TestDecode_nativeFixedSingleTokenAttentionMasked_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention masked"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keyA := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	valueA := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offsetA := FromValue(0)
	maskA := fixedSingleTokenCausalMaskFromHost(1, 4, 0)
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	maskB := fixedSingleTokenCausalMaskFromHost(1, 4, 1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, maskA, keyB, valueB, offsetB, maskB)

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, maskA, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(masked first) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(masked first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(masked second) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)

	keysValid := Slice(secondKeys, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(secondValues, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(second, secondKeys, secondValues, wantSecond); err != nil {
		t.Fatalf("Eval(masked second) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), wantSecond.Floats())
	floatSliceApprox(t, secondKeys.Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, secondValues.Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}

func TestDecode_nativeFixedSingleTokenAttentionRowUpdate_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention row update"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetFixedAttentionDiagnostics(false, false, true))
	requireMetalRuntime(t)

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keyA := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	valueA := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offsetA := FromValue(0)
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	maskB := fixedSingleTokenCausalMaskFromHost(1, 4, 1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, keyB, valueB, offsetB, maskB)

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(row first) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(row first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	floatSliceApprox(t, firstKeys.Floats(), []float32{1, 0, 0, 0, 0, 0, 0, 0})
	floatSliceApprox(t, firstValues.Floats(), []float32{10, 0, 0, 0, 0, 0, 0, 0})

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(row masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(row masked second) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)

	keysValid := Slice(secondKeys, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(secondValues, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(second, secondKeys, secondValues, wantSecond); err != nil {
		t.Fatalf("Eval(row second) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), wantSecond.Floats())
	floatSliceApprox(t, secondKeys.Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, secondValues.Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}

func testGemma4NativeLayerConfig() *Gemma4TextConfig {
	return &Gemma4TextConfig{
		TransformerConfig: TransformerConfig{
			RMSNormEps:        1e-6,
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			HeadDim:           2,
		},
	}
}

func testGemma4NativeLayer() *Gemma4DecoderLayer {
	norm := func() *Array { return FromValues([]float32{1, 1}, 2) }
	linear := func(vals []float32) *Linear {
		return NewLinear(FromValues(vals, 2, 2), nil)
	}
	layer := &Gemma4DecoderLayer{
		InputNormScaled:             norm(),
		PostAttnNormScaled:          norm(),
		PreFFNormScaled:             norm(),
		PostFFNormScaled:            norm(),
		PostPerLayerInputNormScaled: norm(),
		LayerScalar:                 FromValues([]float32{1}, 1),
		Attention: &Gemma4Attention{
			QProj:          linear([]float32{1, 0, 0, 1}),
			KProj:          linear([]float32{1, 0, 0, 1}),
			VProj:          linear([]float32{0.5, 0.25, -0.25, 0.75}),
			OProj:          linear([]float32{1, 0, 0, 1}),
			QNormScaled:    norm(),
			KNormScaled:    norm(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          0.70710677,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &MLP{
			GateProj: linear([]float32{0.5, 0.1, -0.2, 0.3}),
			UpProj:   linear([]float32{0.4, -0.1, 0.2, 0.6}),
			DownProj: linear([]float32{0.7, 0.2, -0.3, 0.5}),
		},
		PerLayerInputGate:  linear([]float32{0.2, 0.1, 0.3, -0.2}),
		PerLayerProjection: linear([]float32{0.6, 0.1, -0.2, 0.4}),
	}
	return layer
}

func testGemma4NativeMoELayer() *Gemma4DecoderLayer {
	layer := testGemma4NativeLayer()
	norm := func() *Array { return FromValues([]float32{1, 1}, 2) }
	switchLinear := func(vals []float32) *SwitchLinear {
		return NewSwitchLinear(FromValues(vals, 2, 2, 2), nil)
	}
	layer.EnableMoE = true
	layer.PreFFNorm2Scaled = norm()
	layer.PostFFNorm1Scaled = norm()
	layer.PostFFNorm2Scaled = norm()
	layer.Router = &Gemma4Router{
		Proj:           NewLinear(FromValues([]float32{1.0, -0.25, -0.5, 0.75}, 2, 2), nil),
		Scale:          norm(),
		ScaleScaled:    norm(),
		PerExpertScale: FromValues([]float32{1.0, 0.75}, 2),
		TopK:           1,
		Eps:            1e-6,
	}
	layer.Experts = &Gemma4Experts{
		GateProj: switchLinear([]float32{
			0.9, 0.1,
			-0.2, 0.8,
			0.3, -0.4,
			0.7, 0.2,
		}),
		UpProj: switchLinear([]float32{
			0.6, -0.1,
			0.2, 0.5,
			-0.3, 0.4,
			0.8, -0.2,
		}),
		DownProj: switchLinear([]float32{
			0.7, 0.2,
			-0.1, 0.6,
			0.4, -0.3,
			0.2, 0.9,
		}),
	}
	return layer
}

func freeTestGemma4NativeLayer(layer *Gemma4DecoderLayer) {
	if layer == nil {
		return
	}
	Free(
		layer.InputNormScaled,
		layer.PostAttnNormScaled,
		layer.PreFFNormScaled,
		layer.PostFFNormScaled,
		layer.PostPerLayerInputNormScaled,
		layer.LayerScalar,
	)
	if layer.Attention != nil {
		Free(
			layer.Attention.QProj.Weight,
			layer.Attention.KProj.Weight,
			layer.Attention.VProj.Weight,
			layer.Attention.OProj.Weight,
			layer.Attention.QNormScaled,
			layer.Attention.KNormScaled,
		)
	}
	if layer.MLP != nil {
		Free(layer.MLP.GateProj.Weight, layer.MLP.UpProj.Weight, layer.MLP.DownProj.Weight)
	}
	Free(layer.PerLayerInputGate.Weight, layer.PerLayerProjection.Weight)
}
