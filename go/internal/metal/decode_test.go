// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func float32Fill(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func TestDecode_nativeGreedyDecodeToken_Good(t *testing.T) {
	target := "nativeGreedyDecodeToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	logits := FromValues([]float32{0.1, 2.5, -1.0}, 1, 1, 3)
	defer Free(logits)

	token, err := nativeGreedyDecodeToken(logits)
	if err != nil {
		t.Fatalf("nativeGreedyDecodeToken() error = %v", err)
	}
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval(token) error = %v", err)
	}
	if got := token.Int(); got != 1 {
		t.Fatalf("token = %d, want 1", got)
	}
}

func TestDecode_nativeGreedyDecodeToken_Bad(t *testing.T) {
	target := "nativeGreedyDecodeToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, err := nativeGreedyDecodeToken(nil); err == nil {
		t.Fatal("nativeGreedyDecodeToken(nil) error = nil, want error")
	}
}

func TestDecode_nativeGreedyDecodeToken_Ugly(t *testing.T) {
	target := "nativeGreedyDecodeToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	logits := FromValues([]float32{9, 1, 0, 0.2, 0.3, 0.4}, 1, 2, 3)
	defer Free(logits)

	token, err := nativeGreedyDecodeToken(logits)
	if err != nil {
		t.Fatalf("nativeGreedyDecodeToken() error = %v", err)
	}
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval(token) error = %v", err)
	}
	if got := token.Int(); got != 2 {
		t.Fatalf("token = %d, want last-position argmax 2", got)
	}
}

func TestDecode_nativeGreedyDecodeAvailable_Good(t *testing.T) {
	target := "nativeGreedyDecodeAvailable"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	logits := Zeros([]int32{1, 1, 3}, DTypeFloat32)
	defer Free(logits)
	cfg := GenerateConfig{}
	if !nativeGreedyDecodeAvailable(cfg, nil, logits) {
		t.Fatal("nativeGreedyDecodeAvailable() = false, want true for unprobed greedy single-step logits")
	}
}

func TestDecode_nativeGreedyDecodeAvailable_Bad(t *testing.T) {
	target := "nativeGreedyDecodeAvailable"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if nativeGreedyDecodeAvailable(GenerateConfig{}, nil, nil) {
		t.Fatal("nativeGreedyDecodeAvailable(nil logits) = true, want false")
	}
}

func TestDecode_nativeGreedyDecodeAvailable_Ugly(t *testing.T) {
	target := "nativeGreedyDecodeAvailable"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	logits := Zeros([]int32{1, 8, 3}, DTypeFloat32)
	defer Free(logits)
	cfg := GenerateConfig{RepeatPenalty: 1.1}
	if nativeGreedyDecodeAvailable(cfg, []int32{1}, logits) {
		t.Fatal("nativeGreedyDecodeAvailable() = true, want false for repeat penalty and variable sequence logits")
	}
}

func TestDecode_nativeLastTokenOutputLogits_Good(t *testing.T) {
	target := "nativeLastTokenOutputLogits"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	got, ok, err := nativeLastTokenOutputLogits(hidden, normWeight, output, 1e-6, 30)
	if err != nil {
		t.Fatalf("nativeLastTokenOutputLogits() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeLastTokenOutputLogits() ok = false, want true")
	}
	defer Free(got)

	normed := RMSNorm(hidden, normWeight, 1e-6)
	wantRaw := output.Forward(normed)
	want := logitSoftcap(wantRaw, 30)
	Free(normed, wantRaw)
	defer Free(want)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(logits) error = %v", err)
	}
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 3 {
		t.Fatalf("native logits shape = %v, want [1 1 3]", shape)
	}

	gotToken, err := nativeGreedyDecodeToken(got)
	if err != nil {
		t.Fatalf("nativeGreedyDecodeToken(got) error = %v", err)
	}
	wantToken, err := nativeGreedyDecodeToken(want)
	if err != nil {
		Free(gotToken)
		t.Fatalf("nativeGreedyDecodeToken(want) error = %v", err)
	}
	defer Free(gotToken, wantToken)
	if err := Eval(gotToken, wantToken); err != nil {
		t.Fatalf("Eval(tokens) error = %v", err)
	}
	if gotID, wantID := gotToken.Int(), wantToken.Int(); gotID != wantID {
		t.Fatalf("token = %d, want %d", gotID, wantID)
	}
}

func TestDecode_nativeLastTokenOutputLogits_Bad(t *testing.T) {
	target := "nativeLastTokenOutputLogits"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}

	if _, ok, err := nativeLastTokenOutputLogits(nil, nil, nil, 1e-6, 30); ok || err != nil {
		t.Fatalf("nativeLastTokenOutputLogits(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeLastTokenOutputLogits_Ugly(t *testing.T) {
	target := "nativeLastTokenOutputLogits"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	if _, ok, err := nativeLastTokenOutputLogits(hidden, normWeight, output, 1e-5, 30); ok || err != nil {
		t.Fatalf("nativeLastTokenOutputLogits(eps=1e-5) = ok %v err %v, want unsupported without error", ok, err)
	}
	if _, ok, err := nativeLastTokenOutputLogits(hidden, normWeight, output, 1e-6, 0); ok || err != nil {
		t.Fatalf("nativeLastTokenOutputLogits(softcap=0) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeLastTokenGreedyToken_Good(t *testing.T) {
	target := "nativeLastTokenGreedyToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	got, ok, err := nativeLastTokenGreedyToken(hidden, normWeight, output, 1e-6)
	if err != nil {
		t.Fatalf("nativeLastTokenGreedyToken() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeLastTokenGreedyToken() ok = false, want true")
	}
	defer Free(got)

	normed := RMSNorm(hidden, normWeight, 1e-6)
	logits := output.Forward(normed)
	want := Argmax(logits, -1, false)
	Free(normed, logits)
	defer Free(want)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(tokens) error = %v", err)
	}
	if gotID, wantID := got.Int(), want.Int(); gotID != wantID {
		t.Fatalf("token = %d, want %d", gotID, wantID)
	}
}

func TestDecode_nativeLastTokenGreedyTokenSuppressesIDs_Good(t *testing.T) {
	target := "nativeLastTokenGreedyToken suppress IDs"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	got, ok, err := nativeLastTokenGreedyToken(hidden, normWeight, output, 1e-6, 2)
	if err != nil {
		t.Fatalf("nativeLastTokenGreedyToken() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeLastTokenGreedyToken() ok = false, want true")
	}
	defer Free(got)

	if err := Eval(got); err != nil {
		t.Fatalf("Eval(tokens) error = %v", err)
	}
	if gotID := got.Int(); gotID != 1 {
		t.Fatalf("suppressed token = %d, want 1 after suppressing argmax ID 2", gotID)
	}
}

func TestDecode_nativeLastTokenGreedyToken_Bad(t *testing.T) {
	target := "nativeLastTokenGreedyToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, ok, err := nativeLastTokenGreedyToken(nil, nil, nil, 1e-6); ok || err != nil {
		t.Fatalf("nativeLastTokenGreedyToken(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeLastTokenGreedyToken_Ugly(t *testing.T) {
	target := "nativeLastTokenGreedyToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	if _, ok, err := nativeLastTokenGreedyToken(hidden, normWeight, output, 1e-5); ok || err != nil {
		t.Fatalf("nativeLastTokenGreedyToken(eps=1e-5) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeMLPGELU_Good(t *testing.T) {
	target := "nativeMLPGELU"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Setenv("GO_MLX_ENABLE_NATIVE_MLP_GELU", "1")
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2}, 1, 1, 2)
	gateW := FromValues([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	upW := FromValues([]float32{
		1, 1,
		1, -1,
		0, 1,
	}, 3, 2)
	downW := FromValues([]float32{
		1, 0, 0,
		0, 1, 1,
	}, 2, 3)
	mlp := &MLP{
		GateProj: NewLinear(gateW, nil),
		UpProj:   NewLinear(upW, nil),
		DownProj: NewLinear(downW, nil),
	}
	defer Free(input, gateW, upW, downW)

	got, ok, err := nativeMLPGELU(input, mlp)
	if err != nil {
		t.Fatalf("nativeMLPGELU() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMLPGELU() ok = false, want true")
	}
	defer Free(got)

	gate := mlp.GateProj.Forward(input)
	up := mlp.UpProj.Forward(input)
	activated := geluGateMul(gate, up)
	want := mlp.DownProj.Forward(activated)
	Free(gate, up, activated)
	defer Free(want)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(MLP) error = %v", err)
	}
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 2 {
		t.Fatalf("native MLP shape = %v, want [1 1 2]", shape)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeMLPGELU_Bad(t *testing.T) {
	target := "nativeMLPGELU"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}

	if _, ok, err := nativeMLPGELU(nil, nil); ok || err != nil {
		t.Fatalf("nativeMLPGELU(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeMLPGELU_Ugly(t *testing.T) {
	target := "nativeMLPGELU"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Setenv("GO_MLX_ENABLE_NATIVE_MLP_GELU", "1")
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2}, 1, 1, 2)
	weight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	bias := FromValues([]float32{1, 1}, 2)
	defer Free(input, weight, bias)

	mlp := &MLP{
		GateProj: NewLinear(weight, bias),
		UpProj:   NewLinear(weight, nil),
		DownProj: NewLinear(weight, nil),
	}
	if _, ok, err := nativeMLPGELU(input, mlp); ok || err != nil {
		t.Fatalf("nativeMLPGELU(biased) = ok %v err %v, want unsupported without error", ok, err)
	}

	scales := FromValues([]float32{1}, 1, 1)
	biases := FromValues([]float32{0}, 1, 1)
	defer Free(scales, biases)
	q4 := NewQuantizedLinear(weight, scales, biases, nil, 64, 4)
	q8 := NewQuantizedLinear(weight, scales, biases, nil, 64, 8)
	mlp = &MLP{GateProj: q4, UpProj: q4, DownProj: q8}
	if _, ok, err := nativeMLPGELU(input, mlp); ok || err != nil {
		t.Fatalf("nativeMLPGELU(mixed quantization) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4LayerLinearAvailable_Good(t *testing.T) {
	target := "nativeGemma4LayerLinearAvailable"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	weight := FromValues([]uint32{0}, 1, 1)
	scales := FromValues([]float32{1}, 1, 1)
	biases := FromValues([]float32{0}, 1, 1)
	defer Free(weight, scales, biases)

	q8 := NewQuantizedLinear(weight, scales, biases, nil, 64, 8)
	if !nativeGemma4LayerLinearAvailable(q8) {
		t.Fatal("nativeGemma4LayerLinearAvailable(q8 affine) = false, want true")
	}

	q8.Bits = 3
	if nativeGemma4LayerLinearAvailable(q8) {
		t.Fatal("nativeGemma4LayerLinearAvailable(3-bit affine) = true, want false")
	}
}

func TestDecode_nativeFixedSingleTokenAttention_Good(t *testing.T) {
	target := "nativeFixedSingleTokenAttention"
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
	keyB := FromValues([]float32{0, 1}, 1, 1, 1, 2)
	valueB := FromValues([]float32{0, 20}, 1, 1, 1, 2)
	offsetB := FromValue(1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, keyB, valueB, offsetB)

	first, firstKeys, firstValues, ok, err := nativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(first) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	wantFirst := ScaledDotProductAttention(query, keyA, valueA, 1, false)
	defer Free(wantFirst)
	if err := Eval(first, firstKeys, firstValues, wantFirst); err != nil {
		t.Fatalf("Eval(first) error = %v", err)
	}
	floatSliceApprox(t, first.Floats(), wantFirst.Floats())
	floatSliceApprox(t, firstKeys.Floats(), []float32{1, 0, 0, 0, 0, 0, 0, 0})
	floatSliceApprox(t, firstValues.Floats(), []float32{10, 0, 0, 0, 0, 0, 0, 0})

	second, secondKeys, secondValues, ok, err := nativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, nil, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(second) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(second) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)
	keysValid := Slice(secondKeys, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	valuesValid := Slice(secondValues, []int32{0, 0, 0, 0}, []int32{1, 1, 2, 2})
	wantSecond := ScaledDotProductAttention(query, keysValid, valuesValid, 1, false)
	defer Free(keysValid, valuesValid, wantSecond)
	if err := Eval(second, secondKeys, secondValues, wantSecond); err != nil {
		t.Fatalf("Eval(second) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), wantSecond.Floats())
	floatSliceApprox(t, secondKeys.Floats(), []float32{1, 0, 0, 1, 0, 0, 0, 0})
	floatSliceApprox(t, secondValues.Floats(), []float32{10, 0, 0, 20, 0, 0, 0, 0})
}

func TestDecode_nativeFixedSingleTokenAttentionMasked_Good(t *testing.T) {
	target := "nativeFixedSingleTokenAttention masked"
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

	first, firstKeys, firstValues, ok, err := nativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, maskA, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(masked first) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(masked first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)

	second, secondKeys, secondValues, ok, err := nativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(masked second) ok = false, want true")
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
	target := "nativeFixedSingleTokenAttention row update"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Setenv("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE", "1")
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

	first, firstKeys, firstValues, ok, err := nativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(row first) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(row first) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	floatSliceApprox(t, firstKeys.Floats(), []float32{1, 0, 0, 0, 0, 0, 0, 0})
	floatSliceApprox(t, firstValues.Floats(), []float32{10, 0, 0, 0, 0, 0, 0, 0})

	second, secondKeys, secondValues, ok, err := nativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, maskB, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(row masked second) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(row masked second) ok = false, want true")
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

func TestDecode_nativeFixedSlidingSingleTokenAttention_Good(t *testing.T) {
	target := "nativeFixedSlidingSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	query := FromValues([]float32{
		1, 0,
		0, 1,
	}, 1, 2, 1, 2)
	keyCache := FromValues([]float32{
		1, 0,
		0, 1,
	}, 1, 1, 2, 2)
	valueCache := FromValues([]float32{
		10, 0,
		0, 20,
	}, 1, 1, 2, 2)
	key := FromValues([]float32{1, 1}, 1, 1, 1, 2)
	value := FromValues([]float32{30, 40}, 1, 1, 1, 2)
	shiftIndices := FromValues([]int32{1, 1}, 2)
	lastIndex := FromValue(1)
	defer Free(query, keyCache, valueCache, key, value, shiftIndices, lastIndex)

	got, gotKeys, gotValues, ok, err := nativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex, 1)
	if err != nil {
		t.Fatalf("nativeFixedSlidingSingleTokenAttention error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSlidingSingleTokenAttention ok = false, want true")
	}
	if !got.Valid() || !gotKeys.Valid() || !gotValues.Valid() {
		t.Fatalf("nativeFixedSlidingSingleTokenAttention returned invalid outputs: out=%v keys=%v values=%v", got.Valid(), gotKeys.Valid(), gotValues.Valid())
	}
	defer Free(got, gotKeys, gotValues)

	wantKeys := FromValues([]float32{
		0, 1,
		1, 1,
	}, 1, 1, 2, 2)
	wantValues := FromValues([]float32{
		0, 20,
		30, 40,
	}, 1, 1, 2, 2)
	want := ScaledDotProductAttention(query, wantKeys, wantValues, 1, false)
	defer Free(wantKeys, wantValues, want)

	if err := Eval(got, gotKeys, gotValues, want); err != nil {
		t.Fatalf("Eval(sliding) error = %v", err)
	}
	floatSliceApprox(t, gotKeys.Floats(), wantKeys.Floats())
	floatSliceApprox(t, gotValues.Floats(), wantValues.Floats())
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeFixedSlidingSingleTokenAttentionGemma4E2BShape_Good(t *testing.T) {
	target := "nativeFixedSlidingSingleTokenAttention Gemma4E2BShape"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	const B, QH, KVH, window, D int32 = 1, 8, 1, 512, 256
	query := RandomUniform(-0.5, 0.5, []int32{B, QH, 1, D}, DTypeBFloat16)
	keyCache := RandomUniform(-0.5, 0.5, []int32{B, KVH, window, D}, DTypeBFloat16)
	valueCache := RandomUniform(-0.5, 0.5, []int32{B, KVH, window, D}, DTypeBFloat16)
	key := RandomUniform(-0.5, 0.5, []int32{B, KVH, 1, D}, DTypeBFloat16)
	value := RandomUniform(-0.5, 0.5, []int32{B, KVH, 1, D}, DTypeBFloat16)
	shiftIndices := FromValues(func() []int32 {
		out := make([]int32, window)
		for i := int32(0); i < window; i++ {
			next := i + 1
			if next >= window {
				next = window - 1
			}
			out[i] = next
		}
		return out
	}(), int(window))
	lastIndex := FromValue(int(window - 1))
	defer Free(query, keyCache, valueCache, key, value, shiftIndices, lastIndex)
	Materialize(query, keyCache, valueCache, key, value, shiftIndices, lastIndex)

	got, gotKeys, gotValues, ok, err := nativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex, 0.0625)
	if err != nil {
		t.Fatalf("nativeFixedSlidingSingleTokenAttention(E2B shape) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSlidingSingleTokenAttention(E2B shape) ok = false, want true")
	}
	defer Free(got, gotKeys, gotValues)
	if err := Eval(got, gotKeys, gotValues); err != nil {
		t.Fatalf("Eval(E2B shape) error = %v", err)
	}
	if !got.Valid() || !gotKeys.Valid() || !gotValues.Valid() {
		t.Fatalf("nativeFixedSlidingSingleTokenAttention(E2B shape) returned invalid outputs: out=%v keys=%v values=%v", got.Valid(), gotKeys.Valid(), gotValues.Valid())
	}
	if got.Dim(1) != int(QH) || gotKeys.Dim(2) != int(window) || gotValues.Dim(2) != int(window) {
		t.Fatalf("E2B shape outputs = out heads:%d key window:%d value window:%d, want heads:%d window:%d", got.Dim(1), gotKeys.Dim(2), gotValues.Dim(2), QH, window)
	}
}

func TestDecode_nativeResidualNormAdd_Good(t *testing.T) {
	target := "nativeResidualNormAdd"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	residual := FromValues([]float32{1, 2}, 1, 1, 2)
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	norm := FromValues([]float32{1, 1}, 2)
	defer Free(residual, input, norm)

	got, ok, err := nativeResidualNormAdd(residual, input, norm, 1e-6)
	if err != nil {
		t.Fatalf("nativeResidualNormAdd() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeResidualNormAdd() ok = false, want true")
	}
	defer Free(got)
	normed := RMSNorm(input, norm, 1e-6)
	want := Add(residual, normed)
	defer Free(normed, want)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(got/want) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeResidualNormAdd_Bad(t *testing.T) {
	target := "nativeResidualNormAdd"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, ok, err := nativeResidualNormAdd(nil, nil, nil, 1e-6); ok || err != nil {
		t.Fatalf("nativeResidualNormAdd(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeResidualNormAdd_Ugly(t *testing.T) {
	target := "nativeResidualNormAdd"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	residual := FromValues([]float32{1, 2}, 1, 1, 2)
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	norm := FromValues([]float32{1, 1}, 2)
	defer Free(residual, input, norm)

	if _, ok, err := nativeResidualNormAdd(residual, input, norm, 1e-5); ok || err != nil {
		t.Fatalf("nativeResidualNormAdd(eps=1e-5) = ok %v err %v, want unsupported without error", ok, err)
	}
	mismatch := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	defer Free(mismatch)
	if _, ok, err := nativeResidualNormAdd(residual, mismatch, norm, 1e-6); ok || err != nil {
		t.Fatalf("nativeResidualNormAdd(shape mismatch) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeFixedSingleTokenAttentionWide_Good(t *testing.T) {
	target := "nativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION", "1")
	requireMetalRuntime(t)

	const headDim = 512
	query := FromValues(float32Fill(2*headDim, 0), 1, 2, 1, headDim)
	keyCache := Zeros([]int32{1, 1, 4, headDim}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, headDim}, DTypeFloat32)
	keyA := FromValues(float32Fill(headDim, 1), 1, 1, 1, headDim)
	valueA := FromValues(float32Fill(headDim, 2), 1, 1, 1, headDim)
	offsetA := FromValue(0)
	keyB := FromValues(float32Fill(headDim, 3), 1, 1, 1, headDim)
	valueB := FromValues(float32Fill(headDim, 4), 1, 1, 1, headDim)
	offsetB := FromValue(1)
	defer Free(query, keyCache, valueCache, keyA, valueA, offsetA, keyB, valueB, offsetB)

	first, firstKeys, firstValues, ok, err := nativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(first wide) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(first wide) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	if err := Eval(first, firstKeys, firstValues); err != nil {
		t.Fatalf("Eval(first wide) error = %v", err)
	}
	floatSliceApprox(t, first.Floats(), float32Fill(2*headDim, 2))
	floatSliceApprox(t, firstKeys.Floats()[:headDim], float32Fill(headDim, 1))
	floatSliceApprox(t, firstValues.Floats()[:headDim], float32Fill(headDim, 2))

	second, secondKeys, secondValues, ok, err := nativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, nil, 1)
	if err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(second wide) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeFixedSingleTokenAttention(second wide) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)
	if err := Eval(second, secondKeys, secondValues); err != nil {
		t.Fatalf("Eval(second wide) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), float32Fill(2*headDim, 3))
	floatSliceApprox(t, secondKeys.Floats()[headDim:2*headDim], float32Fill(headDim, 3))
	floatSliceApprox(t, secondValues.Floats()[headDim:2*headDim], float32Fill(headDim, 4))
}

func TestDecode_nativeFixedSingleTokenAttentionWideGate_Good(t *testing.T) {
	target := "nativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	query := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	keyCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	key := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	value := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	offset := FromValue(0)
	defer Free(query, keyCache, valueCache, key, value, offset)

	if nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, nil) {
		t.Fatal("nativeFixedSingleTokenAttentionAvailable(512 ungated, nil) = true, want false")
	}
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION", "1")
	if !nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, nil) {
		t.Fatal("nativeFixedSingleTokenAttentionAvailable(512 sdpa gate, nil) = false, want true")
	}
}

func TestDecode_nativeFixedSingleTokenAttention_Bad(t *testing.T) {
	target := "nativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, _, _, ok, err := nativeFixedSingleTokenAttention(nil, nil, nil, nil, nil, nil, nil, 1); ok || err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeFixedSingleTokenAttention_Ugly(t *testing.T) {
	target := "nativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	query := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	keyCache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	valueCache := Zeros([]int32{1, 2, 4, 2}, DTypeFloat32)
	key := FromValues([]float32{1, 0}, 1, 1, 1, 2)
	value := FromValues([]float32{10, 0}, 1, 1, 1, 2)
	offset := FromValue(0)
	defer Free(query, keyCache, valueCache, key, value, offset)

	if _, _, _, ok, err := nativeFixedSingleTokenAttention(query, keyCache, valueCache, key, value, offset, nil, 1); ok || err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(mismatched cache heads) = ok %v err %v, want unsupported without error", ok, err)
	}

	wideQuery := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	wideKeyCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	wideValueCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	wideKey := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	wideValue := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	defer Free(wideQuery, wideKeyCache, wideValueCache, wideKey, wideValue)
	if _, _, _, ok, err := nativeFixedSingleTokenAttention(wideQuery, wideKeyCache, wideValueCache, wideKey, wideValue, offset, nil, 1); ok || err != nil {
		t.Fatalf("nativeFixedSingleTokenAttention(512-wide heads without matmul gate) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4FixedOwnerAttentionBlock_Good(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	identity := func() *Array {
		return FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *Array { return FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          NewLinear(identity(), nil),
		KProj:          NewLinear(identity(), nil),
		VProj:          NewLinear(identity(), nil),
		OProj:          NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	fixed := NewFixedKVCache(4)
	paged := NewPagedKVCache(4, 2)
	defer fixed.Reset()
	defer paged.Reset()

	fixedX := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	pagedX := fixedX.Clone()
	defer Free(fixedX, pagedX)

	got, gotKV, ok, err := nativeGemma4FixedOwnerAttentionBlock(fixedX, fixed, nil, attention, cfg)
	if err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionBlock() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedOwnerAttentionBlock() ok = false, want true")
	}
	want, wantKV := attention.forward(pagedX, paged, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer Free(got, want)
	defer gotKV.free()
	defer wantKV.free()
	if !gotKV.Fixed {
		t.Fatal("nativeGemma4FixedOwnerAttentionBlock() did not return fixed shared KV")
	}
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(got/want) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4FixedOwnerAttentionBlockQ4_Good(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionBlock q4"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	q4Identity := func() *Linear {
		const dim = 64
		quantized := make([]uint8, dim*dim)
		for i := 0; i < dim; i++ {
			quantized[i*dim+i] = 1
		}
		weight := FromValues(packMLXAffineQ4TestRows(t, quantized), dim, dim/8)
		scales := FromValues(float32Fill(dim, 1), dim, 1)
		biases := FromValues(float32Fill(dim, 0), dim, 1)
		return NewQuantizedLinear(weight, scales, biases, nil, 64, 4)
	}
	ones := func() *Array { return FromValues(float32Fill(64, 1), 64) }
	attention := &Gemma4Attention{
		QProj:          q4Identity(),
		KProj:          q4Identity(),
		VProj:          q4Identity(),
		OProj:          q4Identity(),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        64,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 64,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        64,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	values := make([]float32, 64)
	values[0] = 0.25
	values[1] = -0.5
	values[2] = 0.125
	fixed := NewFixedKVCache(4)
	paged := NewPagedKVCache(4, 2)
	mask := fixedSingleTokenCausalMaskFromHost(1, 4, 0)
	fixedX := FromValues(values, 1, 1, 64)
	pagedX := fixedX.Clone()
	defer fixed.Reset()
	defer paged.Reset()
	defer Free(mask, fixedX, pagedX)

	got, gotKV, ok, err := nativeGemma4FixedOwnerAttentionBlock(fixedX, fixed, mask, attention, cfg)
	if err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionBlock(q4) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedOwnerAttentionBlock(q4) ok = false, want true")
	}
	want, wantKV := attention.forward(pagedX, paged, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer Free(got, want)
	defer gotKV.free()
	defer wantKV.free()
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(q4 got/want) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4FixedOwnerAttentionResidualBlock_Good(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionResidualBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	identity := func() *Array {
		return FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *Array { return FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          NewLinear(identity(), nil),
		KProj:          NewLinear(identity(), nil),
		VProj:          NewLinear(identity(), nil),
		OProj:          NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	fixed := NewFixedKVCache(4)
	paged := NewPagedKVCache(4, 2)
	residual := FromValues([]float32{1, 2}, 1, 1, 2)
	fixedX := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	pagedX := fixedX.Clone()
	postNorm := FromValues([]float32{1, 1}, 2)
	defer fixed.Reset()
	defer paged.Reset()
	defer Free(residual, fixedX, pagedX, postNorm)

	got, gotKV, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(residual, fixedX, fixed, nil, attention, postNorm, cfg)
	if err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionResidualBlock() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedOwnerAttentionResidualBlock() ok = false, want true")
	}
	attnOut, wantKV := attention.forward(pagedX, paged, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	attnNormed := RMSNorm(attnOut, postNorm, 1e-6)
	want := Add(residual, attnNormed)
	defer Free(got, attnOut, attnNormed, want)
	defer gotKV.free()
	defer wantKV.free()
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(got/want) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4FixedOwnerAttentionResidualBlockQ4_Good(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionResidualBlock q4"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	q4Identity := func() *Linear {
		const dim = 64
		quantized := make([]uint8, dim*dim)
		for i := 0; i < dim; i++ {
			quantized[i*dim+i] = 1
		}
		weight := FromValues(packMLXAffineQ4TestRows(t, quantized), dim, dim/8)
		scales := FromValues(float32Fill(dim, 1), dim, 1)
		biases := FromValues(float32Fill(dim, 0), dim, 1)
		return NewQuantizedLinear(weight, scales, biases, nil, 64, 4)
	}
	ones := func() *Array { return FromValues(float32Fill(64, 1), 64) }
	attention := &Gemma4Attention{
		QProj:          q4Identity(),
		KProj:          q4Identity(),
		VProj:          q4Identity(),
		OProj:          q4Identity(),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        64,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 64,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        64,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	values := make([]float32, 64)
	values[0] = 0.25
	values[1] = -0.5
	values[2] = 0.125
	residualValues := float32Fill(64, 0)
	residualValues[0] = 1
	residualValues[1] = 2
	fixed := NewFixedKVCache(4)
	paged := NewPagedKVCache(4, 2)
	mask := fixedSingleTokenCausalMaskFromHost(1, 4, 0)
	residual := FromValues(residualValues, 1, 1, 64)
	fixedX := FromValues(values, 1, 1, 64)
	pagedX := fixedX.Clone()
	postNorm := ones()
	defer fixed.Reset()
	defer paged.Reset()
	defer Free(mask, residual, fixedX, pagedX, postNorm)

	got, gotKV, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(residual, fixedX, fixed, mask, attention, postNorm, cfg)
	if err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionResidualBlock(q4) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedOwnerAttentionResidualBlock(q4) ok = false, want true")
	}
	attnOut, wantKV := attention.forward(pagedX, paged, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	attnNormed := RMSNorm(attnOut, postNorm, 1e-6)
	want := Add(residual, attnNormed)
	defer Free(got, attnOut, attnNormed, want)
	defer gotKV.free()
	defer wantKV.free()
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(q4 got/want) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4FixedOwnerAttentionBlock_Bad(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, _, ok, err := nativeGemma4FixedOwnerAttentionBlock(nil, nil, nil, nil, nil); ok || err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionBlock(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4FixedOwnerAttentionResidualBlock_Bad(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionResidualBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, _, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(nil, nil, nil, nil, nil, nil, nil); ok || err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionResidualBlock(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4FixedOwnerAttentionBlock_Ugly(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	identity := func() *Array {
		return FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	attention := &Gemma4Attention{
		QProj:          NewLinear(identity(), nil),
		KProj:          NewLinear(identity(), nil),
		VProj:          NewLinear(identity(), nil),
		OProj:          NewLinear(identity(), nil),
		QNormScaled:    FromValues([]float32{1, 1}, 2),
		KNormScaled:    FromValues([]float32{1, 1}, 2),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
		UseKEqV:        true,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	fixed := NewFixedKVCache(4)
	x := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	defer fixed.Reset()
	defer Free(x)

	if _, _, ok, err := nativeGemma4FixedOwnerAttentionBlock(x, fixed, nil, attention, cfg); ok || err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionBlock(UseKEqV) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4FixedOwnerAttentionResidualBlock_Ugly(t *testing.T) {
	target := "nativeGemma4FixedOwnerAttentionResidualBlock"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	identity := func() *Array {
		return FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	attention := &Gemma4Attention{
		QProj:          NewLinear(identity(), nil),
		KProj:          NewLinear(identity(), nil),
		VProj:          NewLinear(identity(), nil),
		OProj:          NewLinear(identity(), nil),
		QNormScaled:    FromValues([]float32{1, 1}, 2),
		KNormScaled:    FromValues([]float32{1, 1}, 2),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	fixed := NewFixedKVCache(4)
	residual := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	x := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	postNorm := FromValues([]float32{1, 1}, 2)
	defer fixed.Reset()
	defer Free(residual, x, postNorm)

	if _, _, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(residual, x, fixed, nil, attention, postNorm, cfg); ok || err != nil {
		t.Fatalf("nativeGemma4FixedOwnerAttentionResidualBlock(mismatched residual) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4DecodeLayer_Good(t *testing.T) {
	target := "nativeGemma4DecodeLayer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer freeTestGemma4NativeLayer(layer)

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantCache := NewPagedKVCache(0, 2)
	want, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, wantPerLayer, sharedKV{}, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)
	defer wantKV.free()
	defer wantCache.Reset()

	enableNativeGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotCache := NewPagedKVCache(0, 2)
	got, gotKV, ok, err := nativeGemma4DecodeLayer(gotInput, gotCache, 1, 1, nil, gotPerLayer, sharedKV{}, layer, cfg, nil)
	if err != nil {
		t.Fatalf("nativeGemma4DecodeLayer() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4DecodeLayer() ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)
	defer gotKV.free()
	defer gotCache.Reset()

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(layer outputs) error = %v", err)
	}
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 2 {
		t.Fatalf("native layer shape = %v, want [1 1 2]", shape)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4DecodeLayer_Bad(t *testing.T) {
	target := "nativeGemma4DecodeLayer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative := enableNativeGemma4Layer
	enableNativeGemma4Layer = false
	t.Cleanup(func() { enableNativeGemma4Layer = oldNative })

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer freeTestGemma4NativeLayer(layer)

	if _, _, ok, err := nativeGemma4DecodeLayer(input, NewPagedKVCache(0, 2), 1, 1, nil, perLayer, sharedKV{}, layer, cfg, nil); ok || err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(gate off) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4DecodeLayer_EmptyPagedCacheBad(t *testing.T) {
	target := "nativeGemma4DecodeLayer empty paged cache"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative := enableNativeGemma4Layer
	enableNativeGemma4Layer = true
	t.Cleanup(func() { enableNativeGemma4Layer = oldNative })

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer freeTestGemma4NativeLayer(layer)

	if _, _, ok, err := nativeGemma4DecodeLayer(input, NewPagedKVCache(0, 2), 1, 1, nil, perLayer, sharedKV{}, layer, cfg, nil); ok || err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(empty paged cache) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4DecodeLayer_MoEGateOffBad(t *testing.T) {
	target := "nativeGemma4DecodeLayer MoE gate"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative := enableNativeGemma4Layer
	enableNativeGemma4Layer = true
	t.Cleanup(func() { enableNativeGemma4Layer = oldNative })

	layer := testGemma4NativeMoELayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	if _, _, ok, err := nativeGemma4DecodeLayer(input, NewPagedKVCache(0, 2), 1, 1, nil, perLayer, sharedKV{}, layer, cfg, nil); ok || err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(MoE gate off) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4DecodeLayer_Ugly(t *testing.T) {
	target := "nativeGemma4DecodeLayer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative := enableNativeGemma4Layer
	enableNativeGemma4Layer = true
	t.Cleanup(func() { enableNativeGemma4Layer = oldNative })

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	key := FromValues([]float32{0.1, 0.2}, 1, 1, 1, 2)
	value := FromValues([]float32{0.3, 0.4}, 1, 1, 1, 2)
	defer Free(input, perLayer, key, value)
	defer freeTestGemma4NativeLayer(layer)

	cache := NewPagedKVCache(1, 1)
	state := cache.UpdatePages(key, value, 1)
	defer state.Free()
	defer cache.Reset()

	if _, _, ok, err := nativeGemma4DecodeLayer(input, cache, 1, 1, nil, perLayer, sharedKV{}, layer, cfg, nil); ok || err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(trimming cache) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeGemma4DecodeLayer_MoEGood(t *testing.T) {
	target := "nativeGemma4DecodeLayer MoE"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "1"))
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeMoELayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantCache := NewPagedKVCache(0, 2)
	want, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, wantPerLayer, sharedKV{}, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)
	defer wantKV.free()
	defer wantCache.Reset()

	enableNativeGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotCache := NewPagedKVCache(0, 2)
	got, gotKV, ok, err := nativeGemma4DecodeLayer(gotInput, gotCache, 1, 1, nil, gotPerLayer, sharedKV{}, layer, cfg, nil)
	if err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(MoE) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4DecodeLayer(MoE) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)
	defer gotKV.free()
	defer gotCache.Reset()

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(native MoE layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4DecodeLayer_FixedCacheMoEGood(t *testing.T) {
	target := "nativeGemma4DecodeLayer fixed cache MoE"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "1"))
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeMoELayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantCache := NewFixedKVCache(4)
	wantCacheK, wantCacheV := wantCache.Update(prevK, prevV, 1)
	Free(wantCacheK, wantCacheV)
	want, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, wantPerLayer, sharedKV{}, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)
	defer wantKV.free()
	defer wantCache.Reset()

	enableNativeGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotCache := NewFixedKVCache(4)
	gotCacheK, gotCacheV := gotCache.Update(prevK, prevV, 1)
	Free(gotCacheK, gotCacheV)
	fixedMask := fixedSingleTokenCausalMaskFromHost(1, 4, gotCache.Offset())
	got, gotKV, ok, err := nativeGemma4DecodeLayer(gotInput, gotCache, 1, 1, nil, gotPerLayer, sharedKV{}, layer, cfg, fixedMask)
	if err != nil {
		t.Fatalf("nativeGemma4DecodeLayer(fixed cache MoE) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4DecodeLayer(fixed cache MoE) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, fixedMask, got)
	defer gotKV.free()
	defer gotCache.Reset()

	if !gotKV.Fixed {
		t.Fatal("native fixed-cache MoE layer returned non-fixed shared KV")
	}
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(native fixed-cache MoE layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_nativeGemma4FixedGreedyToken_Good(t *testing.T) {
	target := "nativeGemma4FixedGreedyToken"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "1"))
	requireMetalRuntime(t)

	cfg := testGemma4NativeLayerConfig()
	cfg.NumHiddenLayers = 2
	layers := []*Gemma4DecoderLayer{
		testGemma4NativeMoELayer(),
		testGemma4NativeLayer(),
	}
	model := &Gemma4Model{
		Cfg:               cfg,
		Layers:            layers,
		PreviousKVs:       []int32{0, 0},
		CacheIndexByLayer: []int32{0, -1},
		NormScaled:        FromValues([]float32{1, 1}, 2),
		Output: NewLinear(FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, 3, 2), nil),
	}
	defer closeGemma4(model)

	hidden := FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	perLayerInputs := []*Array{
		FromValues([]float32{0.1, 0.2}, 1, 1, 2),
		FromValues([]float32{-0.3, 0.4}, 1, 1, 2),
	}
	defer Free(hidden, perLayerInputs[0], perLayerInputs[1])

	wantCache := NewFixedKVCache(4)
	wantMasks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	defer wantMasks.Free()
	wantH := hidden.Clone()
	intermediates := make([]sharedKV, len(layers))
	for i, layer := range layers {
		var cache Cache
		var prev sharedKV
		if model.PreviousKVs[i] == int32(i) {
			cache = wantCache
		} else {
			prev = intermediates[int(model.PreviousKVs[i])]
		}
		fixedMask := wantMasks.ForLayer(cache, prev)
		nextH, kv := layer.forward(wantH, cache, 1, 1, nil, perLayerInputs[i], prev, cfg, fixedMask, nil, false)
		Free(wantH)
		wantH = nextH
		intermediates[i] = kv
	}
	defer Free(wantH)
	want, ok, err := nativeLastTokenGreedyToken(wantH, model.NormScaled, model.Output, cfg.RMSNormEps)
	if err != nil {
		t.Fatalf("nativeLastTokenGreedyToken(want) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeLastTokenGreedyToken(want) ok = false, want true")
	}
	defer Free(want)

	gotCache := NewFixedKVCache(4)
	gotMasks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	defer gotMasks.Free()
	gotHidden := hidden.Clone()
	got, ok, err := nativeGemma4FixedGreedyToken(gotHidden, perLayerInputs, []Cache{gotCache}, model, gotMasks)
	Free(gotHidden)
	if err != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedGreedyToken() ok = false, want true")
	}
	defer Free(got)
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(tokens) error = %v", err)
	}
	if gotID, wantID := got.Int(), want.Int(); gotID != wantID {
		t.Fatalf("token = %d, want %d", gotID, wantID)
	}
	if gotCache.Offset() != 1 || gotCache.Len() != 1 {
		t.Fatalf("got cache offset/len = %d/%d, want 1/1", gotCache.Offset(), gotCache.Len())
	}
}

func TestDecode_nativeGemma4FixedGreedyToken_NoPerLayerInputs_Good(t *testing.T) {
	target := "nativeGemma4FixedGreedyToken NoPerLayerInputs"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1"))
	requireMetalRuntime(t)

	cfg := testGemma4NativeLayerConfig()
	cfg.NumHiddenLayers = 1
	layer := testGemma4NativeLayer()
	model := &Gemma4Model{
		Cfg:               cfg,
		Layers:            []*Gemma4DecoderLayer{layer},
		PreviousKVs:       []int32{0},
		CacheIndexByLayer: []int32{0},
		NormScaled:        FromValues([]float32{1, 1}, 2),
		Output: NewLinear(FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, 3, 2), nil),
	}
	defer closeGemma4(model)

	hidden := FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	wantCache := NewFixedKVCache(4)
	wantMasks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	wantInput := hidden.Clone()
	fixedMask := wantMasks.ForLayer(wantCache, sharedKV{})
	wantH, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, nil, sharedKV{}, cfg, fixedMask, nil, false)
	Free(wantInput)
	defer Free(hidden, wantH)
	defer wantKV.free()
	defer wantCache.Reset()
	defer wantMasks.Free()
	want, ok, err := nativeLastTokenGreedyToken(wantH, model.NormScaled, model.Output, cfg.RMSNormEps)
	if err != nil {
		t.Fatalf("nativeLastTokenGreedyToken(want) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeLastTokenGreedyToken(want) ok = false, want true")
	}
	defer Free(want)

	gotCache := NewFixedKVCache(4)
	gotMasks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	gotHidden := hidden.Clone()
	got, ok, err := nativeGemma4FixedGreedyToken(gotHidden, nil, []Cache{gotCache}, model, gotMasks)
	Free(gotHidden)
	defer gotCache.Reset()
	defer gotMasks.Free()
	if err != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken(nil per-layer) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FixedGreedyToken(nil per-layer) ok = false, want true")
	}
	defer Free(got)
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(tokens) error = %v", err)
	}
	if gotID, wantID := got.Int(), want.Int(); gotID != wantID {
		t.Fatalf("token = %d, want %d", gotID, wantID)
	}
}

func TestDecode_nativeGemma4FixedGreedyToken_MoEGateSkip_Ugly(t *testing.T) {
	target := "nativeGemma4FixedGreedyToken MoEGateSkip"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "0"))
	t.Setenv("GO_MLX_TRACE_FORWARD_EVAL", "1")
	requireMetalRuntime(t)

	cfg := testGemma4NativeLayerConfig()
	cfg.NumHiddenLayers = 1
	layer := testGemma4NativeMoELayer()
	model := &Gemma4Model{
		Cfg:               cfg,
		Layers:            []*Gemma4DecoderLayer{layer},
		PreviousKVs:       []int32{0},
		CacheIndexByLayer: []int32{0},
		NormScaled:        FromValues([]float32{1, 1}, 2),
		Output: NewLinear(FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, 3, 2), nil),
	}
	defer closeGemma4(model)

	hidden := FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	cache := NewFixedKVCache(4)
	masks := newFixedGemma4AttentionMaskSet(1, 1, nil)
	defer Free(hidden, perLayer)
	defer cache.Reset()
	defer masks.Free()

	resetNativePhaseTraceEvents()
	got, ok, err := nativeGemma4FixedGreedyToken(hidden, []*Array{perLayer}, []Cache{cache}, model, masks)
	if err != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken() error = %v", err)
	}
	if ok || got != nil {
		t.Fatalf("nativeGemma4FixedGreedyToken() = ok %v token %v, want skip", ok, got)
	}
	events := takeNativePhaseTraceEvents()
	if len(events) != 1 || events[0].Name != "gemma4.model.greedy_token.skip" || events[0].Error != "layer 00: moe native layer is disabled" {
		t.Fatalf("events = %+v, want model greedy MoE gate skip", events)
	}
}

func TestDecode_compiledGemma4DecodeLayer_Good(t *testing.T) {
	target := "compiledGemma4DecodeLayer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer freeTestGemma4NativeLayer(layer)

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	want, _ := layer.forward(wantInput, nil, 1, 1, nil, wantPerLayer, wantPrev, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)

	enableCompiledGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	got, _, ok, err := compiledGemma4DecodeLayer(gotInput, nil, 1, 1, nil, gotPerLayer, gotPrev, layer, cfg, nil)
	if err != nil {
		t.Fatalf("compiledGemma4DecodeLayer() error = %v", err)
	}
	if !ok {
		t.Fatal("compiledGemma4DecodeLayer() ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(compiled layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_compiledGemma4DecodeLayer_UseKEqVGood(t *testing.T) {
	target := "compiledGemma4DecodeLayer UseKEqV"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeLayer()
	Free(layer.Attention.VProj.Weight)
	layer.Attention.VProj = &Linear{}
	layer.Attention.UseKEqV = true
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer freeTestGemma4NativeLayer(layer)

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	want, _ := layer.forward(wantInput, nil, 1, 1, nil, wantPerLayer, wantPrev, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)

	enableCompiledGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	got, _, ok, err := compiledGemma4DecodeLayer(gotInput, nil, 1, 1, nil, gotPerLayer, gotPrev, layer, cfg, nil)
	if err != nil {
		t.Fatalf("compiledGemma4DecodeLayer(UseKEqV) error = %v", err)
	}
	if !ok {
		t.Fatal("compiledGemma4DecodeLayer(UseKEqV) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(compiled UseKEqV layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_compiledGemma4DecodeLayer_FixedCacheGood(t *testing.T) {
	target := "compiledGemma4DecodeLayer fixed cache"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer freeTestGemma4NativeLayer(layer)

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantCache := NewFixedKVCache(4)
	wantCacheK, wantCacheV := wantCache.Update(prevK, prevV, 1)
	Free(wantCacheK, wantCacheV)
	want, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, wantPerLayer, sharedKV{}, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)
	defer wantKV.free()
	defer wantCache.Reset()

	enableCompiledGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotCache := NewFixedKVCache(4)
	gotCacheK, gotCacheV := gotCache.Update(prevK, prevV, 1)
	Free(gotCacheK, gotCacheV)
	got, gotKV, ok, err := compiledGemma4DecodeLayer(gotInput, gotCache, 1, 1, nil, gotPerLayer, sharedKV{}, layer, cfg, nil)
	if err != nil {
		t.Fatalf("compiledGemma4DecodeLayer(fixed cache) error = %v", err)
	}
	if !ok {
		t.Fatal("compiledGemma4DecodeLayer(fixed cache) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)
	defer gotKV.free()
	defer gotCache.Reset()

	if !gotKV.Fixed {
		t.Fatal("compiled fixed-cache layer returned non-fixed shared KV")
	}
	if state := gotCache.State(); len(state) != 2 || state[0].Dim(2) != 4 || state[1].Dim(2) != 4 {
		t.Fatalf("fixed cache state = %v, want full-capacity K/V", state)
	}
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(compiled fixed-cache layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_compiledGemma4DecodeLayer_MoEGood(t *testing.T) {
	target := "compiledGemma4DecodeLayer MoE"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeMoELayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	want, _ := layer.forward(wantInput, nil, 1, 1, nil, wantPerLayer, wantPrev, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)

	enableCompiledGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotPrev := sharedKV{Keys: prevK, Values: prevV, Offset: 1}
	got, _, ok, err := compiledGemma4DecodeLayer(gotInput, nil, 1, 1, nil, gotPerLayer, gotPrev, layer, cfg, nil)
	if err != nil {
		t.Fatalf("compiledGemma4DecodeLayer(MoE) error = %v", err)
	}
	if !ok {
		t.Fatal("compiledGemma4DecodeLayer(MoE) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, got)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(compiled MoE layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_compiledGemma4DecodeLayer_FixedCacheSharedMaskGood(t *testing.T) {
	target := "compiledGemma4DecodeLayer fixed cache shared mask"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldNative, oldCompiled := enableNativeGemma4Layer, enableCompiledGemma4Layer
	enableNativeGemma4Layer, enableCompiledGemma4Layer = false, false
	t.Cleanup(func() {
		enableNativeGemma4Layer, enableCompiledGemma4Layer = oldNative, oldCompiled
	})

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	prevK := FromValues([]float32{0.05, 0.1}, 1, 1, 1, 2)
	prevV := FromValues([]float32{0.2, -0.1}, 1, 1, 1, 2)
	defer Free(input, perLayer, prevK, prevV)
	defer freeTestGemma4NativeLayer(layer)

	wantInput := input.Clone()
	wantPerLayer := perLayer.Clone()
	wantCache := NewFixedKVCache(4)
	wantCacheK, wantCacheV := wantCache.Update(prevK, prevV, 1)
	Free(wantCacheK, wantCacheV)
	want, wantKV := layer.forward(wantInput, wantCache, 1, 1, nil, wantPerLayer, sharedKV{}, cfg, nil, nil, false)
	defer Free(wantInput, wantPerLayer, want)
	defer wantKV.free()
	defer wantCache.Reset()

	enableCompiledGemma4Layer = true
	gotInput := input.Clone()
	gotPerLayer := perLayer.Clone()
	gotCache := NewFixedKVCache(4)
	gotCacheK, gotCacheV := gotCache.Update(prevK, prevV, 1)
	Free(gotCacheK, gotCacheV)
	fixedMask := fixedSingleTokenCausalMaskFromHost(1, 4, gotCache.Offset())
	got, gotKV, ok, err := compiledGemma4DecodeLayer(gotInput, gotCache, 1, 1, nil, gotPerLayer, sharedKV{}, layer, cfg, fixedMask)
	if err != nil {
		t.Fatalf("compiledGemma4DecodeLayer(fixed cache shared mask) error = %v", err)
	}
	if !ok {
		t.Fatal("compiledGemma4DecodeLayer(fixed cache shared mask) ok = false, want true")
	}
	defer Free(gotInput, gotPerLayer, fixedMask, got)
	defer gotKV.free()
	defer gotCache.Reset()

	if !gotKV.Fixed {
		t.Fatal("compiled fixed-cache shared-mask layer returned non-fixed shared KV")
	}
	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval(compiled fixed-cache shared-mask layer outputs) error = %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestDecode_compiledGemma4DecodeLayer_Bad(t *testing.T) {
	target := "compiledGemma4DecodeLayer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)
	oldCompiled := enableCompiledGemma4Layer
	enableCompiledGemma4Layer = false
	t.Cleanup(func() { enableCompiledGemma4Layer = oldCompiled })

	layer := testGemma4NativeLayer()
	cfg := testGemma4NativeLayerConfig()
	input := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	perLayer := FromValues([]float32{0.1, 0.2}, 1, 1, 2)
	defer Free(input, perLayer)
	defer freeTestGemma4NativeLayer(layer)

	if _, _, ok, err := compiledGemma4DecodeLayer(input, NewPagedKVCache(0, 2), 1, 1, nil, perLayer, sharedKV{}, layer, cfg, nil); ok || err != nil {
		t.Fatalf("compiledGemma4DecodeLayer(gate off) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_gemma4PerLayerDecodeLayerUnavailableReason_Good(t *testing.T) {
	target := "gemma4PerLayerDecodeLayerUnavailableReason"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}

	cfg := &Gemma4TextConfig{HeadDim: 256, GlobalHeadDim: 512}
	layer := &Gemma4DecoderLayer{
		LayerType: "full_attention",
		Attention: &Gemma4Attention{HeadDim: 512},
	}
	const want = "full-attention global head dim requires model-level native boundary"
	if got := gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg); got != want {
		t.Fatalf("gemma4PerLayerDecodeLayerUnavailableReason(full global) = %q, want %q", got, want)
	}

	layer.LayerType = "sliding_attention"
	if got := gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg); got != "" {
		t.Fatalf("gemma4PerLayerDecodeLayerUnavailableReason(sliding) = %q, want empty", got)
	}

	layer.LayerType = "full_attention"
	cfg.GlobalHeadDim = cfg.HeadDim
	if got := gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg); got != "" {
		t.Fatalf("gemma4PerLayerDecodeLayerUnavailableReason(equal dims) = %q, want empty", got)
	}

	if got := gemma4PerLayerDecodeLayerUnavailableReason(nil, cfg); got != "" {
		t.Fatalf("gemma4PerLayerDecodeLayerUnavailableReason(nil layer) = %q, want empty", got)
	}
}

func BenchmarkGemma4PerLayerDecodeLayerUnavailableReason_FullGlobal(b *testing.B) {
	cfg := &Gemma4TextConfig{HeadDim: 256, GlobalHeadDim: 512}
	layer := &Gemma4DecoderLayer{
		LayerType: "full_attention",
		Attention: &Gemma4Attention{HeadDim: 512},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg) == "" {
			b.Fatal("expected per-layer full-attention boundary to be unavailable")
		}
	}
}

func TestDecode_validateGemma4LayerOutputs_Good(t *testing.T) {
	target := "validateGemma4LayerOutputs"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	out := FromValue(float32(1))
	key := FromValue(float32(2))
	value := FromValue(float32(3))
	defer Free(out, key, value)

	if err := validateGemma4LayerOutputs("test", []*Array{out}, false); err != nil {
		t.Fatalf("validateGemma4LayerOutputs(shared) error = %v", err)
	}
	if err := validateGemma4LayerOutputs("test", []*Array{out, key, value}, true); err != nil {
		t.Fatalf("validateGemma4LayerOutputs(owner) error = %v", err)
	}
}

func TestDecode_validateGemma4LayerOutputs_Bad(t *testing.T) {
	target := "validateGemma4LayerOutputs"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}

	if err := validateGemma4LayerOutputs("test", nil, false); err == nil {
		t.Fatal("validateGemma4LayerOutputs(nil shared) error = nil, want error")
	}
	if err := validateGemma4LayerOutputs("test", []*Array{nil}, false); err == nil {
		t.Fatal("validateGemma4LayerOutputs(nil array) error = nil, want error")
	}
	if err := validateGemma4LayerOutputs("test", []*Array{{}}, false); err == nil {
		t.Fatal("validateGemma4LayerOutputs(invalid array) error = nil, want error")
	}
	if err := validateGemma4LayerOutputs("test", []*Array{{}}, true); err == nil {
		t.Fatal("validateGemma4LayerOutputs(owner short outputs) error = nil, want error")
	}
}

func TestDecode_validateGemma4LayerOutputShapes_Good(t *testing.T) {
	target := "validateGemma4LayerOutputShapes"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	x := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	out := FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	prevK := FromValues(float32Fill(8, 0.1), 1, 1, 4, 2)
	prevV := FromValues(float32Fill(8, 0.2), 1, 1, 4, 2)
	newK := FromValues(float32Fill(8, 0.3), 1, 1, 4, 2)
	newV := FromValues(float32Fill(8, 0.4), 1, 1, 4, 2)
	defer Free(x, out, prevK, prevV, newK, newV)

	if err := validateGemma4LayerOutputShapes("test", x, out, newK, newV, prevK, prevV, true, true); err != nil {
		t.Fatalf("validateGemma4LayerOutputShapes(fixed owner) error = %v", err)
	}
	if err := validateGemma4LayerOutputShapes("test", x, out, nil, nil, prevK, prevV, false, true); err != nil {
		t.Fatalf("validateGemma4LayerOutputShapes(shared) error = %v", err)
	}
}

func TestDecode_validateGemma4LayerOutputShapes_Bad(t *testing.T) {
	target := "validateGemma4LayerOutputShapes"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	x := FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	out := FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	badOut := FromValues([]float32{0.5, 0.25}, 1, 2, 1)
	prevK := FromValues(float32Fill(8, 0.1), 1, 1, 4, 2)
	prevV := FromValues(float32Fill(8, 0.2), 1, 1, 4, 2)
	shortK := FromValues([]float32{0.3, 0.4}, 1, 1, 1, 2)
	shortV := FromValues([]float32{0.5, 0.6}, 1, 1, 1, 2)
	defer Free(x, out, badOut, prevK, prevV, shortK, shortV)

	if err := validateGemma4LayerOutputShapes("test", x, badOut, nil, nil, prevK, prevV, false, true); err == nil {
		t.Fatal("validateGemma4LayerOutputShapes(bad output shape) error = nil, want error")
	}
	if err := validateGemma4LayerOutputShapes("test", x, out, shortK, shortV, prevK, prevV, true, true); err == nil {
		t.Fatal("validateGemma4LayerOutputShapes(short fixed K/V) error = nil, want error")
	}
}

func testGemma4NativeLayerConfig() *Gemma4TextConfig {
	return &Gemma4TextConfig{
		RMSNormEps:        1e-6,
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           2,
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
