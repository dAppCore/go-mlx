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
		t.Fatal("nativeGreedyDecodeAvailable() = false, want true for unprobed Greedy single-step logits")
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
	target := "NativeLastTokenOutputLogits"
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

	got, ok, err := NativeLastTokenOutputLogits(hidden, normWeight, output, 1e-6, 30)
	if err != nil {
		t.Fatalf("NativeLastTokenOutputLogits() error = %v", err)
	}
	if !ok {
		t.Fatal("NativeLastTokenOutputLogits() ok = false, want true")
	}
	defer Free(got)

	normed := RMSNorm(hidden, normWeight, 1e-6)
	wantRaw := output.Forward(normed)
	// Reference softcap: 30·tanh(x/30). gemma4.logitSoftcap (which moved to
	// package gemma4 with the Gemma 4 architecture) is this exact expression on
	// the same public metal ops; reconstructed inline so this metal-kernel test
	// (NativeLastTokenOutputLogits + nativeGreedyDecodeToken, both metal-internal)
	// stays in package metal.
	wantScaled := MulScalar(wantRaw, 1.0/30)
	wantCapped := Tanh(wantScaled)
	want := MulScalar(wantCapped, 30)
	Free(normed, wantRaw, wantScaled, wantCapped)
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
	target := "NativeLastTokenOutputLogits"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}

	if _, ok, err := NativeLastTokenOutputLogits(nil, nil, nil, 1e-6, 30); ok || err != nil {
		t.Fatalf("NativeLastTokenOutputLogits(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeLastTokenOutputLogits_Ugly(t *testing.T) {
	target := "NativeLastTokenOutputLogits"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	requireMetalRuntime(t)

	hidden := FromValues([]float32{1, 2}, 1, 1, 2)
	normWeight := FromValues([]float32{1, 1}, 2)
	outputWeight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	output := NewLinear(outputWeight, nil)
	defer Free(hidden, normWeight, outputWeight)

	if _, ok, err := NativeLastTokenOutputLogits(hidden, normWeight, output, 1e-5, 30); ok || err != nil {
		t.Fatalf("NativeLastTokenOutputLogits(eps=1e-5) = ok %v err %v, want unsupported without error", ok, err)
	}
	if _, ok, err := NativeLastTokenOutputLogits(hidden, normWeight, output, 1e-6, 0); ok || err != nil {
		t.Fatalf("NativeLastTokenOutputLogits(softcap=0) = ok %v err %v, want unsupported without error", ok, err)
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

func TestDecode_nativeLastTokenQuantizedOutputBitsAvailable_Good(t *testing.T) {
	target := "nativeLastTokenQuantizedOutputBitsAvailable"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	for _, tc := range []struct {
		bits int
		want bool
	}{
		{bits: 4, want: true},
		{bits: 6, want: false},
		{bits: 8, want: true},
	} {
		if got := nativeLastTokenQuantizedOutputBitsAvailable(tc.bits); got != tc.want {
			t.Fatalf("nativeLastTokenQuantizedOutputBitsAvailable(%d) = %v, want %v", tc.bits, got, tc.want)
		}
	}
}

func TestDecode_nativeMLPGELU_Good(t *testing.T) {
	target := "nativeMLPGELU"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	previous := enableNativeMLPGELU
	enableNativeMLPGELU = true
	t.Cleanup(func() { enableNativeMLPGELU = previous })
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
	activated := GeluGateMul(gate, up)
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
	previous := enableNativeMLPGELU
	enableNativeMLPGELU = true
	t.Cleanup(func() { enableNativeMLPGELU = previous })
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

func TestDecode_nativeFixedSingleTokenAttention_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention"
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

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(first) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(first) ok = false, want true")
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

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(second) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(second) ok = false, want true")
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

func TestDecode_nativeFixedSlidingSingleTokenAttention_Good(t *testing.T) {
	target := "NativeFixedSlidingSingleTokenAttention"
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

	got, gotKeys, gotValues, ok, err := NativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex, 1)
	if err != nil {
		t.Fatalf("NativeFixedSlidingSingleTokenAttention error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSlidingSingleTokenAttention ok = false, want true")
	}
	if !got.Valid() || !gotKeys.Valid() || !gotValues.Valid() {
		t.Fatalf("NativeFixedSlidingSingleTokenAttention returned invalid outputs: out=%v keys=%v values=%v", got.Valid(), gotKeys.Valid(), gotValues.Valid())
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
	target := "NativeFixedSlidingSingleTokenAttention Gemma4E2BShape"
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
		for i := range window {
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

	got, gotKeys, gotValues, ok, err := NativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex, 0.0625)
	if err != nil {
		t.Fatalf("NativeFixedSlidingSingleTokenAttention(E2B shape) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSlidingSingleTokenAttention(E2B shape) ok = false, want true")
	}
	defer Free(got, gotKeys, gotValues)
	if err := Eval(got, gotKeys, gotValues); err != nil {
		t.Fatalf("Eval(E2B shape) error = %v", err)
	}
	if !got.Valid() || !gotKeys.Valid() || !gotValues.Valid() {
		t.Fatalf("NativeFixedSlidingSingleTokenAttention(E2B shape) returned invalid outputs: out=%v keys=%v values=%v", got.Valid(), gotKeys.Valid(), gotValues.Valid())
	}
	if got.Dim(1) != int(QH) || gotKeys.Dim(2) != int(window) || gotValues.Dim(2) != int(window) {
		t.Fatalf("E2B shape outputs = out heads:%d key window:%d value window:%d, want heads:%d window:%d", got.Dim(1), gotKeys.Dim(2), gotValues.Dim(2), QH, window)
	}
}

func TestDecode_nativeFixedSingleTokenAttentionWide_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	t.Cleanup(SetFixedAttentionDiagnostics(false, true, false))
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

	first, firstKeys, firstValues, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, keyA, valueA, offsetA, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(first wide) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(first wide) ok = false, want true")
	}
	defer Free(first, firstKeys, firstValues)
	if err := Eval(first, firstKeys, firstValues); err != nil {
		t.Fatalf("Eval(first wide) error = %v", err)
	}
	floatSliceApprox(t, first.Floats(), float32Fill(2*headDim, 2))
	floatSliceApprox(t, firstKeys.Floats()[:headDim], float32Fill(headDim, 1))
	floatSliceApprox(t, firstValues.Floats()[:headDim], float32Fill(headDim, 2))

	second, secondKeys, secondValues, ok, err := NativeFixedSingleTokenAttention(query, firstKeys, firstValues, keyB, valueB, offsetB, nil, 1)
	if err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(second wide) error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedSingleTokenAttention(second wide) ok = false, want true")
	}
	defer Free(second, secondKeys, secondValues)
	if err := Eval(second, secondKeys, secondValues); err != nil {
		t.Fatalf("Eval(second wide) error = %v", err)
	}
	floatSliceApprox(t, second.Floats(), float32Fill(2*headDim, 3))
	floatSliceApprox(t, secondKeys.Floats()[headDim:2*headDim], float32Fill(headDim, 3))
	floatSliceApprox(t, secondValues.Floats()[headDim:2*headDim], float32Fill(headDim, 4))
}

func TestDecode_nativeFixedSingleTokenAttentionWideDiagnostic_Good(t *testing.T) {
	target := "NativeFixedSingleTokenAttention"
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
	restore := SetFixedAttentionDiagnostics(true, false, false)
	t.Cleanup(restore)
	if !nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, nil) {
		t.Fatal("nativeFixedSingleTokenAttentionAvailable(512 sdpa diagnostic, nil) = false, want true")
	}
}

func TestDecode_nativeFixedSingleTokenAttention_Bad(t *testing.T) {
	target := "NativeFixedSingleTokenAttention"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	if _, _, _, ok, err := NativeFixedSingleTokenAttention(nil, nil, nil, nil, nil, nil, nil, 1); ok || err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(nil) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_nativeFixedSingleTokenAttention_Ugly(t *testing.T) {
	target := "NativeFixedSingleTokenAttention"
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

	if _, _, _, ok, err := NativeFixedSingleTokenAttention(query, keyCache, valueCache, key, value, offset, nil, 1); ok || err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(mismatched cache heads) = ok %v err %v, want unsupported without error", ok, err)
	}

	wideQuery := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	wideKeyCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	wideValueCache := Zeros([]int32{1, 1, 4, 512}, DTypeFloat32)
	wideKey := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	wideValue := Zeros([]int32{1, 1, 1, 512}, DTypeFloat32)
	defer Free(wideQuery, wideKeyCache, wideValueCache, wideKey, wideValue)
	if _, _, _, ok, err := NativeFixedSingleTokenAttention(wideQuery, wideKeyCache, wideValueCache, wideKey, wideValue, offset, nil, 1); ok || err != nil {
		t.Fatalf("NativeFixedSingleTokenAttention(512-wide heads without matmul gate) = ok %v err %v, want unsupported without error", ok, err)
	}
}

func TestDecode_validateGemma4LayerOutputShapes_Good(t *testing.T) {
	target := "ValidateLayerOutputShapes"
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

	if err := ValidateLayerOutputShapes("test", x, out, newK, newV, prevK, prevV, true, true); err != nil {
		t.Fatalf("ValidateLayerOutputShapes(fixed owner) error = %v", err)
	}
	if err := ValidateLayerOutputShapes("test", x, out, nil, nil, prevK, prevV, false, true); err != nil {
		t.Fatalf("ValidateLayerOutputShapes(shared) error = %v", err)
	}
}

func TestDecode_validateGemma4LayerOutputShapes_Bad(t *testing.T) {
	target := "ValidateLayerOutputShapes"
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

	if err := ValidateLayerOutputShapes("test", x, badOut, nil, nil, prevK, prevV, false, true); err == nil {
		t.Fatal("ValidateLayerOutputShapes(bad output shape) error = nil, want error")
	}
	if err := ValidateLayerOutputShapes("test", x, out, shortK, shortV, prevK, prevV, true, true); err == nil {
		t.Fatal("ValidateLayerOutputShapes(short fixed K/V) error = nil, want error")
	}
}

// TestDecode_NativeFixedMultiTokenAttention_Good proves the fused L-token
// fixed-cache attention (the speculative-verify fast path) matches the
// reference: write the L new K/V into the cache at writeIndices, then masked
// SDPA over the updated cache. Correctness here lets the MTP verify swap the
// op-by-op cache-update + SDPA for one fused, compiled call.
func TestDecode_NativeFixedMultiTokenAttention_Good(t *testing.T) {
	requireMetalRuntime(t)

	const (
		heads = 2
		dim   = 2
		cap   = 6
		L     = 2
		off   = 1 // new tokens occupy cache rows off..off+L-1
		scale = 0.5
	)
	// keyCache/valueCache: [1, heads, cap, dim], zero.
	cacheLen := heads * cap * dim
	keyCache := FromValues(make([]float32, cacheLen), 1, heads, cap, dim)
	valueCache := FromValues(make([]float32, cacheLen), 1, heads, cap, dim)
	// q/k/v: [1, heads, L, dim].
	q := FromValues([]float32{0.5, -0.5, 0.25, 0.75, -1, 0.5, 0.5, 1}, 1, heads, L, dim)
	k := FromValues([]float32{1, 0, 0, 1, 0.5, 0.5, -0.5, 1}, 1, heads, L, dim)
	v := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, heads, L, dim)
	// writeIndices: [1, heads, L, dim] int32, value off+i for token i.
	idx := make([]int32, heads*L*dim)
	for h := 0; h < heads; h++ {
		for i := 0; i < L; i++ {
			for d := 0; d < dim; d++ {
				idx[(h*L+i)*dim+d] = int32(off + i)
			}
		}
	}
	writeIndices := FromValues(idx, 1, heads, L, dim)
	// mask: [1,1,L,cap], 0 where j <= off+i else -1e9.
	mdata := make([]float32, L*cap)
	for i := 0; i < L; i++ {
		for j := 0; j < cap; j++ {
			if j <= off+i {
				mdata[i*cap+j] = 0
			} else {
				mdata[i*cap+j] = -1e9
			}
		}
	}
	mask := FromValues(mdata, 1, 1, L, cap)
	defer Free(keyCache, valueCache, q, k, v, writeIndices, mask)

	got, newKeys, newValues, ok, err := NativeFixedMultiTokenAttention(q, keyCache, valueCache, k, v, writeIndices, mask, scale)
	if err != nil {
		t.Fatalf("NativeFixedMultiTokenAttention error = %v", err)
	}
	if !ok {
		t.Fatal("NativeFixedMultiTokenAttention ok = false, want true")
	}
	defer Free(got, newKeys, newValues)

	// Reference: write k/v into the cache at writeIndices, masked SDPA(q*scale).
	wantKeys := PutAlongAxis(keyCache, writeIndices, k, 2)
	wantValues := PutAlongAxis(valueCache, writeIndices, v, 2)
	scaledQ := MulScalar(q, scale)
	wantOut := ScaledDotProductAttentionWithMask(scaledQ, wantKeys, wantValues, mask, 1.0)
	defer Free(wantKeys, wantValues, scaledQ, wantOut)

	if err := Eval(got, newKeys, newValues, wantOut, wantKeys, wantValues); err != nil {
		t.Fatalf("Eval error = %v", err)
	}
	assertFloat32SliceClose(t, got.Floats(), wantOut.Floats(), 1e-4)
	assertFloat32SliceClose(t, newKeys.Floats(), wantKeys.Floats(), 1e-4)
	assertFloat32SliceClose(t, newValues.Floats(), wantValues.Floats(), 1e-4)
	if shape := got.Shape(); len(shape) != 4 || shape[2] != L {
		t.Fatalf("out shape = %v, want [...%d.]", shape, L)
	}
}
