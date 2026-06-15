// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"slices"
	"testing"

	core "dappco.re/go"

	coreio "dappco.re/go/io"
)

type loraResolverTestModel struct {
	modelType string
	layers    map[int]map[string]*Linear
}

func newLoRAResolverTestModel(layer0 map[string]*Linear) *loraResolverTestModel {
	return &loraResolverTestModel{layers: map[int]map[string]*Linear{0: layer0}}
}

func (m *loraResolverTestModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (m *loraResolverTestModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (m *loraResolverTestModel) NewCache() []Cache                                  { return nil }
func (m *loraResolverTestModel) NumLayers() int                                     { return len(m.layers) }
func (m *loraResolverTestModel) Tokenizer() *Tokenizer                              { return nil }
func (m *loraResolverTestModel) ModelType() string {
	if m != nil && m.modelType != "" {
		return m.modelType
	}
	return "lora_resolver_test"
}
func (m *loraResolverTestModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }
func (m *loraResolverTestModel) ResolveLoRALinear(layerIdx int, projPath string) *Linear {
	if m == nil || m.layers == nil {
		return nil
	}
	return m.layers[layerIdx][projPath]
}

func TestLora_NewLoRALinear_Good(t *testing.T) {
	// Create a simple base linear layer: [4, 8] weight
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0) // rank=4, alpha=8

	// Check dimensions
	aShape := lora.A.Shape()
	bShape := lora.B.Shape()

	if aShape[0] != 4 || aShape[1] != 8 {
		t.Errorf("A shape = %v, want [4, 8]", aShape)
	}
	if bShape[0] != 4 || bShape[1] != 4 {
		t.Errorf("B shape = %v, want [4, 4]", bShape)
	}

	// Scale should be alpha/rank = 8/4 = 2
	if math.Abs(float64(lora.Scale)-2.0) > 1e-5 {
		t.Errorf("Scale = %f, want 2.0", lora.Scale)
	}

	// B should be all zeros (LoRA starts as identity)
	Materialize(lora.B)
	bFloats := lora.B.Floats()
	for i, v := range bFloats {
		if v != 0 {
			t.Errorf("B[%d] = %f, want 0", i, v)
		}
	}
}

func TestLora_LoRALinear_ForwardMatchesBase_Good(t *testing.T) {
	// With B=0, LoRA forward should equal base forward
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)

	// Random input [1, 3, 8]
	x := RandomNormal(0, 1, []int32{1, 3, 8}, DTypeFloat32)
	Materialize(x)

	baseOut := base.Forward(x)
	loraOut := lora.Forward(x)
	Materialize(baseOut, loraOut)

	// Should be identical since B is zero
	baseFloats := baseOut.Floats()
	loraFloats := loraOut.Floats()

	if len(baseFloats) != len(loraFloats) {
		t.Fatalf("output sizes differ: base=%d, lora=%d", len(baseFloats), len(loraFloats))
	}

	for i := range baseFloats {
		diff := math.Abs(float64(baseFloats[i] - loraFloats[i]))
		if diff > 1e-4 {
			t.Errorf("output[%d] differs: base=%f, lora=%f", i, baseFloats[i], loraFloats[i])
		}
	}
}

func TestLora_LoRALinear_ForwardWithAdapter_Good(t *testing.T) {
	// Set A and B to known values and verify output changes
	w := Zeros([]int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 2, 4.0) // rank=2, alpha=4, scale=2

	// Set A to identity-like: [[1,0,0,...], [0,1,0,...]]
	a := Zeros([]int32{2, 8}, DTypeFloat32)
	// Set B to ones: [[1,1], [1,1], [1,1], [1,1]]
	b := FromValues([]float32{
		1, 1,
		1, 1,
		1, 1,
		1, 1,
	}, 4, 2)
	Materialize(a, b)
	lora.A = a
	lora.B = b

	// With base=0, A=0, output should also be 0 (scale * x@0@B^T = 0)
	x := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 1, 8)
	result := lora.Forward(x)
	Materialize(result)

	// base(x) = 0 (zero weights), lora = scale * (x @ A^T) @ B^T
	// A is zeros, so x @ A^T = [0, 0], then @ B^T = [0,0,0,0]
	for _, v := range result.Floats() {
		if v != 0 {
			t.Errorf("expected 0 with zero A, got %f", v)
		}
	}
}

func TestLora_LoRALinear_ParamCount_Good(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{64, 128}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 8, 16.0) // rank=8
	// A: [8, 128] = 1024, B: [64, 8] = 512, total = 1536
	expected := 8*128 + 64*8
	if lora.ParamCount() != expected {
		t.Errorf("ParamCount = %d, want %d", lora.ParamCount(), expected)
	}
}

func TestLora_LoRALinear_TrainableParams_Good(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)
	params := lora.TrainableParams()

	if len(params) != 2 {
		t.Fatalf("TrainableParams returned %d arrays, want 2", len(params))
	}

	// First is A, second is B
	if params[0].Shape()[0] != 4 || params[0].Shape()[1] != 8 {
		t.Errorf("param[0] (A) shape = %v, want [4, 8]", params[0].Shape())
	}
	if params[1].Shape()[0] != 4 || params[1].Shape()[1] != 4 {
		t.Errorf("param[1] (B) shape = %v, want [4, 4]", params[1].Shape())
	}
}

func TestLora_NormalizeConfig_RFCAliases_Good(t *testing.T) {
	cfg := normalizeLoRAConfig(LoRAConfig{
		Rank:         8,
		Scale:        1.5,
		TargetLayers: []string{"q_proj", "v_proj"},
	})

	if cfg.Alpha != 12 {
		t.Fatalf("Alpha = %f, want 12", cfg.Alpha)
	}
	if cfg.Scale != 1.5 {
		t.Fatalf("Scale = %f, want 1.5", cfg.Scale)
	}
	if len(cfg.TargetKeys) != 2 || cfg.TargetKeys[0] != "q_proj" || cfg.TargetKeys[1] != "v_proj" {
		t.Fatalf("TargetKeys = %v, want RFC aliases copied", cfg.TargetKeys)
	}
	if cfg.DType != DTypeFloat32 {
		t.Fatalf("DType = %v, want float32 default", cfg.DType)
	}
}

type loraStepTestModel struct {
	layer *LoRALinear
}

func (m *loraStepTestModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *loraStepTestModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array {
	zero := Zeros([]int32{1, 1}, DTypeFloat32)
	logit := Add(m.layer.A, m.layer.B)
	pair := Concatenate([]*Array{zero, logit}, 1)
	logits := Reshape(pair, 1, 1, 2)
	Free(zero, logit, pair)
	return logits
}

func (m *loraStepTestModel) NewCache() []Cache                   { return nil }
func (m *loraStepTestModel) NumLayers() int                      { return 1 }
func (m *loraStepTestModel) Tokenizer() *Tokenizer               { return nil }
func (m *loraStepTestModel) ModelType() string                   { return "lora-step-test" }
func (m *loraStepTestModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func TestLora_Regularization_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValues([]float32{3, 4}, 1, 2)
	b := FromValues([]float32{0, 2}, 1, 2)
	reg := loraRegularization([]*Array{a, b}, 0.1)
	defer Free(a, b, reg)
	Materialize(reg)

	// 0.1 * (mean([9,16]) + mean([0,4])) = 0.1 * (12.5 + 2.0) = 1.45
	if got := reg.Float(); math.Abs(got-1.45) > 1e-5 {
		t.Fatalf("regularization = %f, want 1.45", got)
	}
}

func TestLora_Step_AppliesLambdaRegularization_Good(t *testing.T) {
	requireMetalRuntime(t)

	newAdapter := func(lambda float32) (*LoRAAdapter, *LoRALinear) {
		layer := &LoRALinear{
			A:     FromValues([]float32{0.25}, 1, 1),
			B:     FromValues([]float32{0.5}, 1, 1),
			Scale: 1,
			Rank:  1,
			Alpha: 1,
		}
		return &LoRAAdapter{
			Layers: map[string]*LoRALinear{"model.layers.0.self_attn.q_proj": layer},
			Config: LoRAConfig{Lambda: lambda},
			Model:  &loraStepTestModel{layer: layer},
		}, layer
	}

	batch := Batch{
		Tokens: [][]int{{0}},
		Length: []int{1},
	}
	targets := [][]int{{1}}
	opt := NewAdamW(&AdamWConfig{LearningRate: 0})

	plain, plainLayer := newAdapter(0)
	defer Free(plainLayer.A, plainLayer.B)
	plainLoss := plain.Step(batch, targets, opt)
	if plainLoss == nil {
		t.Fatal("plain Step returned nil loss")
	}
	defer Free(plainLoss)
	Materialize(plainLoss)

	regularized, regularizedLayer := newAdapter(0.5)
	defer Free(regularizedLayer.A, regularizedLayer.B)
	regularizedLoss := regularized.Step(batch, targets, opt)
	if regularizedLoss == nil {
		t.Fatal("regularized Step returned nil loss")
	}
	defer Free(regularizedLoss)
	Materialize(regularizedLoss)

	if got, want := regularizedLoss.Float(), plainLoss.Float(); got <= want {
		t.Fatalf("regularized loss = %f, want > plain loss %f", got, want)
	}
}

func TestLora_Step_EmitsTrainingProbe_Good(t *testing.T) {
	requireMetalRuntime(t)

	layer := &LoRALinear{
		A:     FromValues([]float32{0.25}, 1, 1),
		B:     FromValues([]float32{0.5}, 1, 1),
		Scale: 1,
		Rank:  1,
		Alpha: 1,
	}
	defer Free(layer.A, layer.B)
	var events []ProbeEvent
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{"model.layers.0.self_attn.q_proj": layer},
		Config: LoRAConfig{
			ProbeSink: ProbeSinkFunc(func(event ProbeEvent) {
				events = append(events, event)
			}),
		},
		Model: &loraStepTestModel{layer: layer},
	}
	batch := Batch{
		Tokens: [][]int{{0}},
		Length: []int{1},
	}
	targets := [][]int{{1}}
	opt := NewAdamW(&AdamWConfig{LearningRate: 0.01})

	loss := adapter.Step(batch, targets, opt)
	if loss == nil {
		t.Fatal("Step returned nil loss")
	}
	defer Free(loss)

	if len(events) != 1 {
		t.Fatalf("probe events len = %d, want 1", len(events))
	}
	if events[0].Kind != ProbeEventTraining || events[0].Phase != ProbePhaseTraining {
		t.Fatalf("probe event = %+v", events[0])
	}
	if events[0].Training == nil || events[0].Training.Step != 1 || events[0].Training.Loss <= 0 {
		t.Fatalf("training payload = %+v", events[0].Training)
	}
	if events[0].Training.LearningRate != 0.01 {
		t.Fatalf("learning rate = %f, want 0.01", events[0].Training.LearningRate)
	}
}

// Loss is the validation lane: the same masked cross-entropy Step minimises,
// with NOTHING moved — params identical after the call, no probe emission,
// and no lambda regularization term (validation reads the data surface).
func TestLora_Loss_ForwardOnlyNoUpdate_Good(t *testing.T) {
	requireMetalRuntime(t)

	layer := &LoRALinear{
		A:     FromValues([]float32{0.25}, 1, 1),
		B:     FromValues([]float32{0.5}, 1, 1),
		Scale: 1,
		Rank:  1,
		Alpha: 1,
	}
	defer Free(layer.A, layer.B)
	var events []ProbeEvent
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{"model.layers.0.self_attn.q_proj": layer},
		Config: LoRAConfig{
			// Both set deliberately: Loss must ignore the probe sink AND
			// the regularization term.
			Lambda:    0.5,
			ProbeSink: ProbeSinkFunc(func(event ProbeEvent) { events = append(events, event) }),
		},
		Model: &loraStepTestModel{layer: layer},
	}
	batch := Batch{
		Tokens: [][]int{{0}},
		Length: []int{1},
	}
	targets := [][]int{{1}}

	before := layer.A
	loss := adapter.Loss(batch, targets)
	if loss == nil {
		t.Fatal("Loss returned nil")
	}
	defer Free(loss)
	Materialize(loss)
	got := loss.Float()
	if got <= 0 {
		t.Fatalf("val loss = %f, want > 0", got)
	}
	if layer.A != before {
		t.Fatal("Loss replaced adapter params — validation must not move weights")
	}
	if len(events) != 0 {
		t.Fatalf("probe events = %d, want 0 — Loss is silent, the training loop owns emission", len(events))
	}

	// Same surface as a plain unregularized Step: a zero-LR Step on a
	// lambda-0 twin must read the identical loss value.
	plainLayer := &LoRALinear{
		A:     FromValues([]float32{0.25}, 1, 1),
		B:     FromValues([]float32{0.5}, 1, 1),
		Scale: 1,
		Rank:  1,
		Alpha: 1,
	}
	defer Free(plainLayer.A, plainLayer.B)
	plain := &LoRAAdapter{
		Layers: map[string]*LoRALinear{"model.layers.0.self_attn.q_proj": plainLayer},
		Model:  &loraStepTestModel{layer: plainLayer},
	}
	stepLoss := plain.Step(batch, targets, NewAdamW(&AdamWConfig{LearningRate: 0}))
	if stepLoss == nil {
		t.Fatal("Step returned nil loss")
	}
	defer Free(stepLoss)
	Materialize(stepLoss)
	if want := stepLoss.Float(); math.Abs(got-want) > 1e-6 {
		t.Fatalf("val loss = %f, want %f (Step's data loss)", got, want)
	}
}

func TestLora_Loss_NilAndEmpty_Ugly(t *testing.T) {
	var nilAdapter *LoRAAdapter
	if nilAdapter.Loss(Batch{}, nil) != nil {
		t.Fatal("nil adapter must return nil loss")
	}
	adapter := &LoRAAdapter{Model: &loraStepTestModel{}}
	if adapter.Loss(Batch{}, nil) != nil {
		t.Fatal("empty batch must return nil loss")
	}
}

func TestLora_BatchLengths_Good(t *testing.T) {
	lengths, maxLen := batchLengths(
		Batch{
			Tokens: [][]int{
				{1, 2, 3, 4},
				{5, 6, 7},
			},
			Length: []int{3, 2},
		},
		[][]int{
			{9, 8, 7, 6},
			{4, 3, 2},
		},
	)

	if maxLen != 3 {
		t.Fatalf("maxLen = %d, want 3", maxLen)
	}
	if len(lengths) != 2 || lengths[0] != 3 || lengths[1] != 2 {
		t.Fatalf("lengths = %v, want [3 2]", lengths)
	}
}

func TestLora_BatchLossMask_UsesExplicitMask_Good(t *testing.T) {
	requireMetalRuntime(t)

	mask := batchLossMaskForBatch(
		Batch{
			LossMask: [][]float32{
				{0, 1, 1},
				{1},
			},
		},
		[]int32{3, 2},
		3,
	)
	defer Free(mask)
	Materialize(mask)

	got := mask.Floats()
	want := []float32{0, 1, 1, 1, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("loss mask len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("loss mask[%d] = %f, want %f; full mask %v", i, got[i], want[i], got)
		}
	}
}

func TestLora_FreeReplacedArrays_PreservesLiveReferences_Good(t *testing.T) {
	requireMetalRuntime(t)

	keep := FromValues([]float32{1, 2}, 1, 2)
	replaced := FromValues([]float32{3, 4}, 1, 2)
	current := FromValues([]float32{5, 6}, 1, 2)

	freeReplacedArrays([]*Array{keep, replaced}, []*Array{keep, current})
	defer Free(keep, current)

	Materialize(keep, current)

	if got := keep.Floats(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("keep = %v, want [1 2]", got)
	}
	if got := current.Floats(); len(got) != 2 || got[0] != 5 || got[1] != 6 {
		t.Fatalf("current = %v, want [5 6]", got)
	}
}

func TestLora_LoRALinear_GradientFlows_Good(t *testing.T) {
	// Verify that gradients flow through the LoRA path
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)
	x := RandomNormal(0, 1, []int32{1, 2, 8}, DTypeFloat32)
	Materialize(x)

	// Loss function: sum of LoRA output (differentiating w.r.t. A and B)
	lossFn := func(inputs []*Array) []*Array {
		lora.A = inputs[0]
		lora.B = inputs[1]
		out := lora.Forward(x)
		return []*Array{SumAll(out)}
	}

	grad := ValueAndGrad(lossFn, 0, 1) // grad w.r.t. A and B
	defer grad.Free()

	values, grads, err := grad.Apply(lora.A, lora.B)
	if err != nil {
		t.Fatalf("ValueAndGrad failed: %v", err)
	}

	Materialize(append(values, grads...)...)

	// Loss should be a scalar
	loss := values[0].Float()
	t.Logf("loss = %f", loss)

	// Gradients should be non-zero (A has random init, B is zero but gets grad)
	gradA := grads[0]
	gradB := grads[1]

	aGradFloats := gradA.Floats()
	bGradFloats := gradB.Floats()

	hasNonZeroA := false
	for _, v := range aGradFloats {
		if v != 0 {
			hasNonZeroA = true
			break
		}
	}

	hasNonZeroB := false
	for _, v := range bGradFloats {
		if v != 0 {
			hasNonZeroB = true
			break
		}
	}

	// A gradient might be zero if B is zero (since dL/dA depends on B)
	// But B gradient should be non-zero since A is random
	if !hasNonZeroB {
		t.Error("gradient for B is all zeros — gradients not flowing")
	}
	t.Logf("gradA has non-zero: %v, gradB has non-zero: %v", hasNonZeroA, hasNonZeroB)
}

func TestLora_RandomNormal_Good(t *testing.T) {
	arr := RandomNormal(0, 1, []int32{100}, DTypeFloat32)
	Materialize(arr)

	floats := arr.Floats()
	if len(floats) != 100 {
		t.Fatalf("RandomNormal returned %d elements, want 100", len(floats))
	}

	// Check rough statistics: mean should be near 0, values should have spread
	var sum float64
	for _, f := range floats {
		sum += float64(f)
	}
	mean := sum / 100
	if math.Abs(mean) > 0.5 { // generous tolerance for 100 samples
		t.Errorf("mean = %f, expected near 0", mean)
	}
}

func TestLora_SaveSafetensors_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8, 9, 10}, 3, 2)
	Materialize(a, b)

	path := t.TempDir() + "/test.safetensors"
	err := SaveSafetensors(path, map[string]*Array{
		"layer.lora_a": a,
		"layer.lora_b": b,
	})
	if err != nil {
		t.Fatalf("SaveSafetensors failed: %v", err)
	}

	// Verify file exists
	fileInfo, err := coreio.Local.Stat(path)
	if err != nil {
		t.Fatalf("saved file not found: %v", err)
	}
	if fileInfo.Size() == 0 {
		t.Error("saved file is empty")
	}

	// Load it back
	loaded, err := LoadAllSafetensors(path)
	if err != nil {
		t.Fatalf("LoadAllSafetensors: %v", err)
	}
	Materialize(loaded["layer.lora_a"], loaded["layer.lora_b"])

	aLoaded := loaded["layer.lora_a"].Floats()
	bLoaded := loaded["layer.lora_b"].Floats()

	expectedA := []float32{1, 2, 3, 4}
	expectedB := []float32{5, 6, 7, 8, 9, 10}

	for i, v := range expectedA {
		if aLoaded[i] != v {
			t.Errorf("loaded A[%d] = %f, want %f", i, aLoaded[i], v)
		}
	}
	for i, v := range expectedB {
		if bLoaded[i] != v {
			t.Errorf("loaded B[%d] = %f, want %f", i, bLoaded[i], v)
		}
	}
}

func TestLora_LoRAAdapter_Save_Good(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": NewLoRALinear(base, 4, 8.0),
		},
		Config: DefaultLoRAConfig(),
	}

	path := t.TempDir() + "/adapter.safetensors"
	err := adapter.Save(path)
	if err != nil {
		t.Fatalf("Adapter.Save failed: %v", err)
	}

	// Load and verify
	loaded, err := LoadAllSafetensors(path)
	if err != nil {
		t.Fatalf("LoadAllSafetensors: %v", err)
	}
	aKey := "model.layers.0.self_attn.q_proj.lora_a"
	bKey := "model.layers.0.self_attn.q_proj.lora_b"

	if _, ok := loaded[aKey]; !ok {
		t.Errorf("missing key %s in saved adapter", aKey)
	}
	if _, ok := loaded[bKey]; !ok {
		t.Errorf("missing key %s in saved adapter", bKey)
	}

	config, err := parseAdapterConfig(core.JoinPath(core.PathDir(path), "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if config.Rank != 8 {
		t.Fatalf("config rank = %d, want 8", config.Rank)
	}
	if config.Alpha != 16 {
		t.Fatalf("config alpha = %f, want 16", config.Alpha)
	}
	if config.NumLayers != 1 {
		t.Fatalf("config num_layers = %d, want 1", config.NumLayers)
	}
	found := slices.Contains(config.TargetKeys, "self_attn.q_proj")
	if !found {
		t.Fatalf("config target keys = %v, want self_attn.q_proj", config.TargetKeys)
	}
}

func TestLora_LoRAAdapter_Save_Directory_Good(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.3.self_attn.q_proj": NewLoRALinear(base, 4, 8.0),
		},
		Config: LoRAConfig{
			Rank:       4,
			Alpha:      8,
			TargetKeys: []string{"q_proj"},
		},
	}

	dir := t.TempDir()
	if err := adapter.Save(dir); err != nil {
		t.Fatalf("Adapter.Save failed: %v", err)
	}

	if _, err := coreio.Local.Stat(core.JoinPath(dir, "adapter.safetensors")); err != nil {
		t.Fatalf("saved adapter weights not found: %v", err)
	}
	config, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if config.NumLayers != 4 {
		t.Fatalf("config num_layers = %d, want 4", config.NumLayers)
	}
}

func TestLora_DefaultLoRAConfig_Good(t *testing.T) {
	cfg := DefaultLoRAConfig()
	if cfg.Rank != 8 {
		t.Errorf("Rank = %d, want 8", cfg.Rank)
	}
	if cfg.Alpha != 16 {
		t.Errorf("Alpha = %f, want 16", cfg.Alpha)
	}
	if len(cfg.TargetKeys) != 2 {
		t.Errorf("TargetKeys = %v, want [q_proj, v_proj]", cfg.TargetKeys)
	}
}

func TestLora_NormalizeConfig_NegativeRankUsesDefault_Good(t *testing.T) {
	cfg := normalizeLoRAConfig(LoRAConfig{Rank: -4})
	if cfg.Rank != 8 {
		t.Fatalf("Rank = %d, want 8", cfg.Rank)
	}
	if cfg.Scale != 2 {
		t.Fatalf("Scale = %f, want 2", cfg.Scale)
	}
}

func sameStringSlice(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func loraTestValues(start float32, count int) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = start + float32(i)/10
	}
	return values
}

// --- parseLoRAWeightName ---

func TestLora_ParseLoRAWeightName_Good(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantIdx  int
		wantProj string
		wantSuf  string
	}{
		{
			"standard_lora_a",
			"layers.0.self_attn.q_proj.lora_a",
			0, "self_attn.q_proj", "lora_a",
		},
		{
			"standard_lora_b",
			"layers.5.self_attn.v_proj.lora_b",
			5, "self_attn.v_proj", "lora_b",
		},
		{
			"with_model_prefix",
			"model.layers.12.self_attn.q_proj.lora_a",
			12, "self_attn.q_proj", "lora_a",
		},
		{
			"k_proj",
			"layers.3.self_attn.k_proj.lora_b",
			3, "self_attn.k_proj", "lora_b",
		},
		{
			"o_proj",
			"layers.7.self_attn.o_proj.lora_a",
			7, "self_attn.o_proj", "lora_a",
		},
		{
			"peft_uppercase_lora_a_weight",
			"model.layers.0.self_attn.q_proj.lora_A.weight",
			0, "self_attn.q_proj", "lora_a",
		},
		{
			"peft_suffix_lora_b_weight",
			"model.layers.0.q_proj.lora_B.weight",
			0, "q_proj", "lora_b",
		},
		{
			"peft_base_model_prefix",
			"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
			0, "self_attn.q_proj", "lora_a",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, proj, suf := parseLoRAWeightName(tt.input)
			if idx != tt.wantIdx {
				t.Errorf("layerIdx = %d, want %d", idx, tt.wantIdx)
			}
			if proj != tt.wantProj {
				t.Errorf("projPath = %q, want %q", proj, tt.wantProj)
			}
			if suf != tt.wantSuf {
				t.Errorf("suffix = %q, want %q", suf, tt.wantSuf)
			}
		})
	}
}

func TestLora_ParseLoRAWeightName_Bad(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"no_lora_suffix", "layers.0.self_attn.q_proj.weight"},
		{"no_layers_prefix", "self_attn.q_proj.lora_a"},
		{"empty", ""},
		{"just_layers", "layers."},
		{"no_dot_after_idx", "layers.0lora_a"},
		{"non_numeric_idx", "layers.abc.self_attn.q_proj.lora_a"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, _, _ := parseLoRAWeightName(tt.input)
			if idx != -1 {
				t.Errorf("expected -1 for %q, got %d", tt.input, idx)
			}
		})
	}
}

// --- parseAdapterConfig ---

func TestLora_ParseAdapterConfig_Good(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"rank": 16,
		"alpha": 32.0,
		"num_layers": 4,
		"lora_layers": ["self_attn.q_proj", "self_attn.v_proj"]
	}`
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), cfg)

	parsed, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if parsed.Rank != 16 {
		t.Errorf("Rank = %d, want 16", parsed.Rank)
	}
	if parsed.Alpha != 32.0 {
		t.Errorf("Alpha = %f, want 32.0", parsed.Alpha)
	}
	if parsed.NumLayers != 4 {
		t.Errorf("NumLayers = %d, want 4", parsed.NumLayers)
	}
	if len(parsed.TargetKeys) != 2 {
		t.Errorf("TargetKeys = %v, want 2 entries", parsed.TargetKeys)
	}
}

func TestLora_ParseAdapterConfig_Good_Defaults(t *testing.T) {
	dir := t.TempDir()
	// Minimal config — rank and alpha should get defaults.
	cfg := `{}`
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), cfg)

	parsed, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if parsed.Rank != 8 {
		t.Errorf("default Rank = %d, want 8", parsed.Rank)
	}
	if parsed.Alpha != 16.0 {
		t.Errorf("default Alpha = %f, want 16.0 (2 * rank)", parsed.Alpha)
	}
	if parsed.Scale != 2.0 {
		t.Errorf("default Scale = %f, want 2.0", parsed.Scale)
	}
}

func TestLora_ParseAdapterConfig_Good_PEFTAliases(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"r":4,"lora_alpha":12,"target_modules":["q_proj","k_proj","v_proj","o_proj"]}`
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), cfg)

	parsed, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if parsed.Rank != 4 {
		t.Fatalf("Rank = %d, want PEFT r", parsed.Rank)
	}
	if parsed.Alpha != 12 {
		t.Fatalf("Alpha = %f, want PEFT lora_alpha", parsed.Alpha)
	}
	wantTargets := []string{"q_proj", "k_proj", "v_proj", "o_proj"}
	if !sameStringSlice(parsed.TargetKeys, wantTargets) {
		t.Fatalf("TargetKeys = %v, want PEFT target_modules %v", parsed.TargetKeys, wantTargets)
	}
}

func TestLora_ParseAdapterConfig_UsesSharedTargetPrecedence_Good(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"rank": 4,
		"scale": 2,
		"target_keys": ["explicit"],
		"target_modules": ["peft"],
		"lora_layers": ["mlx-lm"]
	}`
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), cfg)

	parsed, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("parseAdapterConfig: %v", err)
	}
	if parsed.Alpha != 8 || parsed.Scale != 2 {
		t.Fatalf("alpha/scale = %f/%f, want scale-derived alpha", parsed.Alpha, parsed.Scale)
	}
	if !sameStringSlice(parsed.TargetKeys, []string{"explicit"}) {
		t.Fatalf("TargetKeys = %v, want shared explicit target_keys precedence", parsed.TargetKeys)
	}
}

func TestLora_ParseAdapterConfig_Bad_MissingFile(t *testing.T) {
	_, err := parseAdapterConfig("/nonexistent/adapter_config.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLora_ParseAdapterConfig_Bad_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), "{broken")

	_, err := parseAdapterConfig(core.JoinPath(dir, "adapter_config.json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- loadAdapterWeights ---

func TestLora_LoadAdapterWeights_Bad_NoFiles(t *testing.T) {
	dir := t.TempDir()
	_, err := loadAdapterWeights(dir)
	if err == nil {
		t.Fatal("expected error for directory with no safetensors files")
	}
}

func TestLora_LoadAdapterWeights_Good(t *testing.T) {
	dir := t.TempDir()

	// Save a small adapter file.
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	Materialize(a, b)

	err := SaveSafetensors(core.JoinPath(dir, "adapters.safetensors"), map[string]*Array{
		"layers.0.self_attn.q_proj.lora_a": a,
		"layers.0.self_attn.q_proj.lora_b": b,
	})
	if err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	weights, err := loadAdapterWeights(dir)
	if err != nil {
		t.Fatalf("loadAdapterWeights: %v", err)
	}
	if len(weights) != 2 {
		t.Errorf("loaded %d weights, want 2", len(weights))
	}
	if _, ok := weights["layers.0.self_attn.q_proj.lora_a"]; !ok {
		t.Error("missing lora_a weight")
	}
	if _, ok := weights["layers.0.self_attn.q_proj.lora_b"]; !ok {
		t.Error("missing lora_b weight")
	}
}

// --- applyLoadedLoRA integration ---

func TestLora_ApplyLoadedLoRA_Good_SaveAndReload(t *testing.T) {
	// Create a simple base Linear layer and save LoRA weights for it,
	// then load them back with applyLoadedLoRA.

	// Create a small "model" with 1 layer and known dimensions.
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	linear := NewLinear(w, nil)

	// Train a LoRA on this linear, then save.
	lora := NewLoRALinear(linear, 4, 8.0)
	// Set A and B to non-zero values so we can verify they load correctly.
	newA := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
		1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
		2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
	}, 4, 8) // [rank=4, in=8]
	newB := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}, 4, 4) // [out=4, rank=4]
	Materialize(newA, newB)
	lora.A = newA
	lora.B = newB

	// Save the adapter package using the public LoRA save path.
	adapterDir := t.TempDir()
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": lora,
		},
		Config: LoRAConfig{
			Rank:       4,
			Alpha:      8,
			TargetKeys: []string{"q_proj"},
		},
	}
	if err := adapter.Save(adapterDir); err != nil {
		t.Fatalf("adapter.Save: %v", err)
	}

	// Now create a fresh linear with the same base weights (no LoRA).
	linear2 := NewLinear(w, nil)
	if linear2.LoRA != nil {
		t.Fatal("fresh linear should not have LoRA")
	}

	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": linear2})

	// Apply the loaded adapter.
	err := applyLoadedLoRA(qwen, adapterDir)
	if err != nil {
		t.Fatalf("applyLoadedLoRA: %v", err)
	}

	// Verify LoRA was injected.
	if linear2.LoRA == nil {
		t.Fatal("LoRA should have been injected into q_proj")
	}

	// Verify rank and scale.
	if linear2.LoRA.Rank != 4 {
		t.Errorf("Rank = %d, want 4", linear2.LoRA.Rank)
	}
	expectedScale := float32(8.0) / float32(4) // alpha / rank = 2.0
	if math.Abs(float64(linear2.LoRA.Scale-expectedScale)) > 1e-5 {
		t.Errorf("Scale = %f, want %f", linear2.LoRA.Scale, expectedScale)
	}

	// Verify the loaded A weights match what we saved.
	Materialize(linear2.LoRA.A, linear2.LoRA.B)
	loadedA := linear2.LoRA.A.Floats()
	origA := newA.Floats()
	if len(loadedA) != len(origA) {
		t.Fatalf("A size mismatch: %d vs %d", len(loadedA), len(origA))
	}
	for i := range origA {
		if math.Abs(float64(loadedA[i]-origA[i])) > 1e-5 {
			t.Errorf("A[%d] = %f, want %f", i, loadedA[i], origA[i])
			break
		}
	}

	// Verify the loaded B weights match.
	loadedB := linear2.LoRA.B.Floats()
	origB := newB.Floats()
	if len(loadedB) != len(origB) {
		t.Fatalf("B size mismatch: %d vs %d", len(loadedB), len(origB))
	}
	for i := range origB {
		if math.Abs(float64(loadedB[i]-origB[i])) > 1e-5 {
			t.Errorf("B[%d] = %f, want %f", i, loadedB[i], origB[i])
			break
		}
	}
}

func TestLora_LoadLoRAAdapter_ReturnsAdapter_Good(t *testing.T) {
	requireMetalRuntime(t)

	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	sourceLinear := NewLinear(w, nil)
	sourceAdapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": NewLoRALinear(sourceLinear, 2, 4),
		},
		Config: LoRAConfig{Rank: 2, Alpha: 4, TargetKeys: []string{"q_proj"}},
	}
	adapterDir := t.TempDir()
	if err := sourceAdapter.Save(adapterDir); err != nil {
		t.Fatalf("sourceAdapter.Save: %v", err)
	}

	targetLinear := NewLinear(w, nil)
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})

	loaded, err := loadLoRAAdapter(qwen, adapterDir)
	if err != nil {
		t.Fatalf("loadLoRAAdapter: %v", err)
	}
	if loaded == nil {
		t.Fatal("loadLoRAAdapter returned nil adapter")
	}
	if loaded.Model != qwen {
		t.Fatal("loaded adapter should retain target model for resume")
	}
	if loaded.Layers["model.layers.0.self_attn.q_proj"] == nil {
		t.Fatalf("loaded adapter layers = %v, want q_proj entry", loaded.SortedNames())
	}
	if targetLinear.LoRA == nil {
		t.Fatal("target q_proj should have an attached LoRA adapter")
	}
	if loaded.Config.Rank != 2 || loaded.Config.Alpha != 4 || loaded.Config.Scale != 2 {
		t.Fatalf("loaded config = %+v, want rank=2 alpha=4 scale=2", loaded.Config)
	}
}

func TestLora_LoadLoRAAdapter_PEFTConfigAliases_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"r":4,"lora_alpha":12,"target_modules":["q_proj"]}`); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	a := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
		1.7, 1.8, 1.9, 2.0,
		2.1, 2.2, 2.3, 2.4,
		2.5, 2.6, 2.7, 2.8,
		2.9, 3.0, 3.1, 3.2,
	}, 4, 8)
	b := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}, 4, 4)
	Materialize(a, b)
	if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), map[string]*Array{
		"model.layers.0.self_attn.q_proj.lora_a": a,
		"model.layers.0.self_attn.q_proj.lora_b": b,
	}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	Free(a, b)

	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	defer Free(w)
	targetLinear := NewLinear(w, nil)
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})

	loaded, err := loadLoRAAdapter(qwen, dir)
	if err != nil {
		t.Fatalf("loadLoRAAdapter: %v", err)
	}
	if targetLinear.LoRA == nil {
		t.Fatal("target q_proj should have an attached LoRA adapter")
	}
	if loaded.Config.Rank != 4 || loaded.Config.Alpha != 12 || loaded.Config.Scale != 3 {
		t.Fatalf("loaded config = %+v, want PEFT rank=4 alpha=12 scale=3", loaded.Config)
	}
	if !sameStringSlice(loaded.Config.TargetKeys, []string{"q_proj"}) {
		t.Fatalf("loaded target keys = %v, want PEFT target_modules", loaded.Config.TargetKeys)
	}
	if targetLinear.LoRA.Rank != 4 || targetLinear.LoRA.Alpha != 12 || targetLinear.LoRA.Scale != 3 {
		t.Fatalf("attached LoRA = rank:%d alpha:%f scale:%f, want PEFT config", targetLinear.LoRA.Rank, targetLinear.LoRA.Alpha, targetLinear.LoRA.Scale)
	}
}

func TestLora_LoadLoRAAdapter_Gemma4PEFTWeightAliases_Good(t *testing.T) {
	requireMetalRuntime(t)

	for _, modelType := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"gemma4_unified_text",
		"Gemma4ForConditionalGeneration",
		"Gemma4UnifiedForConditionalGeneration",
		"Gemma4ForCausalLM",
		"Gemma4TextForCausalLM",
	} {
		t.Run(modelType, func(t *testing.T) {
			dir := t.TempDir()
			if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"r":2,"lora_alpha":6,"target_modules":["q_proj"]}`); err != nil {
				t.Fatalf("write adapter_config.json: %v", err)
			}

			a := FromValues([]float32{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,
			}, 2, 8)
			b := FromValues([]float32{
				0.1, 0.2,
				0.3, 0.4,
				0.5, 0.6,
				0.7, 0.8,
			}, 4, 2)
			Materialize(a, b)
			if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), map[string]*Array{
				"model.layers.0.q_proj.lora_A.weight": a,
				"model.layers.0.q_proj.lora_B.weight": b,
			}); err != nil {
				t.Fatalf("SaveSafetensors: %v", err)
			}
			Free(a, b)

			w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
			Materialize(w)
			defer Free(w)
			targetLinear := NewLinear(w, nil)
			gemma4Like := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})
			gemma4Like.modelType = modelType

			loaded, err := loadLoRAAdapter(gemma4Like, dir)
			if err != nil {
				t.Fatalf("loadLoRAAdapter: %v", err)
			}
			if targetLinear.LoRA == nil {
				t.Fatal("target Gemma4 q_proj should have an attached LoRA adapter")
			}
			if loaded.Layers["model.layers.0.self_attn.q_proj"] == nil {
				t.Fatalf("loaded adapter layers = %v, want canonical Gemma4 q_proj entry", loaded.SortedNames())
			}
			if !sameStringSlice(loaded.Config.TargetKeys, []string{"q_proj"}) {
				t.Fatalf("loaded target keys = %v, want PEFT target_modules", loaded.Config.TargetKeys)
			}
			if targetLinear.LoRA.Rank != 2 || targetLinear.LoRA.Alpha != 6 || targetLinear.LoRA.Scale != 3 {
				t.Fatalf("attached LoRA = rank:%d alpha:%f scale:%f, want PEFT config", targetLinear.LoRA.Rank, targetLinear.LoRA.Alpha, targetLinear.LoRA.Scale)
			}
		})
	}
}

func TestLora_LoadLoRAAdapter_Gemma4MoEExtendedTargets_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	targetKeys := []string{"router.proj", "per_layer_input_gate", "per_layer_projection"}
	if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"r":2,"lora_alpha":6,"target_modules":["router.proj","per_layer_input_gate","per_layer_projection"]}`); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	adapterWeights := make(map[string]*Array, len(targetKeys)*2)
	savedArrays := make([]*Array, 0, len(targetKeys)*2)
	for i, target := range targetKeys {
		a := FromValues(loraTestValues(float32(i)+0.1, 16), 2, 8)
		b := FromValues(loraTestValues(float32(i)+0.2, 8), 4, 2)
		Materialize(a, b)
		adapterWeights[core.Sprintf("model.layers.0.%s.lora_A.weight", target)] = a
		adapterWeights[core.Sprintf("model.layers.0.%s.lora_B.weight", target)] = b
		savedArrays = append(savedArrays, a, b)
	}
	if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), adapterWeights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	Free(savedArrays...)

	baseWeights := make([]*Array, 0, len(targetKeys))
	targetLinears := make(map[string]*Linear, len(targetKeys))
	for _, target := range targetKeys {
		w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
		baseWeights = append(baseWeights, w)
		targetLinears[target] = NewLinear(w, nil)
	}
	Materialize(baseWeights...)
	defer Free(baseWeights...)
	gemma4MoELike := newLoRAResolverTestModel(targetLinears)
	gemma4MoELike.modelType = "Gemma4ForConditionalGeneration"

	loaded, err := loadLoRAAdapter(gemma4MoELike, dir)
	if err != nil {
		t.Fatalf("loadLoRAAdapter: %v", err)
	}
	for _, target := range targetKeys {
		if loaded.Layers["model.layers.0."+target] == nil {
			t.Fatalf("loaded adapter layers = %v, want %s entry", loaded.SortedNames(), target)
		}
		if targetLinears[target].LoRA == nil {
			t.Fatalf("%s should have an attached LoRA adapter", target)
		}
		if targetLinears[target].LoRA.Rank != 2 || targetLinears[target].LoRA.Alpha != 6 || targetLinears[target].LoRA.Scale != 3 {
			t.Fatalf("%s LoRA = rank:%d alpha:%f scale:%f, want PEFT config",
				target,
				targetLinears[target].LoRA.Rank,
				targetLinears[target].LoRA.Alpha,
				targetLinears[target].LoRA.Scale,
			)
		}
	}
	if !sameStringSlice(loaded.Config.TargetKeys, targetKeys) {
		t.Fatalf("loaded target keys = %v, want PEFT extended target_modules", loaded.Config.TargetKeys)
	}
}

func TestLora_LoadLoRAAdapter_ShapeMismatch_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank":4,"alpha":8,"lora_layers":["self_attn.q_proj"]}`); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	a := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
		0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
		1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
	}, 4, 6)
	b := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}, 4, 4)
	Materialize(a, b)
	if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), map[string]*Array{
		"model.layers.0.self_attn.q_proj.lora_a": a,
		"model.layers.0.self_attn.q_proj.lora_b": b,
	}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	Free(a, b)

	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	defer Free(w)
	targetLinear := NewLinear(w, nil)
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})

	_, err := loadLoRAAdapter(qwen, dir)
	if err == nil {
		t.Fatal("expected shape mismatch error")
	}
	if !core.Contains(err.Error(), "shape mismatch") || !core.Contains(err.Error(), "self_attn.q_proj") {
		t.Fatalf("error = %v, want clear target shape mismatch", err)
	}
	if targetLinear.LoRA != nil {
		t.Fatal("target q_proj should not retain a LoRA adapter after shape mismatch")
	}
}

func TestLora_LoadLoRAAdapter_UnsupportedTarget_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank":4,"alpha":8,"lora_layers":["self_attn.q_proj","self_attn.nope"]}`); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	qA := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
		1.7, 1.8, 1.9, 2.0,
		2.1, 2.2, 2.3, 2.4,
		2.5, 2.6, 2.7, 2.8,
		2.9, 3.0, 3.1, 3.2,
	}, 4, 8)
	qB := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}, 4, 4)
	nopeA := FromValues([]float32{
		3.2, 3.1, 3.0, 2.9,
		2.8, 2.7, 2.6, 2.5,
		2.4, 2.3, 2.2, 2.1,
		2.0, 1.9, 1.8, 1.7,
		1.6, 1.5, 1.4, 1.3,
		1.2, 1.1, 1.0, 0.9,
		0.8, 0.7, 0.6, 0.5,
		0.4, 0.3, 0.2, 0.1,
	}, 4, 8)
	nopeB := FromValues([]float32{
		1.6, 1.5, 1.4, 1.3,
		1.2, 1.1, 1.0, 0.9,
		0.8, 0.7, 0.6, 0.5,
		0.4, 0.3, 0.2, 0.1,
	}, 4, 4)
	Materialize(qA, qB, nopeA, nopeB)
	if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), map[string]*Array{
		"model.layers.0.self_attn.q_proj.lora_a": qA,
		"model.layers.0.self_attn.q_proj.lora_b": qB,
		"model.layers.0.self_attn.nope.lora_a":   nopeA,
		"model.layers.0.self_attn.nope.lora_b":   nopeB,
	}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	Free(qA, qB, nopeA, nopeB)

	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	defer Free(w)
	targetLinear := NewLinear(w, nil)
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})

	loaded, err := loadLoRAAdapter(qwen, dir)
	if loaded != nil {
		t.Cleanup(loaded.Unload)
	}
	if err == nil {
		t.Fatal("expected unsupported target error")
	}
	if !core.Contains(err.Error(), "unsupported target") || !core.Contains(err.Error(), "self_attn.nope") {
		t.Fatalf("error = %v, want clear unsupported target", err)
	}
	if targetLinear.LoRA != nil {
		t.Fatal("target q_proj should not retain a LoRA adapter after unsupported target")
	}
}

func TestLora_LoadLoRAAdapter_UnsupportedQuantizedTarget_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank":2,"alpha":4,"lora_layers":["self_attn.q_proj"]}`); err != nil {
		t.Fatalf("write adapter_config.json: %v", err)
	}

	a := FromValues([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6,
	}, 2, 8)
	b := FromValues([]float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
		0.7, 0.8,
	}, 4, 2)
	Materialize(a, b)
	if err := SaveSafetensors(core.JoinPath(dir, "adapter.safetensors"), map[string]*Array{
		"model.layers.0.self_attn.q_proj.lora_a": a,
		"model.layers.0.self_attn.q_proj.lora_b": b,
	}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	Free(a, b)

	w := RandomNormal(0, 0.01, []int32{4, 1}, DTypeFloat32)
	scales := FromValues([]float32{1, 1, 1, 1}, 4, 1)
	Materialize(w, scales)
	defer Free(w, scales)
	targetLinear := NewQuantizedLinear(w, scales, nil, nil, 0, 6)
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": targetLinear})

	_, err := loadLoRAAdapter(qwen, dir)
	if err == nil {
		t.Fatal("expected unsupported quantized target error")
	}
	if !core.Contains(err.Error(), "unsupported quantized target") ||
		!core.Contains(err.Error(), "self_attn.q_proj") ||
		!core.Contains(err.Error(), "group_size=0") {
		t.Fatalf("error = %v, want clear unsupported quantized target with group size", err)
	}
	if targetLinear.LoRA != nil {
		t.Fatal("target q_proj should not retain a LoRA adapter after unsupported quantized target")
	}
}

func TestLora_ResolveLinear_QwenFamilyMLPTargets_Good(t *testing.T) {
	qProj := &Linear{}
	gateProj := &Linear{}
	upProj := &Linear{}
	downProj := &Linear{}
	model := newLoRAResolverTestModel(map[string]*Linear{
		"self_attn.q_proj": qProj,
		"mlp.gate_proj":    gateProj,
		"mlp.up_proj":      upProj,
		"mlp.down_proj":    downProj,
	})

	if got := resolveLinear(model, 0, "self_attn.q_proj"); got != qProj {
		t.Fatal("resolveLinear should return Qwen q_proj")
	}
	if got := resolveLinear(model, 0, "mlp.gate_proj"); got != gateProj {
		t.Fatal("resolveLinear should return Qwen mlp.gate_proj")
	}
	if got := resolveLinear(model, 0, "mlp.up_proj"); got != upProj {
		t.Fatal("resolveLinear should return Qwen mlp.up_proj")
	}
	if got := resolveLinear(model, 0, "mlp.down_proj"); got != downProj {
		t.Fatal("resolveLinear should return Qwen mlp.down_proj")
	}
}

func TestLora_ApplyLoadedLoRA_Bad_MissingConfig(t *testing.T) {
	dir := t.TempDir()
	// Write safetensors but no config.
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	Materialize(a)
	if err := SaveSafetensors(core.JoinPath(dir, "adapters.safetensors"), map[string]*Array{"x": a}); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	qwen := &loraResolverTestModel{}
	err := applyLoadedLoRA(qwen, dir)
	if err == nil {
		t.Fatal("expected error for missing adapter_config.json")
	}
	if !core.Contains(err.Error(), "adapter_config.json") {
		t.Fatalf("error = %v, want missing adapter_config.json context", err)
	}
}

func TestLora_ApplyLoadedLoRA_Bad_MissingSafetensors(t *testing.T) {
	dir := t.TempDir()
	// Write config but no safetensors.
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank": 8}`)

	qwen := &loraResolverTestModel{}
	err := applyLoadedLoRA(qwen, dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors")
	}
	if !core.Contains(err.Error(), "no .safetensors files found") {
		t.Fatalf("error = %v, want missing safetensors context", err)
	}
}

func TestLora_ApplyLoadedLoRA_Bad_NoMatchingLayers(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank": 4, "alpha": 8.0}`)

	// Save weights that reference layer 99 (which won't exist).
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	Materialize(a, b)
	SaveSafetensors(core.JoinPath(dir, "adapters.safetensors"), map[string]*Array{
		"layers.99.self_attn.q_proj.lora_a": a,
		"layers.99.self_attn.q_proj.lora_b": b,
	})

	qwen := newLoRAResolverTestModel(map[string]*Linear{
		"self_attn.q_proj": NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
	})
	err := applyLoadedLoRA(qwen, dir)
	if err == nil {
		t.Fatal("expected error when no layers are injected")
	}
}

// TestLora_ApplyLoadedLoRA_Good_ForwardProducesOutput validates that a model with a
// loaded LoRA adapter produces different output than the base model alone.
func TestLora_ApplyLoadedLoRA_Good_ForwardProducesOutput(t *testing.T) {
	// Create base linear [4, 8].
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	linear := NewLinear(w, nil)

	// Compute base output.
	x := RandomNormal(0, 1, []int32{1, 2, 8}, DTypeFloat32)
	Materialize(x)
	baseOut := linear.Forward(x)
	Materialize(baseOut)
	baseFloats := baseOut.Floats()

	// Create and save non-trivial adapter weights.
	rank := 4
	loraA := RandomNormal(0, 0.1, []int32{int32(rank), 8}, DTypeFloat32)
	loraB := RandomNormal(0, 0.1, []int32{4, int32(rank)}, DTypeFloat32)
	Materialize(loraA, loraB)

	adapterDir := t.TempDir()
	SaveSafetensors(core.JoinPath(adapterDir, "adapters.safetensors"), map[string]*Array{
		"layers.0.self_attn.q_proj.lora_a": loraA,
		"layers.0.self_attn.q_proj.lora_b": loraB,
	})
	_ = coreio.Local.Write(core.JoinPath(adapterDir, "adapter_config.json"),
		`{"rank": 4, "alpha": 8.0}`)

	// Build a model and apply adapter.
	qwen := newLoRAResolverTestModel(map[string]*Linear{"self_attn.q_proj": linear})

	err := applyLoadedLoRA(qwen, adapterDir)
	if err != nil {
		t.Fatalf("applyLoadedLoRA: %v", err)
	}

	// Now forward should go through LoRA path.
	loraOut := linear.Forward(x)
	Materialize(loraOut)
	loraFloats := loraOut.Floats()

	// Outputs should differ since B is non-zero.
	allSame := true
	for i := range baseFloats {
		if math.Abs(float64(baseFloats[i]-loraFloats[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("expected LoRA output to differ from base output with non-zero B weights")
	}
}

// --- LoadAndInit with adapter ---

func TestLora_LoadAndInit_AdapterMissing_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	// Create a minimal safetensors file so model loading proceeds.
	// The adapter path doesn't exist, so it should fail at the adapter step.
	_, err := LoadAndInit(dir, LoadConfig{AdapterPath: "/nonexistent/adapter"})
	if err == nil {
		t.Fatal("expected error for missing adapter")
	}
}

// TestLora_NormalizeLoRAConfig_Good: the exported config normaliser applies every
// default rule — rank floor, alpha derivation from scale or the 16 fallback,
// scale derivation, target-key/layer mirroring, and the float32 dtype default.
// Pure-Go, no Metal device.
func TestLora_NormalizeLoRAConfig_Good(t *testing.T) {
	// Empty config: rank→8, alpha→16, scale→16/8=2, targets→default pair.
	got := NormalizeLoRAConfig(LoRAConfig{})
	if got.Rank != 8 || got.Alpha != 16 || got.Scale != 2 {
		t.Errorf("empty cfg = rank %d alpha %g scale %g, want 8/16/2", got.Rank, got.Alpha, got.Scale)
	}
	if len(got.TargetKeys) != 2 || got.TargetKeys[0] != "q_proj" || got.TargetKeys[1] != "v_proj" {
		t.Errorf("default TargetKeys = %v, want [q_proj v_proj]", got.TargetKeys)
	}
	if !slices.Equal(got.TargetLayers, got.TargetKeys) {
		t.Errorf("TargetLayers = %v, want mirror of TargetKeys %v", got.TargetLayers, got.TargetKeys)
	}
	if got.DType != DTypeFloat32 {
		t.Errorf("default DType = %v, want float32", got.DType)
	}

	// Alpha derived from an explicit scale: alpha = scale * rank.
	scaled := NormalizeLoRAConfig(LoRAConfig{Rank: 4, Scale: 3})
	if scaled.Alpha != 12 {
		t.Errorf("scale=3 rank=4 → alpha %g, want 12", scaled.Alpha)
	}

	// Explicit alpha drives scale, and TargetLayers seeds TargetKeys when only
	// the former is given.
	fromLayers := NormalizeLoRAConfig(LoRAConfig{Rank: 16, Alpha: 32, TargetLayers: []string{"o_proj"}})
	if fromLayers.Scale != 2 {
		t.Errorf("alpha=32 rank=16 → scale %g, want 2", fromLayers.Scale)
	}
	if len(fromLayers.TargetKeys) != 1 || fromLayers.TargetKeys[0] != "o_proj" {
		t.Errorf("TargetKeys from TargetLayers = %v, want [o_proj]", fromLayers.TargetKeys)
	}
}

// TestLora_loraResultError_Good: the result→error adapter returns the embedded
// error when the Core result carries one, and nil for any non-error value.
func TestLora_loraResultError_Good(t *testing.T) {
	boom := core.NewError("boom")
	if got := loraResultError(core.Result{Value: boom}); got != boom {
		t.Errorf("loraResultError(error value) = %v, want the embedded error", got)
	}
	if got := loraResultError(core.Result{Value: "not an error", OK: true}); got != nil {
		t.Errorf("loraResultError(non-error value) = %v, want nil", got)
	}
	if got := loraResultError(core.Result{}); got != nil {
		t.Errorf("loraResultError(zero result) = %v, want nil", got)
	}
}

// TestLora_StepAccumulated_Bad: the gradient-accumulation step refuses every
// malformed precondition — nil adapter, nil/absent model, nil optimiser, no
// batches, and a batch/target count mismatch — returning a nil loss without
// touching the (absent) model. Pure-Go guard, no training run.
func TestLora_StepAccumulated_Bad(t *testing.T) {
	var nilAdapter *LoRAAdapter
	if got := nilAdapter.StepAccumulated(nil, nil, &AdamW{}); got != nil {
		t.Errorf("nil-adapter StepAccumulated = %v, want nil", got)
	}
	// Adapter with no Model set → the Model==nil guard trips.
	noModel := &LoRAAdapter{}
	if got := noModel.StepAccumulated([]Batch{{}}, [][][]int{{{1}}}, &AdamW{}); got != nil {
		t.Errorf("no-Model StepAccumulated = %v, want nil", got)
	}
	// Nil optimiser, even with other fields, declines.
	if got := noModel.StepAccumulated([]Batch{{}}, [][][]int{{{1}}}, nil); got != nil {
		t.Errorf("nil-optimiser StepAccumulated = %v, want nil", got)
	}
	// Empty batches, and a batch/target length mismatch, both decline.
	if got := noModel.StepAccumulated(nil, nil, &AdamW{}); got != nil {
		t.Errorf("empty-batch StepAccumulated = %v, want nil", got)
	}
	if got := noModel.StepAccumulated([]Batch{{}, {}}, [][][]int{{{1}}}, &AdamW{}); got != nil {
		t.Errorf("mismatched batch/target StepAccumulated = %v, want nil", got)
	}
}

// TestLora_LayerParams_Good: SetParams swaps the trainable A/B arrays in place and
// ParamCount/TotalParams sum their element counts. A synthetic adapter with one
// layer (A=[rank,in], B=[out,rank]) has rank*in + out*rank params. Needs a Metal
// device for the array shapes.
func TestLora_LayerParams_Good(t *testing.T) {
	requireMetalRuntime(t)

	// rank=2, in=4, out=3 → A is [2,4]=8, B is [3,2]=6, total 14.
	a := Zeros([]int32{2, 4}, DTypeFloat32)
	b := Zeros([]int32{3, 2}, DTypeFloat32)
	Materialize(a, b)
	layer := &LoRALinear{Rank: 2}
	layer.SetParams(a, b)
	defer Free(layer.A, layer.B)

	if got := layer.ParamCount(); got != 14 {
		t.Errorf("ParamCount() = %d, want 14 (8 + 6)", got)
	}
	adapter := &LoRAAdapter{Layers: map[string]*LoRALinear{"layer.0": layer}}
	if got := adapter.TotalParams(); got != 14 {
		t.Errorf("TotalParams() over one layer = %d, want 14", got)
	}
}
