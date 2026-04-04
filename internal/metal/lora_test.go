//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	"dappco.re/go/core"

	coreio "forge.lthn.ai/core/go-io"
)

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

	// Save the adapter weights.
	adapterDir := t.TempDir()
	err := SaveSafetensors(core.JoinPath(adapterDir, "adapters.safetensors"), map[string]*Array{
		"layers.0.self_attn.q_proj.lora_a": lora.A,
		"layers.0.self_attn.q_proj.lora_b": lora.B,
	})
	if err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	// Write adapter_config.json.
	configJSON := `{"rank": 4, "alpha": 8.0, "num_layers": 1, "lora_layers": ["self_attn.q_proj"]}`
	_ = coreio.Local.Write(core.JoinPath(adapterDir, "adapter_config.json"), configJSON)

	// Now create a fresh linear with the same base weights (no LoRA).
	linear2 := NewLinear(w, nil)
	if linear2.LoRA != nil {
		t.Fatal("fresh linear should not have LoRA")
	}

	// Build a minimal model for resolveLinear to work.
	qwen := &Qwen3Model{
		Layers: []*Qwen3DecoderLayer{
			{
				Attention: &Qwen3Attention{
					QProj: linear2,
					KProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
					VProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
					OProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
				},
			},
		},
	}

	// Apply the loaded adapter.
	err = applyLoadedLoRA(qwen, adapterDir)
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

func TestLora_ApplyLoadedLoRA_Bad_MissingConfig(t *testing.T) {
	dir := t.TempDir()
	// Write safetensors but no config.
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	Materialize(a)
	SaveSafetensors(core.JoinPath(dir, "adapters.safetensors"), map[string]*Array{"x": a})

	qwen := &Qwen3Model{Layers: []*Qwen3DecoderLayer{}}
	err := applyLoadedLoRA(qwen, dir)
	if err == nil {
		t.Fatal("expected error for missing adapter_config.json")
	}
}

func TestLora_ApplyLoadedLoRA_Bad_MissingSafetensors(t *testing.T) {
	dir := t.TempDir()
	// Write config but no safetensors.
	_ = coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), `{"rank": 8}`)

	qwen := &Qwen3Model{Layers: []*Qwen3DecoderLayer{}}
	err := applyLoadedLoRA(qwen, dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors")
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

	qwen := &Qwen3Model{
		Layers: []*Qwen3DecoderLayer{
			{
				Attention: &Qwen3Attention{
					QProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
				},
			},
		},
	}
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
	qwen := &Qwen3Model{
		Layers: []*Qwen3DecoderLayer{
			{
				Attention: &Qwen3Attention{
					QProj: linear,
					KProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
					VProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
					OProj: NewLinear(RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32), nil),
				},
			},
		},
	}

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
