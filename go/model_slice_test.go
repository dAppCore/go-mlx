// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/safetensors"
)

func TestModelSlice_SliceModel_GoodClientPresetMaterialisesPack(t *testing.T) {
	source := writeModelSliceTestPack(t)
	target := core.PathJoin(t.TempDir(), "client-slice")

	plan, err := (&metalbackend{}).SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: target,
	})
	if err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	if plan.SourcePath != source || plan.OutputPath != target {
		t.Fatalf("paths = source %q output %q, want %q %q", plan.SourcePath, plan.OutputPath, source, target)
	}
	index, err := safetensors.ReadIndex(core.PathJoin(target, "model.safetensors"))
	if err != nil {
		t.Fatalf("ReadIndex(output): %v", err)
	}
	for _, name := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"lm_head.weight",
	} {
		if _, ok := index.Tensors[name]; !ok {
			t.Fatalf("slice tensors = %v, want %q", index.Names, name)
		}
	}
	if _, ok := index.Tensors["model.layers.0.mlp.down_proj.weight"]; ok {
		t.Fatalf("slice tensors = %v, want FFN tensor excluded", index.Names)
	}
	if _, ok := index.Tensors["model.layers.0.mlp.gate_proj.weight"]; ok {
		t.Fatalf("slice tensors = %v, want gate tensor excluded", index.Names)
	}
	if result := core.Stat(core.PathJoin(target, "config.json")); !result.OK {
		t.Fatalf("config.json not copied: %v", result.Value)
	}
	if result := core.Stat(core.PathJoin(target, "tokenizer.json")); !result.OK {
		t.Fatalf("tokenizer.json not copied: %v", result.Value)
	}
	if result := core.Stat(core.PathJoin(target, "slice_manifest.json")); !result.OK {
		t.Fatalf("slice_manifest.json not written: %v", result.Value)
	}
	if plan.Labels["tensor_count"] != "4" {
		t.Fatalf("labels = %+v, want tensor_count=4", plan.Labels)
	}
	if plan.Labels["selected_tensor_bytes"] != "16" || plan.Labels["source_tensor_bytes"] != "24" {
		t.Fatalf("labels = %+v, want selected/source tensor byte counts", plan.Labels)
	}
}

func TestModelSlice_InspectModelSlice_GoodClientRequiresSplitPlacement(t *testing.T) {
	source := writeModelSliceTestPack(t)
	target := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: target,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	inspection, err := InspectModelSlice(target)

	if err != nil {
		t.Fatalf("InspectModelSlice: %v", err)
	}
	if inspection.Standalone || !inspection.RequiresSplitPlacement {
		t.Fatalf("inspection = %+v, want non-standalone split placement", inspection)
	}
	if inspection.LocalTensorBytes != 16 || inspection.SourceTensorBytes != 24 || inspection.OffloadTensorBytes != 8 {
		t.Fatalf("inspection bytes = local:%d source:%d offload:%d, want 16/24/8", inspection.LocalTensorBytes, inspection.SourceTensorBytes, inspection.OffloadTensorBytes)
	}
	if inspection.RetainedTensorRatio != 0.6666666666666666 {
		t.Fatalf("retained ratio = %v, want 2/3", inspection.RetainedTensorRatio)
	}
}

func TestModelSlice_LoadModel_BadClientSliceRequiresSplitPlacement(t *testing.T) {
	source := writeModelSliceTestPack(t)
	target := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: target,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })
	called := false
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (NativeModel, error) {
		called = true
		return &fakeNativeModel{}, nil
	}

	_, err := LoadModel(target)

	if err == nil || !core.Contains(err.Error(), "requires split placement") {
		t.Fatalf("LoadModel(client slice) error = %v, want split placement error", err)
	}
	if called {
		t.Fatal("LoadModel called native loader for non-standalone client slice")
	}
}

func TestModelSlice_SliceModel_BadMissingOutput(t *testing.T) {
	source := writeModelSliceTestPack(t)

	_, err := (&metalbackend{}).SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset: inference.ModelSlicePresetClient,
		Model:  inference.ModelIdentity{Path: source},
	})

	if err == nil {
		t.Fatal("SliceModel missing output error = nil")
	}
}

func TestModelSlice_SliceModel_UglyContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := (&metalbackend{}).SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: core.PathJoin(t.TempDir(), "missing")},
		OutputPath: core.PathJoin(t.TempDir(), "out"),
	})

	if err == nil {
		t.Fatal("SliceModel cancelled context error = nil")
	}
}

func writeModelSliceTestPack(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen2",
		"vocab_size": 16,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 32
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{"chat_template":"{{ messages }}"}`)
	writeModelSliceSafetensors(t, core.PathJoin(dir, "model.safetensors"), map[string][]byte{
		"model.embed_tokens.weight":              {1, 2, 3, 4},
		"model.layers.0.input_layernorm.weight":  {5, 6, 7, 8},
		"model.layers.0.self_attn.q_proj.weight": {9, 10, 11, 12},
		"model.layers.0.mlp.down_proj.weight":    {13, 14, 15, 16},
		"model.layers.0.mlp.gate_proj.weight":    {17, 18, 19, 20},
		"lm_head.weight":                         {21, 22, 23, 24},
	})
	return dir
}

func writeModelSliceSafetensors(t *testing.T, path string, tensors map[string][]byte) {
	t.Helper()
	header := map[string]safetensors.HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		raw := tensors[name]
		header[name] = safetensors.HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(len(raw))},
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

// --- merged from model_slice_classify_test.go (Track A: tests match their source file) ---
// classifyEquivalenceCases enumerates the tensor-name shapes covered by
// the projection-family classifier. Each shape exercises the byte-walk
// branches that distinguish q/k/v/o/out/up/down/gate as well as the
// reject paths (no leading '.', no anchor at all, mixed cases).
var classifyEquivalenceCases = []string{
	// Attention paths through the single-char discriminator.
	"model.layers.0.self_attn.q_proj.weight",
	"model.layers.5.self_attn.k_proj.weight",
	"model.layers.7.self_attn.v_proj.weight",
	"model.layers.12.self_attn.o_proj.weight",
	"model.layers.12.attn.q_proj.bias",
	// Attention via .out_proj.
	"model.layers.0.attn.out_proj.weight",
	"transformer.h.5.attn.out_proj.weight",
	// FFN via .up_proj. / .down_proj.
	"model.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight",
	// Gate via .gate_proj. and .gate.
	"model.layers.0.mlp.gate_proj.weight",
	"model.layers.0.gate.weight",
	// Reject paths — wrong leading byte or no leading '.'.
	"foo_proj.weight",
	"q_proj.weight",    // no leading "."
	"down_proj.weight", // no leading "."
	"out_proj.weight",  // no leading "."
	"_proj.weight",     // anchor at start
	".x_proj.weight",   // unknown discriminator
	"model.embed_tokens.weight",
	"model.layers.0.input_layernorm.weight",
	"lm_head.weight",
	"router.weight",
	// Edge: anchor in the middle but not preceded by valid prefix.
	"foo_bar_proj.weight",
}

func TestModelSliceClassify_ProjectionFamilyEquivalence(t *testing.T) {
	for _, name := range classifyEquivalenceCases {
		fam := modelSliceProjectionFamily(name)

		// Cross-check projAttention against the legacy 5-projection chain.
		wantAttn := false
		if core.Contains(name, "_proj.") {
			wantAttn = modelSliceHasProjection(name, "q_proj") ||
				modelSliceHasProjection(name, "k_proj") ||
				modelSliceHasProjection(name, "v_proj") ||
				modelSliceHasProjection(name, "o_proj") ||
				modelSliceHasProjection(name, "out_proj")
		}
		gotAttn := fam&projAttention != 0
		if gotAttn != wantAttn {
			t.Errorf("name %q: projAttention=%v want %v", name, gotAttn, wantAttn)
		}

		// projFFN — up_proj or down_proj.
		wantFFN := false
		if core.Contains(name, "_proj.") {
			wantFFN = modelSliceHasProjection(name, "up_proj") ||
				modelSliceHasProjection(name, "down_proj")
		}
		gotFFN := fam&projFFN != 0
		if gotFFN != wantFFN {
			t.Errorf("name %q: projFFN=%v want %v", name, gotFFN, wantFFN)
		}

		// projGate — gate_proj.
		wantGate := modelSliceHasProjection(name, "gate_proj")
		gotGate := fam&projGate != 0
		if gotGate != wantGate {
			t.Errorf("name %q: projGate=%v want %v", name, gotGate, wantGate)
		}
	}
}

func TestModelSliceClassify_AttentionFFNGateEquivalence(t *testing.T) {
	for _, name := range classifyEquivalenceCases {
		// Recompute the previous-implementation result so each branch
		// stays pinned to the original semantics post-byte-walk swap.
		oldAttn := false
		if core.Contains(name, "self_attn") || core.Contains(name, "attention") || core.Contains(name, ".attn.") {
			oldAttn = true
		} else if core.Contains(name, "_proj.") {
			oldAttn = modelSliceHasProjection(name, "q_proj") ||
				modelSliceHasProjection(name, "k_proj") ||
				modelSliceHasProjection(name, "v_proj") ||
				modelSliceHasProjection(name, "o_proj") ||
				modelSliceHasProjection(name, "out_proj")
		}
		if got := modelSliceTensorIsAttention(name); got != oldAttn {
			t.Errorf("modelSliceTensorIsAttention(%q) = %v want %v", name, got, oldAttn)
		}

		oldFFN := false
		if core.Contains(name, ".mlp.") || core.Contains(name, "feed_forward") || core.Contains(name, "ffn") {
			oldFFN = true
		} else if core.Contains(name, "_proj.") {
			oldFFN = modelSliceHasProjection(name, "up_proj") ||
				modelSliceHasProjection(name, "down_proj")
		}
		if got := modelSliceTensorIsFFN(name); got != oldFFN {
			t.Errorf("modelSliceTensorIsFFN(%q) = %v want %v", name, got, oldFFN)
		}

		oldGate := modelSliceHasProjection(name, "gate_proj") || core.Contains(name, ".gate.")
		if got := modelSliceTensorIsGate(name); got != oldGate {
			t.Errorf("modelSliceTensorIsGate(%q) = %v want %v", name, got, oldGate)
		}
	}
}

// The MoE-family classifiers (down-meta / router / expert) gate which
// tensors a sparse-model slice keeps. They are pure name predicates, so
// Good = a name the family owns, Bad = a name a sibling family owns,
// Ugly = the boundary spellings the substring rules must still catch.
func TestModelSlice_TensorIsDownMeta_Good(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.down_meta",
		"model.layers.3.mlp.down_proj.meta",
	} {
		if !modelSliceTensorIsDownMeta(name) {
			t.Errorf("modelSliceTensorIsDownMeta(%q) = false, want true", name)
		}
	}
}

func TestModelSlice_TensorIsDownMeta_Bad(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.down_proj.weight", // plain down-proj, not its meta
		"model.layers.0.self_attn.q_proj.weight",
		"model.embed_tokens.weight",
	} {
		if modelSliceTensorIsDownMeta(name) {
			t.Errorf("modelSliceTensorIsDownMeta(%q) = true, want false", name)
		}
	}
}

func TestModelSlice_TensorIsRouter_Good(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.router.weight",
		"model.layers.0.mlp.gate_score",
		"model.layers.0.mlp.gate.weight", // HasSuffix(".gate.weight")
	} {
		if !modelSliceTensorIsRouter(name) {
			t.Errorf("modelSliceTensorIsRouter(%q) = false, want true", name)
		}
	}
}

func TestModelSlice_TensorIsRouter_Bad(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.gate_proj.weight", // gate-proj is FFN, not the router
		"model.layers.0.mlp.gate.bias",        // ".gate." but not the .gate.weight suffix
		"model.layers.0.self_attn.o_proj.weight",
	} {
		if modelSliceTensorIsRouter(name) {
			t.Errorf("modelSliceTensorIsRouter(%q) = true, want false", name)
		}
	}
}

func TestModelSlice_TensorIsExpert_Good(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.experts.0.down_proj.weight",
		"model.layers.0.mlp.expert.up_proj.weight",
	} {
		if !modelSliceTensorIsExpert(name) {
			t.Errorf("modelSliceTensorIsExpert(%q) = false, want true", name)
		}
	}
}

func TestModelSlice_TensorIsExpert_Bad(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.mlp.router.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.embed_tokens.weight",
	} {
		if modelSliceTensorIsExpert(name) {
			t.Errorf("modelSliceTensorIsExpert(%q) = true, want false", name)
		}
	}
}

// modelSliceIncludesTensor is the plan-level entry the slice selector
// runs per tensor: it builds the inclusion mask from the plan components
// (or short-circuits on ExtractLevelAll) then classifies the name.
func TestModelSlice_IncludesTensor_GoodExtractLevelAllKeepsEverything(t *testing.T) {
	plan := inference.ModelSlicePlan{ExtractLevel: inference.ModelExtractLevelAll}
	for _, name := range []string{
		"anything.at.all",
		"model.layers.0.self_attn.q_proj.weight",
	} {
		if !modelSliceIncludesTensor(plan, name) {
			t.Errorf("ExtractLevelAll: modelSliceIncludesTensor(%q) = false, want true", name)
		}
	}
}

func TestModelSlice_IncludesTensor_GoodComponentMaskMatches(t *testing.T) {
	plan := inference.ModelSlicePlan{
		ExtractLevel: inference.ModelExtractLevelCustom,
		Components: []inference.ModelComponent{
			inference.ModelComponentAttention,
			inference.ModelComponentExperts,
			inference.ModelComponentRouter,
			inference.ModelComponentDownMeta,
		},
	}
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight", // attention
		"model.layers.0.mlp.experts.0.up_proj.weight",
		"model.layers.0.mlp.router.weight",
		"model.layers.0.mlp.down_meta",
	} {
		if !modelSliceIncludesTensor(plan, name) {
			t.Errorf("masked: modelSliceIncludesTensor(%q) = false, want true", name)
		}
	}
}

func TestModelSlice_IncludesTensor_BadComponentNotRequested(t *testing.T) {
	// Only attention requested — FFN / embedding / norm tensors fall out.
	plan := inference.ModelSlicePlan{
		ExtractLevel: inference.ModelExtractLevelAttention,
		Components:   []inference.ModelComponent{inference.ModelComponentAttention},
	}
	for _, name := range []string{
		"model.layers.0.mlp.down_proj.weight",
		"model.embed_tokens.weight",
		"model.layers.0.input_layernorm.weight",
	} {
		if modelSliceIncludesTensor(plan, name) {
			t.Errorf("attention-only: modelSliceIncludesTensor(%q) = true, want false", name)
		}
	}
}

func TestModelSlice_IncludesTensor_UglyEmptyPlanKeepsNothing(t *testing.T) {
	// Custom level, no components → the mask is all-false, so no name is
	// ever included even when it spells a real projection.
	plan := inference.ModelSlicePlan{ExtractLevel: inference.ModelExtractLevelCustom}
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"lm_head.weight",
	} {
		if modelSliceIncludesTensor(plan, name) {
			t.Errorf("empty plan: modelSliceIncludesTensor(%q) = true, want false", name)
		}
	}
}

// modelSliceResultError unwraps the error a core.Result carries. Good = a
// failed result whose Value is an error; Bad = an OK result (no error);
// Ugly = a failed result whose Value is not an error, which falls back to
// the package sentinel rather than panicking on the type assertion.
func TestModelSlice_ResultError_Good(t *testing.T) {
	want := core.NewError("boom")
	if got := modelSliceResultError(core.Result{OK: false, Value: want}); got != want {
		t.Errorf("modelSliceResultError(err) = %v, want %v", got, want)
	}
}

func TestModelSlice_ResultError_BadOKResultHasNoError(t *testing.T) {
	if got := modelSliceResultError(core.Result{OK: true}); got != nil {
		t.Errorf("modelSliceResultError(ok) = %v, want nil", got)
	}
}

func TestModelSlice_ResultError_UglyNonErrorValueFallsBackToSentinel(t *testing.T) {
	if got := modelSliceResultError(core.Result{OK: false, Value: "not-an-error"}); got != errModelSliceCoreResultFailed {
		t.Errorf("modelSliceResultError(non-error) = %v, want sentinel", got)
	}
}
