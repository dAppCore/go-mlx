// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
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
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
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
