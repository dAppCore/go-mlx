// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
)

import core "dappco.re/go"

func TestAutoRound_BuiltinProfiles_Good(t *testing.T) {
	profiles := BuiltinProfiles()
	if len(profiles) != 3 {
		t.Fatalf("BuiltinProfiles len = %d, want 3", len(profiles))
	}
	profile, ok := LookupProfile(ProfileAutoRoundBest)
	if !ok {
		t.Fatal("LookupProfile(auto-round-best) ok = false")
	}
	if profile.Iters != 1000 || profile.NSamples != 512 || profile.SeqLen != 2048 {
		t.Fatalf("profile = %+v, want best tuning defaults", profile)
	}
	cfg := ConfigFromProfile(profile)
	if cfg.Bits != 2 || cfg.GroupSize != 32 || !cfg.Symmetric {
		t.Fatalf("ConfigFromProfile = %+v, want W2A16 group 32 symmetric", cfg)
	}
	profiles[0].Notes[0] = "mutated"
	again := BuiltinProfiles()
	if again[0].Notes[0] == "mutated" {
		t.Fatal("BuiltinProfiles returned aliased notes")
	}
}

func TestAutoRound_ReadPackInfo_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, PackConfigFileQuantization)
	if result := core.WriteFile(path, []byte(`{
		"bits": 4,
		"group_size": 128,
		"sym": true,
		"data_type": "int",
		"iters": 1000,
		"nsamples": 512,
		"seqlen": 2048,
		"autoround_version": "0.13.0",
		"quant_method": "auto-round",
		"packing_format": "auto_round:auto_gptq",
		"tensors": [
			{
				"name": "model.layers.0.self_attn.q_proj.weight",
				"packed": "model.layers.0.self_attn.q_proj.weight.packed",
				"scales": "model.layers.0.self_attn.q_proj.weight.scales",
				"zero_points": "model.layers.0.self_attn.q_proj.weight.zeros",
				"shape": [4, 8],
				"bits": 4,
				"group_size": 32,
				"sym": true
			}
		],
		"extra_config": {
			"model.layers.0.self_attn.q_proj": {"bits": 16}
		}
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	info, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo() error = %v", err)
	}
	if info == nil || info.Bits != 4 || info.GroupSize != 128 || info.Scheme != SchemeW4A16 || info.ExportFormat != FormatAutoRound || info.LayerOverrideN != 1 {
		t.Fatalf("ReadPackInfo() = %+v, want W4A16 AutoRound sidecar", info)
	}
	if info.TensorCount != 1 || !info.NativeTensorMap() {
		t.Fatalf("tensor metadata count=%d native=%v, want one native tensor mapping", info.TensorCount, info.NativeTensorMap())
	}
	if !info.NativeFormat() || info.GGUFExport() {
		t.Fatalf("format flags native=%v gguf=%v, want native only", info.NativeFormat(), info.GGUFExport())
	}
}

func TestAutoRound_ReadPackInfoGGUF_Good(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, PackConfigFileAutoRound), []byte(`{
		"bits": 4,
		"group_size": 256,
		"sym": true,
		"quant_method": "autoround",
		"packing_format": "gguf:q4_k_m"
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	info, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo() error = %v", err)
	}
	if info == nil || info.Scheme != SchemeGGUFQ4KM || info.ExportFormat != FormatGGUFQ4KM || !info.GGUFExport() {
		t.Fatalf("ReadPackInfo() = %+v, want GGUF Q4_K_M AutoRound export", info)
	}
}

func TestAutoRound_ReadPackInfoIgnoresOtherQuantization_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, PackConfigFileQuantization), []byte(`{
		"bits": 4,
		"group_size": 128,
		"quant_method": "gptq"
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	info, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo(non-AutoRound quantization_config) error = %v", err)
	}
	if info != nil {
		t.Fatalf("ReadPackInfo(non-AutoRound quantization_config) = %+v, want nil", info)
	}
	if _, err := ReadPackInfoFile(core.PathJoin(dir, PackConfigFileQuantization)); err == nil {
		t.Fatal("ReadPackInfoFile(non-AutoRound) error = nil, want strict direct-file error")
	}
}

func TestAutoRound_ReadPackInfoRejectsInvalidTensorMap_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, PackConfigFileAutoRound), []byte(`{
		"bits": 4,
		"group_size": 32,
		"sym": true,
		"quant_method": "auto-round",
		"packing_format": "auto_round",
		"tensors": [
			{
				"name": "model.layers.0.mlp.gate_proj.weight",
				"packed": "model.layers.0.mlp.gate_proj.weight.packed",
				"scales": "model.layers.0.mlp.gate_proj.weight.scales",
				"zero_points": "model.layers.0.mlp.gate_proj.weight.zeros",
				"shape": [3],
				"bits": 4,
				"group_size": 32,
				"packed_bytes": 1
			}
		]
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	_, err := ReadPackInfo(dir)
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("ReadPackInfo(invalid tensor map) error = %v, want packed length diagnostic", err)
	}
}

func TestAutoRound_ResolveScheme_Good(t *testing.T) {
	w4, ok := ResolveScheme("w4a16")
	if !ok {
		t.Fatal("ResolveScheme(w4a16) ok = false")
	}
	if w4.Scheme != SchemeW4A16 || w4.Bits != 4 || w4.ActivationBits != 16 || w4.GroupSize != 128 {
		t.Fatalf("W4A16 info = %+v, want int4 weight-only defaults", w4)
	}
	gguf, ok := ResolveScheme("gguf:q4_k_m")
	if !ok {
		t.Fatal("ResolveScheme(gguf:q4_k_m) ok = false")
	}
	if gguf.Scheme != SchemeGGUFQ4KM || gguf.ExportFormat != FormatGGUFQ4KM || gguf.GroupSize != 256 {
		t.Fatalf("GGUF info = %+v, want Q4_K_M export defaults", gguf)
	}
}

func TestAutoRound_QuantizeWeightsRTN_Good(t *testing.T) {
	weights := make([]float32, 32)
	for i := range weights {
		weights[i] = float32(i-16) / 8
	}
	got, err := QuantizeWeights(weights, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32, Iters: 0})
	if err != nil {
		t.Fatalf("QuantizeWeights returned error: %v", err)
	}
	if got.Bits != 4 || got.GroupSize != 32 || got.Iters != 0 || len(got.QValues) != len(weights) || len(got.Scales) != 1 || len(got.Dequantized) != len(weights) {
		t.Fatalf("QuantizeWeights = %+v, want one W4A16 RTN group", got)
	}
}

func TestAutoRound_QuantizeWeightsSignRound_Good(t *testing.T) {
	weights := make([]float32, 32)
	weights[0] = 1.4
	weights[1] = 1.4
	weights[2] = 7
	gradients := make([]float32, len(weights))
	gradients[0] = 1
	gradients[1] = -1

	got, err := QuantizeWeights(weights, QuantizeConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		Gradients: gradients,
	})
	if err != nil {
		t.Fatalf("QuantizeWeights returned error: %v", err)
	}
	if got.QValues[0] != 1 || got.QValues[1] != 2 {
		t.Fatalf("qvalues[0:2] = %v, want sign-gradient floor/ceil split", got.QValues[:2])
	}
}

func TestAutoRound_CalibrationPlan_Good(t *testing.T) {
	profile, ok := LookupProfile(ProfileAutoRoundLight)
	if !ok {
		t.Fatal("missing auto-round-light profile")
	}
	cfg := CalibrationConfigFromProfile(profile)
	cfg.NSamples = 2
	cfg.SeqLen = 3
	samples := []CalibrationSample{
		{ID: "a", Text: "one two three four"},
		{ID: "b", TokenN: 2},
		{ID: "c", TokenN: 1},
	}

	plan, err := BuildCalibrationPlan(samples, cfg)
	if err != nil {
		t.Fatalf("BuildCalibrationPlan() error = %v", err)
	}
	if plan.InputSamples != 3 || plan.SelectedSamples != 2 || !plan.Truncated || plan.TokenCount != 5 {
		t.Fatalf("plan = %+v, want two selected samples and bounded token count", plan)
	}
	if plan.Config.Iters != 50 || plan.Config.NSamples != 2 || plan.Config.SeqLen != 3 {
		t.Fatalf("plan config = %+v, want profile calibration defaults with overrides", plan.Config)
	}
}

func TestAutoRound_QuantizeWithCalibration_Good(t *testing.T) {
	weights := make([]float32, 32)
	weights[0] = 1.4
	weights[1] = 1.4
	weights[2] = 7
	gradients := make([]float32, len(weights))
	gradients[0] = 1
	gradients[1] = -1

	run, err := QuantizeWithCalibration(weights, gradients, []CalibrationSample{{ID: "a", TokenN: 4}}, CalibrationConfig{
		Scheme:    SchemeW4A16,
		GroupSize: 32,
		Iters:     1,
		NSamples:  1,
		SeqLen:    8,
	})
	if err != nil {
		t.Fatalf("QuantizeWithCalibration() error = %v", err)
	}
	if run.Plan.SelectedSamples != 1 || run.Weights.QValues[0] != 1 || run.Weights.QValues[1] != 2 {
		t.Fatalf("run = %+v, want calibration plan plus SignRound split", run)
	}
}

func TestAutoRound_PackQuantizedWeightsRoundTripsDequantized_Good(t *testing.T) {
	weights := make([]float32, 32)
	for i := range weights {
		weights[i] = float32(i-16) / 7
	}
	quantized, err := QuantizeWeights(weights, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32, Iters: 0})
	if err != nil {
		t.Fatalf("QuantizeWeights() error = %v", err)
	}

	packed, err := PackQuantizedWeights(quantized, []int32{4, 8})
	if err != nil {
		t.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	got, err := DequantizePackedWeights(packed)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}

	if packed.QMin != -8 || packed.QMax != 7 || len(packed.Packed) != 16 {
		t.Fatalf("packed metadata = %+v, want W4 symmetric byte layout", packed)
	}
	assertAutoRoundFloat32SliceClose(t, got, quantized.Dequantized, 1e-6)
}

func TestAutoRound_PackQuantizedWeightsPreservesSignedQValues_Good(t *testing.T) {
	quantized := QuantizedWeights{
		Scheme:      SchemeW2A16,
		Bits:        2,
		GroupSize:   32,
		Symmetric:   true,
		QValues:     []int16{-2, -1, 0, 1},
		Scales:      []float32{0.5},
		ZeroPoints:  []float32{0},
		Dequantized: []float32{-1, -0.5, 0, 0.5},
	}

	packed, err := PackQuantizedWeights(quantized, []int32{4})
	if err != nil {
		t.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	got, err := DequantizePackedWeights(packed)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}

	if len(packed.Packed) != 1 || packed.Packed[0] != 0b11100100 {
		t.Fatalf("packed bytes = %08b, want signed qmin-offset layout 11100100", packed.Packed)
	}
	assertAutoRoundFloat32SliceClose(t, got, quantized.Dequantized, 1e-6)
}

func TestAutoRound_PackQuantizedWeightsRejectsBadMetadata_Bad(t *testing.T) {
	quantized := QuantizedWeights{
		Scheme:     SchemeW4A16,
		Bits:       4,
		GroupSize:  32,
		Symmetric:  true,
		QValues:    []int16{0, 1},
		Scales:     []float32{1},
		ZeroPoints: []float32{0},
	}
	if _, err := PackQuantizedWeights(quantized, []int32{3}); err == nil || !core.Contains(err.Error(), "shape") {
		t.Fatalf("PackQuantizedWeights(bad shape) error = %v, want shape diagnostic", err)
	}

	quantized.QValues[0] = -9
	if _, err := PackQuantizedWeights(quantized, []int32{2}); err == nil || !core.Contains(err.Error(), "outside range") {
		t.Fatalf("PackQuantizedWeights(bad qvalue) error = %v, want range diagnostic", err)
	}

	packed := PackedWeights{
		Bits:       4,
		GroupSize:  32,
		Shape:      []int32{3},
		Packed:     []byte{0},
		Scales:     []float32{1},
		ZeroPoints: []float32{0},
		QMin:       -8,
		QMax:       7,
	}
	if _, err := DequantizePackedWeights(packed); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("DequantizePackedWeights(bad length) error = %v, want packed length diagnostic", err)
	}
}

func TestAutoRound_LoadPackedProjectionFromSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.weight.scales", []float32{0.5}, 1),
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.weight.zeros", []float32{0}, 1),
		autoRoundF32Tensor("model.layers.0.self_attn.q_proj.bias", []float32{0.25}, 1),
	})
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "model.layers.0.self_attn.q_proj.weight",
			Packed:     "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:     "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints: "model.layers.0.self_attn.q_proj.weight.zeros",
			Bias:       "model.layers.0.self_attn.q_proj.bias",
			Shape:      []int32{1, 4},
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()

	projection, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors() error = %v", err)
	}
	got, err := DequantizePackedWeights(projection.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}

	assertAutoRoundFloat32SliceClose(t, got, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, projection.Bias, []float32{0.25}, 1e-6)
	if projection.Tensor.Name != "model.layers.0.self_attn.q_proj.weight" || projection.Weights.QMin != -2 || projection.Weights.QMax != 1 {
		t.Fatalf("projection metadata = %+v, want qmin/qmax and tensor name", projection)
	}
}

func TestAutoRound_LoadPackedProjectionFromSafetensorsRejectsMissingTensor_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0}},
		autoRoundF32Tensor("weight.scales", []float32{1}, 1),
	})
	info := PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()

	_, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight")
	if err == nil || !core.Contains(err.Error(), "missing") {
		t.Fatalf("LoadPackedProjectionFromSafetensors(missing zero point) error = %v, want missing tensor diagnostic", err)
	}
}

func TestAutoRound_WritePackedProjectionSafetensorsRoundTrips_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "projection.safetensors")
	projection := PackedProjection{
		Tensor: PackTensor{
			Name:        "model.layers.0.self_attn.q_proj.weight",
			Packed:      "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:      "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints:  "model.layers.0.self_attn.q_proj.weight.zeros",
			Bias:        "model.layers.0.self_attn.q_proj.bias",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: PackedWeights{
			Scheme:     SchemeW2A16,
			Format:     FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     []byte{0b11100100},
			Scales:     []float32{0.5},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
		Bias: []float32{0.25},
	}

	if err := WritePackedProjectionSafetensors(context.Background(), path, projection); err != nil {
		t.Fatalf("WritePackedProjectionSafetensors() error = %v", err)
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{projection.Tensor},
	}
	info.normalise()

	got, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projection.Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(exported) error = %v", err)
	}
	dequantized, err := DequantizePackedWeights(got.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(exported) error = %v", err)
	}

	if got.Weights.Packed[0] != projection.Weights.Packed[0] {
		t.Fatalf("packed byte = %08b, want %08b", got.Weights.Packed[0], projection.Weights.Packed[0])
	}
	assertAutoRoundFloat32SliceClose(t, dequantized, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, got.Bias, []float32{0.25}, 1e-6)
}

func TestAutoRound_WritePackedProjectionsSafetensorsRoundTrips_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, []float32{0.25}),
		autoRoundTestProjection("model.layers.0.self_attn.k_proj.weight", []byte{0b00011011}, []float32{0.25}, []float32{0}, nil),
	}

	if err := WritePackedProjectionsSafetensors(context.Background(), path, projections); err != nil {
		t.Fatalf("WritePackedProjectionsSafetensors() error = %v", err)
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{projections[0].Tensor, projections[1].Tensor},
	}
	info.normalise()

	qProj, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projections[0].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(q_proj) error = %v", err)
	}
	kProj, err := LoadPackedProjectionFromSafetensors(info, []string{path}, projections[1].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(k_proj) error = %v", err)
	}
	qValues, err := DequantizePackedWeights(qProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(q_proj) error = %v", err)
	}
	kValues, err := DequantizePackedWeights(kProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(k_proj) error = %v", err)
	}

	assertAutoRoundFloat32SliceClose(t, qValues, []float32{-1, -0.5, 0, 0.5}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, qProj.Bias, []float32{0.25}, 1e-6)
	assertAutoRoundFloat32SliceClose(t, kValues, []float32{0.25, 0, -0.25, -0.5}, 1e-6)
	if len(kProj.Bias) != 0 {
		t.Fatalf("k_proj bias len = %d, want no bias", len(kProj.Bias))
	}
}

func TestAutoRound_WritePackedProjectionsSafetensorsRejectsDuplicateTensorNames_Bad(t *testing.T) {
	dir := t.TempDir()
	first := autoRoundTestProjection("weight", []byte{0}, []float32{1}, []float32{0}, nil)
	second := autoRoundTestProjection("other", []byte{0}, []float32{1}, []float32{0}, nil)
	second.Tensor.Packed = first.Tensor.Packed

	err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(dir, "bad.safetensors"), []PackedProjection{first, second})
	if err == nil || !core.Contains(err.Error(), "duplicate") {
		t.Fatalf("WritePackedProjectionsSafetensors(duplicate tensor) error = %v, want duplicate diagnostic", err)
	}
}

func TestAutoRound_WriteNativePackRoundTripsDirectory_Good(t *testing.T) {
	dir := t.TempDir()
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, []float32{0.25}),
		autoRoundTestProjection("model.layers.0.self_attn.k_proj.weight", []byte{0b00011011}, []float32{0.25}, []float32{0}, nil),
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Scheme:        SchemeW2A16,
		ExportFormat:  FormatAutoRound,
		Iters:         1000,
		NSamples:      512,
		SeqLen:        2048,
	}

	result, err := WriteNativePack(context.Background(), dir, info, projections)
	if err != nil {
		t.Fatalf("WriteNativePack() error = %v", err)
	}
	if result.ConfigPath != core.PathJoin(dir, PackConfigFileAutoRound) || result.WeightPath != core.PathJoin(dir, "model.safetensors") || result.TensorCount != 2 {
		t.Fatalf("WriteNativePack() result = %+v, want config + model.safetensors paths and two tensors", result)
	}

	gotInfo, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo(exported) error = %v", err)
	}
	if gotInfo == nil || gotInfo.TensorCount != 2 || gotInfo.Scheme != SchemeW2A16 || gotInfo.ExportFormat != FormatAutoRound {
		t.Fatalf("ReadPackInfo(exported) = %+v, want W2 native tensor map", gotInfo)
	}
	if err := ValidateSafetensorsTensorMap(*gotInfo, []string{result.WeightPath}); err != nil {
		t.Fatalf("ValidateSafetensorsTensorMap(exported) error = %v", err)
	}
	qProj, err := LoadPackedProjectionFromSafetensors(*gotInfo, []string{result.WeightPath}, projections[0].Tensor.Name)
	if err != nil {
		t.Fatalf("LoadPackedProjectionFromSafetensors(exported pack) error = %v", err)
	}
	values, err := DequantizePackedWeights(qProj.Weights)
	if err != nil {
		t.Fatalf("DequantizePackedWeights(exported pack) error = %v", err)
	}
	assertAutoRoundFloat32SliceClose(t, values, []float32{-1, -0.5, 0, 0.5}, 1e-6)
}

func TestAutoRound_WriteNativePackRejectsEmptyProjectionSet_Bad(t *testing.T) {
	_, err := WriteNativePack(context.Background(), t.TempDir(), PackInfo{Bits: 2, GroupSize: 32, Symmetric: true}, nil)
	if err == nil || !core.Contains(err.Error(), "projection") {
		t.Fatalf("WriteNativePack(empty projections) error = %v, want projection diagnostic", err)
	}
}

func TestAutoRound_WritePackedProjectionSafetensorsRejectsBadProjection_Bad(t *testing.T) {
	projection := PackedProjection{
		Tensor: PackTensor{
			Name:        "weight",
			Packed:      "weight.packed",
			Scales:      "weight.scales",
			ZeroPoints:  "weight.zeros",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: PackedWeights{
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     nil,
			Scales:     []float32{1},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
	}
	if err := WritePackedProjectionSafetensors(context.Background(), "", projection); err == nil || !core.Contains(err.Error(), "path") {
		t.Fatalf("WritePackedProjectionSafetensors(empty path) error = %v, want path diagnostic", err)
	}
	if err := WritePackedProjectionSafetensors(context.Background(), core.PathJoin(t.TempDir(), "bad.safetensors"), projection); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("WritePackedProjectionSafetensors(bad packed) error = %v, want packed length diagnostic", err)
	}
}

func TestAutoRound_Validation_Bad(t *testing.T) {
	cases := []QuantizeConfig{
		{Bits: 5, GroupSize: 32},
		{Bits: 4, GroupSize: 16},
		{Scheme: "missing"},
		{Bits: 4, GroupSize: 32, Iters: -1},
		{Bits: 4, GroupSize: 32, Iters: 1, Gradients: []float32{1}},
	}
	for _, cfg := range cases {
		t.Run(string(cfg.Scheme), func(t *testing.T) {
			if _, err := QuantizeWeights([]float32{1, 2}, cfg); err == nil {
				t.Fatalf("QuantizeWeights(%+v) err = nil, want error", cfg)
			}
		})
	}
	if _, ok := LookupProfile("missing"); ok {
		t.Fatal("LookupProfile(missing) ok = true")
	}
}

func autoRoundTestProjection(name string, packed []byte, scales, zeroPoints, bias []float32) PackedProjection {
	tensor := PackTensor{
		Name:        name,
		Packed:      name + ".packed",
		Scales:      name + ".scales",
		ZeroPoints:  name + ".zeros",
		Shape:       []int32{1, 4},
		Bits:        2,
		GroupSize:   32,
		Symmetric:   true,
		PackedBytes: 1,
		Groups:      1,
		QMin:        -2,
		QMax:        1,
	}
	if len(bias) > 0 {
		tensor.Bias = name + ".bias"
	}
	return PackedProjection{
		Tensor: tensor,
		Weights: PackedWeights{
			Scheme:     SchemeW2A16,
			Format:     FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     core.SliceClone(packed),
			Scales:     core.SliceClone(scales),
			ZeroPoints: core.SliceClone(zeroPoints),
			QMin:       -2,
			QMax:       1,
		},
		Bias: core.SliceClone(bias),
	}
}

type autoRoundSafetensorTensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

func autoRoundF32Tensor(name string, values []float32, shape ...int) autoRoundSafetensorTensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	if len(shape) == 0 {
		shape = []int{len(values)}
	}
	return autoRoundSafetensorTensor{Name: name, DType: "F32", Shape: append([]int(nil), shape...), Raw: raw}
}

func writeAutoRoundSafetensors(t *testing.T, path string, tensors []autoRoundSafetensorTensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.Raw...)
		header[tensor.Name] = entry{
			DType:       tensor.DType,
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func assertAutoRoundFloat32SliceClose(t *testing.T, got, want []float32, epsilon float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
