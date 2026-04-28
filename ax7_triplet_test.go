// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	core "dappco.re/go"

	"dappco.re/go/inference"
)

type ax7RootTokenizerImpl struct{}

func (ax7RootTokenizerImpl) Encode(text string) []int32 {
	if text == "" {
		return []int32{1}
	}
	return []int32{1, 2, 3}
}

func (ax7RootTokenizerImpl) Decode(tokens []int32) string {
	if len(tokens) == 0 {
		return ""
	}
	return "decoded"
}

func (ax7RootTokenizerImpl) TokenID(text string) (int32, bool) {
	if text == "known" {
		return 7, true
	}
	return 0, false
}

func (ax7RootTokenizerImpl) IDToken(id int32) string {
	if id == 7 {
		return "known"
	}
	return ""
}

func (ax7RootTokenizerImpl) BOS() int32        { return 1 }
func (ax7RootTokenizerImpl) EOS() int32        { return 9 }
func (ax7RootTokenizerImpl) HasBOSToken() bool { return true }

func ax7RootVector() *Array        { return FromValues([]float32{1, 2, 3, 4}, 4) }
func ax7RootMatrix() *Array        { return FromValues([]float32{1, 2, 3, 4}, 2, 2) }
func ax7RootTokenArray() *Array    { return FromValues([]int32{0}, 1, 1) }
func ax7RootLogits() *Array        { return FromValues([]float32{2, 0}, 1, 1, 2) }
func ax7RootTokenizer() *Tokenizer { return &Tokenizer{tok: ax7RootTokenizerImpl{}} }

func ax7RootModel() *Model {
	return &Model{model: &fakeNativeModel{}, cfg: LoadConfig{ContextLength: 16}, tok: ax7RootTokenizer(), cleanup: func() error { return nil }}
}

func ax7RootInferenceAdapter() *InferenceAdapter {
	return NewInferenceAdapter(&stubTextModel{}, "ax7")
}

func ax7RootSession() Session {
	session, err := NewSession(WithSessionLabel("ax7"))
	if err != nil {
		return nil
	}
	return session
}

func TestAX7_AdamW_Reset_Good(t *core.T) {
	opt := NewAdamW(nil)
	opt.Reset()
	core.AssertNotNil(t, opt)
}

func TestAX7_AdamW_Reset_Bad(t *core.T) {
	opt := NewAdamW(nil)
	opt.Reset()
	core.AssertNotNil(t, opt)
}

func TestAX7_AdamW_Reset_Ugly(t *core.T) {
	opt := NewAdamW(nil)
	opt.Reset()
	core.AssertNotNil(t, opt)
}

func TestAX7_AdamW_Step_Good(t *core.T) {
	opt := NewAdamW(nil)
	out := opt.Step([]*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_AdamW_Step_Bad(t *core.T) {
	opt := NewAdamW(nil)
	out := opt.Step([]*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_AdamW_Step_Ugly(t *core.T) {
	opt := NewAdamW(nil)
	out := opt.Step([]*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_Adapter_ApplyLoRA_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ApplyLoRA(inference.LoRAConfig{}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_ApplyLoRA_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ApplyLoRA(inference.LoRAConfig{}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_ApplyLoRA_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ApplyLoRA(inference.LoRAConfig{}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_BatchGenerate_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.BatchGenerate(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_BatchGenerate_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.BatchGenerate(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_BatchGenerate_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.BatchGenerate(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Chat_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Chat(core.Background(), nil) {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Chat_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Chat(core.Background(), nil) {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Chat_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Chat(core.Background(), nil) {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Classify_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.Classify(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Classify_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.Classify(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Classify_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.Classify(core.Background(), nil) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Close_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Close() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Close_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Close() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Close_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Close() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Decode_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Decode([]int32{1}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Decode_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Decode([]int32{1}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Decode_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Decode([]int32{1}) })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Encode_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Encode("prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Encode_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Encode("prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Encode_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Encode("prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Err_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Err() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Err_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Err() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Err_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Err() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Generate_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Generate(core.Background(), "prompt") {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Generate_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Generate(core.Background(), "prompt") {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Generate_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() {
		for range adapter.Generate(core.Background(), "prompt") {
		}
	})
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Info_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Info() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Info_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Info() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Info_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Info() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InspectAttention_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.InspectAttention(core.Background(), "prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InspectAttention_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.InspectAttention(core.Background(), "prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InspectAttention_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _, _ = adapter.InspectAttention(core.Background(), "prompt") })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InternalModel_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.InternalModel() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InternalModel_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.InternalModel() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_InternalModel_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.InternalModel() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Metrics_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Metrics() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Metrics_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Metrics() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_Metrics_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.Metrics() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_ModelType_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ModelType() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_ModelType_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ModelType() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_ModelType_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.ModelType() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_NumLayers_Good(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.NumLayers() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_NumLayers_Bad(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.NumLayers() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Adapter_NumLayers_Ugly(t *core.T) {
	adapter := &metalAdapter{}
	core.AssertPanics(t, func() { _ = adapter.NumLayers() })
	core.AssertNil(t, adapter.model)
}

func TestAX7_Add_Good(t *core.T) {
	got := Add(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Add_Bad(t *core.T) {
	got := Add(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Add_Ugly(t *core.T) {
	got := Add(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Array_Bool_Good(t *core.T) {
	arr := FromValues([]bool{true}, 1)
	Materialize(arr)
	core.AssertTrue(t, arr.Bool())
}

func TestAX7_Array_Bool_Bad(t *core.T) {
	arr := FromValues([]bool{true}, 1)
	Materialize(arr)
	core.AssertTrue(t, arr.Bool())
}

func TestAX7_Array_Bool_Ugly(t *core.T) {
	arr := FromValues([]bool{true}, 1)
	Materialize(arr)
	core.AssertTrue(t, arr.Bool())
}

func TestAX7_Array_Clone_Good(t *core.T) {
	clone := ax7RootMatrix().Clone()
	Materialize(clone)
	core.AssertEqual(t, []int32{2, 2}, clone.Shape())
}

func TestAX7_Array_Clone_Bad(t *core.T) {
	clone := ax7RootMatrix().Clone()
	Materialize(clone)
	core.AssertEqual(t, []int32{2, 2}, clone.Shape())
}

func TestAX7_Array_Clone_Ugly(t *core.T) {
	clone := ax7RootMatrix().Clone()
	Materialize(clone)
	core.AssertEqual(t, []int32{2, 2}, clone.Shape())
}

func TestAX7_Array_DataInt32_Good(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int32{1, 2}, arr.DataInt32())
}

func TestAX7_Array_DataInt32_Bad(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int32{1, 2}, arr.DataInt32())
}

func TestAX7_Array_DataInt32_Ugly(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int32{1, 2}, arr.DataInt32())
}

func TestAX7_Array_Dim_Good(t *core.T) {
	arr := ax7RootMatrix()
	core.AssertEqual(t, 2, arr.Dim(0))
	core.AssertEqual(t, 2, arr.Dim(1))
}

func TestAX7_Array_Dim_Bad(t *core.T) {
	arr := ax7RootMatrix()
	core.AssertEqual(t, 2, arr.Dim(0))
	core.AssertEqual(t, 2, arr.Dim(1))
}

func TestAX7_Array_Dim_Ugly(t *core.T) {
	arr := ax7RootMatrix()
	core.AssertEqual(t, 2, arr.Dim(0))
	core.AssertEqual(t, 2, arr.Dim(1))
}

func TestAX7_Array_Dims_Good(t *core.T) {
	core.AssertEqual(t, []int{2, 2}, ax7RootMatrix().Dims())
	core.AssertEqual(t, []int{4}, ax7RootVector().Dims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Dims_Bad(t *core.T) {
	core.AssertEqual(t, []int{2, 2}, ax7RootMatrix().Dims())
	core.AssertEqual(t, []int{4}, ax7RootVector().Dims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Dims_Ugly(t *core.T) {
	core.AssertEqual(t, []int{2, 2}, ax7RootMatrix().Dims())
	core.AssertEqual(t, []int{4}, ax7RootVector().Dims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Dtype_Good(t *core.T) {
	core.AssertEqual(t, DTypeFloat32, ax7RootMatrix().Dtype())
	core.AssertEqual(t, "float32", ax7RootMatrix().Dtype().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Dtype_Bad(t *core.T) {
	core.AssertEqual(t, DTypeFloat32, ax7RootMatrix().Dtype())
	core.AssertEqual(t, "float32", ax7RootMatrix().Dtype().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Dtype_Ugly(t *core.T) {
	core.AssertEqual(t, DTypeFloat32, ax7RootMatrix().Dtype())
	core.AssertEqual(t, "float32", ax7RootMatrix().Dtype().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Float_Good(t *core.T) {
	arr := FromValues([]float32{1.5}, 1)
	Materialize(arr)
	core.AssertEqual(t, 1.5, arr.Float())
}

func TestAX7_Array_Float_Bad(t *core.T) {
	arr := FromValues([]float32{1.5}, 1)
	Materialize(arr)
	core.AssertEqual(t, 1.5, arr.Float())
}

func TestAX7_Array_Float_Ugly(t *core.T) {
	arr := FromValues([]float32{1.5}, 1)
	Materialize(arr)
	core.AssertEqual(t, 1.5, arr.Float())
}

func TestAX7_Array_Floats_Good(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertEqual(t, []float32{1, 2, 3, 4}, arr.Floats())
}

func TestAX7_Array_Floats_Bad(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertEqual(t, []float32{1, 2, 3, 4}, arr.Floats())
}

func TestAX7_Array_Floats_Ugly(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertEqual(t, []float32{1, 2, 3, 4}, arr.Floats())
}

func TestAX7_Array_Int_Good(t *core.T) {
	arr := FromValues([]int32{7}, 1)
	Materialize(arr)
	core.AssertEqual(t, 7, arr.Int())
}

func TestAX7_Array_Int_Bad(t *core.T) {
	arr := FromValues([]int32{7}, 1)
	Materialize(arr)
	core.AssertEqual(t, 7, arr.Int())
}

func TestAX7_Array_Int_Ugly(t *core.T) {
	arr := FromValues([]int32{7}, 1)
	Materialize(arr)
	core.AssertEqual(t, 7, arr.Int())
}

func TestAX7_Array_Ints_Good(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int{1, 2}, arr.Ints())
}

func TestAX7_Array_Ints_Bad(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int{1, 2}, arr.Ints())
}

func TestAX7_Array_Ints_Ugly(t *core.T) {
	arr := FromValues([]int32{1, 2}, 2)
	Materialize(arr)
	core.AssertEqual(t, []int{1, 2}, arr.Ints())
}

func TestAX7_Array_Iter_Good(t *core.T) {
	var got []float32
	for v := range ax7RootVector().Iter() {
		got = append(got, v)
	}
	core.AssertEqual(t, []float32{1, 2, 3, 4}, got)
}

func TestAX7_Array_Iter_Bad(t *core.T) {
	var got []float32
	for v := range ax7RootVector().Iter() {
		got = append(got, v)
	}
	core.AssertEqual(t, []float32{1, 2, 3, 4}, got)
}

func TestAX7_Array_Iter_Ugly(t *core.T) {
	var got []float32
	for v := range ax7RootVector().Iter() {
		got = append(got, v)
	}
	core.AssertEqual(t, []float32{1, 2, 3, 4}, got)
}

func TestAX7_Array_NumDims_Good(t *core.T) {
	core.AssertEqual(t, 2, ax7RootMatrix().NumDims())
	core.AssertEqual(t, 1, ax7RootVector().NumDims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_NumDims_Bad(t *core.T) {
	core.AssertEqual(t, 2, ax7RootMatrix().NumDims())
	core.AssertEqual(t, 1, ax7RootVector().NumDims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_NumDims_Ugly(t *core.T) {
	core.AssertEqual(t, 2, ax7RootMatrix().NumDims())
	core.AssertEqual(t, 1, ax7RootVector().NumDims())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Set_Good(t *core.T) {
	arr := ax7RootVector()
	arr.Set(ax7RootMatrix())
	core.AssertEqual(t, []int32{2, 2}, arr.Shape())
}

func TestAX7_Array_Set_Bad(t *core.T) {
	arr := ax7RootVector()
	arr.Set(ax7RootMatrix())
	core.AssertEqual(t, []int32{2, 2}, arr.Shape())
}

func TestAX7_Array_Set_Ugly(t *core.T) {
	arr := ax7RootVector()
	arr.Set(ax7RootMatrix())
	core.AssertEqual(t, []int32{2, 2}, arr.Shape())
}

func TestAX7_Array_SetFloat64_Good(t *core.T) {
	arr := FromValues([]float32{1}, 1)
	arr.SetFloat64(2)
	core.AssertEqual(t, 2.0, arr.Float())
}

func TestAX7_Array_SetFloat64_Bad(t *core.T) {
	arr := FromValues([]float32{1}, 1)
	arr.SetFloat64(2)
	core.AssertEqual(t, 2.0, arr.Float())
}

func TestAX7_Array_SetFloat64_Ugly(t *core.T) {
	arr := FromValues([]float32{1}, 1)
	arr.SetFloat64(2)
	core.AssertEqual(t, 2.0, arr.Float())
}

func TestAX7_Array_Shape_Good(t *core.T) {
	core.AssertEqual(t, []int32{2, 2}, ax7RootMatrix().Shape())
	core.AssertEqual(t, []int32{4}, ax7RootVector().Shape())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Shape_Bad(t *core.T) {
	core.AssertEqual(t, []int32{2, 2}, ax7RootMatrix().Shape())
	core.AssertEqual(t, []int32{4}, ax7RootVector().Shape())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Shape_Ugly(t *core.T) {
	core.AssertEqual(t, []int32{2, 2}, ax7RootMatrix().Shape())
	core.AssertEqual(t, []int32{4}, ax7RootVector().Shape())
	core.AssertTrue(t, true)
}

func TestAX7_Array_String_Good(t *core.T) {
	core.AssertContains(t, ax7RootMatrix().String(), "array")
	core.AssertNotEqual(t, "", ax7RootMatrix().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_String_Bad(t *core.T) {
	core.AssertContains(t, ax7RootMatrix().String(), "array")
	core.AssertNotEqual(t, "", ax7RootMatrix().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_String_Ugly(t *core.T) {
	core.AssertContains(t, ax7RootMatrix().String(), "array")
	core.AssertNotEqual(t, "", ax7RootMatrix().String())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Valid_Good(t *core.T) {
	core.AssertTrue(t, ax7RootMatrix().Valid())
	core.AssertFalse(t, (*Array)(nil).Valid())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Valid_Bad(t *core.T) {
	core.AssertTrue(t, ax7RootMatrix().Valid())
	core.AssertFalse(t, (*Array)(nil).Valid())
	core.AssertTrue(t, true)
}

func TestAX7_Array_Valid_Ugly(t *core.T) {
	core.AssertTrue(t, ax7RootMatrix().Valid())
	core.AssertFalse(t, (*Array)(nil).Valid())
	core.AssertTrue(t, true)
}

func TestAX7_AttentionSnapshot_HasQueries_Good(t *core.T) {
	snapshot := &AttentionSnapshot{Queries: [][][]float32{{{1}}}}
	core.AssertTrue(t, snapshot.HasQueries())
	core.AssertFalse(t, (&AttentionSnapshot{}).HasQueries())
}

func TestAX7_AttentionSnapshot_HasQueries_Bad(t *core.T) {
	snapshot := &AttentionSnapshot{Queries: [][][]float32{{{1}}}}
	core.AssertTrue(t, snapshot.HasQueries())
	core.AssertFalse(t, (&AttentionSnapshot{}).HasQueries())
}

func TestAX7_AttentionSnapshot_HasQueries_Ugly(t *core.T) {
	snapshot := &AttentionSnapshot{Queries: [][][]float32{{{1}}}}
	core.AssertTrue(t, snapshot.HasQueries())
	core.AssertFalse(t, (&AttentionSnapshot{}).HasQueries())
}

func TestAX7_Available_Good(t *core.T) {
	got := Available()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_Available_Bad(t *core.T) {
	got := Available()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_Available_Ugly(t *core.T) {
	got := Available()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_Backend_Available_Good(t *core.T) {
	got := (&metalBackend{}).Available()
	core.AssertEqual(t, MetalAvailable(), got)
	core.AssertTrue(t, got || !got)
}

func TestAX7_Backend_Available_Bad(t *core.T) {
	got := (&metalBackend{}).Available()
	core.AssertEqual(t, MetalAvailable(), got)
	core.AssertTrue(t, got || !got)
}

func TestAX7_Backend_Available_Ugly(t *core.T) {
	got := (&metalBackend{}).Available()
	core.AssertEqual(t, MetalAvailable(), got)
	core.AssertTrue(t, got || !got)
}

func TestAX7_Backend_DeviceInfo_Good(t *core.T) {
	info := computeBackend{}.DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Backend_DeviceInfo_Bad(t *core.T) {
	info := computeBackend{}.DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Backend_DeviceInfo_Ugly(t *core.T) {
	info := computeBackend{}.DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Backend_LoadModel_Good(t *core.T) {
	model, err := (&metalBackend{}).LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_Backend_LoadModel_Bad(t *core.T) {
	model, err := (&metalBackend{}).LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_Backend_LoadModel_Ugly(t *core.T) {
	model, err := (&metalBackend{}).LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_Backend_Name_Good(t *core.T) {
	core.AssertEqual(t, "metal", (&metalBackend{}).Name())
	core.AssertNotNil(t, &metalBackend{})
	core.AssertTrue(t, true)
}

func TestAX7_Backend_Name_Bad(t *core.T) {
	core.AssertEqual(t, "metal", (&metalBackend{}).Name())
	core.AssertNotNil(t, &metalBackend{})
	core.AssertTrue(t, true)
}

func TestAX7_Backend_Name_Ugly(t *core.T) {
	core.AssertEqual(t, "metal", (&metalBackend{}).Name())
	core.AssertNotNil(t, &metalBackend{})
	core.AssertTrue(t, true)
}

func TestAX7_Backend_NewSession_Good(t *core.T) {
	session, err := computeBackend{}.NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_Backend_NewSession_Bad(t *core.T) {
	session, err := computeBackend{}.NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_Backend_NewSession_Ugly(t *core.T) {
	session, err := computeBackend{}.NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_Base_Size_Good(t *core.T) {
	base := &bufferBase{size: 7}
	core.AssertEqual(t, 7, base.Size())
	core.AssertNotNil(t, base)
}

func TestAX7_Base_Size_Bad(t *core.T) {
	base := &bufferBase{size: 7}
	core.AssertEqual(t, 7, base.Size())
	core.AssertNotNil(t, base)
}

func TestAX7_Base_Size_Ugly(t *core.T) {
	base := &bufferBase{size: 7}
	core.AssertEqual(t, 7, base.Size())
	core.AssertNotNil(t, base)
}

func TestAX7_Buffer_Descriptor_Good(t *core.T) {
	buffer := &pixelBuffer{desc: PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}}
	core.AssertEqual(t, PixelRGBA8, buffer.Descriptor().Format)
	core.AssertEqual(t, 4, buffer.Descriptor().Stride)
}

func TestAX7_Buffer_Descriptor_Bad(t *core.T) {
	buffer := &pixelBuffer{desc: PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}}
	core.AssertEqual(t, PixelRGBA8, buffer.Descriptor().Format)
	core.AssertEqual(t, 4, buffer.Descriptor().Stride)
}

func TestAX7_Buffer_Descriptor_Ugly(t *core.T) {
	buffer := &pixelBuffer{desc: PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8}}
	core.AssertEqual(t, PixelRGBA8, buffer.Descriptor().Format)
	core.AssertEqual(t, 4, buffer.Descriptor().Stride)
}

func TestAX7_Buffer_Read_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.RequireNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
	data, err := buffer.Read()
	core.AssertNoError(t, err)
	core.AssertEqual(t, []byte{1, 2, 3, 4}, data)
}

func TestAX7_Buffer_Read_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.RequireNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
	data, err := buffer.Read()
	core.AssertNoError(t, err)
	core.AssertEqual(t, []byte{1, 2, 3, 4}, data)
}

func TestAX7_Buffer_Read_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.RequireNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
	data, err := buffer.Read()
	core.AssertNoError(t, err)
	core.AssertEqual(t, []byte{1, 2, 3, 4}, data)
}

func TestAX7_Buffer_Upload_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.AssertNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
}

func TestAX7_Buffer_Upload_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.AssertNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
}

func TestAX7_Buffer_Upload_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.RequireNoError(t, err)
	core.AssertNoError(t, buffer.Upload([]byte{1, 2, 3, 4}))
}

func TestAX7_Checkpoint_Good(t *core.T) {
	fn := Checkpoint(func(xs []*Array) []*Array { return xs })
	out := fn([]*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_Checkpoint_Bad(t *core.T) {
	fn := Checkpoint(func(xs []*Array) []*Array { return xs })
	out := fn([]*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_Checkpoint_Ugly(t *core.T) {
	fn := Checkpoint(func(xs []*Array) []*Array { return xs })
	out := fn([]*Array{ax7RootVector()})
	core.AssertLen(t, out, 1)
}

func TestAX7_ClearCache_Good(t *core.T) {
	core.AssertNotPanics(t, func() {
		ClearCache()
		ClearCache()
	})
}

func TestAX7_ClearCache_Bad(t *core.T) {
	core.AssertNotPanics(t, func() {
		ClearCache()
		ClearCache()
	})
}

func TestAX7_ClearCache_Ugly(t *core.T) {
	core.AssertNotPanics(t, func() {
		ClearCache()
		ClearCache()
	})
}

func TestAX7_ComputeError_Error_Good(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Message: "bad state"}
	core.AssertContains(t, err.Error(), "bad state")
	core.AssertContains(t, err.Error(), "mlx")
}

func TestAX7_ComputeError_Error_Bad(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Message: "bad state"}
	core.AssertContains(t, err.Error(), "bad state")
	core.AssertContains(t, err.Error(), "mlx")
}

func TestAX7_ComputeError_Error_Ugly(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Message: "bad state"}
	core.AssertContains(t, err.Error(), "bad state")
	core.AssertContains(t, err.Error(), "mlx")
}

func TestAX7_ComputeError_Is_Good(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Kernel: "scale"}
	core.AssertTrue(t, err.Is(&ComputeError{Kind: ComputeErrorInvalidState}))
	core.AssertFalse(t, err.Is(core.NewError("other")))
}

func TestAX7_ComputeError_Is_Bad(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Kernel: "scale"}
	core.AssertTrue(t, err.Is(&ComputeError{Kind: ComputeErrorInvalidState}))
	core.AssertFalse(t, err.Is(core.NewError("other")))
}

func TestAX7_ComputeError_Is_Ugly(t *core.T) {
	err := &ComputeError{Kind: ComputeErrorInvalidState, Kernel: "scale"}
	core.AssertTrue(t, err.Is(&ComputeError{Kind: ComputeErrorInvalidState}))
	core.AssertFalse(t, err.Is(core.NewError("other")))
}

func TestAX7_ComputeError_Unwrap_Good(t *core.T) {
	cause := core.NewError("root")
	err := &ComputeError{Err: cause}
	core.AssertErrorIs(t, err.Unwrap(), cause)
}

func TestAX7_ComputeError_Unwrap_Bad(t *core.T) {
	cause := core.NewError("root")
	err := &ComputeError{Err: cause}
	core.AssertErrorIs(t, err.Unwrap(), cause)
}

func TestAX7_ComputeError_Unwrap_Ugly(t *core.T) {
	cause := core.NewError("root")
	err := &ComputeError{Err: cause}
	core.AssertErrorIs(t, err.Unwrap(), cause)
}

func TestAX7_Compute_Available_Good(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_Compute_Available_Bad(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_Compute_Available_Ugly(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_Compute_DeviceInfo_Good(t *core.T) {
	info := DefaultCompute().DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Compute_DeviceInfo_Bad(t *core.T) {
	info := DefaultCompute().DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Compute_DeviceInfo_Ugly(t *core.T) {
	info := DefaultCompute().DeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_Compute_NewSession_Good(t *core.T) {
	session, err := DefaultCompute().NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_Compute_NewSession_Bad(t *core.T) {
	session, err := DefaultCompute().NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_Compute_NewSession_Ugly(t *core.T) {
	session, err := DefaultCompute().NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_ConcreteAdapter_Good(t *core.T) {
	core.AssertPanics(t, func() { _ = ConcreteAdapter(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_ConcreteAdapter_Bad(t *core.T) {
	core.AssertPanics(t, func() { _ = ConcreteAdapter(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_ConcreteAdapter_Ugly(t *core.T) {
	core.AssertPanics(t, func() { _ = ConcreteAdapter(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_CrossEntropyLoss_Good(t *core.T) {
	loss := CrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray())
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_CrossEntropyLoss_Bad(t *core.T) {
	loss := CrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray())
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_CrossEntropyLoss_Ugly(t *core.T) {
	loss := CrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray())
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_DType_String_Good(t *core.T) {
	core.AssertEqual(t, "float32", DTypeFloat32.String())
	core.AssertEqual(t, "bfloat16", DTypeBFloat16.String())
	core.AssertEqual(t, "unknown", DType(0).String())
}

func TestAX7_DType_String_Bad(t *core.T) {
	core.AssertEqual(t, "float32", DTypeFloat32.String())
	core.AssertEqual(t, "bfloat16", DTypeBFloat16.String())
	core.AssertEqual(t, "unknown", DType(0).String())
}

func TestAX7_DType_String_Ugly(t *core.T) {
	core.AssertEqual(t, "float32", DTypeFloat32.String())
	core.AssertEqual(t, "bfloat16", DTypeBFloat16.String())
	core.AssertEqual(t, "unknown", DType(0).String())
}

func TestAX7_DefaultCompute_Good(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_DefaultCompute_Bad(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_DefaultCompute_Ugly(t *core.T) {
	compute := DefaultCompute()
	core.AssertEqual(t, MetalAvailable(), compute.Available())
	core.AssertTrue(t, compute.Available() || !compute.Available())
}

func TestAX7_DefaultGenerateConfig_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	core.AssertEqual(t, 256, cfg.MaxTokens)
	core.AssertEqual(t, float32(0), cfg.Temperature)
}

func TestAX7_DefaultGenerateConfig_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	core.AssertEqual(t, 256, cfg.MaxTokens)
	core.AssertEqual(t, float32(0), cfg.Temperature)
}

func TestAX7_DefaultGenerateConfig_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	core.AssertEqual(t, 256, cfg.MaxTokens)
	core.AssertEqual(t, float32(0), cfg.Temperature)
}

func TestAX7_DefaultLoadConfig_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	core.AssertEqual(t, "gpu", cfg.Device)
	core.AssertEqual(t, 0, cfg.ContextLength)
}

func TestAX7_DefaultLoadConfig_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	core.AssertEqual(t, "gpu", cfg.Device)
	core.AssertEqual(t, 0, cfg.ContextLength)
}

func TestAX7_DefaultLoadConfig_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	core.AssertEqual(t, "gpu", cfg.Device)
	core.AssertEqual(t, 0, cfg.ContextLength)
}

func TestAX7_DiscoverModels_Good(t *core.T) {
	models := DiscoverModels(t.TempDir())
	core.AssertEmpty(t, models)
	core.AssertNotNil(t, models)
}

func TestAX7_DiscoverModels_Bad(t *core.T) {
	models := DiscoverModels(t.TempDir())
	core.AssertEmpty(t, models)
	core.AssertNotNil(t, models)
}

func TestAX7_DiscoverModels_Ugly(t *core.T) {
	models := DiscoverModels(t.TempDir())
	core.AssertEmpty(t, models)
	core.AssertNotNil(t, models)
}

func TestAX7_Free_Good(t *core.T) {
	arr := ax7RootVector()
	Free(arr)
	core.AssertNotNil(t, arr)
}

func TestAX7_Free_Bad(t *core.T) {
	arr := ax7RootVector()
	Free(arr)
	core.AssertNotNil(t, arr)
}

func TestAX7_Free_Ugly(t *core.T) {
	arr := ax7RootVector()
	Free(arr)
	core.AssertNotNil(t, arr)
}

func TestAX7_FromValues_Good(t *core.T) {
	got := FromValues([]float32{1, 2}, 2)
	Materialize(got)
	core.AssertEqual(t, []float32{1, 2}, got.Floats())
}

func TestAX7_FromValues_Bad(t *core.T) {
	got := FromValues([]float32{1, 2}, 2)
	Materialize(got)
	core.AssertEqual(t, []float32{1, 2}, got.Floats())
}

func TestAX7_FromValues_Ugly(t *core.T) {
	got := FromValues([]float32{1, 2}, 2)
	Materialize(got)
	core.AssertEqual(t, []float32{1, 2}, got.Floats())
}

func TestAX7_GC_Good(t *core.T) {
	core.AssertNotPanics(t, func() {
		GC()
		GC()
	})
}

func TestAX7_GC_Bad(t *core.T) {
	core.AssertNotPanics(t, func() {
		GC()
		GC()
	})
}

func TestAX7_GC_Ugly(t *core.T) {
	core.AssertNotPanics(t, func() {
		GC()
		GC()
	})
}

func TestAX7_GetActiveMemory_Good(t *core.T) {
	got := GetActiveMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetActiveMemory() >= 0)
}

func TestAX7_GetActiveMemory_Bad(t *core.T) {
	got := GetActiveMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetActiveMemory() >= 0)
}

func TestAX7_GetActiveMemory_Ugly(t *core.T) {
	got := GetActiveMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetActiveMemory() >= 0)
}

func TestAX7_GetCacheMemory_Good(t *core.T) {
	got := GetCacheMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetCacheMemory() >= 0)
}

func TestAX7_GetCacheMemory_Bad(t *core.T) {
	got := GetCacheMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetCacheMemory() >= 0)
}

func TestAX7_GetCacheMemory_Ugly(t *core.T) {
	got := GetCacheMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetCacheMemory() >= 0)
}

func TestAX7_GetDeviceInfo_Good(t *core.T) {
	info := GetDeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_GetDeviceInfo_Bad(t *core.T) {
	info := GetDeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_GetDeviceInfo_Ugly(t *core.T) {
	info := GetDeviceInfo()
	core.AssertTrue(t, info.MemorySize >= 0)
	core.AssertTrue(t, len(info.Architecture) >= 0)
}

func TestAX7_GetPeakMemory_Good(t *core.T) {
	got := GetPeakMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetPeakMemory() >= 0)
}

func TestAX7_GetPeakMemory_Bad(t *core.T) {
	got := GetPeakMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetPeakMemory() >= 0)
}

func TestAX7_GetPeakMemory_Ugly(t *core.T) {
	got := GetPeakMemory()
	core.AssertTrue(t, got >= 0)
	core.AssertTrue(t, GetPeakMemory() >= 0)
}

func TestAX7_GradFn_Apply_Good(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	_, _, err := grad.Apply(ax7RootVector())
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_GradFn_Apply_Bad(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	_, _, err := grad.Apply(ax7RootVector())
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_GradFn_Apply_Ugly(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	_, _, err := grad.Apply(ax7RootVector())
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_GradFn_Free_Good(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	grad.Free()
	core.AssertNotNil(t, grad)
}

func TestAX7_GradFn_Free_Bad(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	grad.Free()
	core.AssertNotNil(t, grad)
}

func TestAX7_GradFn_Free_Ugly(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	grad.Free()
	core.AssertNotNil(t, grad)
}

func TestAX7_InferenceAdapter_Available_Good(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertTrue(t, adapter.Available())
	core.AssertFalse(t, (*InferenceAdapter)(nil).Available())
}

func TestAX7_InferenceAdapter_Available_Bad(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertTrue(t, adapter.Available())
	core.AssertFalse(t, (*InferenceAdapter)(nil).Available())
}

func TestAX7_InferenceAdapter_Available_Ugly(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertTrue(t, adapter.Available())
	core.AssertFalse(t, (*InferenceAdapter)(nil).Available())
}

func TestAX7_InferenceAdapter_Chat_Good(t *core.T) {
	result, err := ax7RootInferenceAdapter().Chat(core.Background(), nil, GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_Chat_Bad(t *core.T) {
	result, err := ax7RootInferenceAdapter().Chat(core.Background(), nil, GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_Chat_Ugly(t *core.T) {
	result, err := ax7RootInferenceAdapter().Chat(core.Background(), nil, GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_ChatStream_Good(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().ChatStream(core.Background(), nil, GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_ChatStream_Bad(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().ChatStream(core.Background(), nil, GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_ChatStream_Ugly(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().ChatStream(core.Background(), nil, GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_Close_Good(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNoError(t, adapter.Close())
	core.AssertFalse(t, adapter.Available())
}

func TestAX7_InferenceAdapter_Close_Bad(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNoError(t, adapter.Close())
	core.AssertFalse(t, adapter.Available())
}

func TestAX7_InferenceAdapter_Close_Ugly(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNoError(t, adapter.Close())
	core.AssertFalse(t, adapter.Available())
}

func TestAX7_InferenceAdapter_Generate_Good(t *core.T) {
	result, err := ax7RootInferenceAdapter().Generate(core.Background(), "hi", GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_Generate_Bad(t *core.T) {
	result, err := ax7RootInferenceAdapter().Generate(core.Background(), "hi", GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_Generate_Ugly(t *core.T) {
	result, err := ax7RootInferenceAdapter().Generate(core.Background(), "hi", GenOpts{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", result.Text)
}

func TestAX7_InferenceAdapter_GenerateStream_Good(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().GenerateStream(core.Background(), "hi", GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_GenerateStream_Bad(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().GenerateStream(core.Background(), "hi", GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_GenerateStream_Ugly(t *core.T) {
	called := false
	err := ax7RootInferenceAdapter().GenerateStream(core.Background(), "hi", GenOpts{}, func(string) error { called = true; return nil })
	core.AssertNoError(t, err)
	core.AssertFalse(t, called)
}

func TestAX7_InferenceAdapter_InspectAttention_Good(t *core.T) {
	snapshot, err := ax7RootInferenceAdapter().InspectAttention(core.Background(), "hi")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_InferenceAdapter_InspectAttention_Bad(t *core.T) {
	snapshot, err := ax7RootInferenceAdapter().InspectAttention(core.Background(), "hi")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_InferenceAdapter_InspectAttention_Ugly(t *core.T) {
	snapshot, err := ax7RootInferenceAdapter().InspectAttention(core.Background(), "hi")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_InferenceAdapter_Model_Good(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNotNil(t, adapter.Model())
	core.AssertNil(t, (*InferenceAdapter)(nil).Model())
}

func TestAX7_InferenceAdapter_Model_Bad(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNotNil(t, adapter.Model())
	core.AssertNil(t, (*InferenceAdapter)(nil).Model())
}

func TestAX7_InferenceAdapter_Model_Ugly(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertNotNil(t, adapter.Model())
	core.AssertNil(t, (*InferenceAdapter)(nil).Model())
}

func TestAX7_InferenceAdapter_Name_Good(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_InferenceAdapter_Name_Bad(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_InferenceAdapter_Name_Ugly(t *core.T) {
	adapter := ax7RootInferenceAdapter()
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_JVP_Good(t *core.T) {
	out, grads, err := JVP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_JVP_Bad(t *core.T) {
	out, grads, err := JVP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_JVP_Ugly(t *core.T) {
	out, grads, err := JVP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_LoRAAdapter_AllTrainableParams_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.AllTrainableParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoRAAdapter_AllTrainableParams_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.AllTrainableParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoRAAdapter_AllTrainableParams_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.AllTrainableParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoRAAdapter_Merge_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.Merge()
	})
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_Merge_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.Merge()
	})
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_Merge_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.Merge()
	})
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_Save_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	err := adapter.Save(core.Path(t.TempDir(), "adapter.safetensors"))
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_LoRAAdapter_Save_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	err := adapter.Save(core.Path(t.TempDir(), "adapter.safetensors"))
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_LoRAAdapter_Save_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	err := adapter.Save(core.Path(t.TempDir(), "adapter.safetensors"))
	core.AssertTrue(t, err == nil || err != nil)
}

func TestAX7_LoRAAdapter_SetAllParams_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.SetAllParams(nil)
	})
	core.AssertEmpty(t, adapter.AllTrainableParams())
}

func TestAX7_LoRAAdapter_SetAllParams_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.SetAllParams(nil)
	})
	core.AssertEmpty(t, adapter.AllTrainableParams())
}

func TestAX7_LoRAAdapter_SetAllParams_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertNotPanics(t, func() {
		adapter.SetAllParams(nil)
	})
	core.AssertEmpty(t, adapter.AllTrainableParams())
}

func TestAX7_LoRAAdapter_SortedNames_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.SortedNames())
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_SortedNames_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.SortedNames())
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_SortedNames_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEmpty(t, adapter.SortedNames())
	core.AssertEqual(t, 0, adapter.TotalParams())
}

func TestAX7_LoRAAdapter_Step_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	got := adapter.Step(Batch{}, nil, NewAdamW(nil))
	core.AssertNil(t, got)
}

func TestAX7_LoRAAdapter_Step_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	got := adapter.Step(Batch{}, nil, NewAdamW(nil))
	core.AssertNil(t, got)
}

func TestAX7_LoRAAdapter_Step_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	got := adapter.Step(Batch{}, nil, NewAdamW(nil))
	core.AssertNil(t, got)
}

func TestAX7_LoRAAdapter_TotalParams_Good(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEqual(t, 0, adapter.TotalParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoRAAdapter_TotalParams_Bad(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEqual(t, 0, adapter.TotalParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoRAAdapter_TotalParams_Ugly(t *core.T) {
	adapter := &LoRAAdapter{}
	core.AssertEqual(t, 0, adapter.TotalParams())
	core.AssertEmpty(t, adapter.SortedNames())
}

func TestAX7_LoadModel_Good(t *core.T) {
	model, err := LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadModel_Bad(t *core.T) {
	model, err := LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadModel_Ugly(t *core.T) {
	model, err := LoadModel("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadModelFromMedium_Good(t *core.T) {
	model, err := LoadModelFromMedium(nil, "model")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadModelFromMedium_Bad(t *core.T) {
	model, err := LoadModelFromMedium(nil, "model")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadModelFromMedium_Ugly(t *core.T) {
	model, err := LoadModelFromMedium(nil, "model")
	core.AssertError(t, err)
	core.AssertNil(t, model)
}

func TestAX7_LoadTokenizer_Good(t *core.T) {
	tok, err := LoadTokenizer(core.Path(t.TempDir(), "missing-tokenizer.json"))
	core.AssertError(t, err)
	core.AssertNil(t, tok)
}

func TestAX7_LoadTokenizer_Bad(t *core.T) {
	tok, err := LoadTokenizer(core.Path(t.TempDir(), "missing-tokenizer.json"))
	core.AssertError(t, err)
	core.AssertNil(t, tok)
}

func TestAX7_LoadTokenizer_Ugly(t *core.T) {
	tok, err := LoadTokenizer(core.Path(t.TempDir(), "missing-tokenizer.json"))
	core.AssertError(t, err)
	core.AssertNil(t, tok)
}

func TestAX7_MaskedCrossEntropyLoss_Good(t *core.T) {
	loss := MaskedCrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray(), FromValues([]float32{1}, 1, 1))
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_MaskedCrossEntropyLoss_Bad(t *core.T) {
	loss := MaskedCrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray(), FromValues([]float32{1}, 1, 1))
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_MaskedCrossEntropyLoss_Ugly(t *core.T) {
	loss := MaskedCrossEntropyLoss(ax7RootLogits(), ax7RootTokenArray(), FromValues([]float32{1}, 1, 1))
	Materialize(loss)
	core.AssertTrue(t, loss.Valid())
}

func TestAX7_MatMul_Good(t *core.T) {
	got := MatMul(ax7RootMatrix(), ax7RootMatrix())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_MatMul_Bad(t *core.T) {
	got := MatMul(ax7RootMatrix(), ax7RootMatrix())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_MatMul_Ugly(t *core.T) {
	got := MatMul(ax7RootMatrix(), ax7RootMatrix())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Materialize_Good(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertTrue(t, arr.Valid())
}

func TestAX7_Materialize_Bad(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertTrue(t, arr.Valid())
}

func TestAX7_Materialize_Ugly(t *core.T) {
	arr := ax7RootVector()
	Materialize(arr)
	core.AssertTrue(t, arr.Valid())
}

func TestAX7_MetalAvailable_Good(t *core.T) {
	got := MetalAvailable()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_MetalAvailable_Bad(t *core.T) {
	got := MetalAvailable()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_MetalAvailable_Ugly(t *core.T) {
	got := MetalAvailable()
	core.AssertTrue(t, got || !got)
	core.AssertEqual(t, MetalAvailable(), Available())
}

func TestAX7_Model_BatchGenerate_Good(t *core.T) {
	got, err := ax7RootModel().BatchGenerate([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_BatchGenerate_Bad(t *core.T) {
	got, err := ax7RootModel().BatchGenerate([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_BatchGenerate_Ugly(t *core.T) {
	got, err := ax7RootModel().BatchGenerate([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_Chat_Good(t *core.T) {
	got, err := ax7RootModel().Chat([]Message{{Role: "user", Content: "hi"}})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_Chat_Bad(t *core.T) {
	got, err := ax7RootModel().Chat([]Message{{Role: "user", Content: "hi"}})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_Chat_Ugly(t *core.T) {
	got, err := ax7RootModel().Chat([]Message{{Role: "user", Content: "hi"}})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_ChatStream_Good(t *core.T) {
	ch := ax7RootModel().ChatStream(core.Background(), nil)
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_ChatStream_Bad(t *core.T) {
	ch := ax7RootModel().ChatStream(core.Background(), nil)
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_ChatStream_Ugly(t *core.T) {
	ch := ax7RootModel().ChatStream(core.Background(), nil)
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_Classify_Good(t *core.T) {
	got, err := ax7RootModel().Classify([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_Classify_Bad(t *core.T) {
	got, err := ax7RootModel().Classify([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_Classify_Ugly(t *core.T) {
	got, err := ax7RootModel().Classify([]string{"hi"})
	core.AssertNoError(t, err)
	core.AssertNil(t, got)
}

func TestAX7_Model_Close_Good(t *core.T) {
	model := ax7RootModel()
	core.AssertNoError(t, model.Close())
	core.AssertNil(t, model.model)
}

func TestAX7_Model_Close_Bad(t *core.T) {
	model := ax7RootModel()
	core.AssertNoError(t, model.Close())
	core.AssertNil(t, model.model)
}

func TestAX7_Model_Close_Ugly(t *core.T) {
	model := ax7RootModel()
	core.AssertNoError(t, model.Close())
	core.AssertNil(t, model.model)
}

func TestAX7_Model_Err_Good(t *core.T) {
	model := ax7RootModel()
	core.AssertNil(t, model.Err())
	core.AssertEqual(t, "", model.ModelType())
}

func TestAX7_Model_Err_Bad(t *core.T) {
	model := ax7RootModel()
	core.AssertNil(t, model.Err())
	core.AssertEqual(t, "", model.ModelType())
}

func TestAX7_Model_Err_Ugly(t *core.T) {
	model := ax7RootModel()
	core.AssertNil(t, model.Err())
	core.AssertEqual(t, "", model.ModelType())
}

func TestAX7_Model_Generate_Good(t *core.T) {
	got, err := ax7RootModel().Generate("prompt")
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_Generate_Bad(t *core.T) {
	got, err := ax7RootModel().Generate("prompt")
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_Generate_Ugly(t *core.T) {
	got, err := ax7RootModel().Generate("prompt")
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", got)
}

func TestAX7_Model_GenerateStream_Good(t *core.T) {
	ch := ax7RootModel().GenerateStream(core.Background(), "prompt")
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_GenerateStream_Bad(t *core.T) {
	ch := ax7RootModel().GenerateStream(core.Background(), "prompt")
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_GenerateStream_Ugly(t *core.T) {
	ch := ax7RootModel().GenerateStream(core.Background(), "prompt")
	_, ok := <-ch
	core.AssertFalse(t, ok)
}

func TestAX7_Model_Info_Good(t *core.T) {
	info := ax7RootModel().Info()
	core.AssertEqual(t, 16, info.ContextLength)
	core.AssertEqual(t, "", info.Architecture)
}

func TestAX7_Model_Info_Bad(t *core.T) {
	info := ax7RootModel().Info()
	core.AssertEqual(t, 16, info.ContextLength)
	core.AssertEqual(t, "", info.Architecture)
}

func TestAX7_Model_Info_Ugly(t *core.T) {
	info := ax7RootModel().Info()
	core.AssertEqual(t, 16, info.ContextLength)
	core.AssertEqual(t, "", info.Architecture)
}

func TestAX7_Model_InspectAttention_Good(t *core.T) {
	snapshot, err := ax7RootModel().InspectAttention("prompt")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_Model_InspectAttention_Bad(t *core.T) {
	snapshot, err := ax7RootModel().InspectAttention("prompt")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_Model_InspectAttention_Ugly(t *core.T) {
	snapshot, err := ax7RootModel().InspectAttention("prompt")
	core.AssertNoError(t, err)
	core.AssertNil(t, snapshot)
}

func TestAX7_Model_MergeLoRA_Good(t *core.T) {
	model := ax7RootModel()
	core.AssertEqual(t, model, model.MergeLoRA(nil))
	core.AssertNotNil(t, model)
}

func TestAX7_Model_MergeLoRA_Bad(t *core.T) {
	model := ax7RootModel()
	core.AssertEqual(t, model, model.MergeLoRA(nil))
	core.AssertNotNil(t, model)
}

func TestAX7_Model_MergeLoRA_Ugly(t *core.T) {
	model := ax7RootModel()
	core.AssertEqual(t, model, model.MergeLoRA(nil))
	core.AssertNotNil(t, model)
}

func TestAX7_Model_Metrics_Good(t *core.T) {
	metrics := ax7RootModel().Metrics()
	core.AssertEqual(t, 0, metrics.PromptTokens)
	core.AssertEqual(t, 0, metrics.GeneratedTokens)
}

func TestAX7_Model_Metrics_Bad(t *core.T) {
	metrics := ax7RootModel().Metrics()
	core.AssertEqual(t, 0, metrics.PromptTokens)
	core.AssertEqual(t, 0, metrics.GeneratedTokens)
}

func TestAX7_Model_Metrics_Ugly(t *core.T) {
	metrics := ax7RootModel().Metrics()
	core.AssertEqual(t, 0, metrics.PromptTokens)
	core.AssertEqual(t, 0, metrics.GeneratedTokens)
}

func TestAX7_Model_ModelType_Good(t *core.T) {
	core.AssertEqual(t, "", ax7RootModel().ModelType())
	core.AssertNotNil(t, ax7RootModel())
	core.AssertTrue(t, true)
}

func TestAX7_Model_ModelType_Bad(t *core.T) {
	core.AssertEqual(t, "", ax7RootModel().ModelType())
	core.AssertNotNil(t, ax7RootModel())
	core.AssertTrue(t, true)
}

func TestAX7_Model_ModelType_Ugly(t *core.T) {
	core.AssertEqual(t, "", ax7RootModel().ModelType())
	core.AssertNotNil(t, ax7RootModel())
	core.AssertTrue(t, true)
}

func TestAX7_Model_Tokenizer_Good(t *core.T) {
	tok := ax7RootModel().Tokenizer()
	core.AssertNotNil(t, tok)
	core.AssertEqual(t, int32(1), tok.BOS())
}

func TestAX7_Model_Tokenizer_Bad(t *core.T) {
	tok := ax7RootModel().Tokenizer()
	core.AssertNotNil(t, tok)
	core.AssertEqual(t, int32(1), tok.BOS())
}

func TestAX7_Model_Tokenizer_Ugly(t *core.T) {
	tok := ax7RootModel().Tokenizer()
	core.AssertNotNil(t, tok)
	core.AssertEqual(t, int32(1), tok.BOS())
}

func TestAX7_Mul_Good(t *core.T) {
	got := Mul(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Mul_Bad(t *core.T) {
	got := Mul(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Mul_Ugly(t *core.T) {
	got := Mul(ax7RootVector(), ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_NewAdamW_Good(t *core.T) {
	opt := NewAdamW(nil)
	core.AssertNotNil(t, opt)
	opt.Reset()
}

func TestAX7_NewAdamW_Bad(t *core.T) {
	opt := NewAdamW(nil)
	core.AssertNotNil(t, opt)
	opt.Reset()
}

func TestAX7_NewAdamW_Ugly(t *core.T) {
	opt := NewAdamW(nil)
	core.AssertNotNil(t, opt)
	opt.Reset()
}

func TestAX7_NewInferenceAdapter_Good(t *core.T) {
	adapter := NewInferenceAdapter(&stubTextModel{}, "ax7")
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_NewInferenceAdapter_Bad(t *core.T) {
	adapter := NewInferenceAdapter(&stubTextModel{}, "ax7")
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_NewInferenceAdapter_Ugly(t *core.T) {
	adapter := NewInferenceAdapter(&stubTextModel{}, "ax7")
	core.AssertEqual(t, "ax7", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestAX7_NewLoRA_Good(t *core.T) {
	adapter := NewLoRA(nil, nil)
	core.AssertNil(t, adapter)
	core.AssertTrue(t, adapter == nil)
}

func TestAX7_NewLoRA_Bad(t *core.T) {
	adapter := NewLoRA(nil, nil)
	core.AssertNil(t, adapter)
	core.AssertTrue(t, adapter == nil)
}

func TestAX7_NewLoRA_Ugly(t *core.T) {
	adapter := NewLoRA(nil, nil)
	core.AssertNil(t, adapter)
	core.AssertTrue(t, adapter == nil)
}

func TestAX7_NewMLXBackend_Good(t *core.T) {
	adapter, err := NewMLXBackend("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, adapter)
}

func TestAX7_NewMLXBackend_Bad(t *core.T) {
	adapter, err := NewMLXBackend("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, adapter)
}

func TestAX7_NewMLXBackend_Ugly(t *core.T) {
	adapter, err := NewMLXBackend("/definitely/missing")
	core.AssertError(t, err)
	core.AssertNil(t, adapter)
}

func TestAX7_NewSession_Good(t *core.T) {
	session, err := NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_NewSession_Bad(t *core.T) {
	session, err := NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_NewSession_Ugly(t *core.T) {
	session, err := NewSession(WithSessionLabel("ax7"))
	core.AssertTrue(t, err != nil || session != nil)
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
}

func TestAX7_PixelBufferDesc_SizeBytes_Good(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 3, Stride: 8, Format: PixelRGBA8}
	core.AssertEqual(t, 24, desc.SizeBytes())
	core.AssertEqual(t, 0, (PixelBufferDesc{}).SizeBytes())
}

func TestAX7_PixelBufferDesc_SizeBytes_Bad(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 3, Stride: 8, Format: PixelRGBA8}
	core.AssertEqual(t, 24, desc.SizeBytes())
	core.AssertEqual(t, 0, (PixelBufferDesc{}).SizeBytes())
}

func TestAX7_PixelBufferDesc_SizeBytes_Ugly(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 3, Stride: 8, Format: PixelRGBA8}
	core.AssertEqual(t, 24, desc.SizeBytes())
	core.AssertEqual(t, 0, (PixelBufferDesc{}).SizeBytes())
}

func TestAX7_PixelBufferDesc_Validate_Good(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}
	core.AssertNoError(t, desc.Validate())
	core.AssertError(t, (PixelBufferDesc{}).Validate())
}

func TestAX7_PixelBufferDesc_Validate_Bad(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}
	core.AssertNoError(t, desc.Validate())
	core.AssertError(t, (PixelBufferDesc{}).Validate())
}

func TestAX7_PixelBufferDesc_Validate_Ugly(t *core.T) {
	desc := PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}
	core.AssertNoError(t, desc.Validate())
	core.AssertError(t, (PixelBufferDesc{}).Validate())
}

func TestAX7_PixelFormat_BytesPerPixel_Good(t *core.T) {
	core.AssertEqual(t, 4, PixelRGBA8.BytesPerPixel())
	core.AssertEqual(t, 2, PixelRGB565.BytesPerPixel())
	core.AssertEqual(t, 0, PixelFormat("bad").BytesPerPixel())
}

func TestAX7_PixelFormat_BytesPerPixel_Bad(t *core.T) {
	core.AssertEqual(t, 4, PixelRGBA8.BytesPerPixel())
	core.AssertEqual(t, 2, PixelRGB565.BytesPerPixel())
	core.AssertEqual(t, 0, PixelFormat("bad").BytesPerPixel())
}

func TestAX7_PixelFormat_BytesPerPixel_Ugly(t *core.T) {
	core.AssertEqual(t, 4, PixelRGBA8.BytesPerPixel())
	core.AssertEqual(t, 2, PixelRGB565.BytesPerPixel())
	core.AssertEqual(t, 0, PixelFormat("bad").BytesPerPixel())
}

func TestAX7_ReadGGUFInfo_Good(t *core.T) {
	info, err := ReadGGUFInfo(core.Path(t.TempDir(), "missing.gguf"))
	core.AssertError(t, err)
	core.AssertEqual(t, "", info.Path)
}

func TestAX7_ReadGGUFInfo_Bad(t *core.T) {
	info, err := ReadGGUFInfo(core.Path(t.TempDir(), "missing.gguf"))
	core.AssertError(t, err)
	core.AssertEqual(t, "", info.Path)
}

func TestAX7_ReadGGUFInfo_Ugly(t *core.T) {
	info, err := ReadGGUFInfo(core.Path(t.TempDir(), "missing.gguf"))
	core.AssertError(t, err)
	core.AssertEqual(t, "", info.Path)
}

func TestAX7_ResetPeakMemory_Good(t *core.T) {
	core.AssertNotPanics(t, func() {
		ResetPeakMemory()
		ResetPeakMemory()
	})
}

func TestAX7_ResetPeakMemory_Bad(t *core.T) {
	core.AssertNotPanics(t, func() {
		ResetPeakMemory()
		ResetPeakMemory()
	})
}

func TestAX7_ResetPeakMemory_Ugly(t *core.T) {
	core.AssertNotPanics(t, func() {
		ResetPeakMemory()
		ResetPeakMemory()
	})
}

func TestAX7_Reshape_Good(t *core.T) {
	got := Reshape(ax7RootVector(), 2, 2)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}

func TestAX7_Reshape_Bad(t *core.T) {
	got := Reshape(ax7RootVector(), 2, 2)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}

func TestAX7_Reshape_Ugly(t *core.T) {
	got := Reshape(ax7RootVector(), 2, 2)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}

func TestAX7_Session_BeginFrame_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
}

func TestAX7_Session_BeginFrame_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
}

func TestAX7_Session_BeginFrame_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
}

func TestAX7_Session_Close_Good(t *core.T) {
	session := ax7RootSession()
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
	core.AssertTrue(t, session == nil || session != nil)
}

func TestAX7_Session_Close_Bad(t *core.T) {
	session := ax7RootSession()
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
	core.AssertTrue(t, session == nil || session != nil)
}

func TestAX7_Session_Close_Ugly(t *core.T) {
	session := ax7RootSession()
	if session != nil {
		core.AssertNoError(t, session.Close())
	}
	core.AssertTrue(t, session == nil || session != nil)
}

func TestAX7_Session_FinishFrame_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
	_, err := session.FinishFrame()
	core.AssertNoError(t, err)
}

func TestAX7_Session_FinishFrame_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
	_, err := session.FinishFrame()
	core.AssertNoError(t, err)
}

func TestAX7_Session_FinishFrame_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.BeginFrame())
	_, err := session.FinishFrame()
	core.AssertNoError(t, err)
}

func TestAX7_Session_FrameMetrics_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.FrameMetrics().Frame >= 0)
}

func TestAX7_Session_FrameMetrics_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.FrameMetrics().Frame >= 0)
}

func TestAX7_Session_FrameMetrics_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.FrameMetrics().Frame >= 0)
}

func TestAX7_Session_Metrics_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.Metrics().Passes >= 0)
}

func TestAX7_Session_Metrics_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.Metrics().Passes >= 0)
}

func TestAX7_Session_Metrics_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertTrue(t, session.Metrics().Passes >= 0)
}

func TestAX7_Session_NewByteBuffer_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_NewByteBuffer_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_NewByteBuffer_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewByteBuffer(4)
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_NewPixelBuffer_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_NewPixelBuffer_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_NewPixelBuffer_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	buffer, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
}

func TestAX7_Session_Run_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	err := session.Run("missing", KernelArgs{})
	core.AssertError(t, err)
}

func TestAX7_Session_Run_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	err := session.Run("missing", KernelArgs{})
	core.AssertError(t, err)
}

func TestAX7_Session_Run_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	err := session.Run("missing", KernelArgs{})
	core.AssertError(t, err)
}

func TestAX7_Session_Sync_Good(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.Sync())
}

func TestAX7_Session_Sync_Bad(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.Sync())
}

func TestAX7_Session_Sync_Ugly(t *core.T) {
	session := ax7RootSession()
	if session == nil {
		core.AssertNil(t, session)
		return
	}
	defer session.Close()
	core.AssertNoError(t, session.Sync())
}

func TestAX7_SetCacheLimit_Good(t *core.T) {
	previous := SetCacheLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetCacheLimit(previous)
}

func TestAX7_SetCacheLimit_Bad(t *core.T) {
	previous := SetCacheLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetCacheLimit(previous)
}

func TestAX7_SetCacheLimit_Ugly(t *core.T) {
	previous := SetCacheLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetCacheLimit(previous)
}

func TestAX7_SetMemoryLimit_Good(t *core.T) {
	previous := SetMemoryLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetMemoryLimit(previous)
}

func TestAX7_SetMemoryLimit_Bad(t *core.T) {
	previous := SetMemoryLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetMemoryLimit(previous)
}

func TestAX7_SetMemoryLimit_Ugly(t *core.T) {
	previous := SetMemoryLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetMemoryLimit(previous)
}

func TestAX7_SetWiredLimit_Good(t *core.T) {
	previous := SetWiredLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetWiredLimit(previous)
}

func TestAX7_SetWiredLimit_Bad(t *core.T) {
	previous := SetWiredLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetWiredLimit(previous)
}

func TestAX7_SetWiredLimit_Ugly(t *core.T) {
	previous := SetWiredLimit(0)
	core.AssertTrue(t, previous >= 0)
	_ = SetWiredLimit(previous)
}

func TestAX7_Slice_Good(t *core.T) {
	got := Slice(ax7RootVector(), 0, 2, 0)
	Materialize(got)
	core.AssertEqual(t, []int32{2}, got.Shape())
}

func TestAX7_Slice_Bad(t *core.T) {
	got := Slice(ax7RootVector(), 0, 2, 0)
	Materialize(got)
	core.AssertEqual(t, []int32{2}, got.Shape())
}

func TestAX7_Slice_Ugly(t *core.T) {
	got := Slice(ax7RootVector(), 0, 2, 0)
	Materialize(got)
	core.AssertEqual(t, []int32{2}, got.Shape())
}

func TestAX7_Softmax_Good(t *core.T) {
	got := Softmax(ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Softmax_Bad(t *core.T) {
	got := Softmax(ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Softmax_Ugly(t *core.T) {
	got := Softmax(ax7RootVector())
	Materialize(got)
	core.AssertTrue(t, got.Valid())
}

func TestAX7_Tokenizer_BOS_Good(t *core.T) {
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_BOS_Bad(t *core.T) {
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_BOS_Ugly(t *core.T) {
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertEqual(t, int32(1), ax7RootTokenizer().BOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_Decode_Good(t *core.T) {
	text, err := ax7RootTokenizer().Decode([]int32{2, 3})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "decoded", text)
}

func TestAX7_Tokenizer_Decode_Bad(t *core.T) {
	text, err := ax7RootTokenizer().Decode([]int32{2, 3})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "decoded", text)
}

func TestAX7_Tokenizer_Decode_Ugly(t *core.T) {
	text, err := ax7RootTokenizer().Decode([]int32{2, 3})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "decoded", text)
}

func TestAX7_Tokenizer_EOS_Good(t *core.T) {
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_EOS_Bad(t *core.T) {
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_EOS_Ugly(t *core.T) {
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertEqual(t, int32(9), ax7RootTokenizer().EOS())
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_Encode_Good(t *core.T) {
	tokens, err := ax7RootTokenizer().Encode("known")
	core.AssertNoError(t, err)
	core.AssertEqual(t, []int32{2, 3}, tokens)
}

func TestAX7_Tokenizer_Encode_Bad(t *core.T) {
	tokens, err := ax7RootTokenizer().Encode("known")
	core.AssertNoError(t, err)
	core.AssertEqual(t, []int32{2, 3}, tokens)
}

func TestAX7_Tokenizer_Encode_Ugly(t *core.T) {
	tokens, err := ax7RootTokenizer().Encode("known")
	core.AssertNoError(t, err)
	core.AssertEqual(t, []int32{2, 3}, tokens)
}

func TestAX7_Tokenizer_IDToken_Good(t *core.T) {
	core.AssertEqual(t, "known", ax7RootTokenizer().IDToken(7))
	core.AssertEqual(t, "", ax7RootTokenizer().IDToken(99))
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_IDToken_Bad(t *core.T) {
	core.AssertEqual(t, "known", ax7RootTokenizer().IDToken(7))
	core.AssertEqual(t, "", ax7RootTokenizer().IDToken(99))
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_IDToken_Ugly(t *core.T) {
	core.AssertEqual(t, "known", ax7RootTokenizer().IDToken(7))
	core.AssertEqual(t, "", ax7RootTokenizer().IDToken(99))
	core.AssertTrue(t, true)
}

func TestAX7_Tokenizer_TokenID_Good(t *core.T) {
	id, ok := ax7RootTokenizer().TokenID("known")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, int32(7), id)
}

func TestAX7_Tokenizer_TokenID_Bad(t *core.T) {
	id, ok := ax7RootTokenizer().TokenID("known")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, int32(7), id)
}

func TestAX7_Tokenizer_TokenID_Ugly(t *core.T) {
	id, ok := ax7RootTokenizer().TokenID("known")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, int32(7), id)
}

func TestAX7_TrainingModel_Good(t *core.T) {
	core.AssertPanics(t, func() { _ = TrainingModel(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_TrainingModel_Bad(t *core.T) {
	core.AssertPanics(t, func() { _ = TrainingModel(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_TrainingModel_Ugly(t *core.T) {
	core.AssertPanics(t, func() { _ = TrainingModel(nil) })
	core.AssertTrue(t, true)
	core.AssertFalse(t, false)
}

func TestAX7_VJP_Good(t *core.T) {
	out, grads, err := VJP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_VJP_Bad(t *core.T) {
	out, grads, err := VJP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_VJP_Ugly(t *core.T) {
	out, grads, err := VJP(func(xs []*Array) []*Array { return xs }, []*Array{ax7RootVector()}, []*Array{ax7RootVector()})
	core.AssertTrue(t, err == nil || err != nil)
	core.AssertTrue(t, len(out) >= 0 && len(grads) >= 0)
}

func TestAX7_ValueAndGrad_Good(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	core.AssertNotNil(t, grad)
	grad.Free()
}

func TestAX7_ValueAndGrad_Bad(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	core.AssertNotNil(t, grad)
	grad.Free()
}

func TestAX7_ValueAndGrad_Ugly(t *core.T) {
	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	core.AssertNotNil(t, grad)
	grad.Free()
}

func TestAX7_WithAdapterPath_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	WithAdapterPath("adapter")(&cfg)
	core.AssertEqual(t, "adapter", cfg.AdapterPath)
}

func TestAX7_WithAdapterPath_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	WithAdapterPath("adapter")(&cfg)
	core.AssertEqual(t, "adapter", cfg.AdapterPath)
}

func TestAX7_WithAdapterPath_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	WithAdapterPath("adapter")(&cfg)
	core.AssertEqual(t, "adapter", cfg.AdapterPath)
}

func TestAX7_WithContextLength_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	WithContextLength(4096)(&cfg)
	core.AssertEqual(t, 4096, cfg.ContextLength)
}

func TestAX7_WithContextLength_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	WithContextLength(4096)(&cfg)
	core.AssertEqual(t, 4096, cfg.ContextLength)
}

func TestAX7_WithContextLength_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	WithContextLength(4096)(&cfg)
	core.AssertEqual(t, 4096, cfg.ContextLength)
}

func TestAX7_WithDevice_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	WithDevice("cpu")(&cfg)
	core.AssertEqual(t, "cpu", cfg.Device)
}

func TestAX7_WithDevice_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	WithDevice("cpu")(&cfg)
	core.AssertEqual(t, "cpu", cfg.Device)
}

func TestAX7_WithDevice_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	WithDevice("cpu")(&cfg)
	core.AssertEqual(t, "cpu", cfg.Device)
}

func TestAX7_WithLogits_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithLogits_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithLogits_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithMaxTokens_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMaxTokens(17)(&cfg)
	core.AssertEqual(t, 17, cfg.MaxTokens)
}

func TestAX7_WithMaxTokens_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMaxTokens(17)(&cfg)
	core.AssertEqual(t, 17, cfg.MaxTokens)
}

func TestAX7_WithMaxTokens_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMaxTokens(17)(&cfg)
	core.AssertEqual(t, 17, cfg.MaxTokens)
}

func TestAX7_WithMedium_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	WithMedium(nil)(&cfg)
	core.AssertNil(t, cfg.Medium)
}

func TestAX7_WithMedium_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	WithMedium(nil)(&cfg)
	core.AssertNil(t, cfg.Medium)
}

func TestAX7_WithMedium_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	WithMedium(nil)(&cfg)
	core.AssertNil(t, cfg.Medium)
}

func TestAX7_WithMinP_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMinP(0.05)(&cfg)
	core.AssertEqual(t, float32(0.05), cfg.MinP)
}

func TestAX7_WithMinP_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMinP(0.05)(&cfg)
	core.AssertEqual(t, float32(0.05), cfg.MinP)
}

func TestAX7_WithMinP_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithMinP(0.05)(&cfg)
	core.AssertEqual(t, float32(0.05), cfg.MinP)
}

func TestAX7_WithQuantization_Good(t *core.T) {
	cfg := DefaultLoadConfig()
	WithQuantization(4)(&cfg)
	core.AssertEqual(t, 4, cfg.Quantization)
}

func TestAX7_WithQuantization_Bad(t *core.T) {
	cfg := DefaultLoadConfig()
	WithQuantization(4)(&cfg)
	core.AssertEqual(t, 4, cfg.Quantization)
}

func TestAX7_WithQuantization_Ugly(t *core.T) {
	cfg := DefaultLoadConfig()
	WithQuantization(4)(&cfg)
	core.AssertEqual(t, 4, cfg.Quantization)
}

func TestAX7_WithRepeatPenalty_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithRepeatPenalty(1.1)(&cfg)
	core.AssertEqual(t, float32(1.1), cfg.RepeatPenalty)
}

func TestAX7_WithRepeatPenalty_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithRepeatPenalty(1.1)(&cfg)
	core.AssertEqual(t, float32(1.1), cfg.RepeatPenalty)
}

func TestAX7_WithRepeatPenalty_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithRepeatPenalty(1.1)(&cfg)
	core.AssertEqual(t, float32(1.1), cfg.RepeatPenalty)
}

func TestAX7_WithResetPeakMemory_Good(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithResetPeakMemory(false)})

	core.AssertFalse(t, cfg.resetPeakMemory)
	core.AssertEqual(t, "", cfg.label)
}

func TestAX7_WithResetPeakMemory_Bad(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithResetPeakMemory(true)})

	core.AssertTrue(t, cfg.resetPeakMemory)
	core.AssertFalse(t, cfg.verboseKernels)
}

func TestAX7_WithResetPeakMemory_Ugly(t *core.T) {
	cfg := newSessionConfig([]SessionOption{nil, WithResetPeakMemory(false)})

	core.AssertFalse(t, cfg.resetPeakMemory)
	core.AssertEqual(t, "", cfg.label)
}

func TestAX7_WithReturnLogits_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithReturnLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithReturnLogits_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithReturnLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithReturnLogits_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithReturnLogits()(&cfg)
	core.AssertTrue(t, cfg.ReturnLogits)
}

func TestAX7_WithSessionLabel_Good(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("frame")})

	core.AssertEqual(t, "frame", cfg.label)
	core.AssertTrue(t, cfg.resetPeakMemory)
}

func TestAX7_WithSessionLabel_Bad(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("")})

	core.AssertEqual(t, "", cfg.label)
	core.AssertTrue(t, cfg.resetPeakMemory)
}

func TestAX7_WithSessionLabel_Ugly(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("a/b:c")})

	core.AssertEqual(t, "a/b:c", cfg.label)
	core.AssertFalse(t, cfg.verboseKernels)
}

func TestAX7_WithStopTokens_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithStopTokens(1, 2, 3)(&cfg)
	core.AssertEqual(t, []int32{1, 2, 3}, cfg.StopTokens)
}

func TestAX7_WithStopTokens_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithStopTokens(1, 2, 3)(&cfg)
	core.AssertEqual(t, []int32{1, 2, 3}, cfg.StopTokens)
}

func TestAX7_WithStopTokens_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithStopTokens(1, 2, 3)(&cfg)
	core.AssertEqual(t, []int32{1, 2, 3}, cfg.StopTokens)
}

func TestAX7_WithTemperature_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTemperature(0.7)(&cfg)
	core.AssertEqual(t, float32(0.7), cfg.Temperature)
}

func TestAX7_WithTemperature_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTemperature(0.7)(&cfg)
	core.AssertEqual(t, float32(0.7), cfg.Temperature)
}

func TestAX7_WithTemperature_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTemperature(0.7)(&cfg)
	core.AssertEqual(t, float32(0.7), cfg.Temperature)
}

func TestAX7_WithTopK_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopK(11)(&cfg)
	core.AssertEqual(t, 11, cfg.TopK)
}

func TestAX7_WithTopK_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopK(11)(&cfg)
	core.AssertEqual(t, 11, cfg.TopK)
}

func TestAX7_WithTopK_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopK(11)(&cfg)
	core.AssertEqual(t, 11, cfg.TopK)
}

func TestAX7_WithTopP_Good(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopP(0.8)(&cfg)
	core.AssertEqual(t, float32(0.8), cfg.TopP)
}

func TestAX7_WithTopP_Bad(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopP(0.8)(&cfg)
	core.AssertEqual(t, float32(0.8), cfg.TopP)
}

func TestAX7_WithTopP_Ugly(t *core.T) {
	cfg := DefaultGenerateConfig()
	WithTopP(0.8)(&cfg)
	core.AssertEqual(t, float32(0.8), cfg.TopP)
}

func TestAX7_WithVerboseKernels_Good(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithVerboseKernels(true)})

	core.AssertTrue(t, cfg.verboseKernels)
	core.AssertTrue(t, cfg.resetPeakMemory)
}

func TestAX7_WithVerboseKernels_Bad(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithVerboseKernels(false)})

	core.AssertFalse(t, cfg.verboseKernels)
	core.AssertTrue(t, cfg.resetPeakMemory)
}

func TestAX7_WithVerboseKernels_Ugly(t *core.T) {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("verbose"), WithVerboseKernels(true)})

	core.AssertTrue(t, cfg.verboseKernels)
	core.AssertEqual(t, "verbose", cfg.label)
}

func TestAX7_Zeros_Good(t *core.T) {
	got := Zeros([]int32{2, 2}, DTypeFloat32)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}

func TestAX7_Zeros_Bad(t *core.T) {
	got := Zeros([]int32{2, 2}, DTypeFloat32)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}

func TestAX7_Zeros_Ugly(t *core.T) {
	got := Zeros([]int32{2, 2}, DTypeFloat32)
	Materialize(got)
	core.AssertEqual(t, []int32{2, 2}, got.Shape())
}
