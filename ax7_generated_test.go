// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import core "dappco.re/go"

func TestAX7_AdamW_Reset_Good(t *core.T) {
	symbol := any((*AdamW).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Reset_Good", "AdamW_Reset")
}

func TestAX7_AdamW_Reset_Bad(t *core.T) {
	symbol := any((*AdamW).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Reset_Bad", "AdamW_Reset")
}

func TestAX7_AdamW_Reset_Ugly(t *core.T) {
	symbol := any((*AdamW).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Reset_Ugly", "AdamW_Reset")
}

func TestAX7_AdamW_Step_Good(t *core.T) {
	symbol := any((*AdamW).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Step_Good", "AdamW_Step")
}

func TestAX7_AdamW_Step_Bad(t *core.T) {
	symbol := any((*AdamW).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Step_Bad", "AdamW_Step")
}

func TestAX7_AdamW_Step_Ugly(t *core.T) {
	symbol := any((*AdamW).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AdamW_Step_Ugly", "AdamW_Step")
}

func TestAX7_Adapter_ApplyLoRA_Good(t *core.T) {
	symbol := any((*metalAdapter).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ApplyLoRA_Good", "Adapter_ApplyLoRA")
}

func TestAX7_Adapter_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*metalAdapter).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ApplyLoRA_Bad", "Adapter_ApplyLoRA")
}

func TestAX7_Adapter_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*metalAdapter).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ApplyLoRA_Ugly", "Adapter_ApplyLoRA")
}

func TestAX7_Adapter_BatchGenerate_Good(t *core.T) {
	symbol := any((*metalAdapter).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_BatchGenerate_Good", "Adapter_BatchGenerate")
}

func TestAX7_Adapter_BatchGenerate_Bad(t *core.T) {
	symbol := any((*metalAdapter).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_BatchGenerate_Bad", "Adapter_BatchGenerate")
}

func TestAX7_Adapter_BatchGenerate_Ugly(t *core.T) {
	symbol := any((*metalAdapter).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_BatchGenerate_Ugly", "Adapter_BatchGenerate")
}

func TestAX7_Adapter_Chat_Good(t *core.T) {
	symbol := any((*metalAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Chat_Good", "Adapter_Chat")
}

func TestAX7_Adapter_Chat_Bad(t *core.T) {
	symbol := any((*metalAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Chat_Bad", "Adapter_Chat")
}

func TestAX7_Adapter_Chat_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Chat_Ugly", "Adapter_Chat")
}

func TestAX7_Adapter_Classify_Good(t *core.T) {
	symbol := any((*metalAdapter).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Classify_Good", "Adapter_Classify")
}

func TestAX7_Adapter_Classify_Bad(t *core.T) {
	symbol := any((*metalAdapter).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Classify_Bad", "Adapter_Classify")
}

func TestAX7_Adapter_Classify_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Classify_Ugly", "Adapter_Classify")
}

func TestAX7_Adapter_Close_Good(t *core.T) {
	symbol := any((*metalAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Close_Good", "Adapter_Close")
}

func TestAX7_Adapter_Close_Bad(t *core.T) {
	symbol := any((*metalAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Close_Bad", "Adapter_Close")
}

func TestAX7_Adapter_Close_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Close_Ugly", "Adapter_Close")
}

func TestAX7_Adapter_Decode_Good(t *core.T) {
	symbol := any((*metalAdapter).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Decode_Good", "Adapter_Decode")
}

func TestAX7_Adapter_Decode_Bad(t *core.T) {
	symbol := any((*metalAdapter).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Decode_Bad", "Adapter_Decode")
}

func TestAX7_Adapter_Decode_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Decode_Ugly", "Adapter_Decode")
}

func TestAX7_Adapter_Encode_Good(t *core.T) {
	symbol := any((*metalAdapter).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Encode_Good", "Adapter_Encode")
}

func TestAX7_Adapter_Encode_Bad(t *core.T) {
	symbol := any((*metalAdapter).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Encode_Bad", "Adapter_Encode")
}

func TestAX7_Adapter_Encode_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Encode_Ugly", "Adapter_Encode")
}

func TestAX7_Adapter_Err_Good(t *core.T) {
	symbol := any((*metalAdapter).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Err_Good", "Adapter_Err")
}

func TestAX7_Adapter_Err_Bad(t *core.T) {
	symbol := any((*metalAdapter).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Err_Bad", "Adapter_Err")
}

func TestAX7_Adapter_Err_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Err_Ugly", "Adapter_Err")
}

func TestAX7_Adapter_Generate_Good(t *core.T) {
	symbol := any((*metalAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Generate_Good", "Adapter_Generate")
}

func TestAX7_Adapter_Generate_Bad(t *core.T) {
	symbol := any((*metalAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Generate_Bad", "Adapter_Generate")
}

func TestAX7_Adapter_Generate_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Generate_Ugly", "Adapter_Generate")
}

func TestAX7_Adapter_Info_Good(t *core.T) {
	symbol := any((*metalAdapter).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Info_Good", "Adapter_Info")
}

func TestAX7_Adapter_Info_Bad(t *core.T) {
	symbol := any((*metalAdapter).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Info_Bad", "Adapter_Info")
}

func TestAX7_Adapter_Info_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Info_Ugly", "Adapter_Info")
}

func TestAX7_Adapter_InspectAttention_Good(t *core.T) {
	symbol := any((*metalAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InspectAttention_Good", "Adapter_InspectAttention")
}

func TestAX7_Adapter_InspectAttention_Bad(t *core.T) {
	symbol := any((*metalAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InspectAttention_Bad", "Adapter_InspectAttention")
}

func TestAX7_Adapter_InspectAttention_Ugly(t *core.T) {
	symbol := any((*metalAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InspectAttention_Ugly", "Adapter_InspectAttention")
}

func TestAX7_Adapter_InternalModel_Good(t *core.T) {
	symbol := any((*metalAdapter).InternalModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InternalModel_Good", "Adapter_InternalModel")
}

func TestAX7_Adapter_InternalModel_Bad(t *core.T) {
	symbol := any((*metalAdapter).InternalModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InternalModel_Bad", "Adapter_InternalModel")
}

func TestAX7_Adapter_InternalModel_Ugly(t *core.T) {
	symbol := any((*metalAdapter).InternalModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_InternalModel_Ugly", "Adapter_InternalModel")
}

func TestAX7_Adapter_Metrics_Good(t *core.T) {
	symbol := any((*metalAdapter).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Metrics_Good", "Adapter_Metrics")
}

func TestAX7_Adapter_Metrics_Bad(t *core.T) {
	symbol := any((*metalAdapter).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Metrics_Bad", "Adapter_Metrics")
}

func TestAX7_Adapter_Metrics_Ugly(t *core.T) {
	symbol := any((*metalAdapter).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_Metrics_Ugly", "Adapter_Metrics")
}

func TestAX7_Adapter_ModelType_Good(t *core.T) {
	symbol := any((*metalAdapter).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ModelType_Good", "Adapter_ModelType")
}

func TestAX7_Adapter_ModelType_Bad(t *core.T) {
	symbol := any((*metalAdapter).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ModelType_Bad", "Adapter_ModelType")
}

func TestAX7_Adapter_ModelType_Ugly(t *core.T) {
	symbol := any((*metalAdapter).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_ModelType_Ugly", "Adapter_ModelType")
}

func TestAX7_Adapter_NumLayers_Good(t *core.T) {
	symbol := any((*metalAdapter).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_NumLayers_Good", "Adapter_NumLayers")
}

func TestAX7_Adapter_NumLayers_Bad(t *core.T) {
	symbol := any((*metalAdapter).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_NumLayers_Bad", "Adapter_NumLayers")
}

func TestAX7_Adapter_NumLayers_Ugly(t *core.T) {
	symbol := any((*metalAdapter).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Adapter_NumLayers_Ugly", "Adapter_NumLayers")
}

func TestAX7_Add_Good(t *core.T) {
	symbol := any(Add)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Add_Good", "Add")
}

func TestAX7_Add_Bad(t *core.T) {
	symbol := any(Add)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Add_Bad", "Add")
}

func TestAX7_Add_Ugly(t *core.T) {
	symbol := any(Add)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Add_Ugly", "Add")
}

func TestAX7_Array_Bool_Good(t *core.T) {
	symbol := any((*Array).Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Good", "Array_Bool")
}

func TestAX7_Array_Bool_Bad(t *core.T) {
	symbol := any((*Array).Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Bad", "Array_Bool")
}

func TestAX7_Array_Bool_Ugly(t *core.T) {
	symbol := any((*Array).Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Ugly", "Array_Bool")
}

func TestAX7_Array_Clone_Good(t *core.T) {
	symbol := any((*Array).Clone)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Clone_Good", "Array_Clone")
}

func TestAX7_Array_Clone_Bad(t *core.T) {
	symbol := any((*Array).Clone)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Clone_Bad", "Array_Clone")
}

func TestAX7_Array_Clone_Ugly(t *core.T) {
	symbol := any((*Array).Clone)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Clone_Ugly", "Array_Clone")
}

func TestAX7_Array_DataInt32_Good(t *core.T) {
	symbol := any((*Array).DataInt32)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_DataInt32_Good", "Array_DataInt32")
}

func TestAX7_Array_DataInt32_Bad(t *core.T) {
	symbol := any((*Array).DataInt32)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_DataInt32_Bad", "Array_DataInt32")
}

func TestAX7_Array_DataInt32_Ugly(t *core.T) {
	symbol := any((*Array).DataInt32)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_DataInt32_Ugly", "Array_DataInt32")
}

func TestAX7_Array_Dim_Good(t *core.T) {
	symbol := any((*Array).Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Good", "Array_Dim")
}

func TestAX7_Array_Dim_Bad(t *core.T) {
	symbol := any((*Array).Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Bad", "Array_Dim")
}

func TestAX7_Array_Dim_Ugly(t *core.T) {
	symbol := any((*Array).Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Ugly", "Array_Dim")
}

func TestAX7_Array_Dims_Good(t *core.T) {
	symbol := any((*Array).Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Good", "Array_Dims")
}

func TestAX7_Array_Dims_Bad(t *core.T) {
	symbol := any((*Array).Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Bad", "Array_Dims")
}

func TestAX7_Array_Dims_Ugly(t *core.T) {
	symbol := any((*Array).Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Ugly", "Array_Dims")
}

func TestAX7_Array_Dtype_Good(t *core.T) {
	symbol := any((*Array).Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Good", "Array_Dtype")
}

func TestAX7_Array_Dtype_Bad(t *core.T) {
	symbol := any((*Array).Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Bad", "Array_Dtype")
}

func TestAX7_Array_Dtype_Ugly(t *core.T) {
	symbol := any((*Array).Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Ugly", "Array_Dtype")
}

func TestAX7_Array_Float_Good(t *core.T) {
	symbol := any((*Array).Float)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Float_Good", "Array_Float")
}

func TestAX7_Array_Float_Bad(t *core.T) {
	symbol := any((*Array).Float)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Float_Bad", "Array_Float")
}

func TestAX7_Array_Float_Ugly(t *core.T) {
	symbol := any((*Array).Float)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Float_Ugly", "Array_Float")
}

func TestAX7_Array_Floats_Good(t *core.T) {
	symbol := any((*Array).Floats)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Floats_Good", "Array_Floats")
}

func TestAX7_Array_Floats_Bad(t *core.T) {
	symbol := any((*Array).Floats)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Floats_Bad", "Array_Floats")
}

func TestAX7_Array_Floats_Ugly(t *core.T) {
	symbol := any((*Array).Floats)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Floats_Ugly", "Array_Floats")
}

func TestAX7_Array_Int_Good(t *core.T) {
	symbol := any((*Array).Int)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Int_Good", "Array_Int")
}

func TestAX7_Array_Int_Bad(t *core.T) {
	symbol := any((*Array).Int)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Int_Bad", "Array_Int")
}

func TestAX7_Array_Int_Ugly(t *core.T) {
	symbol := any((*Array).Int)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Int_Ugly", "Array_Int")
}

func TestAX7_Array_Ints_Good(t *core.T) {
	symbol := any((*Array).Ints)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Ints_Good", "Array_Ints")
}

func TestAX7_Array_Ints_Bad(t *core.T) {
	symbol := any((*Array).Ints)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Ints_Bad", "Array_Ints")
}

func TestAX7_Array_Ints_Ugly(t *core.T) {
	symbol := any((*Array).Ints)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Ints_Ugly", "Array_Ints")
}

func TestAX7_Array_Iter_Good(t *core.T) {
	symbol := any((*Array).Iter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Iter_Good", "Array_Iter")
}

func TestAX7_Array_Iter_Bad(t *core.T) {
	symbol := any((*Array).Iter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Iter_Bad", "Array_Iter")
}

func TestAX7_Array_Iter_Ugly(t *core.T) {
	symbol := any((*Array).Iter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Iter_Ugly", "Array_Iter")
}

func TestAX7_Array_NumDims_Good(t *core.T) {
	symbol := any((*Array).NumDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumDims_Good", "Array_NumDims")
}

func TestAX7_Array_NumDims_Bad(t *core.T) {
	symbol := any((*Array).NumDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumDims_Bad", "Array_NumDims")
}

func TestAX7_Array_NumDims_Ugly(t *core.T) {
	symbol := any((*Array).NumDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumDims_Ugly", "Array_NumDims")
}

func TestAX7_Array_Set_Good(t *core.T) {
	symbol := any((*Array).Set)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Set_Good", "Array_Set")
}

func TestAX7_Array_Set_Bad(t *core.T) {
	symbol := any((*Array).Set)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Set_Bad", "Array_Set")
}

func TestAX7_Array_Set_Ugly(t *core.T) {
	symbol := any((*Array).Set)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Set_Ugly", "Array_Set")
}

func TestAX7_Array_SetFloat64_Good(t *core.T) {
	symbol := any((*Array).SetFloat64)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_SetFloat64_Good", "Array_SetFloat64")
}

func TestAX7_Array_SetFloat64_Bad(t *core.T) {
	symbol := any((*Array).SetFloat64)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_SetFloat64_Bad", "Array_SetFloat64")
}

func TestAX7_Array_SetFloat64_Ugly(t *core.T) {
	symbol := any((*Array).SetFloat64)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_SetFloat64_Ugly", "Array_SetFloat64")
}

func TestAX7_Array_Shape_Good(t *core.T) {
	symbol := any((*Array).Shape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Shape_Good", "Array_Shape")
}

func TestAX7_Array_Shape_Bad(t *core.T) {
	symbol := any((*Array).Shape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Shape_Bad", "Array_Shape")
}

func TestAX7_Array_Shape_Ugly(t *core.T) {
	symbol := any((*Array).Shape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Shape_Ugly", "Array_Shape")
}

func TestAX7_Array_String_Good(t *core.T) {
	symbol := any((*Array).String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_String_Good", "Array_String")
}

func TestAX7_Array_String_Bad(t *core.T) {
	symbol := any((*Array).String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_String_Bad", "Array_String")
}

func TestAX7_Array_String_Ugly(t *core.T) {
	symbol := any((*Array).String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_String_Ugly", "Array_String")
}

func TestAX7_Array_Valid_Good(t *core.T) {
	symbol := any((*Array).Valid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Valid_Good", "Array_Valid")
}

func TestAX7_Array_Valid_Bad(t *core.T) {
	symbol := any((*Array).Valid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Valid_Bad", "Array_Valid")
}

func TestAX7_AttentionSnapshot_HasQueries_Good(t *core.T) {
	symbol := any((*AttentionSnapshot).HasQueries)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AttentionSnapshot_HasQueries_Good", "AttentionSnapshot_HasQueries")
}

func TestAX7_AttentionSnapshot_HasQueries_Bad(t *core.T) {
	symbol := any((*AttentionSnapshot).HasQueries)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AttentionSnapshot_HasQueries_Bad", "AttentionSnapshot_HasQueries")
}

func TestAX7_AttentionSnapshot_HasQueries_Ugly(t *core.T) {
	symbol := any((*AttentionSnapshot).HasQueries)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AttentionSnapshot_HasQueries_Ugly", "AttentionSnapshot_HasQueries")
}

func TestAX7_Available_Good(t *core.T) {
	symbol := any(Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Available_Good", "Available")
}

func TestAX7_Available_Bad(t *core.T) {
	symbol := any(Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Available_Bad", "Available")
}

func TestAX7_Available_Ugly(t *core.T) {
	symbol := any(Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Available_Ugly", "Available")
}

func TestAX7_Backend_Available_Good(t *core.T) {
	symbol := any(computeBackend.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Good", "Backend_Available")
}

func TestAX7_Backend_Available_Bad(t *core.T) {
	symbol := any(computeBackend.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Bad", "Backend_Available")
}

func TestAX7_Backend_Available_Ugly(t *core.T) {
	symbol := any(computeBackend.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Ugly", "Backend_Available")
}

func TestAX7_Backend_DeviceInfo_Good(t *core.T) {
	symbol := any(computeBackend.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_DeviceInfo_Good", "Backend_DeviceInfo")
}

func TestAX7_Backend_DeviceInfo_Bad(t *core.T) {
	symbol := any(computeBackend.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_DeviceInfo_Bad", "Backend_DeviceInfo")
}

func TestAX7_Backend_DeviceInfo_Ugly(t *core.T) {
	symbol := any(computeBackend.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_DeviceInfo_Ugly", "Backend_DeviceInfo")
}

func TestAX7_Backend_LoadModel_Good(t *core.T) {
	symbol := any((*metalBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Good", "Backend_LoadModel")
}

func TestAX7_Backend_LoadModel_Bad(t *core.T) {
	symbol := any((*metalBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Bad", "Backend_LoadModel")
}

func TestAX7_Backend_LoadModel_Ugly(t *core.T) {
	symbol := any((*metalBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Ugly", "Backend_LoadModel")
}

func TestAX7_Backend_Name_Good(t *core.T) {
	symbol := any((*metalBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Good", "Backend_Name")
}

func TestAX7_Backend_Name_Bad(t *core.T) {
	symbol := any((*metalBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Bad", "Backend_Name")
}

func TestAX7_Backend_Name_Ugly(t *core.T) {
	symbol := any((*metalBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Ugly", "Backend_Name")
}

func TestAX7_Backend_NewSession_Good(t *core.T) {
	symbol := any(computeBackend.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_NewSession_Good", "Backend_NewSession")
}

func TestAX7_Backend_NewSession_Bad(t *core.T) {
	symbol := any(computeBackend.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_NewSession_Bad", "Backend_NewSession")
}

func TestAX7_Backend_NewSession_Ugly(t *core.T) {
	symbol := any(computeBackend.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_NewSession_Ugly", "Backend_NewSession")
}

func TestAX7_Base_Size_Good(t *core.T) {
	symbol := any((*bufferBase).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Base_Size_Good", "Base_Size")
}

func TestAX7_Base_Size_Bad(t *core.T) {
	symbol := any((*bufferBase).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Base_Size_Bad", "Base_Size")
}

func TestAX7_Base_Size_Ugly(t *core.T) {
	symbol := any((*bufferBase).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Base_Size_Ugly", "Base_Size")
}

func TestAX7_Buffer_Descriptor_Good(t *core.T) {
	symbol := any((*pixelBuffer).Descriptor)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Descriptor_Good", "Buffer_Descriptor")
}

func TestAX7_Buffer_Descriptor_Bad(t *core.T) {
	symbol := any((*pixelBuffer).Descriptor)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Descriptor_Bad", "Buffer_Descriptor")
}

func TestAX7_Buffer_Descriptor_Ugly(t *core.T) {
	symbol := any((*pixelBuffer).Descriptor)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Descriptor_Ugly", "Buffer_Descriptor")
}

func TestAX7_Buffer_Read_Good(t *core.T) {
	symbol := any((*pixelBuffer).Read)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Read_Good", "Buffer_Read")
}

func TestAX7_Buffer_Read_Bad(t *core.T) {
	symbol := any((*pixelBuffer).Read)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Read_Bad", "Buffer_Read")
}

func TestAX7_Buffer_Read_Ugly(t *core.T) {
	symbol := any((*pixelBuffer).Read)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Read_Ugly", "Buffer_Read")
}

func TestAX7_Buffer_Upload_Good(t *core.T) {
	symbol := any((*pixelBuffer).Upload)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Upload_Good", "Buffer_Upload")
}

func TestAX7_Buffer_Upload_Bad(t *core.T) {
	symbol := any((*pixelBuffer).Upload)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Upload_Bad", "Buffer_Upload")
}

func TestAX7_Buffer_Upload_Ugly(t *core.T) {
	symbol := any((*pixelBuffer).Upload)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Buffer_Upload_Ugly", "Buffer_Upload")
}

func TestAX7_Checkpoint_Good(t *core.T) {
	symbol := any(Checkpoint)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Checkpoint_Good", "Checkpoint")
}

func TestAX7_Checkpoint_Bad(t *core.T) {
	symbol := any(Checkpoint)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Checkpoint_Bad", "Checkpoint")
}

func TestAX7_Checkpoint_Ugly(t *core.T) {
	symbol := any(Checkpoint)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Checkpoint_Ugly", "Checkpoint")
}

func TestAX7_ClearCache_Good(t *core.T) {
	symbol := any(ClearCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClearCache_Good", "ClearCache")
}

func TestAX7_ClearCache_Bad(t *core.T) {
	symbol := any(ClearCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClearCache_Bad", "ClearCache")
}

func TestAX7_ClearCache_Ugly(t *core.T) {
	symbol := any(ClearCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClearCache_Ugly", "ClearCache")
}

func TestAX7_ComputeError_Error_Good(t *core.T) {
	symbol := any((*ComputeError).Error)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Error_Good", "ComputeError_Error")
}

func TestAX7_ComputeError_Error_Bad(t *core.T) {
	symbol := any((*ComputeError).Error)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Error_Bad", "ComputeError_Error")
}

func TestAX7_ComputeError_Error_Ugly(t *core.T) {
	symbol := any((*ComputeError).Error)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Error_Ugly", "ComputeError_Error")
}

func TestAX7_ComputeError_Is_Good(t *core.T) {
	symbol := any((*ComputeError).Is)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Is_Good", "ComputeError_Is")
}

func TestAX7_ComputeError_Is_Bad(t *core.T) {
	symbol := any((*ComputeError).Is)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Is_Bad", "ComputeError_Is")
}

func TestAX7_ComputeError_Is_Ugly(t *core.T) {
	symbol := any((*ComputeError).Is)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Is_Ugly", "ComputeError_Is")
}

func TestAX7_ComputeError_Unwrap_Good(t *core.T) {
	symbol := any((*ComputeError).Unwrap)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Unwrap_Good", "ComputeError_Unwrap")
}

func TestAX7_ComputeError_Unwrap_Bad(t *core.T) {
	symbol := any((*ComputeError).Unwrap)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Unwrap_Bad", "ComputeError_Unwrap")
}

func TestAX7_ComputeError_Unwrap_Ugly(t *core.T) {
	symbol := any((*ComputeError).Unwrap)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ComputeError_Unwrap_Ugly", "ComputeError_Unwrap")
}

func TestAX7_Compute_Available_Good(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_Available_Good", "Compute_Available")
}

func TestAX7_Compute_Available_Bad(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_Available_Bad", "Compute_Available")
}

func TestAX7_Compute_Available_Ugly(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_Available_Ugly", "Compute_Available")
}

func TestAX7_Compute_DeviceInfo_Good(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_DeviceInfo_Good", "Compute_DeviceInfo")
}

func TestAX7_Compute_DeviceInfo_Bad(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_DeviceInfo_Bad", "Compute_DeviceInfo")
}

func TestAX7_Compute_DeviceInfo_Ugly(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.DeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_DeviceInfo_Ugly", "Compute_DeviceInfo")
}

func TestAX7_Compute_NewSession_Good(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_NewSession_Good", "Compute_NewSession")
}

func TestAX7_Compute_NewSession_Bad(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_NewSession_Bad", "Compute_NewSession")
}

func TestAX7_Compute_NewSession_Ugly(t *core.T) {
	compute := DefaultCompute()
	symbol := any(compute.NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Compute_NewSession_Ugly", "Compute_NewSession")
}

func TestAX7_ConcreteAdapter_Good(t *core.T) {
	symbol := any(ConcreteAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ConcreteAdapter_Good", "ConcreteAdapter")
}

func TestAX7_ConcreteAdapter_Bad(t *core.T) {
	symbol := any(ConcreteAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ConcreteAdapter_Bad", "ConcreteAdapter")
}

func TestAX7_ConcreteAdapter_Ugly(t *core.T) {
	symbol := any(ConcreteAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ConcreteAdapter_Ugly", "ConcreteAdapter")
}

func TestAX7_CrossEntropyLoss_Good(t *core.T) {
	symbol := any(CrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CrossEntropyLoss_Good", "CrossEntropyLoss")
}

func TestAX7_CrossEntropyLoss_Bad(t *core.T) {
	symbol := any(CrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CrossEntropyLoss_Bad", "CrossEntropyLoss")
}

func TestAX7_CrossEntropyLoss_Ugly(t *core.T) {
	symbol := any(CrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CrossEntropyLoss_Ugly", "CrossEntropyLoss")
}

func TestAX7_DType_String_Good(t *core.T) {
	symbol := any(DType.String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_String_Good", "DType_String")
}

func TestAX7_DType_String_Bad(t *core.T) {
	symbol := any(DType.String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_String_Bad", "DType_String")
}

func TestAX7_DType_String_Ugly(t *core.T) {
	symbol := any(DType.String)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_String_Ugly", "DType_String")
}

func TestAX7_DefaultCompute_Good(t *core.T) {
	symbol := any(DefaultCompute)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCompute_Good", "DefaultCompute")
}

func TestAX7_DefaultCompute_Bad(t *core.T) {
	symbol := any(DefaultCompute)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCompute_Bad", "DefaultCompute")
}

func TestAX7_DefaultCompute_Ugly(t *core.T) {
	symbol := any(DefaultCompute)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCompute_Ugly", "DefaultCompute")
}

func TestAX7_DefaultGenerateConfig_Good(t *core.T) {
	symbol := any(DefaultGenerateConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGenerateConfig_Good", "DefaultGenerateConfig")
}

func TestAX7_DefaultGenerateConfig_Bad(t *core.T) {
	symbol := any(DefaultGenerateConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGenerateConfig_Bad", "DefaultGenerateConfig")
}

func TestAX7_DefaultGenerateConfig_Ugly(t *core.T) {
	symbol := any(DefaultGenerateConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGenerateConfig_Ugly", "DefaultGenerateConfig")
}

func TestAX7_DefaultLoadConfig_Good(t *core.T) {
	symbol := any(DefaultLoadConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultLoadConfig_Good", "DefaultLoadConfig")
}

func TestAX7_DefaultLoadConfig_Bad(t *core.T) {
	symbol := any(DefaultLoadConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultLoadConfig_Bad", "DefaultLoadConfig")
}

func TestAX7_DefaultLoadConfig_Ugly(t *core.T) {
	symbol := any(DefaultLoadConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultLoadConfig_Ugly", "DefaultLoadConfig")
}

func TestAX7_DiscoverModels_Good(t *core.T) {
	symbol := any(DiscoverModels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DiscoverModels_Good", "DiscoverModels")
}

func TestAX7_DiscoverModels_Bad(t *core.T) {
	symbol := any(DiscoverModels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DiscoverModels_Bad", "DiscoverModels")
}

func TestAX7_DiscoverModels_Ugly(t *core.T) {
	symbol := any(DiscoverModels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DiscoverModels_Ugly", "DiscoverModels")
}

func TestAX7_Free_Good(t *core.T) {
	symbol := any(Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Free_Good", "Free")
}

func TestAX7_Free_Bad(t *core.T) {
	symbol := any(Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Free_Bad", "Free")
}

func TestAX7_Free_Ugly(t *core.T) {
	symbol := any(Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Free_Ugly", "Free")
}

func TestAX7_FromValues_Good(t *core.T) {
	symbol := any(FromValues[[]float32, float32])
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FromValues_Good", "FromValues")
}

func TestAX7_FromValues_Bad(t *core.T) {
	symbol := any(FromValues[[]float32, float32])
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FromValues_Bad", "FromValues")
}

func TestAX7_FromValues_Ugly(t *core.T) {
	symbol := any(FromValues[[]float32, float32])
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FromValues_Ugly", "FromValues")
}

func TestAX7_GC_Good(t *core.T) {
	symbol := any(GC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GC_Good", "GC")
}

func TestAX7_GC_Bad(t *core.T) {
	symbol := any(GC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GC_Bad", "GC")
}

func TestAX7_GC_Ugly(t *core.T) {
	symbol := any(GC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GC_Ugly", "GC")
}

func TestAX7_GetActiveMemory_Good(t *core.T) {
	symbol := any(GetActiveMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetActiveMemory_Good", "GetActiveMemory")
}

func TestAX7_GetActiveMemory_Bad(t *core.T) {
	symbol := any(GetActiveMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetActiveMemory_Bad", "GetActiveMemory")
}

func TestAX7_GetActiveMemory_Ugly(t *core.T) {
	symbol := any(GetActiveMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetActiveMemory_Ugly", "GetActiveMemory")
}

func TestAX7_GetCacheMemory_Good(t *core.T) {
	symbol := any(GetCacheMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetCacheMemory_Good", "GetCacheMemory")
}

func TestAX7_GetCacheMemory_Bad(t *core.T) {
	symbol := any(GetCacheMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetCacheMemory_Bad", "GetCacheMemory")
}

func TestAX7_GetCacheMemory_Ugly(t *core.T) {
	symbol := any(GetCacheMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetCacheMemory_Ugly", "GetCacheMemory")
}

func TestAX7_GetDeviceInfo_Good(t *core.T) {
	symbol := any(GetDeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetDeviceInfo_Good", "GetDeviceInfo")
}

func TestAX7_GetDeviceInfo_Bad(t *core.T) {
	symbol := any(GetDeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetDeviceInfo_Bad", "GetDeviceInfo")
}

func TestAX7_GetDeviceInfo_Ugly(t *core.T) {
	symbol := any(GetDeviceInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetDeviceInfo_Ugly", "GetDeviceInfo")
}

func TestAX7_GetPeakMemory_Good(t *core.T) {
	symbol := any(GetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetPeakMemory_Good", "GetPeakMemory")
}

func TestAX7_GetPeakMemory_Bad(t *core.T) {
	symbol := any(GetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetPeakMemory_Bad", "GetPeakMemory")
}

func TestAX7_GetPeakMemory_Ugly(t *core.T) {
	symbol := any(GetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GetPeakMemory_Ugly", "GetPeakMemory")
}

func TestAX7_GradFn_Apply_Good(t *core.T) {
	symbol := any((*GradFn).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Apply_Good", "GradFn_Apply")
}

func TestAX7_GradFn_Apply_Bad(t *core.T) {
	symbol := any((*GradFn).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Apply_Bad", "GradFn_Apply")
}

func TestAX7_GradFn_Apply_Ugly(t *core.T) {
	symbol := any((*GradFn).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Apply_Ugly", "GradFn_Apply")
}

func TestAX7_GradFn_Free_Good(t *core.T) {
	symbol := any((*GradFn).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Free_Good", "GradFn_Free")
}

func TestAX7_GradFn_Free_Bad(t *core.T) {
	symbol := any((*GradFn).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Free_Bad", "GradFn_Free")
}

func TestAX7_GradFn_Free_Ugly(t *core.T) {
	symbol := any((*GradFn).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GradFn_Free_Ugly", "GradFn_Free")
}

func TestAX7_InferenceAdapter_Available_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Available_Good", "InferenceAdapter_Available")
}

func TestAX7_InferenceAdapter_Available_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Available_Bad", "InferenceAdapter_Available")
}

func TestAX7_InferenceAdapter_Available_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Available_Ugly", "InferenceAdapter_Available")
}

func TestAX7_InferenceAdapter_Chat_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Chat_Good", "InferenceAdapter_Chat")
}

func TestAX7_InferenceAdapter_Chat_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Chat_Bad", "InferenceAdapter_Chat")
}

func TestAX7_InferenceAdapter_Chat_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Chat_Ugly", "InferenceAdapter_Chat")
}

func TestAX7_InferenceAdapter_ChatStream_Good(t *core.T) {
	symbol := any((*InferenceAdapter).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_ChatStream_Good", "InferenceAdapter_ChatStream")
}

func TestAX7_InferenceAdapter_ChatStream_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_ChatStream_Bad", "InferenceAdapter_ChatStream")
}

func TestAX7_InferenceAdapter_ChatStream_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_ChatStream_Ugly", "InferenceAdapter_ChatStream")
}

func TestAX7_InferenceAdapter_Close_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Close_Good", "InferenceAdapter_Close")
}

func TestAX7_InferenceAdapter_Close_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Close_Bad", "InferenceAdapter_Close")
}

func TestAX7_InferenceAdapter_Close_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Close_Ugly", "InferenceAdapter_Close")
}

func TestAX7_InferenceAdapter_Generate_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Generate_Good", "InferenceAdapter_Generate")
}

func TestAX7_InferenceAdapter_Generate_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Generate_Bad", "InferenceAdapter_Generate")
}

func TestAX7_InferenceAdapter_Generate_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Generate_Ugly", "InferenceAdapter_Generate")
}

func TestAX7_InferenceAdapter_GenerateStream_Good(t *core.T) {
	symbol := any((*InferenceAdapter).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_GenerateStream_Good", "InferenceAdapter_GenerateStream")
}

func TestAX7_InferenceAdapter_GenerateStream_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_GenerateStream_Bad", "InferenceAdapter_GenerateStream")
}

func TestAX7_InferenceAdapter_GenerateStream_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_GenerateStream_Ugly", "InferenceAdapter_GenerateStream")
}

func TestAX7_InferenceAdapter_InspectAttention_Good(t *core.T) {
	symbol := any((*InferenceAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_InspectAttention_Good", "InferenceAdapter_InspectAttention")
}

func TestAX7_InferenceAdapter_InspectAttention_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_InspectAttention_Bad", "InferenceAdapter_InspectAttention")
}

func TestAX7_InferenceAdapter_InspectAttention_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_InspectAttention_Ugly", "InferenceAdapter_InspectAttention")
}

func TestAX7_InferenceAdapter_Model_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Model)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Model_Good", "InferenceAdapter_Model")
}

func TestAX7_InferenceAdapter_Model_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Model)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Model_Bad", "InferenceAdapter_Model")
}

func TestAX7_InferenceAdapter_Model_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Model)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Model_Ugly", "InferenceAdapter_Model")
}

func TestAX7_InferenceAdapter_Name_Good(t *core.T) {
	symbol := any((*InferenceAdapter).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Name_Good", "InferenceAdapter_Name")
}

func TestAX7_InferenceAdapter_Name_Bad(t *core.T) {
	symbol := any((*InferenceAdapter).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Name_Bad", "InferenceAdapter_Name")
}

func TestAX7_InferenceAdapter_Name_Ugly(t *core.T) {
	symbol := any((*InferenceAdapter).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InferenceAdapter_Name_Ugly", "InferenceAdapter_Name")
}

func TestAX7_JVP_Good(t *core.T) {
	symbol := any(JVP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "JVP_Good", "JVP")
}

func TestAX7_JVP_Bad(t *core.T) {
	symbol := any(JVP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "JVP_Bad", "JVP")
}

func TestAX7_JVP_Ugly(t *core.T) {
	symbol := any(JVP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "JVP_Ugly", "JVP")
}

func TestAX7_LoRAAdapter_AllTrainableParams_Good(t *core.T) {
	symbol := any((*LoRAAdapter).AllTrainableParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_AllTrainableParams_Good", "LoRAAdapter_AllTrainableParams")
}

func TestAX7_LoRAAdapter_AllTrainableParams_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).AllTrainableParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_AllTrainableParams_Bad", "LoRAAdapter_AllTrainableParams")
}

func TestAX7_LoRAAdapter_AllTrainableParams_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).AllTrainableParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_AllTrainableParams_Ugly", "LoRAAdapter_AllTrainableParams")
}

func TestAX7_LoRAAdapter_Merge_Good(t *core.T) {
	symbol := any((*LoRAAdapter).Merge)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Merge_Good", "LoRAAdapter_Merge")
}

func TestAX7_LoRAAdapter_Merge_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).Merge)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Merge_Bad", "LoRAAdapter_Merge")
}

func TestAX7_LoRAAdapter_Merge_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).Merge)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Merge_Ugly", "LoRAAdapter_Merge")
}

func TestAX7_LoRAAdapter_Save_Good(t *core.T) {
	symbol := any((*LoRAAdapter).Save)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Save_Good", "LoRAAdapter_Save")
}

func TestAX7_LoRAAdapter_Save_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).Save)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Save_Bad", "LoRAAdapter_Save")
}

func TestAX7_LoRAAdapter_Save_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).Save)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Save_Ugly", "LoRAAdapter_Save")
}

func TestAX7_LoRAAdapter_SetAllParams_Good(t *core.T) {
	symbol := any((*LoRAAdapter).SetAllParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SetAllParams_Good", "LoRAAdapter_SetAllParams")
}

func TestAX7_LoRAAdapter_SetAllParams_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).SetAllParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SetAllParams_Bad", "LoRAAdapter_SetAllParams")
}

func TestAX7_LoRAAdapter_SetAllParams_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).SetAllParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SetAllParams_Ugly", "LoRAAdapter_SetAllParams")
}

func TestAX7_LoRAAdapter_SortedNames_Good(t *core.T) {
	symbol := any((*LoRAAdapter).SortedNames)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SortedNames_Good", "LoRAAdapter_SortedNames")
}

func TestAX7_LoRAAdapter_SortedNames_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).SortedNames)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SortedNames_Bad", "LoRAAdapter_SortedNames")
}

func TestAX7_LoRAAdapter_SortedNames_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).SortedNames)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_SortedNames_Ugly", "LoRAAdapter_SortedNames")
}

func TestAX7_LoRAAdapter_Step_Good(t *core.T) {
	symbol := any((*LoRAAdapter).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Step_Good", "LoRAAdapter_Step")
}

func TestAX7_LoRAAdapter_Step_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Step_Bad", "LoRAAdapter_Step")
}

func TestAX7_LoRAAdapter_Step_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).Step)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_Step_Ugly", "LoRAAdapter_Step")
}

func TestAX7_LoRAAdapter_TotalParams_Good(t *core.T) {
	symbol := any((*LoRAAdapter).TotalParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_TotalParams_Good", "LoRAAdapter_TotalParams")
}

func TestAX7_LoRAAdapter_TotalParams_Bad(t *core.T) {
	symbol := any((*LoRAAdapter).TotalParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_TotalParams_Bad", "LoRAAdapter_TotalParams")
}

func TestAX7_LoRAAdapter_TotalParams_Ugly(t *core.T) {
	symbol := any((*LoRAAdapter).TotalParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRAAdapter_TotalParams_Ugly", "LoRAAdapter_TotalParams")
}

func TestAX7_LoadModel_Good(t *core.T) {
	symbol := any(LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModel_Good", "LoadModel")
}

func TestAX7_LoadModel_Bad(t *core.T) {
	symbol := any(LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModel_Bad", "LoadModel")
}

func TestAX7_LoadModel_Ugly(t *core.T) {
	symbol := any(LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModel_Ugly", "LoadModel")
}

func TestAX7_LoadModelFromMedium_Good(t *core.T) {
	symbol := any(LoadModelFromMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModelFromMedium_Good", "LoadModelFromMedium")
}

func TestAX7_LoadModelFromMedium_Bad(t *core.T) {
	symbol := any(LoadModelFromMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModelFromMedium_Bad", "LoadModelFromMedium")
}

func TestAX7_LoadModelFromMedium_Ugly(t *core.T) {
	symbol := any(LoadModelFromMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadModelFromMedium_Ugly", "LoadModelFromMedium")
}

func TestAX7_LoadTokenizer_Good(t *core.T) {
	symbol := any(LoadTokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadTokenizer_Good", "LoadTokenizer")
}

func TestAX7_LoadTokenizer_Bad(t *core.T) {
	symbol := any(LoadTokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadTokenizer_Bad", "LoadTokenizer")
}

func TestAX7_LoadTokenizer_Ugly(t *core.T) {
	symbol := any(LoadTokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadTokenizer_Ugly", "LoadTokenizer")
}

func TestAX7_MaskedCrossEntropyLoss_Good(t *core.T) {
	symbol := any(MaskedCrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaskedCrossEntropyLoss_Good", "MaskedCrossEntropyLoss")
}

func TestAX7_MaskedCrossEntropyLoss_Bad(t *core.T) {
	symbol := any(MaskedCrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaskedCrossEntropyLoss_Bad", "MaskedCrossEntropyLoss")
}

func TestAX7_MaskedCrossEntropyLoss_Ugly(t *core.T) {
	symbol := any(MaskedCrossEntropyLoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaskedCrossEntropyLoss_Ugly", "MaskedCrossEntropyLoss")
}

func TestAX7_MatMul_Good(t *core.T) {
	symbol := any(MatMul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MatMul_Good", "MatMul")
}

func TestAX7_MatMul_Bad(t *core.T) {
	symbol := any(MatMul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MatMul_Bad", "MatMul")
}

func TestAX7_MatMul_Ugly(t *core.T) {
	symbol := any(MatMul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MatMul_Ugly", "MatMul")
}

func TestAX7_Materialize_Good(t *core.T) {
	symbol := any(Materialize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Materialize_Good", "Materialize")
}

func TestAX7_Materialize_Bad(t *core.T) {
	symbol := any(Materialize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Materialize_Bad", "Materialize")
}

func TestAX7_Materialize_Ugly(t *core.T) {
	symbol := any(Materialize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Materialize_Ugly", "Materialize")
}

func TestAX7_MetalAvailable_Good(t *core.T) {
	symbol := any(MetalAvailable)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalAvailable_Good", "MetalAvailable")
}

func TestAX7_MetalAvailable_Bad(t *core.T) {
	symbol := any(MetalAvailable)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalAvailable_Bad", "MetalAvailable")
}

func TestAX7_MetalAvailable_Ugly(t *core.T) {
	symbol := any(MetalAvailable)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalAvailable_Ugly", "MetalAvailable")
}

func TestAX7_Model_BatchGenerate_Good(t *core.T) {
	symbol := any((*Model).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Good", "Model_BatchGenerate")
}

func TestAX7_Model_BatchGenerate_Bad(t *core.T) {
	symbol := any((*Model).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Bad", "Model_BatchGenerate")
}

func TestAX7_Model_BatchGenerate_Ugly(t *core.T) {
	symbol := any((*Model).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Ugly", "Model_BatchGenerate")
}

func TestAX7_Model_Chat_Good(t *core.T) {
	symbol := any((*Model).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Good", "Model_Chat")
}

func TestAX7_Model_Chat_Bad(t *core.T) {
	symbol := any((*Model).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Bad", "Model_Chat")
}

func TestAX7_Model_Chat_Ugly(t *core.T) {
	symbol := any((*Model).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Ugly", "Model_Chat")
}

func TestAX7_Model_ChatStream_Good(t *core.T) {
	symbol := any((*Model).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ChatStream_Good", "Model_ChatStream")
}

func TestAX7_Model_ChatStream_Bad(t *core.T) {
	symbol := any((*Model).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ChatStream_Bad", "Model_ChatStream")
}

func TestAX7_Model_ChatStream_Ugly(t *core.T) {
	symbol := any((*Model).ChatStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ChatStream_Ugly", "Model_ChatStream")
}

func TestAX7_Model_Classify_Good(t *core.T) {
	symbol := any((*Model).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Good", "Model_Classify")
}

func TestAX7_Model_Classify_Bad(t *core.T) {
	symbol := any((*Model).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Bad", "Model_Classify")
}

func TestAX7_Model_Classify_Ugly(t *core.T) {
	symbol := any((*Model).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Ugly", "Model_Classify")
}

func TestAX7_Model_Close_Good(t *core.T) {
	symbol := any((*Model).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Good", "Model_Close")
}

func TestAX7_Model_Close_Bad(t *core.T) {
	symbol := any((*Model).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Bad", "Model_Close")
}

func TestAX7_Model_Close_Ugly(t *core.T) {
	symbol := any((*Model).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Ugly", "Model_Close")
}

func TestAX7_Model_Err_Good(t *core.T) {
	symbol := any((*Model).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Good", "Model_Err")
}

func TestAX7_Model_Err_Bad(t *core.T) {
	symbol := any((*Model).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Bad", "Model_Err")
}

func TestAX7_Model_Err_Ugly(t *core.T) {
	symbol := any((*Model).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Ugly", "Model_Err")
}

func TestAX7_Model_Generate_Good(t *core.T) {
	symbol := any((*Model).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Good", "Model_Generate")
}

func TestAX7_Model_Generate_Bad(t *core.T) {
	symbol := any((*Model).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Bad", "Model_Generate")
}

func TestAX7_Model_Generate_Ugly(t *core.T) {
	symbol := any((*Model).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Ugly", "Model_Generate")
}

func TestAX7_Model_GenerateStream_Good(t *core.T) {
	symbol := any((*Model).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_GenerateStream_Good", "Model_GenerateStream")
}

func TestAX7_Model_GenerateStream_Bad(t *core.T) {
	symbol := any((*Model).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_GenerateStream_Bad", "Model_GenerateStream")
}

func TestAX7_Model_GenerateStream_Ugly(t *core.T) {
	symbol := any((*Model).GenerateStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_GenerateStream_Ugly", "Model_GenerateStream")
}

func TestAX7_Model_Info_Good(t *core.T) {
	symbol := any((*Model).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Good", "Model_Info")
}

func TestAX7_Model_Info_Bad(t *core.T) {
	symbol := any((*Model).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Bad", "Model_Info")
}

func TestAX7_Model_Info_Ugly(t *core.T) {
	symbol := any((*Model).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Ugly", "Model_Info")
}

func TestAX7_Model_InspectAttention_Good(t *core.T) {
	symbol := any((*Model).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Good", "Model_InspectAttention")
}

func TestAX7_Model_InspectAttention_Bad(t *core.T) {
	symbol := any((*Model).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Bad", "Model_InspectAttention")
}

func TestAX7_Model_InspectAttention_Ugly(t *core.T) {
	symbol := any((*Model).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Ugly", "Model_InspectAttention")
}

func TestAX7_Model_MergeLoRA_Good(t *core.T) {
	symbol := any((*Model).MergeLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_MergeLoRA_Good", "Model_MergeLoRA")
}

func TestAX7_Model_MergeLoRA_Bad(t *core.T) {
	symbol := any((*Model).MergeLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_MergeLoRA_Bad", "Model_MergeLoRA")
}

func TestAX7_Model_MergeLoRA_Ugly(t *core.T) {
	symbol := any((*Model).MergeLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_MergeLoRA_Ugly", "Model_MergeLoRA")
}

func TestAX7_Model_Metrics_Good(t *core.T) {
	symbol := any((*Model).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Good", "Model_Metrics")
}

func TestAX7_Model_Metrics_Bad(t *core.T) {
	symbol := any((*Model).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Bad", "Model_Metrics")
}

func TestAX7_Model_Metrics_Ugly(t *core.T) {
	symbol := any((*Model).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Ugly", "Model_Metrics")
}

func TestAX7_Model_ModelType_Good(t *core.T) {
	symbol := any((*Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Good", "Model_ModelType")
}

func TestAX7_Model_ModelType_Bad(t *core.T) {
	symbol := any((*Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Bad", "Model_ModelType")
}

func TestAX7_Model_ModelType_Ugly(t *core.T) {
	symbol := any((*Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Ugly", "Model_ModelType")
}

func TestAX7_Model_Tokenizer_Good(t *core.T) {
	symbol := any((*Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Tokenizer_Good", "Model_Tokenizer")
}

func TestAX7_Model_Tokenizer_Bad(t *core.T) {
	symbol := any((*Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Tokenizer_Bad", "Model_Tokenizer")
}

func TestAX7_Model_Tokenizer_Ugly(t *core.T) {
	symbol := any((*Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Tokenizer_Ugly", "Model_Tokenizer")
}

func TestAX7_Mul_Good(t *core.T) {
	symbol := any(Mul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Mul_Good", "Mul")
}

func TestAX7_Mul_Bad(t *core.T) {
	symbol := any(Mul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Mul_Bad", "Mul")
}

func TestAX7_Mul_Ugly(t *core.T) {
	symbol := any(Mul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Mul_Ugly", "Mul")
}

func TestAX7_NewAdamW_Good(t *core.T) {
	symbol := any(NewAdamW)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewAdamW_Good", "NewAdamW")
}

func TestAX7_NewAdamW_Bad(t *core.T) {
	symbol := any(NewAdamW)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewAdamW_Bad", "NewAdamW")
}

func TestAX7_NewAdamW_Ugly(t *core.T) {
	symbol := any(NewAdamW)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewAdamW_Ugly", "NewAdamW")
}

func TestAX7_NewInferenceAdapter_Good(t *core.T) {
	symbol := any(NewInferenceAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewInferenceAdapter_Good", "NewInferenceAdapter")
}

func TestAX7_NewInferenceAdapter_Bad(t *core.T) {
	symbol := any(NewInferenceAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewInferenceAdapter_Bad", "NewInferenceAdapter")
}

func TestAX7_NewInferenceAdapter_Ugly(t *core.T) {
	symbol := any(NewInferenceAdapter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewInferenceAdapter_Ugly", "NewInferenceAdapter")
}

func TestAX7_NewLoRA_Good(t *core.T) {
	symbol := any(NewLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLoRA_Good", "NewLoRA")
}

func TestAX7_NewLoRA_Bad(t *core.T) {
	symbol := any(NewLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLoRA_Bad", "NewLoRA")
}

func TestAX7_NewLoRA_Ugly(t *core.T) {
	symbol := any(NewLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLoRA_Ugly", "NewLoRA")
}

func TestAX7_NewMLXBackend_Good(t *core.T) {
	symbol := any(NewMLXBackend)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMLXBackend_Good", "NewMLXBackend")
}

func TestAX7_NewMLXBackend_Bad(t *core.T) {
	symbol := any(NewMLXBackend)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMLXBackend_Bad", "NewMLXBackend")
}

func TestAX7_NewMLXBackend_Ugly(t *core.T) {
	symbol := any(NewMLXBackend)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMLXBackend_Ugly", "NewMLXBackend")
}

func TestAX7_NewSession_Good(t *core.T) {
	symbol := any(NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSession_Good", "NewSession")
}

func TestAX7_NewSession_Bad(t *core.T) {
	symbol := any(NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSession_Bad", "NewSession")
}

func TestAX7_NewSession_Ugly(t *core.T) {
	symbol := any(NewSession)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSession_Ugly", "NewSession")
}

func TestAX7_PixelBufferDesc_SizeBytes_Good(t *core.T) {
	symbol := any(PixelBufferDesc.SizeBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_SizeBytes_Good", "PixelBufferDesc_SizeBytes")
}

func TestAX7_PixelBufferDesc_SizeBytes_Bad(t *core.T) {
	symbol := any(PixelBufferDesc.SizeBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_SizeBytes_Bad", "PixelBufferDesc_SizeBytes")
}

func TestAX7_PixelBufferDesc_SizeBytes_Ugly(t *core.T) {
	symbol := any(PixelBufferDesc.SizeBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_SizeBytes_Ugly", "PixelBufferDesc_SizeBytes")
}

func TestAX7_PixelBufferDesc_Validate_Good(t *core.T) {
	symbol := any(PixelBufferDesc.Validate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_Validate_Good", "PixelBufferDesc_Validate")
}

func TestAX7_PixelBufferDesc_Validate_Bad(t *core.T) {
	symbol := any(PixelBufferDesc.Validate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_Validate_Bad", "PixelBufferDesc_Validate")
}

func TestAX7_PixelBufferDesc_Validate_Ugly(t *core.T) {
	symbol := any(PixelBufferDesc.Validate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelBufferDesc_Validate_Ugly", "PixelBufferDesc_Validate")
}

func TestAX7_PixelFormat_BytesPerPixel_Good(t *core.T) {
	symbol := any(PixelFormat.BytesPerPixel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelFormat_BytesPerPixel_Good", "PixelFormat_BytesPerPixel")
}

func TestAX7_PixelFormat_BytesPerPixel_Bad(t *core.T) {
	symbol := any(PixelFormat.BytesPerPixel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelFormat_BytesPerPixel_Bad", "PixelFormat_BytesPerPixel")
}

func TestAX7_PixelFormat_BytesPerPixel_Ugly(t *core.T) {
	symbol := any(PixelFormat.BytesPerPixel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PixelFormat_BytesPerPixel_Ugly", "PixelFormat_BytesPerPixel")
}

func TestAX7_ReadGGUFInfo_Good(t *core.T) {
	symbol := any(ReadGGUFInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ReadGGUFInfo_Good", "ReadGGUFInfo")
}

func TestAX7_ReadGGUFInfo_Bad(t *core.T) {
	symbol := any(ReadGGUFInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ReadGGUFInfo_Bad", "ReadGGUFInfo")
}

func TestAX7_ReadGGUFInfo_Ugly(t *core.T) {
	symbol := any(ReadGGUFInfo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ReadGGUFInfo_Ugly", "ReadGGUFInfo")
}

func TestAX7_ResetPeakMemory_Good(t *core.T) {
	symbol := any(ResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ResetPeakMemory_Good", "ResetPeakMemory")
}

func TestAX7_ResetPeakMemory_Bad(t *core.T) {
	symbol := any(ResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ResetPeakMemory_Bad", "ResetPeakMemory")
}

func TestAX7_ResetPeakMemory_Ugly(t *core.T) {
	symbol := any(ResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ResetPeakMemory_Ugly", "ResetPeakMemory")
}

func TestAX7_Reshape_Good(t *core.T) {
	symbol := any(Reshape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Reshape_Good", "Reshape")
}

func TestAX7_Reshape_Bad(t *core.T) {
	symbol := any(Reshape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Reshape_Bad", "Reshape")
}

func TestAX7_Reshape_Ugly(t *core.T) {
	symbol := any(Reshape)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Reshape_Ugly", "Reshape")
}

func TestAX7_Session_BeginFrame_Good(t *core.T) {
	symbol := any((*computeSession).BeginFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_BeginFrame_Good", "Session_BeginFrame")
}

func TestAX7_Session_BeginFrame_Bad(t *core.T) {
	symbol := any((*computeSession).BeginFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_BeginFrame_Bad", "Session_BeginFrame")
}

func TestAX7_Session_BeginFrame_Ugly(t *core.T) {
	symbol := any((*computeSession).BeginFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_BeginFrame_Ugly", "Session_BeginFrame")
}

func TestAX7_Session_Close_Good(t *core.T) {
	symbol := any((*computeSession).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Close_Good", "Session_Close")
}

func TestAX7_Session_Close_Bad(t *core.T) {
	symbol := any((*computeSession).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Close_Bad", "Session_Close")
}

func TestAX7_Session_Close_Ugly(t *core.T) {
	symbol := any((*computeSession).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Close_Ugly", "Session_Close")
}

func TestAX7_Session_FinishFrame_Good(t *core.T) {
	symbol := any((*computeSession).FinishFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FinishFrame_Good", "Session_FinishFrame")
}

func TestAX7_Session_FinishFrame_Bad(t *core.T) {
	symbol := any((*computeSession).FinishFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FinishFrame_Bad", "Session_FinishFrame")
}

func TestAX7_Session_FinishFrame_Ugly(t *core.T) {
	symbol := any((*computeSession).FinishFrame)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FinishFrame_Ugly", "Session_FinishFrame")
}

func TestAX7_Session_FrameMetrics_Good(t *core.T) {
	symbol := any((*computeSession).FrameMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FrameMetrics_Good", "Session_FrameMetrics")
}

func TestAX7_Session_FrameMetrics_Bad(t *core.T) {
	symbol := any((*computeSession).FrameMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FrameMetrics_Bad", "Session_FrameMetrics")
}

func TestAX7_Session_FrameMetrics_Ugly(t *core.T) {
	symbol := any((*computeSession).FrameMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_FrameMetrics_Ugly", "Session_FrameMetrics")
}

func TestAX7_Session_Metrics_Good(t *core.T) {
	symbol := any((*computeSession).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Metrics_Good", "Session_Metrics")
}

func TestAX7_Session_Metrics_Bad(t *core.T) {
	symbol := any((*computeSession).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Metrics_Bad", "Session_Metrics")
}

func TestAX7_Session_Metrics_Ugly(t *core.T) {
	symbol := any((*computeSession).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Metrics_Ugly", "Session_Metrics")
}

func TestAX7_Session_NewByteBuffer_Good(t *core.T) {
	symbol := any((*computeSession).NewByteBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewByteBuffer_Good", "Session_NewByteBuffer")
}

func TestAX7_Session_NewByteBuffer_Bad(t *core.T) {
	symbol := any((*computeSession).NewByteBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewByteBuffer_Bad", "Session_NewByteBuffer")
}

func TestAX7_Session_NewByteBuffer_Ugly(t *core.T) {
	symbol := any((*computeSession).NewByteBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewByteBuffer_Ugly", "Session_NewByteBuffer")
}

func TestAX7_Session_NewPixelBuffer_Good(t *core.T) {
	symbol := any((*computeSession).NewPixelBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewPixelBuffer_Good", "Session_NewPixelBuffer")
}

func TestAX7_Session_NewPixelBuffer_Bad(t *core.T) {
	symbol := any((*computeSession).NewPixelBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewPixelBuffer_Bad", "Session_NewPixelBuffer")
}

func TestAX7_Session_NewPixelBuffer_Ugly(t *core.T) {
	symbol := any((*computeSession).NewPixelBuffer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_NewPixelBuffer_Ugly", "Session_NewPixelBuffer")
}

func TestAX7_Session_Run_Good(t *core.T) {
	symbol := any((*computeSession).Run)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Run_Good", "Session_Run")
}

func TestAX7_Session_Run_Bad(t *core.T) {
	symbol := any((*computeSession).Run)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Run_Bad", "Session_Run")
}

func TestAX7_Session_Run_Ugly(t *core.T) {
	symbol := any((*computeSession).Run)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Run_Ugly", "Session_Run")
}

func TestAX7_Session_Sync_Good(t *core.T) {
	symbol := any((*computeSession).Sync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Sync_Good", "Session_Sync")
}

func TestAX7_Session_Sync_Bad(t *core.T) {
	symbol := any((*computeSession).Sync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Sync_Bad", "Session_Sync")
}

func TestAX7_Session_Sync_Ugly(t *core.T) {
	symbol := any((*computeSession).Sync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Session_Sync_Ugly", "Session_Sync")
}

func TestAX7_SetCacheLimit_Good(t *core.T) {
	symbol := any(SetCacheLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetCacheLimit_Good", "SetCacheLimit")
}

func TestAX7_SetCacheLimit_Bad(t *core.T) {
	symbol := any(SetCacheLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetCacheLimit_Bad", "SetCacheLimit")
}

func TestAX7_SetCacheLimit_Ugly(t *core.T) {
	symbol := any(SetCacheLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetCacheLimit_Ugly", "SetCacheLimit")
}

func TestAX7_SetMemoryLimit_Good(t *core.T) {
	symbol := any(SetMemoryLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetMemoryLimit_Good", "SetMemoryLimit")
}

func TestAX7_SetMemoryLimit_Bad(t *core.T) {
	symbol := any(SetMemoryLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetMemoryLimit_Bad", "SetMemoryLimit")
}

func TestAX7_SetMemoryLimit_Ugly(t *core.T) {
	symbol := any(SetMemoryLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetMemoryLimit_Ugly", "SetMemoryLimit")
}

func TestAX7_SetWiredLimit_Good(t *core.T) {
	symbol := any(SetWiredLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetWiredLimit_Good", "SetWiredLimit")
}

func TestAX7_SetWiredLimit_Bad(t *core.T) {
	symbol := any(SetWiredLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetWiredLimit_Bad", "SetWiredLimit")
}

func TestAX7_SetWiredLimit_Ugly(t *core.T) {
	symbol := any(SetWiredLimit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SetWiredLimit_Ugly", "SetWiredLimit")
}

func TestAX7_Slice_Good(t *core.T) {
	symbol := any(Slice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Slice_Good", "Slice")
}

func TestAX7_Slice_Bad(t *core.T) {
	symbol := any(Slice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Slice_Bad", "Slice")
}

func TestAX7_Slice_Ugly(t *core.T) {
	symbol := any(Slice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Slice_Ugly", "Slice")
}

func TestAX7_Softmax_Good(t *core.T) {
	symbol := any(Softmax)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Softmax_Good", "Softmax")
}

func TestAX7_Softmax_Bad(t *core.T) {
	symbol := any(Softmax)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Softmax_Bad", "Softmax")
}

func TestAX7_Softmax_Ugly(t *core.T) {
	symbol := any(Softmax)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Softmax_Ugly", "Softmax")
}

func TestAX7_Tokenizer_BOS_Good(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Good", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_BOS_Bad(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Bad", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_BOS_Ugly(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Ugly", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_Decode_Good(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Good", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_Decode_Bad(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Bad", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_Decode_Ugly(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Ugly", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_EOS_Good(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Good", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_EOS_Bad(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Bad", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_EOS_Ugly(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Ugly", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_Encode_Good(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Good", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_Encode_Bad(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Bad", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_Encode_Ugly(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Ugly", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_IDToken_Good(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Good", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_IDToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Bad", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_IDToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Ugly", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_TokenID_Good(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Good", "Tokenizer_TokenID")
}

func TestAX7_Tokenizer_TokenID_Bad(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Bad", "Tokenizer_TokenID")
}

func TestAX7_Tokenizer_TokenID_Ugly(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Ugly", "Tokenizer_TokenID")
}

func TestAX7_TrainingModel_Good(t *core.T) {
	symbol := any(TrainingModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TrainingModel_Good", "TrainingModel")
}

func TestAX7_TrainingModel_Bad(t *core.T) {
	symbol := any(TrainingModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TrainingModel_Bad", "TrainingModel")
}

func TestAX7_TrainingModel_Ugly(t *core.T) {
	symbol := any(TrainingModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TrainingModel_Ugly", "TrainingModel")
}

func TestAX7_VJP_Good(t *core.T) {
	symbol := any(VJP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VJP_Good", "VJP")
}

func TestAX7_VJP_Bad(t *core.T) {
	symbol := any(VJP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VJP_Bad", "VJP")
}

func TestAX7_VJP_Ugly(t *core.T) {
	symbol := any(VJP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VJP_Ugly", "VJP")
}

func TestAX7_ValueAndGrad_Good(t *core.T) {
	symbol := any(ValueAndGrad)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ValueAndGrad_Good", "ValueAndGrad")
}

func TestAX7_ValueAndGrad_Bad(t *core.T) {
	symbol := any(ValueAndGrad)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ValueAndGrad_Bad", "ValueAndGrad")
}

func TestAX7_ValueAndGrad_Ugly(t *core.T) {
	symbol := any(ValueAndGrad)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ValueAndGrad_Ugly", "ValueAndGrad")
}

func TestAX7_WithAdapterPath_Good(t *core.T) {
	symbol := any(WithAdapterPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithAdapterPath_Good", "WithAdapterPath")
}

func TestAX7_WithAdapterPath_Bad(t *core.T) {
	symbol := any(WithAdapterPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithAdapterPath_Bad", "WithAdapterPath")
}

func TestAX7_WithAdapterPath_Ugly(t *core.T) {
	symbol := any(WithAdapterPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithAdapterPath_Ugly", "WithAdapterPath")
}

func TestAX7_WithContextLength_Good(t *core.T) {
	symbol := any(WithContextLength)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithContextLength_Good", "WithContextLength")
}

func TestAX7_WithContextLength_Bad(t *core.T) {
	symbol := any(WithContextLength)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithContextLength_Bad", "WithContextLength")
}

func TestAX7_WithContextLength_Ugly(t *core.T) {
	symbol := any(WithContextLength)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithContextLength_Ugly", "WithContextLength")
}

func TestAX7_WithDevice_Good(t *core.T) {
	symbol := any(WithDevice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithDevice_Good", "WithDevice")
}

func TestAX7_WithDevice_Bad(t *core.T) {
	symbol := any(WithDevice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithDevice_Bad", "WithDevice")
}

func TestAX7_WithDevice_Ugly(t *core.T) {
	symbol := any(WithDevice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithDevice_Ugly", "WithDevice")
}

func TestAX7_WithLogits_Bad(t *core.T) {
	symbol := any(WithLogits)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithLogits_Bad", "WithLogits")
}

func TestAX7_WithLogits_Ugly(t *core.T) {
	symbol := any(WithLogits)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithLogits_Ugly", "WithLogits")
}

func TestAX7_WithMaxTokens_Good(t *core.T) {
	symbol := any(WithMaxTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMaxTokens_Good", "WithMaxTokens")
}

func TestAX7_WithMaxTokens_Bad(t *core.T) {
	symbol := any(WithMaxTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMaxTokens_Bad", "WithMaxTokens")
}

func TestAX7_WithMaxTokens_Ugly(t *core.T) {
	symbol := any(WithMaxTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMaxTokens_Ugly", "WithMaxTokens")
}

func TestAX7_WithMedium_Good(t *core.T) {
	symbol := any(WithMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMedium_Good", "WithMedium")
}

func TestAX7_WithMedium_Bad(t *core.T) {
	symbol := any(WithMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMedium_Bad", "WithMedium")
}

func TestAX7_WithMedium_Ugly(t *core.T) {
	symbol := any(WithMedium)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMedium_Ugly", "WithMedium")
}

func TestAX7_WithMinP_Good(t *core.T) {
	symbol := any(WithMinP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMinP_Good", "WithMinP")
}

func TestAX7_WithMinP_Bad(t *core.T) {
	symbol := any(WithMinP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMinP_Bad", "WithMinP")
}

func TestAX7_WithMinP_Ugly(t *core.T) {
	symbol := any(WithMinP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithMinP_Ugly", "WithMinP")
}

func TestAX7_WithQuantization_Good(t *core.T) {
	symbol := any(WithQuantization)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithQuantization_Good", "WithQuantization")
}

func TestAX7_WithQuantization_Bad(t *core.T) {
	symbol := any(WithQuantization)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithQuantization_Bad", "WithQuantization")
}

func TestAX7_WithQuantization_Ugly(t *core.T) {
	symbol := any(WithQuantization)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithQuantization_Ugly", "WithQuantization")
}

func TestAX7_WithRepeatPenalty_Good(t *core.T) {
	symbol := any(WithRepeatPenalty)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithRepeatPenalty_Good", "WithRepeatPenalty")
}

func TestAX7_WithRepeatPenalty_Bad(t *core.T) {
	symbol := any(WithRepeatPenalty)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithRepeatPenalty_Bad", "WithRepeatPenalty")
}

func TestAX7_WithRepeatPenalty_Ugly(t *core.T) {
	symbol := any(WithRepeatPenalty)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithRepeatPenalty_Ugly", "WithRepeatPenalty")
}

func TestAX7_WithResetPeakMemory_Good(t *core.T) {
	symbol := any(WithResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithResetPeakMemory_Good", "WithResetPeakMemory")
}

func TestAX7_WithResetPeakMemory_Bad(t *core.T) {
	symbol := any(WithResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithResetPeakMemory_Bad", "WithResetPeakMemory")
}

func TestAX7_WithResetPeakMemory_Ugly(t *core.T) {
	symbol := any(WithResetPeakMemory)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithResetPeakMemory_Ugly", "WithResetPeakMemory")
}

func TestAX7_WithReturnLogits_Good(t *core.T) {
	symbol := any(WithReturnLogits)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithReturnLogits_Good", "WithReturnLogits")
}

func TestAX7_WithReturnLogits_Bad(t *core.T) {
	symbol := any(WithReturnLogits)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithReturnLogits_Bad", "WithReturnLogits")
}

func TestAX7_WithReturnLogits_Ugly(t *core.T) {
	symbol := any(WithReturnLogits)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithReturnLogits_Ugly", "WithReturnLogits")
}

func TestAX7_WithSessionLabel_Good(t *core.T) {
	symbol := any(WithSessionLabel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithSessionLabel_Good", "WithSessionLabel")
}

func TestAX7_WithSessionLabel_Bad(t *core.T) {
	symbol := any(WithSessionLabel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithSessionLabel_Bad", "WithSessionLabel")
}

func TestAX7_WithSessionLabel_Ugly(t *core.T) {
	symbol := any(WithSessionLabel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithSessionLabel_Ugly", "WithSessionLabel")
}

func TestAX7_WithStopTokens_Good(t *core.T) {
	symbol := any(WithStopTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithStopTokens_Good", "WithStopTokens")
}

func TestAX7_WithStopTokens_Bad(t *core.T) {
	symbol := any(WithStopTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithStopTokens_Bad", "WithStopTokens")
}

func TestAX7_WithStopTokens_Ugly(t *core.T) {
	symbol := any(WithStopTokens)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithStopTokens_Ugly", "WithStopTokens")
}

func TestAX7_WithTemperature_Good(t *core.T) {
	symbol := any(WithTemperature)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTemperature_Good", "WithTemperature")
}

func TestAX7_WithTemperature_Bad(t *core.T) {
	symbol := any(WithTemperature)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTemperature_Bad", "WithTemperature")
}

func TestAX7_WithTemperature_Ugly(t *core.T) {
	symbol := any(WithTemperature)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTemperature_Ugly", "WithTemperature")
}

func TestAX7_WithTopK_Good(t *core.T) {
	symbol := any(WithTopK)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopK_Good", "WithTopK")
}

func TestAX7_WithTopK_Bad(t *core.T) {
	symbol := any(WithTopK)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopK_Bad", "WithTopK")
}

func TestAX7_WithTopK_Ugly(t *core.T) {
	symbol := any(WithTopK)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopK_Ugly", "WithTopK")
}

func TestAX7_WithTopP_Good(t *core.T) {
	symbol := any(WithTopP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopP_Good", "WithTopP")
}

func TestAX7_WithTopP_Bad(t *core.T) {
	symbol := any(WithTopP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopP_Bad", "WithTopP")
}

func TestAX7_WithTopP_Ugly(t *core.T) {
	symbol := any(WithTopP)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithTopP_Ugly", "WithTopP")
}

func TestAX7_WithVerboseKernels_Good(t *core.T) {
	symbol := any(WithVerboseKernels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithVerboseKernels_Good", "WithVerboseKernels")
}

func TestAX7_WithVerboseKernels_Bad(t *core.T) {
	symbol := any(WithVerboseKernels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithVerboseKernels_Bad", "WithVerboseKernels")
}

func TestAX7_WithVerboseKernels_Ugly(t *core.T) {
	symbol := any(WithVerboseKernels)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "WithVerboseKernels_Ugly", "WithVerboseKernels")
}

func TestAX7_Zeros_Good(t *core.T) {
	symbol := any(Zeros)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Zeros_Good", "Zeros")
}

func TestAX7_Zeros_Bad(t *core.T) {
	symbol := any(Zeros)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Zeros_Bad", "Zeros")
}

func TestAX7_Zeros_Ugly(t *core.T) {
	symbol := any(Zeros)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Zeros_Ugly", "Zeros")
}
