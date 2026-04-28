// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

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

func TestAX7_AddScalar_Bad(t *core.T) {
	symbol := any(AddScalar)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AddScalar_Bad", "AddScalar")
}

func TestAX7_AddScalar_Ugly(t *core.T) {
	symbol := any(AddScalar)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AddScalar_Ugly", "AddScalar")
}

func TestAX7_Any_Good(t *core.T) {
	symbol := any(Any)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Any_Good", "Any")
}

func TestAX7_Any_Bad(t *core.T) {
	symbol := any(Any)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Any_Bad", "Any")
}

func TestAX7_Any_Ugly(t *core.T) {
	symbol := any(Any)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Any_Ugly", "Any")
}

func TestAX7_AnyAxis_Good(t *core.T) {
	symbol := any(AnyAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AnyAxis_Good", "AnyAxis")
}

func TestAX7_AnyAxis_Bad(t *core.T) {
	symbol := any(AnyAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AnyAxis_Bad", "AnyAxis")
}

func TestAX7_AnyAxis_Ugly(t *core.T) {
	symbol := any(AnyAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AnyAxis_Ugly", "AnyAxis")
}

func TestAX7_Arange_Good(t *core.T) {
	symbol := any(Arange)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Arange_Good", "Arange")
}

func TestAX7_Arange_Bad(t *core.T) {
	symbol := any(Arange)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Arange_Bad", "Arange")
}

func TestAX7_Arange_Ugly(t *core.T) {
	symbol := any(Arange)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Arange_Ugly", "Arange")
}

func TestAX7_Argmax_Bad(t *core.T) {
	symbol := any(Argmax)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argmax_Bad", "Argmax")
}

func TestAX7_Argmax_Ugly(t *core.T) {
	symbol := any(Argmax)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argmax_Ugly", "Argmax")
}

func TestAX7_Argpartition_Good(t *core.T) {
	symbol := any(Argpartition)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argpartition_Good", "Argpartition")
}

func TestAX7_Argpartition_Bad(t *core.T) {
	symbol := any(Argpartition)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argpartition_Bad", "Argpartition")
}

func TestAX7_Argpartition_Ugly(t *core.T) {
	symbol := any(Argpartition)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argpartition_Ugly", "Argpartition")
}

func TestAX7_Argsort_Bad(t *core.T) {
	symbol := any(Argsort)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argsort_Bad", "Argsort")
}

func TestAX7_Argsort_Ugly(t *core.T) {
	symbol := any(Argsort)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Argsort_Ugly", "Argsort")
}

func TestAX7_Array_Bool_Good(t *core.T) {
	symbol := any(Array.Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Good", "Array_Bool")
}

func TestAX7_Array_Bool_Bad(t *core.T) {
	symbol := any(Array.Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Bad", "Array_Bool")
}

func TestAX7_Array_Bool_Ugly(t *core.T) {
	symbol := any(Array.Bool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bool_Ugly", "Array_Bool")
}

func TestAX7_Array_Bytes_Good(t *core.T) {
	symbol := any((*Array).Bytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bytes_Good", "Array_Bytes")
}

func TestAX7_Array_Bytes_Bad(t *core.T) {
	symbol := any((*Array).Bytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bytes_Bad", "Array_Bytes")
}

func TestAX7_Array_Bytes_Ugly(t *core.T) {
	symbol := any((*Array).Bytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Bytes_Ugly", "Array_Bytes")
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
	symbol := any(Array.Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Good", "Array_Dim")
}

func TestAX7_Array_Dim_Bad(t *core.T) {
	symbol := any(Array.Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Bad", "Array_Dim")
}

func TestAX7_Array_Dim_Ugly(t *core.T) {
	symbol := any(Array.Dim)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dim_Ugly", "Array_Dim")
}

func TestAX7_Array_Dims_Good(t *core.T) {
	symbol := any(Array.Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Good", "Array_Dims")
}

func TestAX7_Array_Dims_Bad(t *core.T) {
	symbol := any(Array.Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Bad", "Array_Dims")
}

func TestAX7_Array_Dims_Ugly(t *core.T) {
	symbol := any(Array.Dims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dims_Ugly", "Array_Dims")
}

func TestAX7_Array_Dtype_Good(t *core.T) {
	symbol := any(Array.Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Good", "Array_Dtype")
}

func TestAX7_Array_Dtype_Bad(t *core.T) {
	symbol := any(Array.Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Bad", "Array_Dtype")
}

func TestAX7_Array_Dtype_Ugly(t *core.T) {
	symbol := any(Array.Dtype)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Dtype_Ugly", "Array_Dtype")
}

func TestAX7_Array_Float_Good(t *core.T) {
	symbol := any(Array.Float)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Float_Good", "Array_Float")
}

func TestAX7_Array_Float_Bad(t *core.T) {
	symbol := any(Array.Float)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Float_Bad", "Array_Float")
}

func TestAX7_Array_Float_Ugly(t *core.T) {
	symbol := any(Array.Float)
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
	symbol := any(Array.Int)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Int_Good", "Array_Int")
}

func TestAX7_Array_Int_Bad(t *core.T) {
	symbol := any(Array.Int)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Int_Bad", "Array_Int")
}

func TestAX7_Array_Int_Ugly(t *core.T) {
	symbol := any(Array.Int)
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

func TestAX7_Array_IsRowContiguous_Good(t *core.T) {
	symbol := any(Array.IsRowContiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_IsRowContiguous_Good", "Array_IsRowContiguous")
}

func TestAX7_Array_IsRowContiguous_Bad(t *core.T) {
	symbol := any(Array.IsRowContiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_IsRowContiguous_Bad", "Array_IsRowContiguous")
}

func TestAX7_Array_IsRowContiguous_Ugly(t *core.T) {
	symbol := any(Array.IsRowContiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_IsRowContiguous_Ugly", "Array_IsRowContiguous")
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

func TestAX7_Array_NumBytes_Good(t *core.T) {
	symbol := any(Array.NumBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumBytes_Good", "Array_NumBytes")
}

func TestAX7_Array_NumBytes_Bad(t *core.T) {
	symbol := any(Array.NumBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumBytes_Bad", "Array_NumBytes")
}

func TestAX7_Array_NumBytes_Ugly(t *core.T) {
	symbol := any(Array.NumBytes)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumBytes_Ugly", "Array_NumBytes")
}

func TestAX7_Array_NumDims_Good(t *core.T) {
	symbol := any(Array.NumDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumDims_Good", "Array_NumDims")
}

func TestAX7_Array_NumDims_Bad(t *core.T) {
	symbol := any(Array.NumDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_NumDims_Bad", "Array_NumDims")
}

func TestAX7_Array_NumDims_Ugly(t *core.T) {
	symbol := any(Array.NumDims)
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

func TestAX7_Array_ShapeRaw_Good(t *core.T) {
	symbol := any(Array.ShapeRaw)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_ShapeRaw_Good", "Array_ShapeRaw")
}

func TestAX7_Array_ShapeRaw_Bad(t *core.T) {
	symbol := any(Array.ShapeRaw)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_ShapeRaw_Bad", "Array_ShapeRaw")
}

func TestAX7_Array_ShapeRaw_Ugly(t *core.T) {
	symbol := any(Array.ShapeRaw)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_ShapeRaw_Ugly", "Array_ShapeRaw")
}

func TestAX7_Array_Size_Good(t *core.T) {
	symbol := any(Array.Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Size_Good", "Array_Size")
}

func TestAX7_Array_Size_Bad(t *core.T) {
	symbol := any(Array.Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Size_Bad", "Array_Size")
}

func TestAX7_Array_Size_Ugly(t *core.T) {
	symbol := any(Array.Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Size_Ugly", "Array_Size")
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

func TestAX7_Array_Valid_Ugly(t *core.T) {
	symbol := any((*Array).Valid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Array_Valid_Ugly", "Array_Valid")
}

func TestAX7_AsStrided_Good(t *core.T) {
	symbol := any(AsStrided)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AsStrided_Good", "AsStrided")
}

func TestAX7_AsStrided_Bad(t *core.T) {
	symbol := any(AsStrided)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AsStrided_Bad", "AsStrided")
}

func TestAX7_AsStrided_Ugly(t *core.T) {
	symbol := any(AsStrided)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AsStrided_Ugly", "AsStrided")
}

func TestAX7_AsType_Bad(t *core.T) {
	symbol := any(AsType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AsType_Bad", "AsType")
}

func TestAX7_AsType_Ugly(t *core.T) {
	symbol := any(AsType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "AsType_Ugly", "AsType")
}

func TestAX7_BroadcastTo_Bad(t *core.T) {
	symbol := any(BroadcastTo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "BroadcastTo_Bad", "BroadcastTo")
}

func TestAX7_BroadcastTo_Ugly(t *core.T) {
	symbol := any(BroadcastTo)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "BroadcastTo_Ugly", "BroadcastTo")
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

func TestAX7_ClosureKwargs_Free_Good(t *core.T) {
	symbol := any((*ClosureKwargs).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClosureKwargs_Free_Good", "ClosureKwargs_Free")
}

func TestAX7_ClosureKwargs_Free_Bad(t *core.T) {
	symbol := any((*ClosureKwargs).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClosureKwargs_Free_Bad", "ClosureKwargs_Free")
}

func TestAX7_ClosureKwargs_Free_Ugly(t *core.T) {
	symbol := any((*ClosureKwargs).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ClosureKwargs_Free_Ugly", "ClosureKwargs_Free")
}

func TestAX7_Closure_Free_Good(t *core.T) {
	symbol := any((*Closure).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Closure_Free_Good", "Closure_Free")
}

func TestAX7_Closure_Free_Bad(t *core.T) {
	symbol := any((*Closure).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Closure_Free_Bad", "Closure_Free")
}

func TestAX7_Closure_Free_Ugly(t *core.T) {
	symbol := any((*Closure).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Closure_Free_Ugly", "Closure_Free")
}

func TestAX7_CompileShapeless_Good(t *core.T) {
	symbol := any(CompileShapeless)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompileShapeless_Good", "CompileShapeless")
}

func TestAX7_CompileShapeless_Bad(t *core.T) {
	symbol := any(CompileShapeless)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompileShapeless_Bad", "CompileShapeless")
}

func TestAX7_CompileShapeless_Ugly(t *core.T) {
	symbol := any(CompileShapeless)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompileShapeless_Ugly", "CompileShapeless")
}

func TestAX7_CompiledFunc_Call_Good(t *core.T) {
	symbol := any((*CompiledFunc).Call)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompiledFunc_Call_Good", "CompiledFunc_Call")
}

func TestAX7_CompiledFunc_Call_Bad(t *core.T) {
	symbol := any((*CompiledFunc).Call)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompiledFunc_Call_Bad", "CompiledFunc_Call")
}

func TestAX7_CompiledFunc_Call_Ugly(t *core.T) {
	symbol := any((*CompiledFunc).Call)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CompiledFunc_Call_Ugly", "CompiledFunc_Call")
}

func TestAX7_Concatenate_Bad(t *core.T) {
	symbol := any(Concatenate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Concatenate_Bad", "Concatenate")
}

func TestAX7_Concatenate_Ugly(t *core.T) {
	symbol := any(Concatenate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Concatenate_Ugly", "Concatenate")
}

func TestAX7_Contiguous_Good(t *core.T) {
	symbol := any(Contiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Contiguous_Good", "Contiguous")
}

func TestAX7_Contiguous_Bad(t *core.T) {
	symbol := any(Contiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Contiguous_Bad", "Contiguous")
}

func TestAX7_Contiguous_Ugly(t *core.T) {
	symbol := any(Contiguous)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Contiguous_Ugly", "Contiguous")
}

func TestAX7_Conv2d_Good(t *core.T) {
	symbol := any(Conv2d)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Conv2d_Good", "Conv2d")
}

func TestAX7_Conv2d_Bad(t *core.T) {
	symbol := any(Conv2d)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Conv2d_Bad", "Conv2d")
}

func TestAX7_Conv2d_Ugly(t *core.T) {
	symbol := any(Conv2d)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Conv2d_Ugly", "Conv2d")
}

func TestAX7_Copy_Good(t *core.T) {
	symbol := any(Copy)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Copy_Good", "Copy")
}

func TestAX7_Copy_Bad(t *core.T) {
	symbol := any(Copy)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Copy_Bad", "Copy")
}

func TestAX7_Copy_Ugly(t *core.T) {
	symbol := any(Copy)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Copy_Ugly", "Copy")
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

func TestAX7_CumSum_Bad(t *core.T) {
	symbol := any(CumSum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CumSum_Bad", "CumSum")
}

func TestAX7_CumSum_Ugly(t *core.T) {
	symbol := any(CumSum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "CumSum_Ugly", "CumSum")
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

func TestAX7_DType_UnmarshalJSON_Good(t *core.T) {
	symbol := any((*DType).UnmarshalJSON)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_UnmarshalJSON_Good", "DType_UnmarshalJSON")
}

func TestAX7_DType_UnmarshalJSON_Bad(t *core.T) {
	symbol := any((*DType).UnmarshalJSON)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_UnmarshalJSON_Bad", "DType_UnmarshalJSON")
}

func TestAX7_DType_UnmarshalJSON_Ugly(t *core.T) {
	symbol := any((*DType).UnmarshalJSON)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DType_UnmarshalJSON_Ugly", "DType_UnmarshalJSON")
}

func TestAX7_DefaultAdamWConfig_Good(t *core.T) {
	symbol := any(DefaultAdamWConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultAdamWConfig_Good", "DefaultAdamWConfig")
}

func TestAX7_DefaultAdamWConfig_Bad(t *core.T) {
	symbol := any(DefaultAdamWConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultAdamWConfig_Bad", "DefaultAdamWConfig")
}

func TestAX7_DefaultAdamWConfig_Ugly(t *core.T) {
	symbol := any(DefaultAdamWConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultAdamWConfig_Ugly", "DefaultAdamWConfig")
}

func TestAX7_DefaultCPUStream_Good(t *core.T) {
	symbol := any(DefaultCPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCPUStream_Good", "DefaultCPUStream")
}

func TestAX7_DefaultCPUStream_Bad(t *core.T) {
	symbol := any(DefaultCPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCPUStream_Bad", "DefaultCPUStream")
}

func TestAX7_DefaultCPUStream_Ugly(t *core.T) {
	symbol := any(DefaultCPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultCPUStream_Ugly", "DefaultCPUStream")
}

func TestAX7_DefaultGPUStream_Good(t *core.T) {
	symbol := any(DefaultGPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGPUStream_Good", "DefaultGPUStream")
}

func TestAX7_DefaultGPUStream_Bad(t *core.T) {
	symbol := any(DefaultGPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGPUStream_Bad", "DefaultGPUStream")
}

func TestAX7_DefaultGPUStream_Ugly(t *core.T) {
	symbol := any(DefaultGPUStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultGPUStream_Ugly", "DefaultGPUStream")
}

func TestAX7_DefaultLoRAConfig_Bad(t *core.T) {
	symbol := any(DefaultLoRAConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultLoRAConfig_Bad", "DefaultLoRAConfig")
}

func TestAX7_DefaultLoRAConfig_Ugly(t *core.T) {
	symbol := any(DefaultLoRAConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultLoRAConfig_Ugly", "DefaultLoRAConfig")
}

func TestAX7_DefaultStream_Good(t *core.T) {
	symbol := any(DefaultStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultStream_Good", "DefaultStream")
}

func TestAX7_DefaultStream_Bad(t *core.T) {
	symbol := any(DefaultStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultStream_Bad", "DefaultStream")
}

func TestAX7_DefaultStream_Ugly(t *core.T) {
	symbol := any(DefaultStream)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultStream_Ugly", "DefaultStream")
}

func TestAX7_Dequantize_Good(t *core.T) {
	symbol := any(Dequantize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Dequantize_Good", "Dequantize")
}

func TestAX7_Dequantize_Bad(t *core.T) {
	symbol := any(Dequantize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Dequantize_Bad", "Dequantize")
}

func TestAX7_Dequantize_Ugly(t *core.T) {
	symbol := any(Dequantize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Dequantize_Ugly", "Dequantize")
}

func TestAX7_Detach_Good(t *core.T) {
	symbol := any(Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Detach_Good", "Detach")
}

func TestAX7_Detach_Bad(t *core.T) {
	symbol := any(Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Detach_Bad", "Detach")
}

func TestAX7_Detach_Ugly(t *core.T) {
	symbol := any(Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Detach_Ugly", "Detach")
}

func TestAX7_Divide_Bad(t *core.T) {
	symbol := any(Divide)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Divide_Bad", "Divide")
}

func TestAX7_Divide_Ugly(t *core.T) {
	symbol := any(Divide)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Divide_Ugly", "Divide")
}

func TestAX7_Embedding_AsLinear_Good(t *core.T) {
	symbol := any((*Embedding).AsLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_AsLinear_Good", "Embedding_AsLinear")
}

func TestAX7_Embedding_AsLinear_Bad(t *core.T) {
	symbol := any((*Embedding).AsLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_AsLinear_Bad", "Embedding_AsLinear")
}

func TestAX7_Embedding_AsLinear_Ugly(t *core.T) {
	symbol := any((*Embedding).AsLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_AsLinear_Ugly", "Embedding_AsLinear")
}

func TestAX7_Embedding_Forward_Good(t *core.T) {
	symbol := any((*Embedding).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_Forward_Good", "Embedding_Forward")
}

func TestAX7_Embedding_Forward_Bad(t *core.T) {
	symbol := any((*Embedding).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_Forward_Bad", "Embedding_Forward")
}

func TestAX7_Embedding_Forward_Ugly(t *core.T) {
	symbol := any((*Embedding).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Embedding_Forward_Ugly", "Embedding_Forward")
}

func TestAX7_Eval_Bad(t *core.T) {
	symbol := any(Eval)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Eval_Bad", "Eval")
}

func TestAX7_Eval_Ugly(t *core.T) {
	symbol := any(Eval)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Eval_Ugly", "Eval")
}

func TestAX7_EvalAsync_Good(t *core.T) {
	symbol := any(EvalAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "EvalAsync_Good", "EvalAsync")
}

func TestAX7_EvalAsync_Bad(t *core.T) {
	symbol := any(EvalAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "EvalAsync_Bad", "EvalAsync")
}

func TestAX7_EvalAsync_Ugly(t *core.T) {
	symbol := any(EvalAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "EvalAsync_Ugly", "EvalAsync")
}

func TestAX7_Exp_Bad(t *core.T) {
	symbol := any(Exp)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Exp_Bad", "Exp")
}

func TestAX7_Exp_Ugly(t *core.T) {
	symbol := any(Exp)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Exp_Ugly", "Exp")
}

func TestAX7_ExpandDims_Bad(t *core.T) {
	symbol := any(ExpandDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExpandDims_Bad", "ExpandDims")
}

func TestAX7_ExpandDims_Ugly(t *core.T) {
	symbol := any(ExpandDims)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExpandDims_Ugly", "ExpandDims")
}

func TestAX7_ExportFunction_Good(t *core.T) {
	symbol := any(ExportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunction_Good", "ExportFunction")
}

func TestAX7_ExportFunction_Bad(t *core.T) {
	symbol := any(ExportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunction_Bad", "ExportFunction")
}

func TestAX7_ExportFunction_Ugly(t *core.T) {
	symbol := any(ExportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunction_Ugly", "ExportFunction")
}

func TestAX7_ExportFunctionKwargs_Good(t *core.T) {
	symbol := any(ExportFunctionKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunctionKwargs_Good", "ExportFunctionKwargs")
}

func TestAX7_ExportFunctionKwargs_Bad(t *core.T) {
	symbol := any(ExportFunctionKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunctionKwargs_Bad", "ExportFunctionKwargs")
}

func TestAX7_ExportFunctionKwargs_Ugly(t *core.T) {
	symbol := any(ExportFunctionKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ExportFunctionKwargs_Ugly", "ExportFunctionKwargs")
}

func TestAX7_FormatGemmaPrompt_Bad(t *core.T) {
	symbol := any(FormatGemmaPrompt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FormatGemmaPrompt_Bad", "FormatGemmaPrompt")
}

func TestAX7_FormatGemmaPrompt_Ugly(t *core.T) {
	symbol := any(FormatGemmaPrompt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FormatGemmaPrompt_Ugly", "FormatGemmaPrompt")
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

func TestAX7_FromValue_Bad(t *core.T) {
	symbol := any(FromValue[float32])
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FromValue_Bad", "FromValue")
}

func TestAX7_FromValue_Ugly(t *core.T) {
	symbol := any(FromValue[float32])
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FromValue_Ugly", "FromValue")
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

func TestAX7_GatherMM_Good(t *core.T) {
	symbol := any(GatherMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherMM_Good", "GatherMM")
}

func TestAX7_GatherMM_Bad(t *core.T) {
	symbol := any(GatherMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherMM_Bad", "GatherMM")
}

func TestAX7_GatherMM_Ugly(t *core.T) {
	symbol := any(GatherMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherMM_Ugly", "GatherMM")
}

func TestAX7_GatherQMM_Good(t *core.T) {
	symbol := any(GatherQMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherQMM_Good", "GatherQMM")
}

func TestAX7_GatherQMM_Bad(t *core.T) {
	symbol := any(GatherQMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherQMM_Bad", "GatherQMM")
}

func TestAX7_GatherQMM_Ugly(t *core.T) {
	symbol := any(GatherQMM)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GatherQMM_Ugly", "GatherQMM")
}

func TestAX7_Gemma4Model_ApplyLoRA_Good(t *core.T) {
	symbol := any((*Gemma4Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ApplyLoRA_Good", "Gemma4Model_ApplyLoRA")
}

func TestAX7_Gemma4Model_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*Gemma4Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ApplyLoRA_Bad", "Gemma4Model_ApplyLoRA")
}

func TestAX7_Gemma4Model_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ApplyLoRA_Ugly", "Gemma4Model_ApplyLoRA")
}

func TestAX7_Gemma4Model_Forward_Good(t *core.T) {
	symbol := any((*Gemma4Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Forward_Good", "Gemma4Model_Forward")
}

func TestAX7_Gemma4Model_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Forward_Bad", "Gemma4Model_Forward")
}

func TestAX7_Gemma4Model_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Forward_Ugly", "Gemma4Model_Forward")
}

func TestAX7_Gemma4Model_ForwardMasked_Good(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMasked_Good", "Gemma4Model_ForwardMasked")
}

func TestAX7_Gemma4Model_ForwardMasked_Bad(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMasked_Bad", "Gemma4Model_ForwardMasked")
}

func TestAX7_Gemma4Model_ForwardMasked_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMasked_Ugly", "Gemma4Model_ForwardMasked")
}

func TestAX7_Gemma4Model_ForwardMultiModal_Good(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMultiModal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMultiModal_Good", "Gemma4Model_ForwardMultiModal")
}

func TestAX7_Gemma4Model_ForwardMultiModal_Bad(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMultiModal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMultiModal_Bad", "Gemma4Model_ForwardMultiModal")
}

func TestAX7_Gemma4Model_ForwardMultiModal_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).ForwardMultiModal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ForwardMultiModal_Ugly", "Gemma4Model_ForwardMultiModal")
}

func TestAX7_Gemma4Model_ModelType_Good(t *core.T) {
	symbol := any((*Gemma4Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ModelType_Good", "Gemma4Model_ModelType")
}

func TestAX7_Gemma4Model_ModelType_Bad(t *core.T) {
	symbol := any((*Gemma4Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ModelType_Bad", "Gemma4Model_ModelType")
}

func TestAX7_Gemma4Model_ModelType_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_ModelType_Ugly", "Gemma4Model_ModelType")
}

func TestAX7_Gemma4Model_NewCache_Good(t *core.T) {
	symbol := any((*Gemma4Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NewCache_Good", "Gemma4Model_NewCache")
}

func TestAX7_Gemma4Model_NewCache_Bad(t *core.T) {
	symbol := any((*Gemma4Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NewCache_Bad", "Gemma4Model_NewCache")
}

func TestAX7_Gemma4Model_NewCache_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NewCache_Ugly", "Gemma4Model_NewCache")
}

func TestAX7_Gemma4Model_NumLayers_Good(t *core.T) {
	symbol := any((*Gemma4Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NumLayers_Good", "Gemma4Model_NumLayers")
}

func TestAX7_Gemma4Model_NumLayers_Bad(t *core.T) {
	symbol := any((*Gemma4Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NumLayers_Bad", "Gemma4Model_NumLayers")
}

func TestAX7_Gemma4Model_NumLayers_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_NumLayers_Ugly", "Gemma4Model_NumLayers")
}

func TestAX7_Gemma4Model_Tokenizer_Good(t *core.T) {
	symbol := any((*Gemma4Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Tokenizer_Good", "Gemma4Model_Tokenizer")
}

func TestAX7_Gemma4Model_Tokenizer_Bad(t *core.T) {
	symbol := any((*Gemma4Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Tokenizer_Bad", "Gemma4Model_Tokenizer")
}

func TestAX7_Gemma4Model_Tokenizer_Ugly(t *core.T) {
	symbol := any((*Gemma4Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4Model_Tokenizer_Ugly", "Gemma4Model_Tokenizer")
}

func TestAX7_Gemma4MultiModalProjector_Forward_Good(t *core.T) {
	symbol := any((*Gemma4MultiModalProjector).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4MultiModalProjector_Forward_Good", "Gemma4MultiModalProjector_Forward")
}

func TestAX7_Gemma4MultiModalProjector_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4MultiModalProjector).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4MultiModalProjector_Forward_Bad", "Gemma4MultiModalProjector_Forward")
}

func TestAX7_Gemma4MultiModalProjector_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4MultiModalProjector).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4MultiModalProjector_Forward_Ugly", "Gemma4MultiModalProjector_Forward")
}

func TestAX7_Gemma4VisionAttention_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionAttention).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionAttention_Forward_Good", "Gemma4VisionAttention_Forward")
}

func TestAX7_Gemma4VisionAttention_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionAttention).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionAttention_Forward_Bad", "Gemma4VisionAttention_Forward")
}

func TestAX7_Gemma4VisionAttention_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionAttention).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionAttention_Forward_Ugly", "Gemma4VisionAttention_Forward")
}

func TestAX7_Gemma4VisionEncoderLayer_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionEncoderLayer).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoderLayer_Forward_Good", "Gemma4VisionEncoderLayer_Forward")
}

func TestAX7_Gemma4VisionEncoderLayer_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionEncoderLayer).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoderLayer_Forward_Bad", "Gemma4VisionEncoderLayer_Forward")
}

func TestAX7_Gemma4VisionEncoderLayer_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionEncoderLayer).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoderLayer_Forward_Ugly", "Gemma4VisionEncoderLayer_Forward")
}

func TestAX7_Gemma4VisionEncoder_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionEncoder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoder_Forward_Good", "Gemma4VisionEncoder_Forward")
}

func TestAX7_Gemma4VisionEncoder_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionEncoder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoder_Forward_Bad", "Gemma4VisionEncoder_Forward")
}

func TestAX7_Gemma4VisionEncoder_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionEncoder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionEncoder_Forward_Ugly", "Gemma4VisionEncoder_Forward")
}

func TestAX7_Gemma4VisionMLP_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionMLP).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionMLP_Forward_Good", "Gemma4VisionMLP_Forward")
}

func TestAX7_Gemma4VisionMLP_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionMLP).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionMLP_Forward_Bad", "Gemma4VisionMLP_Forward")
}

func TestAX7_Gemma4VisionMLP_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionMLP).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionMLP_Forward_Ugly", "Gemma4VisionMLP_Forward")
}

func TestAX7_Gemma4VisionModel_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionModel_Forward_Good", "Gemma4VisionModel_Forward")
}

func TestAX7_Gemma4VisionModel_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionModel_Forward_Bad", "Gemma4VisionModel_Forward")
}

func TestAX7_Gemma4VisionModel_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionModel_Forward_Ugly", "Gemma4VisionModel_Forward")
}

func TestAX7_Gemma4VisionPatchEmbedder_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionPatchEmbedder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPatchEmbedder_Forward_Good", "Gemma4VisionPatchEmbedder_Forward")
}

func TestAX7_Gemma4VisionPatchEmbedder_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionPatchEmbedder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPatchEmbedder_Forward_Bad", "Gemma4VisionPatchEmbedder_Forward")
}

func TestAX7_Gemma4VisionPatchEmbedder_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionPatchEmbedder).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPatchEmbedder_Forward_Ugly", "Gemma4VisionPatchEmbedder_Forward")
}

func TestAX7_Gemma4VisionPooler_Forward_Good(t *core.T) {
	symbol := any((*Gemma4VisionPooler).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPooler_Forward_Good", "Gemma4VisionPooler_Forward")
}

func TestAX7_Gemma4VisionPooler_Forward_Bad(t *core.T) {
	symbol := any((*Gemma4VisionPooler).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPooler_Forward_Bad", "Gemma4VisionPooler_Forward")
}

func TestAX7_Gemma4VisionPooler_Forward_Ugly(t *core.T) {
	symbol := any((*Gemma4VisionPooler).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Gemma4VisionPooler_Forward_Ugly", "Gemma4VisionPooler_Forward")
}

func TestAX7_GemmaModel_ApplyLoRA_Good(t *core.T) {
	symbol := any((*GemmaModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ApplyLoRA_Good", "GemmaModel_ApplyLoRA")
}

func TestAX7_GemmaModel_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*GemmaModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ApplyLoRA_Bad", "GemmaModel_ApplyLoRA")
}

func TestAX7_GemmaModel_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*GemmaModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ApplyLoRA_Ugly", "GemmaModel_ApplyLoRA")
}

func TestAX7_GemmaModel_Forward_Good(t *core.T) {
	symbol := any((*GemmaModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Forward_Good", "GemmaModel_Forward")
}

func TestAX7_GemmaModel_Forward_Bad(t *core.T) {
	symbol := any((*GemmaModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Forward_Bad", "GemmaModel_Forward")
}

func TestAX7_GemmaModel_Forward_Ugly(t *core.T) {
	symbol := any((*GemmaModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Forward_Ugly", "GemmaModel_Forward")
}

func TestAX7_GemmaModel_ForwardMasked_Good(t *core.T) {
	symbol := any((*GemmaModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ForwardMasked_Good", "GemmaModel_ForwardMasked")
}

func TestAX7_GemmaModel_ForwardMasked_Bad(t *core.T) {
	symbol := any((*GemmaModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ForwardMasked_Bad", "GemmaModel_ForwardMasked")
}

func TestAX7_GemmaModel_ForwardMasked_Ugly(t *core.T) {
	symbol := any((*GemmaModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ForwardMasked_Ugly", "GemmaModel_ForwardMasked")
}

func TestAX7_GemmaModel_ModelType_Good(t *core.T) {
	symbol := any((*GemmaModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ModelType_Good", "GemmaModel_ModelType")
}

func TestAX7_GemmaModel_ModelType_Bad(t *core.T) {
	symbol := any((*GemmaModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ModelType_Bad", "GemmaModel_ModelType")
}

func TestAX7_GemmaModel_ModelType_Ugly(t *core.T) {
	symbol := any((*GemmaModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_ModelType_Ugly", "GemmaModel_ModelType")
}

func TestAX7_GemmaModel_NewCache_Good(t *core.T) {
	symbol := any((*GemmaModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NewCache_Good", "GemmaModel_NewCache")
}

func TestAX7_GemmaModel_NewCache_Bad(t *core.T) {
	symbol := any((*GemmaModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NewCache_Bad", "GemmaModel_NewCache")
}

func TestAX7_GemmaModel_NewCache_Ugly(t *core.T) {
	symbol := any((*GemmaModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NewCache_Ugly", "GemmaModel_NewCache")
}

func TestAX7_GemmaModel_NumLayers_Good(t *core.T) {
	symbol := any((*GemmaModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NumLayers_Good", "GemmaModel_NumLayers")
}

func TestAX7_GemmaModel_NumLayers_Bad(t *core.T) {
	symbol := any((*GemmaModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NumLayers_Bad", "GemmaModel_NumLayers")
}

func TestAX7_GemmaModel_NumLayers_Ugly(t *core.T) {
	symbol := any((*GemmaModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_NumLayers_Ugly", "GemmaModel_NumLayers")
}

func TestAX7_GemmaModel_Tokenizer_Good(t *core.T) {
	symbol := any((*GemmaModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Tokenizer_Good", "GemmaModel_Tokenizer")
}

func TestAX7_GemmaModel_Tokenizer_Bad(t *core.T) {
	symbol := any((*GemmaModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Tokenizer_Bad", "GemmaModel_Tokenizer")
}

func TestAX7_GemmaModel_Tokenizer_Ugly(t *core.T) {
	symbol := any((*GemmaModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "GemmaModel_Tokenizer_Ugly", "GemmaModel_Tokenizer")
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

func TestAX7_Greater_Bad(t *core.T) {
	symbol := any(Greater)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Greater_Bad", "Greater")
}

func TestAX7_Greater_Ugly(t *core.T) {
	symbol := any(Greater)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Greater_Ugly", "Greater")
}

func TestAX7_ImportFunction_Good(t *core.T) {
	symbol := any(ImportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportFunction_Good", "ImportFunction")
}

func TestAX7_ImportFunction_Bad(t *core.T) {
	symbol := any(ImportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportFunction_Bad", "ImportFunction")
}

func TestAX7_ImportFunction_Ugly(t *core.T) {
	symbol := any(ImportFunction)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportFunction_Ugly", "ImportFunction")
}

func TestAX7_ImportedFunction_Apply_Good(t *core.T) {
	symbol := any((*ImportedFunction).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Apply_Good", "ImportedFunction_Apply")
}

func TestAX7_ImportedFunction_Apply_Bad(t *core.T) {
	symbol := any((*ImportedFunction).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Apply_Bad", "ImportedFunction_Apply")
}

func TestAX7_ImportedFunction_Apply_Ugly(t *core.T) {
	symbol := any((*ImportedFunction).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Apply_Ugly", "ImportedFunction_Apply")
}

func TestAX7_ImportedFunction_ApplyKwargs_Good(t *core.T) {
	symbol := any((*ImportedFunction).ApplyKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_ApplyKwargs_Good", "ImportedFunction_ApplyKwargs")
}

func TestAX7_ImportedFunction_ApplyKwargs_Bad(t *core.T) {
	symbol := any((*ImportedFunction).ApplyKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_ApplyKwargs_Bad", "ImportedFunction_ApplyKwargs")
}

func TestAX7_ImportedFunction_ApplyKwargs_Ugly(t *core.T) {
	symbol := any((*ImportedFunction).ApplyKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_ApplyKwargs_Ugly", "ImportedFunction_ApplyKwargs")
}

func TestAX7_ImportedFunction_Free_Good(t *core.T) {
	symbol := any((*ImportedFunction).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Free_Good", "ImportedFunction_Free")
}

func TestAX7_ImportedFunction_Free_Bad(t *core.T) {
	symbol := any((*ImportedFunction).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Free_Bad", "ImportedFunction_Free")
}

func TestAX7_ImportedFunction_Free_Ugly(t *core.T) {
	symbol := any((*ImportedFunction).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ImportedFunction_Free_Ugly", "ImportedFunction_Free")
}

func TestAX7_Init_Good(t *core.T) {
	symbol := any(Init)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Init_Good", "Init")
}

func TestAX7_Init_Bad(t *core.T) {
	symbol := any(Init)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Init_Bad", "Init")
}

func TestAX7_Init_Ugly(t *core.T) {
	symbol := any(Init)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Init_Ugly", "Init")
}

func TestAX7_InternalModel_ApplyLoRA_Good(t *core.T) {
	symbol := any((*deviceInternalModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ApplyLoRA_Good", "InternalModel_ApplyLoRA")
}

func TestAX7_InternalModel_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ApplyLoRA_Bad", "InternalModel_ApplyLoRA")
}

func TestAX7_InternalModel_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ApplyLoRA_Ugly", "InternalModel_ApplyLoRA")
}

func TestAX7_InternalModel_Forward_Good(t *core.T) {
	symbol := any((*deviceInternalModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Forward_Good", "InternalModel_Forward")
}

func TestAX7_InternalModel_Forward_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Forward_Bad", "InternalModel_Forward")
}

func TestAX7_InternalModel_Forward_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Forward_Ugly", "InternalModel_Forward")
}

func TestAX7_InternalModel_ForwardMasked_Good(t *core.T) {
	symbol := any((*deviceInternalModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ForwardMasked_Good", "InternalModel_ForwardMasked")
}

func TestAX7_InternalModel_ForwardMasked_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ForwardMasked_Bad", "InternalModel_ForwardMasked")
}

func TestAX7_InternalModel_ForwardMasked_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ForwardMasked_Ugly", "InternalModel_ForwardMasked")
}

func TestAX7_InternalModel_ModelType_Good(t *core.T) {
	symbol := any((*deviceInternalModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ModelType_Good", "InternalModel_ModelType")
}

func TestAX7_InternalModel_ModelType_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ModelType_Bad", "InternalModel_ModelType")
}

func TestAX7_InternalModel_ModelType_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_ModelType_Ugly", "InternalModel_ModelType")
}

func TestAX7_InternalModel_NewCache_Good(t *core.T) {
	symbol := any((*deviceInternalModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NewCache_Good", "InternalModel_NewCache")
}

func TestAX7_InternalModel_NewCache_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NewCache_Bad", "InternalModel_NewCache")
}

func TestAX7_InternalModel_NewCache_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NewCache_Ugly", "InternalModel_NewCache")
}

func TestAX7_InternalModel_NumLayers_Good(t *core.T) {
	symbol := any((*deviceInternalModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NumLayers_Good", "InternalModel_NumLayers")
}

func TestAX7_InternalModel_NumLayers_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NumLayers_Bad", "InternalModel_NumLayers")
}

func TestAX7_InternalModel_NumLayers_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_NumLayers_Ugly", "InternalModel_NumLayers")
}

func TestAX7_InternalModel_Tokenizer_Good(t *core.T) {
	symbol := any((*deviceInternalModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Tokenizer_Good", "InternalModel_Tokenizer")
}

func TestAX7_InternalModel_Tokenizer_Bad(t *core.T) {
	symbol := any((*deviceInternalModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Tokenizer_Bad", "InternalModel_Tokenizer")
}

func TestAX7_InternalModel_Tokenizer_Ugly(t *core.T) {
	symbol := any((*deviceInternalModel).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "InternalModel_Tokenizer_Ugly", "InternalModel_Tokenizer")
}

func TestAX7_IsNaN_Good(t *core.T) {
	symbol := any(IsNaN)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "IsNaN_Good", "IsNaN")
}

func TestAX7_IsNaN_Bad(t *core.T) {
	symbol := any(IsNaN)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "IsNaN_Bad", "IsNaN")
}

func TestAX7_IsNaN_Ugly(t *core.T) {
	symbol := any(IsNaN)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "IsNaN_Ugly", "IsNaN")
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

func TestAX7_KVCache_Detach_Good(t *core.T) {
	symbol := any((*KVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Detach_Good", "KVCache_Detach")
}

func TestAX7_KVCache_Detach_Bad(t *core.T) {
	symbol := any((*KVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Detach_Bad", "KVCache_Detach")
}

func TestAX7_KVCache_Detach_Ugly(t *core.T) {
	symbol := any((*KVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Detach_Ugly", "KVCache_Detach")
}

func TestAX7_KVCache_Len_Good(t *core.T) {
	symbol := any((*KVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Len_Good", "KVCache_Len")
}

func TestAX7_KVCache_Len_Bad(t *core.T) {
	symbol := any((*KVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Len_Bad", "KVCache_Len")
}

func TestAX7_KVCache_Len_Ugly(t *core.T) {
	symbol := any((*KVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Len_Ugly", "KVCache_Len")
}

func TestAX7_KVCache_Offset_Good(t *core.T) {
	symbol := any((*KVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Offset_Good", "KVCache_Offset")
}

func TestAX7_KVCache_Offset_Bad(t *core.T) {
	symbol := any((*KVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Offset_Bad", "KVCache_Offset")
}

func TestAX7_KVCache_Offset_Ugly(t *core.T) {
	symbol := any((*KVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Offset_Ugly", "KVCache_Offset")
}

func TestAX7_KVCache_Reset_Good(t *core.T) {
	symbol := any((*KVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Reset_Good", "KVCache_Reset")
}

func TestAX7_KVCache_Reset_Bad(t *core.T) {
	symbol := any((*KVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Reset_Bad", "KVCache_Reset")
}

func TestAX7_KVCache_Reset_Ugly(t *core.T) {
	symbol := any((*KVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Reset_Ugly", "KVCache_Reset")
}

func TestAX7_KVCache_State_Good(t *core.T) {
	symbol := any((*KVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_State_Good", "KVCache_State")
}

func TestAX7_KVCache_State_Bad(t *core.T) {
	symbol := any((*KVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_State_Bad", "KVCache_State")
}

func TestAX7_KVCache_State_Ugly(t *core.T) {
	symbol := any((*KVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_State_Ugly", "KVCache_State")
}

func TestAX7_KVCache_Update_Good(t *core.T) {
	symbol := any((*KVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Update_Good", "KVCache_Update")
}

func TestAX7_KVCache_Update_Bad(t *core.T) {
	symbol := any((*KVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Update_Bad", "KVCache_Update")
}

func TestAX7_KVCache_Update_Ugly(t *core.T) {
	symbol := any((*KVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "KVCache_Update_Ugly", "KVCache_Update")
}

func TestAX7_LayerNorm_Bad(t *core.T) {
	symbol := any(LayerNorm)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LayerNorm_Bad", "LayerNorm")
}

func TestAX7_LayerNorm_Ugly(t *core.T) {
	symbol := any(LayerNorm)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LayerNorm_Ugly", "LayerNorm")
}

func TestAX7_Linear_Forward_Good(t *core.T) {
	symbol := any((*Linear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Linear_Forward_Good", "Linear_Forward")
}

func TestAX7_Linear_Forward_Bad(t *core.T) {
	symbol := any((*Linear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Linear_Forward_Bad", "Linear_Forward")
}

func TestAX7_Linear_Forward_Ugly(t *core.T) {
	symbol := any((*Linear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Linear_Forward_Ugly", "Linear_Forward")
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

func TestAX7_LoRALinear_Forward_Good(t *core.T) {
	symbol := any((*LoRALinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_Forward_Good", "LoRALinear_Forward")
}

func TestAX7_LoRALinear_Forward_Bad(t *core.T) {
	symbol := any((*LoRALinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_Forward_Bad", "LoRALinear_Forward")
}

func TestAX7_LoRALinear_Forward_Ugly(t *core.T) {
	symbol := any((*LoRALinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_Forward_Ugly", "LoRALinear_Forward")
}

func TestAX7_LoRALinear_ParamCount_Bad(t *core.T) {
	symbol := any((*LoRALinear).ParamCount)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_ParamCount_Bad", "LoRALinear_ParamCount")
}

func TestAX7_LoRALinear_ParamCount_Ugly(t *core.T) {
	symbol := any((*LoRALinear).ParamCount)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_ParamCount_Ugly", "LoRALinear_ParamCount")
}

func TestAX7_LoRALinear_SetParams_Good(t *core.T) {
	symbol := any((*LoRALinear).SetParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_SetParams_Good", "LoRALinear_SetParams")
}

func TestAX7_LoRALinear_SetParams_Bad(t *core.T) {
	symbol := any((*LoRALinear).SetParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_SetParams_Bad", "LoRALinear_SetParams")
}

func TestAX7_LoRALinear_SetParams_Ugly(t *core.T) {
	symbol := any((*LoRALinear).SetParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_SetParams_Ugly", "LoRALinear_SetParams")
}

func TestAX7_LoRALinear_TrainableParams_Bad(t *core.T) {
	symbol := any((*LoRALinear).TrainableParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_TrainableParams_Bad", "LoRALinear_TrainableParams")
}

func TestAX7_LoRALinear_TrainableParams_Ugly(t *core.T) {
	symbol := any((*LoRALinear).TrainableParams)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoRALinear_TrainableParams_Ugly", "LoRALinear_TrainableParams")
}

func TestAX7_LoadAllGGUF_Good(t *core.T) {
	symbol := any(LoadAllGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllGGUF_Good", "LoadAllGGUF")
}

func TestAX7_LoadAllGGUF_Bad(t *core.T) {
	symbol := any(LoadAllGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllGGUF_Bad", "LoadAllGGUF")
}

func TestAX7_LoadAllGGUF_Ugly(t *core.T) {
	symbol := any(LoadAllGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllGGUF_Ugly", "LoadAllGGUF")
}

func TestAX7_LoadAllSafetensors_Good(t *core.T) {
	symbol := any(LoadAllSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensors_Good", "LoadAllSafetensors")
}

func TestAX7_LoadAllSafetensors_Bad(t *core.T) {
	symbol := any(LoadAllSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensors_Bad", "LoadAllSafetensors")
}

func TestAX7_LoadAllSafetensors_Ugly(t *core.T) {
	symbol := any(LoadAllSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensors_Ugly", "LoadAllSafetensors")
}

func TestAX7_LoadAllSafetensorsFromReader_Good(t *core.T) {
	symbol := any(LoadAllSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensorsFromReader_Good", "LoadAllSafetensorsFromReader")
}

func TestAX7_LoadAllSafetensorsFromReader_Bad(t *core.T) {
	symbol := any(LoadAllSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensorsFromReader_Bad", "LoadAllSafetensorsFromReader")
}

func TestAX7_LoadAllSafetensorsFromReader_Ugly(t *core.T) {
	symbol := any(LoadAllSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAllSafetensorsFromReader_Ugly", "LoadAllSafetensorsFromReader")
}

func TestAX7_LoadAndInit_Good(t *core.T) {
	symbol := any(LoadAndInit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAndInit_Good", "LoadAndInit")
}

func TestAX7_LoadAndInit_Bad(t *core.T) {
	symbol := any(LoadAndInit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAndInit_Bad", "LoadAndInit")
}

func TestAX7_LoadAndInit_Ugly(t *core.T) {
	symbol := any(LoadAndInit)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadAndInit_Ugly", "LoadAndInit")
}

func TestAX7_LoadGGUF_Good(t *core.T) {
	symbol := any(LoadGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGGUF_Good", "LoadGGUF")
}

func TestAX7_LoadGGUF_Bad(t *core.T) {
	symbol := any(LoadGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGGUF_Bad", "LoadGGUF")
}

func TestAX7_LoadGGUF_Ugly(t *core.T) {
	symbol := any(LoadGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGGUF_Ugly", "LoadGGUF")
}

func TestAX7_LoadGemma3_Good(t *core.T) {
	symbol := any(LoadGemma3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma3_Good", "LoadGemma3")
}

func TestAX7_LoadGemma3_Bad(t *core.T) {
	symbol := any(LoadGemma3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma3_Bad", "LoadGemma3")
}

func TestAX7_LoadGemma3_Ugly(t *core.T) {
	symbol := any(LoadGemma3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma3_Ugly", "LoadGemma3")
}

func TestAX7_LoadGemma4_Good(t *core.T) {
	symbol := any(LoadGemma4)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma4_Good", "LoadGemma4")
}

func TestAX7_LoadGemma4_Bad(t *core.T) {
	symbol := any(LoadGemma4)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma4_Bad", "LoadGemma4")
}

func TestAX7_LoadGemma4_Ugly(t *core.T) {
	symbol := any(LoadGemma4)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadGemma4_Ugly", "LoadGemma4")
}

func TestAX7_LoadQwen3_Good(t *core.T) {
	symbol := any(LoadQwen3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadQwen3_Good", "LoadQwen3")
}

func TestAX7_LoadQwen3_Bad(t *core.T) {
	symbol := any(LoadQwen3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadQwen3_Bad", "LoadQwen3")
}

func TestAX7_LoadQwen3_Ugly(t *core.T) {
	symbol := any(LoadQwen3)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadQwen3_Ugly", "LoadQwen3")
}

func TestAX7_LoadSafetensors_Good(t *core.T) {
	symbol := any(LoadSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensors_Good", "LoadSafetensors")
}

func TestAX7_LoadSafetensors_Bad(t *core.T) {
	symbol := any(LoadSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensors_Bad", "LoadSafetensors")
}

func TestAX7_LoadSafetensors_Ugly(t *core.T) {
	symbol := any(LoadSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensors_Ugly", "LoadSafetensors")
}

func TestAX7_LoadSafetensorsFromReader_Good(t *core.T) {
	symbol := any(LoadSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensorsFromReader_Good", "LoadSafetensorsFromReader")
}

func TestAX7_LoadSafetensorsFromReader_Bad(t *core.T) {
	symbol := any(LoadSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensorsFromReader_Bad", "LoadSafetensorsFromReader")
}

func TestAX7_LoadSafetensorsFromReader_Ugly(t *core.T) {
	symbol := any(LoadSafetensorsFromReader)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadSafetensorsFromReader_Ugly", "LoadSafetensorsFromReader")
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

func TestAX7_Log_Good(t *core.T) {
	symbol := any(Log)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Log_Good", "Log")
}

func TestAX7_Log_Bad(t *core.T) {
	symbol := any(Log)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Log_Bad", "Log")
}

func TestAX7_Log_Ugly(t *core.T) {
	symbol := any(Log)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Log_Ugly", "Log")
}

func TestAX7_LogSumExp_Bad(t *core.T) {
	symbol := any(LogSumExp)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LogSumExp_Bad", "LogSumExp")
}

func TestAX7_LogSumExp_Ugly(t *core.T) {
	symbol := any(LogSumExp)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LogSumExp_Ugly", "LogSumExp")
}

func TestAX7_MSELoss_Bad(t *core.T) {
	symbol := any(MSELoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MSELoss_Bad", "MSELoss")
}

func TestAX7_MSELoss_Ugly(t *core.T) {
	symbol := any(MSELoss)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MSELoss_Ugly", "MSELoss")
}

func TestAX7_MapGet_Good(t *core.T) {
	symbol := any(MapGet)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MapGet_Good", "MapGet")
}

func TestAX7_MapGet_Bad(t *core.T) {
	symbol := any(MapGet)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MapGet_Bad", "MapGet")
}

func TestAX7_MapGet_Ugly(t *core.T) {
	symbol := any(MapGet)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MapGet_Ugly", "MapGet")
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

func TestAX7_MaterializeAsync_Good(t *core.T) {
	symbol := any(MaterializeAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaterializeAsync_Good", "MaterializeAsync")
}

func TestAX7_MaterializeAsync_Bad(t *core.T) {
	symbol := any(MaterializeAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaterializeAsync_Bad", "MaterializeAsync")
}

func TestAX7_MaterializeAsync_Ugly(t *core.T) {
	symbol := any(MaterializeAsync)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaterializeAsync_Ugly", "MaterializeAsync")
}

func TestAX7_Matmul_Bad(t *core.T) {
	symbol := any(Matmul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Matmul_Bad", "Matmul")
}

func TestAX7_Matmul_Ugly(t *core.T) {
	symbol := any(Matmul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Matmul_Ugly", "Matmul")
}

func TestAX7_MaxAxis_Bad(t *core.T) {
	symbol := any(MaxAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaxAxis_Bad", "MaxAxis")
}

func TestAX7_MaxAxis_Ugly(t *core.T) {
	symbol := any(MaxAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MaxAxis_Ugly", "MaxAxis")
}

func TestAX7_Maximum_Bad(t *core.T) {
	symbol := any(Maximum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Maximum_Bad", "Maximum")
}

func TestAX7_Maximum_Ugly(t *core.T) {
	symbol := any(Maximum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Maximum_Ugly", "Maximum")
}

func TestAX7_Mean_Bad(t *core.T) {
	symbol := any(Mean)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Mean_Bad", "Mean")
}

func TestAX7_Mean_Ugly(t *core.T) {
	symbol := any(Mean)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Mean_Ugly", "Mean")
}

func TestAX7_MeanAll_Bad(t *core.T) {
	symbol := any(MeanAll)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MeanAll_Bad", "MeanAll")
}

func TestAX7_MeanAll_Ugly(t *core.T) {
	symbol := any(MeanAll)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MeanAll_Ugly", "MeanAll")
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

func TestAX7_MetalKernelConfig_AddOutputArg_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).AddOutputArg)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddOutputArg_Good", "MetalKernelConfig_AddOutputArg")
}

func TestAX7_MetalKernelConfig_AddOutputArg_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).AddOutputArg)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddOutputArg_Bad", "MetalKernelConfig_AddOutputArg")
}

func TestAX7_MetalKernelConfig_AddOutputArg_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).AddOutputArg)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddOutputArg_Ugly", "MetalKernelConfig_AddOutputArg")
}

func TestAX7_MetalKernelConfig_AddTemplateBool_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateBool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateBool_Good", "MetalKernelConfig_AddTemplateBool")
}

func TestAX7_MetalKernelConfig_AddTemplateBool_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateBool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateBool_Bad", "MetalKernelConfig_AddTemplateBool")
}

func TestAX7_MetalKernelConfig_AddTemplateBool_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateBool)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateBool_Ugly", "MetalKernelConfig_AddTemplateBool")
}

func TestAX7_MetalKernelConfig_AddTemplateDType_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateDType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateDType_Good", "MetalKernelConfig_AddTemplateDType")
}

func TestAX7_MetalKernelConfig_AddTemplateDType_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateDType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateDType_Bad", "MetalKernelConfig_AddTemplateDType")
}

func TestAX7_MetalKernelConfig_AddTemplateDType_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateDType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateDType_Ugly", "MetalKernelConfig_AddTemplateDType")
}

func TestAX7_MetalKernelConfig_AddTemplateInt_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateInt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateInt_Good", "MetalKernelConfig_AddTemplateInt")
}

func TestAX7_MetalKernelConfig_AddTemplateInt_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateInt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateInt_Bad", "MetalKernelConfig_AddTemplateInt")
}

func TestAX7_MetalKernelConfig_AddTemplateInt_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).AddTemplateInt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_AddTemplateInt_Ugly", "MetalKernelConfig_AddTemplateInt")
}

func TestAX7_MetalKernelConfig_Free_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_Free_Good", "MetalKernelConfig_Free")
}

func TestAX7_MetalKernelConfig_Free_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_Free_Bad", "MetalKernelConfig_Free")
}

func TestAX7_MetalKernelConfig_Free_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_Free_Ugly", "MetalKernelConfig_Free")
}

func TestAX7_MetalKernelConfig_SetGrid_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).SetGrid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetGrid_Good", "MetalKernelConfig_SetGrid")
}

func TestAX7_MetalKernelConfig_SetGrid_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).SetGrid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetGrid_Bad", "MetalKernelConfig_SetGrid")
}

func TestAX7_MetalKernelConfig_SetGrid_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).SetGrid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetGrid_Ugly", "MetalKernelConfig_SetGrid")
}

func TestAX7_MetalKernelConfig_SetInitValue_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).SetInitValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetInitValue_Good", "MetalKernelConfig_SetInitValue")
}

func TestAX7_MetalKernelConfig_SetInitValue_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).SetInitValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetInitValue_Bad", "MetalKernelConfig_SetInitValue")
}

func TestAX7_MetalKernelConfig_SetInitValue_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).SetInitValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetInitValue_Ugly", "MetalKernelConfig_SetInitValue")
}

func TestAX7_MetalKernelConfig_SetThreadGroup_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).SetThreadGroup)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetThreadGroup_Good", "MetalKernelConfig_SetThreadGroup")
}

func TestAX7_MetalKernelConfig_SetThreadGroup_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).SetThreadGroup)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetThreadGroup_Bad", "MetalKernelConfig_SetThreadGroup")
}

func TestAX7_MetalKernelConfig_SetThreadGroup_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).SetThreadGroup)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetThreadGroup_Ugly", "MetalKernelConfig_SetThreadGroup")
}

func TestAX7_MetalKernelConfig_SetVerbose_Good(t *core.T) {
	symbol := any((*MetalKernelConfig).SetVerbose)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetVerbose_Good", "MetalKernelConfig_SetVerbose")
}

func TestAX7_MetalKernelConfig_SetVerbose_Bad(t *core.T) {
	symbol := any((*MetalKernelConfig).SetVerbose)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetVerbose_Bad", "MetalKernelConfig_SetVerbose")
}

func TestAX7_MetalKernelConfig_SetVerbose_Ugly(t *core.T) {
	symbol := any((*MetalKernelConfig).SetVerbose)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernelConfig_SetVerbose_Ugly", "MetalKernelConfig_SetVerbose")
}

func TestAX7_MetalKernel_Apply_Good(t *core.T) {
	symbol := any((*MetalKernel).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Apply_Good", "MetalKernel_Apply")
}

func TestAX7_MetalKernel_Apply_Bad(t *core.T) {
	symbol := any((*MetalKernel).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Apply_Bad", "MetalKernel_Apply")
}

func TestAX7_MetalKernel_Apply_Ugly(t *core.T) {
	symbol := any((*MetalKernel).Apply)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Apply_Ugly", "MetalKernel_Apply")
}

func TestAX7_MetalKernel_Free_Good(t *core.T) {
	symbol := any((*MetalKernel).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Free_Good", "MetalKernel_Free")
}

func TestAX7_MetalKernel_Free_Bad(t *core.T) {
	symbol := any((*MetalKernel).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Free_Bad", "MetalKernel_Free")
}

func TestAX7_MetalKernel_Free_Ugly(t *core.T) {
	symbol := any((*MetalKernel).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MetalKernel_Free_Ugly", "MetalKernel_Free")
}

func TestAX7_MinPSampler_Sample_Good(t *core.T) {
	symbol := any(MinPSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MinPSampler_Sample_Good", "MinPSampler_Sample")
}

func TestAX7_MinPSampler_Sample_Bad(t *core.T) {
	symbol := any(MinPSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MinPSampler_Sample_Bad", "MinPSampler_Sample")
}

func TestAX7_MinPSampler_Sample_Ugly(t *core.T) {
	symbol := any(MinPSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MinPSampler_Sample_Ugly", "MinPSampler_Sample")
}

func TestAX7_Minimum_Bad(t *core.T) {
	symbol := any(Minimum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Minimum_Bad", "Minimum")
}

func TestAX7_Minimum_Ugly(t *core.T) {
	symbol := any(Minimum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Minimum_Ugly", "Minimum")
}

func TestAX7_Model_ApplyLoRA_Good(t *core.T) {
	symbol := any((*Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ApplyLoRA_Good", "Model_ApplyLoRA")
}

func TestAX7_Model_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ApplyLoRA_Bad", "Model_ApplyLoRA")
}

func TestAX7_Model_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ApplyLoRA_Ugly", "Model_ApplyLoRA")
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

func TestAX7_Model_Decode_Good(t *core.T) {
	symbol := any((*Model).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Decode_Good", "Model_Decode")
}

func TestAX7_Model_Decode_Bad(t *core.T) {
	symbol := any((*Model).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Decode_Bad", "Model_Decode")
}

func TestAX7_Model_Decode_Ugly(t *core.T) {
	symbol := any((*Model).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Decode_Ugly", "Model_Decode")
}

func TestAX7_Model_Encode_Good(t *core.T) {
	symbol := any((*Model).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Encode_Good", "Model_Encode")
}

func TestAX7_Model_Encode_Bad(t *core.T) {
	symbol := any((*Model).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Encode_Bad", "Model_Encode")
}

func TestAX7_Model_Encode_Ugly(t *core.T) {
	symbol := any((*Model).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Encode_Ugly", "Model_Encode")
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

func TestAX7_Model_Internal_Good(t *core.T) {
	symbol := any((*Model).Internal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Internal_Good", "Model_Internal")
}

func TestAX7_Model_Internal_Bad(t *core.T) {
	symbol := any((*Model).Internal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Internal_Bad", "Model_Internal")
}

func TestAX7_Model_Internal_Ugly(t *core.T) {
	symbol := any((*Model).Internal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Internal_Ugly", "Model_Internal")
}

func TestAX7_Model_LastMetrics_Good(t *core.T) {
	symbol := any((*Model).LastMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_LastMetrics_Good", "Model_LastMetrics")
}

func TestAX7_Model_LastMetrics_Bad(t *core.T) {
	symbol := any((*Model).LastMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_LastMetrics_Bad", "Model_LastMetrics")
}

func TestAX7_Model_LastMetrics_Ugly(t *core.T) {
	symbol := any((*Model).LastMetrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_LastMetrics_Ugly", "Model_LastMetrics")
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

func TestAX7_Model_NumLayers_Good(t *core.T) {
	symbol := any((*Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_NumLayers_Good", "Model_NumLayers")
}

func TestAX7_Model_NumLayers_Bad(t *core.T) {
	symbol := any((*Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_NumLayers_Bad", "Model_NumLayers")
}

func TestAX7_Model_NumLayers_Ugly(t *core.T) {
	symbol := any((*Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_NumLayers_Ugly", "Model_NumLayers")
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

func TestAX7_MulScalar_Bad(t *core.T) {
	symbol := any(MulScalar)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MulScalar_Bad", "MulScalar")
}

func TestAX7_MulScalar_Ugly(t *core.T) {
	symbol := any(MulScalar)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "MulScalar_Ugly", "MulScalar")
}

func TestAX7_Negative_Ugly(t *core.T) {
	symbol := any(Negative)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Negative_Ugly", "Negative")
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

func TestAX7_NewClosure_Good(t *core.T) {
	symbol := any(NewClosure)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosure_Good", "NewClosure")
}

func TestAX7_NewClosure_Bad(t *core.T) {
	symbol := any(NewClosure)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosure_Bad", "NewClosure")
}

func TestAX7_NewClosure_Ugly(t *core.T) {
	symbol := any(NewClosure)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosure_Ugly", "NewClosure")
}

func TestAX7_NewClosureKwargs_Good(t *core.T) {
	symbol := any(NewClosureKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosureKwargs_Good", "NewClosureKwargs")
}

func TestAX7_NewClosureKwargs_Bad(t *core.T) {
	symbol := any(NewClosureKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosureKwargs_Bad", "NewClosureKwargs")
}

func TestAX7_NewClosureKwargs_Ugly(t *core.T) {
	symbol := any(NewClosureKwargs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewClosureKwargs_Ugly", "NewClosureKwargs")
}

func TestAX7_NewKVCache_Good(t *core.T) {
	symbol := any(NewKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewKVCache_Good", "NewKVCache")
}

func TestAX7_NewKVCache_Bad(t *core.T) {
	symbol := any(NewKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewKVCache_Bad", "NewKVCache")
}

func TestAX7_NewKVCache_Ugly(t *core.T) {
	symbol := any(NewKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewKVCache_Ugly", "NewKVCache")
}

func TestAX7_NewLinear_Good(t *core.T) {
	symbol := any(NewLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLinear_Good", "NewLinear")
}

func TestAX7_NewLinear_Bad(t *core.T) {
	symbol := any(NewLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLinear_Bad", "NewLinear")
}

func TestAX7_NewLinear_Ugly(t *core.T) {
	symbol := any(NewLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLinear_Ugly", "NewLinear")
}

func TestAX7_NewLoRALinear_Bad(t *core.T) {
	symbol := any(NewLoRALinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLoRALinear_Bad", "NewLoRALinear")
}

func TestAX7_NewLoRALinear_Ugly(t *core.T) {
	symbol := any(NewLoRALinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewLoRALinear_Ugly", "NewLoRALinear")
}

func TestAX7_NewMetalKernel_Good(t *core.T) {
	symbol := any(NewMetalKernel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernel_Good", "NewMetalKernel")
}

func TestAX7_NewMetalKernel_Bad(t *core.T) {
	symbol := any(NewMetalKernel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernel_Bad", "NewMetalKernel")
}

func TestAX7_NewMetalKernel_Ugly(t *core.T) {
	symbol := any(NewMetalKernel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernel_Ugly", "NewMetalKernel")
}

func TestAX7_NewMetalKernelConfig_Good(t *core.T) {
	symbol := any(NewMetalKernelConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernelConfig_Good", "NewMetalKernelConfig")
}

func TestAX7_NewMetalKernelConfig_Bad(t *core.T) {
	symbol := any(NewMetalKernelConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernelConfig_Bad", "NewMetalKernelConfig")
}

func TestAX7_NewMetalKernelConfig_Ugly(t *core.T) {
	symbol := any(NewMetalKernelConfig)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewMetalKernelConfig_Ugly", "NewMetalKernelConfig")
}

func TestAX7_NewQuantizedLinear_Good(t *core.T) {
	symbol := any(NewQuantizedLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedLinear_Good", "NewQuantizedLinear")
}

func TestAX7_NewQuantizedLinear_Bad(t *core.T) {
	symbol := any(NewQuantizedLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedLinear_Bad", "NewQuantizedLinear")
}

func TestAX7_NewQuantizedLinear_Ugly(t *core.T) {
	symbol := any(NewQuantizedLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedLinear_Ugly", "NewQuantizedLinear")
}

func TestAX7_NewQuantizedSwitchLinear_Good(t *core.T) {
	symbol := any(NewQuantizedSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedSwitchLinear_Good", "NewQuantizedSwitchLinear")
}

func TestAX7_NewQuantizedSwitchLinear_Bad(t *core.T) {
	symbol := any(NewQuantizedSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedSwitchLinear_Bad", "NewQuantizedSwitchLinear")
}

func TestAX7_NewQuantizedSwitchLinear_Ugly(t *core.T) {
	symbol := any(NewQuantizedSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewQuantizedSwitchLinear_Ugly", "NewQuantizedSwitchLinear")
}

func TestAX7_NewRotatingKVCache_Good(t *core.T) {
	symbol := any(NewRotatingKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRotatingKVCache_Good", "NewRotatingKVCache")
}

func TestAX7_NewRotatingKVCache_Bad(t *core.T) {
	symbol := any(NewRotatingKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRotatingKVCache_Bad", "NewRotatingKVCache")
}

func TestAX7_NewRotatingKVCache_Ugly(t *core.T) {
	symbol := any(NewRotatingKVCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRotatingKVCache_Ugly", "NewRotatingKVCache")
}

func TestAX7_NewSwitchLinear_Good(t *core.T) {
	symbol := any(NewSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSwitchLinear_Good", "NewSwitchLinear")
}

func TestAX7_NewSwitchLinear_Bad(t *core.T) {
	symbol := any(NewSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSwitchLinear_Bad", "NewSwitchLinear")
}

func TestAX7_NewSwitchLinear_Ugly(t *core.T) {
	symbol := any(NewSwitchLinear)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewSwitchLinear_Ugly", "NewSwitchLinear")
}

func TestAX7_NewVectorArray_Good(t *core.T) {
	symbol := any(NewVectorArray)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArray_Good", "NewVectorArray")
}

func TestAX7_NewVectorArray_Bad(t *core.T) {
	symbol := any(NewVectorArray)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArray_Bad", "NewVectorArray")
}

func TestAX7_NewVectorArray_Ugly(t *core.T) {
	symbol := any(NewVectorArray)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArray_Ugly", "NewVectorArray")
}

func TestAX7_NewVectorArrayFromValue_Good(t *core.T) {
	symbol := any(NewVectorArrayFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArrayFromValue_Good", "NewVectorArrayFromValue")
}

func TestAX7_NewVectorArrayFromValue_Bad(t *core.T) {
	symbol := any(NewVectorArrayFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArrayFromValue_Bad", "NewVectorArrayFromValue")
}

func TestAX7_NewVectorArrayFromValue_Ugly(t *core.T) {
	symbol := any(NewVectorArrayFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorArrayFromValue_Ugly", "NewVectorArrayFromValue")
}

func TestAX7_NewVectorString_Good(t *core.T) {
	symbol := any(NewVectorString)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorString_Good", "NewVectorString")
}

func TestAX7_NewVectorString_Bad(t *core.T) {
	symbol := any(NewVectorString)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorString_Bad", "NewVectorString")
}

func TestAX7_NewVectorString_Ugly(t *core.T) {
	symbol := any(NewVectorString)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorString_Ugly", "NewVectorString")
}

func TestAX7_NewVectorStringFromSlice_Good(t *core.T) {
	symbol := any(NewVectorStringFromSlice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromSlice_Good", "NewVectorStringFromSlice")
}

func TestAX7_NewVectorStringFromSlice_Bad(t *core.T) {
	symbol := any(NewVectorStringFromSlice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromSlice_Bad", "NewVectorStringFromSlice")
}

func TestAX7_NewVectorStringFromSlice_Ugly(t *core.T) {
	symbol := any(NewVectorStringFromSlice)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromSlice_Ugly", "NewVectorStringFromSlice")
}

func TestAX7_NewVectorStringFromValue_Good(t *core.T) {
	symbol := any(NewVectorStringFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromValue_Good", "NewVectorStringFromValue")
}

func TestAX7_NewVectorStringFromValue_Bad(t *core.T) {
	symbol := any(NewVectorStringFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromValue_Bad", "NewVectorStringFromValue")
}

func TestAX7_NewVectorStringFromValue_Ugly(t *core.T) {
	symbol := any(NewVectorStringFromValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewVectorStringFromValue_Ugly", "NewVectorStringFromValue")
}

func TestAX7_OnesLike_Bad(t *core.T) {
	symbol := any(OnesLike)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "OnesLike_Bad", "OnesLike")
}

func TestAX7_OnesLike_Ugly(t *core.T) {
	symbol := any(OnesLike)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "OnesLike_Ugly", "OnesLike")
}

func TestAX7_Power_Bad(t *core.T) {
	symbol := any(Power)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Power_Bad", "Power")
}

func TestAX7_Power_Ugly(t *core.T) {
	symbol := any(Power)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Power_Ugly", "Power")
}

func TestAX7_PutAlongAxis_Good(t *core.T) {
	symbol := any(PutAlongAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PutAlongAxis_Good", "PutAlongAxis")
}

func TestAX7_PutAlongAxis_Bad(t *core.T) {
	symbol := any(PutAlongAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PutAlongAxis_Bad", "PutAlongAxis")
}

func TestAX7_PutAlongAxis_Ugly(t *core.T) {
	symbol := any(PutAlongAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "PutAlongAxis_Ugly", "PutAlongAxis")
}

func TestAX7_QuantizedMatmul_Good(t *core.T) {
	symbol := any(QuantizedMatmul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "QuantizedMatmul_Good", "QuantizedMatmul")
}

func TestAX7_QuantizedMatmul_Bad(t *core.T) {
	symbol := any(QuantizedMatmul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "QuantizedMatmul_Bad", "QuantizedMatmul")
}

func TestAX7_QuantizedMatmul_Ugly(t *core.T) {
	symbol := any(QuantizedMatmul)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "QuantizedMatmul_Ugly", "QuantizedMatmul")
}

func TestAX7_Qwen3Model_ApplyLoRA_Good(t *core.T) {
	symbol := any((*Qwen3Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ApplyLoRA_Good", "Qwen3Model_ApplyLoRA")
}

func TestAX7_Qwen3Model_ApplyLoRA_Bad(t *core.T) {
	symbol := any((*Qwen3Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ApplyLoRA_Bad", "Qwen3Model_ApplyLoRA")
}

func TestAX7_Qwen3Model_ApplyLoRA_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).ApplyLoRA)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ApplyLoRA_Ugly", "Qwen3Model_ApplyLoRA")
}

func TestAX7_Qwen3Model_Forward_Good(t *core.T) {
	symbol := any((*Qwen3Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Forward_Good", "Qwen3Model_Forward")
}

func TestAX7_Qwen3Model_Forward_Bad(t *core.T) {
	symbol := any((*Qwen3Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Forward_Bad", "Qwen3Model_Forward")
}

func TestAX7_Qwen3Model_Forward_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Forward_Ugly", "Qwen3Model_Forward")
}

func TestAX7_Qwen3Model_ForwardMasked_Good(t *core.T) {
	symbol := any((*Qwen3Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ForwardMasked_Good", "Qwen3Model_ForwardMasked")
}

func TestAX7_Qwen3Model_ForwardMasked_Bad(t *core.T) {
	symbol := any((*Qwen3Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ForwardMasked_Bad", "Qwen3Model_ForwardMasked")
}

func TestAX7_Qwen3Model_ForwardMasked_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).ForwardMasked)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ForwardMasked_Ugly", "Qwen3Model_ForwardMasked")
}

func TestAX7_Qwen3Model_ModelType_Good(t *core.T) {
	symbol := any((*Qwen3Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ModelType_Good", "Qwen3Model_ModelType")
}

func TestAX7_Qwen3Model_ModelType_Bad(t *core.T) {
	symbol := any((*Qwen3Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ModelType_Bad", "Qwen3Model_ModelType")
}

func TestAX7_Qwen3Model_ModelType_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_ModelType_Ugly", "Qwen3Model_ModelType")
}

func TestAX7_Qwen3Model_NewCache_Good(t *core.T) {
	symbol := any((*Qwen3Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NewCache_Good", "Qwen3Model_NewCache")
}

func TestAX7_Qwen3Model_NewCache_Bad(t *core.T) {
	symbol := any((*Qwen3Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NewCache_Bad", "Qwen3Model_NewCache")
}

func TestAX7_Qwen3Model_NewCache_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).NewCache)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NewCache_Ugly", "Qwen3Model_NewCache")
}

func TestAX7_Qwen3Model_NumLayers_Good(t *core.T) {
	symbol := any((*Qwen3Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NumLayers_Good", "Qwen3Model_NumLayers")
}

func TestAX7_Qwen3Model_NumLayers_Bad(t *core.T) {
	symbol := any((*Qwen3Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NumLayers_Bad", "Qwen3Model_NumLayers")
}

func TestAX7_Qwen3Model_NumLayers_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).NumLayers)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_NumLayers_Ugly", "Qwen3Model_NumLayers")
}

func TestAX7_Qwen3Model_Tokenizer_Good(t *core.T) {
	symbol := any((*Qwen3Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Tokenizer_Good", "Qwen3Model_Tokenizer")
}

func TestAX7_Qwen3Model_Tokenizer_Bad(t *core.T) {
	symbol := any((*Qwen3Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Tokenizer_Bad", "Qwen3Model_Tokenizer")
}

func TestAX7_Qwen3Model_Tokenizer_Ugly(t *core.T) {
	symbol := any((*Qwen3Model).Tokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Qwen3Model_Tokenizer_Ugly", "Qwen3Model_Tokenizer")
}

func TestAX7_RMSNorm_Bad(t *core.T) {
	symbol := any(RMSNorm)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNorm_Bad", "RMSNorm")
}

func TestAX7_RMSNorm_Ugly(t *core.T) {
	symbol := any(RMSNorm)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNorm_Ugly", "RMSNorm")
}

func TestAX7_RMSNormModule_Forward_Good(t *core.T) {
	symbol := any((*RMSNormModule).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormModule_Forward_Good", "RMSNormModule_Forward")
}

func TestAX7_RMSNormModule_Forward_Bad(t *core.T) {
	symbol := any((*RMSNormModule).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormModule_Forward_Bad", "RMSNormModule_Forward")
}

func TestAX7_RMSNormModule_Forward_Ugly(t *core.T) {
	symbol := any((*RMSNormModule).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormModule_Forward_Ugly", "RMSNormModule_Forward")
}

func TestAX7_RMSNormNoScale_Good(t *core.T) {
	symbol := any(RMSNormNoScale)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormNoScale_Good", "RMSNormNoScale")
}

func TestAX7_RMSNormNoScale_Bad(t *core.T) {
	symbol := any(RMSNormNoScale)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormNoScale_Bad", "RMSNormNoScale")
}

func TestAX7_RMSNormNoScale_Ugly(t *core.T) {
	symbol := any(RMSNormNoScale)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RMSNormNoScale_Ugly", "RMSNormNoScale")
}

func TestAX7_RandomCategorical_Bad(t *core.T) {
	symbol := any(RandomCategorical)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomCategorical_Bad", "RandomCategorical")
}

func TestAX7_RandomCategorical_Ugly(t *core.T) {
	symbol := any(RandomCategorical)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomCategorical_Ugly", "RandomCategorical")
}

func TestAX7_RandomNormal_Bad(t *core.T) {
	symbol := any(RandomNormal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomNormal_Bad", "RandomNormal")
}

func TestAX7_RandomNormal_Ugly(t *core.T) {
	symbol := any(RandomNormal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomNormal_Ugly", "RandomNormal")
}

func TestAX7_RandomUniform_Bad(t *core.T) {
	symbol := any(RandomUniform)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomUniform_Bad", "RandomUniform")
}

func TestAX7_RandomUniform_Ugly(t *core.T) {
	symbol := any(RandomUniform)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RandomUniform_Ugly", "RandomUniform")
}

func TestAX7_Reciprocal_Bad(t *core.T) {
	symbol := any(Reciprocal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Reciprocal_Bad", "Reciprocal")
}

func TestAX7_Reciprocal_Ugly(t *core.T) {
	symbol := any(Reciprocal)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Reciprocal_Ugly", "Reciprocal")
}

func TestAX7_RepeatKV_Good(t *core.T) {
	symbol := any(RepeatKV)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RepeatKV_Good", "RepeatKV")
}

func TestAX7_RepeatKV_Bad(t *core.T) {
	symbol := any(RepeatKV)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RepeatKV_Bad", "RepeatKV")
}

func TestAX7_RepeatKV_Ugly(t *core.T) {
	symbol := any(RepeatKV)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RepeatKV_Ugly", "RepeatKV")
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

func TestAX7_RoPE_Bad(t *core.T) {
	symbol := any(RoPE)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RoPE_Bad", "RoPE")
}

func TestAX7_RoPE_Ugly(t *core.T) {
	symbol := any(RoPE)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RoPE_Ugly", "RoPE")
}

func TestAX7_RoPEWithFreqs_Good(t *core.T) {
	symbol := any(RoPEWithFreqs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RoPEWithFreqs_Good", "RoPEWithFreqs")
}

func TestAX7_RoPEWithFreqs_Bad(t *core.T) {
	symbol := any(RoPEWithFreqs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RoPEWithFreqs_Bad", "RoPEWithFreqs")
}

func TestAX7_RoPEWithFreqs_Ugly(t *core.T) {
	symbol := any(RoPEWithFreqs)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RoPEWithFreqs_Ugly", "RoPEWithFreqs")
}

func TestAX7_RotatingKVCache_Detach_Good(t *core.T) {
	symbol := any((*RotatingKVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Detach_Good", "RotatingKVCache_Detach")
}

func TestAX7_RotatingKVCache_Detach_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Detach_Bad", "RotatingKVCache_Detach")
}

func TestAX7_RotatingKVCache_Detach_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).Detach)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Detach_Ugly", "RotatingKVCache_Detach")
}

func TestAX7_RotatingKVCache_Len_Good(t *core.T) {
	symbol := any((*RotatingKVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Len_Good", "RotatingKVCache_Len")
}

func TestAX7_RotatingKVCache_Len_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Len_Bad", "RotatingKVCache_Len")
}

func TestAX7_RotatingKVCache_Len_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).Len)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Len_Ugly", "RotatingKVCache_Len")
}

func TestAX7_RotatingKVCache_Offset_Good(t *core.T) {
	symbol := any((*RotatingKVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Offset_Good", "RotatingKVCache_Offset")
}

func TestAX7_RotatingKVCache_Offset_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Offset_Bad", "RotatingKVCache_Offset")
}

func TestAX7_RotatingKVCache_Offset_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).Offset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Offset_Ugly", "RotatingKVCache_Offset")
}

func TestAX7_RotatingKVCache_Reset_Good(t *core.T) {
	symbol := any((*RotatingKVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Reset_Good", "RotatingKVCache_Reset")
}

func TestAX7_RotatingKVCache_Reset_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Reset_Bad", "RotatingKVCache_Reset")
}

func TestAX7_RotatingKVCache_Reset_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).Reset)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Reset_Ugly", "RotatingKVCache_Reset")
}

func TestAX7_RotatingKVCache_State_Good(t *core.T) {
	symbol := any((*RotatingKVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_State_Good", "RotatingKVCache_State")
}

func TestAX7_RotatingKVCache_State_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_State_Bad", "RotatingKVCache_State")
}

func TestAX7_RotatingKVCache_State_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).State)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_State_Ugly", "RotatingKVCache_State")
}

func TestAX7_RotatingKVCache_Update_Good(t *core.T) {
	symbol := any((*RotatingKVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Update_Good", "RotatingKVCache_Update")
}

func TestAX7_RotatingKVCache_Update_Bad(t *core.T) {
	symbol := any((*RotatingKVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Update_Bad", "RotatingKVCache_Update")
}

func TestAX7_RotatingKVCache_Update_Ugly(t *core.T) {
	symbol := any((*RotatingKVCache).Update)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RotatingKVCache_Update_Ugly", "RotatingKVCache_Update")
}

func TestAX7_Rsqrt_Bad(t *core.T) {
	symbol := any(Rsqrt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Rsqrt_Bad", "Rsqrt")
}

func TestAX7_Rsqrt_Ugly(t *core.T) {
	symbol := any(Rsqrt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Rsqrt_Ugly", "Rsqrt")
}

func TestAX7_RuntimeGC_Good(t *core.T) {
	symbol := any(RuntimeGC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RuntimeGC_Good", "RuntimeGC")
}

func TestAX7_RuntimeGC_Bad(t *core.T) {
	symbol := any(RuntimeGC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RuntimeGC_Bad", "RuntimeGC")
}

func TestAX7_RuntimeGC_Ugly(t *core.T) {
	symbol := any(RuntimeGC)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "RuntimeGC_Ugly", "RuntimeGC")
}

func TestAX7_SaveGGUF_Good(t *core.T) {
	symbol := any(SaveGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveGGUF_Good", "SaveGGUF")
}

func TestAX7_SaveGGUF_Bad(t *core.T) {
	symbol := any(SaveGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveGGUF_Bad", "SaveGGUF")
}

func TestAX7_SaveGGUF_Ugly(t *core.T) {
	symbol := any(SaveGGUF)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveGGUF_Ugly", "SaveGGUF")
}

func TestAX7_SaveSafetensors_Bad(t *core.T) {
	symbol := any(SaveSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveSafetensors_Bad", "SaveSafetensors")
}

func TestAX7_SaveSafetensors_Ugly(t *core.T) {
	symbol := any(SaveSafetensors)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveSafetensors_Ugly", "SaveSafetensors")
}

func TestAX7_SaveSafetensorsToWriter_Good(t *core.T) {
	symbol := any(SaveSafetensorsToWriter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveSafetensorsToWriter_Good", "SaveSafetensorsToWriter")
}

func TestAX7_SaveSafetensorsToWriter_Bad(t *core.T) {
	symbol := any(SaveSafetensorsToWriter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveSafetensorsToWriter_Bad", "SaveSafetensorsToWriter")
}

func TestAX7_SaveSafetensorsToWriter_Ugly(t *core.T) {
	symbol := any(SaveSafetensorsToWriter)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SaveSafetensorsToWriter_Ugly", "SaveSafetensorsToWriter")
}

func TestAX7_ScaledDotProductAttention_Good(t *core.T) {
	symbol := any(ScaledDotProductAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ScaledDotProductAttention_Good", "ScaledDotProductAttention")
}

func TestAX7_ScaledDotProductAttention_Bad(t *core.T) {
	symbol := any(ScaledDotProductAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ScaledDotProductAttention_Bad", "ScaledDotProductAttention")
}

func TestAX7_ScaledDotProductAttention_Ugly(t *core.T) {
	symbol := any(ScaledDotProductAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ScaledDotProductAttention_Ugly", "ScaledDotProductAttention")
}

func TestAX7_ScaledDotProductAttentionWithMask_Bad(t *core.T) {
	symbol := any(ScaledDotProductAttentionWithMask)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ScaledDotProductAttentionWithMask_Bad", "ScaledDotProductAttentionWithMask")
}

func TestAX7_ScaledDotProductAttentionWithMask_Ugly(t *core.T) {
	symbol := any(ScaledDotProductAttentionWithMask)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "ScaledDotProductAttentionWithMask_Ugly", "ScaledDotProductAttentionWithMask")
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

func TestAX7_SiLU_Bad(t *core.T) {
	symbol := any(SiLU)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SiLU_Bad", "SiLU")
}

func TestAX7_SiLU_Ugly(t *core.T) {
	symbol := any(SiLU)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SiLU_Ugly", "SiLU")
}

func TestAX7_Sigmoid_Bad(t *core.T) {
	symbol := any(Sigmoid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sigmoid_Bad", "Sigmoid")
}

func TestAX7_Sigmoid_Ugly(t *core.T) {
	symbol := any(Sigmoid)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sigmoid_Ugly", "Sigmoid")
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

func TestAX7_SliceAxis_Bad(t *core.T) {
	symbol := any(SliceAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SliceAxis_Bad", "SliceAxis")
}

func TestAX7_SliceAxis_Ugly(t *core.T) {
	symbol := any(SliceAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SliceAxis_Ugly", "SliceAxis")
}

func TestAX7_SliceUpdateInplace_Bad(t *core.T) {
	symbol := any(SliceUpdateInplace)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SliceUpdateInplace_Bad", "SliceUpdateInplace")
}

func TestAX7_SliceUpdateInplace_Ugly(t *core.T) {
	symbol := any(SliceUpdateInplace)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SliceUpdateInplace_Ugly", "SliceUpdateInplace")
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

func TestAX7_Sort_Bad(t *core.T) {
	symbol := any(Sort)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sort_Bad", "Sort")
}

func TestAX7_Sort_Ugly(t *core.T) {
	symbol := any(Sort)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sort_Ugly", "Sort")
}

func TestAX7_Sqrt_Bad(t *core.T) {
	symbol := any(Sqrt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sqrt_Bad", "Sqrt")
}

func TestAX7_Sqrt_Ugly(t *core.T) {
	symbol := any(Sqrt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sqrt_Ugly", "Sqrt")
}

func TestAX7_Square_Bad(t *core.T) {
	symbol := any(Square)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Square_Bad", "Square")
}

func TestAX7_Square_Ugly(t *core.T) {
	symbol := any(Square)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Square_Ugly", "Square")
}

func TestAX7_Squeeze_Bad(t *core.T) {
	symbol := any(Squeeze)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Squeeze_Bad", "Squeeze")
}

func TestAX7_Squeeze_Ugly(t *core.T) {
	symbol := any(Squeeze)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Squeeze_Ugly", "Squeeze")
}

func TestAX7_Subtract_Bad(t *core.T) {
	symbol := any(Subtract)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Subtract_Bad", "Subtract")
}

func TestAX7_Subtract_Ugly(t *core.T) {
	symbol := any(Subtract)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Subtract_Ugly", "Subtract")
}

func TestAX7_Sum_Bad(t *core.T) {
	symbol := any(Sum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sum_Bad", "Sum")
}

func TestAX7_Sum_Ugly(t *core.T) {
	symbol := any(Sum)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Sum_Ugly", "Sum")
}

func TestAX7_SumAll_Bad(t *core.T) {
	symbol := any(SumAll)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SumAll_Bad", "SumAll")
}

func TestAX7_SumAll_Ugly(t *core.T) {
	symbol := any(SumAll)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SumAll_Ugly", "SumAll")
}

func TestAX7_SwitchLinear_Forward_Good(t *core.T) {
	symbol := any((*SwitchLinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SwitchLinear_Forward_Good", "SwitchLinear_Forward")
}

func TestAX7_SwitchLinear_Forward_Bad(t *core.T) {
	symbol := any((*SwitchLinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SwitchLinear_Forward_Bad", "SwitchLinear_Forward")
}

func TestAX7_SwitchLinear_Forward_Ugly(t *core.T) {
	symbol := any((*SwitchLinear).Forward)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "SwitchLinear_Forward_Ugly", "SwitchLinear_Forward")
}

func TestAX7_Synchronize_Good(t *core.T) {
	symbol := any(Synchronize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Synchronize_Good", "Synchronize")
}

func TestAX7_Synchronize_Bad(t *core.T) {
	symbol := any(Synchronize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Synchronize_Bad", "Synchronize")
}

func TestAX7_Synchronize_Ugly(t *core.T) {
	symbol := any(Synchronize)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Synchronize_Ugly", "Synchronize")
}

func TestAX7_Take_Bad(t *core.T) {
	symbol := any(Take)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Take_Bad", "Take")
}

func TestAX7_Take_Ugly(t *core.T) {
	symbol := any(Take)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Take_Ugly", "Take")
}

func TestAX7_TakeAlongAxis_Bad(t *core.T) {
	symbol := any(TakeAlongAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TakeAlongAxis_Bad", "TakeAlongAxis")
}

func TestAX7_TakeAlongAxis_Ugly(t *core.T) {
	symbol := any(TakeAlongAxis)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TakeAlongAxis_Ugly", "TakeAlongAxis")
}

func TestAX7_Tanh_Bad(t *core.T) {
	symbol := any(Tanh)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tanh_Bad", "Tanh")
}

func TestAX7_Tanh_Ugly(t *core.T) {
	symbol := any(Tanh)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tanh_Ugly", "Tanh")
}

func TestAX7_Temperature_Sample_Good(t *core.T) {
	symbol := any(Temperature.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Temperature_Sample_Good", "Temperature_Sample")
}

func TestAX7_Temperature_Sample_Bad(t *core.T) {
	symbol := any(Temperature.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Temperature_Sample_Bad", "Temperature_Sample")
}

func TestAX7_Temperature_Sample_Ugly(t *core.T) {
	symbol := any(Temperature.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Temperature_Sample_Ugly", "Temperature_Sample")
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

func TestAX7_Tokenizer_BOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Good", "Tokenizer_BOSToken")
}

func TestAX7_Tokenizer_BOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Bad", "Tokenizer_BOSToken")
}

func TestAX7_Tokenizer_BOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Ugly", "Tokenizer_BOSToken")
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

func TestAX7_Tokenizer_DecodeToken_Good(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Good", "Tokenizer_DecodeToken")
}

func TestAX7_Tokenizer_DecodeToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Bad", "Tokenizer_DecodeToken")
}

func TestAX7_Tokenizer_DecodeToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Ugly", "Tokenizer_DecodeToken")
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

func TestAX7_Tokenizer_EOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Good", "Tokenizer_EOSToken")
}

func TestAX7_Tokenizer_EOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Bad", "Tokenizer_EOSToken")
}

func TestAX7_Tokenizer_EOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Ugly", "Tokenizer_EOSToken")
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

func TestAX7_Tokenizer_HasBOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Good", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasBOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Bad", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasBOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Ugly", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Good", "Tokenizer_HasEOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Bad", "Tokenizer_HasEOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Ugly", "Tokenizer_HasEOSToken")
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

func TestAX7_TopK_Bad(t *core.T) {
	symbol := any(TopK)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopK_Bad", "TopK")
}

func TestAX7_TopK_Ugly(t *core.T) {
	symbol := any(TopK)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopK_Ugly", "TopK")
}

func TestAX7_TopKSampler_Sample_Good(t *core.T) {
	symbol := any(TopKSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopKSampler_Sample_Good", "TopKSampler_Sample")
}

func TestAX7_TopKSampler_Sample_Bad(t *core.T) {
	symbol := any(TopKSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopKSampler_Sample_Bad", "TopKSampler_Sample")
}

func TestAX7_TopKSampler_Sample_Ugly(t *core.T) {
	symbol := any(TopKSampler.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopKSampler_Sample_Ugly", "TopKSampler_Sample")
}

func TestAX7_TopP_Sample_Good(t *core.T) {
	symbol := any(TopP.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopP_Sample_Good", "TopP_Sample")
}

func TestAX7_TopP_Sample_Bad(t *core.T) {
	symbol := any(TopP.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopP_Sample_Bad", "TopP_Sample")
}

func TestAX7_TopP_Sample_Ugly(t *core.T) {
	symbol := any(TopP.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "TopP_Sample_Ugly", "TopP_Sample")
}

func TestAX7_Transpose_Bad(t *core.T) {
	symbol := any(Transpose)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Transpose_Bad", "Transpose")
}

func TestAX7_Transpose_Ugly(t *core.T) {
	symbol := any(Transpose)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Transpose_Ugly", "Transpose")
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

func TestAX7_VectorArray_Append_Good(t *core.T) {
	symbol := any((*VectorArray).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Append_Good", "VectorArray_Append")
}

func TestAX7_VectorArray_Append_Bad(t *core.T) {
	symbol := any((*VectorArray).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Append_Bad", "VectorArray_Append")
}

func TestAX7_VectorArray_Append_Ugly(t *core.T) {
	symbol := any((*VectorArray).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Append_Ugly", "VectorArray_Append")
}

func TestAX7_VectorArray_Free_Good(t *core.T) {
	symbol := any((*VectorArray).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Free_Good", "VectorArray_Free")
}

func TestAX7_VectorArray_Free_Bad(t *core.T) {
	symbol := any((*VectorArray).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Free_Bad", "VectorArray_Free")
}

func TestAX7_VectorArray_Free_Ugly(t *core.T) {
	symbol := any((*VectorArray).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Free_Ugly", "VectorArray_Free")
}

func TestAX7_VectorArray_Get_Good(t *core.T) {
	symbol := any((*VectorArray).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Get_Good", "VectorArray_Get")
}

func TestAX7_VectorArray_Get_Bad(t *core.T) {
	symbol := any((*VectorArray).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Get_Bad", "VectorArray_Get")
}

func TestAX7_VectorArray_Get_Ugly(t *core.T) {
	symbol := any((*VectorArray).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Get_Ugly", "VectorArray_Get")
}

func TestAX7_VectorArray_SetValue_Good(t *core.T) {
	symbol := any((*VectorArray).SetValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_SetValue_Good", "VectorArray_SetValue")
}

func TestAX7_VectorArray_SetValue_Bad(t *core.T) {
	symbol := any((*VectorArray).SetValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_SetValue_Bad", "VectorArray_SetValue")
}

func TestAX7_VectorArray_SetValue_Ugly(t *core.T) {
	symbol := any((*VectorArray).SetValue)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_SetValue_Ugly", "VectorArray_SetValue")
}

func TestAX7_VectorArray_Size_Good(t *core.T) {
	symbol := any((*VectorArray).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Size_Good", "VectorArray_Size")
}

func TestAX7_VectorArray_Size_Bad(t *core.T) {
	symbol := any((*VectorArray).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Size_Bad", "VectorArray_Size")
}

func TestAX7_VectorArray_Size_Ugly(t *core.T) {
	symbol := any((*VectorArray).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorArray_Size_Ugly", "VectorArray_Size")
}

func TestAX7_VectorString_Append_Good(t *core.T) {
	symbol := any((*VectorString).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Append_Good", "VectorString_Append")
}

func TestAX7_VectorString_Append_Bad(t *core.T) {
	symbol := any((*VectorString).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Append_Bad", "VectorString_Append")
}

func TestAX7_VectorString_Append_Ugly(t *core.T) {
	symbol := any((*VectorString).Append)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Append_Ugly", "VectorString_Append")
}

func TestAX7_VectorString_Free_Good(t *core.T) {
	symbol := any((*VectorString).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Free_Good", "VectorString_Free")
}

func TestAX7_VectorString_Free_Bad(t *core.T) {
	symbol := any((*VectorString).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Free_Bad", "VectorString_Free")
}

func TestAX7_VectorString_Free_Ugly(t *core.T) {
	symbol := any((*VectorString).Free)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Free_Ugly", "VectorString_Free")
}

func TestAX7_VectorString_Get_Good(t *core.T) {
	symbol := any((*VectorString).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Get_Good", "VectorString_Get")
}

func TestAX7_VectorString_Get_Bad(t *core.T) {
	symbol := any((*VectorString).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Get_Bad", "VectorString_Get")
}

func TestAX7_VectorString_Get_Ugly(t *core.T) {
	symbol := any((*VectorString).Get)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Get_Ugly", "VectorString_Get")
}

func TestAX7_VectorString_Size_Good(t *core.T) {
	symbol := any((*VectorString).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Size_Good", "VectorString_Size")
}

func TestAX7_VectorString_Size_Bad(t *core.T) {
	symbol := any((*VectorString).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Size_Bad", "VectorString_Size")
}

func TestAX7_VectorString_Size_Ugly(t *core.T) {
	symbol := any((*VectorString).Size)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "VectorString_Size_Ugly", "VectorString_Size")
}

func TestAX7_Version_Good(t *core.T) {
	symbol := any(Version)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Version_Good", "Version")
}

func TestAX7_Version_Bad(t *core.T) {
	symbol := any(Version)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Version_Bad", "Version")
}

func TestAX7_Version_Ugly(t *core.T) {
	symbol := any(Version)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Version_Ugly", "Version")
}

func TestAX7_Where_Bad(t *core.T) {
	symbol := any(Where)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Where_Bad", "Where")
}

func TestAX7_Where_Ugly(t *core.T) {
	symbol := any(Where)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Where_Ugly", "Where")
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

func TestAX7_chain_Sample_Good(t *core.T) {
	symbol := any(chain.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "chain_Sample_Good", "chain_Sample")
}

func TestAX7_chain_Sample_Bad(t *core.T) {
	symbol := any(chain.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "chain_Sample_Bad", "chain_Sample")
}

func TestAX7_chain_Sample_Ugly(t *core.T) {
	symbol := any(chain.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "chain_Sample_Ugly", "chain_Sample")
}

func TestAX7_greedy_Sample_Good(t *core.T) {
	symbol := any(greedy.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "greedy_Sample_Good", "greedy_Sample")
}

func TestAX7_greedy_Sample_Bad(t *core.T) {
	symbol := any(greedy.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "greedy_Sample_Bad", "greedy_Sample")
}

func TestAX7_greedy_Sample_Ugly(t *core.T) {
	symbol := any(greedy.Sample)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "greedy_Sample_Ugly", "greedy_Sample")
}
