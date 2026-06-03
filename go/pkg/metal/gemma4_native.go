// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// gemma4_native.go holds the Gemma 4 runtime's fused cgo kernels (RFC.model-sdk
// Cat 3). The cgo C types (C.mlx_array, C.go_mlx_gemma4_*) are package-private to
// metal, so a pure-Go model package cannot compile cgo that uses them — the fused
// kernels therefore live here, in package metal, beside the C types.
//
// The model architecture (package gemma4) drives these kernels through request
// structs of *Array + *Linear/*SwitchLinear + scalars; no concrete Gemma4* type is
// named in this file. The architecture fills the request from its own types and
// calls the exported NativeGemma4* entry points.
package metal

/*
#include <stdlib.h>
#include "decode_bridge.h"

int go_mlx_gemma4_decode_layer(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_layer_args* args,
	const mlx_stream stream);
int go_mlx_gemma4_fixed_greedy_token(
	mlx_array* token,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_model_greedy_args* args,
	const mlx_stream stream);
int go_mlx_gemma4_fixed_owner_attention(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_fixed_attention_args* args,
	const mlx_stream stream);
int go_mlx_gemma4_fixed_owner_attention_residual(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_fixed_attention_args* args,
	const mlx_stream stream);
*/
import "C"

import (
	"runtime"
	"unsafe"

	core "dappco.re/go"
)

// SharedKV is the Gemma 4 runtime's per-layer K/V hand-off between fused kernels
// and the architecture forward pass. It carries only metal-owned arrays + page
// state + scalars (no model architecture type), so it lives in package metal where
// both the kernels (producers) and the model package (consumer) can reference it.
type SharedKV struct {
	Keys     *Array
	Values   *Array
	Pages    PagedKVState
	Offset   int
	Fixed    bool
	Borrowed bool
}

// HasState reports whether the shared K/V carries usable contiguous tensors or
// pages.
func (kv SharedKV) HasState() bool {
	return (kv.Keys != nil && kv.Keys.Valid() && kv.Values != nil && kv.Values.Valid()) || kv.HasPages()
}

// HasPages reports whether the shared K/V carries a complete paged state.
func (kv SharedKV) HasPages() bool {
	if len(kv.Pages.Keys) == 0 || len(kv.Pages.Keys) != len(kv.Pages.Values) {
		return false
	}
	for i := range kv.Pages.Keys {
		if kv.Pages.Keys[i] == nil || !kv.Pages.Keys[i].Valid() || kv.Pages.Values[i] == nil || !kv.Pages.Values[i].Valid() {
			return false
		}
	}
	return true
}

// Free releases the shared K/V handles. Borrowed states leave their (cache-owned)
// contiguous tensors alone and only free page state.
func (kv SharedKV) Free() {
	if !kv.Borrowed {
		Free(kv.Keys, kv.Values)
	}
	kv.Pages.Free()
}

// Clone deep-copies the shared K/V state.
func (kv SharedKV) Clone() SharedKV {
	out := SharedKV{
		Offset: kv.Offset,
		Fixed:  kv.Fixed,
	}
	if kv.Keys != nil && kv.Keys.Valid() {
		out.Keys = kv.Keys.Clone()
	}
	if kv.Values != nil && kv.Values.Valid() {
		out.Values = kv.Values.Clone()
	}
	out.Pages = clonePagedKVState(kv.Pages)
	return out
}

// MoveSharedKV transfers ownership of the shared K/V out of *kv, leaving it zeroed.
func MoveSharedKV(kv *SharedKV) SharedKV {
	if kv == nil {
		return SharedKV{}
	}
	out := *kv
	*kv = SharedKV{}
	return out
}

func clonePagedKVState(state PagedKVState) PagedKVState {
	out := PagedKVState{Length: state.Length}
	if len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return out
	}
	out.Keys = make([]*Array, len(state.Keys))
	out.Values = make([]*Array, len(state.Values))
	out.Owned = make([]*Array, 0, len(state.Keys)+len(state.Values))
	for i := range state.Keys {
		if state.Keys[i] != nil && state.Keys[i].Valid() {
			out.Keys[i] = state.Keys[i].Clone()
			out.Owned = append(out.Owned, out.Keys[i])
		}
		if state.Values[i] != nil && state.Values[i].Valid() {
			out.Values[i] = state.Values[i].Clone()
			out.Owned = append(out.Owned, out.Values[i])
		}
	}
	return out
}

func gemma4ValidKV(k, v *Array) bool {
	return k != nil && k.Valid() && v != nil && v.Valid()
}

func gemma4DefaultStream() C.mlx_stream {
	var s C.mlx_stream
	s.ctx = DefaultStreamHandle()
	return s
}

// Gemma4FixedAttentionRequest carries the static inputs for the fused fixed-owner
// attention kernels. KeyCache/ValueCache/Offset/Scale are resolved by the caller
// from the live FixedKVCache; the projections + norms + RoPE come from the model
// architecture.
type Gemma4FixedAttentionRequest struct {
	X        *Array
	Residual *Array // nil for the non-residual variant
	Mask     *Array

	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear

	QNorm        *Array
	KNorm        *Array
	PostAttnNorm *Array // nil for the non-residual variant
	RopeFreqs    *Array

	Scale             float32
	NumAttentionHeads int32
	NumKeyValueHeads  int32
	HeadDim           int32
	RopeDims          int32
	RopeBase          float32

	// keyCache/valueCache/offset/scaleArray are resolved by the kernel from the
	// live FixedKVCache before the C call; they are not part of the public input.
	keyCache, valueCache, offset, scaleArray *Array
}

func (req Gemma4FixedAttentionRequest) cArgs() C.go_mlx_gemma4_fixed_attention_args {
	args := C.go_mlx_gemma4_fixed_attention_args{
		x:                   cArray(req.X),
		residual:            cArray(req.Residual),
		key_cache:           cArray(req.keyCache),
		value_cache:         cArray(req.valueCache),
		offset:              cArray(req.offset),
		scale:               cArray(req.scaleArray),
		mask:                cArray(req.Mask),
		q_weight:            cArray(req.QProj.Weight),
		q_scales:            cArray(req.QProj.Scales),
		q_biases:            cArray(req.QProj.Biases),
		k_weight:            cArray(req.KProj.Weight),
		k_scales:            cArray(req.KProj.Scales),
		k_biases:            cArray(req.KProj.Biases),
		v_weight:            cArray(req.VProj.Weight),
		v_scales:            cArray(req.VProj.Scales),
		v_biases:            cArray(req.VProj.Biases),
		o_weight:            cArray(req.OProj.Weight),
		o_scales:            cArray(req.OProj.Scales),
		o_biases:            cArray(req.OProj.Biases),
		q_norm:              cArray(req.QNorm),
		k_norm:              cArray(req.KNorm),
		post_attn_norm:      cArray(req.PostAttnNorm),
		rope_freqs:          cArray(req.RopeFreqs),
		num_attention_heads: C.int(req.NumAttentionHeads),
		num_key_value_heads: C.int(req.NumKeyValueHeads),
		head_dim:            C.int(req.HeadDim),
		rope_dims:           C.int(req.RopeDims),
		rope_base:           C.float(req.RopeBase),
	}
	if req.Mask != nil && req.Mask.Valid() {
		args.has_mask = 1
	}
	if req.RopeFreqs != nil && req.RopeFreqs.Valid() {
		args.has_rope_freqs = 1
	}
	return args
}

// NativeGemma4FixedOwnerAttention runs the fused fixed-owner attention kernel.
// The caller has already established that the native path is available for this
// layer (architecture-side predicate) and that fixed is a usable FixedKVCache;
// this entry point ensures the cache shape, borrows its state, runs the kernel and
// writes the updated K/V back. It returns ok=false only when the borrowed state is
// unusable.
func NativeGemma4FixedOwnerAttention(req Gemma4FixedAttentionRequest, fixed *FixedKVCache) (*Array, SharedKV, bool, error) {
	fixed.ensureShape(int32(req.X.Dim(0)), req.NumKeyValueHeads, req.HeadDim, req.HeadDim, req.X.Dtype(), req.X.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, SharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := FromValue(offset)
	scaleArray := FromValue(req.Scale)
	defer Free(offsetArray, scaleArray)

	req.keyCache = state.Keys
	req.valueCache = state.Values
	req.offset = offsetArray
	req.scaleArray = scaleArray

	out := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION", req.X, state.Keys, state.Values)
	newKeys := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_K", state.Keys)
	newValues := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_V", state.Values)
	args := req.cArgs()
	rc := C.go_mlx_gemma4_fixed_owner_attention(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := LastError(); err != nil {
			return nil, SharedKV{}, true, err
		}
		return nil, SharedKV{}, true, core.E("mlx.NativeGemma4FixedOwnerAttention", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if err := ValidateGemma4LayerOutputs("mlx.NativeGemma4FixedOwnerAttention", []*Array{out, newKeys, newValues}, true); err != nil {
		Free(out, newKeys, newValues)
		return nil, SharedKV{}, true, err
	}
	if err := ValidateGemma4LayerOutputShapes("mlx.NativeGemma4FixedOwnerAttention", req.X, out, newKeys, newValues, state.Keys, state.Values, true, true); err != nil {
		Free(out, newKeys, newValues)
		return nil, SharedKV{}, true, err
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
		Free(out)
		return nil, SharedKV{}, true, core.E("mlx.NativeGemma4FixedOwnerAttention", "native wrapper updated cache without valid K/V state", nil)
	}
	return out, SharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
}

// NativeGemma4FixedOwnerAttentionResidual runs the fused fixed-owner attention
// kernel with a fused residual add + post-attention norm. req.Residual and
// req.PostAttnNorm must be set.
func NativeGemma4FixedOwnerAttentionResidual(req Gemma4FixedAttentionRequest, fixed *FixedKVCache) (*Array, SharedKV, bool, error) {
	fixed.ensureShape(int32(req.X.Dim(0)), req.NumKeyValueHeads, req.HeadDim, req.HeadDim, req.X.Dtype(), req.X.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, SharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := FromValue(offset)
	scaleArray := FromValue(req.Scale)
	defer Free(offsetArray, scaleArray)

	req.keyCache = state.Keys
	req.valueCache = state.Values
	req.offset = offsetArray
	req.scaleArray = scaleArray

	out := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", req.Residual, req.X, state.Keys, state.Values)
	newKeys := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_K", state.Keys)
	newValues := NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_V", state.Values)
	args := req.cArgs()
	rc := C.go_mlx_gemma4_fixed_owner_attention_residual(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := LastError(); err != nil {
			return nil, SharedKV{}, true, err
		}
		return nil, SharedKV{}, true, core.E("mlx.NativeGemma4FixedOwnerAttentionResidual", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if err := ValidateGemma4LayerOutputs("mlx.NativeGemma4FixedOwnerAttentionResidual", []*Array{out, newKeys, newValues}, true); err != nil {
		Free(out, newKeys, newValues)
		return nil, SharedKV{}, true, err
	}
	if err := ValidateGemma4LayerOutputShapes("mlx.NativeGemma4FixedOwnerAttentionResidual", req.Residual, out, newKeys, newValues, state.Keys, state.Values, true, true); err != nil {
		Free(out, newKeys, newValues)
		return nil, SharedKV{}, true, err
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
		Free(out)
		return nil, SharedKV{}, true, core.E("mlx.NativeGemma4FixedOwnerAttentionResidual", "native wrapper updated cache without valid K/V state", nil)
	}
	return out, SharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
}

// Gemma4LayerRequest carries the static per-layer inputs for the fused decode
// kernels — everything the kernel needs that does not depend on cache state. The
// dynamic state (prev keys/values, offset, fixed mask, ownsKV/fixedKV) is supplied
// per call. Projections are passed as *Linear/*SwitchLinear (metal-owned bundles of
// weight + scales + biases + group/bits), the rest as norm arrays + scalars.
type Gemma4LayerRequest struct {
	InputNorm             *Array
	PostAttnNorm          *Array
	PreFFNorm             *Array
	PreFFNorm2            *Array
	PostFFNorm1           *Array
	PostFFNorm2           *Array
	PostFFNorm            *Array
	PostPerLayerInputNorm *Array
	LayerScalar           *Array

	QProj     *Linear
	KProj     *Linear
	VProj     *Linear
	OProj     *Linear
	QNorm     *Array
	KNorm     *Array
	RopeFreqs *Array

	MLPGate *Linear
	MLPUp   *Linear
	MLPDown *Linear

	PerLayerInputGate  *Linear
	PerLayerProjection *Linear

	EnableMoE      bool
	UseKEqV        bool
	RouterProj     *Linear
	RouterScale    *Array // unscaled
	RouterScaled   *Array // pre-scaled (preferred when valid)
	PerExpertScale *Array
	RouterTopK     int32
	RouterEps      float32
	RouterRootSize float32

	ExpertGate   *SwitchLinear
	ExpertUp     *SwitchLinear
	ExpertGateUp *SwitchLinear
	ExpertDown   *SwitchLinear

	NumAttentionHeads int32
	NumKeyValueHeads  int32
	HeadDim           int32
	RopeDims          int32
	RopeBase          float32
	AttentionScale    float32
}

// cArgs builds the C layer-args struct from the static request plus the per-call
// dynamic inputs. It is the relocated nativeGemma4LayerArgs.
func (req Gemma4LayerRequest) cArgs(x, prevKeys, prevValues, perLayerInput, fixedMask *Array, ownsKV, fixedKV bool, offset int) C.go_mlx_gemma4_layer_args {
	args := C.go_mlx_gemma4_layer_args{
		x:                         cArray(x),
		prev_keys:                 cArray(prevKeys),
		prev_values:               cArray(prevValues),
		per_layer_input:           cArray(perLayerInput),
		fixed_mask:                cArray(fixedMask),
		input_norm:                cArray(req.InputNorm),
		post_attn_norm:            cArray(req.PostAttnNorm),
		pre_ff_norm:               cArray(req.PreFFNorm),
		pre_ff_norm2:              cArray(req.PreFFNorm2),
		post_ff_norm1:             cArray(req.PostFFNorm1),
		post_ff_norm2:             cArray(req.PostFFNorm2),
		post_ff_norm:              cArray(req.PostFFNorm),
		post_per_layer_input_norm: cArray(req.PostPerLayerInputNorm),
		layer_scalar:              cArray(req.LayerScalar),
		q_weight:                  cArray(req.QProj.Weight),
		q_scales:                  cArray(req.QProj.Scales),
		q_biases:                  cArray(req.QProj.Biases),
		k_weight:                  cArray(req.KProj.Weight),
		k_scales:                  cArray(req.KProj.Scales),
		k_biases:                  cArray(req.KProj.Biases),
		o_weight:                  cArray(req.OProj.Weight),
		o_scales:                  cArray(req.OProj.Scales),
		o_biases:                  cArray(req.OProj.Biases),
		q_norm:                    cArray(req.QNorm),
		k_norm:                    cArray(req.KNorm),
		rope_freqs:                cArray(req.RopeFreqs),
		q_group_size:              C.int(req.QProj.GroupSize),
		q_bits:                    C.int(req.QProj.Bits),
		k_group_size:              C.int(req.KProj.GroupSize),
		k_bits:                    C.int(req.KProj.Bits),
		o_group_size:              C.int(req.OProj.GroupSize),
		o_bits:                    C.int(req.OProj.Bits),
		mlp_gate_weight:           cArray(req.MLPGate.Weight),
		mlp_gate_scales:           cArray(req.MLPGate.Scales),
		mlp_gate_biases:           cArray(req.MLPGate.Biases),
		mlp_gate_group_size:       C.int(req.MLPGate.GroupSize),
		mlp_gate_bits:             C.int(req.MLPGate.Bits),
		mlp_up_weight:             cArray(req.MLPUp.Weight),
		mlp_up_scales:             cArray(req.MLPUp.Scales),
		mlp_up_biases:             cArray(req.MLPUp.Biases),
		mlp_up_group_size:         C.int(req.MLPUp.GroupSize),
		mlp_up_bits:               C.int(req.MLPUp.Bits),
		mlp_down_weight:           cArray(req.MLPDown.Weight),
		mlp_down_scales:           cArray(req.MLPDown.Scales),
		mlp_down_biases:           cArray(req.MLPDown.Biases),
		mlp_down_group_size:       C.int(req.MLPDown.GroupSize),
		mlp_down_bits:             C.int(req.MLPDown.Bits),
		num_attention_heads:       C.int(req.NumAttentionHeads),
		num_key_value_heads:       C.int(req.NumKeyValueHeads),
		head_dim:                  C.int(req.HeadDim),
		rope_dims:                 C.int(req.RopeDims),
		offset:                    C.int(offset),
		rope_base:                 C.float(req.RopeBase),
		attention_scale:           C.float(req.AttentionScale),
	}
	if prevKeys != nil && prevValues != nil {
		args.has_prev = 1
	}
	if perLayerInput != nil && perLayerInput.Valid() {
		args.has_per_layer_input = 1
		args.per_layer_gate_weight = cArray(req.PerLayerInputGate.Weight)
		args.per_layer_gate_scales = cArray(req.PerLayerInputGate.Scales)
		args.per_layer_gate_biases = cArray(req.PerLayerInputGate.Biases)
		args.per_layer_gate_group_size = C.int(req.PerLayerInputGate.GroupSize)
		args.per_layer_gate_bits = C.int(req.PerLayerInputGate.Bits)
		args.per_layer_projection_weight = cArray(req.PerLayerProjection.Weight)
		args.per_layer_projection_scales = cArray(req.PerLayerProjection.Scales)
		args.per_layer_projection_biases = cArray(req.PerLayerProjection.Biases)
		args.per_layer_projection_group_size = C.int(req.PerLayerProjection.GroupSize)
		args.per_layer_projection_bits = C.int(req.PerLayerProjection.Bits)
	}
	if ownsKV {
		args.owns_kv = 1
	}
	if fixedKV {
		args.fixed_kv = 1
	}
	if fixedMask != nil && fixedMask.Valid() {
		args.has_fixed_mask = 1
	}
	if req.RopeFreqs != nil && req.RopeFreqs.Valid() {
		args.has_rope_freqs = 1
	}
	if req.UseKEqV {
		args.use_k_eq_v = 1
	} else if req.VProj != nil {
		args.v_weight = cArray(req.VProj.Weight)
		args.v_scales = cArray(req.VProj.Scales)
		args.v_biases = cArray(req.VProj.Biases)
		args.v_group_size = C.int(req.VProj.GroupSize)
		args.v_bits = C.int(req.VProj.Bits)
	}
	if req.EnableMoE && req.RouterProj != nil && req.ExpertDown != nil {
		args.has_moe = 1
		args.router_weight = cArray(req.RouterProj.Weight)
		args.router_scales = cArray(req.RouterProj.Scales)
		args.router_biases = cArray(req.RouterProj.Biases)
		args.router_group_size = C.int(req.RouterProj.GroupSize)
		args.router_bits = C.int(req.RouterProj.Bits)
		if req.RouterScaled != nil && req.RouterScaled.Valid() {
			args.router_scale = cArray(req.RouterScaled)
			args.has_router_scale_scaled = 1
		} else {
			args.router_scale = cArray(req.RouterScale)
		}
		args.router_per_expert_scale = cArray(req.PerExpertScale)
		args.router_top_k = C.int(req.RouterTopK)
		args.router_eps = C.float(req.RouterEps)
		args.router_root_size = C.float(req.RouterRootSize)

		if req.ExpertGate != nil {
			args.expert_gate_weight = cArray(req.ExpertGate.Weight)
			args.expert_gate_scales = cArray(req.ExpertGate.Scales)
			args.expert_gate_biases = cArray(req.ExpertGate.Biases)
			args.expert_gate_bias = cArray(req.ExpertGate.Bias)
			args.expert_gate_group_size = C.int(req.ExpertGate.GroupSize)
			args.expert_gate_bits = C.int(req.ExpertGate.Bits)
		}
		if req.ExpertUp != nil {
			args.expert_up_weight = cArray(req.ExpertUp.Weight)
			args.expert_up_scales = cArray(req.ExpertUp.Scales)
			args.expert_up_biases = cArray(req.ExpertUp.Biases)
			args.expert_up_bias = cArray(req.ExpertUp.Bias)
			args.expert_up_group_size = C.int(req.ExpertUp.GroupSize)
			args.expert_up_bits = C.int(req.ExpertUp.Bits)
		}
		if req.ExpertGateUp != nil {
			args.expert_gate_up_weight = cArray(req.ExpertGateUp.Weight)
			args.expert_gate_up_scales = cArray(req.ExpertGateUp.Scales)
			args.expert_gate_up_biases = cArray(req.ExpertGateUp.Biases)
			args.expert_gate_up_bias = cArray(req.ExpertGateUp.Bias)
			args.expert_gate_up_group_size = C.int(req.ExpertGateUp.GroupSize)
			args.expert_gate_up_bits = C.int(req.ExpertGateUp.Bits)
		}
		args.expert_down_weight = cArray(req.ExpertDown.Weight)
		args.expert_down_scales = cArray(req.ExpertDown.Scales)
		args.expert_down_biases = cArray(req.ExpertDown.Biases)
		args.expert_down_bias = cArray(req.ExpertDown.Bias)
		args.expert_down_group_size = C.int(req.ExpertDown.GroupSize)
		args.expert_down_bits = C.int(req.ExpertDown.Bits)
	}
	return args
}

// NativeGemma4DecodeLayer runs the fused single-layer decode kernel. The caller
// has already established (architecture-side) that the native layer path is
// available; this entry point resolves the cache state, runs the kernel and writes
// the K/V back, returning ok=false only when the cache state is unusable.
func NativeGemma4DecodeLayer(req Gemma4LayerRequest, x *Array, c Cache, B, L int32, perLayerInput *Array, prev SharedKV, fixedMask *Array) (*Array, SharedKV, bool, error) {
	offset := 0
	var prevKeys, prevValues *Array
	var pageState PagedKVState
	var fixedState FixedKVState
	ownsKV := !prev.HasState()
	fixedKV := prev.Fixed
	if ownsKV {
		switch cache := c.(type) {
		case *PagedKVCache:
			offset = cache.Offset()
			pageState = cache.PageState()
			if len(pageState.Keys) != 1 || len(pageState.Values) != 1 {
				pageState.Free()
				return nil, SharedKV{}, false, nil
			}
			prevKeys = pageState.Keys[0]
			prevValues = pageState.Values[0]
			defer pageState.Free()
		case *FixedKVCache:
			offset = cache.Offset()
			fixedState = cache.BorrowedFixedState()
			if fixedState.Keys == nil || fixedState.Values == nil {
				return nil, SharedKV{}, false, nil
			}
			prevKeys = fixedState.Keys
			prevValues = fixedState.Values
			fixedKV = true
		default:
			return nil, SharedKV{}, false, nil
		}
	} else {
		offset = prev.Offset
		switch {
		case prev.Keys != nil && prev.Values != nil:
			prevKeys, prevValues = prev.Keys, prev.Values
		case prev.HasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
			prevKeys, prevValues = prev.Pages.Keys[0], prev.Pages.Values[0]
		default:
			return nil, SharedKV{}, false, nil
		}
	}
	if prevKeys == nil || prevValues == nil || !prevKeys.Valid() || !prevValues.Valid() {
		return nil, SharedKV{}, false, nil
	}

	out := NewArray("FAST_GEMMA4_DECODE_LAYER", x, prevKeys, prevValues, perLayerInput)
	newK := NewArray("FAST_GEMMA4_DECODE_LAYER_K", x)
	newV := NewArray("FAST_GEMMA4_DECODE_LAYER_V", x)
	args := req.cArgs(x, prevKeys, prevValues, perLayerInput, fixedMask, ownsKV, fixedKV, offset)
	rc := C.go_mlx_gemma4_decode_layer(&out.ctx, &newK.ctx, &newV.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		Free(out, newK, newV)
		if err := LastError(); err != nil {
			return nil, SharedKV{}, true, err
		}
		return nil, SharedKV{}, true, core.E("mlx.NativeGemma4DecodeLayer", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}

	if ownsKV {
		if err := ValidateGemma4LayerOutputs("mlx.NativeGemma4DecodeLayer", []*Array{out, newK, newV}, true); err != nil {
			Free(out, newK, newV)
			return nil, SharedKV{}, true, err
		}
		if err := ValidateGemma4LayerOutputShapes("mlx.NativeGemma4DecodeLayer", x, out, newK, newV, prevKeys, prevValues, true, fixedKV); err != nil {
			Free(out, newK, newV)
			return nil, SharedKV{}, true, err
		}
		if fixedKV {
			fixed, _ := c.(*FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(newK, newV, int(L))
			return out, SharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
		}
		paged, _ := c.(*PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(newK, newV, int(L))
		return out, SharedKV{Pages: pages, Offset: offset}, true, nil
	}
	if err := ValidateGemma4LayerOutputs("mlx.NativeGemma4DecodeLayer", []*Array{out}, false); err != nil {
		Free(out, newK, newV)
		return nil, SharedKV{}, true, err
	}
	if err := ValidateGemma4LayerOutputShapes("mlx.NativeGemma4DecodeLayer", x, out, nil, nil, prevKeys, prevValues, false, fixedKV); err != nil {
		Free(out, newK, newV)
		return nil, SharedKV{}, true, err
	}
	Free(newK, newV)
	return out, prev, true, nil
}

// Gemma4FixedMaskSet is the per-layer fixed attention mask provider the fused
// model greedy kernel consults. The model architecture implements it (its mask set
// satisfies this interface); metal reaches the masks only through it, so no model
// type is named here.
type Gemma4FixedMaskSet interface {
	ForLayer(cache Cache, prev SharedKV) *Array
}

// Gemma4GreedyRequest carries the whole-model inputs for the fused fixed greedy
// decode kernel. Layers holds the per-layer static args in model order; the kernel
// merges them with the per-layer cache state it resolves from Caches.
type Gemma4GreedyRequest struct {
	Hidden            *Array
	Layers            []Gemma4LayerRequest
	PreviousKVs       []int32
	CacheIndexByLayer []int32
	Caches            []Cache
	PerLayerInputs    []*Array
	FixedMasks        Gemma4FixedMaskSet

	FinalNorm *Array
	Output    *Linear
}

// NativeGemma4FixedGreedyToken runs the fused single-token greedy decode over the
// whole model in one native call. The caller has already established the path is
// available; this entry point resolves per-layer cache state, builds the layer args
// and writes K/V back. suppress may be nil; suppressTokens, when present and
// suppress is nil, are materialised into a token-id array.
func NativeGemma4FixedGreedyToken(req Gemma4GreedyRequest, suppress *Array, suppressTokens ...int32) (*Array, bool, error) {
	h := req.Hidden
	model := req.Layers
	caches := req.Caches
	fixedMasks := req.FixedMasks

	layerCount := len(model)
	var layerArgsStack [64]C.go_mlx_gemma4_layer_args
	var previousKVsStack [64]C.int
	var newKCtxStack [64]C.mlx_array
	var newVCtxStack [64]C.mlx_array
	var layerArgs []C.go_mlx_gemma4_layer_args
	var previousKVs []C.int
	var newKCtx []C.mlx_array
	var newVCtx []C.mlx_array
	var layerArgsPtr *C.go_mlx_gemma4_layer_args
	var previousKVsPtr *C.int
	var newKCtxPtr *C.mlx_array
	var newVCtxPtr *C.mlx_array
	var cgoPinner runtime.Pinner
	defer cgoPinner.Unpin()
	if layerCount <= len(layerArgsStack) {
		layerArgs = layerArgsStack[:layerCount]
		previousKVs = previousKVsStack[:layerCount]
		newKCtx = newKCtxStack[:layerCount]
		newVCtx = newVCtxStack[:layerCount]
		layerArgsPtr = &layerArgs[0]
		previousKVsPtr = &previousKVs[0]
		newKCtxPtr = &newKCtx[0]
		newVCtxPtr = &newVCtx[0]
		cgoPinner.Pin(layerArgsPtr)
		cgoPinner.Pin(previousKVsPtr)
		cgoPinner.Pin(newKCtxPtr)
		cgoPinner.Pin(newVCtxPtr)
	} else {
		layerArgsPtr = (*C.go_mlx_gemma4_layer_args)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.go_mlx_gemma4_layer_args{}))))
		previousKVsPtr = (*C.int)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.int(0)))))
		newKCtxPtr = (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
		newVCtxPtr = (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
		if layerArgsPtr == nil || previousKVsPtr == nil || newKCtxPtr == nil || newVCtxPtr == nil {
			if layerArgsPtr != nil {
				C.free(unsafe.Pointer(layerArgsPtr))
			}
			if previousKVsPtr != nil {
				C.free(unsafe.Pointer(previousKVsPtr))
			}
			if newKCtxPtr != nil {
				C.free(unsafe.Pointer(newKCtxPtr))
			}
			if newVCtxPtr != nil {
				C.free(unsafe.Pointer(newVCtxPtr))
			}
			return nil, true, core.NewError("mlx.NativeGemma4FixedGreedyToken: allocate C argument buffers failed")
		}
		defer C.free(unsafe.Pointer(layerArgsPtr))
		defer C.free(unsafe.Pointer(previousKVsPtr))
		defer C.free(unsafe.Pointer(newKCtxPtr))
		defer C.free(unsafe.Pointer(newVCtxPtr))
		layerArgs = unsafe.Slice(layerArgsPtr, layerCount)
		previousKVs = unsafe.Slice(previousKVsPtr, layerCount)
		newKCtx = unsafe.Slice(newKCtxPtr, layerCount)
		newVCtx = unsafe.Slice(newVCtxPtr, layerCount)
	}
	var fixedByLayerStack [64]*FixedKVCache
	var statesStack [64]FixedKVState
	var offsetsStack [64]int
	var fixedByLayer []*FixedKVCache
	var states []FixedKVState
	var offsets []int
	if layerCount <= len(statesStack) {
		fixedByLayer = fixedByLayerStack[:layerCount]
		states = statesStack[:layerCount]
		offsets = offsetsStack[:layerCount]
	} else {
		fixedByLayer = make([]*FixedKVCache, layerCount)
		states = make([]FixedKVState, layerCount)
		offsets = make([]int, layerCount)
	}
	defer func() {
		for i := range states {
			states[i].Free()
		}
	}()

	B := int32(h.Dim(0))
	for i := range model {
		prevIdx := int(req.PreviousKVs[i])
		previousKVs[i] = C.int(prevIdx)
		ownsKV := prevIdx == i
		var fixed *FixedKVCache
		var prev SharedKV
		var prevKeys, prevValues *Array
		var offset int
		if ownsKV {
			cacheIdx := int(req.CacheIndexByLayer[i])
			fixed = caches[cacheIdx].(*FixedKVCache)
			fixed.ensureShape(B, model[i].NumKeyValueHeads, model[i].HeadDim, model[i].HeadDim, h.Dtype(), h.Dtype())
			state := fixed.BorrowedFixedState()
			if state.Keys == nil || state.Values == nil {
				return nil, false, nil
			}
			states[i] = state
			fixedByLayer[i] = fixed
			prevKeys, prevValues = state.Keys, state.Values
			offset = fixed.Offset()
			offsets[i] = offset
		} else {
			state := states[prevIdx]
			if state.Keys == nil || state.Values == nil {
				return nil, false, nil
			}
			prevKeys, prevValues = state.Keys, state.Values
			offset = offsets[prevIdx]
			prev = SharedKV{Keys: prevKeys, Values: prevValues, Offset: offset, Fixed: true, Borrowed: true}
		}
		var perLayerInput *Array
		if req.PerLayerInputs != nil {
			perLayerInput = req.PerLayerInputs[i]
		}
		fixedMask := fixedMasks.ForLayer(fixed, prev)
		layerArgs[i] = model[i].cArgs(h, prevKeys, prevValues, perLayerInput, fixedMask, ownsKV, true, offset)
	}

	out := NewArray("FAST_GEMMA4_MODEL_GREEDY_TOKEN", h, req.FinalNorm, req.Output.Weight, req.Output.Scales, req.Output.Biases)
	args := C.go_mlx_gemma4_model_greedy_args{
		hidden:           cArray(h),
		layers:           layerArgsPtr,
		previous_kvs:     previousKVsPtr,
		layer_count:      C.int(layerCount),
		final_norm:       cArray(req.FinalNorm),
		output_weight:    cArray(req.Output.Weight),
		output_scales:    cArray(req.Output.Scales),
		output_biases:    cArray(req.Output.Biases),
		output_quantized: 0,
	}
	ownsSuppress := false
	if len(suppressTokens) == 0 {
		suppress = nil
	} else if suppress == nil || !suppress.Valid() {
		suppress = SuppressTokenArray(suppressTokens)
		ownsSuppress = true
	}
	if ownsSuppress {
		defer Free(suppress)
	}
	if suppress != nil {
		args.suppress_token_ids = suppress.ctx
		args.has_suppress_token_ids = 1
	}
	if req.Output.Scales != nil && req.Output.Scales.Valid() {
		args.output_quantized = 1
	}
	cgoPinner.Pin(&args)
	rc := C.go_mlx_gemma4_fixed_greedy_token(
		&out.ctx,
		newKCtxPtr,
		newVCtxPtr,
		&args,
		gemma4DefaultStream(),
	)
	if rc != 0 {
		Free(out)
		FreeCArrayHandles(newKCtx)
		FreeCArrayHandles(newVCtx)
		if err := LastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.NativeGemma4FixedGreedyToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() {
		Free(out)
		FreeCArrayHandles(newKCtx)
		FreeCArrayHandles(newVCtx)
		return nil, true, core.E("mlx.NativeGemma4FixedGreedyToken", "native wrapper returned invalid token", nil)
	}

	for i, fixed := range fixedByLayer {
		if fixed == nil {
			continue
		}
		newKeys := NewArray("FAST_GEMMA4_MODEL_GREEDY_K", h)
		newValues := NewArray("FAST_GEMMA4_MODEL_GREEDY_V", h)
		newKeys.ctx = newKCtx[i]
		newValues.ctx = newVCtx[i]
		if !newKeys.Valid() || !newValues.Valid() {
			Free(out, newKeys, newValues)
			return nil, true, core.E("mlx.NativeGemma4FixedGreedyToken", "native wrapper returned invalid KV outputs", nil)
		}
		Free(fixed.keys, fixed.values)
		fixed.keys = newKeys
		fixed.values = newValues
		fixed.offset++
		fixed.length = min(fixed.offset, fixed.maxSize)
	}
	return out, true, nil
}

// cArray rebuilds this package's C.mlx_array from an *Array. In package metal the
// ctx field is directly accessible, but ArrayHandle keeps the construction in one
// place and nil-safe.
func cArray(a *Array) C.mlx_array {
	var r C.mlx_array
	if a != nil {
		r.ctx = ArrayHandle(a)
	}
	return r
}
