// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"os"
	"sync"

	"dappco.re/go/mlx/pkg/metal"
)

// Whole-stack compiled decode (graph-eval): the entire single-token forward —
// embedding scale already applied upstream, then every decoder layer, the final
// norm, and the output projection — as ONE mlx_compile'd closure, re-applied per
// token. The per-layer compiled path already collapses each layer's graph BUILD,
// but still pays ~N closure-applies of host scheduling + Go marshalling per token
// (the 8.6 ms "logits graph" prefetch). Composing the layers into a single trace
// collapses that to ONE apply: MLX schedules the whole step once and re-evals it
// with new input arrays (token hidden, offset, caches) — MLX owns the input
// buffers, so per-token state advances cleanly (unlike raw command replay, which
// froze MLX's internal per-token buffers in the capture).
//
// Reuses gemma4CompiledLayerStep per layer inside the stack trace: each layer's
// op-builder is invoked with its sub-slice of the flat input vector, chaining the
// hidden state layer to layer. Decode-only, steady state (preCap owners +
// consumers, L==1); anything else declines to the per-layer / uncompiled paths.

func compiledStackDecodeEnabled() bool {
	v := os.Getenv("MLX_COMPILED_STACK")
	return v != "" && v != "0"
}

// gemma4StackPlan is the per-token assembly plan: the ordered per-layer keys +
// states and the shared-KV routing, resolved host-side, cached on the model.
type gemma4StackPlan struct {
	keys    []gemma4CompiledLayerKey
	states  []*gemma4CompiledLayerState
	ownerOf []int // for a consumer layer i, the layer index whose new K/V it reads; -1 for owners
}

var gemma4StackFnCache sync.Map // stackTraceKey -> *metal.CompiledFunc

// stackTraceKey identifies a compiled stack trace: the ordered per-layer keys.
// Different cache capacities / regimes re-key (shapeless=false, like the layer fn).
type stackTraceKey struct {
	n   int
	sig [128]gemma4CompiledLayerKey
}

func makeStackTraceKey(keys []gemma4CompiledLayerKey) stackTraceKey {
	var k stackTraceKey
	k.n = len(keys)
	for i, key := range keys {
		if i >= len(k.sig) {
			break
		}
		k.sig[i] = key
	}
	return k
}

// gemma4CompiledStackFn returns the compiled whole-stack closure for the plan,
// composing every layer's traced ops into one trace. Returns the last layer's
// hidden state (pre-final-norm) + the new owner K/V; the final norm + output
// projection stay in the native tail (ForwardLastTokenLogitsAndHidden).
func gemma4CompiledStackFn(plan *gemma4StackPlan) *metal.CompiledFunc {
	tk := makeStackTraceKey(plan.keys)
	if fn, ok := gemma4StackFnCache.Load(tk); ok {
		return fn.(*metal.CompiledFunc)
	}
	fn := metal.CompileShapeless(gemma4CompiledStackStep(plan), false)
	actual, _ := gemma4StackFnCache.LoadOrStore(tk, fn)
	return actual.(*metal.CompiledFunc)
}

// gemma4CompiledStackStep builds the whole-step trace. Input layout (flat):
//
//	[0]            h        — embedded+scaled hidden for this token
//	[1]            offset   — cache write position (shared by all layers)
//	then per layer in order: cacheK, cacheV, [pli], <state.weights...>
//	  (consumer layers contribute NO cacheK/cacheV — they read the owner's new
//	   K/V chained in-trace; postCap is declined upstream so no shift/last here)
//
// Output: [hidden, then per owner layer: newK, newV] in layer order.
func gemma4CompiledStackStep(plan *gemma4StackPlan) func([]*metal.Array) []*metal.Array {
	return func(in []*metal.Array) []*metal.Array {
		h := in[0]
		offset := in[1]
		idx := 2
		ownerK := make([]*metal.Array, len(plan.keys))
		ownerV := make([]*metal.Array, len(plan.keys))
		outs := []*metal.Array{nil} // [0] reserved for the final hidden
		for i, key := range plan.keys {
			st := plan.states[i]
			var sub []*metal.Array
			if key.regime == gemma4LayerConsumer {
				// Consumer reads the owner's freshly-written K/V (chained), shares
				// offset. Per-layer order: x, cacheK, cacheV, offset, [pli], weights.
				oi := plan.ownerOf[i]
				sub = append(sub, h, ownerK[oi], ownerV[oi], offset)
				if key.pliDim > 0 {
					sub = append(sub, in[idx])
					idx++
				}
				sub = append(sub, st.weights...)
				res := gemma4CompiledLayerStep(key)(sub)
				h = res[0]
			} else {
				cacheK, cacheV := in[idx], in[idx+1]
				idx += 2
				sub = append(sub, h, cacheK, cacheV, offset)
				if key.pliDim > 0 {
					sub = append(sub, in[idx])
					idx++
				}
				sub = append(sub, st.weights...)
				res := gemma4CompiledLayerStep(key)(sub)
				h = res[0]
				ownerK[i] = res[1]
				ownerV[i] = res[2]
				outs = append(outs, res[1], res[2])
			}
		}
		outs[0] = h // last layer hidden; native tail applies norm + output projection
		return outs
	}
}

// compiledStackHidden runs the whole-stack graph-eval for one token. h is the
// embedded+scaled hidden ([1,1,hidden]); caches/offset advance in place via the
// returned new owner K/V. Returns (last-layer hidden, true) on success, or
// (nil,false) to decline to the per-layer path. Steady-state decode only.
func (m *Gemma4Model) compiledStackHidden(h *metal.Array, caches []metal.Cache, perLayerInput *metal.Array, B, L int32) (*metal.Array, bool) {
	if !compiledStackDecodeEnabled() || B != 1 || L != 1 {
		return nil, false
	}
	plan, owners, ok := m.buildStackPlan(caches)
	if !ok {
		return nil, false
	}
	// Assemble the flat input vector matching gemma4CompiledStackStep's layout.
	off := 0
	for i, key := range plan.keys {
		if key.regime != gemma4LayerConsumer {
			off = owners[i].fixed.Offset()
			break
		}
	}
	offsetArr := metal.FromValue(off)
	in := append(m.stackIn[:0], h, offsetArr) // reuse the cached input buffer (zero alloc steady state)
	// Flat input carries ONLY the dynamic per-token arrays (cacheK/V, pli); the
	// trace uses each layer's weights as captured constants (gemma4CompiledLayerStep
	// reads them from plan.states), so the apply must NOT append them here or the
	// trace's idx walk over `in` misaligns.
	for i, key := range plan.keys {
		if key.regime != gemma4LayerConsumer {
			fs := owners[i].fixed.BorrowedFixedState()
			in = append(in, fs.Keys, fs.Values)
		}
		if key.pliDim > 0 {
			pli := m.perLayerInputForLayer(perLayerInput, B, L, int32(i))
			in = append(in, pli)
			defer metal.Free(pli)
		}
	}
	m.stackIn = in // keep the grown backing array for next token
	fn := gemma4CompiledStackFn(plan)
	outs := fn.Call(in...)
	metal.Free(offsetArr)
	if len(outs) < 1 || outs[0] == nil || !outs[0].Valid() {
		metal.Free(outs...)
		return nil, false
	}
	// Write the new owner K/V back into the fixed caches (outs[1:] in owner order).
	oi := 1
	for i, key := range plan.keys {
		if key.regime == gemma4LayerConsumer {
			continue
		}
		owners[i].fixed.ReplaceFixedFromNativeBorrowed(outs[oi], outs[oi+1], int(L))
		oi += 2
	}
	return outs[0], true
}

// stackOwner pairs a layer's cache with its FixedKVCache view.
type stackOwner struct {
	cache metal.Cache
	fixed *metal.FixedKVCache
}

// buildStackPlan resolves the per-layer keys/states + shared-KV routing for the
// current steady-state decode, or declines. Conservative: every layer must be a
// preCap owner or a consumer of a preCap owner (no postCap, no LoRA, no MoE-only
// declines), all sized FixedKVCaches.
func (m *Gemma4Model) buildStackPlan(caches []metal.Cache) (*gemma4StackPlan, []stackOwner, bool) {
	if !metal.CompiledLayerDecodeEnabled() {
		return nil, nil, false
	}
	n := len(m.Layers)
	// Reuse the cached plan/owners scratch — overwritten in place, zero per-token
	// allocation at steady state (the worked-code pattern; was 13 KB/tok of churn).
	if m.stackPlan == nil || len(m.stackPlan.keys) != n {
		m.stackPlan = &gemma4StackPlan{
			keys:    make([]gemma4CompiledLayerKey, n),
			states:  make([]*gemma4CompiledLayerState, n),
			ownerOf: make([]int, n),
		}
		m.stackOwners = make([]stackOwner, n)
	}
	plan := m.stackPlan
	owners := m.stackOwners
	for i := range m.Layers {
		l := m.Layers[i]
		consumer := m.PreviousKVs[i] != int32(i)
		ownerIdx := i
		if consumer {
			ownerIdx = int(m.PreviousKVs[i])
		}
		st := l.compiledLayerState(consumer, m.Cfg)
		if st == nil || st.declined || st.consumer != consumer {
			return nil, nil, false
		}
		for _, lin := range st.linears {
			if lin.LoRA != nil {
				return nil, nil, false
			}
		}
		key := st.key
		plan.ownerOf[i] = -1
		if consumer {
			key.regime = gemma4LayerConsumer
			plan.ownerOf[i] = ownerIdx
			// Consumer reads the owner's cache — derive band/capacity/dtype from it
			// (the owner's new K/V, chained in-trace), matching the per-layer path.
			oci := m.CacheIndexByLayer[ownerIdx]
			if oci < 0 || int(oci) >= len(caches) {
				return nil, nil, false
			}
			ofixed, isFixed := caches[oci].(*metal.FixedKVCache)
			if !isFixed || ofixed == nil || ofixed.MaxSize() <= 0 {
				return nil, nil, false
			}
			ofixed.EnsureDecodeCapacityFor(1)
			ofs := ofixed.BorrowedFixedState()
			if !gemma4ValidKV(ofs.Keys, ofs.Values) {
				return nil, nil, false
			}
			ooff := ofixed.Offset()
			if ooff+1 > ofixed.MaxSize() || ooff+1 > int(ofs.Keys.Dim(2)) {
				return nil, nil, false
			}
			key.capacity = int32(ofs.Keys.Dim(2))
			key.cacheDType = ofs.Keys.Dtype()
			key.attnBand = gemma4AttnBandFor(int32(ooff)+1, key.capacity)
			key.seqLen = 1
		} else {
			ci := m.CacheIndexByLayer[i]
			if ci < 0 || int(ci) >= len(caches) {
				return nil, nil, false
			}
			fixed, isFixed := caches[ci].(*metal.FixedKVCache)
			if !isFixed || fixed == nil || fixed.MaxSize() <= 0 {
				return nil, nil, false
			}
			fixed.EnsureDecodeCapacityFor(1)
			fs := fixed.BorrowedFixedState()
			if !gemma4ValidKV(fs.Keys, fs.Values) {
				return nil, nil, false
			}
			off := fixed.Offset()
			if off+1 > fixed.MaxSize() {
				return nil, nil, false // postCap (sliding) — decline for now
			}
			band := int(fs.Keys.Dim(2))
			if off+1 > band {
				return nil, nil, false
			}
			key.regime = gemma4LayerOwnerPreCap
			key.capacity = int32(fs.Keys.Dim(2))
			key.cacheDType = fs.Keys.Dtype()
			key.attnBand = gemma4AttnBandFor(int32(off)+1, key.capacity)
			key.seqLen = 1
			owners[i] = stackOwner{cache: caches[ci], fixed: fixed}
		}
		key.xDType = metal.DTypeBFloat16
		plan.keys[i] = key
		plan.states[i] = st
	}
	// Need at least one owner (for the shared offset).
	for i := range plan.keys {
		if plan.keys[i].regime != gemma4LayerConsumer {
			return plan, owners, true
		}
	}
	return nil, nil, false
}
