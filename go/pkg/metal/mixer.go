// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// MixerCtx is the per-forward context the engine hands a sequence mixer for one
// chunk. It carries only universal pieces — the state-holder, the chunk
// geometry, the decode position, and the attention masks a softmax mixer needs.
// A recurrent mixer (GLA, RetNet, Mamba, RWKV, …) reads x plus its state from
// Cache and ignores the mask/window fields; a softmax mixer uses them.
// Architecture specifics (head counts, RoPE base, norm scales, the runtime-mask
// machinery) are captured in the mixer instance at construction, never here —
// so this contract stays family-agnostic and the decoder loop calls every
// mixer the same way, whatever it is.
type MixerCtx struct {
	Cache       Cache    // the state-holder the mixer declared it needs: a KV cache (softmax) or a recurrent-state holder (SSM/linear-attn)
	Prev        SharedKV // the per-layer shared-KV hand-off (Gemma-4 shared-KV); zero value for mixers that do not share KV
	B           int32    // batch size of this chunk
	L           int32    // sequence length of this chunk
	Step        int      // decode position (0 on prefill)
	Mask        *Array   // additive attention mask for this chunk (softmax mixers only; nil for recurrent)
	FixedMask   *Array   // the fixed sliding-window mask (softmax sliding layers; nil otherwise)
	Window      int32    // sliding-window span; 0 = full attention / not applicable
	Materialize bool     // materialise paged-KV for reuse (softmax + paged cache)
	// Extra is the arch-specific per-forward context the universal contract does
	// not model — e.g. Gemma-4's *Gemma4TextConfig + its per-forward runtime
	// mask cache. The architecture that builds the MixerCtx puts its own type
	// here and its mixer casts it back; every other mixer (the recurrent ones)
	// ignores it. This keeps the contract family-agnostic without a god-struct.
	Extra any
}

// Recurrent returns the recurrent-state holder for this layer when the mixer's
// cache is one — a StateRecurrent mixer's Forward reads its prior state and
// writes the advanced state through it, instead of the KV Update path a softmax
// mixer takes. Reports (nil,false) for a KV cache; the load-time Compatible()
// gate guarantees a recurrent mixer is only ever paired with the holder, so
// false signals a misconfiguration the mixer can treat as fatal.
//
//	if rc, ok := ctx.Recurrent(); ok {
//		prior := rc.RecurrentState()
//		out, next := scan(x, prior)
//		rc.SetRecurrentState(next)
//	}
func (ctx *MixerCtx) Recurrent() (RecurrentCache, bool) {
	rc, ok := ctx.Cache.(RecurrentCache)
	return rc, ok
}

// MixerCompute is the driver-side contract a registered sequence mixer fulfils
// on the metal Engine. It identifies itself and declares its state kind (via the
// embedded pure-Go scheme.Mixer) and mixes one chunk of hidden states. The
// decoder loop resolves the mixer with scheme.MixerFor(kind) and calls Forward;
// a new mixer — any member of the flash-linear-attention family — implements
// this in its own file and RegisterMixer's it, with no edit to the loop.
//
// Forward returns the mixed output and the advanced shared-KV hand-off (the
// zero SharedKV when the mixer keeps all of its state inside ctx.Cache, which is
// every recurrent mixer and softmax layers that do not share KV).
//
//	out, kv := mx.Forward(normed, &MixerCtx{Cache: c, Prev: prev, B: B, L: L, Mask: mask, Window: window})
type MixerCompute interface {
	scheme.Mixer
	Forward(x *Array, ctx *MixerCtx) (*Array, SharedKV)
}

// MixerCloser is the optional capability a mixer implements when it owns Metal
// weight arrays that must be released on model Close. The composed model asserts
// it per layer and calls CloseMixer when present; a mixer that holds no arrays of
// its own (or is not yet wired for release) simply does not implement it, and the
// composed model skips it — Close stays best-effort, never a nil-deref.
type MixerCloser interface {
	CloseMixer()
}

// MixerComputeFor resolves a registered mixer by kind and asserts it carries the
// metal compute surface. The pure-Go registry (scheme.MixerFor) holds identity +
// state kind; the metal Engine needs Forward, so a metadata-only catalogue entry
// (no compute attached yet) reports (nil,false) here — the engine then refuses
// the model cleanly rather than calling a half-registered scheme.
func MixerComputeFor(kind string) (MixerCompute, bool) {
	m, ok := scheme.MixerFor(kind)
	if !ok {
		return nil, false
	}
	mc, ok := m.(MixerCompute)
	return mc, ok
}
