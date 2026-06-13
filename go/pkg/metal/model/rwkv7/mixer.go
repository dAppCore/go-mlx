// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// mixer.go makes the RWKV-7 WKV7 recurrence satisfy the engine's pluggable
// sequence-mixer contract (metal.MixerCompute + the pure-Go scheme.Mixer it
// embeds), so the decoder loop dispatches it through the scheme registry
// exactly like the softmax-hybrid reference. It mirrors
// gemma4/softmax_mixer.go: declare a Kind + State, then a single Forward that
// maps the universal metal.MixerCtx onto WKV7.

// Mixer is the RWKV-7 sequence mixer. It carries the per-layer geometry the
// recurrence needs; a model's load path constructs one per layer from the
// checkpoint and registers a representative instance with the scheme catalogue.
// Forward delegates to WKV7, which holds the family-agnostic recurrence.
type Mixer struct {
	cfg *Config
}

// Config is the per-layer WKV7 geometry the mixer needs: head count, key_dim
// (K), and value_dim (V). A model's config maps onto this at load.
type Config struct {
	NumHeads int32 // H
	KeyDim   int32 // K
	ValueDim int32 // V
}

// rwkv7MixerExtra is the per-forward arch context an RWKV-7 model hands its
// mixer via metal.MixerCtx.Extra: the already-projected WKV7 inputs for this
// chunk. The universal MixerCtx carries the chunk geometry and the recurrent
// state holder (ctx.Cache); the arch-specific projected tensors ride the escape
// hatch, exactly as Gemma-4 rides its config + mask cache there.
type rwkv7MixerExtra struct {
	in StepInput
}

// Kind is the config token an RWKV-7 build declares. The engine resolves this
// mixer with scheme.MixerFor("rwkv7").
func (m *Mixer) Kind() string { return "rwkv7" }

// State declares this mixer keeps a fixed-size recurrent [K,V] state (no growing
// K/V cache), so the engine pairs it with a recurrent cache scheme and rejects a
// KV-cache scheme for it at load (the mixer-owns-state contract).
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// Forward mixes one chunk by running the WKV7 recurrence over the projected
// inputs carried in ctx.Extra, reading the prior state from ctx.Cache and
// writing the advanced state back.
//
// TODO(#1-remainder): ctx.Cache is the KV-shaped metal.Cache today; the
// recurrent-state holder (a cache scheme whose Serves() == StateRecurrent) is
// landing in the #1 engine-rewire remainder. Until it exposes a typed
// recurrent get/put, this reads no prior state (fresh-sequence path); the
// holder will own persistence. The recurrence itself is complete and
// state-correct; only the holder wiring is deferred.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	extra, _ := ctx.Extra.(rwkv7MixerExtra)

	var prior *metal.Array
	// Recurrent holder not yet attached (see TODO above); prior stays nil →
	// WKV7 starts from a zero state. When the holder lands, prior comes from
	// ctx.Cache's recurrent get for this layer.

	o, _ := WKV7(extra.in, prior)
	// The advanced state is owned by the recurrent holder once wired; a recurrent
	// mixer keeps all state inside ctx.Cache, so it returns the zero SharedKV.
	return o, metal.SharedKV{}
}

// compile-time proof Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)
