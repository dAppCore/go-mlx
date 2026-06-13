// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// mixer.go makes the Mamba-2 selective scan satisfy the engine's pluggable
// sequence-mixer contract (metal.MixerCompute + the pure-Go scheme.Mixer it
// embeds), so the decoder loop dispatches it through the scheme registry
// exactly like the softmax-hybrid reference. It mirrors
// gemma4/softmax_mixer.go: declare a Kind + State, then a single Forward that
// maps the universal metal.MixerCtx onto SSDScan.

// Mixer is the Mamba-2 sequence mixer. It carries the per-layer SSD projection
// weights the scan needs; a model's load path constructs one per layer from the
// checkpoint and registers a representative instance with the scheme catalogue.
// Forward delegates to SSDScan, which holds the family-agnostic recurrence.
type Mixer struct {
	// Projections are applied by the model load path before SSDScan; this scaffold
	// holds the constructed inputs for the active chunk via mamba2MixerExtra. The
	// scan kernel (scan.go) is the value deliverable and is exercised directly by
	// the kernel test, independent of this scaffold.
	cfg *Config
}

// Config is the per-layer SSD geometry the mixer needs: head count, head_dim,
// and d_state. A model's config maps onto this at load.
type Config struct {
	NumHeads int32 // H
	HeadDim  int32 // P
	StateDim int32 // N (d_state)
}

// mamba2MixerExtra is the per-forward arch context a Mamba-2 model hands its
// mixer via metal.MixerCtx.Extra: the already-projected SSD inputs for this
// chunk. The universal MixerCtx carries the chunk geometry and the recurrent
// state holder (ctx.Cache); the arch-specific projected tensors ride the escape
// hatch, exactly as Gemma-4 rides its config + mask cache there.
type mamba2MixerExtra struct {
	in ScanInput
}

// Kind is the config token a Mamba-2 build declares. The engine resolves this
// mixer with scheme.MixerFor("mamba2").
func (m *Mixer) Kind() string { return "mamba2" }

// State declares this mixer keeps a fixed-size recurrent SSM state (no growing
// K/V), so the engine pairs it with a recurrent cache scheme and rejects a
// KV-cache scheme for it at load (the mixer-owns-state contract).
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// Forward mixes one chunk by running the SSD selective scan over the projected
// inputs carried in ctx.Extra, reading the prior SSM state from ctx.Cache and
// writing the advanced state back.
//
// TODO(#1-remainder): ctx.Cache is the KV-shaped metal.Cache today; the
// recurrent-state holder (a cache scheme whose Serves() == StateRecurrent) is
// landing in the #1 engine-rewire remainder. Until it exposes a typed
// recurrent get/put, this reads no prior state (fresh-sequence path) and
// returns the advanced state to the caller via the zero SharedKV's absence —
// the holder will own persistence. The scan kernel itself is complete and
// state-correct; only the holder wiring is deferred.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	extra, _ := ctx.Extra.(mamba2MixerExtra)

	var prior *metal.Array
	// Recurrent holder not yet attached (see TODO above); prior stays nil →
	// SSDScan starts from a zero state. When the holder lands, prior comes from
	// ctx.Cache's recurrent get for this layer.

	y, _ := SSDScan(extra.in, prior)
	// The advanced state is owned by the recurrent holder once wired; a recurrent
	// mixer keeps all state inside ctx.Cache, so it returns the zero SharedKV
	// (the per-layer shared-KV hand-off is a softmax-shared-KV concept).
	return y, metal.SharedKV{}
}

// compile-time proof Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)
