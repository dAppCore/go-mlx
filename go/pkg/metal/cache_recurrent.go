// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// RecurrentCache is the StateRecurrent state-holder a recurrent sequence mixer
// (GLA, RetNet, DeltaNet, Mamba2, RWKV7, GSA) reads and writes in its Forward —
// the sibling of the KV-shaped Cache. Where a softmax mixer appends K/V via
// Cache.Update, a recurrent mixer threads its own fixed-size state: read the
// prior state, run the recurrence, write the advanced state back. One holder per
// recurrent layer (the decoder hands each layer its own cache), so there is no
// layer index here.
//
//	rc, _ := ctx.Recurrent()
//	prior := rc.RecurrentState()        // nil on the first forward → mixer starts from zero
//	out, next := mixer.scan(x, prior)   // mixer owns the state shape
//	rc.SetRecurrentState(next)
type RecurrentCache interface {
	Cache
	// RecurrentState returns the mixer's stored state slots, or nil before the
	// first SetRecurrentState (the mixer then starts from its zero state). The
	// mixer owns the count and shapes — one slot for a single-matrix recurrence
	// ([B,H,Dk,Dv] linear-attn, [B,H,K,V] RWKV7), two for Mamba2 (ssm-state +
	// conv-state ring). The holder treats them opaquely.
	RecurrentState() []*Array
	// SetRecurrentState adopts the mixer's advanced state for the next forward,
	// freeing the slots it replaces. The mixer hands ownership over and must not
	// reuse the prior handles after the call.
	SetRecurrentState(state []*Array)
}

// recurrentCache is the engine's StateRecurrent holder. It persists a recurrent
// mixer's fixed-size state across forward calls and rides the same lifecycle
// (Reset/Detach/State) as a KV cache, so prompt-cache, session reset, and
// graph-detach treat a recurrent layer like any other. Its KV Update is a no-op
// that only advances the token offset — a recurrent mixer appends no K/V; it
// threads state through RecurrentState/SetRecurrentState.
type recurrentCache struct {
	state  []*Array
	offset int
}

// NewRecurrentCache creates an empty per-layer recurrent-state holder.
//
//	cache := metal.NewRecurrentCache() // one per StateRecurrent layer
func NewRecurrentCache() RecurrentCache { return &recurrentCache{} }

// Update advances the token position and returns no K/V: a recurrent layer
// contributes nothing to softmax attention, and its mixer threads state through
// RecurrentState/SetRecurrentState rather than this KV-append path. Present only
// to satisfy the Cache interface so the holder rides the shared cache lifecycle.
func (c *recurrentCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	c.offset += seqLen
	return nil, nil
}

func (c *recurrentCache) Offset() int { return c.offset }
func (c *recurrentCache) Len() int    { return c.offset }

// State returns the live recurrent-state slots so the engine's prompt-cache and
// session-snapshot paths persist a recurrent layer alongside the KV layers.
func (c *recurrentCache) State() []*Array { return c.state }

func (c *recurrentCache) Reset() {
	Free(c.state...)
	c.state = nil
	c.offset = 0
}

func (c *recurrentCache) Detach() {
	if len(c.state) > 0 {
		Detach(c.state...)
	}
}

func (c *recurrentCache) RecurrentState() []*Array { return c.state }

func (c *recurrentCache) SetRecurrentState(state []*Array) {
	// Free the superseded slots, leaving any handle the mixer re-used in place
	// (an in-place recurrence may hand the same array straight back).
	for _, old := range c.state {
		if old == nil {
			continue
		}
		reused := false
		for _, n := range state {
			if n == old {
				reused = true
				break
			}
		}
		if !reused {
			Free(old)
		}
	}
	c.state = state
}

// recurrentScheme is the StateRecurrent cache scheme — it Serves StateRecurrent
// so scheme.Compatible pairs it only with a recurrent mixer, and builds the
// per-layer holder. Registering it here overwrites the metadata-only "recurrent"
// catalogue seed (pkg/scheme/builtin.go) with the compute-bearing version — the
// standard driver-attaches-compute pattern.
type recurrentScheme struct{}

func (recurrentScheme) Mode() string               { return "recurrent" }
func (recurrentScheme) Serves() scheme.StateKind   { return scheme.StateRecurrent }
func (recurrentScheme) NewCache(CacheParams) Cache { return NewRecurrentCache() }

func init() { scheme.RegisterCache(recurrentScheme{}) }

// compile-time proofs the holder + scheme satisfy their contracts.
var (
	_ RecurrentCache = (*recurrentCache)(nil)
	_ CacheCompute   = recurrentScheme{}
)
