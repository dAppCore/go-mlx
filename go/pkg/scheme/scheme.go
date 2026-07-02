// SPDX-Licence-Identifier: EUPL-1.2

// Package scheme is a thin compatibility layer over the shared
// dappco.re/go/inference/scheme package — the engine's pluggable-component
// contract layer (weight quant, KV/state cache, sequence mixer) that
// relocated there so every inference engine (metal on Apple, rocm on
// AMD/CUDA/CPU) inherits ONE scheme catalogue instead of a per-engine copy.
// Every type below is a genuine Go type alias and every function delegates
// straight through, so this package and the shared one read and write the
// SAME registry: a scheme registered via either import path resolves via the
// other. This import path is the compatibility shim that keeps pkg/native
// and pkg/metal compiling unchanged until they migrate to importing the
// shared package directly.
//
//	q, _ := scheme.QuantFor(cfg.QuantKind)   // "affine", "q4_0", "mxfp4", …
//	m, _ := scheme.MixerFor(cfg.MixerKind)   // "softmax-hybrid", "gla", "mamba2", …
//	c, _ := scheme.CacheFor(cfg.KVCacheMode) // "q8", "turboquant", "compaction", …
//	if !scheme.Compatible(m, c) { /* mixer needs a state this cache can't hold */ }
//
// See dappco.re/go/inference/scheme for the full contract documentation —
// StateKind/Mixer/CacheScheme/QuantScheme/DType and the mixer-owns-state
// design note live there now.
package scheme

import (
	core "dappco.re/go"
	shared "dappco.re/go/inference/scheme"
)

// StateKind aliases the shared StateKind — see
// dappco.re/go/inference/scheme.StateKind for the full contract (String()
// comes along for free: an alias is the same type, not a new one).
type StateKind = shared.StateKind

// StateNone, StateKVCache, StateRecurrent alias the shared StateKind values.
const (
	StateNone      = shared.StateNone
	StateKVCache   = shared.StateKVCache
	StateRecurrent = shared.StateRecurrent
)

// Mixer, CacheScheme, QuantScheme, DType alias the shared contracts — see
// dappco.re/go/inference/scheme for the full documentation of each.
type (
	Mixer       = shared.Mixer
	CacheScheme = shared.CacheScheme
	QuantScheme = shared.QuantScheme
	DType       = shared.DType
)

// RegisterMixer, RegisterCache, RegisterQuant, RegisterDType delegate to the
// shared registry — ONE store, two import paths: a scheme registered here
// resolves through dappco.re/go/inference/scheme too, and vice versa.
func RegisterMixer(m Mixer) core.Result       { return shared.RegisterMixer(m) }
func RegisterCache(c CacheScheme) core.Result { return shared.RegisterCache(c) }
func RegisterQuant(q QuantScheme) core.Result { return shared.RegisterQuant(q) }
func RegisterDType(d DType) core.Result       { return shared.RegisterDType(d) }

// MixerFor, CacheFor, QuantFor, DTypeFor resolve a registered scheme by its
// config token, delegating to the shared registry.
func MixerFor(kind string) (Mixer, bool)       { return shared.MixerFor(kind) }
func CacheFor(mode string) (CacheScheme, bool) { return shared.CacheFor(mode) }
func QuantFor(kind string) (QuantScheme, bool) { return shared.QuantFor(kind) }
func DTypeFor(name string) (DType, bool)       { return shared.DTypeFor(name) }

// MixerKinds, CacheModes, QuantKinds, DTypeNames list the registered names in
// registration order — the engine's "what can I load" catalogue.
func MixerKinds() []string { return shared.MixerKinds() }
func CacheModes() []string { return shared.CacheModes() }
func QuantKinds() []string { return shared.QuantKinds() }
func DTypeNames() []string { return shared.DTypeNames() }

// Compatible enforces the mixer-owns-state contract: a cache scheme may serve
// a mixer only if it holds the state kind the mixer declares it needs.
func Compatible(m Mixer, c CacheScheme) bool { return shared.Compatible(m, c) }
