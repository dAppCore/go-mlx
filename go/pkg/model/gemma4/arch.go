// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma4 is the backend-agnostic DECLARATION of the Gemma 4 decode
// architecture — the "what", separated from pkg/metal/model/gemma4's imperative
// Forward (the "how"). pkg/metal currently bakes the per-layer structure (which
// layers are sliding vs global, the KV-cache-sharing map) into its forward pass and
// loader; this package owns that derivation in a pure-Go form a backend executor
// (pkg/native, pkg/metal, future go-rocm) can consume. Part three of pkg/model: the
// declarative arch over the backend contract.
//
// This file is the cache-topology axis (attention type + KV-share) — the part most
// tangled into the metal forward. MoE / per-layer-input derivation and the executor
// that runs an Arch on a backend are later slices; their LayerSpec/Arch fields are
// declared here so the shape is whole.
package gemma4

// AttentionType is a layer's attention span.
type AttentionType uint8

const (
	GlobalAttention  AttentionType = iota // full_attention — attends the whole context
	SlidingAttention                      // sliding_attention — windowed
)

// LayerSpec declares one decode layer's structure, backend-agnostic.
type LayerSpec struct {
	Attention   AttentionType
	KVShareFrom int  // index of the layer whose KV cache this layer reads (== own index if it owns its cache)
	CacheIndex  int  // cache slot for an owner; -1 if this layer shares another's cache
	MoE         bool // sparse-expert MLP instead of dense (derivation: a later slice)
}

// OwnsCache reports whether this layer holds its own KV cache (vs sharing).
func (l LayerSpec) OwnsCache() bool { return l.CacheIndex >= 0 }

// Arch is the full backend-agnostic Gemma 4 decode declaration: the neutral
// transformer dims + the gemma4-specifics + the derived per-layer specs. Built from
// a model config; consumed by a backend executor. (Dims are plain fields the loader
// fills from config; the per-layer derivation is DeriveLayers.)
type Arch struct {
	Hidden, Heads, KVHeads, HeadDim, FF, Vocab int
	Eps                                        float32
	RopeBase, RopeScale                        float32
	SoftCap                                    float32 // final logit soft-cap (0 = none)
	SlidingWindow                              int
	PerLayerInputVocab, PerLayerInputHidden    int  // gemma4 per-layer-input aux embedding (0 = absent)
	AttentionKEqV                              bool // K == V (shared projection)
	Layer                                      []LayerSpec
}

// DeriveLayers resolves the per-layer attention type and KV-cache-sharing map from a
// gemma4 config — a faithful backend-agnostic lift of pkg/metal/model/gemma4's
// buildGemma4CacheLayout plus the layer_types rule. layerTypes is the config's
// per-layer "sliding_attention"/"full_attention"; numKVShared is
// num_kv_shared_layers. Rule: the first (n − numKVShared) layers OWN their cache;
// each later layer SHARES the KV cache of the most recent owner of the same
// attention type (and is itself promoted to owner if no such owner exists yet — the
// toy-config edge). Parity-gated against the metal impl (no model load needed).
func DeriveLayers(layerTypes []string, numKVShared int) []LayerSpec {
	n := len(layerTypes)
	specs := make([]LayerSpec, n)
	firstShared := n - numKVShared
	if firstShared < 0 {
		firstShared = 0
	}
	if firstShared > n {
		firstShared = n
	}
	latestByType := map[AttentionType]int{}
	nextCache := 0
	for i := 0; i < n; i++ {
		at := GlobalAttention
		if layerTypes[i] == "sliding_attention" {
			at = SlidingAttention
		}
		specs[i] = LayerSpec{Attention: at, KVShareFrom: i, CacheIndex: -1}
		owns := i < firstShared
		if !owns {
			if prev, ok := latestByType[at]; ok {
				specs[i].KVShareFrom = prev
			} else {
				owns = true // first layer of this type lands in the shared region → promote to owner
			}
		}
		if owns {
			specs[i].KVShareFrom = i
			latestByType[at] = i
			specs[i].CacheIndex = nextCache
			nextCache++
		}
	}
	return specs
}
