// SPDX-Licence-Identifier: EUPL-1.2

package kvconv

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
)

// kv_snapshot_convert.go: marshalling between the root kv.Snapshot surface and
// metal.KVSnapshot — TurboQuant reference payloads and KV head dtype tagging.

func ToRootKVSnapshot(result *metal.KVSnapshot) *kv.Snapshot {
	if result == nil {
		return nil
	}
	resultLayers := result.Layers
	layers := make([]kv.LayerSnapshot, len(resultLayers))
	// Single arena allocation for all per-layer Heads slices. Avoids N
	// small allocations on a path that runs per KV capture / restore.
	totalHeads := 0
	totalKey := 0
	totalValue := 0
	totalKeyBytes := 0
	totalValueBytes := 0
	// totalInt32 covers per-layer KeyShape + ValueShape AND the top-level
	// Tokens + Generated + LogitShape slices — all share the same int32
	// element type and the same once-per-snapshot lifetime, so they share
	// one arena. Drops 3 + 2×layers small clones to 1 outer alloc.
	totalInt32 := len(result.Tokens) + len(result.Generated) + len(result.LogitShape)
	totalLogits := len(result.Logits)
	for i := range resultLayers {
		layer := &resultLayers[i]
		heads := layer.Heads
		totalHeads += len(heads)
		totalInt32 += len(layer.KeyShape) + len(layer.ValueShape)
		for j := range heads {
			head := &heads[j]
			totalKey += len(head.Key)
			totalValue += len(head.Value)
			totalKeyBytes += len(head.KeyBytes)
			totalValueBytes += len(head.ValueBytes)
		}
	}
	headsSlab := make([]kv.HeadSnapshot, totalHeads)
	// One float32 slab covers per-head Key + per-head Value + top-level
	// Logits — all are []float32 with once-per-snapshot lifetime. Previous
	// shape: 2 head-family slabs + 1 standalone Logits clone = 3 allocs;
	// unified: 1 alloc regardless of (layers × heads × Logits len).
	// keyOffset / valueOffset / logitsOffset partition the slab into the
	// three regions without ever overlapping (offsets are monotonic and
	// total exactly totalFloat32). 3-cap sub-slicing keeps each sub-region
	// safely append-bounded against neighbours.
	totalFloat32 := totalKey + totalValue + totalLogits
	var float32Slab []float32
	if totalFloat32 > 0 {
		float32Slab = make([]float32, totalFloat32)
	}
	// Same pattern for per-head KeyBytes + ValueBytes — both []byte, both
	// once-per-snapshot — one byteSlab instead of two outer allocs.
	totalBytes := totalKeyBytes + totalValueBytes
	var byteSlab []byte
	if totalBytes > 0 {
		byteSlab = make([]byte, totalBytes)
	}
	var int32Slab []int32
	if totalInt32 > 0 {
		int32Slab = make([]int32, totalInt32)
	}
	headsOffset := 0
	keyOffset := 0
	// value region begins where key region ends.
	valueOffset := totalKey
	// logits region begins where value region ends (we lay it down at the
	// end below).
	logitsOffset := totalKey + totalValue
	keyBytesOffset := 0
	// valueBytes region begins where keyBytes region ends.
	valueBytesOffset := totalKeyBytes
	int32Offset := 0
	// Index iteration on both loops — KVLayerSnapshot is ~136 B (4 slice
	// headers + 2 strings + 2 byte-slice headers) and KVHeadSnapshot is
	// ~160 B (6 slice headers + 2 dtype strings); for deep models (Gemma
	// 4 E4B = 30 layers × 16 heads = 480 head-copies per snapshot)
	// the range-and-copy intermediate variable was 100+ KB of redundant
	// stack copies per capture. Read fields direct from resultLayers[i].
	for i := range resultLayers {
		layer := &resultLayers[i]
		layerHeadsSrc := layer.Heads
		headsEnd := headsOffset + len(layerHeadsSrc)
		layerHeads := headsSlab[headsOffset:headsEnd:headsEnd]
		// Per-layer shape clones cut from the shared int32 arena.
		var keyShape, valueShape []int32
		switch {
		case layer.KeyShape == nil:
		case len(layer.KeyShape) == 0:
			keyShape = []int32{}
		default:
			end := int32Offset + len(layer.KeyShape)
			keyShape = int32Slab[int32Offset:end:end]
			copy(keyShape, layer.KeyShape)
			int32Offset = end
		}
		switch {
		case layer.ValueShape == nil:
		case len(layer.ValueShape) == 0:
			valueShape = []int32{}
		default:
			end := int32Offset + len(layer.ValueShape)
			valueShape = int32Slab[int32Offset:end:end]
			copy(valueShape, layer.ValueShape)
			int32Offset = end
		}
		layers[i] = kv.LayerSnapshot{
			Layer:              layer.Layer,
			CacheIndex:         layer.CacheIndex,
			CacheMode:          string(layer.CacheMode),
			TurboQuantPayloads: rootTurboQuantPayloads(layer.TurboQuantPayloads),
			KeyDType:           RootKVHeadDType(layer.KeyDType, layer.KeyBytes),
			KeyBytes:           layer.KeyBytes,
			KeyShape:           keyShape,
			ValueDType:         RootKVHeadDType(layer.ValueDType, layer.ValueBytes),
			ValueBytes:         layer.ValueBytes,
			ValueShape:         valueShape,
			Heads:              layerHeads,
		}
		for j := range layerHeadsSrc {
			head := &layerHeadsSrc[j]
			// Allocate per-head slices out of the pre-sized arenas. Each
			// branch preserves the prior nil-in -> nil-out / empty-in ->
			// empty-out semantics of core.SliceClone so downstream
			// callers see identical post-clone shape.
			var headKey []float32
			switch {
			case head.Key == nil:
				// nil in -> nil out
			case len(head.Key) == 0:
				headKey = []float32{}
			default:
				end := keyOffset + len(head.Key)
				headKey = float32Slab[keyOffset:end:end]
				copy(headKey, head.Key)
				keyOffset = end
			}
			var headValue []float32
			switch {
			case head.Value == nil:
			case len(head.Value) == 0:
				headValue = []float32{}
			default:
				end := valueOffset + len(head.Value)
				headValue = float32Slab[valueOffset:end:end]
				copy(headValue, head.Value)
				valueOffset = end
			}
			var headKeyBytes []byte
			switch {
			case head.KeyBytes == nil:
			case len(head.KeyBytes) == 0:
				headKeyBytes = []byte{}
			default:
				end := keyBytesOffset + len(head.KeyBytes)
				headKeyBytes = byteSlab[keyBytesOffset:end:end]
				copy(headKeyBytes, head.KeyBytes)
				keyBytesOffset = end
			}
			var headValueBytes []byte
			switch {
			case head.ValueBytes == nil:
			case len(head.ValueBytes) == 0:
				headValueBytes = []byte{}
			default:
				end := valueBytesOffset + len(head.ValueBytes)
				headValueBytes = byteSlab[valueBytesOffset:end:end]
				copy(headValueBytes, head.ValueBytes)
				valueBytesOffset = end
			}
			layerHeads[j] = kv.HeadSnapshot{
				Key:        headKey,
				KeyDType:   RootKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   headKeyBytes,
				Value:      headValue,
				ValueDType: RootKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: headValueBytes,
			}
		}
		headsOffset = headsEnd
	}
	// Top-level int32 slices share the same arena as the per-layer shape
	// clones — preserves the same nil-in/empty-in/non-empty semantics
	// core.SliceClone provided so downstream callers see no change.
	var tokens, generated, logitShape []int32
	switch {
	case result.Tokens == nil:
	case len(result.Tokens) == 0:
		tokens = []int32{}
	default:
		end := int32Offset + len(result.Tokens)
		tokens = int32Slab[int32Offset:end:end]
		copy(tokens, result.Tokens)
		int32Offset = end
	}
	switch {
	case result.Generated == nil:
	case len(result.Generated) == 0:
		generated = []int32{}
	default:
		end := int32Offset + len(result.Generated)
		generated = int32Slab[int32Offset:end:end]
		copy(generated, result.Generated)
		int32Offset = end
	}
	switch {
	case result.LogitShape == nil:
	case len(result.LogitShape) == 0:
		logitShape = []int32{}
	default:
		end := int32Offset + len(result.LogitShape)
		logitShape = int32Slab[int32Offset:end:end]
		copy(logitShape, result.LogitShape)
		int32Offset = end
	}
	// Top-level Logits sits in the tail region of the shared float32 slab.
	var topLogits []float32
	switch {
	case result.Logits == nil:
	case len(result.Logits) == 0:
		topLogits = []float32{}
	default:
		end := logitsOffset + len(result.Logits)
		topLogits = float32Slab[logitsOffset:end:end]
		copy(topLogits, result.Logits)
		logitsOffset = end
	}
	return &kv.Snapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    logitShape,
		Logits:        topLogits,
		Layers:        layers,
	}
}

// kvLayerHasNativeSlab reports whether a layer carries native K/V slab
// bytes. When true the metal restorer pins those bytes zero-copy and never
// reads the layer's per-head float32, so ToMetalKVSnapshot can skip the
// per-head materialisation. Both K and V must be present — a half-native
// layer would still hit the heads decode path on the missing side.
//
//	kvLayerHasNativeSlab(&kv.LayerSnapshot{KeyBytes: b, ValueBytes: b}) // true
func kvLayerHasNativeSlab(layer *kv.LayerSnapshot) bool {
	return len(layer.KeyBytes) > 0 && len(layer.ValueBytes) > 0
}

func rootTurboQuantPayloads(payloads []metal.TurboQuantKVReferencePagePayload) [][]byte {
	if len(payloads) == 0 {
		return nil
	}
	out := make([][]byte, 0, len(payloads))
	for idx := range payloads {
		encoded := core.JSONMarshal(payloads[idx])
		if !encoded.OK {
			return nil
		}
		out = append(out, core.SliceClone(encoded.Value.([]byte)))
	}
	return out
}

func metalTurboQuantPayloads(payloads [][]byte) []metal.TurboQuantKVReferencePagePayload {
	if len(payloads) == 0 {
		return nil
	}
	out := make([]metal.TurboQuantKVReferencePagePayload, 0, len(payloads))
	for idx := range payloads {
		if len(payloads[idx]) == 0 {
			return nil
		}
		var payload metal.TurboQuantKVReferencePagePayload
		if result := core.JSONUnmarshal(payloads[idx], &payload); !result.OK {
			return nil
		}
		if err := payload.Layout.Validate(); err != nil {
			return nil
		}
		out = append(out, payload)
	}
	return out
}

func ToMetalKVSnapshot(result *kv.Snapshot) *metal.KVSnapshot {
	if result == nil {
		return nil
	}
	resultLayers := result.Layers
	layers := make([]metal.KVLayerSnapshot, len(resultLayers))
	// Single arena allocations for the per-layer Heads slices and the
	// per-head Key + Value tensor copies. The inverse direction only
	// clones Key + Value (KeyBytes / ValueBytes pass through by reference
	// from the root side), so the per-head alloc budget is 2 instead of
	// ToRootKVSnapshot's 4. Coalescing into single float32 slabs drops
	// 2×heads small allocations to 2 outer allocations regardless of
	// (layers × heads). Gemma 4 E4B (30 × 16 = 480 heads) goes from 960
	// to 2 per snapshot.
	totalHeads := 0
	totalKey := 0
	totalValue := 0
	// totalInt32 covers per-layer KeyShape + ValueShape AND the top-level
	// Tokens + Generated + LogitShape slices — all share the same int32
	// element type and the same once-per-snapshot lifetime, so they share
	// one arena. Drops 3 + 2×layers small clones to 1 outer alloc.
	totalInt32 := len(result.Tokens) + len(result.Generated) + len(result.LogitShape)
	totalLogits := len(result.Logits)
	for i := range resultLayers {
		layer := &resultLayers[i]
		heads := layer.Heads
		totalHeads += len(heads)
		totalInt32 += len(layer.KeyShape) + len(layer.ValueShape)
		// When a layer carries native K/V slab bytes the metal restorer
		// reads ONLY those bytes (kvLayerArrays takes the native-slab
		// branch and ignores per-head Key/Value); the decoded per-head
		// float32 are dead weight. A v4 snapshot loaded with the default
		// (non-RawKVOnly) options populates BOTH — copying the heads here
		// would materialise the entire prefix cache a second time alongside
		// the byte slab the restorer actually pins zero-copy. Skip them.
		if kvLayerHasNativeSlab(layer) {
			continue
		}
		for j := range heads {
			head := &heads[j]
			totalKey += len(head.Key)
			totalValue += len(head.Value)
		}
	}
	headsSlab := make([]metal.KVHeadSnapshot, totalHeads)
	// One float32 slab covers per-head Key + per-head Value + top-level
	// Logits — all []float32, all once-per-snapshot. Previous shape was
	// 2 head-family slabs + 1 standalone Logits clone = 3 outer allocs;
	// unified: 1 alloc regardless of (layers × heads × Logits len).
	totalFloat32 := totalKey + totalValue + totalLogits
	var float32Slab []float32
	if totalFloat32 > 0 {
		float32Slab = make([]float32, totalFloat32)
	}
	var int32Slab []int32
	if totalInt32 > 0 {
		int32Slab = make([]int32, totalInt32)
	}
	headsOffset := 0
	keyOffset := 0
	// value region begins where key region ends.
	valueOffset := totalKey
	// logits region begins where value region ends.
	logitsOffset := totalKey + totalValue
	int32Offset := 0
	// Index iteration — see ToRootKVSnapshot for rationale; same N×layer
	// + N×head struct-copy elision on the inverse direction.
	for i := range resultLayers {
		layer := &resultLayers[i]
		layerHeadsSrc := layer.Heads
		headsEnd := headsOffset + len(layerHeadsSrc)
		layerHeads := headsSlab[headsOffset:headsEnd:headsEnd]
		// Per-layer shape clones cut from the shared arena.
		var keyShape, valueShape []int32
		switch {
		case layer.KeyShape == nil:
		case len(layer.KeyShape) == 0:
			keyShape = []int32{}
		default:
			end := int32Offset + len(layer.KeyShape)
			keyShape = int32Slab[int32Offset:end:end]
			copy(keyShape, layer.KeyShape)
			int32Offset = end
		}
		switch {
		case layer.ValueShape == nil:
		case len(layer.ValueShape) == 0:
			valueShape = []int32{}
		default:
			end := int32Offset + len(layer.ValueShape)
			valueShape = int32Slab[int32Offset:end:end]
			copy(valueShape, layer.ValueShape)
			int32Offset = end
		}
		layers[i] = metal.KVLayerSnapshot{
			Layer:              layer.Layer,
			CacheIndex:         layer.CacheIndex,
			CacheMode:          metal.KVCacheMode(layer.CacheMode),
			TurboQuantPayloads: metalTurboQuantPayloads(layer.TurboQuantPayloads),
			KeyDType:           MetalKVHeadDType(layer.KeyDType, layer.KeyBytes),
			KeyBytes:           layer.KeyBytes,
			KeyShape:           keyShape,
			ValueDType:         MetalKVHeadDType(layer.ValueDType, layer.ValueBytes),
			ValueBytes:         layer.ValueBytes,
			ValueShape:         valueShape,
			Heads:              layerHeads,
		}
		// Native-slab layers never have their per-head float32 read by the
		// restorer (see the sizing-loop note), so pass the source slices
		// through by reference — same ownership contract as KeyBytes above,
		// where the source snapshot already outlives the metal snapshot for
		// the duration of the restore call. Zero copy, zero slab footprint.
		layerNative := kvLayerHasNativeSlab(layer)
		for j := range layerHeadsSrc {
			head := &layerHeadsSrc[j]
			// Allocate per-head Key + Value out of the pre-sized arenas;
			// preserve the prior nil-in -> nil-out / empty-in -> empty-out
			// shape of core.SliceClone so downstream metal sees no
			// behavioural change.
			var headKey []float32
			switch {
			case layerNative:
				headKey = head.Key
			case head.Key == nil:
				// nil in -> nil out
			case len(head.Key) == 0:
				headKey = []float32{}
			default:
				end := keyOffset + len(head.Key)
				headKey = float32Slab[keyOffset:end:end]
				copy(headKey, head.Key)
				keyOffset = end
			}
			var headValue []float32
			switch {
			case layerNative:
				headValue = head.Value
			case head.Value == nil:
			case len(head.Value) == 0:
				headValue = []float32{}
			default:
				end := valueOffset + len(head.Value)
				headValue = float32Slab[valueOffset:end:end]
				copy(headValue, head.Value)
				valueOffset = end
			}
			layerHeads[j] = metal.KVHeadSnapshot{
				Key:        headKey,
				KeyDType:   MetalKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   head.KeyBytes,
				Value:      headValue,
				ValueDType: MetalKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: head.ValueBytes,
			}
		}
		headsOffset = headsEnd
	}
	// Top-level int32 slices share the same arena as the per-layer shape
	// clones — preserves the same nil-in/empty-in/non-empty semantics
	// core.SliceClone provided so downstream callers see no change.
	var tokens, generated, logitShape []int32
	switch {
	case result.Tokens == nil:
	case len(result.Tokens) == 0:
		tokens = []int32{}
	default:
		end := int32Offset + len(result.Tokens)
		tokens = int32Slab[int32Offset:end:end]
		copy(tokens, result.Tokens)
		int32Offset = end
	}
	switch {
	case result.Generated == nil:
	case len(result.Generated) == 0:
		generated = []int32{}
	default:
		end := int32Offset + len(result.Generated)
		generated = int32Slab[int32Offset:end:end]
		copy(generated, result.Generated)
		int32Offset = end
	}
	switch {
	case result.LogitShape == nil:
	case len(result.LogitShape) == 0:
		logitShape = []int32{}
	default:
		end := int32Offset + len(result.LogitShape)
		logitShape = int32Slab[int32Offset:end:end]
		copy(logitShape, result.LogitShape)
		int32Offset = end
	}
	// Top-level Logits sits in the tail region of the shared float32 slab.
	var topLogits []float32
	switch {
	case result.Logits == nil:
	case len(result.Logits) == 0:
		topLogits = []float32{}
	default:
		end := logitsOffset + len(result.Logits)
		topLogits = float32Slab[logitsOffset:end:end]
		copy(topLogits, result.Logits)
		logitsOffset = end
	}
	return &metal.KVSnapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    logitShape,
		Logits:        topLogits,
		Layers:        layers,
	}
}

func ToMetalKVSnapshotCaptureOptions(opts kv.CaptureOptions) metal.KVSnapshotCaptureOptions {
	return metal.KVSnapshotCaptureOptions{RawKVOnly: opts.RawKVOnly}
}

func RootKVHeadDType(dtype metal.DType, raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	// Inline the three KV-supported dtype names to avoid the dtype.String()
	// map lookup. Called per-head inside the KV snapshot clone hot path —
	// thousands of invocations per snapshot.
	switch dtype {
	case metal.DTypeFloat32:
		return "float32"
	case metal.DTypeFloat16:
		return "float16"
	case metal.DTypeBFloat16:
		return "bfloat16"
	default:
		return ""
	}
}

func MetalKVHeadDType(dtype string, raw []byte) metal.DType {
	if len(raw) == 0 {
		return 0
	}
	switch dtype {
	case "float32", "F32":
		return metal.DTypeFloat32
	case "float16", "F16":
		return metal.DTypeFloat16
	case "bfloat16", "BF16":
		return metal.DTypeBFloat16
	default:
		return 0
	}
}
