// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"math"

	core "dappco.re/go"
)

// exampleNativeSnapshot builds a single-layer native-dtype snapshot whose key
// is float16 and value is bfloat16 — the raw-byte capture shape produced by an
// MLX cache export, used by the round-trip examples below.
func exampleNativeSnapshot() *Snapshot {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(0.25)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(-0.75)>>16))
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1},
		TokenOffset:   1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:        []float32{1.5, -2},
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				Value:      []float32{0.25, -0.75},
				ValueDType: "bfloat16",
				ValueBytes: valueBytes,
			}},
		}},
	}
}

// ExampleSnapshot_MarshalBinary round-trips a snapshot through the
// encoding.BinaryMarshaler interface — the in-memory serialisation path State
// stores and session caches use. The decoded snapshot recovers the same
// architecture and token count as the source.
func ExampleSnapshot_MarshalBinary() {
	data, err := testSnapshot().MarshalBinary()
	if err != nil {
		core.Println("marshal error:", err)
		return
	}

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		core.Println("unmarshal error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	core.Println("tokens:", len(loaded.Tokens))
	// Output:
	// architecture: gemma4_text
	// tokens: 2
}

// ExampleSnapshot_MarshalBinary_nativeDtypes round-trips a native-dtype
// snapshot in memory. The float16 key and bfloat16 value dtype tags survive the
// encode/decode (the decoder's dtype-string reader recognises the canonical
// vocabulary), and the raw byte payloads are preserved bit-exact.
func ExampleSnapshot_MarshalBinary_nativeDtypes() {
	source := exampleNativeSnapshot()
	data, err := source.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		core.Println("encode error:", err)
		return
	}

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		core.Println("decode error:", err)
		return
	}
	head := loaded.Layers[0].Heads[0]
	core.Println("key dtype:", head.KeyDType)
	core.Println("value dtype:", head.ValueDType)
	core.Println("key bytes preserved:", equalBytes(head.KeyBytes, source.Layers[0].Heads[0].KeyBytes))
	// Output:
	// key dtype: float16
	// value dtype: bfloat16
	// key bytes preserved: true
}

// ExampleSnapshot_Head reads a single head out of a snapshot by (layer, head)
// index, returning a clone. An out-of-range head index reports ok=false.
func ExampleSnapshot_Head() {
	snapshot := testSnapshot()

	head, ok := snapshot.Head(0, 0)
	core.Println("ok:", ok, "key len:", len(head.Key))

	_, missing := snapshot.Head(0, 99)
	core.Println("missing head ok:", missing)
	// Output:
	// ok: true key len: 4
	// missing head ok: false
}

// ExampleSnapshot_Clone produces a deep copy: mutating the clone's head data
// leaves the original untouched.
func ExampleSnapshot_Clone() {
	original := testSnapshot()
	clone := original.Clone()
	clone.Layers[0].Heads[0].Key[0] = -999

	core.Println("original intact:", original.Layers[0].Heads[0].Key[0] == 1)
	core.Println("clone mutated:", clone.Layers[0].Heads[0].Key[0] == -999)
	// Output:
	// original intact: true
	// clone mutated: true
}

// ExampleLoad shows the file round-trip: Save writes a snapshot to a path and
// Load reads it back, recovering the architecture.
func ExampleLoad() {
	dir := core.MkdirTemp("", "kv-load-example-*").Value.(string)
	path := core.PathJoin(dir, "snapshot.kvbin")
	if err := testSnapshot().Save(path); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := Load(path)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output:
	// architecture: gemma4_text
}
