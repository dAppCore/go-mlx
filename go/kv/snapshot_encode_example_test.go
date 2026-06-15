// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
)

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
