// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

func TestKVSnapshot_Clone_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Tokens:       []int32{1, 2},
		Generated:    []int32{2},
		TokenOffset:  4,
		Architecture: "gemma4_text",
		LogitShape:   []int32{1, 1, 3},
		Logits:       []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	cloned := snapshot.Clone()
	cloned.Tokens[0] = 99
	cloned.Generated[0] = 88
	cloned.Logits[0] = 0.9
	cloned.LogitShape[0] = 9
	cloned.Layers[0].Heads[0].Key[0] = 88

	if snapshot.Tokens[0] != 1 || snapshot.Generated[0] != 2 || snapshot.Logits[0] != 0.1 || snapshot.LogitShape[0] != 1 || snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Clone() returned aliased snapshot data")
	}
}

func TestKVSnapshot_SaveLoadRestorable_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{11, 12},
		Generated:     []int32{12},
		TokenOffset:   9,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 4},
		Logits:        []float32{0.1, 0.2, 0.3, 0.4},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "restorable.kvbin")

	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := Load(path)

	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Version != SnapshotVersion || loaded.TokenOffset != 9 || loaded.Generated[0] != 12 {
		t.Fatalf("loaded version/offset/generated = %d/%d/%v", loaded.Version, loaded.TokenOffset, loaded.Generated)
	}
	if len(loaded.LogitShape) != 3 || loaded.LogitShape[2] != 4 || len(loaded.Logits) != 4 || loaded.Logits[3] != 0.4 {
		t.Fatalf("loaded logits = shape %v values %v", loaded.LogitShape, loaded.Logits)
	}
}

func TestKVSnapshot_MarshalUnmarshalBinary_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{11, 12},
		Generated:     []int32{12},
		TokenOffset:   9,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	data, err := snapshot.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	if legacy, err := snapshot.bytes(); err != nil || !equalBytes(data, legacy) {
		t.Fatalf("bytes() = %d/%v, want MarshalBinary bytes %d", len(legacy), err, len(data))
	}
	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary() error = %v", err)
	}
	if loaded.TokenOffset != 9 || len(loaded.Tokens) != 2 || loaded.Layers[0].Heads[0].Value[3] != 8 {
		t.Fatalf("loaded snapshot = %+v, want marshalled state", loaded)
	}
	parsed, err := parseKVSnapshot(data)
	if err != nil {
		t.Fatalf("parseKVSnapshot() error = %v", err)
	}
	if parsed.Architecture != snapshot.Architecture || parsed.NumHeads != 1 {
		t.Fatalf("parsed snapshot = %+v, want architecture metadata", parsed)
	}
}

func TestKVSnapshot_Q8ValidateBitTricks_Good(t *testing.T) {
	// Bit-trick validate (NaN/Inf detect via exp mask + abs via bit-clear)
	// must produce maxAbs identical to the prior math.Abs walk and reject
	// the same NaN/Inf inputs as math.IsNaN/math.IsInf would.
	probes := []struct {
		name string
		vals []float32
		ok   bool
		max  float32
	}{
		{name: "positive", vals: []float32{0.5, 1.0, 1.5, 0.25}, ok: true, max: 1.5},
		{name: "negative", vals: []float32{-0.5, -1.0, -1.5, -0.25}, ok: true, max: 1.5},
		{name: "mixed", vals: []float32{-1.0, 2.0, -3.0, 0.5, -0.25, 0.75, 1.25, -1.5}, ok: true, max: 3.0},
		{name: "zero", vals: []float32{0, 0, 0, 0}, ok: true, max: 0},
		{name: "scalar-tail", vals: []float32{0.5, -0.5, 1.0}, ok: true, max: 1.0},
		{name: "nan-in-block", vals: []float32{1, 2, float32(math.NaN()), 3}, ok: false},
		{name: "nan-in-tail", vals: []float32{1, 2, 3, 4, float32(math.NaN())}, ok: false},
		{name: "posinf", vals: []float32{1, 2, float32(math.Inf(1))}, ok: false},
		{name: "neginf", vals: []float32{1, 2, float32(math.Inf(-1))}, ok: false},
	}
	for _, probe := range probes {
		maxAbs, ok := kvSnapshotQ8Validate(probe.vals)
		if ok != probe.ok {
			t.Fatalf("%s: ok = %v, want %v", probe.name, ok, probe.ok)
		}
		if ok && maxAbs != probe.max {
			t.Fatalf("%s: maxAbs = %v, want %v", probe.name, maxAbs, probe.max)
		}
	}
}

func TestKVSnapshot_SaveLoadQuantizedQ8_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 2},
		Logits:        []float32{0.25, 0.75},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{-1, -0.5, 0.5, 1},
				Value: []float32{0, 0.25, -0.25, 0.75},
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "quantized-q8.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingQ8}); err != nil {
		t.Fatalf("SaveWithOptions() error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if loaded.Version != SnapshotVersion {
		t.Fatalf("loaded Version = %d, want %d", loaded.Version, SnapshotVersion)
	}
	for i, want := range snapshot.Layers[0].Heads[0].Key {
		if diff := loaded.Layers[0].Heads[0].Key[i] - want; diff < -0.01 || diff > 0.01 {
			t.Fatalf("loaded key[%d] = %f, want near %f", i, loaded.Layers[0].Heads[0].Key[i], want)
		}
	}
	if loaded.Logits[1] != 0.75 {
		t.Fatalf("loaded logits = %v, want unquantized logits preserved", loaded.Logits)
	}
}

func TestKVSnapshot_SaveLoadNativeDType_Good(t *testing.T) {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(0.25)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(-0.75)>>16))
	snapshot := &Snapshot{
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
	path := core.PathJoin(t.TempDir(), "native-dtype.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		t.Fatalf("SaveWithOptions(native) error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	head := loaded.Layers[0].Heads[0]
	if head.KeyDType != "float16" || head.ValueDType != "bfloat16" {
		t.Fatalf("loaded dtypes = %q/%q, want float16/bfloat16", head.KeyDType, head.ValueDType)
	}
	if !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
		t.Fatalf("loaded native bytes = %v/%v, want %v/%v", head.KeyBytes, head.ValueBytes, keyBytes, valueBytes)
	}
	if diff := head.Key[0] - 1.5; diff < -0.001 || diff > 0.001 {
		t.Fatalf("loaded f16 key[0] = %f, want near 1.5", head.Key[0])
	}
	if got := binary.LittleEndian.Uint16(head.ValueBytes); got != binary.LittleEndian.Uint16(valueBytes) {
		t.Fatalf("loaded bf16 value bits = %#x, want %#x", got, binary.LittleEndian.Uint16(valueBytes))
	}
}

func TestKVSnapshot_SaveLoadNativeRawOnly_Good(t *testing.T) {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(3))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(4))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(5)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(6)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(7)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(8)>>16))
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				ValueDType: "bfloat16",
				ValueBytes: valueBytes,
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "native-raw-only.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		t.Fatalf("SaveWithOptions(native raw-only) error = %v", err)
	}
	rawOnly, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadWithOptions(raw-only) error = %v", err)
	}
	head := rawOnly.Layers[0].Heads[0]
	if len(head.Key) != 0 || len(head.Value) != 0 {
		t.Fatalf("raw-only load decoded float32 key/value lengths = %d/%d, want 0/0", len(head.Key), len(head.Value))
	}
	if head.KeyDType != "float16" || head.ValueDType != "bfloat16" || !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
		t.Fatalf("raw-only head = %+v, want native bytes preserved", head)
	}

	decoded, err := Load(path)
	if err != nil {
		t.Fatalf("Load(default) error = %v", err)
	}
	decodedHead := decoded.Layers[0].Heads[0]
	if len(decodedHead.Key) != 4 || len(decodedHead.Value) != 4 || decodedHead.Key[3] != 4 {
		t.Fatalf("default load head = %+v, want decoded float32 values for debugging", decodedHead)
	}
}

func TestKVSnapshot_SaveLoadNativeLayerRawOnly_Good(t *testing.T) {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(3))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(4))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(5)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(6)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(7)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(8)>>16))
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 2, 1},
			ValueDType: "bfloat16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 2, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}
	path := core.PathJoin(t.TempDir(), "native-layer-raw-only.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		t.Fatalf("SaveWithOptions(native layer raw-only) error = %v", err)
	}
	loaded, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadWithOptions(native layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if loaded.Version != SnapshotVersion || !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("loaded native layer = version:%d key:%v value:%v", loaded.Version, layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 || len(layer.Heads[1].ValueBytes) != 0 {
		t.Fatalf("loaded heads = %+v, want shape-only heads without duplicated raw bytes", layer.Heads)
	}
	if len(layer.KeyShape) != 4 || layer.KeyShape[1] != 2 || layer.KeyShape[2] != 2 {
		t.Fatalf("loaded key shape = %v, want [1 2 2 1]", layer.KeyShape)
	}
}

func TestKVSnapshot_EncodedSizeMatchesSerialisedBytes_Good(t *testing.T) {
	nativeKey := appendUint16LE(nil, float32ToFloat16(1))
	nativeKey = appendUint16LE(nativeKey, float32ToFloat16(2))
	nativeValue := appendUint16LE(nil, uint16(math.Float32bits(3)>>16))
	nativeValue = appendUint16LE(nativeValue, uint16(math.Float32bits(4)>>16))
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{3},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 2},
		Logits:        []float32{0.25, 0.75},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:        []float32{1, 2},
				KeyDType:   "float16",
				KeyBytes:   nativeKey,
				Value:      []float32{3, 4},
				ValueDType: "bfloat16",
				ValueBytes: nativeValue,
			}},
		}},
	}
	for _, opts := range []SaveOptions{
		{},
		{KVEncoding: EncodingQ8},
		{KVEncoding: EncodingNative},
	} {
		size, err := snapshot.encodedSizeWithOptions(opts)
		if err != nil {
			t.Fatalf("encodedSizeWithOptions(%q) error = %v", opts.KVEncoding, err)
		}
		data, err := snapshot.bytesWithOptions(opts)
		if err != nil {
			t.Fatalf("bytesWithOptions(%q) error = %v", opts.KVEncoding, err)
		}
		if size != len(data) {
			t.Fatalf("encodedSizeWithOptions(%q) = %d, serialised bytes = %d", opts.KVEncoding, size, len(data))
		}
	}
}

func TestKVSnapshot_SaveWithOptions_Bad(t *testing.T) {
	snapshot := &Snapshot{Version: SnapshotVersion}

	err := snapshot.SaveWithOptions(core.PathJoin(t.TempDir(), "bad.kvbin"), SaveOptions{KVEncoding: "q2"})

	if err == nil {
		t.Fatal("SaveWithOptions() error = nil, want unsupported encoding error")
	}
}

func TestKVSnapshot_TurboQuantPayloadMetadata_Bad(t *testing.T) {
	withPayload := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1},
		TokenOffset:   1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:              0,
			CacheIndex:         0,
			CacheMode:          "paged",
			TurboQuantPayloads: [][]byte{{1, 2, 3}},
		}},
	}

	if _, err := withPayload.MarshalBinary(); err == nil || !core.Contains(err.Error(), "TurboQuant KV payload requires turboquant cache mode") {
		t.Fatalf("MarshalBinary() error = %v, want TurboQuant cache-mode mismatch", err)
	}

	missingPayload := kvSnapshotTurboQuantNoPayloadBytes()
	var loaded Snapshot
	if err := loaded.UnmarshalBinary(missingPayload); err == nil || !core.Contains(err.Error(), "turboquant cache mode requires TurboQuant KV payload") {
		t.Fatalf("UnmarshalBinary(turboquant without payload) error = %v, want fail-closed TurboQuant payload error", err)
	}
}

func TestKVSnapshot_BinaryAPIs_Bad(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.MarshalBinary(); err == nil {
		t.Fatal("MarshalBinary(nil) error = nil")
	}
	if err := snapshot.UnmarshalBinary([]byte(kvSnapshotMagic)); err == nil {
		t.Fatal("UnmarshalBinary(nil) error = nil")
	}
}

func kvSnapshotTurboQuantNoPayloadBytes() []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, SnapshotVersion)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // layers
	data = appendKVU32(data, 0) // heads
	data = appendKVU32(data, 0) // seq len
	data = appendKVU32(data, 0) // head dim
	data = appendKVU32(data, 0) // query heads
	data = appendKVU32(data, 0) // token offset
	data = appendKVU32(data, 0) // tokens
	data = appendKVU32(data, 0) // generated
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0)
	data = appendKVI32(data, 0)
	data = appendKVU32(data, 0) // head count
	data = appendKVBytes(data, core.AsBytes("turboquant"))
	data = appendKVU32(data, 0) // TurboQuant payload count
	data = appendKVU32(data, 0) // max size (v6)
	data = appendKVI32s(data, nil)
	data = appendKVU32(data, 0) // key tensor encoding
	data = appendKVU32(data, 0) // key tensor values
	data = appendKVI32s(data, nil)
	data = appendKVU32(data, 0) // value tensor encoding
	data = appendKVU32(data, 0) // value tensor values
	data = appendKVU32(data, 0) // logit shape
	data = appendKVF32s(data, nil)
	return data
}

func TestKVSnapshot_SaveLoadShortFormDType_Good(t *testing.T) {
	// The native reader/writer accept both long ("float16") and short
	// ("F16") dtype tags. The short forms travel a separate dtypeString
	// fast-path; round-trip them to assert the canonical short tag and
	// raw bytes survive bit-exact.
	keyBytes := appendUint16LE(nil, float32ToFloat16(1))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(3)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(4)>>16))
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{7, 8},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				KeyDType:   "F16",
				KeyBytes:   keyBytes,
				ValueDType: "BF16",
				ValueBytes: valueBytes,
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "short-dtype.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		t.Fatalf("SaveWithOptions(native short dtype) error = %v", err)
	}
	loaded, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadWithOptions(raw-only) error = %v", err)
	}
	head := loaded.Layers[0].Heads[0]
	// normalizeKVSnapshotTensorDType maps "F16"→"float16", "BF16"→"bfloat16".
	if head.KeyDType != "float16" || head.ValueDType != "bfloat16" {
		t.Fatalf("loaded dtypes = %q/%q, want canonicalised float16/bfloat16", head.KeyDType, head.ValueDType)
	}
	if !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
		t.Fatalf("loaded native bytes = %v/%v, want %v/%v (bit-exact)", head.KeyBytes, head.ValueBytes, keyBytes, valueBytes)
	}
}

func TestKVSnapshot_SaveLoadEmptyTensor_Ugly(t *testing.T) {
	// A layer head with no Key/Value at all encodes a zero-length float32
	// tensor (encoding 0, size 0). The reader's case-0 size<=0 arm must
	// return an empty (non-nil) slice rather than read past the buffer.
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1},
		TokenOffset:   1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        0,
		HeadDim:       0,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads:      []HeadSnapshot{{}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "empty-tensor.kvbin")

	if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: KVSnapshotEncodingFloat32}); err != nil {
		t.Fatalf("SaveWithOptions(empty tensor) error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load(empty tensor) error = %v", err)
	}
	head := loaded.Layers[0].Heads[0]
	if len(head.Key) != 0 || len(head.Value) != 0 {
		t.Fatalf("loaded empty head = %+v, want zero-length key/value", head)
	}
}

func TestKVSnapshot_UnmarshalTruncated_Bad(t *testing.T) {
	// A valid serialised buffer fed in truncated prefixes must fail closed
	// at the reader's bounds guard rather than panic. Walking several cut
	// points exercises the read/u32 truncation branches across the header
	// and the tensor body without hand-building a byte layout.
	full, err := testSnapshot().MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	if len(full) < 8 {
		t.Fatalf("serialised snapshot len = %d, want a non-trivial buffer", len(full))
	}
	// Magic-length prefix, mid-header, and one-byte-short all truncate.
	for _, cut := range []int{len(kvSnapshotMagic), len(kvSnapshotMagic) + 2, len(full) / 2, len(full) - 1} {
		var loaded Snapshot
		if err := loaded.UnmarshalBinary(full[:cut]); err == nil {
			t.Fatalf("UnmarshalBinary(truncated to %d/%d) error = nil, want truncation error", cut, len(full))
		}
	}
	// Sanity: the untruncated buffer still round-trips.
	var ok Snapshot
	if err := ok.UnmarshalBinary(full); err != nil {
		t.Fatalf("UnmarshalBinary(full) error = %v, want clean decode", err)
	}
}

func TestKVSnapshot_NativeTensorValidation_Bad(t *testing.T) {
	if _, err := validateKVSnapshotNativeTensor("int4", []byte{1}, 1); err == nil {
		t.Fatal("validateKVSnapshotNativeTensor(bad dtype) error = nil")
	}
	if _, err := validateKVSnapshotNativeTensor("float16", []byte{1}, 1); err == nil {
		t.Fatal("validateKVSnapshotNativeTensor(length mismatch) error = nil")
	}
	if _, err := decodeKVSnapshotNativeTensor("float16", []byte{1}, 1); err == nil {
		t.Fatal("decodeKVSnapshotNativeTensor(length mismatch) error = nil")
	}
	if _, _, _, _, err := kvSnapshotNativeTensorInfo([]float32{1, 2}, "float16", []byte{1, 2}); err == nil {
		t.Fatal("kvSnapshotNativeTensorInfo(element mismatch) error = nil")
	}
	if got := appendKVEncodedF32s(nil, []float32{1, 2}, KVSnapshotEncodingFloat32); len(got) == 0 {
		t.Fatal("appendKVEncodedF32s() returned empty encoding")
	}
}

func TestKVSnapshot_DropFloat32_Good(t *testing.T) {
	DropFloat32(nil)
	snapshot := &Snapshot{Layers: []LayerSnapshot{{
		Heads: []HeadSnapshot{{
			Key:        []float32{1},
			KeyBytes:   []byte{1, 2},
			Value:      []float32{2},
			ValueBytes: []byte{3, 4},
		}},
	}}}

	DropFloat32(snapshot)

	head := snapshot.Layers[0].Heads[0]
	if len(head.Key) != 0 || len(head.Value) != 0 || len(head.KeyBytes) != 2 || len(head.ValueBytes) != 2 {
		t.Fatalf("DropFloat32() head = %+v, want raw bytes retained and float32 dropped", head)
	}
}

func TestKVSnapshot_Head_Ugly(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer: 7,
			Heads: []HeadSnapshot{{
				Key:   []float32{1},
				Value: []float32{2},
			}},
		}},
	}

	if _, ok := snapshot.Head(0, 0); ok {
		t.Fatal("Head(0, 0) ok = true for sparse layer 7")
	}
	if head, ok := snapshot.Head(7, 0); !ok || head.Key[0] != 1 || head.Value[0] != 2 {
		t.Fatalf("Head(7, 0) = %+v/%v, want sparse layer data", head, ok)
	}

	// Guard branches: nil receiver, negative indices, and a head index past
	// the layer's head slice must all report ok = false.
	var nilSnapshot *Snapshot
	if _, ok := nilSnapshot.Head(0, 0); ok {
		t.Fatal("Head(nil receiver) ok = true, want false")
	}
	if _, ok := snapshot.Head(-1, 0); ok {
		t.Fatal("Head(negative layer) ok = true, want false")
	}
	if _, ok := snapshot.Head(7, -1); ok {
		t.Fatal("Head(negative head) ok = true, want false")
	}
	if _, ok := snapshot.Head(7, 5); ok {
		t.Fatal("Head(out-of-range head) ok = true, want false")
	}
}

// TestKVSnapshot_ResultError_GoodBadUgly covers ResultError's three value
// shapes: an error value passes through (Good), a string value is wrapped into
// an error (Bad), and an unrecognised value type falls back to the unknown
// filesystem sentinel (Ugly).
func TestKVSnapshot_ResultError_GoodBadUgly(t *testing.T) {
	sentinel := core.NewError("boom")
	if got := ResultError(core.Result{Value: sentinel}); got != sentinel {
		t.Fatalf("ResultError(error) = %v, want passthrough of %v", got, sentinel)
	}

	if got := ResultError(core.Result{Value: "text failure"}); got == nil || got.Error() != "text failure" {
		t.Fatalf("ResultError(string) = %v, want wrapped error", got)
	}

	if got := ResultError(core.Result{Value: 42}); got == nil {
		t.Fatal("ResultError(unknown type) = nil, want fallback error")
	}
}

// TestKVSnapshot_EffectiveSeqLen_GoodBadUgly covers the three branches: a
// populated SeqLen (Good), a nil snapshot (Bad), and a zero SeqLen that falls
// back to the token count (Ugly).
func TestKVSnapshot_EffectiveSeqLen_GoodBadUgly(t *testing.T) {
	if got := EffectiveSeqLen(&Snapshot{SeqLen: 9}); got != 9 {
		t.Fatalf("EffectiveSeqLen(SeqLen=9) = %d, want 9", got)
	}
	if got := EffectiveSeqLen(nil); got != 0 {
		t.Fatalf("EffectiveSeqLen(nil) = %d, want 0", got)
	}
	if got := EffectiveSeqLen(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveSeqLen(zero SeqLen) = %d, want token count 3", got)
	}
}

// TestKVSnapshot_HashSnapshot_GoodBadUgly covers HashSnapshot: a normal float32
// snapshot hashes deterministically (Good), a nil snapshot errors (Bad), and a
// raw-native-only snapshot (Value present only as ValueBytes) takes the native
// encoding branch yet still hashes (Ugly).
func TestKVSnapshot_HashSnapshot_GoodBadUgly(t *testing.T) {
	snapshot := testSnapshot()
	hash, err := HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("HashSnapshot() error = %v", err)
	}
	again, err := HashSnapshot(snapshot)
	if err != nil || hash == "" || hash != again {
		t.Fatalf("HashSnapshot() = %q/%q, want stable non-empty hash", hash, again)
	}

	if _, err := HashSnapshot(nil); err == nil {
		t.Fatal("HashSnapshot(nil) error = nil, want snapshot error")
	}

	// Raw native head: float32 Value dropped, only ValueBytes present. This
	// drives requiresNativeEncoding down the ValueBytes branch.
	native := testSnapshot()
	head := &native.Layers[0].Heads[0]
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, float32ToFloat16(value))
	}
	head.Value = nil
	head.ValueDType = "float16"
	nativeHash, err := HashSnapshot(native)
	if err != nil || nativeHash == "" {
		t.Fatalf("HashSnapshot(native) = %q, err = %v, want non-empty hash", nativeHash, err)
	}
}

func TestKVSnapshot_Clone_Bad(t *testing.T) {
	var snapshot *Snapshot

	if snapshot.Clone() != nil {
		t.Fatal("Clone() on nil snapshot returned non-nil")
	}
}

func TestKVSnapshot_Clone_Ugly(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{Layer: 7}},
	}

	cloned := snapshot.Clone()

	if len(cloned.Layers) != 1 || cloned.Layers[0].Layer != 7 || cloned.Layers[0].Heads != nil {
		t.Fatalf("Clone() sparse layer = %+v, want preserved sparse metadata", cloned.Layers)
	}
}

func TestKVSnapshot_Save_Bad(t *testing.T) {
	var snapshot *Snapshot

	if err := snapshot.Save(core.PathJoin(t.TempDir(), "nil.kvbin")); err == nil {
		t.Fatal("Save() error = nil, want nil snapshot error")
	}
}

func TestLoadKVSnapshot_Bad(t *testing.T) {
	_, err := Load(core.PathJoin(t.TempDir(), "missing.kvbin"))

	if err == nil {
		t.Fatal("Load() error = nil, want missing file error")
	}
}

func TestLoadKVSnapshot_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.kvbin")
	if result := core.WriteFile(path, []byte("not-a-kv-snapshot"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := Load(path)

	if err == nil {
		t.Fatal("Load() error = nil, want corrupt file error")
	}
}

// TestKVSnapshot_ParseTokensCorrupt_Bad drives parseKVSnapshotTokens down its
// magic (snapshot.go:647), version (651), and token-count overflow (663)
// guards, plus the post-header reader.err arm (678) reached when the header
// truncates mid-field so tokenCount reads 0 and the token block is skipped.
// parseKVSnapshotTokens wraps via core.E with a nil cause, so assert on the
// message rather than errors.Is against the exported sentinels.
func TestKVSnapshot_ParseTokensCorrupt_Bad(t *testing.T) {
	if _, err := parseKVSnapshotTokens([]byte("xx")); err == nil || !core.Contains(err.Error(), "magic") {
		t.Fatalf("parseKVSnapshotTokens(short) error = %v, want magic error", err)
	}

	badVersion := append([]byte(kvSnapshotMagic), 0, 0, 0, 0) // version 0
	if _, err := parseKVSnapshotTokens(badVersion); err == nil || !core.Contains(err.Error(), "version") {
		t.Fatalf("parseKVSnapshotTokens(version 0) error = %v, want version error", err)
	}

	// Header that claims more tokens than the trailing bytes can supply must
	// trip the overflow guard before the token block read.
	overflow := snapshotErrTokenHeader(SnapshotVersion, 1_000_000)
	if _, err := parseKVSnapshotTokens(overflow); err == nil || !core.Contains(err.Error(), "token count") {
		t.Fatalf("parseKVSnapshotTokens(overflow) error = %v, want token-count error", err)
	}

	// Valid magic + version but the architecture-length u32 truncates mid-read:
	// reader.err is set, tokenCount falls through as 0, and the block is
	// skipped, landing on the trailing reader.err guard (snapshot.go:678).
	truncHeader := append([]byte(kvSnapshotMagic), 6, 0, 0, 0) // version 6, then nothing
	truncHeader = append(truncHeader, 0, 0)                    // 2 of 4 archLen bytes
	if _, err := parseKVSnapshotTokens(truncHeader); err == nil || !core.Contains(err.Error(), "State tokens") {
		t.Fatalf("parseKVSnapshotTokens(truncated header) error = %v, want parse-State-tokens error", err)
	}
}

// TestKVSnapshot_ParseTokensInto_Bad drives parseKVSnapshotTokensInto down its
// bare-sentinel guards: magic (snapshot.go:689 → errInvalidSnapshotMagic),
// version (693 → errUnsupportedSnapshotVersion), and token-count overflow
// (705 → errStateTokenBlockTokenCount). The Good arm appends a real token
// block onto a non-empty dst, exercising the slice-extension path.
func TestKVSnapshot_ParseTokensInto_Bad(t *testing.T) {
	dst := []int32{99}

	out, err := parseKVSnapshotTokensInto(dst, []byte("xx"))
	if err == nil || !equalInt32s(out, dst) {
		t.Fatalf("parseKVSnapshotTokensInto(short) = %v/%v, want unchanged dst + magic error", out, err)
	}

	badVersion := append([]byte(kvSnapshotMagic), 0, 0, 0, 0)
	if _, err := parseKVSnapshotTokensInto(dst, badVersion); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(version 0) error = nil, want version error")
	}

	overflow := snapshotErrTokenHeader(SnapshotVersion, 1_000_000)
	if _, err := parseKVSnapshotTokensInto(dst, overflow); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(overflow) error = nil, want token-count error")
	}

	// Good: two real tokens appended to the existing dst.
	withTokens := snapshotErrTokenHeader(SnapshotVersion, 2)
	withTokens = appendKVI32sRaw(withTokens, []int32{5, 6})
	got, err := parseKVSnapshotTokensInto(dst, withTokens)
	if err != nil || !equalInt32s(got, []int32{99, 5, 6}) {
		t.Fatalf("parseKVSnapshotTokensInto(valid) = %v/%v, want [99 5 6]", got, err)
	}
}

// TestKVSnapshot_ParseTokens_Good covers the clean parseKVSnapshotTokens path
// (zero-token header returns an empty slice; a populated header decodes the
// block) so the function's success arms are exercised alongside the Bad cases.
func TestKVSnapshot_ParseTokens_Good(t *testing.T) {
	empty, err := parseKVSnapshotTokens(snapshotErrTokenHeader(SnapshotVersion, 0))
	if err != nil || len(empty) != 0 {
		t.Fatalf("parseKVSnapshotTokens(zero) = %v/%v, want empty slice", empty, err)
	}

	buf := snapshotErrTokenHeader(SnapshotVersion, 3)
	buf = appendKVI32sRaw(buf, []int32{7, 8, 9})
	tokens, err := parseKVSnapshotTokens(buf)
	if err != nil || !equalInt32s(tokens, []int32{7, 8, 9}) {
		t.Fatalf("parseKVSnapshotTokens(three) = %v/%v, want [7 8 9]", tokens, err)
	}
}

// TestKVSnapshot_UnsupportedEncoding_Bad clones the hand-built valid buffer but
// stamps encoding tag 3 on the head key tensor, driving the encodedTensor
// reader's default arm (snapshot.go:1323 → errUnsupportedTensorEncoding).
func TestKVSnapshot_UnsupportedEncoding_Bad(t *testing.T) {
	data := snapshotBadEncodingBytes(3)

	var loaded Snapshot
	err := loaded.UnmarshalBinary(data)
	if err == nil || !core.Contains(err.Error(), "unsupported KV tensor encoding") {
		t.Fatalf("UnmarshalBinary(encoding tag 3) error = %v, want unsupported-encoding error", err)
	}
}

// TestKVSnapshot_ReaderCase0Truncated_Bad stamps a float32 (encoding 0) head
// tensor with a size larger than the trailing bytes, driving the case-0
// chunk==nil arm in the encodedTensor reader (snapshot.go:1282) via the
// underlying read() truncation guard.
func TestKVSnapshot_ReaderCase0Truncated_Bad(t *testing.T) {
	data := snapshotBadEncodingBytes(0)
	// snapshotBadEncodingBytes(0) writes encoding 0 with size 1 and one f32
	// (4 bytes). Rewrite the size to claim 9999 elements without supplying the
	// bytes — the batched read(size*4) overruns and returns nil.
	patchKVU32(data, snapshotKeyTensorSizeOffset(), 9999)

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err == nil || !core.Contains(err.Error(), "truncated") {
		t.Fatalf("UnmarshalBinary(case-0 oversized) error = %v, want truncation error", err)
	}
}

// TestKVSnapshot_LoadWithOptionsParseError_Bad writes a corrupt-but-present
// file so LoadWithOptions reaches parseKVSnapshotWithOptions and returns its
// parse error (the read succeeds; the parse fails). Complements the existing
// missing-file Bad case.
func TestKVSnapshot_LoadWithOptionsParseError_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "badmagic.kvbin")
	if result := core.WriteFile(path, []byte("XXXXXXXX____"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	if _, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true}); err == nil || !core.Contains(err.Error(), "magic") {
		t.Fatalf("LoadWithOptions(bad magic) error = %v, want magic parse error", err)
	}
}

// TestKVSnapshot_ParseLegacyV2_Good hand-builds a version-2 buffer whose heads
// carry plain float32 Key/Value blocks (no per-tensor encoding header). This is
// the only way to drive the version<3 head read arm in parseKVSnapshotWithOptions
// (snapshot.go:611-614, the reader.f32s() fallback) — the writer always emits
// the current version, so a round-trip can't reach it.
func TestKVSnapshot_ParseLegacyV2_Good(t *testing.T) {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, 2) // version 2 (<3 → f32s head path, ≥2 → token offset/generated/logits)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 2) // SeqLen
	data = appendKVU32(data, 2) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVU32(data, 2) // TokenOffset (v>=2)
	data = appendKVI32s(data, []int32{1, 2})
	data = appendKVI32s(data, []int32{2}) // generated (v>=2)
	data = appendKVU32(data, 1)           // layer count
	data = appendKVI32(data, 0)           // Layer
	data = appendKVI32(data, 0)           // CacheIndex
	data = appendKVU32(data, 1)           // head count
	data = appendKVF32s(data, []float32{1, 0, 0, 1})
	data = appendKVF32s(data, []float32{0, 1, 1, 0})
	data = appendKVI32s(data, []int32{1, 1, 3}) // logit shape (v>=2)
	data = appendKVF32s(data, []float32{0.1, 0.2, 0.7})

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(v2 legacy) error = %v", err)
	}
	if loaded.Version != 2 || len(loaded.Layers) != 1 {
		t.Fatalf("loaded v2 = version %d / %d layers, want version 2 / 1 layer", loaded.Version, len(loaded.Layers))
	}
	head := loaded.Layers[0].Heads[0]
	if len(head.Key) != 4 || head.Key[0] != 1 || len(head.Value) != 4 || head.Value[1] != 1 {
		t.Fatalf("loaded v2 head = %+v, want float32 key/value from the legacy read path", head)
	}
}

// TestKVSnapshot_TokenOffsetDefault_Ugly loads a v1 buffer that omits the token
// offset field, so the parser's trailing `TokenOffset == 0 → len(Tokens)`
// fixup fires (snapshot.go:639). v1 has no per-tensor encoding header, so the
// head goes through the same f32s path as the v2 case.
func TestKVSnapshot_TokenOffsetDefault_Ugly(t *testing.T) {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, 1) // version 1 (no TokenOffset/Generated/Logits fields)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 2) // SeqLen
	data = appendKVU32(data, 2) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVI32s(data, []int32{3, 4})
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0) // Layer
	data = appendKVI32(data, 0) // CacheIndex
	data = appendKVU32(data, 1) // head count
	data = appendKVF32s(data, []float32{1, 2, 3, 4})
	data = appendKVF32s(data, []float32{4, 3, 2, 1})

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(v1) error = %v", err)
	}
	if loaded.TokenOffset != 2 {
		t.Fatalf("loaded v1 TokenOffset = %d, want default to token count 2", loaded.TokenOffset)
	}
}

// TestKVSnapshot_NativeTensorInfo_Bad covers the two early-return error arms of
// kvSnapshotNativeTensorInfo: an unknown dtype with raw bytes present
// (snapshot.go:862) and a raw length that is not a whole number of elements for
// the dtype (snapshot.go:865).
func TestKVSnapshot_NativeTensorInfo_Bad(t *testing.T) {
	if _, _, _, ok, err := kvSnapshotNativeTensorInfo(nil, "int8", []byte{1, 2}); ok || err == nil {
		t.Fatalf("kvSnapshotNativeTensorInfo(unknown dtype) = ok %v/err %v, want false + error", ok, err)
	}
	// float16 = 2 bytes/value; 3 raw bytes is not a whole number of elements.
	if _, _, _, ok, err := kvSnapshotNativeTensorInfo(nil, "float16", []byte{1, 2, 3}); ok || err == nil {
		t.Fatalf("kvSnapshotNativeTensorInfo(odd length) = ok %v/err %v, want false + error", ok, err)
	}
}

// TestKVSnapshot_EncodedTensorSize_GoodBadUgly covers kvSnapshotEncodedTensorSize:
// a native tensor with an unknown dtype surfaces the info error (snapshot.go:843,
// Bad); empty values with raw bytes under a non-native encoding hits the
// raw-requires-native guard (snapshot.go:850, Ugly); a plain float32 tensor
// returns the 8+4N size (Good).
func TestKVSnapshot_EncodedTensorSize_GoodBadUgly(t *testing.T) {
	if _, err := kvSnapshotEncodedTensorSize(nil, "int8", []byte{1, 2}, EncodingNative); err == nil {
		t.Fatal("kvSnapshotEncodedTensorSize(native bad dtype) error = nil, want native-info error")
	}
	if _, err := kvSnapshotEncodedTensorSize(nil, "", []byte{1, 2, 3}, KVSnapshotEncodingFloat32); err == nil {
		t.Fatal("kvSnapshotEncodedTensorSize(raw without native) error = nil, want raw-needs-native error")
	}
	size, err := kvSnapshotEncodedTensorSize([]float32{1, 2}, "", nil, KVSnapshotEncodingFloat32)
	if err != nil || size != 8+2*4 {
		t.Fatalf("kvSnapshotEncodedTensorSize(float32) = %d/%v, want %d", size, err, 8+2*4)
	}
}

// TestKVSnapshot_DecodeNativeFloat32_Good drives decodeKVSnapshotNativeTensor's
// float32 reinterpret-cast arm (snapshot.go:1347-1351), which the existing
// validation-error test never reaches (it only feeds mismatched lengths).
func TestKVSnapshot_DecodeNativeFloat32_Good(t *testing.T) {
	raw := appendKVF32Raw(nil, []float32{1.5, -2.25})
	values, err := decodeKVSnapshotNativeTensor("float32", raw, 2)
	if err != nil || len(values) != 2 || values[0] != 1.5 || values[1] != -2.25 {
		t.Fatalf("decodeKVSnapshotNativeTensor(float32) = %v/%v, want [1.5 -2.25]", values, err)
	}
}

// TestKVSnapshot_NilPredicates_Bad exercises the nil-snapshot guards that the
// happy-path tests never reach: validateKVSnapshotCompressedPayloads
// (snapshot.go:1482), requiresNativeEncoding (1498), and
// snapshotHasLayerNativeTensors (1518). cloneKVLayers(nil) covers the empty
// guard at 1367.
func TestKVSnapshot_NilPredicates_Bad(t *testing.T) {
	if err := validateKVSnapshotCompressedPayloads(nil); err == nil {
		t.Fatal("validateKVSnapshotCompressedPayloads(nil) error = nil, want snapshot-nil error")
	}
	if requiresNativeEncoding(nil) {
		t.Fatal("requiresNativeEncoding(nil) = true, want false")
	}
	if snapshotHasLayerNativeTensors(nil) {
		t.Fatal("snapshotHasLayerNativeTensors(nil) = true, want false")
	}
	if cloneKVLayers(nil) != nil {
		t.Fatal("cloneKVLayers(nil) != nil, want nil")
	}
}

// TestKVSnapshot_LayerNativeTensors_Good drives the positive arms of
// snapshotHasLayerNativeTensors (layer.KeyBytes present, snapshot.go:1522) and
// requiresNativeEncoding (which short-circuits true through it, 1501), plus
// cloneKVLayers over a fully-populated layer (the per-layer clone body, 1376).
func TestKVSnapshot_LayerNativeTensors_Good(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer:      3,
			CacheIndex: 1,
			KeyDType:   "float16",
			KeyBytes:   []byte{1, 2},
			KeyShape:   []int32{1, 1},
		}},
	}
	if !snapshotHasLayerNativeTensors(snapshot) {
		t.Fatal("snapshotHasLayerNativeTensors(layer bytes) = false, want true")
	}
	if !requiresNativeEncoding(snapshot) {
		t.Fatal("requiresNativeEncoding(layer bytes) = false, want true")
	}
	cloned := cloneKVLayers(snapshot.Layers)
	if len(cloned) != 1 || cloned[0].Layer != 3 || !equalBytes(cloned[0].KeyBytes, []byte{1, 2}) {
		t.Fatalf("cloneKVLayers(populated) = %+v, want deep copy with KeyBytes", cloned)
	}
	// requiresNativeEncoding's head-bytes arm (snapshot.go:1506/1509): a head
	// with ValueBytes but no float32 Value, no layer-level native bytes.
	headOnly := &Snapshot{Layers: []LayerSnapshot{{Heads: []HeadSnapshot{{
		ValueBytes: []byte{9, 9},
		ValueDType: "float16",
	}}}}}
	if !requiresNativeEncoding(headOnly) {
		t.Fatal("requiresNativeEncoding(head bytes) = false, want true")
	}
}

// TestKVSnapshot_FirstNonEmpty_GoodBadUgly covers firstNonEmpty: a real value
// is returned (Good), all-empty inputs fall through to "" (snapshot.go:1466,
// Bad), and a whitespace-only value is skipped in favour of a later real one
// via the core.Trim branch (Ugly).
func TestKVSnapshot_FirstNonEmpty_GoodBadUgly(t *testing.T) {
	if got := firstNonEmpty("", "real"); got != "real" {
		t.Fatalf("firstNonEmpty(empty, real) = %q, want \"real\"", got)
	}
	if got := firstNonEmpty("", ""); got != "" {
		t.Fatalf("firstNonEmpty(all empty) = %q, want empty string", got)
	}
	if got := firstNonEmpty("   ", "kept"); got != "kept" {
		t.Fatalf("firstNonEmpty(whitespace, kept) = %q, want \"kept\"", got)
	}
}

// TestKVSnapshot_HashSnapshotNativeError_Bad drives HashSnapshot's
// writeWithOptions error arm (snapshot.go:1546): a head carrying KeyBytes with
// an empty dtype forces requiresNativeEncoding true, so HashSnapshot selects
// native encoding, and the native encoder rejects the unknown dtype mid-write.
func TestKVSnapshot_HashSnapshotNativeError_Bad(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		NumLayers:     1,
		NumHeads:      1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Heads: []HeadSnapshot{{
				KeyBytes: []byte{1, 2, 3}, // raw bytes, empty dtype → native encode fails
			}},
		}},
	}

	if _, err := HashSnapshot(snapshot); err == nil {
		t.Fatal("HashSnapshot(native bad dtype) error = nil, want native-encode error")
	}
}

// snapshotErrTokenHeader builds the State-block header parseKVSnapshotTokens and
// parseKVSnapshotTokensInto consume: magic, version, length-prefixed
// architecture, five u32 dimension fields, the v>=2 token-offset field, and the
// token count. Callers append the token bytes (or omit them to trip the
// overflow guard).
func snapshotErrTokenHeader(version, tokenCount uint32) []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, version)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	for range 5 {
		data = appendKVU32(data, 0) // NumLayers/NumHeads/SeqLen/HeadDim/NumQueryHeads
	}
	if version >= 2 {
		data = appendKVU32(data, 0) // TokenOffset
	}
	data = appendKVU32(data, tokenCount)
	return data
}

// snapshotBadEncodingBytes builds a complete valid single-head v6 buffer (the
// kvSnapshotTurboQuantNoPayloadBytes layout, minus the turboquant cache mode)
// but writes encoding tag `encodingTag` on the key tensor, with one float32
// element of payload. With tag 3 it drives the reader's default arm; with tag 0
// it is a valid float32 tensor whose size field can be patched to overrun.
func snapshotBadEncodingBytes(encodingTag uint32) []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, SnapshotVersion)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 1) // SeqLen
	data = appendKVU32(data, 1) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVU32(data, 1) // TokenOffset (v>=2)
	data = appendKVI32s(data, []int32{1})
	data = appendKVU32(data, 0) // generated count (v>=2)
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0) // Layer
	data = appendKVI32(data, 0) // CacheIndex
	data = appendKVU32(data, 1) // head count
	data = appendKVBytes(data, core.AsBytes(""))
	data = appendKVU32(data, 0)    // TurboQuant payload count (v>=5)
	data = appendKVU32(data, 0)    // MaxSize (v>=6)
	data = appendKVI32s(data, nil) // KeyShape (v>=4)
	data = appendKVU32(data, 0)    // key tensor encoding (RawKVOnly path)
	data = appendKVU32(data, 0)    // key tensor size
	data = appendKVI32s(data, nil) // ValueShape (v>=4)
	data = appendKVU32(data, 0)    // value tensor encoding
	data = appendKVU32(data, 0)    // value tensor size
	// Head 0 (v>=3): key tensor with the chosen encoding tag, then a clean
	// value tensor. snapshotKeyTensorSizeOffset() points at the size u32 below.
	data = appendKVU32(data, encodingTag) // key tensor encoding
	data = appendKVU32(data, 1)           // key tensor size (1 element)
	data = appendKVF32Raw(data, []float32{1})
	data = appendKVU32(data, 0) // value tensor encoding (float32)
	data = appendKVU32(data, 0) // value tensor size
	data = appendKVU32(data, 0) // logit shape (v>=2)
	data = appendKVF32s(data, nil)
	return data
}

// snapshotKeyTensorSizeOffset returns the byte offset of the head-0 key
// tensor's size u32 within a snapshotBadEncodingBytes buffer, so a test can
// rewrite the size to overrun the trailing bytes. It is the position of the
// "key tensor size" field written after the head-0 encoding tag.
func snapshotKeyTensorSizeOffset() int {
	// Recompute by re-walking the prefix the builder writes up to (and
	// including) the head-0 encoding tag. Mirrors snapshotBadEncodingBytes.
	var prefix []byte
	prefix = append(prefix, kvSnapshotMagic...)
	prefix = appendKVU32(prefix, SnapshotVersion)
	prefix = appendKVBytes(prefix, core.AsBytes("gemma4_text"))
	for range 5 {
		prefix = appendKVU32(prefix, 1)
	}
	prefix = appendKVU32(prefix, 1)           // TokenOffset
	prefix = appendKVI32s(prefix, []int32{1}) // tokens
	prefix = appendKVU32(prefix, 0)           // generated count
	prefix = appendKVU32(prefix, 1)           // layer count
	prefix = appendKVI32(prefix, 0)           // Layer
	prefix = appendKVI32(prefix, 0)           // CacheIndex
	prefix = appendKVU32(prefix, 1)           // head count
	prefix = appendKVBytes(prefix, core.AsBytes(""))
	prefix = appendKVU32(prefix, 0)    // TurboQuant payload count
	prefix = appendKVU32(prefix, 0)    // MaxSize
	prefix = appendKVI32s(prefix, nil) // KeyShape
	prefix = appendKVU32(prefix, 0)    // layer key encoding
	prefix = appendKVU32(prefix, 0)    // layer key size
	prefix = appendKVI32s(prefix, nil) // ValueShape
	prefix = appendKVU32(prefix, 0)    // layer value encoding
	prefix = appendKVU32(prefix, 0)    // layer value size
	prefix = appendKVU32(prefix, 0)    // head-0 key encoding tag
	return len(prefix)                 // next u32 written is the key size
}

// patchKVU32 overwrites the little-endian u32 at offset within buf.
func patchKVU32(buf []byte, offset int, value uint32) {
	binary.LittleEndian.PutUint32(buf[offset:offset+4], value)
}

// equalInt32s reports whether two int32 slices hold the same values.
func equalInt32s(left, right []int32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func equalBytes(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

// TestSnapshot_QuantizeKVSnapshotQ8_Good covers the validate+quantise wrapper
// quantizeKVSnapshotQ8 (snapshot.go), which computes maxAbs then forwards to
// quantizeKVSnapshotQ8WithMaxAbs. The returned scale is maxAbs/127 and the
// largest-magnitude value must dequantise back to ~itself.
func TestSnapshot_QuantizeKVSnapshotQ8_Good(t *testing.T) {
	values := []float32{0, 1.27, -1.27, 0.635}

	scale, quant := quantizeKVSnapshotQ8(values)
	if len(quant) != len(values) {
		t.Fatalf("quant len = %d, want %d", len(quant), len(values))
	}
	wantScale := float32(1.27) / 127
	if scale != wantScale {
		t.Fatalf("scale = %v, want %v", scale, wantScale)
	}
	// The +1.27 peak quantises to +127 and the -1.27 to -127.
	if int8(quant[1]) != 127 || int8(quant[2]) != -127 {
		t.Fatalf("quant peaks = %d/%d, want 127/-127", int8(quant[1]), int8(quant[2]))
	}

	// All-zero input keeps scale 1 (no divide-by-zero) and zeroed output.
	zScale, zQuant := quantizeKVSnapshotQ8([]float32{0, 0, 0})
	if zScale != 1 {
		t.Fatalf("all-zero scale = %v, want 1", zScale)
	}
	for i, b := range zQuant {
		if b != 0 {
			t.Fatalf("all-zero quant[%d] = %d, want 0", i, b)
		}
	}
}

// TestSnapshot_ReaderEncodedF32s_Good covers the kvSnapshotReader.encodedF32s
// wrapper (snapshot.go), which forwards encodedTensor(LoadOptions{}).Values.
// A hand-built encoding-0 (float32) tensor block is decoded back to its values.
func TestSnapshot_ReaderEncodedF32s_Good(t *testing.T) {
	want := []float32{1.5, -2.25, 3.75}
	buf := appendKVEncodedF32s(nil, want, KVSnapshotEncodingFloat32)

	reader := &kvSnapshotReader{data: buf}
	got := reader.encodedF32s()
	if reader.err != nil {
		t.Fatalf("encodedF32s reader.err = %v", reader.err)
	}
	if len(got) != len(want) {
		t.Fatalf("encodedF32s len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("encodedF32s = %v, want %v", got, want)
		}
	}
}

// TestSnapshot_DtypeString_GoodBadUgly covers every arm of the
// kvSnapshotReader.dtypeString length-prefixed tag reader: the canonical
// short-form (F32/F16/BF16) and long-form (float32/float16/bfloat16) tags each
// return their literal, an unknown tag is returned verbatim, and a truncated
// length prefix yields the empty string (the read-nil guard).
func TestSnapshot_DtypeString_GoodBadUgly(t *testing.T) {
	// dtypeTag builds a length-prefixed dtype buffer the reader consumes.
	dtypeTag := func(tag string) []byte {
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(len(tag)))
		return append(buf, tag...)
	}

	for _, tag := range []string{"F32", "F16", "BF16", "float32", "float16", "bfloat16"} {
		reader := &kvSnapshotReader{data: dtypeTag(tag)}
		if got := reader.dtypeString(); got != tag {
			t.Fatalf("dtypeString(%q) = %q, want the canonical literal", tag, got)
		}
	}

	// Unknown tag of a recognised length is returned verbatim (validator
	// rejects it downstream).
	if got := (&kvSnapshotReader{data: dtypeTag("abc")}).dtypeString(); got != "abc" {
		t.Fatalf("dtypeString(unknown 3-byte) = %q, want \"abc\"", got)
	}
	// Unknown tag of an unrecognised length also falls through to verbatim.
	if got := (&kvSnapshotReader{data: dtypeTag("int8")}).dtypeString(); got != "int8" {
		t.Fatalf("dtypeString(unknown 4-byte) = %q, want \"int8\"", got)
	}

	// Ugly: a length prefix claiming more bytes than remain → read returns
	// nil → dtypeString returns "".
	truncated := make([]byte, 4)
	binary.LittleEndian.PutUint32(truncated, 99)
	if got := (&kvSnapshotReader{data: truncated}).dtypeString(); got != "" {
		t.Fatalf("dtypeString(truncated) = %q, want empty string", got)
	}
}

// TestSnapshot_EncodeErrors_Bad drives the encode-path guards shared by
// encodedSizeWithOptions / bytesWithOptions / writeWithOptions: an invalid
// KVEncoding is rejected up front, and a snapshot carrying a malformed native
// layer tensor (a dtype/shape the encoder can't size) surfaces the encode error
// rather than producing a corrupt buffer.
func TestSnapshot_EncodeErrors_Bad(t *testing.T) {
	// Invalid encoding rejected by all three entry points.
	bad := SaveOptions{KVEncoding: "not-an-encoding"}
	if _, err := testSnapshot().encodedSizeWithOptions(bad); err == nil {
		t.Fatal("encodedSizeWithOptions(bad encoding) error = nil")
	}
	if _, err := testSnapshot().bytesWithOptions(bad); err == nil {
		t.Fatal("bytesWithOptions(bad encoding) error = nil")
	}

	// A head carrying raw native bytes but NO float32 values cannot be encoded
	// under a non-native encoding (Q8): the encoder needs EncodingNative to
	// pass raw payloads through, so the size pass surfaces errRawTensorNeedsNative.
	rawOnly := &Snapshot{
		Version: SnapshotVersion, Architecture: "gemma4_text",
		Tokens: []int32{1}, TokenOffset: 1,
		NumLayers: 1, NumHeads: 1, SeqLen: 1, HeadDim: 1, NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				KeyDType: "float16",
				KeyBytes: []byte{1, 0}, // raw, no float32 Key alongside
			}},
		}},
	}
	if _, err := rawOnly.encodedSizeWithOptions(SaveOptions{KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("encodedSizeWithOptions(raw-only head, Q8) error = nil, want errRawTensorNeedsNative")
	}
	if _, err := rawOnly.bytesWithOptions(SaveOptions{KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("bytesWithOptions(raw-only head, Q8) error = nil, want errRawTensorNeedsNative")
	}
}

// TestSnapshot_NormalizeSnapshot_GoodUgly covers normalizeSnapshot
// (snapshot.go): the nil guard (Ugly), the Version==0 default fill, and the
// TokenOffset==0 → len(Tokens) default fill (Good).
func TestSnapshot_NormalizeSnapshot_GoodUgly(t *testing.T) {
	// Ugly: nil snapshot must be a no-op (no panic).
	normalizeSnapshot(nil)

	// Good: zero Version and zero TokenOffset both get filled.
	snapshot := &Snapshot{Tokens: []int32{1, 2, 3}}
	normalizeSnapshot(snapshot)
	if snapshot.Version != SnapshotVersion {
		t.Fatalf("Version = %d, want default %d", snapshot.Version, SnapshotVersion)
	}
	if snapshot.TokenOffset != 3 {
		t.Fatalf("TokenOffset = %d, want len(Tokens) = 3", snapshot.TokenOffset)
	}

	// A snapshot already carrying both fields is left untouched.
	preset := &Snapshot{Version: 2, TokenOffset: 9, Tokens: []int32{1}}
	normalizeSnapshot(preset)
	if preset.Version != 2 || preset.TokenOffset != 9 {
		t.Fatalf("preset normalised to %d/%d, want 2/9 unchanged", preset.Version, preset.TokenOffset)
	}
}

// kvSnapshotRichV6 builds a version-6 snapshot exercising every version-gated
// encode arm: Generated tokens (v2), per-head float32 K/V (v3), a native layer
// raw tensor (v4), a TurboQuant compressed layer (v5), a MaxSize window clamp
// (v6), and LogitShape/Logits. SeqLen 2 so a single block holds it whole (the
// TurboQuant payload requires a full-range block).
func kvSnapshotRichV6() *Snapshot {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
	valueBytes := appendUint16LE(nil, float32ToFloat16(0.25))
	valueBytes = appendUint16LE(valueBytes, float32ToFloat16(-0.75))
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{
			{
				// Native layer raw tensor (v4) + MaxSize clamp (v6).
				Layer:      0,
				CacheIndex: 0,
				MaxSize:    4096,
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				KeyShape:   []int32{1, 1, 2, 1},
				ValueDType: "float16",
				ValueBytes: valueBytes,
				ValueShape: []int32{1, 1, 2, 1},
				Heads:      make([]HeadSnapshot, 1),
			},
			{
				// TurboQuant compressed layer (v5) — requires the turboquant
				// cache mode and at least one payload.
				Layer:              1,
				CacheIndex:         1,
				MaxSize:            4096,
				CacheMode:          "turboquant",
				TurboQuantPayloads: [][]byte{{1, 2, 3, 4}},
				Heads:              make([]HeadSnapshot, 1),
			},
		},
	}
}

// TestSnapshot_RichVersion6_EncodeRoundTrip_Good drives the version-gated encode
// arms shared by encodedSizeWithOptions / bytesWithOptions / writeWithOptions /
// the stream encoder across three usage surfaces: the in-memory MarshalBinary
// round-trip, a SaveStateBlocks to a streaming store (BinaryStreamWriter →
// kvSnapshotStreamWriter), and HashSnapshot. Each recovers the rich snapshot's
// observable shape.
func TestSnapshot_RichVersion6_EncodeRoundTrip_Good(t *testing.T) {
	source := kvSnapshotRichV6()

	// Surface 1: in-memory binary round-trip under native encoding
	// (bytesWithOptions + encodedSizeWithOptions). Native is required because
	// the rich snapshot carries raw layer tensors; the default float32
	// MarshalBinary cannot encode raw payloads (errRawTensorNeedsNative).
	data, err := source.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("bytesWithOptions(rich v6, native) error = %v", err)
	}
	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(rich v6) error = %v", err)
	}
	if loaded.Version != SnapshotVersion {
		t.Fatalf("loaded version = %d, want %d", loaded.Version, SnapshotVersion)
	}
	if len(loaded.Generated) != 1 || len(loaded.Logits) != 3 {
		t.Fatalf("loaded generated/logits = %d/%d, want 1/3", len(loaded.Generated), len(loaded.Logits))
	}
	if loaded.Layers[0].MaxSize != 4096 || loaded.Layers[1].CacheMode != "turboquant" {
		t.Fatalf("loaded layer metadata = maxsize %d / mode %q, want 4096 / turboquant", loaded.Layers[0].MaxSize, loaded.Layers[1].CacheMode)
	}

	// Surface 2: stream-save path (writeWithOptions via kvSnapshotStreamWriter).
	stream := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	bundle, err := source.SaveStateBlocks(context.Background(), stream, StateBlockOptions{
		BlockSize:  2, // whole snapshot in one block (TurboQuant needs full range)
		KVEncoding: EncodingNative,
		URI:        "mlx://rich-v6",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(rich v6, stream store) error = %v", err)
	}
	if stream.streamPuts == 0 {
		t.Fatal("stream store recorded no PutBytesStream calls, want the stream-write path exercised")
	}
	if len(bundle.Blocks) != 1 {
		t.Fatalf("bundle blocks = %d, want 1 whole-snapshot block", len(bundle.Blocks))
	}

	// Surface 3: HashSnapshot (writeWithOptions to a hash sink) is stable.
	hash, err := HashSnapshot(source)
	if err != nil || len(hash) != 64 {
		t.Fatalf("HashSnapshot(rich v6) = %q / %v, want 64-hex digest", hash, err)
	}
}
