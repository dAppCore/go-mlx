// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
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
