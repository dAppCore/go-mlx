// SPDX-Licence-Identifier: EUPL-1.2

package kvconv

import (
	"reflect"
	"testing"

	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
)

// kvconv_test.go: Good/Bad/Ugly coverage for the root<->metal KV snapshot
// bridge — dtype tagging, capture-option mapping, and the full snapshot
// conversions including the nil-in/empty-in/populated three-way branches the
// production code goes out of its way to preserve, plus the native-slab
// passthrough on the metal direction.

// --- RootKVHeadDType -------------------------------------------------------

func TestKvconv_RootKVHeadDType_Good(t *testing.T) {
	raw := []byte{0, 0}
	cases := []struct {
		dtype metal.DType
		want  string
	}{
		{metal.DTypeFloat32, "float32"},
		{metal.DTypeFloat16, "float16"},
		{metal.DTypeBFloat16, "bfloat16"},
	}
	for _, tc := range cases {
		if got := RootKVHeadDType(tc.dtype, raw); got != tc.want {
			t.Fatalf("RootKVHeadDType(%v) = %q, want %q", tc.dtype, got, tc.want)
		}
	}
}

func TestKvconv_RootKVHeadDType_Bad(t *testing.T) {
	// Empty raw bytes mean the head carries no tensor — dtype tag must be
	// blank regardless of the dtype value, so a downstream consumer does not
	// mistake an absent tensor for a typed one.
	if got := RootKVHeadDType(metal.DTypeFloat32, nil); got != "" {
		t.Fatalf("RootKVHeadDType(float32, nil) = %q, want empty", got)
	}
	if got := RootKVHeadDType(metal.DTypeFloat16, []byte{}); got != "" {
		t.Fatalf("RootKVHeadDType(float16, empty) = %q, want empty", got)
	}
}

func TestKvconv_RootKVHeadDType_Ugly(t *testing.T) {
	// A dtype outside the three KV-supported kinds (int32 here) must map to
	// the empty tag rather than leak a partial name.
	if got := RootKVHeadDType(metal.DTypeInt32, []byte{1, 2, 3, 4}); got != "" {
		t.Fatalf("RootKVHeadDType(int32, raw) = %q, want empty for unsupported dtype", got)
	}
}

// --- MetalKVHeadDType ------------------------------------------------------

func TestKvconv_MetalKVHeadDType_Good(t *testing.T) {
	raw := []byte{0, 0}
	cases := []struct {
		name string
		want metal.DType
	}{
		{"float32", metal.DTypeFloat32},
		{"float16", metal.DTypeFloat16},
		{"bfloat16", metal.DTypeBFloat16},
	}
	for _, tc := range cases {
		if got := MetalKVHeadDType(tc.name, raw); got != tc.want {
			t.Fatalf("MetalKVHeadDType(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestKvconv_MetalKVHeadDType_Bad(t *testing.T) {
	// No raw bytes -> zero DType, mirroring RootKVHeadDType's empty-tag rule.
	if got := MetalKVHeadDType("float32", nil); got != 0 {
		t.Fatalf("MetalKVHeadDType(float32, nil) = %v, want 0", got)
	}
	if got := MetalKVHeadDType("float16", []byte{}); got != 0 {
		t.Fatalf("MetalKVHeadDType(float16, empty) = %v, want 0", got)
	}
}

func TestKvconv_MetalKVHeadDType_Ugly(t *testing.T) {
	raw := []byte{0, 0}
	// The safetensors-style short aliases (F32/F16/BF16) must resolve the
	// same as the long names — both encodings reach this path from persisted
	// snapshots.
	aliases := map[string]metal.DType{
		"F32":  metal.DTypeFloat32,
		"F16":  metal.DTypeFloat16,
		"BF16": metal.DTypeBFloat16,
	}
	for name, want := range aliases {
		if got := MetalKVHeadDType(name, raw); got != want {
			t.Fatalf("MetalKVHeadDType(%q) = %v, want %v", name, got, want)
		}
	}
	// An unknown dtype name with present bytes maps to the zero DType.
	if got := MetalKVHeadDType("complex128", raw); got != 0 {
		t.Fatalf("MetalKVHeadDType(unknown) = %v, want 0", got)
	}
}

// --- ToMetalKVSnapshotCaptureOptions ---------------------------------------

func TestKvconv_ToMetalKVSnapshotCaptureOptions_Good(t *testing.T) {
	in := kv.CaptureOptions{RawKVOnly: true, BlockStartToken: 42}
	got := ToMetalKVSnapshotCaptureOptions(in)
	if !got.RawKVOnly || got.BlockStartToken != 42 {
		t.Fatalf("ToMetalKVSnapshotCaptureOptions(%+v) = %+v, want fields carried through", in, got)
	}
}

func TestKvconv_ToMetalKVSnapshotCaptureOptions_Bad(t *testing.T) {
	// The zero options must map to zero options — no field is silently
	// defaulted on the way across the boundary.
	got := ToMetalKVSnapshotCaptureOptions(kv.CaptureOptions{})
	if got.RawKVOnly || got.BlockStartToken != 0 {
		t.Fatalf("ToMetalKVSnapshotCaptureOptions(zero) = %+v, want zero metal options", got)
	}
}

// --- ToRootKVSnapshot ------------------------------------------------------

func TestKvconv_ToRootKVSnapshot_Good(t *testing.T) {
	src := &metal.KVSnapshot{
		Version:       4,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		Generated:     []int32{4, 5},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        3,
		HeadDim:       4,
		NumQueryHeads: 2,
		LogitShape:    []int32{1, 8},
		Logits:        []float32{0.1, 0.2, 0.3},
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  metal.KVCacheModeFP16,
			MaxSize:    16,
			KeyShape:   []int32{1, 2, 3, 4},
			ValueShape: []int32{1, 2, 3, 4},
			Heads: []metal.KVHeadSnapshot{{
				Key:        []float32{1, 2, 3, 4},
				KeyDType:   metal.DTypeFloat32,
				KeyBytes:   []byte{0x10, 0x20},
				Value:      []float32{5, 6, 7, 8},
				ValueDType: metal.DTypeFloat32,
				ValueBytes: []byte{0x30, 0x40},
			}},
		}},
	}
	got := ToRootKVSnapshot(src)
	if got == nil {
		t.Fatal("ToRootKVSnapshot() = nil, want snapshot")
	}
	// Scalars carry across verbatim.
	if got.Version != 4 || got.Architecture != "gemma4_text" || got.TokenOffset != 3 {
		t.Fatalf("scalar fields = {%d %q %d}, want {4 gemma4_text 3}", got.Version, got.Architecture, got.TokenOffset)
	}
	// Populated per-layer shapes carry across (the non-empty default branch).
	if !reflect.DeepEqual(got.Layers[0].KeyShape, []int32{1, 2, 3, 4}) || !reflect.DeepEqual(got.Layers[0].ValueShape, []int32{1, 2, 3, 4}) {
		t.Fatalf("layer shapes = key:%v value:%v, want [1 2 3 4] each", got.Layers[0].KeyShape, got.Layers[0].ValueShape)
	}
	if got.NumLayers != 1 || got.NumHeads != 2 || got.SeqLen != 3 || got.HeadDim != 4 || got.NumQueryHeads != 2 {
		t.Fatalf("shape fields = {%d %d %d %d %d}, want {1 2 3 4 2}", got.NumLayers, got.NumHeads, got.SeqLen, got.HeadDim, got.NumQueryHeads)
	}
	// Slices carry values across and the CacheMode string-converts.
	if !reflect.DeepEqual(got.Tokens, []int32{1, 2, 3}) || !reflect.DeepEqual(got.Generated, []int32{4, 5}) {
		t.Fatalf("tokens/generated = %v/%v, want [1 2 3]/[4 5]", got.Tokens, got.Generated)
	}
	if len(got.Layers) != 1 || got.Layers[0].CacheMode != "fp16" {
		t.Fatalf("layer cache mode = %q, want fp16", got.Layers[0].CacheMode)
	}
	head := got.Layers[0].Heads[0]
	// The dtype tag is derived from the head's *bytes* (RootKVHeadDType reads
	// KeyBytes/ValueBytes, not the decoded float32) — so a head with bytes
	// present carries the resolved name.
	if !reflect.DeepEqual(head.Key, []float32{1, 2, 3, 4}) || head.KeyDType != "float32" {
		t.Fatalf("head key = %v dtype %q, want [1 2 3 4] float32", head.Key, head.KeyDType)
	}
	if !reflect.DeepEqual(head.Value, []float32{5, 6, 7, 8}) || head.ValueDType != "float32" {
		t.Fatalf("head value = %v dtype %q, want [5 6 7 8] float32", head.Value, head.ValueDType)
	}
	// The copied head slices must not alias the source — mutating the source
	// after conversion must not be visible through the root snapshot.
	src.Layers[0].Heads[0].Key[0] = 999
	if got.Layers[0].Heads[0].Key[0] == 999 {
		t.Fatal("ToRootKVSnapshot aliased the source head Key; want an independent copy")
	}
}

func TestKvconv_ToRootKVSnapshot_Bad(t *testing.T) {
	// A nil source returns nil — the conversion never panics on the
	// not-captured case (multi-turn restore with no prior snapshot).
	if got := ToRootKVSnapshot(nil); got != nil {
		t.Fatalf("ToRootKVSnapshot(nil) = %+v, want nil", got)
	}
}

func TestKvconv_ToRootKVSnapshot_Ugly(t *testing.T) {
	// The nil-in -> nil-out vs empty-in -> empty-[]T{} distinction is
	// load-bearing: downstream serialisation treats nil and empty
	// differently, and the conversion deliberately preserves which one each
	// field was. Pin both ends per field.
	// headA exercises one nil/empty assignment per field; headB exercises the
	// complementary one, so every arm of every per-head three-way switch is
	// hit in a single conversion.
	headA := metal.KVHeadSnapshot{
		Key:        []float32{}, // empty -> empty
		KeyBytes:   nil,         // nil -> nil
		Value:      nil,         // nil -> nil
		ValueBytes: []byte{},    // empty -> empty
	}
	headB := metal.KVHeadSnapshot{
		Key:        nil,         // nil -> nil
		KeyBytes:   []byte{},    // empty -> empty
		Value:      []float32{}, // empty -> empty
		ValueBytes: nil,         // nil -> nil
	}
	src := &metal.KVSnapshot{
		Version:    4,
		Tokens:     []int32{},   // empty -> empty
		Generated:  nil,         // nil -> nil
		LogitShape: nil,         // nil -> nil
		Logits:     []float32{}, // empty -> empty
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			KeyShape:   nil,       // nil -> nil
			ValueShape: []int32{}, // empty -> empty
			Heads:      []metal.KVHeadSnapshot{headA},
		}, {
			Layer:      1,
			KeyShape:   []int32{}, // empty -> empty
			ValueShape: nil,       // nil -> nil
			Heads:      []metal.KVHeadSnapshot{headB},
		}},
	}
	got := ToRootKVSnapshot(src)
	if got == nil {
		t.Fatal("ToRootKVSnapshot() = nil, want snapshot")
	}
	if got.Tokens == nil || len(got.Tokens) != 0 {
		t.Fatalf("Tokens = %v, want non-nil empty preserved", got.Tokens)
	}
	if got.Generated != nil {
		t.Fatalf("Generated = %v, want nil preserved", got.Generated)
	}
	if got.LogitShape != nil {
		t.Fatalf("LogitShape = %v, want nil preserved", got.LogitShape)
	}
	if got.Logits == nil || len(got.Logits) != 0 {
		t.Fatalf("Logits = %v, want non-nil empty preserved", got.Logits)
	}
	// Layer 0: KeyShape nil, ValueShape empty.
	if got.Layers[0].KeyShape != nil {
		t.Fatalf("layer0 KeyShape = %v, want nil preserved", got.Layers[0].KeyShape)
	}
	if got.Layers[0].ValueShape == nil || len(got.Layers[0].ValueShape) != 0 {
		t.Fatalf("layer0 ValueShape = %v, want non-nil empty preserved", got.Layers[0].ValueShape)
	}
	// Layer 1: KeyShape empty, ValueShape nil (complementary).
	if got.Layers[1].KeyShape == nil || len(got.Layers[1].KeyShape) != 0 {
		t.Fatalf("layer1 KeyShape = %v, want non-nil empty preserved", got.Layers[1].KeyShape)
	}
	if got.Layers[1].ValueShape != nil {
		t.Fatalf("layer1 ValueShape = %v, want nil preserved", got.Layers[1].ValueShape)
	}
	// headA: Key empty, KeyBytes nil, Value nil, ValueBytes empty.
	a := got.Layers[0].Heads[0]
	if a.Key == nil || len(a.Key) != 0 {
		t.Fatalf("headA.Key = %v, want non-nil empty preserved", a.Key)
	}
	if a.KeyBytes != nil {
		t.Fatalf("headA.KeyBytes = %v, want nil preserved", a.KeyBytes)
	}
	if a.Value != nil {
		t.Fatalf("headA.Value = %v, want nil preserved", a.Value)
	}
	if a.ValueBytes == nil || len(a.ValueBytes) != 0 {
		t.Fatalf("headA.ValueBytes = %v, want non-nil empty preserved", a.ValueBytes)
	}
	// headB: Key nil, KeyBytes empty, Value empty, ValueBytes nil.
	b := got.Layers[1].Heads[0]
	if b.Key != nil {
		t.Fatalf("headB.Key = %v, want nil preserved", b.Key)
	}
	if b.KeyBytes == nil || len(b.KeyBytes) != 0 {
		t.Fatalf("headB.KeyBytes = %v, want non-nil empty preserved", b.KeyBytes)
	}
	if b.Value == nil || len(b.Value) != 0 {
		t.Fatalf("headB.Value = %v, want non-nil empty preserved", b.Value)
	}
	if b.ValueBytes != nil {
		t.Fatalf("headB.ValueBytes = %v, want nil preserved", b.ValueBytes)
	}
	// Empty tensors carry no dtype tag (RootKVHeadDType returns "" for empty).
	if a.KeyDType != "" {
		t.Fatalf("headA.KeyDType = %q, want empty for empty bytes", a.KeyDType)
	}
}

func TestKvconv_ToRootKVSnapshot_EmptyTopLevelSlices_Ugly(t *testing.T) {
	// The _Ugly case above pins nil Generated / nil LogitShape; this pins the
	// complementary empty-but-non-nil arm of the SAME two top-level switches
	// (empty -> []int32{}, not nil). Together they cover every arm of the
	// Tokens/Generated/LogitShape three-way at the snapshot top level.
	src := &metal.KVSnapshot{
		Version:    4,
		Tokens:     nil,       // nil -> nil
		Generated:  []int32{}, // empty -> empty (the uncovered arm)
		LogitShape: []int32{}, // empty -> empty (the uncovered arm)
		Logits:     nil,       // nil -> nil
		Layers:     nil,
	}
	got := ToRootKVSnapshot(src)
	if got == nil {
		t.Fatal("ToRootKVSnapshot() = nil, want snapshot")
	}
	if got.Tokens != nil {
		t.Fatalf("Tokens = %v, want nil preserved", got.Tokens)
	}
	if got.Generated == nil || len(got.Generated) != 0 {
		t.Fatalf("Generated = %v, want non-nil empty preserved", got.Generated)
	}
	if got.LogitShape == nil || len(got.LogitShape) != 0 {
		t.Fatalf("LogitShape = %v, want non-nil empty preserved", got.LogitShape)
	}
	if got.Logits != nil {
		t.Fatalf("Logits = %v, want nil preserved", got.Logits)
	}
}

// --- ToMetalKVSnapshot -----------------------------------------------------

func TestKvconv_ToMetalKVSnapshot_Good(t *testing.T) {
	// Heads-float32 source (no native slab): per-head K/V must be copied into
	// the metal snapshot and the dtype tags resolved to metal.DType.
	src := &kv.Snapshot{
		Version:       4,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  "fp16",
			Heads: []kv.HeadSnapshot{{
				Key:        []float32{1, 2},
				KeyDType:   "float32",
				KeyBytes:   []byte{0x10, 0x20},
				Value:      []float32{3, 4},
				ValueDType: "float32",
				ValueBytes: []byte{0x30, 0x40},
			}},
		}},
	}
	got := ToMetalKVSnapshot(src)
	if got == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if got.Version != 4 || got.Architecture != "gemma4_text" || got.NumHeads != 2 {
		t.Fatalf("scalars = {%d %q %d}, want {4 gemma4_text 2}", got.Version, got.Architecture, got.NumHeads)
	}
	if len(got.Layers) != 1 || got.Layers[0].CacheMode != metal.KVCacheModeFP16 {
		t.Fatalf("cache mode = %q, want fp16", got.Layers[0].CacheMode)
	}
	head := got.Layers[0].Heads[0]
	// Head-level bytes are present (but the LAYER has no native slab, so this
	// is still the per-head copy path) — the dtype tag resolves from those
	// head bytes via MetalKVHeadDType.
	if !reflect.DeepEqual(head.Key, []float32{1, 2}) || head.KeyDType != metal.DTypeFloat32 {
		t.Fatalf("head key = %v dtype %v, want [1 2] float32", head.Key, head.KeyDType)
	}
	if !reflect.DeepEqual(head.Value, []float32{3, 4}) || head.ValueDType != metal.DTypeFloat32 {
		t.Fatalf("head value = %v dtype %v, want [3 4] float32", head.Value, head.ValueDType)
	}
	// Heads-path copies are independent of the source.
	src.Layers[0].Heads[0].Key[0] = 999
	if got.Layers[0].Heads[0].Key[0] == 999 {
		t.Fatal("ToMetalKVSnapshot aliased the source head Key on the heads path; want a copy")
	}
}

func TestKvconv_ToMetalKVSnapshot_Bad(t *testing.T) {
	if got := ToMetalKVSnapshot(nil); got != nil {
		t.Fatalf("ToMetalKVSnapshot(nil) = %+v, want nil", got)
	}
}

func TestKvconv_ToMetalKVSnapshot_Ugly(t *testing.T) {
	// Native-slab layer: the metal restorer reads only KeyBytes/ValueBytes,
	// so the per-head float32 must pass through BY REFERENCE (zero-copy, no
	// fresh slab). A native layer is one where both KeyBytes and ValueBytes
	// are populated. Verify the heads alias the source rather than being
	// copied — this is the doubling-avoidance contract.
	key := []float32{1, 2, 3, 4}
	value := []float32{5, 6, 7, 8}
	src := &kv.Snapshot{
		Version:    4,
		NumLayers:  2,
		NumHeads:   1,
		Tokens:     []int32{}, // empty -> empty
		Generated:  []int32{}, // empty -> empty
		LogitShape: nil,       // nil -> nil
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			KeyDType:   "float16",
			KeyBytes:   []byte{0xAA, 0xBB},
			ValueDType: "float16",
			ValueBytes: []byte{0xCC, 0xDD},
			Heads: []kv.HeadSnapshot{{
				Key:        key,
				KeyDType:   "float32",
				Value:      value,
				ValueDType: "float32",
			}},
		}, {
			// Non-native layer (no slab bytes) with empty shapes and an empty
			// head — exercises the heads-path three-way switches the native
			// passthrough above skips.
			Layer:      1,
			KeyShape:   []int32{}, // empty -> empty
			ValueShape: []int32{}, // empty -> empty
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{}, // empty -> empty
				Value: []float32{}, // empty -> empty
			}},
		}},
	}
	got := ToMetalKVSnapshot(src)
	if got == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	// Top-level empty/nil slices preserved on the metal direction too.
	if got.Tokens == nil || len(got.Tokens) != 0 {
		t.Fatalf("Tokens = %v, want non-nil empty preserved", got.Tokens)
	}
	if got.LogitShape != nil {
		t.Fatalf("LogitShape = %v, want nil preserved", got.LogitShape)
	}
	// Non-native layer's empty head slices preserved as empty (not nil).
	nh := got.Layers[1].Heads[0]
	if nh.Key == nil || len(nh.Key) != 0 || nh.Value == nil || len(nh.Value) != 0 {
		t.Fatalf("non-native head = key:%v value:%v, want non-nil empty each", nh.Key, nh.Value)
	}
	if got.Layers[1].KeyShape == nil || len(got.Layers[1].KeyShape) != 0 {
		t.Fatalf("non-native KeyShape = %v, want non-nil empty preserved", got.Layers[1].KeyShape)
	}
	layer := got.Layers[0]
	// Layer bytes pass through by reference and the dtype tag resolves.
	if !reflect.DeepEqual(layer.KeyBytes, []byte{0xAA, 0xBB}) || layer.KeyDType != metal.DTypeFloat16 {
		t.Fatalf("layer key bytes/dtype = %v/%v, want [AA BB]/float16", layer.KeyBytes, layer.KeyDType)
	}
	gotHead := layer.Heads[0]
	// The per-head float32 must be the SAME backing array as the source
	// (reference passthrough). Mutating the source is therefore visible.
	src.Layers[0].Heads[0].Key[0] = 777
	if gotHead.Key[0] != 777 {
		t.Fatal("native-slab head Key was copied; want reference passthrough (zero-copy)")
	}
	src.Layers[0].Heads[0].Value[0] = 888
	if gotHead.Value[0] != 888 {
		t.Fatal("native-slab head Value was copied; want reference passthrough (zero-copy)")
	}
}

func TestKvconv_ToMetalKVSnapshot_PopulatedTopLevelSlices_Good(t *testing.T) {
	// The metal direction lays Tokens/Generated/LogitShape down in a shared
	// int32 arena and Logits in the tail of the float32 slab. The _Good /
	// _Ugly cases above leave Generated, LogitShape and Logits empty or nil, so
	// the non-empty default arm of each of those three top-level switches went
	// unexercised. Populate all four (with a heads-path layer so the float32
	// slab is allocated) and confirm each carries across with values intact.
	src := &kv.Snapshot{
		Version:    4,
		Tokens:     []int32{1, 2, 3},
		Generated:  []int32{4, 5},
		LogitShape: []int32{1, 8},
		Logits:     []float32{0.1, 0.2, 0.3, 0.4},
		NumLayers:  1,
		Layers: []kv.LayerSnapshot{{
			Layer: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}
	got := ToMetalKVSnapshot(src)
	if got == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if !reflect.DeepEqual(got.Tokens, []int32{1, 2, 3}) {
		t.Fatalf("Tokens = %v, want [1 2 3]", got.Tokens)
	}
	if !reflect.DeepEqual(got.Generated, []int32{4, 5}) {
		t.Fatalf("Generated = %v, want [4 5]", got.Generated)
	}
	if !reflect.DeepEqual(got.LogitShape, []int32{1, 8}) {
		t.Fatalf("LogitShape = %v, want [1 8]", got.LogitShape)
	}
	if !reflect.DeepEqual(got.Logits, []float32{0.1, 0.2, 0.3, 0.4}) {
		t.Fatalf("Logits = %v, want [0.1 0.2 0.3 0.4]", got.Logits)
	}
	// The top-level int32 slices are cut from the shared arena; mutating the
	// source must not be visible through the converted snapshot.
	src.Generated[0] = 999
	if got.Generated[0] == 999 {
		t.Fatal("ToMetalKVSnapshot aliased the source Generated; want an independent copy")
	}
}

func TestKvconv_ToMetalKVSnapshot_EmptyTopLevelSlices_Ugly(t *testing.T) {
	// Complements _PopulatedTopLevelSlices_Good: that case pins the non-empty
	// default arm of the metal-direction LogitShape / Logits switches; the
	// existing _Ugly pins their nil arms. This pins the empty-but-non-nil arm
	// (empty in -> empty out) of both, completing the three-way on the metal
	// direction's top-level LogitShape and Logits.
	src := &kv.Snapshot{
		Version:    4,
		Tokens:     nil,         // nil -> nil
		Generated:  nil,         // nil -> nil
		LogitShape: []int32{},   // empty -> empty (the uncovered arm)
		Logits:     []float32{}, // empty -> empty (the uncovered arm)
		Layers:     nil,
	}
	got := ToMetalKVSnapshot(src)
	if got == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if got.LogitShape == nil || len(got.LogitShape) != 0 {
		t.Fatalf("LogitShape = %v, want non-nil empty preserved", got.LogitShape)
	}
	if got.Logits == nil || len(got.Logits) != 0 {
		t.Fatalf("Logits = %v, want non-nil empty preserved", got.Logits)
	}
}

// --- round trip ------------------------------------------------------------

func TestKvconv_RoundTripMetalRootMetal_Good(t *testing.T) {
	// A heads-float32 metal snapshot survives metal->root->metal with values
	// intact. This guards against a regression where one direction drops or
	// reshapes a field the other preserves. The round trip alone can hide
	// compensating bugs, so the per-direction asserts above carry the real
	// weight; this is the integration backstop.
	src := &metal.KVSnapshot{
		Version:       4,
		Architecture:  "round",
		Tokens:        []int32{7, 8, 9},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  metal.KVCacheModeQ8,
			MaxSize:    8,
			Heads: []metal.KVHeadSnapshot{{
				Key:        []float32{1.5, 2.5},
				KeyDType:   metal.DTypeFloat32,
				KeyBytes:   []byte{0x01, 0x02},
				Value:      []float32{3.5, 4.5},
				ValueDType: metal.DTypeFloat32,
				ValueBytes: []byte{0x03, 0x04},
			}},
		}},
	}
	back := ToMetalKVSnapshot(ToRootKVSnapshot(src))
	if back == nil {
		t.Fatal("round trip produced nil")
	}
	if back.Version != src.Version || back.Architecture != src.Architecture {
		t.Fatalf("scalars drifted: got {%d %q}, want {%d %q}", back.Version, back.Architecture, src.Version, src.Architecture)
	}
	if !reflect.DeepEqual(back.Tokens, src.Tokens) {
		t.Fatalf("tokens drifted: got %v, want %v", back.Tokens, src.Tokens)
	}
	if back.Layers[0].CacheMode != metal.KVCacheModeQ8 {
		t.Fatalf("cache mode drifted: got %q, want q8", back.Layers[0].CacheMode)
	}
	h := back.Layers[0].Heads[0]
	if !reflect.DeepEqual(h.Key, []float32{1.5, 2.5}) || !reflect.DeepEqual(h.Value, []float32{3.5, 4.5}) {
		t.Fatalf("head tensors drifted: key %v value %v", h.Key, h.Value)
	}
	if h.KeyDType != metal.DTypeFloat32 || h.ValueDType != metal.DTypeFloat32 {
		t.Fatalf("head dtypes drifted: key %v value %v", h.KeyDType, h.ValueDType)
	}
}

// --- TurboQuant payloads ---------------------------------------------------

// validTurboQuantPayload mirrors the metal package's own valid reference page
// layout (kept here because that helper lives in an internal _test.go). The
// layout passes TurboQuantKVPageLayout.Validate(), which the metal direction
// runs on every payload — so this is the success-path fixture for both
// rootTurboQuantPayloads (marshal) and metalTurboQuantPayloads (unmarshal +
// validate).
func validTurboQuantPayload() metal.TurboQuantKVReferencePagePayload {
	return metal.TurboQuantKVReferencePagePayload{
		Layout: metal.TurboQuantKVPageLayout{
			Version:     metal.TurboQuantKVLayoutVersion,
			Codec:       metal.TurboQuantKVCodecName,
			CacheIndex:  1,
			Layer:       5,
			LayerType:   "full_attention",
			SharedOwner: 5,
			Shape:       metal.TurboQuantKVShape{Batch: 1, Heads: 2, SeqLen: 2, HeadDim: 8},
			TokenOffset: 16,
			PageTokens:  2,
			PageSize:    2,
			Key: metal.TurboQuantKVCodec{
				Algorithm:          metal.TurboQuantKVAlgorithmProd,
				NormalBits:         5,
				NormPolicy:         metal.TurboQuantKVNormPolicyExplicitVectorBF16V1,
				ResidualNormPolicy: metal.TurboQuantKVResidualNormPolicyExplicitVectorBF16V1,
				RotationSeed:       0x6b,
				QJLSeed:            0x7c,
				CodebookID:         metal.TurboQuantKVReferenceCodebookUniform,
			},
			Value: metal.TurboQuantKVCodec{
				Algorithm:    metal.TurboQuantKVAlgorithmMSE,
				NormalBits:   5,
				NormPolicy:   metal.TurboQuantKVNormPolicyExplicitVectorBF16V1,
				RotationSeed: 0x56,
				CodebookID:   metal.TurboQuantKVReferenceCodebookUniform,
			},
		},
		Endian:    "little",
		Alignment: 16,
		Data:      []byte{1, 2, 3, 4},
	}
}

func TestKvconv_TurboQuantPayloadsRoundTrip_Good(t *testing.T) {
	// A layer carrying a valid TurboQuant reference payload survives
	// metal->root (JSON marshal) and root->metal (JSON unmarshal + layout
	// validate). This is the only path that covers both unexported payload
	// helpers' success branches.
	payload := validTurboQuantPayload()
	src := &metal.KVSnapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:              0,
			CacheMode:          metal.KVCacheModeTurboQuant,
			TurboQuantPayloads: []metal.TurboQuantKVReferencePagePayload{payload},
		}},
	}
	root := ToRootKVSnapshot(src)
	if root == nil || len(root.Layers) != 1 {
		t.Fatalf("ToRootKVSnapshot() = %+v, want one layer", root)
	}
	// The root side stores payloads as opaque JSON byte blobs.
	if len(root.Layers[0].TurboQuantPayloads) != 1 || len(root.Layers[0].TurboQuantPayloads[0]) == 0 {
		t.Fatalf("root TurboQuant payloads = %v, want one non-empty JSON blob", root.Layers[0].TurboQuantPayloads)
	}
	back := ToMetalKVSnapshot(root)
	if back == nil || len(back.Layers) != 1 {
		t.Fatalf("ToMetalKVSnapshot() = %+v, want one layer", back)
	}
	got := back.Layers[0].TurboQuantPayloads
	if len(got) != 1 {
		t.Fatalf("metal TurboQuant payloads = %d, want 1 (valid layout survives the round trip)", len(got))
	}
	if got[0].Layout.Codec != metal.TurboQuantKVCodecName || got[0].Layout.Layer != 5 {
		t.Fatalf("payload layout drifted: codec %q layer %d", got[0].Layout.Codec, got[0].Layout.Layer)
	}
	if !reflect.DeepEqual(got[0].Data, []byte{1, 2, 3, 4}) {
		t.Fatalf("payload data drifted: %v", got[0].Data)
	}
}

func TestKvconv_TurboQuantPayloadsRoundTrip_Bad(t *testing.T) {
	// An INVALID layout (version 0 fails TurboQuantKVPageLayout.Validate) must
	// be rejected by the metal direction: metalTurboQuantPayloads returns nil
	// for the whole layer rather than admitting an unvalidatable payload.
	bad := validTurboQuantPayload()
	bad.Layout.Version = 0 // unsupported version -> Validate() fails
	src := &metal.KVSnapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:              0,
			TurboQuantPayloads: []metal.TurboQuantKVReferencePagePayload{bad},
		}},
	}
	root := ToRootKVSnapshot(src) // marshal still succeeds (no validation there)
	if root == nil || len(root.Layers[0].TurboQuantPayloads) != 1 {
		t.Fatalf("ToRootKVSnapshot() dropped the payload unexpectedly: %+v", root)
	}
	back := ToMetalKVSnapshot(root)
	if back == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if back.Layers[0].TurboQuantPayloads != nil {
		t.Fatalf("metal TurboQuant payloads = %v, want nil for an invalid layout", back.Layers[0].TurboQuantPayloads)
	}
}

func TestKvconv_TurboQuantPayloadsMalformedJSON_Bad(t *testing.T) {
	// A non-empty payload blob that is not valid JSON fails the unmarshal step
	// (distinct from the empty-blob arm in _Ugly and the invalid-layout arm in
	// _Bad above): metalTurboQuantPayloads returns nil for the whole layer
	// rather than admitting a half-decoded payload.
	root := &kv.Snapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:              0,
			TurboQuantPayloads: [][]byte{[]byte("{not valid json")}, // non-empty, unparseable
		}},
	}
	back := ToMetalKVSnapshot(root)
	if back == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if back.Layers[0].TurboQuantPayloads != nil {
		t.Fatalf("metal TurboQuant payloads = %v, want nil for unparseable JSON", back.Layers[0].TurboQuantPayloads)
	}
}

func TestKvconv_TurboQuantPayloadsRoundTrip_Ugly(t *testing.T) {
	// An empty payload byte blob on the root side is treated as corrupt: the
	// metal direction bails on the first zero-length entry and returns nil for
	// the layer rather than a partial slice.
	root := &kv.Snapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:              0,
			TurboQuantPayloads: [][]byte{{}}, // empty blob -> reject
		}},
	}
	back := ToMetalKVSnapshot(root)
	if back == nil {
		t.Fatal("ToMetalKVSnapshot() = nil, want snapshot")
	}
	if back.Layers[0].TurboQuantPayloads != nil {
		t.Fatalf("metal TurboQuant payloads = %v, want nil for an empty blob", back.Layers[0].TurboQuantPayloads)
	}
}
