// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"errors"
	"testing"
)

// session_snapshot_shape_cover_test.go covers the pure snapshot shape-inference,
// validation, and logits-restore helpers session.go exposes for KV restore — the
// branches the model-driven snapshot tests leave dark. All synthetic: plain
// KVSnapshot / KVHeadSnapshot structs and a fakeModel, no weights.

func TestInt32ShapeToInts_Good(t *testing.T) {
	if got := int32ShapeToInts(nil); len(got) != 0 {
		t.Errorf("nil shape → %v, want empty", got)
	}
	got := int32ShapeToInts([]int32{1, 16, 4, 128})
	want := []int{1, 16, 4, 128}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("int32ShapeToInts = %v, want %v", got, want)
		}
	}
}

func TestInferSnapshotTensorElementCacheShape_Branches_Good(t *testing.T) {
	cases := []struct {
		name                    string
		elems, seqLen, fallback int
		wantLen, wantDim        int
	}{
		{"zero elements", 0, 4, 8, 0, 0},
		{"divides by seqLen", 32, 4, 8, 4, 8},              // 32/4 = 8 dim
		{"falls back to head dim", 24, 5, 8, 3, 8},         // 24%5!=0 → 24/8 = 3 len
		{"neither divides", 7, 4, 8, 0, 0},                 // 7%4!=0, 7%8!=0
		{"seqLen preferred over fallback", 16, 4, 8, 4, 4}, // 16%4==0 wins
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gl, gd := inferSnapshotTensorElementCacheShape(tc.elems, tc.seqLen, tc.fallback)
			if gl != tc.wantLen || gd != tc.wantDim {
				t.Fatalf("got (%d,%d), want (%d,%d)", gl, gd, tc.wantLen, tc.wantDim)
			}
		})
	}
}

func TestInferSnapshotTensorCacheShape_FloatPath_Good(t *testing.T) {
	if l, d := inferSnapshotTensorCacheShape(nil, 4, 8); l != 0 || d != 0 {
		t.Fatalf("empty values → (%d,%d), want (0,0)", l, d)
	}
	vals := make([]float32, 32)
	if l, d := inferSnapshotTensorCacheShape(vals, 4, 8); l != 4 || d != 8 {
		t.Fatalf("32 floats, seqLen 4 → (%d,%d), want (4,8)", l, d)
	}
}

func TestInferSnapshotHeadDim_Good(t *testing.T) {
	if got := inferSnapshotHeadDim(make([]float32, 32), 0); got != 0 {
		t.Errorf("seqLen 0 → %d, want 0", got)
	}
	if got := inferSnapshotHeadDim(make([]float32, 30), 4); got != 0 {
		t.Errorf("30%%4!=0 → %d, want 0", got)
	}
	if got := inferSnapshotHeadDim(make([]float32, 32), 4); got != 8 {
		t.Errorf("32/4 → %d, want 8", got)
	}
}

func TestInferSnapshotHeadTensorCacheShape_FloatAndRaw_Good(t *testing.T) {
	// Float key path.
	floatHead := KVHeadSnapshot{Key: make([]float32, 32)}
	if l, d := inferSnapshotHeadTensorCacheShape(floatHead, 4, 8, true); l != 4 || d != 8 {
		t.Fatalf("float key → (%d,%d), want (4,8)", l, d)
	}
	// Raw-bytes value path (no float slice): 16 float32 bytes = 4 elements.
	rawHead := KVHeadSnapshot{ValueBytes: make([]byte, 16*4), ValueDType: DTypeFloat32}
	if l, d := inferSnapshotHeadTensorCacheShape(rawHead, 4, 4, false); l != 4 || d != 4 {
		t.Fatalf("raw value → (%d,%d), want (4,4)", l, d)
	}
	// No data at all → (0,0).
	if l, d := inferSnapshotHeadTensorCacheShape(KVHeadSnapshot{}, 4, 8, true); l != 0 || d != 0 {
		t.Fatalf("empty head → (%d,%d), want (0,0)", l, d)
	}
}

func TestInferSnapshotLayerCacheShape_Branches_Good(t *testing.T) {
	if _, _, _, err := inferSnapshotLayerCacheShape(nil, 4, 8); !errors.Is(err, errSnapshotLayerNoHeads) {
		t.Fatalf("no heads → %v, want errSnapshotLayerNoHeads", err)
	}

	// Valid: key + value both 32 floats at seqLen 4 → len 4, dim 8.
	good := []KVHeadSnapshot{{Key: make([]float32, 32), Value: make([]float32, 32)}}
	kl, kd, vd, err := inferSnapshotLayerCacheShape(good, 4, 8)
	if err != nil || kl != 4 || kd != 8 || vd != 8 {
		t.Fatalf("valid layer → (%d,%d,%d,%v), want (4,8,8,nil)", kl, kd, vd, err)
	}

	// Invalid head dims (no resolvable shape).
	bad := []KVHeadSnapshot{{Key: []float32{1}, Value: []float32{1}}}
	if _, _, _, err := inferSnapshotLayerCacheShape(bad, 0, 0); !errors.Is(err, errSnapshotInvalidHeadDims) {
		t.Fatalf("unresolvable dims → %v, want errSnapshotInvalidHeadDims", err)
	}

	// Key/value lengths differ → errSnapshotKVLenDiffer. The seqLen path always
	// reports len==seqLen, so the lengths only diverge when one side takes the
	// fallback-head-dim path instead. At seqLen 7, fallback 8: key 56 floats →
	// 56%7==0 → len 7; value 32 floats → 32%7!=0 → fallback 32/8 → len 4.
	differ := []KVHeadSnapshot{{Key: make([]float32, 56), Value: make([]float32, 32)}}
	if _, _, _, err := inferSnapshotLayerCacheShape(differ, 7, 8); !errors.Is(err, errSnapshotKVLenDiffer) {
		t.Fatalf("kv len differ → %v, want errSnapshotKVLenDiffer", err)
	}
}

func TestValidateSnapshotHeadTensorCacheShape_Branches_Good(t *testing.T) {
	// Bad dims.
	if err := validateSnapshotHeadTensorCacheShape(KVHeadSnapshot{}, 0, 8, true); !errors.Is(err, errSnapshotInvalidHeadDims) {
		t.Fatalf("zero seqLen → %v, want errSnapshotInvalidHeadDims", err)
	}
	// Float slice with wrong element count → key tensor size error.
	wrong := KVHeadSnapshot{Key: make([]float32, 30)}
	if err := validateSnapshotHeadTensorCacheShape(wrong, 4, 8, true); !errors.Is(err, errSnapshotKeyTensorSize) {
		t.Fatalf("wrong key floats → %v, want errSnapshotKeyTensorSize", err)
	}
	// Float slice with the right count and no raw bytes → ok.
	right := KVHeadSnapshot{Key: make([]float32, 32)}
	if err := validateSnapshotHeadTensorCacheShape(right, 4, 8, true); err != nil {
		t.Fatalf("matching key floats → %v, want nil", err)
	}
	// No floats and no raw → value tensor size error on the value side.
	if err := validateSnapshotHeadTensorCacheShape(KVHeadSnapshot{}, 4, 8, false); !errors.Is(err, errSnapshotValueTensorSize) {
		t.Fatalf("empty value → %v, want errSnapshotValueTensorSize", err)
	}
	// Raw bytes with the right byte length → ok.
	rawOK := KVHeadSnapshot{KeyBytes: make([]byte, 32*4), KeyDType: DTypeFloat32}
	if err := validateSnapshotHeadTensorCacheShape(rawOK, 4, 8, true); err != nil {
		t.Fatalf("matching raw key → %v, want nil", err)
	}
	// Raw bytes with the wrong byte length → native key size error.
	rawBad := KVHeadSnapshot{KeyBytes: make([]byte, 10), KeyDType: DTypeFloat32}
	if err := validateSnapshotHeadTensorCacheShape(rawBad, 4, 8, true); !errors.Is(err, errSnapshotNativeKeySize) {
		t.Fatalf("wrong raw key bytes → %v, want errSnapshotNativeKeySize", err)
	}
}

func TestValidateKVSnapshot_Branches_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := &Model{model: &fakeModel{numLayers: 1}, modelType: "fake"}

	if err := m.validateKVSnapshot(nil); !errors.Is(err, errSnapshotNil) {
		t.Fatalf("nil snapshot → %v, want errSnapshotNil", err)
	}
	if err := m.validateKVSnapshot(&KVSnapshot{Version: 0}); !errors.Is(err, errUnsupportedSnapshotVersion) {
		t.Fatalf("version 0 → %v, want errUnsupportedSnapshotVersion", err)
	}
	if err := m.validateKVSnapshot(&KVSnapshot{Version: KVSnapshotVersion + 1}); !errors.Is(err, errUnsupportedSnapshotVersion) {
		t.Fatalf("future version → %v, want errUnsupportedSnapshotVersion", err)
	}
	// Architecture mismatch (model is "fake", snapshot says "other").
	if err := m.validateKVSnapshot(&KVSnapshot{Version: 1, Architecture: "other", SeqLen: 4, HeadDim: 8, Layers: []KVLayerSnapshot{{}}}); !errors.Is(err, errSnapshotArchMismatch) {
		t.Fatalf("arch mismatch → %v, want errSnapshotArchMismatch", err)
	}
	// Invalid tensor dims.
	if err := m.validateKVSnapshot(&KVSnapshot{Version: 1, SeqLen: 0, HeadDim: 8, Layers: []KVLayerSnapshot{{}}}); !errors.Is(err, errSnapshotInvalidTensorDims) {
		t.Fatalf("zero seqLen → %v, want errSnapshotInvalidTensorDims", err)
	}
	// No layers.
	if err := m.validateKVSnapshot(&KVSnapshot{Version: 1, SeqLen: 4, HeadDim: 8}); !errors.Is(err, errSnapshotNoLayers) {
		t.Fatalf("no layers → %v, want errSnapshotNoLayers", err)
	}
	// Fully valid (architecture left empty → arch check skipped).
	if err := m.validateKVSnapshot(&KVSnapshot{Version: 1, SeqLen: 4, HeadDim: 8, Layers: []KVLayerSnapshot{{}}}); err != nil {
		t.Fatalf("valid snapshot → %v, want nil", err)
	}
}

func TestRestoreSnapshotLogits_Branches_Good(t *testing.T) {
	requireMetalRuntime(t)

	if _, err := restoreSnapshotLogits(nil); !errors.Is(err, errSnapshotNil) {
		t.Fatalf("nil → %v, want errSnapshotNil", err)
	}
	if _, err := restoreSnapshotLogits(&KVSnapshot{}); !errors.Is(err, errSnapshotNoRestorableLogits) {
		t.Fatalf("no logits → %v, want errSnapshotNoRestorableLogits", err)
	}
	// Non-positive dimension in the shape.
	bad := &KVSnapshot{Logits: []float32{1, 2}, LogitShape: []int32{0, 2}}
	if _, err := restoreSnapshotLogits(bad); !errors.Is(err, errSnapshotLogitShape) {
		t.Fatalf("zero dim → %v, want errSnapshotLogitShape", err)
	}
	// Shape product mismatches the logits length.
	mismatch := &KVSnapshot{Logits: []float32{1, 2, 3}, LogitShape: []int32{1, 4}}
	if _, err := restoreSnapshotLogits(mismatch); !errors.Is(err, errSnapshotLogitsShapeMismatch) {
		t.Fatalf("shape mismatch → %v, want errSnapshotLogitsShapeMismatch", err)
	}
	// Valid: 4 logits, shape [1,4].
	ok := &KVSnapshot{Logits: []float32{1, 2, 3, 4}, LogitShape: []int32{1, 4}}
	arr, err := restoreSnapshotLogits(ok)
	if err != nil {
		t.Fatalf("valid logits → %v, want nil", err)
	}
	defer Free(arr)
	if arr.Dim(0) != 1 || arr.Dim(1) != 4 {
		t.Fatalf("restored logits shape = %v, want [1 4]", arr.Shape())
	}
}

func TestSnapshotCacheLength_Branches_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Explicit length wins.
	if got := snapshotCacheLength(cacheSnapshot{length: 7, offset: 3}); got != 7 {
		t.Errorf("explicit length → %d, want 7", got)
	}
	// Falls back to the key tensor's seq dim (axis 2 of a rank-4 [1,H,L,D]).
	keys := Zeros([]int32{1, 2, 5, 8}, DTypeFloat32)
	defer Free(keys)
	if got := snapshotCacheLength(cacheSnapshot{keys: keys}); got != 5 {
		t.Errorf("key seq-dim → %d, want 5", got)
	}
	// No length, no keys → offset.
	if got := snapshotCacheLength(cacheSnapshot{offset: 9}); got != 9 {
		t.Errorf("offset fallback → %d, want 9", got)
	}
}
