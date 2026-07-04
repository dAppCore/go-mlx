// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestSharedKV_HasState_Bad(t *testing.T) {
	// The zero value carries neither contiguous tensors nor pages.
	if (SharedKV{}).HasState() {
		t.Fatal("zero SharedKV.HasState() = true, want false")
	}
	// Only one of K/V present is not usable state.
	if (SharedKV{Keys: &Array{}}).HasState() {
		t.Fatal("Keys-only SharedKV.HasState() = true, want false (invalid + no Values)")
	}
}

func TestSharedKV_HasPages_Bad(t *testing.T) {
	// Empty page state, and mismatched-length K/V page slices, are both not a
	// complete paged state — these are pure-Go length/validity checks.
	if (SharedKV{}).HasPages() {
		t.Fatal("zero SharedKV.HasPages() = true, want false")
	}
	mismatched := SharedKV{Pages: PagedKVState{
		Keys:   []*Array{{}},
		Values: []*Array{{}, {}},
	}}
	if mismatched.HasPages() {
		t.Fatal("mismatched K/V page counts HasPages() = true, want false")
	}
	// Equal-length but with a nil/invalid page → still incomplete.
	withNil := SharedKV{Pages: PagedKVState{
		Keys:   []*Array{nil},
		Values: []*Array{nil},
	}}
	if withNil.HasPages() {
		t.Fatal("page slice with nil entries HasPages() = true, want false")
	}
}

func TestSharedKV_MoveSharedKV_Ugly(t *testing.T) {
	// Move on a nil pointer yields the zero value, not a panic.
	if got := MoveSharedKV(nil); got.HasState() || got.Offset != 0 || got.Fixed {
		t.Fatalf("MoveSharedKV(nil) = %+v, want zero SharedKV", got)
	}
	// Move transfers the scalars out and leaves the source zeroed.
	src := SharedKV{Offset: 7, Fixed: true}
	moved := MoveSharedKV(&src)
	if moved.Offset != 7 || !moved.Fixed {
		t.Fatalf("moved = %+v, want Offset=7 Fixed=true", moved)
	}
	if src.Offset != 0 || src.Fixed {
		t.Fatalf("source after move = %+v, want zeroed", src)
	}
}

func TestSharedKV_Clone_Bad(t *testing.T) {
	// Cloning a zero/scalar-only state copies the scalars and needs no Metal —
	// there are no valid tensors to deep-copy.
	src := SharedKV{Offset: 5, Fixed: true}
	clone := src.Clone()
	if clone.Offset != 5 || !clone.Fixed || clone.Keys != nil || clone.Values != nil {
		t.Fatalf("Clone of scalar-only = %+v, want scalars copied and nil tensors", clone)
	}
	if clone.HasPages() {
		t.Fatal("clone of empty pages HasPages() = true, want false")
	}
	// Free on a zero/borrowed-free state must be a safe no-op.
	src.Free()
	(SharedKV{Borrowed: true}).Free()
}

func TestSharedKV_HasState_Good(t *testing.T) {
	requireMetalRuntime(t)

	// A state with valid contiguous K and V tensors is usable, and Clone deep
	// copies them (distinct handles, identical shapes).
	kv := SharedKV{
		Keys:   FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2),
		Values: FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2),
		Offset: 2,
	}
	defer kv.Free()
	if !kv.HasState() {
		t.Fatal("populated SharedKV.HasState() = false, want true")
	}

	clone := kv.Clone()
	defer clone.Free()
	if !clone.HasState() || clone.Offset != 2 {
		t.Fatalf("clone = %+v, want HasState and Offset=2", clone)
	}
	if clone.Keys == kv.Keys || clone.Values == kv.Values {
		t.Fatal("Clone returned aliased tensor handles, want deep copies")
	}
	if err := Eval(clone.Keys, clone.Values); err != nil {
		t.Fatalf("Eval(clone tensors): %v", err)
	}
}
