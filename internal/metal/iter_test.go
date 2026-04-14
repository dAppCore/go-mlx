//go:build darwin && arm64 && !nomlx

package metal

import (
	"testing"
)

func TestArray_Iter_Good(t *testing.T) {
	a := FromValues([]float32{1.5, 2.5, 3.5, 4.5}, 4)
	Materialize(a)

	var got []float32
	for v := range a.Iter() {
		got = append(got, v)
	}

	want := []float32{1.5, 2.5, 3.5, 4.5}
	if len(got) != len(want) {
		t.Fatalf("Iter: got %d elements, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Iter[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestArray_Iter_2D_Good(t *testing.T) {
	// Iter flattens — [2,3] yields 6 elements.
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	Materialize(a)

	count := 0
	sum := float32(0)
	for v := range a.Iter() {
		count++
		sum += v
	}

	if count != 6 {
		t.Errorf("Iter 2D: got %d elements, want 6", count)
	}
	if sum != 21 {
		t.Errorf("Iter 2D: sum = %f, want 21", sum)
	}
}

func TestArray_Iter_Transposed_Good(t *testing.T) {
	// Iter on a non-contiguous (transposed) array should still work.
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2) // [[1,2],[3,4]]
	tr := Transpose(a)                             // [[1,3],[2,4]]
	Materialize(tr)

	var got []float32
	for v := range tr.Iter() {
		got = append(got, v)
	}

	want := []float32{1, 3, 2, 4}
	if len(got) != len(want) {
		t.Fatalf("Iter transposed: got %d elements, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Iter transposed[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestArray_Iter_EarlyBreak_Good(t *testing.T) {
	a := FromValues([]float32{10, 20, 30, 40, 50}, 5)
	Materialize(a)

	var got []float32
	for v := range a.Iter() {
		got = append(got, v)
		if len(got) == 3 {
			break
		}
	}

	if len(got) != 3 {
		t.Fatalf("Iter early break: got %d elements, want 3", len(got))
	}
	want := []float32{10, 20, 30}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Iter early break[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
