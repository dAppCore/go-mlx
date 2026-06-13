// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestGreaterEqual_Good pins the a>=b primitive that replaces the local
// ¬(b>a) composition in NSA/MoBA: equal and greater are true, strictly-less is
// false. Bool arrays read back as 0/1 floats.
func TestGreaterEqual_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{2, 2, 2}, 3)
	ge := GreaterEqual(a, b)
	Materialize(ge)
	// 1>=2 false, 2>=2 true, 3>=2 true.
	floatSliceApprox(t, ge.Floats(), []float32{0, 1, 1})
}

// TestTopNIndices_Good pins the top-n index selector, largest-first. Scores
// [3,1,2] → top-2 indices = [0,2] (block0=3 largest, block2=2 next).
func TestTopNIndices_Good(t *testing.T) {
	scores := FromValues([]float32{3, 1, 2}, 1, 1, 1, 3)
	idx := TopNIndices(scores, 2)
	Materialize(idx)
	got := idx.DataInt32()
	want := []int32{0, 2}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("topN[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestTopNIndices_ClampsN_Good confirms n is clamped to the block count: asking
// for more blocks than exist returns all of them, largest-first.
func TestTopNIndices_ClampsN_Good(t *testing.T) {
	scores := FromValues([]float32{5, 9, 1}, 1, 1, 1, 3)
	idx := TopNIndices(scores, 10) // clamp to 3
	Materialize(idx)
	got := idx.DataInt32()
	want := []int32{1, 0, 2} // 9, 5, 1 descending
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3 (clamped)", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("topN[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestReverseLastAxis_Good pins the descending-gather reversal helper.
func TestReverseLastAxis_Good(t *testing.T) {
	a := FromValues([]int32{10, 20, 30, 40}, 1, 4)
	rev := reverseLastAxis(a)
	Materialize(rev)
	got := rev.DataInt32()
	want := []int32{40, 30, 20, 10}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rev[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestGatherSelectedBlocks_Good pins the block gather. 3 blocks of dim 2:
// [[10,11],[20,21],[30,31]]; select indices [0,2] for one query → the gathered
// rows are block0 and block2 = [[10,11],[30,31]], shape [B,H,L,n,blockDim].
func TestGatherSelectedBlocks_Good(t *testing.T) {
	blocks := FromValues([]float32{10, 11, 20, 21, 30, 31}, 1, 1, 3, 2)
	indices := FromValues([]int32{0, 2}, 1, 1, 1, 2)
	sel := GatherSelectedBlocks(blocks, indices)
	Materialize(sel)

	dims := sel.Dims()
	wantShape := []int{1, 1, 1, 2, 2}
	if len(dims) != len(wantShape) {
		t.Fatalf("rank = %d, want %d (%v)", len(dims), len(wantShape), dims)
	}
	for i := range wantShape {
		if dims[i] != wantShape[i] {
			t.Fatalf("shape = %v, want %v", dims, wantShape)
		}
	}
	floatSliceApprox(t, sel.Floats(), []float32{10, 11, 30, 31})
}

// TestGatherSelectedBlocks_RoundTrip_Good is the GATE: gathering ALL blocks in
// natural order reproduces the input exactly (the gather is the identity when
// indices = [0,1,…,nBlocks-1]). Proves the gather op round-trips.
func TestGatherSelectedBlocks_RoundTrip_Good(t *testing.T) {
	blocks := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 1, 4, 2)
	// Two query positions, both selecting all 4 blocks in order.
	indices := FromValues([]int32{
		0, 1, 2, 3,
		0, 1, 2, 3,
	}, 1, 1, 2, 4)
	sel := GatherSelectedBlocks(blocks, indices)
	Materialize(sel)

	// Each query's gather = the full block tensor flattened.
	want := []float32{
		1, 2, 3, 4, 5, 6, 7, 8, // query 0
		1, 2, 3, 4, 5, 6, 7, 8, // query 1
	}
	floatSliceApprox(t, sel.Floats(), want)
}
