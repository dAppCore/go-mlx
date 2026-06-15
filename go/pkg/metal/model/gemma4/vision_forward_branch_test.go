// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Branch-level coverage for the two Gemma4VisionPatchEmbedder seams a loaded
// SigLIP checkpoint never reaches: the rank-3 (separate x/y) position-embedding
// table arm of positionEmbeddings, and the no-PatchConvWeight guard in
// prepareRawNHWC. The package-local test constructs the embedder struct directly
// in the exact shape the loader never produces — a checkpoint ships either a
// rank-2 flat table (the covered fallback) or a conv kernel, never the rank-3
// dual table standing alone, so the production path is only observable here.

// synthPositionTableRank3 builds a [2, slots, hidden] position table: row 0 is
// the x table, row 1 the y table — the gemma2-style separate-axis layout the
// rank-3 arm of positionEmbeddings reads. slots must cover the largest grid
// coordinate (max(gridH, gridW)); hidden is the embedding width that flows out.
func synthPositionTableRank3(slots, hidden int) *metal.Array {
	return seqArray(0.02, 2, slots, hidden)
}

// TestVisionForward_positionEmbeddings_Rank3_Good drives the separate x/y table
// arm: a [2, slots, hidden] table is sliced into an x-table and a y-table, each
// gathered by its coordinate ids, then summed. For a 2x2 grid the result is one
// hidden row per patch position — [batch, gridH*gridW, hidden]. This is the arm
// the flat rank-2 fallback (the loader's shape) can never reach.
func TestVisionForward_positionEmbeddings_Rank3_Good(t *testing.T) {
	requireMetalRuntime(t)

	const hidden = gemma4VisionHidden
	const gridH, gridW = int32(2), int32(2)
	embedder := &Gemma4VisionPatchEmbedder{
		PositionEmbeddingTable: synthPositionTableRank3(int(gridW), hidden),
		HiddenSize:             hidden,
	}
	defer metal.Free(embedder.PositionEmbeddingTable)

	pos := embedder.positionEmbeddings(1, gridH, gridW)
	if pos == nil || !pos.Valid() {
		t.Fatal("positionEmbeddings(rank-3 table) returned nil, want the summed x+y embeddings")
	}
	defer metal.Free(pos)
	if err := metal.Eval(pos); err != nil {
		t.Fatalf("Eval pos: %v", err)
	}

	// The arm gathers per (batch, position): [batch, gridH*gridW, hidden].
	shape := pos.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != gridH*gridW || shape[2] != hidden {
		t.Fatalf("rank-3 position embeddings shape = %v, want [1 %d %d]", shape, gridH*gridW, hidden)
	}

	// x table (row 0) and y table (row 1) are distinct, so a position whose
	// (x,y) differ must not collapse to the x==y diagonal — the two halves are
	// genuinely both consulted, not one table read twice.
	got := pos.Floats()
	allZero := true
	for _, v := range got {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("rank-3 position embeddings are all zero — the x/y gather never populated the table")
	}
}

// TestVisionForward_positionEmbeddings_RankTooLow_Ugly pins the degenerate guard:
// a rank-1 table has no position axis to gather, so the arm returns nil rather
// than indexing past the end (a malformed checkpoint must fail soft here, not
// crash the patch embed).
func TestVisionForward_positionEmbeddings_RankTooLow_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	embedder := &Gemma4VisionPatchEmbedder{
		PositionEmbeddingTable: seqArray(0.02, gemma4VisionHidden), // rank-1
		HiddenSize:             gemma4VisionHidden,
	}
	defer metal.Free(embedder.PositionEmbeddingTable)

	if pos := embedder.positionEmbeddings(1, 2, 2); pos != nil {
		metal.Free(pos)
		t.Fatal("positionEmbeddings(rank-1 table) returned non-nil, want nil for the malformed-table guard")
	}
}

// TestVisionForward_prepareRawNHWC_NoConvWeight_Bad drives the missing-kernel
// guard: prepareRawNHWC is reached with a valid rank-4 NHWC tensor but a nil
// PatchConvWeight (a patch embedder that only ships a flat input projection, no
// conv). The guard must free the owned input and return (nil, false, 0, 0) — the
// conv branch must never run without a kernel.
func TestVisionForward_prepareRawNHWC_NoConvWeight_Bad(t *testing.T) {
	requireMetalRuntime(t)

	embedder := &Gemma4VisionPatchEmbedder{
		PatchSize:   gemma4VisionPatch,
		NumChannels: 3,
		// PatchConvWeight deliberately nil.
	}

	nhwc := metal.Zeros([]int32{1, gemma4VisionPatch, gemma4VisionPatch, 3}, metal.DTypeFloat32)
	patches, projected, gridH, gridW := embedder.prepareRawNHWC(nhwc, true)
	if patches != nil {
		metal.Free(patches)
		t.Fatal("prepareRawNHWC(no conv weight) returned non-nil patches, want nil")
	}
	if projected || gridH != 0 || gridW != 0 {
		t.Fatalf("no-conv-weight prepareRawNHWC = (projected=%v, grid %dx%d), want (false, 0x0)", projected, gridH, gridW)
	}
}

// TestVisionForward_prepareRawNHWC_WrongRank_Bad drives the same guard via a
// non-rank-4 tensor: even with a conv kernel present, a tensor that is not NHWC
// rank-4 must hit the guard and free the owned input rather than convolving a
// wrong-rank tensor.
func TestVisionForward_prepareRawNHWC_WrongRank_Bad(t *testing.T) {
	requireMetalRuntime(t)

	embedder := &Gemma4VisionPatchEmbedder{
		PatchSize:       gemma4VisionPatch,
		NumChannels:     3,
		PatchConvWeight: seqArray(0.01, gemma4VisionHidden, gemma4VisionPatch, gemma4VisionPatch, 3),
	}
	defer metal.Free(embedder.PatchConvWeight)

	rank3 := metal.Zeros([]int32{gemma4VisionPatch, gemma4VisionPatch, 3}, metal.DTypeFloat32)
	patches, projected, gridH, gridW := embedder.prepareRawNHWC(rank3, true)
	if patches != nil {
		metal.Free(patches)
		t.Fatal("prepareRawNHWC(rank-3 input) returned non-nil patches, want nil")
	}
	if projected || gridH != 0 || gridW != 0 {
		t.Fatalf("wrong-rank prepareRawNHWC = (projected=%v, grid %dx%d), want (false, 0x0)", projected, gridH, gridW)
	}
}
