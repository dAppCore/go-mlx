// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestBankFile_SaveBank_WriteFault_Ugly drives the two persistence faults inside
// SaveBank that the happy path never reaches: a directory-creation failure (the
// target path's parent is an existing regular file) and a write failure (the
// target path is an existing directory). Both surface an error after the bank
// has already passed validation.
func TestBankFile_SaveBank_WriteFault_Ugly(t *testing.T) {
	bank := sampleBank(t)

	// Parent is a regular file, so MkdirAll for the bank directory fails.
	dir := t.TempDir()
	fileParent := core.PathJoin(dir, "afile")
	writeFile(t, fileParent, "x")
	if err := SaveBank(core.PathJoin(fileParent, "bank.json"), bank); err == nil {
		t.Fatal("SaveBank(parent is a file) error = nil, want mkdir failure")
	}

	// The target path itself is an existing directory, so WriteFile fails after
	// the directory guard is satisfied.
	if err := SaveBank(dir, bank); err == nil {
		t.Fatal("SaveBank(path is a directory) error = nil, want write failure")
	}
}

// TestBankFile_SaveBank_MarshalFault_Ugly reaches the JSON marshal-failure return
// in SaveBank. Validation only finiteness-checks block embeddings, not node
// centroids, so a bank with finite blocks but a non-finite value in a node
// centroid passes validateBank and then fails encoding (JSON cannot represent
// NaN), exercising the marshal error path before any directory or file work.
func TestBankFile_SaveBank_MarshalFault_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{ID: "a", Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, float32(math.NaN())}, BlockIDs: []int{0}}},
		Root:      0,
	}
	path := core.PathJoin(t.TempDir(), "bank.json")
	if err := SaveBank(path, bank); err == nil {
		t.Fatal("SaveBank(non-finite centroid) error = nil, want marshal failure")
	}
	if core.ReadFile(path).OK {
		t.Fatal("SaveBank() wrote a file despite a marshal failure")
	}
}

// TestBankFile_validateBank_BlockValidationError_Ugly reaches the block-validation
// error return in validateBank: the bank declares a positive dimension but its
// only block carries a non-finite embedding value, so validateBlocks rejects it
// before the dimension cross-check.
func TestBankFile_validateBank_BlockValidationError_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{ID: "a", Embedding: []float32{1, float32(math.NaN())}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: []int{0}}},
		Root:      0,
	}
	if err := validateBank(bank); err == nil {
		t.Fatal("validateBank(non-finite block embedding) error = nil, want block validation failure")
	}
}
