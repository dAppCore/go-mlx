// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
)

// ffnMemoryCoverageBank builds a minimal valid FFN memory bank for the
// persistence-fault tests.
func ffnMemoryCoverageBank(t *testing.T) *FFNMemoryBank {
	t.Helper()
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:      2,
		Layers:          1,
		MemoryLevels:    []string{"1"},
		FFNMemoryTokens: []int{1},
		NumClusters:     []int{2},
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	return bank
}

// TestFfnMemoryFile_SaveFFNMemoryBank_WriteFault_Ugly drives the directory-create
// and write faults inside SaveFFNMemoryBank that the round-trip happy path never
// reaches: the target path's parent is an existing regular file (MkdirAll fails),
// and the target path is an existing directory (WriteFile fails). Both occur
// after the bank has already validated.
func TestFfnMemoryFile_SaveFFNMemoryBank_WriteFault_Ugly(t *testing.T) {
	bank := ffnMemoryCoverageBank(t)

	dir := t.TempDir()
	fileParent := core.PathJoin(dir, "afile")
	writeFile(t, fileParent, "x")
	if err := SaveFFNMemoryBank(core.PathJoin(fileParent, "ffn.json"), bank); err == nil {
		t.Fatal("SaveFFNMemoryBank(parent is a file) error = nil, want mkdir failure")
	}

	if err := SaveFFNMemoryBank(dir, bank); err == nil {
		t.Fatal("SaveFFNMemoryBank(path is a directory) error = nil, want write failure")
	}
}
