// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/mlx/memorypretrain"
)

// ExampleSaveFFNMemoryBank persists an FFN memory bank and reloads it through
// the versioned JSON envelope.
func ExampleSaveFFNMemoryBank() {
	bank, err := memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-example-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "ffn.json")
	if err := memorypretrain.SaveFFNMemoryBank(path, bank); err != nil {
		fmt.Println("error:", err)
		return
	}
	loaded, err := memorypretrain.LoadFFNMemoryBank(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(loaded.HiddenSize, len(loaded.Layers))
	// Output: 2 1
}
