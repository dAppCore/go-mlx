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

// ExampleFFNMemoryBank_Save persists an FFN memory bank through the
// (*FFNMemoryBank).Save method and reloads it.
func ExampleFFNMemoryBank_Save() {
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
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-ffn-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "ffn.json")
	if err := bank.Save(path); err != nil {
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

// ExampleLoadFFNMemoryBank reloads a saved FFN memory bank and applies the
// generic-memory fallback to a one-token FFN output.
func ExampleLoadFFNMemoryBank() {
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
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-ffn-*")
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
	_, _, stats, err := loaded.AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(stats.Applied)
	// Output: true
}
