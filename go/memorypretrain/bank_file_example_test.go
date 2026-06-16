// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/mlx/memorypretrain"
)

func exampleBank() (*memorypretrain.Bank, error) {
	return memorypretrain.BuildBank([]memorypretrain.Block{
		{ID: "go", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "poem", Text: "winter proof poem", Embedding: []float32{0, 1}},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
}

// ExampleBank_Save writes a bank to a temporary file through the (*Bank).Save
// method and reloads it through the versioned JSON envelope.
func ExampleBank_Save() {
	bank, err := exampleBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-bank-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "bank.json")
	if err := bank.Save(path); err != nil {
		fmt.Println("error:", err)
		return
	}
	loaded, err := memorypretrain.LoadBank(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(loaded.Dimension, len(loaded.Blocks))
	// Output: 2 2
}

// ExampleSaveBank persists a bank with the package-level SaveBank function.
func ExampleSaveBank() {
	bank, err := exampleBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-bank-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "bank.json")
	if err := memorypretrain.SaveBank(path, bank); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("saved")
	// Output: saved
}

// ExampleLoadBank reloads a saved bank and routes a query through it.
func ExampleLoadBank() {
	bank, err := exampleBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-bank-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "bank.json")
	if err := memorypretrain.SaveBank(path, bank); err != nil {
		fmt.Println("error:", err)
		return
	}
	loaded, err := memorypretrain.LoadBank(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	got, err := loaded.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(got[0].BlockID)
	// Output: go
}
