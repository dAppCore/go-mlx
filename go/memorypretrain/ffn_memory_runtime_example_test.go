// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/memorypretrain"
)

func exampleRuntimeBank() (*memorypretrain.FFNMemoryBank, error) {
	return memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
}

// ExampleNewFFNMemoryRuntime builds a memory-only runtime facade. A nil router
// selects the generic-memory fallback and needs no embedder.
func ExampleNewFFNMemoryRuntime() {
	bank, err := exampleRuntimeBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	runtime, err := memorypretrain.NewFFNMemoryRuntime(bank, nil, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(runtime.Router == nil, runtime.Embedder == nil)
	// Output: true true
}

// ExampleFFNMemoryRuntime_AddTextToFFNOutput applies the generic-memory fallback
// to a one-token FFN output. With no router configured the query text is ignored
// and the final cluster slot is selected; a fresh bank's W3 is zero so the output
// is unchanged.
func ExampleFFNMemoryRuntime_AddTextToFFNOutput() {
	bank, err := exampleRuntimeBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	runtime, err := memorypretrain.NewFFNMemoryRuntime(bank, nil, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	out, ids, stats, err := runtime.AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "", 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(out, ids, stats.Applied)
	// Output: [1 2] [2] true
}
