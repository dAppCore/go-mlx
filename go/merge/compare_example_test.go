// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"
)

// ExampleComparePacks compares a base model pack against a fine-tuned pack
// whose single tensor differs by a constant, and prints how many tensors
// changed. ComparePacks performs weight-space archaeology — it indexes and
// streams the safetensors shards without loading either model through the
// runtime. The example reuses the ExamplePacks pack-writing helpers.
func ExampleComparePacks() {
	base := exampleWritePack("qwen3", 1, 2, 3, 4)
	tuned := exampleWritePack("qwen3", 1, 2, 5, 8)
	defer core.RemoveAll(base)
	defer core.RemoveAll(tuned)

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:      examplePack(base),
		FineTuned: examplePack(tuned),
	})
	if err != nil {
		core.Println("error:", err)
		return
	}

	core.Println("changed tensors:", report.ChangedTensors, "elements:", report.ElementsCompared)
	// Output: changed tensors: 1 elements: 4
}
