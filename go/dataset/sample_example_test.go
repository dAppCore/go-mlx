// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"fmt"

	"dappco.re/go/mlx/dataset"
)

// ExampleNewSliceDataset shows the replayable in-memory dataset: iterate to
// exhaustion, Reset, and iterate again.
func ExampleNewSliceDataset() {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "2+2", Response: "4"},
		{Text: "raw corpus row"},
	})

	count := 0
	for {
		s, ok, err := ds.Next()
		if err != nil {
			panic(err)
		}
		if !ok {
			break
		}
		count++
		_ = s
	}
	fmt.Println("pass1:", count)

	_ = ds.Reset()
	s, _, _ := ds.Next()
	fmt.Println("after reset:", s.Prompt, s.Response)
	// Output:
	// pass1: 2
	// after reset: 2+2 4
}

// ExampleFunc adapts a generator function into a Dataset.
func ExampleFunc() {
	rows := []dataset.Sample{{Text: "a"}, {Text: "b"}}
	i := 0
	ds := dataset.Func(func() (dataset.Sample, bool, error) {
		if i >= len(rows) {
			return dataset.Sample{}, false, nil
		}
		s := rows[i]
		i++
		return s, true, nil
	})

	for {
		s, ok, _ := ds.Next()
		if !ok {
			break
		}
		fmt.Println(s.Text)
	}
	// Output:
	// a
	// b
}

// ExampleCloneSample shows the defensive deep copy: mutating the clone's
// Meta does not touch the original.
func ExampleCloneSample() {
	original := dataset.Sample{Text: "doc", Meta: map[string]string{"split": "train"}}
	clone := dataset.CloneSample(original)
	clone.Meta["split"] = "test"

	fmt.Println("original:", original.Meta["split"])
	fmt.Println("clone:", clone.Meta["split"])
	// Output:
	// original: train
	// clone: test
}
