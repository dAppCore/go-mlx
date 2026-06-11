// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func Example_chainSample() {
	logits := FromValues([]float32{-100, 1, 100, -100}, 1, 4)
	token := chain{steps: []Sampler{TopKSampler(1)}}.Sample(logits)
	defer Free(logits, token)
	Materialize(token)

	core.Println(token.Int())
	// Output: 2
}

func Example_greedySample() {
	logits := FromValues([]float32{-10, 1, 7, 3}, 1, 4)
	token := Greedy{}.Sample(logits)
	defer Free(logits, token)
	Materialize(token)

	core.Println(token.Int())
	// Output: 2
}

func ExampleTemperature_Sample() {
	logits := FromValues([]float32{1, 2, 3}, 1, 3)
	scaled := Temperature(0.5).Sample(logits)
	defer Free(logits, scaled)
	Materialize(scaled)

	core.Println(scaled.Floats())
	// Output: [2 4 6]
}

func ExampleTopKSampler_Sample() {
	logits := FromValues([]float32{1, 10, 3, 2}, 1, 4)
	filtered := TopKSampler(2).Sample(logits)
	defer Free(logits, filtered)
	Materialize(filtered)
	got := filtered.Floats()

	core.Println(got[1], got[2], got[0] < got[2], got[3] < got[2])
	// Output: 10 3 true true
}

func ExampleTopP_Sample() {
	logits := FromValues([]float32{10, 1, 0}, 1, 3)
	filtered := TopP(0.8).Sample(logits)
	defer Free(logits, filtered)
	Materialize(filtered)
	got := filtered.Floats()

	core.Println(got[0], got[1] < got[0], got[2] < got[0])
	// Output: 10 true true
}

func ExampleMinPSampler_Sample() {
	logits := FromValues([]float32{10, 9, 0}, 1, 3)
	filtered := MinPSampler(0.1).Sample(logits)
	defer Free(logits, filtered)
	Materialize(filtered)
	got := filtered.Floats()

	core.Println(got[0], got[1], got[2] < got[1])
	// Output: 10 9 true
}
