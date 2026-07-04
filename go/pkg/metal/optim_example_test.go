// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleDefaultAdamWConfig() {
	cfg := DefaultAdamWConfig()
	core.Println(core.Sprintf("lr=%.0e beta1=%.1f beta2=%.3f wd=%.2f packed=%v",
		cfg.LearningRate,
		cfg.Beta1,
		cfg.Beta2,
		cfg.WeightDecay,
		cfg.PackedState,
	))
	// Output: lr=1e-05 beta1=0.9 beta2=0.999 wd=0.01 packed=true
}

func ExampleNewAdamW() {
	optimizer := NewAdamW(&AdamWConfig{
		LearningRate:   3e-4,
		Beta1:          0.85,
		WeightDecay:    0,
		WeightDecaySet: true,
		PackedState:    false,
		PackedStateSet: true,
	})

	core.Println(core.Sprintf("lr=%.0e beta1=%.2f weight_decay=%.0f packed=%v",
		optimizer.LR,
		optimizer.Beta1,
		optimizer.WeightDecay,
		optimizer.PackedState,
	))
	// Output: lr=3e-04 beta1=0.85 weight_decay=0 packed=false
}

func ExampleAdamW_Step() {
	parameter := FromValues([]float32{1}, 1)
	gradient := FromValues([]float32{0.5}, 1)
	optimizer := NewAdamW(&AdamWConfig{
		LearningRate:   0.1,
		WeightDecay:    0,
		WeightDecaySet: true,
		PackedState:    false,
		PackedStateSet: true,
	})
	updated := optimizer.Step([]*Array{parameter}, []*Array{gradient})
	defer Free(parameter, gradient)
	defer Free(updated...)
	defer optimizer.Reset()

	Materialize(updated[0])
	core.Println(core.Sprintf("value=%.3f step=%d moments=%d", updated[0].Floats()[0], optimizer.step, len(optimizer.m)))
	// Output: value=0.900 step=1 moments=1
}

func ExampleAdamW_Reset() {
	parameter := FromValues([]float32{1}, 1)
	gradient := FromValues([]float32{0.5}, 1)
	optimizer := NewAdamW(&AdamWConfig{PackedState: false, PackedStateSet: true})
	updated := optimizer.Step([]*Array{parameter}, []*Array{gradient})
	defer Free(parameter, gradient)
	defer Free(updated...)

	core.Println(core.Sprintf("before step=%d moments=%d", optimizer.step, len(optimizer.m)))
	optimizer.Reset()
	core.Println(core.Sprintf("after step=%d moments=%d", optimizer.step, len(optimizer.m)))
	// Output:
	// before step=1 moments=1
	// after step=0 moments=0
}
