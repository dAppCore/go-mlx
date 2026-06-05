// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleRMSNorm() {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)
	out := RMSNorm(x, weight, 1e-5)
	defer Free(x, weight, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.3f %.3f %.3f %.3f", got[0], got[1], got[2], got[3]))
	// Output: 0.365 0.730 1.095 1.461
}

func ExampleRMSNormNoScale() {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	out := RMSNormNoScale(x, 1e-5)
	defer Free(x, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.3f %.3f %.3f %.3f", got[0], got[1], got[2], got[3]))
	// Output: 0.365 0.730 1.095 1.461
}

func ExampleLayerNorm() {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)
	bias := FromValues([]float32{0, 0, 0, 0}, 4)
	out := LayerNorm(x, weight, bias, 1e-5)
	defer Free(x, weight, bias, out)
	Materialize(out)

	got := out.Floats()
	core.Println(core.Sprintf("%.3f %.3f %.3f %.3f", got[0], got[1], got[2], got[3]))
	// Output: -1.342 -0.447 0.447 1.342
}

func ExampleRoPE() {
	x := FromValues([]float32{1, 0, 1, 0}, 1, 1, 1, 4)
	out := RoPE(x, 4, false, 10000, 1, 0)
	defer Free(x, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 1 1 4] [1 0 1 0]
}

func ExampleRoPEWithFreqs() {
	x := FromValues([]float32{1, 0, 1, 0}, 1, 1, 1, 4)
	freqs := FromValues([]float32{1, 0.01}, 2)
	out := RoPEWithFreqs(x, 4, false, 0, 1, 0, freqs)
	defer Free(x, freqs, out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 1 1 4] [1 0 1 0]
}

func ExampleScaledDotProductAttention() {
	q := FromValues([]float32{1, 0, 0, 1, 1, 1}, 1, 1, 3, 2)
	k := FromValues([]float32{1, 0, 0, 1, 1, 1}, 1, 1, 3, 2)
	v := FromValues([]float32{1, 0, 0, 1, 0.5, 0.5}, 1, 1, 3, 2)
	out := ScaledDotProductAttention(q, k, v, 0.70710677, true)
	defer Free(q, k, v, out)
	Materialize(out)

	got := out.Floats()
	core.Println(out.Shape(), core.Sprintf("%.2f %.2f", got[0], got[1]))
	// Output: [1 1 3 2] 1.00 0.00
}

func ExampleScaledDotProductAttentionWithMask() {
	q := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := FromValues([]float32{10, 0, 0, 10}, 1, 1, 2, 2)
	mask := FromValues([]float32{0, 0, -1e9, 0}, 1, 1, 2, 2)
	out := ScaledDotProductAttentionWithMask(q, k, v, mask, 0.70710677)
	defer Free(q, k, v, mask, out)
	Materialize(out)

	got := out.Floats()
	core.Println(out.Shape(), core.Sprintf("%.2f %.2f", got[2], got[3]))
	// Output: [1 1 2 2] 0.00 10.00
}
