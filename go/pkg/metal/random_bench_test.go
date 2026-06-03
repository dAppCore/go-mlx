// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkRandomCategorical_Vocab32k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)

	b.ReportAllocs()
	for b.Loop() {
		token := RandomCategorical(logits)
		if err := Eval(token); err != nil {
			Free(token)
			b.Fatalf("Eval(RandomCategorical): %v", err)
		}
		Free(token)
	}
}

func BenchmarkRandomCategorical_Vocab262k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)

	b.ReportAllocs()
	for b.Loop() {
		token := RandomCategorical(logits)
		if err := Eval(token); err != nil {
			Free(token)
			b.Fatalf("Eval(RandomCategorical): %v", err)
		}
		Free(token)
	}
}
