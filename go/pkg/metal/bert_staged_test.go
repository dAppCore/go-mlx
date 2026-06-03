// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestBERTPoolCLS_Good(t *testing.T) {
	coverageTokens := "BERT PoolCLS"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	hidden := FromValues([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 2, 2, 3)
	defer Free(hidden)

	pooled, ok := bertPoolCLS(hidden)
	if !ok {
		t.Fatal("bertPoolCLS ok = false, want true")
	}
	defer Free(pooled)
	Materialize(pooled)

	if gotShape := pooled.Shape(); len(gotShape) != 2 || gotShape[0] != 2 || gotShape[1] != 3 {
		t.Fatalf("shape = %v, want [2 3]", gotShape)
	}
	assertFloat32SliceClose(t, pooled.Floats(), []float32{1, 2, 3, 7, 8, 9}, 1e-5)
}

func TestBERTPoolMean_Masked_Good(t *testing.T) {
	coverageTokens := "BERT PoolMean Masked"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	hidden := FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		10, 20,
		30, 40,
		50, 60,
	}, 2, 3, 2)
	mask := FromValues([]int32{
		1, 1, 0,
		1, 0, 0,
	}, 2, 3)
	defer Free(hidden, mask)

	pooled, ok := bertPoolMean(hidden, mask)
	if !ok {
		t.Fatal("bertPoolMean ok = false, want true")
	}
	defer Free(pooled)
	Materialize(pooled)

	if gotShape := pooled.Shape(); len(gotShape) != 2 || gotShape[0] != 2 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", gotShape)
	}
	assertFloat32SliceClose(t, pooled.Floats(), []float32{2, 3, 10, 20}, 1e-5)
}

func TestBERTRerankHead_Score_Good(t *testing.T) {
	coverageTokens := "BERT RerankHead Score"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	hidden := FromValues([]float32{
		2, 3,
		4, 5,
	}, 1, 2, 2)
	weight := FromValues([]float32{
		1, 2,
		-1, 1,
	}, 2, 2)
	bias := FromValues([]float32{0.5, -0.5}, 2)
	head := bertRerankHead{
		Classifier: NewLinear(weight, bias),
		PoolMode:   bertPoolingCLS,
	}
	defer Free(hidden, weight, bias)

	logits, ok := head.Score(hidden, nil)
	if !ok {
		t.Fatal("Score ok = false, want true")
	}
	defer Free(logits)
	Materialize(logits)

	if gotShape := logits.Shape(); len(gotShape) != 2 || gotShape[0] != 1 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [1 2]", gotShape)
	}
	assertFloat32SliceClose(t, logits.Floats(), []float32{8.5, 0.5}, 1e-5)
}

func TestBERTPoolMean_Bad(t *testing.T) {
	coverageTokens := "BERT PoolMean Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	hidden := FromValues([]float32{1, 2, 3, 4}, 1, 2, 2)
	mask := FromValues([]int32{1, 1, 1}, 1, 3)
	defer Free(hidden, mask)

	if pooled, ok := bertPoolMean(hidden, mask); ok || pooled != nil {
		Free(pooled)
		t.Fatalf("bertPoolMean ok = %v pooled=%v, want false nil for wrong mask shape", ok, pooled)
	}
}
