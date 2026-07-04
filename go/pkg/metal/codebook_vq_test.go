// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestCodebookVQ_MatVecMatchesCPUReference_Good(t *testing.T) {
	requireMetalRuntime(t)

	input := FromValues([]float32{3, 4, 5, 6}, 1, 4)
	codes := FromValues([]uint32{0, 1, 2, 1}, 4)
	codebook := FromValues([]float32{
		1, 0,
		0, 1,
		2, -1,
	}, 3, 2)
	bias := FromValues([]float32{0.5, -1}, 2)

	gotArray, err := CodebookVQMatVec(input, codes, codebook, bias, []int32{2, 4}, 2)
	if err != nil {
		t.Fatalf("CodebookVQMatVec() error = %v", err)
	}
	Materialize(gotArray)

	assertFloat32SliceClose(t, gotArray.Floats(), []float32{9.5, 7}, 1e-5)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != 1 || shape[1] != 2 {
		t.Fatalf("shape = %+v, want [1 2]", shape)
	}
}

func TestCodebookVQ_MatVecRejectsBadMetadata_Bad(t *testing.T) {
	requireMetalRuntime(t)

	_, err := CodebookVQMatVec(
		FromValues([]float32{1, 2, 3}, 1, 3),
		FromValues([]uint32{0, 1, 2, 1}, 4),
		FromValues([]float32{1, 0, 0, 1}, 2, 2),
		nil,
		[]int32{2, 4},
		2,
	)
	if err == nil || !core.Contains(err.Error(), "input") {
		t.Fatalf("error = %v, want input shape diagnostic", err)
	}
}
