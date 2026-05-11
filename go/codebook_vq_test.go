// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestCodebookVQ_DescriptorValidatesAndMatVec_Good(t *testing.T) {
	profile := CodebookQuantizationProfile{
		Format:       CodebookFormatVQ,
		CodebookSize: 3,
		CodeDim:      2,
		IndexBits:    16,
	}

	desc, err := NewCodebookTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{2, 4}, profile)
	if err != nil {
		t.Fatalf("NewCodebookTensorDescriptor() error = %v", err)
	}
	if desc.Elements != 8 || desc.CodeCount != 4 || desc.CodebookSize != 3 || desc.CodeDim != 2 {
		t.Fatalf("descriptor = %+v, want 8 elements, 4 codes, 3-entry codebook with 2D vectors", desc)
	}
	if desc.IndexBytes != 8 {
		t.Fatalf("IndexBytes = %d, want four 16-bit indices", desc.IndexBytes)
	}

	got, err := CodebookVQMatVec(desc, []float32{3, 4, 5, 6}, []uint32{0, 1, 2, 1}, []float32{
		1, 0,
		0, 1,
		2, -1,
	}, []float32{0.5, -1})
	if err != nil {
		t.Fatalf("CodebookVQMatVec() error = %v", err)
	}
	assertCloseSlice(t, got, []float32{9.5, 7}, 1e-5)
}

func TestCodebookVQ_DescriptorRejectsUnalignedShape_Bad(t *testing.T) {
	_, err := NewCodebookTensorDescriptor("bad.weight", []uint64{3, 3}, CodebookQuantizationProfile{
		Format:       CodebookFormatVQ,
		CodebookSize: 16,
		CodeDim:      4,
		IndexBits:    8,
	})
	if err == nil || !core.Contains(err.Error(), "divisible") {
		t.Fatalf("error = %v, want code-dim divisibility diagnostic", err)
	}
}

func TestCodebookVQ_MatVecRejectsOutOfRangeCode_Bad(t *testing.T) {
	desc, err := NewCodebookTensorDescriptor("ok.weight", []uint64{1, 2}, CodebookQuantizationProfile{
		Format:       CodebookFormatVQ,
		CodebookSize: 2,
		CodeDim:      1,
		IndexBits:    8,
	})
	if err != nil {
		t.Fatalf("NewCodebookTensorDescriptor() error = %v", err)
	}

	_, err = CodebookVQMatVec(desc, []float32{1, 2}, []uint32{0, 4}, []float32{1, 2}, nil)
	if err == nil || !core.Contains(err.Error(), "code id") {
		t.Fatalf("error = %v, want out-of-range code diagnostic", err)
	}
}

func TestCodebookVQ_ParseConfig_Good(t *testing.T) {
	profile, err := ParseCodebookQuantizationProfile([]byte(`{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 4,
		"code_dim": 2,
		"index_bits": 8,
		"tensors": [
			{
				"name": "model.layers.0.mlp.down_proj.weight",
				"shape": [2, 4],
				"codes": "model.layers.0.mlp.down_proj.weight.codes",
				"codebook": "model.layers.0.mlp.down_proj.weight.codebook"
			}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseCodebookQuantizationProfile() error = %v", err)
	}
	if profile.Type != CodebookQuantizationType || profile.Format != CodebookFormatVQ || len(profile.Tensors) != 1 {
		t.Fatalf("profile = %+v, want one VQ tensor", profile)
	}
	if tensor := profile.Tensors[0]; tensor.CodeCount != 4 || tensor.CodesName == "" || tensor.CodebookName == "" {
		t.Fatalf("tensor = %+v, want resolved sidecar names and code count", tensor)
	}
}

func assertCloseSlice(t *testing.T, got, want []float32, epsilon float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if float64(diff) > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
