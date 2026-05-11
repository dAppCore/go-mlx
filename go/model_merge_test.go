// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

func TestMergeModelPacks_LinearSafetensors_Good(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{0, 2, 4, 6}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{10, 12, 14, 16}},
	})
	output := core.PathJoin(t.TempDir(), "merged-linear")

	result, err := MergeModelPacks(context.Background(), ModelMergeOptions{
		OutputPath: output,
		Method:     ModelMergeLinear,
		Sources: []ModelMergeSource{
			{Path: left, Weight: 0.25},
			{Path: right, Weight: 0.75},
		},
	})
	if err != nil {
		t.Fatalf("MergeModelPacks() error = %v", err)
	}
	if result.Method != ModelMergeLinear || result.TensorCount != 1 || result.MergedTensors != 1 {
		t.Fatalf("result = %+v", result)
	}
	if result.WeightPath != core.PathJoin(output, "model.safetensors") {
		t.Fatalf("WeightPath = %q", result.WeightPath)
	}
	if !result.Pack.Valid() || result.Pack.Format != mp.ModelPackFormatSafetensors {
		t.Fatalf("pack = %+v", result.Pack)
	}

	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	assertMergedTensorValues(t, tensors, []float32{7.5, 9.5, 11.5, 13.5})
	if stat := core.Stat(core.PathJoin(output, ModelMergeProvenanceFile)); !stat.OK {
		t.Fatalf("provenance was not written: %v", stat.Value)
	}
}

func TestMergeModelPacks_SLERPSafetensors_Good(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{1, 0}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{0, 1}},
	})

	result, err := MergeModelPacks(context.Background(), ModelMergeOptions{
		OutputPath: core.PathJoin(t.TempDir(), "merged-slerp"),
		Method:     ModelMergeSLERP,
		T:          0.5,
		Sources: []ModelMergeSource{
			{Path: left},
			{Path: right},
		},
	})
	if err != nil {
		t.Fatalf("MergeModelPacks() error = %v", err)
	}

	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertMergedTensorValues(t, tensors, []float32{want, want})
}

func TestMergeModelPacks_AllowTensorMismatchCopiesBaseTensor_Good(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{5, 7}},
	})

	result, err := MergeModelPacks(context.Background(), ModelMergeOptions{
		OutputPath:          core.PathJoin(t.TempDir(), "merged-mismatch"),
		Method:              ModelMergeLinear,
		AllowTensorMismatch: true,
		Sources: []ModelMergeSource{
			{Path: left},
			{Path: right},
		},
		Labels: map[string]string{"suite": "mismatch"},
	})
	if err != nil {
		t.Fatalf("MergeModelPacks(allow mismatch) error = %v", err)
	}
	if result.MergedTensors != 1 || result.CopiedTensors != 1 || len(result.SkippedTensors) != 1 {
		t.Fatalf("result = %+v, want one merged and one copied tensor", result)
	}
	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	if len(tensors) != 2 {
		t.Fatalf("tensor count = %d, want 2", len(tensors))
	}
	for _, tensor := range tensors {
		switch tensor.Name {
		case "model.embed_tokens.weight":
			assertFloat32Values(t, tensor.Data, []float32{3, 4})
		case "model.norm.weight":
			assertFloat32Values(t, tensor.Data, []float32{3, 4.5})
		default:
			t.Fatalf("unexpected tensor %q", tensor.Name)
		}
	}
}

func TestModelMerge_WriteLinearMergedTensorChunks_Good(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.layers.0.mlp.down_proj.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{0, 2, 4, 6, 8}},
	})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{10, 12, 14, 16, 18}},
	})
	leftIndex, err := indexSafetensorFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := indexSafetensorFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}
	outPath := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)

	err = writeLinearMergedTensorChunks(context.Background(), file, []safetensorTensorRef{
		leftIndex.Tensors[name],
		rightIndex.Tensors[name],
	}, []float64{0.25, 0.75}, 2)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("writeLinearMergedTensorChunks() error = %v", err)
	}

	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := decodeSafetensorFloatData("F32", read.Value.([]byte), 5)
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	assertFloat32Values(t, values, []float32{7.5, 9.5, 11.5, 13.5, 15.5})
}

func TestModelMerge_WriteSLERPMergedTensorChunks_Good(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.embed_tokens.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{
		{Name: name, Shape: []int{2}, Data: []float32{1, 0}},
	})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{
		{Name: name, Shape: []int{2}, Data: []float32{0, 1}},
	})
	leftIndex, err := indexSafetensorFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := indexSafetensorFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}
	outPath := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)

	err = writeSLERPMergedTensorChunks(context.Background(), file, []safetensorTensorRef{
		leftIndex.Tensors[name],
		rightIndex.Tensors[name],
	}, 0.5, 1)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("writeSLERPMergedTensorChunks() error = %v", err)
	}

	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := decodeSafetensorFloatData("F32", read.Value.([]byte), 2)
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertFloat32Values(t, values, []float32{want, want})
}

func TestModelMerge_SafetensorChunkHelpers_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "source.safetensors")
	name := "model.embed_tokens.weight"
	writeTestSafetensorsF32(t, path, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{0, 2, 4, 6, 8}},
	})
	index, err := indexSafetensorFiles([]string{path})
	if err != nil {
		t.Fatalf("index source: %v", err)
	}
	ref := index.Tensors[name]
	chunk, err := readSafetensorRefFloat32Chunk(ref, 1, 2)
	if err != nil {
		t.Fatalf("read chunk: %v", err)
	}
	assertFloat32Values(t, chunk, []float32{2, 4})

	outPath := core.PathJoin(t.TempDir(), "copy.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	err = writeSafetensorRefFloat32Chunks(context.Background(), file, ref, 2)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("write copy chunks: %v", err)
	}
	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := decodeSafetensorFloatData("F32", read.Value.([]byte), 5)
	if err != nil {
		t.Fatalf("decode copy: %v", err)
	}
	assertFloat32Values(t, values, []float32{0, 2, 4, 6, 8})
}

func TestModelMerge_ValueMergeHelpers_Good(t *testing.T) {
	linear, err := mergeTensorValues([][]float32{
		{0, 2, 4},
		{10, 12, 14},
	}, ModelMergeLinear, 0, []float64{0.25, 0.75})
	if err != nil {
		t.Fatalf("mergeTensorValues(linear) error = %v", err)
	}
	assertFloat32Values(t, linear, []float32{7.5, 9.5, 11.5})

	slerp, err := mergeTensorValues([][]float32{
		{1, 0},
		{0, 1},
	}, ModelMergeSLERP, 0.5, nil)
	if err != nil {
		t.Fatalf("mergeTensorValues(slerp) error = %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertFloat32Values(t, slerp, []float32{want, want})

	linearFallback, err := slerpMergeTensorValues([][]float32{{0, 0}, {2, 4}}, 0.25)
	if err != nil {
		t.Fatalf("slerpMergeTensorValues(zero norm) error = %v", err)
	}
	assertFloat32Values(t, linearFallback, []float32{0.5, 1})
	if got := clampFloat64(-2, -1, 1); got != -1 {
		t.Fatalf("clamp low = %f, want -1", got)
	}
	if got := clampFloat64(2, -1, 1); got != 1 {
		t.Fatalf("clamp high = %f, want 1", got)
	}
	if got := clampFloat64(0.5, -1, 1); got != 0.5 {
		t.Fatalf("clamp mid = %f, want 0.5", got)
	}
}

func TestModelMerge_ReadMergeTensorValues_Good(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.norm.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{1, 2}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{3, 4}}})
	leftIndex, err := indexSafetensorFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := indexSafetensorFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}

	values, complete, err := readMergeTensorValues([]safetensorIndex{leftIndex, rightIndex}, name)
	if err != nil {
		t.Fatalf("readMergeTensorValues() error = %v", err)
	}
	if !complete || len(values) != 2 {
		t.Fatalf("values len/complete = %d/%v, want 2/true", len(values), complete)
	}
	assertFloat32Values(t, values[0], []float32{1, 2})
	assertFloat32Values(t, values[1], []float32{3, 4})
}

func TestModelMerge_ChunkHelperErrors_Bad(t *testing.T) {
	if _, err := safetensorDTypeByteSize("F16"); err != nil {
		t.Fatalf("F16 byte size: %v", err)
	}
	if _, err := safetensorDTypeByteSize("BF16"); err != nil {
		t.Fatalf("BF16 byte size: %v", err)
	}
	if _, err := safetensorDTypeByteSize("F64"); err != nil {
		t.Fatalf("F64 byte size: %v", err)
	}
	if _, err := safetensorDTypeByteSize("I32"); err == nil {
		t.Fatal("expected unsupported dtype error")
	}
	if err := writeLinearMergedTensorChunks(context.Background(), nil, nil, nil, 2); err == nil {
		t.Fatal("expected no tensors error")
	}
	if err := writeLinearMergedTensorChunks(context.Background(), nil, []safetensorTensorRef{{Elements: 1}}, nil, 2); err == nil {
		t.Fatal("expected weight/source mismatch error")
	}
	if _, err := readSafetensorRefFloat32Chunk(safetensorTensorRef{DType: "F32", Elements: 1}, 1, 1); err == nil {
		t.Fatal("expected chunk bounds error")
	}
	if err := modelMergeResultError(core.Ok("ok")); err != nil {
		t.Fatalf("modelMergeResultError(ok) = %v", err)
	}
	if err := modelMergeResultError(core.Result{Value: "bad", OK: false}); err == nil {
		t.Fatal("expected non-error core result failure")
	}
}

func TestModelMerge_ValueMergeHelpers_Bad(t *testing.T) {
	if _, err := mergeTensorValues([][]float32{{1}}, "bad", 0, []float64{1}); err == nil {
		t.Fatal("mergeTensorValues(unsupported) error = nil")
	}
	if _, err := linearMergeTensorValues(nil, nil); err == nil {
		t.Fatal("linearMergeTensorValues(nil) error = nil")
	}
	if _, err := linearMergeTensorValues([][]float32{{1}, {1, 2}}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("linearMergeTensorValues(length mismatch) error = nil")
	}
	if _, err := slerpMergeTensorValues([][]float32{{1}}, 0.5); err == nil {
		t.Fatal("slerpMergeTensorValues(one tensor) error = nil")
	}
	if _, err := slerpMergeTensorValues([][]float32{{1}, {1, 2}}, 0.5); err == nil {
		t.Fatal("slerpMergeTensorValues(length mismatch) error = nil")
	}
	if _, err := normalizedMergeWeights([]ModelMergeSource{{Weight: math.NaN()}}); err == nil {
		t.Fatal("normalizedMergeWeights(NaN) error = nil")
	}
	if _, err := normalizedMergeWeights([]ModelMergeSource{{Weight: 1}, {Weight: -1}}); err == nil {
		t.Fatal("normalizedMergeWeights(zero sum) error = nil")
	}
}

func TestPrepareModelMerge_Bad_Validation(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}}})
	other := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{2}}})
	occupied := t.TempDir()
	writeModelPackFile(t, core.PathJoin(occupied, "model.safetensors"), "occupied")
	cases := []struct {
		name string
		opts ModelMergeOptions
	}{
		{name: "not enough sources", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []ModelMergeSource{{Path: source}}}},
		{name: "missing output", opts: ModelMergeOptions{Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "file output", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out.safetensors"), Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "unsupported method", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: "bad", Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "future method", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: ModelMergeTIES, Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "slerp source count", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: ModelMergeSLERP, Sources: []ModelMergeSource{{Path: source}, {Path: other}, {Path: other}}}},
		{name: "bad t", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), T: 2, Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "empty source", opts: ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []ModelMergeSource{{Path: source}, {}}}},
		{name: "same output", opts: ModelMergeOptions{OutputPath: source, Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
		{name: "occupied output", opts: ModelMergeOptions{OutputPath: occupied, Sources: []ModelMergeSource{{Path: source}, {Path: other}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := prepareModelMerge(context.Background(), tc.opts); err == nil {
				t.Fatal("prepareModelMerge() error = nil")
			}
		})
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := prepareModelMerge(cancelled, ModelMergeOptions{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []ModelMergeSource{{Path: source}, {Path: other}}}); err == nil {
		t.Fatal("prepareModelMerge(cancelled) error = nil")
	}
}

func TestMergeModelPacks_RejectsArchitectureMismatch_Bad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "gemma3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})

	_, err := MergeModelPacks(context.Background(), ModelMergeOptions{
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
		Method:     ModelMergeLinear,
		Sources: []ModelMergeSource{
			{Path: left},
			{Path: right},
		},
	})
	if err == nil {
		t.Fatal("expected architecture mismatch")
	}
	if !core.Contains(err.Error(), "architecture") {
		t.Fatalf("error = %v, want architecture context", err)
	}
}

func TestMergeModelPacks_RejectsTensorShapeMismatch_Ugly(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{3, 4, 5}},
	})

	_, err := MergeModelPacks(context.Background(), ModelMergeOptions{
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
		Method:     ModelMergeLinear,
		Sources: []ModelMergeSource{
			{Path: left},
			{Path: right},
		},
	})
	if err == nil {
		t.Fatal("expected tensor shape mismatch")
	}
	if !core.Contains(err.Error(), "shape") {
		t.Fatalf("error = %v, want shape context", err)
	}
}

func TestModelMerge_SafetensorIndexErrors_Bad(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.norm.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{1}, Data: []float32{1}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{1}, Data: []float32{2}}})
	if _, err := indexSafetensorFiles([]string{leftPath, rightPath}); err == nil {
		t.Fatal("indexSafetensorFiles(duplicate tensor) error = nil")
	}
	if _, err := readSafetensorIndex(core.PathJoin(t.TempDir(), "missing.safetensors")); err == nil {
		t.Fatal("readSafetensorIndex(missing) error = nil")
	}
	if _, err := safetensorRefFromHeader("bad.safetensors", "bad", safetensorHeaderEntry{DType: "F32", Shape: []int64{1}, DataOffsets: []int64{1}}, 8); err == nil {
		t.Fatal("safetensorRefFromHeader(bad offsets len) error = nil")
	}
	if _, err := safetensorRefFromHeader("bad.safetensors", "bad", safetensorHeaderEntry{DType: "F32", Shape: []int64{0}, DataOffsets: []int64{0, 4}}, 8); err == nil {
		t.Fatal("safetensorRefFromHeader(bad shape) error = nil")
	}
	if err := validateModelMergeTensorIndexes([]safetensorIndex{
		{Names: []string{"a"}, Tensors: map[string]safetensorTensorRef{"a": {Name: "a", Shape: []uint64{1}}}},
		{Names: []string{"b"}, Tensors: map[string]safetensorTensorRef{"b": {Name: "b", Shape: []uint64{1}}}},
	}, false); err == nil {
		t.Fatal("validateModelMergeTensorIndexes(missing tensor) error = nil")
	}
	if err := validateModelMergeTensorIndexes([]safetensorIndex{
		{Names: []string{"a"}, Tensors: map[string]safetensorTensorRef{"a": {Name: "a", Shape: []uint64{1}}}},
		{Names: []string{"a", "b"}, Tensors: map[string]safetensorTensorRef{"a": {Name: "a", Shape: []uint64{1}}, "b": {Name: "b", Shape: []uint64{1}}}},
	}, false); err == nil {
		t.Fatal("validateModelMergeTensorIndexes(extra tensor) error = nil")
	}
}

func assertMergedTensorValues(t *testing.T, tensors []denseSafetensor, want []float32) {
	t.Helper()
	if len(tensors) != 1 {
		t.Fatalf("tensor count = %d, want 1", len(tensors))
	}
	if len(tensors[0].Data) != len(want) {
		t.Fatalf("data length = %d, want %d", len(tensors[0].Data), len(want))
	}
	assertFloat32Values(t, tensors[0].Data, want)
}

func assertFloat32Values(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("data length = %d, want %d", len(got), len(want))
	}
	for i, value := range got {
		if math.Abs(float64(value-want[i])) > 1e-5 {
			t.Fatalf("data[%d] = %f, want %f (all=%v)", i, value, want[i], got)
		}
	}
}
