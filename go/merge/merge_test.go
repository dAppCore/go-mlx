// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// TestMerge_Packs_Good is the canonical AX-7 happy-path triplet leg for the
// public Packs symbol: two compatible single-tensor safetensors packs are
// merged with a linear 0.25/0.75 weighting. Packs writes a loadable merged
// pack whose tensor equals 0.25*left + 0.75*right, plus a provenance file.
func TestMerge_Packs_Good(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{0, 2, 4, 6}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{10, 12, 14, 16}},
	})
	output := core.PathJoin(t.TempDir(), "merged-canonical")

	result, err := Packs(context.Background(), Options{
		OutputPath: output,
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left), Weight: 0.25},
			{Pack: testPack(right), Weight: 0.75},
		},
	})
	if err != nil {
		t.Fatalf("Packs() error = %v", err)
	}
	if result.Method != MethodLinear || result.TensorCount != 1 || result.MergedTensors != 1 {
		t.Fatalf("result = %+v, want one linearly merged tensor", result)
	}
	if stat := core.Stat(core.PathJoin(output, ProvenanceFile)); !stat.OK {
		t.Fatalf("provenance was not written: %v", stat.Value)
	}
	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	assertMergedTensorValues(t, tensors, []float32{7.5, 9.5, 11.5, 13.5})
}

// TestMerge_Packs_Bad is the canonical AX-7 invalid-input triplet leg: Packs
// must reject a merge whose two sources declare different architectures (the
// default, no-AllowArchitectureMismatch policy) with an error that names the
// architecture conflict, and must not produce a merged pack.
func TestMerge_Packs_Bad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "gemma3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-bad-arch"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPackArch(left, "qwen3")},
			{Pack: testPackArch(right, "gemma3")},
		},
	})
	if err == nil {
		t.Fatal("Packs(architecture mismatch) error = nil, want rejection")
	}
	if !core.Contains(err.Error(), "architecture") {
		t.Fatalf("error = %v, want architecture context", err)
	}
}

// TestMerge_Packs_Ugly is the canonical AX-7 edge-case triplet leg: a SLERP
// merge where one source tensor is the zero vector. The SLERP chunk scan finds
// a zero norm and falls back to the plain linear blend with weights (1-t, t),
// so the merged output must equal t*right rather than a sin-ratio
// interpolation. Packs still succeeds and writes a valid pack.
func TestMerge_Packs_Ugly(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{3}, Data: []float32{0, 0, 0}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{3}, Data: []float32{2, 4, 8}},
	})

	result, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-canonical-slerp-zero"),
		Method:     MethodSLERP,
		T:          0.25,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err != nil {
		t.Fatalf("Packs(slerp zero norm) error = %v", err)
	}
	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	// Linear fallback with weights (1-t, t) = (0.75, 0.25) => 0.25 * right.
	assertMergedTensorValues(t, tensors, []float32{0.5, 1, 2})
}

func TestMergeModelPacks_LinearSafetensorsGood(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{0, 2, 4, 6}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{4}, Data: []float32{10, 12, 14, 16}},
	})
	output := core.PathJoin(t.TempDir(), "merged-linear")

	result, err := Packs(context.Background(), Options{
		OutputPath: output,
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left), Weight: 0.25},
			{Pack: testPack(right), Weight: 0.75},
		},
	})
	if err != nil {
		t.Fatalf("Packs() error = %v", err)
	}
	if result.Method != MethodLinear || result.TensorCount != 1 || result.MergedTensors != 1 {
		t.Fatalf("result = %+v", result)
	}
	if result.WeightPath != core.PathJoin(output, "model.safetensors") {
		t.Fatalf("WeightPath = %q", result.WeightPath)
	}
	if stat := core.Stat(result.WeightPath); !stat.OK {
		t.Fatalf("weight path missing: %v", stat.Value)
	}

	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	assertMergedTensorValues(t, tensors, []float32{7.5, 9.5, 11.5, 13.5})
	if stat := core.Stat(core.PathJoin(output, ProvenanceFile)); !stat.OK {
		t.Fatalf("provenance was not written: %v", stat.Value)
	}
}

func TestMergeModelPacks_SLERPSafetensorsGood(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{1, 0}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{0, 1}},
	})

	result, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-slerp"),
		Method:     MethodSLERP,
		T:          0.5,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err != nil {
		t.Fatalf("Packs() error = %v", err)
	}

	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertMergedTensorValues(t, tensors, []float32{want, want})
}

func TestMergeModelPacks_AllowTensorMismatchCopiesBaseTensorGood(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
		{Name: "model.embed_tokens.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{5, 7}},
	})

	result, err := Packs(context.Background(), Options{
		OutputPath:          core.PathJoin(t.TempDir(), "merged-mismatch"),
		Method:              MethodLinear,
		AllowTensorMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
		Labels: map[string]string{"suite": "mismatch"},
	})
	if err != nil {
		t.Fatalf("Packs(allow mismatch) error = %v", err)
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

// TestMergeModelPacks_MultiTensorMixedSizes_Good merges a pack holding
// several tensors of different sizes in descending-then-ascending order.
// The writeMergedSafetensors loop shares one out/decode/write scratch
// across every tensor (cap-checked reuse), so a large tensor followed by a
// smaller one re-slices the same backing buffer — this guards that the
// merged bytes stay correct (no stale tail bleed, no length confusion)
// regardless of tensor-size ordering. Linear merge, equal weights, so each
// element is the mean of the two sources.
func TestMergeModelPacks_MultiTensorMixedSizesGood(t *testing.T) {
	big := make([]float32, 4096)
	bigRight := make([]float32, 4096)
	bigWant := make([]float32, 4096)
	for i := range big {
		big[i] = float32(i)
		bigRight[i] = float32(i) + 2
		bigWant[i] = float32(i) + 1
	}
	leftTensors := []safetensorTestTensor{
		{Name: "model.layers.0.mlp.gate_proj.weight", Shape: []int{4096}, Data: big},                               // large
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{1, 2, 3}},                                     // small after large
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{8}, Data: []float32{0, 1, 2, 3, 4, 5, 6, 7}}, // medium after small
	}
	rightTensors := []safetensorTestTensor{
		{Name: "model.layers.0.mlp.gate_proj.weight", Shape: []int{4096}, Data: bigRight},
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{5, 6, 7}},
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{8}, Data: []float32{2, 3, 4, 5, 6, 7, 8, 9}},
	}
	left := writeDenseSafetensorsPack(t, "qwen3", leftTensors)
	right := writeDenseSafetensorsPack(t, "qwen3", rightTensors)

	result, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-mixed"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left), Weight: 0.5},
			{Pack: testPack(right), Weight: 0.5},
		},
	})
	if err != nil {
		t.Fatalf("Packs() error = %v", err)
	}
	if result.MergedTensors != 3 {
		t.Fatalf("merged tensors = %d, want 3", result.MergedTensors)
	}
	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	if len(tensors) != 3 {
		t.Fatalf("tensor count = %d, want 3", len(tensors))
	}
	for _, tensor := range tensors {
		switch tensor.Name {
		case "model.layers.0.mlp.gate_proj.weight":
			assertFloat32Values(t, tensor.Data, bigWant)
		case "model.norm.weight":
			assertFloat32Values(t, tensor.Data, []float32{3, 4, 5})
		case "model.layers.0.self_attn.q_proj.weight":
			assertFloat32Values(t, tensor.Data, []float32{1, 2, 3, 4, 5, 6, 7, 8})
		default:
			t.Fatalf("unexpected tensor %q", tensor.Name)
		}
	}
}

func TestModelMerge_ValueMergeHelpersGood(t *testing.T) {
	linear, err := mergeTensorValues([][]float32{
		{0, 2, 4},
		{10, 12, 14},
	}, MethodLinear, 0, []float64{0.25, 0.75})
	if err != nil {
		t.Fatalf("mergeTensorValues(linear) error = %v", err)
	}
	assertFloat32Values(t, linear, []float32{7.5, 9.5, 11.5})

	slerp, err := mergeTensorValues([][]float32{
		{1, 0},
		{0, 1},
	}, MethodSLERP, 0.5, nil)
	if err != nil {
		t.Fatalf("mergeTensorValues(slerp) error = %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertFloat32Values(t, slerp, []float32{want, want})

	linearFallback, err := slerpMerge([][]float32{{0, 0}, {2, 4}}, 0.25)
	if err != nil {
		t.Fatalf("slerpMerge(zero norm) error = %v", err)
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

func TestModelMerge_ValueMergeHelpersBad(t *testing.T) {
	if _, err := mergeTensorValues([][]float32{{1}}, "bad", 0, []float64{1}); err == nil {
		t.Fatal("mergeTensorValues(unsupported) error = nil")
	}
	if _, err := linearMerge(nil, nil); err == nil {
		t.Fatal("linearMerge(nil) error = nil")
	}
	if _, err := linearMerge([][]float32{{1}, {1, 2}}, []float64{0.5, 0.5}); err == nil {
		t.Fatal("linearMerge(length mismatch) error = nil")
	}
	if _, err := slerpMerge([][]float32{{1}}, 0.5); err == nil {
		t.Fatal("slerpMerge(one tensor) error = nil")
	}
	if _, err := slerpMerge([][]float32{{1}, {1, 2}}, 0.5); err == nil {
		t.Fatal("slerpMerge(length mismatch) error = nil")
	}
	if _, err := normalizedWeights([]Source{{Weight: math.NaN()}}); err == nil {
		t.Fatal("normalizedWeights(NaN) error = nil")
	}
	if _, err := normalizedWeights([]Source{{Weight: 1}, {Weight: -1}}); err == nil {
		t.Fatal("normalizedWeights(zero sum) error = nil")
	}
}

func TestPrepareModelMerge_Bad_Validation(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}}})
	other := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{2}}})
	occupied := t.TempDir()
	writeModelPackFile(t, core.PathJoin(occupied, "model.safetensors"), "occupied")
	cases := []struct {
		name string
		opts Options
	}{
		{name: "not enough sources", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []Source{{Pack: testPack(source)}}}},
		{name: "missing output", opts: Options{Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "file output", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out.safetensors"), Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "unsupported method", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: "bad", Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "future method", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: MethodTIES, Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "slerp source count", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Method: MethodSLERP, Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}, {Pack: testPack(other)}}}},
		{name: "bad t", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), T: 2, Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "empty source", opts: Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []Source{{Pack: testPack(source)}, {}}}},
		{name: "same output", opts: Options{OutputPath: source, Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
		{name: "occupied output", opts: Options{OutputPath: occupied, Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := prepare(context.Background(), tc.opts); err == nil {
				t.Fatal("prepare() error = nil")
			}
		})
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := prepare(cancelled, Options{OutputPath: core.PathJoin(t.TempDir(), "out"), Sources: []Source{{Pack: testPack(source)}, {Pack: testPack(other)}}}); err == nil {
		t.Fatal("prepare(cancelled) error = nil")
	}
}

func TestMergeModelPacks_RejectsArchitectureMismatchBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "gemma3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPackArch(left, "qwen3")},
			{Pack: testPackArch(right, "gemma3")},
		},
	})
	if err == nil {
		t.Fatal("expected architecture mismatch")
	}
	if !core.Contains(err.Error(), "architecture") {
		t.Fatalf("error = %v, want architecture context", err)
	}
}

func TestMergeModelPacks_RejectsTensorShapeMismatchUgly(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{3, 4, 5}},
	})

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("expected tensor shape mismatch")
	}
	if !core.Contains(err.Error(), "shape") {
		t.Fatalf("error = %v, want shape context", err)
	}
}

func TestModelMerge_SafetensorIndexErrorsBad(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.norm.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{1}, Data: []float32{1}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{1}, Data: []float32{2}}})
	if _, err := safetensors.IndexFiles([]string{leftPath, rightPath}); err == nil {
		t.Fatal("safetensors.IndexFiles(duplicate tensor) error = nil")
	}
	if _, err := safetensors.ReadIndex(core.PathJoin(t.TempDir(), "missing.safetensors")); err == nil {
		t.Fatal("safetensors.ReadIndex(missing) error = nil")
	}
	if _, err := safetensors.RefFromHeader("bad.safetensors", "bad", safetensors.HeaderEntry{DType: "F32", Shape: []int64{1}, DataOffsets: []int64{1}}, 8); err == nil {
		t.Fatal("safetensors.RefFromHeader(bad offsets len) error = nil")
	}
	if _, err := safetensors.RefFromHeader("bad.safetensors", "bad", safetensors.HeaderEntry{DType: "F32", Shape: []int64{0}, DataOffsets: []int64{0, 4}}, 8); err == nil {
		t.Fatal("safetensors.RefFromHeader(bad shape) error = nil")
	}
	if err := validateTensorIndexes([]safetensors.Index{
		{Names: []string{"a"}, Tensors: map[string]safetensors.TensorRef{"a": {Name: "a", Shape: []uint64{1}}}},
		{Names: []string{"b"}, Tensors: map[string]safetensors.TensorRef{"b": {Name: "b", Shape: []uint64{1}}}},
	}, false); err == nil {
		t.Fatal("validateTensorIndexes(missing tensor) error = nil")
	}
	if err := validateTensorIndexes([]safetensors.Index{
		{Names: []string{"a"}, Tensors: map[string]safetensors.TensorRef{"a": {Name: "a", Shape: []uint64{1}}}},
		{Names: []string{"a", "b"}, Tensors: map[string]safetensors.TensorRef{"a": {Name: "a", Shape: []uint64{1}}, "b": {Name: "b", Shape: []uint64{1}}}},
	}, false); err == nil {
		t.Fatal("validateTensorIndexes(extra tensor) error = nil")
	}
}

// TestMergeModelPacks_SLERPZeroNormFallback_Ugly drives the full Packs path
// with one source tensor that is the zero vector. The SLERP chunk scan finds
// normA == 0, so slerpChunkedWeightsFromReaders falls back to the linear
// weight pair (1-t, t) rather than the sin-ratio interpolation — the merged
// output must equal the plain linear blend.
func TestMergeModelPacks_SLERPZeroNormFallbackUgly(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{3}, Data: []float32{0, 0, 0}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.embed_tokens.weight", Shape: []int{3}, Data: []float32{2, 4, 8}},
	})

	result, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-slerp-zero"),
		Method:     MethodSLERP,
		T:          0.25,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err != nil {
		t.Fatalf("Packs(slerp zero norm) error = %v", err)
	}
	tensors, err := loadDenseSafetensors([]string{result.WeightPath})
	if err != nil {
		t.Fatalf("load merged safetensors: %v", err)
	}
	// Linear fallback with weights (1-t, t) = (0.75, 0.25): 0.25 * right.
	assertMergedTensorValues(t, tensors, []float32{0.5, 1, 2})
}

// TestMergeModelPacks_CopiesMetadataFiles_Good confirms Packs copies the base
// pack's metadata files (config.json, *.txt) into the merged output while
// skipping weight-adjacent metadata (anything named *.safetensors* or *.gguf*
// and adapter_provenance.json) — exercising copyModelPackMetadata's suffix
// match and isModelWeightMetadataCopySkip filter through the public API.
func TestMergeModelPacks_CopiesMetadataFilesGood(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// Extra metadata the merge should carry over.
	writeModelPackFile(t, core.PathJoin(left, "special_tokens_map.txt"), "<eos>")
	// Weight-adjacent metadata the merge must NOT copy.
	writeModelPackFile(t, core.PathJoin(left, "model.safetensors.index.json"), `{"weight_map":{}}`)
	writeModelPackFile(t, core.PathJoin(left, "adapter_provenance.json"), `{}`)

	output := core.PathJoin(t.TempDir(), "merged-meta")
	if _, err := Packs(context.Background(), Options{
		OutputPath: output,
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	}); err != nil {
		t.Fatalf("Packs() error = %v", err)
	}
	if stat := core.Stat(core.PathJoin(output, "config.json")); !stat.OK {
		t.Fatalf("config.json not copied: %v", stat.Value)
	}
	if stat := core.Stat(core.PathJoin(output, "special_tokens_map.txt")); !stat.OK {
		t.Fatalf("special_tokens_map.txt not copied: %v", stat.Value)
	}
	if stat := core.Stat(core.PathJoin(output, "model.safetensors.index.json")); stat.OK {
		t.Fatal("weight index metadata should be skipped, but it was copied")
	}
	if stat := core.Stat(core.PathJoin(output, "adapter_provenance.json")); stat.OK {
		t.Fatal("adapter_provenance.json should be skipped, but it was copied")
	}
}

// TestMergeModelPacks_ContextCancelledMidMerge_Bad cancels the context before
// Packs runs, so the prepare()-stage ctx.Err() guard rejects the merge and no
// output directory is produced.
func TestMergeModelPacks_ContextCancelledMidMergeBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Packs(ctx, Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-cancelled"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	}); err == nil {
		t.Fatal("Packs(cancelled) error = nil")
	}
}

// TestModelMerge_ReadTensorRefs_Good covers the single-call readTensorRefs
// wrapper (the nil-scratch variant of readTensorRefsInto). A matched name
// present in both indexes returns both refs and complete == true.
func TestModelMerge_ReadTensorRefsGood(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.norm.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{1, 2}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{3, 4}}})
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}

	refs, complete, err := readTensorRefs([]safetensors.Index{leftIndex, rightIndex}, name)
	if err != nil {
		t.Fatalf("readTensorRefs() error = %v", err)
	}
	if !complete || len(refs) != 2 {
		t.Fatalf("refs len/complete = %d/%v, want 2/true", len(refs), complete)
	}

	// A name absent from the second index yields complete == false.
	_, complete, err = readTensorRefs([]safetensors.Index{leftIndex, rightIndex}, "model.absent.weight")
	if err != nil {
		t.Fatalf("readTensorRefs(absent) error = %v", err)
	}
	if complete {
		t.Fatal("readTensorRefs(absent) complete = true, want false")
	}
}

// TestModelMerge_WriteFloat32Values_Good covers the single-call
// writeFloat32Values wrapper (nil-scratch variant). The little-endian byte
// view it writes must round-trip back to the input values.
func TestModelMerge_WriteFloat32ValuesGood(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "values.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	values := []float32{0, 1.5, -2.25, 1024}
	err := writeFloat32Values(file, values)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("writeFloat32Values() error = %v", err)
	}
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	decoded, err := safetensors.DecodeFloatData("F32", read.Value.([]byte), len(values))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	assertFloat32Values(t, decoded, values)

	// Empty input writes nothing and must not error.
	empty := core.Create(core.PathJoin(t.TempDir(), "empty.bin"))
	if !empty.OK {
		t.Fatalf("create empty: %v", empty.Value)
	}
	emptyFile := empty.Value.(*core.OSFile)
	if err := writeFloat32Values(emptyFile, nil); err != nil {
		t.Fatalf("writeFloat32Values(nil) error = %v", err)
	}
	if err := emptyFile.Close(); err != nil {
		t.Fatalf("close empty: %v", err)
	}
}

// TestMergeModelPacks_TokenizerMismatch_Bad rejects a merge whose sources have
// different tokenizers (default policy), then confirms AllowTokenizerMismatch
// lets the same merge through — exercising validatePackCompatibility's
// tokenizer-hash legs and hashFile.
func TestMergeModelPacks_TokenizerMismatchBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// Diverge the right pack's tokenizer so the hashes differ.
	writeModelPackFile(t, core.PathJoin(right, "tokenizer.json"), `{"model":{"type":"BPE","vocab":{"b":0},"merges":[]}}`)

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-tok"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(tokenizer mismatch) error = nil")
	}

	// With the mismatch explicitly allowed the merge succeeds.
	if _, err := Packs(context.Background(), Options{
		OutputPath:             core.PathJoin(t.TempDir(), "merged-tok-ok"),
		Method:                 MethodLinear,
		AllowTokenizerMismatch: true,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	}); err != nil {
		t.Fatalf("Packs(allow tokenizer mismatch) error = %v", err)
	}
}

// TestMergeModelPacks_UnreadableMetadataFile_Ugly chmod-000s a copyable
// metadata file in the base pack so copyModelPackLocalFile's source-open fails,
// driving the otherwise-unreached error leg and modelPackCopyResultError. The
// merge aborts with that copy error before any weights are written.
//
// chmod-000 denies the file's owner only when the process is not root; the
// guard below skips the case under a root-owned CI runner where the open would
// still succeed (and invert the expectation).
func TestMergeModelPacks_UnreadableMetadataFileUgly(t *testing.T) {
	if isRoot(t) {
		t.Skip("chmod-000 does not deny root; skipping copy-failure injection")
	}
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// config.json is a copyable metadata file (not weight-adjacent); make it
	// unreadable so copyModelPackMetadata -> copyModelPackLocalFile fails to open
	// the source.
	configPath := core.PathJoin(left, "config.json")
	if result := core.Chmod(configPath, 0o000); !result.OK {
		t.Fatalf("chmod config.json 000: %v", result.Value)
	}
	t.Cleanup(func() { core.Chmod(configPath, 0o644) })

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-unreadable-meta"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(unreadable metadata) error = nil, want copy failure")
	}
}

// TestMergeModelPacks_UnreadableTokenizer_Bad chmod-000s the base pack's
// tokenizer so validatePackCompatibility's hashFile read fails on the
// default (no AllowTokenizerMismatch) policy — the otherwise-unreached
// hashFile error leg and its resultError fallback.
func TestMergeModelPacks_UnreadableTokenizerBad(t *testing.T) {
	if isRoot(t) {
		t.Skip("chmod-000 does not deny root; skipping tokenizer-read-failure injection")
	}
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	tokPath := core.PathJoin(left, "tokenizer.json")
	if result := core.Chmod(tokPath, 0o000); !result.OK {
		t.Fatalf("chmod tokenizer.json 000: %v", result.Value)
	}
	t.Cleanup(func() { core.Chmod(tokPath, 0o644) })

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-unreadable-tok"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(unreadable tokenizer) error = nil, want hash failure")
	}
}

// TestMergeModelPacks_OutputParentIsFile_Bad points OutputPath at a path whose
// parent is a regular file. Stat of the output then fails with something other
// than not-exist (ENOTDIR), driving ensureEmptyDestination's stat-error leg
// (the non-IsNotExist return) rather than the empty-or-occupied legs the other
// tests cover.
func TestMergeModelPacks_OutputParentIsFileBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	// A regular file standing in for a directory: <file>/merged cannot be
	// stat-ed because <file> is not a directory.
	parentFile := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, parentFile, "x")

	_, err := Packs(context.Background(), Options{
		OutputPath: core.PathJoin(parentFile, "merged"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	})
	if err == nil {
		t.Fatal("Packs(output parent is a file) error = nil, want stat failure")
	}
}

// countingCancelContext returns nil from Err() for the first cancelAfter
// calls, then context.Canceled for every call after. It lets a test drive
// Packs past the early prepare()-stage ctx.Err() guard and trip the
// per-tensor cancel check inside the writeMergedSafetensors loop, exercising
// the mid-merge cancellation leg through the public API (ctx is a genuine
// production seam — no fault-injected writer/reader needed).
type countingCancelContext struct {
	context.Context
	calls       int
	cancelAfter int
}

func (c *countingCancelContext) Err() error {
	c.calls++
	if c.calls > c.cancelAfter {
		return context.Canceled
	}
	return nil
}

// TestMergeModelPacks_CancelDuringTensorLoop_Bad cancels the merge after the
// prepare-stage context check passes but before the first tensor is written,
// so the writeMergedSafetensors per-tensor ctx.Err() guard aborts the write
// loop. prepare() consumes the first Err() call; cancelAfter=1 lets that one
// through and trips the next (the loop's first iteration).
func TestMergeModelPacks_CancelDuringTensorLoopBad(t *testing.T) {
	left := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	right := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	ctx := &countingCancelContext{Context: context.Background(), cancelAfter: 1}

	if _, err := Packs(ctx, Options{
		OutputPath: core.PathJoin(t.TempDir(), "merged-cancel-loop"),
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: testPack(left)},
			{Pack: testPack(right)},
		},
	}); err == nil {
		t.Fatal("Packs(cancel during tensor loop) error = nil, want cancellation")
	}
}

// isRoot reports whether the test process runs with uid 0, where chmod-000
// does not deny the owner and the permission-injection tests would invert.
func isRoot(t *testing.T) bool {
	t.Helper()
	current := core.UserCurrent()
	if !current.OK {
		return false
	}
	user, ok := current.Value.(*core.User)
	if !ok {
		return false
	}
	return user.Uid == "0"
}

// assertMergedTensorValues asserts the single merged tensor equals want.
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
