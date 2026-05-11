// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"sort"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// ModelMergeMethod names the tensor merge algorithm.
type ModelMergeMethod string

const (
	ModelMergeLinear ModelMergeMethod = "linear"
	ModelMergeSLERP  ModelMergeMethod = "slerp"
	ModelMergeTIES   ModelMergeMethod = "ties"
	ModelMergeDARE   ModelMergeMethod = "dare"

	ModelMergeProvenanceFile      = "model_merge_provenance.json"
	modelMergeOutputWeights       = "model.safetensors"
	modelMergeTensorChunkElements = 1 << 20
)

// ModelMergeSource identifies one local model pack participating in a merge.
type ModelMergeSource struct {
	Path   string  `json:"path"`
	Weight float64 `json:"weight,omitempty"`
}

// ModelMergeOptions configures local model-pack tensor merging.
type ModelMergeOptions struct {
	Sources                   []ModelMergeSource `json:"sources"`
	OutputPath                string             `json:"output_path"`
	Method                    ModelMergeMethod   `json:"method,omitempty"`
	T                         float64            `json:"t,omitempty"`
	AllowArchitectureMismatch bool               `json:"allow_architecture_mismatch,omitempty"`
	AllowTokenizerMismatch    bool               `json:"allow_tokenizer_mismatch,omitempty"`
	AllowTensorMismatch       bool               `json:"allow_tensor_mismatch,omitempty"`
	Labels                    map[string]string  `json:"labels,omitempty"`
}

// ModelMergeResult reports the generated merged model pack.
type ModelMergeResult struct {
	OutputPath     string           `json:"output_path"`
	WeightPath     string           `json:"weight_path"`
	ProvenancePath string           `json:"provenance_path"`
	Method         ModelMergeMethod `json:"method"`
	T              float64          `json:"t,omitempty"`
	Sources        []mp.ModelPack      `json:"sources"`
	Pack           mp.ModelPack        `json:"pack"`
	TensorCount    int              `json:"tensor_count"`
	MergedTensors  int              `json:"merged_tensors"`
	CopiedTensors  int              `json:"copied_tensors,omitempty"`
	SkippedTensors []string         `json:"skipped_tensors,omitempty"`
}

// ModelMergeProvenance records how a merged pack was produced.
type ModelMergeProvenance struct {
	Version        int                `json:"version"`
	Method         ModelMergeMethod   `json:"method"`
	T              float64            `json:"t,omitempty"`
	Sources        []ModelMergeSource `json:"sources"`
	SourcePacks    []mp.ModelPack        `json:"source_packs"`
	OutputWeight   string             `json:"output_weight"`
	MergedTensors  int                `json:"merged_tensors"`
	CopiedTensors  int                `json:"copied_tensors,omitempty"`
	SkippedTensors []string           `json:"skipped_tensors,omitempty"`
	Labels         map[string]string  `json:"labels,omitempty"`
}

type modelMergePrepared struct {
	Method  ModelMergeMethod
	T       float64
	Sources []ModelMergeSource
	Packs   []mp.ModelPack
	Output  string
}

// MergeModelPacks merges compatible local safetensors model packs and writes a loadable pack.
func MergeModelPacks(ctx context.Context, opts ModelMergeOptions) (*ModelMergeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepareModelMerge(ctx, opts)
	if err != nil {
		return nil, err
	}

	indexes, err := indexModelMergeSources(prepared.Packs)
	if err != nil {
		return nil, err
	}
	if err := validateModelMergeTensorIndexes(indexes, opts.AllowTensorMismatch); err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(prepared.Output, modelMergeOutputWeights)
	merged, copied, skipped, err := writeMergedSafetensors(ctx, weightPath, indexes, prepared.Method, prepared.T, prepared.Sources, opts.AllowTensorMismatch)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prepared.Output, ModelMergeProvenanceFile)
	if err := writeModelMergeProvenance(provenancePath, ModelMergeProvenance{
		Version:        1,
		Method:         prepared.Method,
		T:              prepared.T,
		Sources:        prepared.Sources,
		SourcePacks:    prepared.Packs,
		OutputWeight:   core.PathBase(weightPath),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
		Labels:         opts.Labels,
	}); err != nil {
		return nil, err
	}

	pack, err := ValidateModelPack(prepared.Output)
	if err != nil {
		return nil, core.E("MergeModelPacks", "validate generated model pack", err)
	}
	return &ModelMergeResult{
		OutputPath:     prepared.Output,
		WeightPath:     weightPath,
		ProvenancePath: provenancePath,
		Method:         prepared.Method,
		T:              prepared.T,
		Sources:        prepared.Packs,
		Pack:           pack,
		TensorCount:    len(indexes[0].Names),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
	}, nil
}

func prepareModelMerge(ctx context.Context, opts ModelMergeOptions) (modelMergePrepared, error) {
	if err := ctx.Err(); err != nil {
		return modelMergePrepared{}, err
	}
	if len(opts.Sources) < 2 {
		return modelMergePrepared{}, core.NewError("mlx: model merge requires at least two sources")
	}
	if opts.OutputPath == "" {
		return modelMergePrepared{}, core.NewError("mlx: merged model output path is required")
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") || core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") {
		return modelMergePrepared{}, core.NewError("mlx: merged output path must be a model-pack directory")
	}

	method := opts.Method
	if method == "" {
		method = ModelMergeLinear
	}
	switch method {
	case ModelMergeLinear, ModelMergeSLERP:
	case ModelMergeTIES, ModelMergeDARE:
		return modelMergePrepared{}, core.NewError("mlx: model merge method " + string(method) + " is reserved as a future sparse-merge hook and is not implemented yet")
	default:
		return modelMergePrepared{}, core.NewError("mlx: unsupported model merge method: " + string(method))
	}
	if method == ModelMergeSLERP && len(opts.Sources) != 2 {
		return modelMergePrepared{}, core.NewError("mlx: SLERP model merge requires exactly two sources")
	}
	if opts.T < 0 || opts.T > 1 {
		return modelMergePrepared{}, core.NewError("mlx: model merge t must be between 0 and 1")
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if err := ensureEmptyModelMergeDestination(output); err != nil {
		return modelMergePrepared{}, err
	}

	packs := make([]mp.ModelPack, 0, len(opts.Sources))
	normalizedSources := make([]ModelMergeSource, 0, len(opts.Sources))
	for _, source := range opts.Sources {
		if source.Path == "" {
			return modelMergePrepared{}, core.NewError("mlx: model merge source path is required")
		}
		pack, err := ValidateModelPack(source.Path)
		if err != nil {
			return modelMergePrepared{}, core.E("MergeModelPacks", "validate source model pack", err)
		}
		if pack.Format != mp.ModelPackFormatSafetensors {
			return modelMergePrepared{}, core.NewError("mlx: model merge currently requires safetensors source weights")
		}
		if samePath(pack.Root, output) {
			return modelMergePrepared{}, core.NewError("mlx: merged output path must differ from source model path")
		}
		normalized := source
		normalized.Path = pack.Root
		packs = append(packs, pack)
		normalizedSources = append(normalizedSources, normalized)
	}

	if err := validateModelMergePackCompatibility(packs, opts); err != nil {
		return modelMergePrepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return modelMergePrepared{}, core.E("MergeModelPacks", "create merged model directory", modelMergeResultError(result))
	}
	if err := copyModelPackMetadata(packs[0].Root, output); err != nil {
		return modelMergePrepared{}, err
	}

	return modelMergePrepared{
		Method:  method,
		T:       opts.T,
		Sources: normalizedSources,
		Packs:   packs,
		Output:  output,
	}, nil
}

func ensureEmptyModelMergeDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("MergeModelPacks", "inspect output path", modelMergeResultError(stat))
	}
	weights := append(core.PathGlob(core.PathJoin(output, "*.safetensors")), core.PathGlob(core.PathJoin(output, "*.gguf"))...)
	if len(weights) > 0 {
		return core.NewError("mlx: merged output path already contains model weights")
	}
	return nil
}

func validateModelMergePackCompatibility(packs []mp.ModelPack, opts ModelMergeOptions) error {
	base := packs[0]
	for i := 1; i < len(packs); i++ {
		pack := packs[i]
		if !opts.AllowArchitectureMismatch && pack.Architecture != base.Architecture {
			return core.NewError(core.Sprintf("mlx: model merge architecture mismatch: %s vs %s", base.Architecture, pack.Architecture))
		}
		if opts.AllowTokenizerMismatch {
			continue
		}
		baseHash, err := StateBundleFileHash(base.TokenizerPath)
		if err != nil {
			return core.E("MergeModelPacks", "hash base tokenizer", err)
		}
		hash, err := StateBundleFileHash(pack.TokenizerPath)
		if err != nil {
			return core.E("MergeModelPacks", "hash tokenizer", err)
		}
		if hash != baseHash {
			return core.NewError("mlx: model merge tokenizer mismatch")
		}
	}
	return nil
}

func indexModelMergeSources(packs []mp.ModelPack) ([]safetensors.Index, error) {
	indexes := make([]safetensors.Index, 0, len(packs))
	for _, pack := range packs {
		index, err := safetensors.IndexFiles(pack.WeightFiles)
		if err != nil {
			return nil, err
		}
		indexes = append(indexes, index)
	}
	return indexes, nil
}

func validateModelMergeTensorIndexes(indexes []safetensors.Index, allowMismatch bool) error {
	base := indexes[0]
	for i := 1; i < len(indexes); i++ {
		index := indexes[i]
		for _, name := range base.Names {
			baseRef := base.Tensors[name]
			ref, ok := index.Tensors[name]
			if !ok {
				if allowMismatch {
					continue
				}
				return core.NewError("mlx: model merge tensor missing from source: " + name)
			}
			if !sameUint64Slice(baseRef.Shape, ref.Shape) {
				if allowMismatch {
					continue
				}
				return core.NewError("mlx: model merge tensor shape mismatch: " + name)
			}
		}
		if allowMismatch {
			continue
		}
		for _, name := range index.Names {
			if _, ok := base.Tensors[name]; !ok {
				return core.NewError("mlx: model merge extra tensor in source: " + name)
			}
		}
	}
	return nil
}

func writeMergedSafetensors(ctx context.Context, path string, indexes []safetensors.Index, method ModelMergeMethod, t float64, sources []ModelMergeSource, allowMismatch bool) (int, int, []string, error) {
	header := buildMergedSafetensorsHeader(indexes[0])
	created := core.Create(path)
	if !created.OK {
		return 0, 0, nil, modelMergeResultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return 0, 0, nil, modelMergeResultError(encoded)
	}
	headerBytes := encoded.Value.([]byte)
	if err := binary.Write(file, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return 0, 0, nil, err
	}
	if _, err := file.Write(headerBytes); err != nil {
		return 0, 0, nil, err
	}

	linearWeights, err := normalizedMergeWeights(sources)
	if err != nil {
		return 0, 0, nil, err
	}

	var merged int
	var copied int
	var skipped []string
	for _, name := range indexes[0].Names {
		if err := ctx.Err(); err != nil {
			return 0, 0, nil, err
		}
		if method == ModelMergeLinear || method == ModelMergeSLERP {
			refs, complete, err := readMergeTensorRefs(indexes, name)
			if err != nil {
				return 0, 0, nil, err
			}
			switch {
			case complete:
				var err error
				if method == ModelMergeSLERP {
					err = writeSLERPMergedTensorChunks(ctx, file, refs, t, modelMergeTensorChunkElements)
				} else {
					err = writeLinearMergedTensorChunks(ctx, file, refs, linearWeights, modelMergeTensorChunkElements)
				}
				if err != nil {
					return 0, 0, nil, err
				}
				merged++
			case allowMismatch && len(refs) > 0:
				if err := safetensors.WriteRefFloat32Chunks(ctx, file, refs[0], modelMergeTensorChunkElements); err != nil {
					return 0, 0, nil, err
				}
				copied++
				skipped = append(skipped, name)
			default:
				return 0, 0, nil, core.NewError("mlx: model merge tensor mismatch: " + name)
			}
			continue
		}
		values, complete, err := readMergeTensorValues(indexes, name)
		if err != nil {
			return 0, 0, nil, err
		}
		var out []float32
		switch {
		case complete:
			out, err = mergeTensorValues(values, method, t, linearWeights)
			if err != nil {
				return 0, 0, nil, err
			}
			merged++
		case allowMismatch:
			out = values[0]
			copied++
			skipped = append(skipped, name)
		default:
			return 0, 0, nil, core.NewError("mlx: model merge tensor mismatch: " + name)
		}
		if err := writeFloat32Values(file, out); err != nil {
			return 0, 0, nil, err
		}
	}
	return merged, copied, skipped, nil
}

func readMergeTensorRefs(indexes []safetensors.Index, name string) ([]safetensors.TensorRef, bool, error) {
	refs := make([]safetensors.TensorRef, 0, len(indexes))
	var shape []uint64
	complete := true
	for _, index := range indexes {
		ref, ok := index.Tensors[name]
		if !ok {
			complete = false
			continue
		}
		if shape == nil {
			shape = ref.Shape
		} else if !sameUint64Slice(shape, ref.Shape) {
			complete = false
			continue
		}
		refs = append(refs, ref)
	}
	return refs, complete && len(refs) == len(indexes), nil
}

func buildMergedSafetensorsHeader(index safetensors.Index) map[string]safetensors.HeaderEntry {
	header := make(map[string]safetensors.HeaderEntry, len(index.Names))
	var offset int64
	for _, name := range index.Names {
		ref := index.Tensors[name]
		byteLen := int64(ref.Elements * 4)
		shape := make([]int64, 0, len(ref.Shape))
		for _, dim := range ref.Shape {
			shape = append(shape, int64(dim))
		}
		header[name] = safetensors.HeaderEntry{
			DType:       "F32",
			Shape:       shape,
			DataOffsets: []int64{offset, offset + byteLen},
		}
		offset += byteLen
	}
	return header
}

func readMergeTensorValues(indexes []safetensors.Index, name string) ([][]float32, bool, error) {
	values := make([][]float32, 0, len(indexes))
	var shape []uint64
	complete := true
	for _, index := range indexes {
		ref, ok := index.Tensors[name]
		if !ok {
			complete = false
			continue
		}
		if shape == nil {
			shape = ref.Shape
		} else if !sameUint64Slice(shape, ref.Shape) {
			complete = false
			continue
		}
		tensor, err := safetensors.ReadRefValues(ref)
		if err != nil {
			return nil, false, err
		}
		values = append(values, tensor)
	}
	return values, complete && len(values) == len(indexes), nil
}

func writeLinearMergedTensorChunks(ctx context.Context, file *core.OSFile, refs []safetensors.TensorRef, weights []float64, chunkElements int) error {
	if len(refs) == 0 {
		return core.NewError("mlx: no tensors to merge")
	}
	if len(refs) != len(weights) {
		return core.NewError("mlx: tensor merge weights do not match source count")
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	elements := refs[0].Elements
	for _, ref := range refs {
		if ref.Elements != elements {
			return core.NewError("mlx: tensor length mismatch during linear merge")
		}
	}
	readers, err := safetensors.OpenReaders(refs)
	if err != nil {
		return err
	}
	defer safetensors.CloseReaders(readers)
	for offset := 0; offset < elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(chunkElements, elements-offset)
		out := make([]float32, count)
		for sourceIndex, reader := range readers {
			values, err := reader.ReadFloat32Chunk(offset, count)
			if err != nil {
				return err
			}
			weight := weights[sourceIndex]
			for i, value := range values {
				out[i] += float32(float64(value) * weight)
			}
		}
		if err := writeFloat32Values(file, out); err != nil {
			return err
		}
	}
	return nil
}

func writeSLERPMergedTensorChunks(ctx context.Context, file *core.OSFile, refs []safetensors.TensorRef, t float64, chunkElements int) error {
	weights, err := slerpChunkedWeights(ctx, refs, t, chunkElements)
	if err != nil {
		return err
	}
	return writeLinearMergedTensorChunks(ctx, file, refs, weights, chunkElements)
}

func slerpChunkedWeights(ctx context.Context, refs []safetensors.TensorRef, t float64, chunkElements int) ([]float64, error) {
	if len(refs) != 2 {
		return nil, core.NewError("mlx: SLERP tensor merge requires exactly two tensors")
	}
	if refs[0].Elements != refs[1].Elements {
		return nil, core.NewError("mlx: tensor length mismatch during SLERP merge")
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	readers, err := safetensors.OpenReaders(refs)
	if err != nil {
		return nil, err
	}
	defer safetensors.CloseReaders(readers)

	var dot float64
	var normA float64
	var normB float64
	for offset := 0; offset < refs[0].Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := min(chunkElements, refs[0].Elements-offset)
		a, err := readers[0].ReadFloat32Chunk(offset, count)
		if err != nil {
			return nil, err
		}
		b, err := readers[1].ReadFloat32Chunk(offset, count)
		if err != nil {
			return nil, err
		}
		for i := range a {
			av := float64(a[i])
			bv := float64(b[i])
			dot += av * bv
			normA += av * av
			normB += bv * bv
		}
	}
	if normA == 0 || normB == 0 {
		return []float64{1 - t, t}, nil
	}
	cosTheta := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	cosTheta = clampFloat64(cosTheta, -1, 1)
	if math.Abs(cosTheta) > 0.9995 {
		return []float64{1 - t, t}, nil
	}
	theta := math.Acos(cosTheta)
	sinTheta := math.Sin(theta)
	return []float64{
		math.Sin((1-t)*theta) / sinTheta,
		math.Sin(t*theta) / sinTheta,
	}, nil
}

func mergeTensorValues(values [][]float32, method ModelMergeMethod, t float64, weights []float64) ([]float32, error) {
	switch method {
	case ModelMergeLinear:
		return linearMergeTensorValues(values, weights)
	case ModelMergeSLERP:
		return slerpMergeTensorValues(values, t)
	default:
		return nil, core.NewError("mlx: unsupported model merge method: " + string(method))
	}
}

func linearMergeTensorValues(values [][]float32, weights []float64) ([]float32, error) {
	if len(values) == 0 {
		return nil, core.NewError("mlx: no tensors to merge")
	}
	out := make([]float32, len(values[0]))
	for sourceIndex, source := range values {
		if len(source) != len(out) {
			return nil, core.NewError("mlx: tensor length mismatch during linear merge")
		}
		weight := weights[sourceIndex]
		for i, value := range source {
			out[i] += float32(float64(value) * weight)
		}
	}
	return out, nil
}

func slerpMergeTensorValues(values [][]float32, t float64) ([]float32, error) {
	if len(values) != 2 {
		return nil, core.NewError("mlx: SLERP tensor merge requires exactly two tensors")
	}
	a := values[0]
	b := values[1]
	if len(a) != len(b) {
		return nil, core.NewError("mlx: tensor length mismatch during SLERP merge")
	}
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return linearMergeTensorValues(values, []float64{1 - t, t})
	}
	cosTheta := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	cosTheta = clampFloat64(cosTheta, -1, 1)
	if math.Abs(cosTheta) > 0.9995 {
		return linearMergeTensorValues(values, []float64{1 - t, t})
	}
	theta := math.Acos(cosTheta)
	sinTheta := math.Sin(theta)
	scaleA := math.Sin((1-t)*theta) / sinTheta
	scaleB := math.Sin(t*theta) / sinTheta
	return linearMergeTensorValues(values, []float64{scaleA, scaleB})
}

func normalizedMergeWeights(sources []ModelMergeSource) ([]float64, error) {
	weights := make([]float64, len(sources))
	var total float64
	var explicit bool
	for i, source := range sources {
		if math.IsNaN(source.Weight) || math.IsInf(source.Weight, 0) {
			return nil, core.NewError("mlx: model merge source weight must be finite")
		}
		if source.Weight != 0 {
			explicit = true
		}
		weights[i] = source.Weight
		total += source.Weight
	}
	if !explicit {
		equal := 1 / float64(len(sources))
		for i := range weights {
			weights[i] = equal
		}
		return weights, nil
	}
	if total == 0 {
		return nil, core.NewError("mlx: model merge source weights sum to zero")
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights, nil
}

func writeFloat32Values(file *core.OSFile, values []float32) error {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	_, err := file.Write(raw)
	return err
}

func writeModelMergeProvenance(path string, provenance ModelMergeProvenance) error {
	slices := append([]string(nil), provenance.SkippedTensors...)
	sort.Strings(slices)
	provenance.SkippedTensors = slices
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("MergeModelPacks", "marshal merge provenance", modelMergeResultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("MergeModelPacks", "write merge provenance", modelMergeResultError(result))
	}
	return nil
}

func sameUint64Slice(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func clampFloat64(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func modelMergeResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

func samePath(a, b string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	patterns := []string{"*.json", "*.model", "*.txt"}
	seen := map[string]struct{}{}
	for _, pattern := range patterns {
		for _, sourcePath := range core.PathGlob(core.PathJoin(sourceRoot, pattern)) {
			name := core.PathBase(sourcePath)
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			if isModelWeightMetadataCopySkip(name) {
				continue
			}
			if err := copyModelPackLocalFile(sourcePath, core.PathJoin(outputRoot, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func isModelWeightMetadataCopySkip(name string) bool {
	lower := core.Lower(name)
	return lower == "adapter_provenance.json" ||
		core.Contains(lower, ".safetensors") ||
		core.Contains(lower, ".gguf") ||
		core.HasSuffix(lower, ".safetensors") ||
		core.HasSuffix(lower, ".gguf")
}

func copyModelPackLocalFile(sourcePath, destinationPath string) error {
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return modelPackCopyResultError(read)
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return modelPackCopyResultError(result)
	}
	return nil
}

func modelPackCopyResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("model pack metadata copy failed")
}
