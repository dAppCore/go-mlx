// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"encoding/binary"
	"math"
	"sort"
	"unsafe"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// Method names the tensor merge algorithm.
type Method string

const (
	MethodLinear Method = "linear"
	MethodSLERP  Method = "slerp"
	MethodTIES   Method = "ties"
	MethodDARE   Method = "dare"

	ProvenanceFile                = "model_merge_provenance.json"
	modelMergeOutputWeights       = "model.safetensors"
	modelMergeTensorChunkElements = 1 << 20
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Sharing instances also makes errors.Is comparable for
// callers distinguishing "no tensors" from "len mismatch" without
// parsing message text.
var (
	errSLERPLenMismatch        = core.NewError("mlx: tensor length mismatch during SLERP merge")
	errSLERPNeedTwoTensors     = core.NewError("mlx: SLERP tensor merge requires exactly two tensors")
	errLinearLenMismatch       = core.NewError("mlx: tensor length mismatch during linear merge")
	errNoTensors               = core.NewError("mlx: no tensors to merge")
	errOutputHasWeights        = core.NewError("mlx: merged output path already contains model weights")
	errPackMetadataCopy        = core.NewError("model pack metadata copy failed")
	errWeightsSourceCount      = core.NewError("mlx: tensor merge weights do not match source count")
	errSLERPNeedTwoReaders     = core.NewError("mlx: SLERP tensor merge requires exactly two readers")
	errSLERPNeedTwoSources     = core.NewError("mlx: SLERP model merge requires exactly two sources")
	errTokenizerMismatch       = core.NewError("mlx: model merge tokenizer mismatch")
	errMergeTOutOfRange        = core.NewError("mlx: model merge t must be between 0 and 1")
	errMergeWeightsSumZero     = core.NewError("mlx: model merge source weights sum to zero")
	errMergeWeightNotFinite    = core.NewError("mlx: model merge source weight must be finite")
	errMergeSourcePackRequired = core.NewError("mlx: model merge source pack is required")
	errMergeNeedTwoSources     = core.NewError("mlx: model merge requires at least two sources")
	errMergeNeedsSafetensors   = core.NewError("mlx: model merge currently requires safetensors source weights")
	errOutputSameAsSource      = core.NewError("mlx: merged output path must differ from source model path")
	errOutputNotPackDir        = core.NewError("mlx: merged output path must be a model-pack directory")
	errOutputPathRequired      = core.NewError("mlx: merged model output path is required")
	errReadNonByteData         = core.NewError("merge: read file returned non-byte data")
	errCoreResultFailed        = core.NewError("core result failed")
)

// Source identifies a pre-validated model pack participating in a merge.
// Callers run mlx.ValidateModelPack on each source before invoking merge.Packs.
type Source struct {
	Pack   mp.ModelPack `json:"pack"`
	Weight float64      `json:"weight,omitempty"`
}

// Options configures local model-pack tensor merging.
type Options struct {
	Sources                   []Source          `json:"sources"`
	OutputPath                string            `json:"output_path"`
	Method                    Method            `json:"method,omitempty"`
	T                         float64           `json:"t,omitempty"`
	AllowArchitectureMismatch bool              `json:"allow_architecture_mismatch,omitempty"`
	AllowTokenizerMismatch    bool              `json:"allow_tokenizer_mismatch,omitempty"`
	AllowTensorMismatch       bool              `json:"allow_tensor_mismatch,omitempty"`
	Labels                    map[string]string `json:"labels,omitempty"`
}

// Result reports the paths of the generated merged model pack and its
// per-tensor counts. Callers re-validate via mlx.ValidateModelPack(OutputPath)
// when they need a populated pack.ModelPack.
type Result struct {
	OutputPath     string         `json:"output_path"`
	WeightPath     string         `json:"weight_path"`
	ProvenancePath string         `json:"provenance_path"`
	Method         Method         `json:"method"`
	T              float64        `json:"t,omitempty"`
	Sources        []mp.ModelPack `json:"sources"`
	TensorCount    int            `json:"tensor_count"`
	MergedTensors  int            `json:"merged_tensors"`
	CopiedTensors  int            `json:"copied_tensors,omitempty"`
	SkippedTensors []string       `json:"skipped_tensors,omitempty"`
}

// Provenance records how a merged pack was produced.
type Provenance struct {
	Version        int               `json:"version"`
	Method         Method            `json:"method"`
	T              float64           `json:"t,omitempty"`
	Sources        []Source          `json:"sources"`
	SourcePacks    []mp.ModelPack    `json:"source_packs"`
	OutputWeight   string            `json:"output_weight"`
	MergedTensors  int               `json:"merged_tensors"`
	CopiedTensors  int               `json:"copied_tensors,omitempty"`
	SkippedTensors []string          `json:"skipped_tensors,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
}

type prepared struct {
	Method  Method
	T       float64
	Sources []Source
	Packs   []mp.ModelPack
	Output  string
}

// Packs merges compatible local safetensors model packs and writes a loadable pack.
func Packs(ctx context.Context, opts Options) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepare(ctx, opts)
	if err != nil {
		return nil, err
	}

	indexes, err := indexSources(prepared.Packs)
	if err != nil {
		return nil, err
	}
	if err := validateTensorIndexes(indexes, opts.AllowTensorMismatch); err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(prepared.Output, modelMergeOutputWeights)
	merged, copied, skipped, err := writeMergedSafetensors(ctx, weightPath, indexes, prepared.Method, prepared.T, prepared.Sources, opts.AllowTensorMismatch)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prepared.Output, ProvenanceFile)
	if err := writeProvenance(provenancePath, Provenance{
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

	return &Result{
		OutputPath:     prepared.Output,
		WeightPath:     weightPath,
		ProvenancePath: provenancePath,
		Method:         prepared.Method,
		T:              prepared.T,
		Sources:        prepared.Packs,
		TensorCount:    len(indexes[0].Names),
		MergedTensors:  merged,
		CopiedTensors:  copied,
		SkippedTensors: skipped,
	}, nil
}

func prepare(ctx context.Context, opts Options) (prepared, error) {
	if err := ctx.Err(); err != nil {
		return prepared{}, err
	}
	if len(opts.Sources) < 2 {
		return prepared{}, errMergeNeedTwoSources
	}
	if opts.OutputPath == "" {
		return prepared{}, errOutputPathRequired
	}
	// hasSuffixFold replaces core.Lower(opts.OutputPath) which allocated a
	// full copy of the (potentially long) output path string just to test
	// two short suffixes.
	if hasSuffixFold(opts.OutputPath, ".safetensors") || hasSuffixFold(opts.OutputPath, ".gguf") {
		return prepared{}, errOutputNotPackDir
	}

	method := opts.Method
	if method == "" {
		method = MethodLinear
	}
	switch method {
	case MethodLinear, MethodSLERP:
	case MethodTIES, MethodDARE:
		return prepared{}, core.NewError("mlx: model merge method " + string(method) + " is reserved as a future sparse-merge hook and is not implemented yet")
	default:
		return prepared{}, core.NewError("mlx: unsupported model merge method: " + string(method))
	}
	if method == MethodSLERP && len(opts.Sources) != 2 {
		return prepared{}, errSLERPNeedTwoSources
	}
	if opts.T < 0 || opts.T > 1 {
		return prepared{}, errMergeTOutOfRange
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if err := ensureEmptyDestination(output); err != nil {
		return prepared{}, err
	}

	packs := make([]mp.ModelPack, 0, len(opts.Sources))
	normalizedSources := make([]Source, 0, len(opts.Sources))
	for _, source := range opts.Sources {
		pack := source.Pack
		if pack.Root == "" {
			return prepared{}, errMergeSourcePackRequired
		}
		if pack.Format != mp.ModelPackFormatSafetensors {
			return prepared{}, errMergeNeedsSafetensors
		}
		if samePathResolved(pack.Root, output) {
			return prepared{}, errOutputSameAsSource
		}
		packs = append(packs, pack)
		normalizedSources = append(normalizedSources, source)
	}

	if err := validatePackCompatibility(packs, opts); err != nil {
		return prepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return prepared{}, core.E("Packs", "create merged model directory", resultError(result))
	}
	if err := copyModelPackMetadata(packs[0].Root, output); err != nil {
		return prepared{}, err
	}

	return prepared{
		Method:  method,
		T:       opts.T,
		Sources: normalizedSources,
		Packs:   packs,
		Output:  output,
	}, nil
}

func ensureEmptyDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("Packs", "inspect output path", resultError(stat))
	}
	// Check the two glob patterns independently — the previous append form
	// always allocated a combined slice even when the first pattern was
	// already non-empty. Short-circuit on the first non-empty pattern.
	if len(core.PathGlob(core.PathJoin(output, "*.safetensors"))) > 0 {
		return errOutputHasWeights
	}
	if len(core.PathGlob(core.PathJoin(output, "*.gguf"))) > 0 {
		return errOutputHasWeights
	}
	return nil
}

func validatePackCompatibility(packs []mp.ModelPack, opts Options) error {
	base := packs[0]
	// Hash the base tokenizer once up front, lazily — only if we actually
	// need it (any non-AllowTokenizerMismatch source). Previously the
	// inner loop re-read + re-hashed the base file once per source pack,
	// turning an O(1) check into O(N) IO + crypto for the N-source case.
	var baseHash string
	var baseHashErr error
	baseHashLoaded := opts.AllowTokenizerMismatch
	for i := 1; i < len(packs); i++ {
		pack := packs[i]
		if !opts.AllowArchitectureMismatch && pack.Architecture != base.Architecture {
			// core.Concat is ~4x cheaper than core.Sprintf for fixed-string
			// composition. Architecture names are short identifiers; the fmt
			// machinery is pure overhead here.
			return core.NewError(core.Concat(
				"mlx: model merge architecture mismatch: ",
				base.Architecture,
				" vs ",
				pack.Architecture,
			))
		}
		if opts.AllowTokenizerMismatch {
			continue
		}
		if !baseHashLoaded {
			baseHash, baseHashErr = hashFile(base.TokenizerPath)
			baseHashLoaded = true
		}
		if baseHashErr != nil {
			return core.E("Packs", "hash base tokenizer", baseHashErr)
		}
		hash, err := hashFile(pack.TokenizerPath)
		if err != nil {
			return core.E("Packs", "hash tokenizer", err)
		}
		if hash != baseHash {
			return errTokenizerMismatch
		}
	}
	return nil
}

func indexSources(packs []mp.ModelPack) ([]safetensors.Index, error) {
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

func validateTensorIndexes(indexes []safetensors.Index, allowMismatch bool) error {
	base := indexes[0]
	for i := 1; i < len(indexes); i++ {
		index := indexes[i]
		for _, name := range base.Names {
			ref, ok := index.Tensors[name]
			if !ok {
				if allowMismatch {
					continue
				}
				return core.NewError("mlx: model merge tensor missing from source: " + name)
			}
			// baseRef is only needed when we actually compare shapes — lift
			// the lookup inside the if-ok branch. Saves one map probe per
			// matched-name iteration (the dominant path).
			baseRef := base.Tensors[name]
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

func writeMergedSafetensors(ctx context.Context, path string, indexes []safetensors.Index, method Method, t float64, sources []Source, allowMismatch bool) (int, int, []string, error) {
	header := buildMergedHeader(indexes[0])
	created := core.Create(path)
	if !created.OK {
		return 0, 0, nil, resultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return 0, 0, nil, resultError(encoded)
	}
	headerBytes := encoded.Value.([]byte)
	// binary.Write goes through reflection — for a single uint64 that's
	// significant overhead. PutUint64 + file.Write is the direct form.
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	if _, err := file.Write(lenBuf[:]); err != nil {
		return 0, 0, nil, err
	}
	if _, err := file.Write(headerBytes); err != nil {
		return 0, 0, nil, err
	}

	linearWeights, err := normalizedWeights(sources)
	if err != nil {
		return 0, 0, nil, err
	}

	var merged int
	var copied int
	var skipped []string
	// One shard-handle cache for the whole write pass. The merge write
	// loop re-reads the same handful of source shard files once per
	// tensor; opening each (os.newFile + the path→C-string
	// syscall.ByteSliceFromString) per tensor made file-open the dominant
	// Packs alloc count (OpenReaders was ~33% of alloc_objects at 32
	// tensors). The cache opens each distinct shard once and hands
	// ReadAt-addressed readers over the shared handle — exactly the
	// pattern ComparePacks already uses. Closed once for the whole pass.
	cache := newFileCache()
	defer cache.close()
	// One write scratch for the whole pass — the per-tensor accumulator and
	// decode/write buffers grow to the largest tensor once and are reused
	// for every subsequent tensor instead of re-allocated per tensor (the
	// dominant Packs alloc count left after the shard cache).
	writeBuffers := &writeScratch{}
	// Reuse the refs scratch slice across tensors — readTensorRefsInto
	// rewinds length to 0 each call and only re-mallocs when capacity is
	// insufficient. Drops N-1 per-tensor make() allocs (where N = number
	// of tensors, typically 200+ for qwen3-class checkpoints).
	var refsScratch []safetensors.TensorRef
	for _, name := range indexes[0].Names {
		if err := ctx.Err(); err != nil {
			return 0, 0, nil, err
		}
		if method == MethodLinear || method == MethodSLERP {
			refs, complete, err := readTensorRefsInto(indexes, name, refsScratch)
			if err != nil {
				return 0, 0, nil, err
			}
			refsScratch = refs
			switch {
			case complete:
				var err error
				if method == MethodSLERP {
					err = writeSLERPChunksCached(ctx, file, cache, refs, t, modelMergeTensorChunkElements, writeBuffers)
				} else {
					err = writeLinearChunksCached(ctx, file, cache, refs, linearWeights, modelMergeTensorChunkElements, writeBuffers)
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
		values, complete, err := readTensorValues(indexes, name)
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

func readTensorRefs(indexes []safetensors.Index, name string) ([]safetensors.TensorRef, bool, error) {
	return readTensorRefsInto(indexes, name, nil)
}

// readTensorRefsInto is the scratch-slice-reusing variant of
// readTensorRefs. The caller passes a previously-returned slice (or
// nil) and we reset its length to 0 before refilling — the backing
// array is reused across iterations in writeMergedSafetensors so the
// per-tensor make() goes away after the first call.
func readTensorRefsInto(indexes []safetensors.Index, name string, scratch []safetensors.TensorRef) ([]safetensors.TensorRef, bool, error) {
	refs := scratch[:0]
	if cap(refs) < len(indexes) {
		refs = make([]safetensors.TensorRef, 0, len(indexes))
	}
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

func buildMergedHeader(index safetensors.Index) map[string]safetensors.HeaderEntry {
	header := make(map[string]safetensors.HeaderEntry, len(index.Names))
	// Pool both shape and DataOffsets backing arrays into one contiguous
	// []int64 slab. Previously each tensor cost 2 small heap allocations
	// (shape + 2-element DataOffsets). Now each tensor's Shape and
	// DataOffsets are sub-slices into the slab; total allocs drop from
	// 2*N to 1 across the whole header build.
	totalDims := 0
	for _, name := range index.Names {
		totalDims += len(index.Tensors[name].Shape)
	}
	// Reserve 2 trailing slots per tensor for DataOffsets.
	slab := make([]int64, totalDims+2*len(index.Names))
	shapeCursor := 0
	offsetsCursor := totalDims
	var offset int64
	for _, name := range index.Names {
		ref := index.Tensors[name]
		byteLen := int64(ref.Elements * 4)
		dims := len(ref.Shape)
		shape := slab[shapeCursor : shapeCursor : shapeCursor+dims]
		for _, dim := range ref.Shape {
			shape = append(shape, int64(dim))
		}
		shapeCursor += dims
		dataOffsets := slab[offsetsCursor : offsetsCursor+2 : offsetsCursor+2]
		dataOffsets[0] = offset
		dataOffsets[1] = offset + byteLen
		offsetsCursor += 2
		header[name] = safetensors.HeaderEntry{
			DType:       "F32",
			Shape:       shape,
			DataOffsets: dataOffsets,
		}
		offset += byteLen
	}
	return header
}

func readTensorValues(indexes []safetensors.Index, name string) ([][]float32, bool, error) {
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

// writeScratch holds the per-chunk buffers the linear write path reuses.
// In the writeMergedSafetensors loop one writeScratch is shared across
// every tensor, so the accumulator (out), the byte write buffer
// (writeBuf) and the decode buffers (rawRead, valuesRead) are grown to
// the largest tensor once instead of re-allocated per tensor — those four
// per-tensor makes were the dominant Packs alloc count left after the
// shard cache. Single-call (test/bench) sites pass a fresh &writeScratch{}
// so their behaviour is unchanged.
type writeScratch struct {
	out        []float32
	writeBuf   []byte
	rawRead    []byte
	valuesRead []float32
	// slerpWeights backs the two-element SLERP weight slice the cached
	// SLERP path computes per tensor. Returning a fresh []float64{1-t, t}
	// (or the sin-ratio pair) per tensor was the dominant SLERP-merge alloc
	// count — one heap slice per tensor, ~N for an N-tensor checkpoint.
	// writeSLERPChunksCached hands this array to slerpChunkedWeightsFromReaders
	// as its output buffer, and the weights are fully consumed by the
	// immediately-following writeLinearChunksUsing call before the next
	// tensor reuses the array, so a single shared backing array is safe.
	slerpWeights [2]float64
}

func writeLinearChunks(ctx context.Context, file *core.OSFile, refs []safetensors.TensorRef, weights []float64, chunkElements int) error {
	if len(refs) == 0 {
		return errNoTensors
	}
	if len(refs) != len(weights) {
		return errWeightsSourceCount
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	elements := refs[0].Elements
	for _, ref := range refs {
		if ref.Elements != elements {
			return errLinearLenMismatch
		}
	}
	readers, err := safetensors.OpenReaders(refs)
	if err != nil {
		return err
	}
	defer safetensors.CloseReaders(readers)
	return writeLinearChunksUsing(ctx, file, readers, elements, weights, chunkElements, &writeScratch{})
}

// writeLinearChunksCached is the shard-cache variant of writeLinearChunks
// used by the writeMergedSafetensors per-tensor loop. Instead of opening
// (and closing) the source shard files once per tensor, it borrows
// ReadAt-addressed readers over handles the pass-level cache opened once.
// The cache owns the *core.OSFile lifetimes, so the readers are never
// closed here. Decoding and writing are byte-identical to writeLinearChunks.
func writeLinearChunksCached(ctx context.Context, file *core.OSFile, cache *fileCache, refs []safetensors.TensorRef, weights []float64, chunkElements int, scratch *writeScratch) error {
	if len(refs) == 0 {
		return errNoTensors
	}
	if len(refs) != len(weights) {
		return errWeightsSourceCount
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	elements := refs[0].Elements
	for _, ref := range refs {
		if ref.Elements != elements {
			return errLinearLenMismatch
		}
	}
	readers, err := cache.readers(refs)
	if err != nil {
		return err
	}
	return writeLinearChunksUsing(ctx, file, readers, elements, weights, chunkElements, scratch)
}

// writeSLERPChunksCached is the shard-cache variant of writeSLERPChunks.
// Same readers-from-cache discipline as writeLinearChunksCached: the SLERP
// dot/norm scan and the merge write pass share cache-borrowed readers, so
// the two-source shards are opened once for the whole merge instead of once
// per tensor (and never twice per tensor). Output is byte-identical.
func writeSLERPChunksCached(ctx context.Context, file *core.OSFile, cache *fileCache, refs []safetensors.TensorRef, t float64, chunkElements int, scratch *writeScratch) error {
	if len(refs) != 2 {
		return errSLERPNeedTwoTensors
	}
	if refs[0].Elements != refs[1].Elements {
		return errSLERPLenMismatch
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	readers, err := cache.readers(refs)
	if err != nil {
		return err
	}
	weights, err := slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, scratch.slerpWeights[:])
	if err != nil {
		return err
	}
	return writeLinearChunksUsing(ctx, file, readers, refs[0].Elements, weights, chunkElements, scratch)
}

// writeLinearChunksUsing is the readers-already-open variant of
// writeLinearChunks. Pulled out so writeSLERPChunks can share the
// readers it opened for the SLERP weight scan instead of paying for a
// second OpenReaders / per-chunk-per-reader file read pass.
func writeLinearChunksUsing(ctx context.Context, file *core.OSFile, readers []safetensors.TensorReader, elements int, weights []float64, chunkElements int, scratch *writeScratch) error {
	// Reuse the out + write/decode buffers across chunks AND across tensors
	// (scratch is owned by the writeMergedSafetensors loop). out is sized to
	// the span we ever fill: the loop only writes out[:count] where count =
	// min(chunkElements, elements-offset), so a tensor smaller than one
	// chunk (the common case — modelMergeTensorChunkElements is 1<<20 but
	// most tensors hold far fewer elements) does not allocate a full
	// 1M-element (4 MiB) buffer. cap-checked reuse keeps the writes
	// byte-identical: out[i] is initialised (not accumulated) from source 0
	// each chunk, so any stale tail left by a larger previous tensor is
	// never read.
	bufLen := chunkElements
	if elements < bufLen {
		bufLen = elements
	}
	out := scratch.out
	if cap(out) < bufLen {
		out = make([]float32, bufLen)
		scratch.out = out
	}
	// rawRead + valuesRead are reused across every reader, chunk and tensor:
	// each reader's decoded chunk is folded into out immediately before the
	// next reader reads, so a single shared pair is safe. Replaces the
	// per-reader-per-chunk fresh []byte + []float32 that ReadFloat32Chunk
	// allocated.
	rawRead := scratch.rawRead
	valuesRead := scratch.valuesRead
	for offset := 0; offset < elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(chunkElements, elements-offset)
		out = out[:count]
		for sourceIndex, reader := range readers {
			var values []float32
			var err error
			rawRead, valuesRead, values, err = reader.ReadFloat32ChunkInto(offset, count, rawRead, valuesRead)
			if err != nil {
				scratch.rawRead, scratch.valuesRead = rawRead, valuesRead
				return err
			}
			// Cast weight to float32 once outside the inner accumulator
			// loop — same precision argument as linearMerge (the inputs
			// are float32, the weights are normalised in [0,1]).
			weight32 := float32(weights[sourceIndex])
			if sourceIndex == 0 {
				// Initialise out from the first source — saves the
				// zero-loop the previous form did before accumulating.
				for i, value := range values {
					out[i] = value * weight32
				}
			} else {
				for i, value := range values {
					out[i] += value * weight32
				}
			}
		}
		var err error
		scratch.writeBuf, err = writeFloat32ValuesScratch(file, out, scratch.writeBuf)
		if err != nil {
			scratch.rawRead, scratch.valuesRead = rawRead, valuesRead
			return err
		}
	}
	scratch.rawRead, scratch.valuesRead = rawRead, valuesRead
	return nil
}

func writeSLERPChunks(ctx context.Context, file *core.OSFile, refs []safetensors.TensorRef, t float64, chunkElements int) error {
	if len(refs) != 2 {
		return errSLERPNeedTwoTensors
	}
	if refs[0].Elements != refs[1].Elements {
		return errSLERPLenMismatch
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	// Open readers ONCE — previously the SLERP write path opened readers
	// twice (here for the dot/norm scan, then again inside
	// writeLinearChunks for the merge write). Sharing readers across the
	// two passes drops len(refs)*2 OpenReader allocs + 2x per-chunk
	// ReadFloat32Chunk file I/O.
	readers, err := safetensors.OpenReaders(refs)
	if err != nil {
		return err
	}
	defer safetensors.CloseReaders(readers)
	weights, err := slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, nil)
	if err != nil {
		return err
	}
	return writeLinearChunksUsing(ctx, file, readers, refs[0].Elements, weights, chunkElements, &writeScratch{})
}

func slerpChunkedWeights(ctx context.Context, refs []safetensors.TensorRef, t float64, chunkElements int) ([]float64, error) {
	if len(refs) != 2 {
		return nil, errSLERPNeedTwoTensors
	}
	if refs[0].Elements != refs[1].Elements {
		return nil, errSLERPLenMismatch
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	readers, err := safetensors.OpenReaders(refs)
	if err != nil {
		return nil, err
	}
	defer safetensors.CloseReaders(readers)
	return slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, nil)
}

// slerpChunkedWeightsFromReaders is the readers-already-open variant
// for the SLERP dot/norm scan. Lets writeSLERPChunks share readers
// across the SLERP weight scan and the writeLinearChunks pass.
//
// out is an optional two-element scratch the result is written into and
// returned as out[:2] — the cached per-tensor path passes a writeScratch
// field so the two-element weight slice isn't heap-allocated once per
// tensor. Pass nil (single-call sites) to get a freshly-allocated slice.
func slerpChunkedWeightsFromReaders(ctx context.Context, readers []safetensors.TensorReader, elements int, t float64, chunkElements int, out []float64) ([]float64, error) {
	if len(readers) != 2 {
		return nil, errSLERPNeedTwoReaders
	}
	if chunkElements <= 0 {
		chunkElements = modelMergeTensorChunkElements
	}
	var dot float64
	var normA float64
	var normB float64
	// a and b are live simultaneously each iteration, so each reader gets
	// its own raw + values scratch — reusing one pair across both would let
	// the b read clobber a's values mid-scan. Reused across chunks, this
	// drops the two fresh []byte + two fresh []float32 ReadFloat32Chunk
	// allocated per chunk down to a one-time grow.
	var rawA, rawB []byte
	var valuesA, valuesB []float32
	for offset := 0; offset < elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := min(chunkElements, elements-offset)
		var a, b []float32
		var err error
		rawA, valuesA, a, err = readers[0].ReadFloat32ChunkInto(offset, count, rawA, valuesA)
		if err != nil {
			return nil, err
		}
		rawB, valuesB, b, err = readers[1].ReadFloat32ChunkInto(offset, count, rawB, valuesB)
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
		return slerpWeightPair(out, 1-t, t), nil
	}
	cosTheta := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	cosTheta = clampFloat64(cosTheta, -1, 1)
	if math.Abs(cosTheta) > 0.9995 {
		return slerpWeightPair(out, 1-t, t), nil
	}
	theta := math.Acos(cosTheta)
	sinTheta := math.Sin(theta)
	return slerpWeightPair(out, math.Sin((1-t)*theta)/sinTheta, math.Sin(t*theta)/sinTheta), nil
}

// slerpWeightPair returns [a, b] written into out when out has room for
// two elements, otherwise into a freshly-allocated slice. The cached
// SLERP write path supplies a writeScratch-owned array so the two-element
// weight slice is not allocated per tensor.
func slerpWeightPair(out []float64, a, b float64) []float64 {
	if cap(out) < 2 {
		out = make([]float64, 2)
	}
	out = out[:2]
	out[0] = a
	out[1] = b
	return out
}

func mergeTensorValues(values [][]float32, method Method, t float64, weights []float64) ([]float32, error) {
	switch method {
	case MethodLinear:
		return linearMerge(values, weights)
	case MethodSLERP:
		return slerpMerge(values, t)
	default:
		return nil, core.NewError("mlx: unsupported model merge method: " + string(method))
	}
}

func linearMerge(values [][]float32, weights []float64) ([]float32, error) {
	if len(values) == 0 {
		return nil, errNoTensors
	}
	out := make([]float32, len(values[0]))
	for sourceIndex, source := range values {
		if len(source) != len(out) {
			return nil, errLinearLenMismatch
		}
		// Cast the weight to float32 once outside the inner loop —
		// previously every element did a float32->float64->mul->float32
		// round-trip. Linear merge weights are normalised in [0,1] so
		// float32 precision is sufficient (matches the source tensor
		// dtype anyway).
		weight32 := float32(weights[sourceIndex])
		for i, value := range source {
			out[i] += value * weight32
		}
	}
	return out, nil
}

func slerpMerge(values [][]float32, t float64) ([]float32, error) {
	if len(values) != 2 {
		return nil, errSLERPNeedTwoTensors
	}
	a := values[0]
	b := values[1]
	if len(a) != len(b) {
		return nil, errSLERPLenMismatch
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
		return linearMerge(values, []float64{1 - t, t})
	}
	cosTheta := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	cosTheta = clampFloat64(cosTheta, -1, 1)
	if math.Abs(cosTheta) > 0.9995 {
		return linearMerge(values, []float64{1 - t, t})
	}
	theta := math.Acos(cosTheta)
	sinTheta := math.Sin(theta)
	scaleA := math.Sin((1-t)*theta) / sinTheta
	scaleB := math.Sin(t*theta) / sinTheta
	return linearMerge(values, []float64{scaleA, scaleB})
}

func normalizedWeights(sources []Source) ([]float64, error) {
	weights := make([]float64, len(sources))
	var total float64
	var explicit bool
	for i, source := range sources {
		if math.IsNaN(source.Weight) || math.IsInf(source.Weight, 0) {
			return nil, errMergeWeightNotFinite
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
		return nil, errMergeWeightsSumZero
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights, nil
}

func writeFloat32Values(file *core.OSFile, values []float32) error {
	_, err := writeFloat32ValuesScratch(file, values, nil)
	return err
}

// writeFloat32ValuesScratch is the byte-buffer-reusing variant for the
// chunked write paths. The caller owns scratch so the same backing array
// is reused across chunks instead of one make per chunk. The returned
// slice (possibly the same as scratch) carries forward the now-grown
// capacity for the caller's next call. Pass nil for scratch on a single
// call site.
func writeFloat32ValuesScratch(file *core.OSFile, values []float32, scratch []byte) ([]byte, error) {
	needed := len(values) * 4
	if cap(scratch) < needed {
		scratch = make([]byte, needed)
	} else {
		scratch = scratch[:needed]
	}
	if needed > 0 {
		// Reinterpret-cast the source []float32 as bytes — float32 storage
		// is little-endian on both Go-supported architectures (arm64 and
		// amd64), so the byte view of a []float32 already matches what
		// binary.LittleEndian.PutUint32(buf, math.Float32bits(v)) writes
		// element-by-element. One memcpy vs N×(PutUint32 + Float32bits).
		// Pattern is established in go/kv/snapshot.go f32sRaw (~4.3× on
		// 2048-element runs) and go/pkg/metal/io_custom.go.
		src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), needed)
		copy(scratch, src)
	}
	_, err := file.Write(scratch)
	return scratch, err
}

func writeProvenance(path string, provenance Provenance) error {
	// core.SliceClone — exact-cap clone, avoids growslice over-allocation
	// from append([]string(nil), src...). Also takes the empty-slice fast
	// path internally so we don't waste an alloc on a typical merge with
	// no skipped tensors.
	sorted := core.SliceClone(provenance.SkippedTensors)
	sort.Strings(sorted)
	provenance.SkippedTensors = sorted
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("Packs", "marshal merge provenance", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("Packs", "write merge provenance", resultError(result))
	}
	return nil
}

// hasSuffixFold reports whether s ends with suffix using ASCII case
// folding. Suffix is required to be lowercase. Pure scan, no allocations —
// replaces the core.Lower(s) + core.HasSuffix pattern that always allocated
// a lowered copy of s regardless of input.
func hasSuffixFold(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	off := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		c := s[off+i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
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

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errCoreResultFailed
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

// samePathResolved is the per-source-loop variant where the right-hand
// side is already absolute. Saves a core.PathAbs call (and any associated
// filesystem inspection) per iteration.
func samePathResolved(a, absB string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	return absA == absB
}

// modelPackMetadataPatterns is the canonical pattern list — hoisted out
// of copyModelPackMetadata so the slice literal isn't rebuilt per call.
var modelPackMetadataPatterns = [...]string{"*.json", "*.model", "*.txt"}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	// Typical metadata footprint: config.json, tokenizer.json,
	// tokenizer_config.json, special_tokens_map.json, generation_config.json
	// — ~5-8 entries. Pre-size the seen map to skip the initial maphint
	// rebalances.
	seen := make(map[string]struct{}, 8)
	for _, pattern := range modelPackMetadataPatterns {
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
	// Two prior issues in this predicate:
	//
	// 1. core.Lower(name) allocated a fresh copy of every filename even
	//    though most metadata filenames are already lowercase.
	// 2. The Contains(".safetensors")|HasSuffix(".safetensors") pair is
	//    redundant — HasSuffix is a strict subset of Contains for the same
	//    suffix. Same for ".gguf". Drop the HasSuffix legs entirely.
	//
	// We keep the Contains semantics (legacy: filters anything *named*
	// with .safetensors in its path, e.g. .safetensors.index.json) by
	// using a case-folding containsFold helper.
	if equalFold(name, "adapter_provenance.json") {
		return true
	}
	return containsFold(name, ".safetensors") || containsFold(name, ".gguf")
}

// equalFold is len-prefixed ASCII case-insensitive equality. Zero allocations.
func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// containsFold reports whether s contains substr using ASCII case folding.
// substr is required to be lowercase. Zero allocations.
func containsFold(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(substr) > len(s) {
		return false
	}
	last := len(s) - len(substr)
outer:
	for i := 0; i <= last; i++ {
		for j := 0; j < len(substr); j++ {
			c := s[i+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != substr[j] {
				continue outer
			}
		}
		return true
	}
	return false
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
	return errPackMetadataCopy
}

func hashFile(path string) (string, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return "", resultError(read)
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return "", errReadNonByteData
	}
	return core.SHA256Hex(data), nil
}
