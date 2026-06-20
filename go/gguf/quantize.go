// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"sort"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// QuantizeFormat names the GGUF quantization format requested by the caller.
type QuantizeFormat string

const (
	QuantizeQ8_0   QuantizeFormat = "q8_0"
	QuantizeQ4_0   QuantizeFormat = "q4_0"
	QuantizeQ5_0   QuantizeFormat = "q5_0"
	QuantizeQ4_K_M QuantizeFormat = "q4_k_m"
	QuantizeQ4_K   QuantizeFormat = "q4_k"
	QuantizeQ5_K   QuantizeFormat = "q5_k"
	QuantizeQ6_K   QuantizeFormat = "q6_k"
	QuantizeQ8_K   QuantizeFormat = "q8_k"
	QuantizeQ3_K   QuantizeFormat = "q3_k"
	QuantizeQ2_K   QuantizeFormat = "q2_k"

	ggufQuantizeOutputWeights      = "model.gguf"
	ggufQuantizeChunkBlockElements = 32 << 15
)

// QuantizeOptions configures native Go safetensors-to-GGUF quantization.
//
// SourcePack must be a validated safetensors-format model pack; callers
// validate via mlx.ValidateModelPack before invoking gguf.QuantizeModelPack.
// This shape keeps the gguf package free of the mlx-root cycle.
type QuantizeOptions struct {
	SourcePack mp.ModelPack      `json:"source_pack"`
	OutputPath string            `json:"output_path"`
	Format     QuantizeFormat    `json:"format,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// QuantizeResult reports the paths of the generated GGUF model pack and
// its metadata. Callers re-validate via mlx.ValidateModelPack(OutputPath)
// when they need a populated pack.ModelPack for downstream use.
type QuantizeResult struct {
	OutputPath       string         `json:"output_path"`
	WeightPath       string         `json:"weight_path"`
	RequestedFormat  QuantizeFormat `json:"requested_format"`
	Format           QuantizeFormat `json:"format"`
	SourcePack       mp.ModelPack   `json:"source_pack"`
	Info             Info           `json:"info"`
	TensorCount      int            `json:"tensor_count"`
	QuantizedTensors int            `json:"quantized_tensors"`
	Notes            []string       `json:"notes,omitempty"`
}

type denseSafetensor struct {
	Name  string
	Shape []uint64
	Data  []float32
}

type ggufQuantizedTensor struct {
	Name   string
	Type   uint32
	Shape  []uint64
	Offset uint64
	Size   uint64
	Data   []byte
}

type ggufMetadataEntry struct {
	Key       string
	ValueType uint32
	Value     any
}

// QuantizeModelPack converts a dense safetensors model pack into a GGUF pack.
func QuantizeModelPack(ctx context.Context, opts QuantizeOptions) (*QuantizeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts.SourcePack.Root == "" {
		return nil, core.NewError("mlx: source pack is required")
	}
	if opts.OutputPath == "" {
		return nil, core.NewError("mlx: GGUF output path is required")
	}
	if core.HasSuffix(core.Lower(opts.OutputPath), ".gguf") || core.HasSuffix(core.Lower(opts.OutputPath), ".safetensors") {
		return nil, core.NewError("mlx: GGUF output path must be a model-pack directory")
	}

	requested, format, notes, err := resolveGGUFQuantizeFormat(opts.Format)
	if err != nil {
		return nil, err
	}

	source := opts.SourcePack
	if source.Format != mp.ModelPackFormatSafetensors {
		return nil, core.NewError("mlx: GGUF quantization currently requires dense safetensors source weights")
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(source.Root, output) {
		return nil, core.NewError("mlx: GGUF output path must differ from source model path")
	}
	if err := ensureEmptyGGUFQuantizeDestination(output); err != nil {
		return nil, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return nil, core.E("QuantizeModelPack", "create output directory", quantizeGGUFResultError(result))
	}
	if err := copyModelPackMetadata(source.Root, output); err != nil {
		return nil, err
	}

	index, err := safetensors.IndexFiles(source.WeightFiles)
	if err != nil {
		return nil, core.E("QuantizeModelPack", "index dense safetensors", err)
	}
	quantized, refs, err := buildStreamingGGUFQuantizedTensors(index, format)
	if err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(output, ggufQuantizeOutputWeights)
	metadata := ggufQuantizeMetadata(source, format, opts.Labels)
	if err := writeQuantizedGGUFStream(ctx, weightPath, metadata, quantized, refs, format, ggufQuantizeChunkBlockElements); err != nil {
		return nil, core.E("QuantizeModelPack", "write GGUF", err)
	}

	info, err := ReadInfo(weightPath)
	if err != nil {
		return nil, core.E("QuantizeModelPack", "read generated GGUF", err)
	}
	if !info.Valid() {
		return nil, core.NewError("mlx: generated GGUF failed metadata validation: " + ValidationSummary(info.ValidationIssues))
	}

	return &QuantizeResult{
		OutputPath:       output,
		WeightPath:       weightPath,
		RequestedFormat:  requested,
		Format:           format,
		SourcePack:       source,
		Info:             info,
		TensorCount:      len(quantized),
		QuantizedTensors: len(quantized),
		Notes:            notes,
	}, nil
}

func resolveGGUFQuantizeFormat(format QuantizeFormat) (requested, used QuantizeFormat, notes []string, err error) {
	if format == "" {
		format = QuantizeQ8_0
	}
	normalized := QuantizeFormat(NormalizeQuantType(string(format)))
	switch normalized {
	case QuantizeQ8_0:
		return normalized, QuantizeQ8_0, nil, nil
	case QuantizeQ4_0:
		return normalized, QuantizeQ4_0, nil, nil
	case QuantizeQ5_0:
		return normalized, QuantizeQ5_0, nil, nil
	case QuantizeQ4_K_M:
		return normalized, QuantizeQ4_K, nil, nil
	case QuantizeQ4_K:
		return normalized, QuantizeQ4_K, nil, nil
	case QuantizeQ5_K:
		return normalized, QuantizeQ5_K, nil, nil
	case QuantizeQ6_K:
		return normalized, QuantizeQ6_K, nil, nil
	case QuantizeQ8_K:
		return normalized, QuantizeQ8_K, nil, nil
	case QuantizeQ3_K:
		return normalized, QuantizeQ3_K, nil, nil
	case QuantizeQ2_K:
		return normalized, QuantizeQ2_K, nil, nil
	default:
		return normalized, "", nil, core.NewError("mlx: unsupported GGUF quantization format: " + string(format))
	}
}

func ensureEmptyGGUFQuantizeDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("QuantizeModelPack", "inspect output path", quantizeGGUFResultError(stat))
	}
	weights := append(core.PathGlob(core.PathJoin(output, "*.safetensors")), core.PathGlob(core.PathJoin(output, "*.gguf"))...)
	if len(weights) > 0 {
		return core.NewError("mlx: GGUF output path already contains model weights")
	}
	return nil
}

func loadDenseSafetensors(paths []string) ([]denseSafetensor, error) {
	if len(paths) == 0 {
		return nil, core.NewError("mlx: no safetensors weight files available")
	}
	var out []denseSafetensor
	seen := map[string]struct{}{}
	for _, path := range paths {
		tensors, err := readDenseSafetensors(path)
		if err != nil {
			return nil, err
		}
		for _, tensor := range tensors {
			if _, ok := seen[tensor.Name]; ok {
				return nil, core.NewError("mlx: duplicate tensor in safetensors shards: " + tensor.Name)
			}
			seen[tensor.Name] = struct{}{}
			out = append(out, tensor)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func readDenseSafetensors(path string) ([]denseSafetensor, error) {
	// Read only the header — ReadIndex opens the file, reads the 8-byte
	// length prefix + header bytes, and hand-rolls the JSON parse (W8-I +
	// W8-K: interned dtype strings, one shared shape slab, ~1 alloc per
	// tensor). It does NOT read tensor payloads, so the whole-file
	// core.ReadFile that previously dominated this path's bytes (a GB
	// shard's payload held resident just to slice per-tensor windows) is
	// gone — the payload is streamed per tensor below via ReadAt.
	index, err := safetensors.ReadIndex(path)
	if err != nil {
		return nil, err
	}
	open := core.Open(path)
	if !open.OK {
		return nil, quantizeGGUFResultError(open)
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()

	// Validate every tensor window against the real file size BEFORE
	// sizing the scratch. ParseHeaderRefs cannot bound ByteLen — it only
	// sees the header bytes, not the file — so a corrupt header declaring
	// a wild ByteLen must be rejected here (one fstat) rather than letting
	// make([]byte, maxByteLen) attempt a giant allocation. This restores
	// the old whole-file path's `end > len(data)` check exactly (against
	// fileSize) with the same error, so malformed-input behaviour is
	// identical. maxByteLen is found in the same pass so the reused scratch
	// is sized to the largest valid tensor.
	stat, statErr := file.Stat()
	if statErr != nil {
		return nil, core.E("QuantizeModelPack", "stat "+path, statErr)
	}
	fileSize := stat.Size()
	var maxByteLen int64
	for _, name := range index.Names {
		ref := index.Tensors[name]
		// end < DataStart catches int64 overflow on a corrupt-huge ByteLen
		// (DataStart+ByteLen wrapping negative would otherwise pass the
		// > fileSize arm and reach make([]byte, ByteLen)) — mirrors the
		// old whole-file path's three-armed bound exactly.
		end := ref.DataStart + ref.ByteLen
		if ref.DataStart < 0 || ref.ByteLen < 0 || end < ref.DataStart || end > fileSize {
			return nil, core.NewError("mlx: safetensors tensor offsets exceed payload: " + ref.Name)
		}
		if ref.ByteLen > maxByteLen {
			maxByteLen = ref.ByteLen
		}
	}

	// One reused raw-byte scratch, sized to the largest tensor's on-disk
	// payload, replaces the whole-file buffer: cumulative B/op drops from
	// (filesize + Σdecoded) to (max_tensor_bytes + Σdecoded). The raw bytes
	// are transient — fully overwritten by each ReadAt and consumed by
	// DecodeFloatData before the next read — so a single buffer is safe.
	// The DECODED []float32 must stay fresh per tensor (each is retained in
	// the returned slice), so DecodeFloatData allocates a new output every
	// call, byte-identical to the prior whole-file slice path.
	raw := make([]byte, maxByteLen)
	tensors := make([]denseSafetensor, 0, len(index.Tensors))
	for _, name := range index.Names {
		ref := index.Tensors[name]
		buf := raw[:ref.ByteLen]
		if _, err := file.ReadAt(buf, ref.DataStart); err != nil {
			return nil, core.E("QuantizeModelPack", "read "+ref.Path+" tensor "+ref.Name, err)
		}
		values, err := safetensors.DecodeFloatData(ref.DType, buf, ref.Elements)
		if err != nil {
			return nil, core.E("QuantizeModelPack", "decode "+ref.Path+" tensor "+ref.Name, err)
		}
		tensors = append(tensors, denseSafetensor{Name: ref.Name, Shape: ref.Shape, Data: values})
	}
	return tensors, nil
}

func decodeDenseSafetensor(path, name string, entry safetensors.HeaderEntry, payload []byte) (denseSafetensor, error) {
	if len(entry.DataOffsets) != 2 {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor has invalid data_offsets: " + name)
	}
	begin := entry.DataOffsets[0]
	end := entry.DataOffsets[1]
	if begin < 0 || end < begin || end > int64(len(payload)) {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor offsets exceed payload: " + name)
	}
	if len(entry.Shape) == 0 {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor shape is empty: " + name)
	}
	shape := make([]uint64, len(entry.Shape))
	elements := uint64(1)
	for i, dim := range entry.Shape {
		if dim <= 0 {
			return denseSafetensor{}, core.NewError("mlx: safetensors tensor has invalid shape: " + name)
		}
		shape[i] = uint64(dim)
		elements *= uint64(dim)
	}
	raw := payload[begin:end]
	values, err := safetensors.DecodeFloatData(core.Upper(entry.DType), raw, int(elements))
	if err != nil {
		return denseSafetensor{}, core.E("QuantizeModelPack", "decode "+path+" tensor "+name, err)
	}
	return denseSafetensor{Name: name, Shape: shape, Data: values}, nil
}

func quantizeGGUFTensors(ctx context.Context, tensors []denseSafetensor, format QuantizeFormat) ([]ggufQuantizedTensor, error) {
	out := make([]ggufQuantizedTensor, 0, len(tensors))
	for _, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		quantized, err := quantizeGGUFTensor(tensor, format)
		if err != nil {
			return nil, err
		}
		out = append(out, quantized)
	}
	return out, nil
}

func quantizeGGUFTensor(tensor denseSafetensor, format QuantizeFormat) (ggufQuantizedTensor, error) {
	tensorType, blockSize, _, err := ggufQuantizeLayout(format)
	if err != nil {
		return ggufQuantizedTensor{}, err
	}
	if len(tensor.Data)%blockSize != 0 {
		return ggufQuantizedTensor{}, core.NewError(core.Sprintf("mlx: tensor %s has %d values, not divisible by GGUF block size %d", tensor.Name, len(tensor.Data), blockSize))
	}
	if len(tensor.Shape) == 0 || tensor.Shape[0]%uint64(blockSize) != 0 {
		return ggufQuantizedTensor{}, core.NewError(core.Sprintf("mlx: tensor %s first dimension is not divisible by GGUF block size %d", tensor.Name, blockSize))
	}
	var data []byte
	switch format {
	case QuantizeQ8_0:
		data = quantizeQ8_0(tensor.Data)
	case QuantizeQ4_0:
		data = quantizeQ4_0(tensor.Data)
	case QuantizeQ5_0:
		data = quantizeQ5_0(tensor.Data)
	case QuantizeQ4_K:
		data = quantizeQ4_K(tensor.Data)
	case QuantizeQ5_K:
		data = quantizeQ5_K(tensor.Data)
	case QuantizeQ6_K:
		data = quantizeQ6_K(tensor.Data)
	case QuantizeQ8_K:
		data = quantizeQ8_K(tensor.Data)
	case QuantizeQ3_K:
		data = quantizeQ3_K(tensor.Data)
	case QuantizeQ2_K:
		data = quantizeQ2_K(tensor.Data)
	}
	return ggufQuantizedTensor{
		Name:  tensor.Name,
		Type:  tensorType,
		Shape: core.SliceClone(tensor.Shape),
		Data:  data,
	}, nil
}

func buildStreamingGGUFQuantizedTensors(index safetensors.Index, format QuantizeFormat) ([]ggufQuantizedTensor, []safetensors.TensorRef, error) {
	tensorType, blockSize, bytesPerBlock, err := ggufQuantizeLayout(format)
	if err != nil {
		return nil, nil, err
	}
	tensors := make([]ggufQuantizedTensor, 0, len(index.Names))
	refs := make([]safetensors.TensorRef, 0, len(index.Names))
	for _, name := range index.Names {
		ref := index.Tensors[name]
		if _, err := safetensors.DTypeByteSize(ref.DType); err != nil {
			return nil, nil, err
		}
		if ref.Elements%blockSize != 0 {
			return nil, nil, core.NewError(core.Sprintf("mlx: tensor %s has %d values, not divisible by GGUF block size %d", ref.Name, ref.Elements, blockSize))
		}
		if len(ref.Shape) == 0 || ref.Shape[0]%uint64(blockSize) != 0 {
			return nil, nil, core.NewError(core.Sprintf("mlx: tensor %s first dimension is not divisible by GGUF block size %d", ref.Name, blockSize))
		}
		// Alias ref.Shape rather than cloning: the Index's shape slab is
		// allocated fresh per IndexFiles/ParseHeaderRefs call (one
		// make([]uint64, 0, totalDims) carved into per-tensor subslices,
		// never pooled or shared across calls), and nothing downstream
		// mutates ggufQuantizedTensor.Shape — assignGGUFTensorOffsets writes
		// only .Offset, and writeGGUFTensorInfo reads the dims without
		// appending. Dropping the per-tensor clone removes one alloc per
		// tensor (scales with model size) byte-identically.
		tensors = append(tensors, ggufQuantizedTensor{
			Name:  ref.Name,
			Type:  tensorType,
			Shape: ref.Shape,
			Size:  uint64(ref.Elements/blockSize) * uint64(bytesPerBlock),
		})
		refs = append(refs, ref)
	}
	return tensors, refs, nil
}

func ggufQuantizeLayout(format QuantizeFormat) (tensorType uint32, blockSize int, bytesPerBlock int, err error) {
	switch format {
	case QuantizeQ8_0:
		return TensorTypeQ8_0, 32, 34, nil
	case QuantizeQ4_0:
		return TensorTypeQ4_0, 32, 18, nil
	case QuantizeQ5_0:
		return ggufTensorTypeQ5_0, 32, 24, nil
	case QuantizeQ4_K:
		return ggufTensorTypeQ4K, 256, 144, nil
	case QuantizeQ5_K:
		return ggufTensorTypeQ5K, 256, 176, nil
	case QuantizeQ6_K:
		return ggufTensorTypeQ6K, 256, 210, nil
	case QuantizeQ8_K:
		// Canonical block_q8_K: float32 d + 256 int8 qs + 16 int16 bsums.
		return ggufTensorTypeQ8K, 256, 292, nil
	case QuantizeQ3_K:
		return ggufTensorTypeQ3K, 256, 110, nil
	case QuantizeQ2_K:
		// Canonical block_q2_K is 84 (16 scales + 64 qs + f16 d + f16
		// dmin). The gguflib type-size table's 82 drops dmin; its decoder
		// nonetheless advances 84, and upstream static_assert is 84.
		return ggufTensorTypeQ2K, 256, 84, nil
	default:
		return 0, 0, 0, core.NewError("mlx: unsupported resolved GGUF format: " + string(format))
	}
}

func quantizeGGUFResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

// ValidationSummary joins GGUF validation issue codes into a human-readable
// string. Used by callers that report failures from the gguf validation path.
//
//	msg := gguf.ValidationSummary(info.ValidationIssues)
func ValidationSummary(issues []ValidationIssue) string {
	if len(issues) == 0 {
		return "unknown validation failure"
	}
	parts := make([]string, 0, len(issues))
	for _, issue := range issues {
		if issue.Tensor != "" {
			parts = append(parts, core.Concat(issue.Code, ":", issue.Tensor))
			continue
		}
		parts = append(parts, issue.Code)
	}
	return core.Join(", ", parts...)
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
			if err := copyLocalFile(sourcePath, core.PathJoin(outputRoot, name)); err != nil {
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

// metadataCopyStreamThreshold is the file size at or below which copyLocalFile
// reads the whole file into one buffer and writes it back (core.ReadFile +
// core.WriteFile), and above which it streams source→destination through
// core.Copy's fixed staging buffer. Below ~128 KiB a single read/write is the
// cheaper path — the slurp buffer is small and a dedicated copy buffer would
// cost more than the read it replaces (trap #5's small-file caveat). Above it
// the slurp is a large transient buffer the size of the whole file
// (tokenizer.json is multiple MB on real checkpoints), so streaming wins on
// B/op without changing a copied byte.
const metadataCopyStreamThreshold = 128 << 10

func copyLocalFile(sourcePath, destinationPath string) error {
	// Size-gate: small files take the direct read/write (byte- and
	// mode-identical to the historical core.ReadFile + core.WriteFile);
	// large files stream. A failed/absent stat falls through to the direct
	// read, whose own failure surfaces the real error — never silently skip.
	if stat := core.Stat(sourcePath); stat.OK {
		if info, ok := stat.Value.(core.FsFileInfo); ok && info.Size() > metadataCopyStreamThreshold {
			return streamLocalFile(sourcePath, destinationPath)
		}
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return quantizeGGUFResultError(read)
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return quantizeGGUFResultError(result)
	}
	return nil
}

// streamLocalFile copies source→destination through core.Copy (io.Copy's
// fixed ~32 KiB staging buffer, or the kernel copy fast-path between two
// *os.File handles) instead of slurping the whole file into a heap []byte.
// The destination is opened with the same O_WRONLY|O_CREATE|O_TRUNC flags and
// 0o644 mode core.WriteFile used, so the written bytes and file mode are
// identical to the direct path. Mirrors merge.copyModelPackLocalFile.
func streamLocalFile(sourcePath, destinationPath string) error {
	srcOpen := core.Open(sourcePath)
	if !srcOpen.OK {
		return quantizeGGUFResultError(srcOpen)
	}
	src := srcOpen.Value.(*core.OSFile)
	defer src.Close()
	dstOpen := core.OpenFile(destinationPath, core.O_WRONLY|core.O_CREATE|core.O_TRUNC, 0o644)
	if !dstOpen.OK {
		return quantizeGGUFResultError(dstOpen)
	}
	dst := dstOpen.Value.(*core.OSFile)
	if result := core.Copy(dst, src); !result.OK {
		// The copy already failed; close the partial destination best-effort
		// and surface the copy error, not the close error.
		dst.Close()
		return quantizeGGUFResultError(result)
	}
	if err := dst.Close(); err != nil {
		return err
	}
	return nil
}
