// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"sort"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/gguf"
)

// GGUFQuantizeFormat names the GGUF quantization format requested by the caller.
type GGUFQuantizeFormat string

const (
	GGUFQuantizeQ8_0   GGUFQuantizeFormat = "q8_0"
	GGUFQuantizeQ4_0   GGUFQuantizeFormat = "q4_0"
	GGUFQuantizeQ4_K_M GGUFQuantizeFormat = "q4_k_m"

	ggufQuantizeOutputWeights      = "model.gguf"
	ggufQuantizeChunkBlockElements = 32 << 15
)

// QuantizeGGUFOptions configures native Go safetensors-to-GGUF quantization.
type QuantizeGGUFOptions struct {
	ModelPath  string             `json:"model_path"`
	OutputPath string             `json:"output_path"`
	Format     GGUFQuantizeFormat `json:"format,omitempty"`
	Labels     map[string]string  `json:"labels,omitempty"`
}

// QuantizeGGUFResult reports the generated GGUF model pack.
type QuantizeGGUFResult struct {
	OutputPath       string             `json:"output_path"`
	WeightPath       string             `json:"weight_path"`
	RequestedFormat  GGUFQuantizeFormat `json:"requested_format"`
	Format           GGUFQuantizeFormat `json:"format"`
	SourcePack       mp.ModelPack          `json:"source_pack"`
	Pack             mp.ModelPack          `json:"pack"`
	Info             gguf.Info           `json:"info"`
	TensorCount      int                `json:"tensor_count"`
	QuantizedTensors int                `json:"quantized_tensors"`
	Notes            []string           `json:"notes,omitempty"`
}

type denseSafetensor struct {
	Name  string
	Shape []uint64
	Data  []float32
}

type safetensorHeaderEntry struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
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

// QuantizeModelPackToGGUF converts a dense safetensors model pack into a GGUF pack.
func QuantizeModelPackToGGUF(ctx context.Context, opts QuantizeGGUFOptions) (*QuantizeGGUFResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts.ModelPath == "" {
		return nil, core.NewError("mlx: source model path is required")
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

	source, err := ValidateModelPack(opts.ModelPath)
	if err != nil {
		return nil, core.E("QuantizeModelPackToGGUF", "validate source model pack", err)
	}
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
		return nil, core.E("QuantizeModelPackToGGUF", "create output directory", quantizeGGUFResultError(result))
	}
	if err := copyModelPackMetadata(source.Root, output); err != nil {
		return nil, err
	}

	index, err := indexSafetensorFiles(source.WeightFiles)
	if err != nil {
		return nil, core.E("QuantizeModelPackToGGUF", "index dense safetensors", err)
	}
	quantized, refs, err := buildStreamingGGUFQuantizedTensors(index, format)
	if err != nil {
		return nil, err
	}

	weightPath := core.PathJoin(output, ggufQuantizeOutputWeights)
	metadata := ggufQuantizeMetadata(source, format, opts.Labels)
	if err := writeQuantizedGGUFStream(ctx, weightPath, metadata, quantized, refs, format, ggufQuantizeChunkBlockElements); err != nil {
		return nil, core.E("QuantizeModelPackToGGUF", "write GGUF", err)
	}

	info, err := gguf.ReadInfo(weightPath)
	if err != nil {
		return nil, core.E("QuantizeModelPackToGGUF", "read generated GGUF", err)
	}
	if !info.Valid() {
		return nil, core.NewError("mlx: generated GGUF failed metadata validation: " + ggufValidationSummary(info.ValidationIssues))
	}
	pack, err := ValidateModelPack(output)
	if err != nil {
		return nil, core.E("QuantizeModelPackToGGUF", "validate generated model pack", err)
	}

	return &QuantizeGGUFResult{
		OutputPath:       output,
		WeightPath:       weightPath,
		RequestedFormat:  requested,
		Format:           format,
		SourcePack:       source,
		Pack:             pack,
		Info:             info,
		TensorCount:      len(quantized),
		QuantizedTensors: len(quantized),
		Notes:            notes,
	}, nil
}

func resolveGGUFQuantizeFormat(format GGUFQuantizeFormat) (requested, used GGUFQuantizeFormat, notes []string, err error) {
	if format == "" {
		format = GGUFQuantizeQ8_0
	}
	normalized := GGUFQuantizeFormat(gguf.NormalizeQuantType(string(format)))
	switch normalized {
	case GGUFQuantizeQ8_0:
		return normalized, GGUFQuantizeQ8_0, nil, nil
	case GGUFQuantizeQ4_0:
		return normalized, GGUFQuantizeQ4_0, nil, nil
	case GGUFQuantizeQ4_K_M:
		return normalized, GGUFQuantizeQ4_0, []string{"q4_k_m writing is not implemented yet; emitted q4_0 as the closest native Go 4-bit GGUF format"}, nil
	default:
		return normalized, "", nil, core.NewError("mlx: unsupported GGUF quantization format: " + string(format))
	}
}

func ensureEmptyGGUFQuantizeDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("QuantizeModelPackToGGUF", "inspect output path", quantizeGGUFResultError(stat))
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
	read := core.ReadFile(path)
	if !read.OK {
		return nil, quantizeGGUFResultError(read)
	}
	data := read.Value.([]byte)
	if len(data) < 8 {
		return nil, core.NewError("mlx: safetensors file is too small: " + path)
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	headerStart := 8
	headerEnd := headerStart + int(headerLen)
	if headerLen > uint64(len(data)-8) || headerEnd > len(data) {
		return nil, core.NewError("mlx: safetensors header exceeds file size: " + path)
	}
	var header map[string]safetensorHeaderEntry
	if result := core.JSONUnmarshal(data[headerStart:headerEnd], &header); !result.OK {
		return nil, quantizeGGUFResultError(result)
	}
	tensors := make([]denseSafetensor, 0, len(header))
	for name, entry := range header {
		if name == "__metadata__" {
			continue
		}
		tensor, err := decodeDenseSafetensor(path, name, entry, data[headerEnd:])
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, tensor)
	}
	return tensors, nil
}

func decodeDenseSafetensor(path, name string, entry safetensorHeaderEntry, payload []byte) (denseSafetensor, error) {
	if len(entry.DataOffsets) != 2 {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor has invalid data_offsets: " + name)
	}
	begin := entry.DataOffsets[0]
	end := entry.DataOffsets[1]
	if begin < 0 || end < begin || end > int64(len(payload)) {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor offsets exceed payload: " + name)
	}
	shape := make([]uint64, 0, len(entry.Shape))
	elements := uint64(1)
	for _, dim := range entry.Shape {
		if dim <= 0 {
			return denseSafetensor{}, core.NewError("mlx: safetensors tensor has invalid shape: " + name)
		}
		shape = append(shape, uint64(dim))
		elements *= uint64(dim)
	}
	if len(shape) == 0 {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor shape is empty: " + name)
	}
	raw := payload[begin:end]
	values, err := decodeSafetensorFloatData(core.Upper(entry.DType), raw, int(elements))
	if err != nil {
		return denseSafetensor{}, core.E("QuantizeModelPackToGGUF", "decode "+path+" tensor "+name, err)
	}
	return denseSafetensor{Name: name, Shape: shape, Data: values}, nil
}

func decodeSafetensorFloatData(dtype string, raw []byte, elements int) ([]float32, error) {
	values := make([]float32, elements)
	switch dtype {
	case "F32":
		if len(raw) != elements*4 {
			return nil, core.NewError("F32 payload length does not match tensor shape")
		}
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		if len(raw) != elements*2 {
			return nil, core.NewError("F16 payload length does not match tensor shape")
		}
		for i := range values {
			values[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		if len(raw) != elements*2 {
			return nil, core.NewError("BF16 payload length does not match tensor shape")
		}
		for i := range values {
			values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case "F64":
		if len(raw) != elements*8 {
			return nil, core.NewError("F64 payload length does not match tensor shape")
		}
		for i := range values {
			values[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
	default:
		return nil, core.NewError("unsupported dense safetensors dtype: " + dtype)
	}
	return values, nil
}

func quantizeGGUFTensors(ctx context.Context, tensors []denseSafetensor, format GGUFQuantizeFormat) ([]ggufQuantizedTensor, error) {
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

func quantizeGGUFTensor(tensor denseSafetensor, format GGUFQuantizeFormat) (ggufQuantizedTensor, error) {
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
	case GGUFQuantizeQ8_0:
		data = quantizeQ8_0(tensor.Data)
	case GGUFQuantizeQ4_0:
		data = quantizeQ4_0(tensor.Data)
	}
	return ggufQuantizedTensor{
		Name:  tensor.Name,
		Type:  tensorType,
		Shape: append([]uint64(nil), tensor.Shape...),
		Data:  data,
	}, nil
}

func buildStreamingGGUFQuantizedTensors(index safetensorIndex, format GGUFQuantizeFormat) ([]ggufQuantizedTensor, []safetensorTensorRef, error) {
	tensorType, blockSize, bytesPerBlock, err := ggufQuantizeLayout(format)
	if err != nil {
		return nil, nil, err
	}
	tensors := make([]ggufQuantizedTensor, 0, len(index.Names))
	refs := make([]safetensorTensorRef, 0, len(index.Names))
	for _, name := range index.Names {
		ref := index.Tensors[name]
		if _, err := safetensorDTypeByteSize(ref.DType); err != nil {
			return nil, nil, err
		}
		if ref.Elements%blockSize != 0 {
			return nil, nil, core.NewError(core.Sprintf("mlx: tensor %s has %d values, not divisible by GGUF block size %d", ref.Name, ref.Elements, blockSize))
		}
		if len(ref.Shape) == 0 || ref.Shape[0]%uint64(blockSize) != 0 {
			return nil, nil, core.NewError(core.Sprintf("mlx: tensor %s first dimension is not divisible by GGUF block size %d", ref.Name, blockSize))
		}
		tensors = append(tensors, ggufQuantizedTensor{
			Name:  ref.Name,
			Type:  tensorType,
			Shape: append([]uint64(nil), ref.Shape...),
			Size:  uint64(ref.Elements/blockSize) * uint64(bytesPerBlock),
		})
		refs = append(refs, ref)
	}
	return tensors, refs, nil
}

func ggufQuantizeLayout(format GGUFQuantizeFormat) (tensorType uint32, blockSize int, bytesPerBlock int, err error) {
	switch format {
	case GGUFQuantizeQ8_0:
		return gguf.TensorTypeQ8_0, 32, 34, nil
	case GGUFQuantizeQ4_0:
		return gguf.TensorTypeQ4_0, 32, 18, nil
	default:
		return 0, 0, 0, core.NewError("mlx: unsupported resolved GGUF format: " + string(format))
	}
}

func quantizeQ8_0(values []float32) []byte {
	out := make([]byte, 0, len(values)/32*34)
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		out = appendUint16LE(out, float32ToFloat16(scale))
		for _, value := range block {
			var q int
			if scale != 0 {
				q = int(math.Round(float64(value / scale)))
			}
			q = clampInt(q, -127, 127)
			out = append(out, byte(int8(q)))
		}
	}
	return out
}

func quantizeQ4_0(values []float32) []byte {
	out := make([]byte, 0, len(values)/32*18)
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 7
		}
		out = appendUint16LE(out, float32ToFloat16(scale))
		packed := make([]byte, 16)
		for i, value := range block {
			var q int
			if scale != 0 {
				q = int(math.Round(float64(value/scale))) + 8
			}
			q = clampInt(q, 0, 15)
			if i < 16 {
				packed[i] = byte(q)
			} else {
				packed[i-16] |= byte(q << 4)
			}
		}
		out = append(out, packed...)
	}
	return out
}

func ggufQuantizeMetadata(source mp.ModelPack, format GGUFQuantizeFormat, labels map[string]string) []ggufMetadataEntry {
	fileType := uint32(7)
	quantizationType := string(GGUFQuantizeQ8_0)
	if format == GGUFQuantizeQ4_0 {
		fileType = 2
		quantizationType = string(GGUFQuantizeQ4_0)
	}
	architecture := source.Architecture
	metadata := []ggufMetadataEntry{
		{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: architecture},
		{Key: "general.file_type", ValueType: gguf.ValueTypeUint32, Value: fileType},
		{Key: "general.quantization_version", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: "general.quantization_type", ValueType: gguf.ValueTypeString, Value: quantizationType},
		{Key: "general.alignment", ValueType: gguf.ValueTypeUint32, Value: uint32(32)},
	}
	if source.VocabSize > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".vocab_size", ValueType: gguf.ValueTypeUint32, Value: uint32(source.VocabSize)})
	}
	if source.HiddenSize > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".embedding_length", ValueType: gguf.ValueTypeUint32, Value: uint32(source.HiddenSize)})
	}
	if source.NumLayers > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".block_count", ValueType: gguf.ValueTypeUint32, Value: uint32(source.NumLayers)})
	}
	if source.ContextLength > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(source.ContextLength)})
	}
	if len(labels) > 0 {
		keys := make([]string, 0, len(labels))
		for key := range labels {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			metadata = append(metadata, ggufMetadataEntry{Key: "go_mlx.label." + key, ValueType: gguf.ValueTypeString, Value: labels[key]})
		}
	}
	return metadata
}

func writeQuantizedGGUF(path string, metadata []ggufMetadataEntry, tensors []ggufQuantizedTensor) error {
	created := core.Create(path)
	if !created.OK {
		return quantizeGGUFResultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	assignGGUFTensorOffsets(tensors, 32)
	if err := writeQuantizedGGUFHeader(file, metadata, tensors); err != nil {
		return err
	}
	var written uint64
	for _, tensor := range tensors {
		if tensor.Offset < written {
			return core.NewError("mlx: GGUF tensor offsets are not monotonic")
		}
		if err := writePadding(file, tensor.Offset-written); err != nil {
			return err
		}
		if _, err := file.Write(tensor.Data); err != nil {
			return err
		}
		written = tensor.Offset + ggufQuantizedTensorDataSize(tensor)
	}
	return nil
}

func writeQuantizedGGUFStream(ctx context.Context, path string, metadata []ggufMetadataEntry, tensors []ggufQuantizedTensor, refs []safetensorTensorRef, format GGUFQuantizeFormat, chunkElements int) error {
	if len(tensors) != len(refs) {
		return core.NewError("mlx: GGUF tensor metadata and source refs are not aligned")
	}
	_, blockSize, _, err := ggufQuantizeLayout(format)
	if err != nil {
		return err
	}
	if chunkElements <= 0 {
		chunkElements = ggufQuantizeChunkBlockElements
	}
	chunkElements = (chunkElements / blockSize) * blockSize
	if chunkElements <= 0 {
		chunkElements = blockSize
	}

	created := core.Create(path)
	if !created.OK {
		return quantizeGGUFResultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	assignGGUFTensorOffsets(tensors, 32)
	if err := writeQuantizedGGUFHeader(file, metadata, tensors); err != nil {
		return err
	}
	var written uint64
	for i, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if tensor.Offset < written {
			return core.NewError("mlx: GGUF tensor offsets are not monotonic")
		}
		if err := writePadding(file, tensor.Offset-written); err != nil {
			return err
		}
		dataSize, err := writeQuantizedGGUFTensorStream(ctx, file, refs[i], format, chunkElements)
		if err != nil {
			return err
		}
		if dataSize != ggufQuantizedTensorDataSize(tensor) {
			return core.NewError(core.Sprintf("mlx: streamed GGUF tensor %s wrote %d bytes, want %d", tensor.Name, dataSize, ggufQuantizedTensorDataSize(tensor)))
		}
		written = tensor.Offset + ggufQuantizedTensorDataSize(tensor)
	}
	return nil
}

func writeQuantizedGGUFHeader(file *core.OSFile, metadata []ggufMetadataEntry, tensors []ggufQuantizedTensor) error {
	write := func(value any) error {
		return binary.Write(file, binary.LittleEndian, value)
	}
	if _, err := file.Write([]byte("GGUF")); err != nil {
		return err
	}
	if err := write(uint32(3)); err != nil {
		return err
	}
	if err := write(uint64(len(tensors))); err != nil {
		return err
	}
	if err := write(uint64(len(metadata))); err != nil {
		return err
	}
	for _, entry := range metadata {
		if err := writeGGUFMetadataEntry(file, entry); err != nil {
			return err
		}
	}
	for _, tensor := range tensors {
		if err := writeGGUFTensorInfo(file, tensor); err != nil {
			return err
		}
	}
	position, err := file.Seek(0, 1)
	if err != nil {
		return err
	}
	if err := writePadding(file, alignPadding(uint64(position), 32)); err != nil {
		return err
	}
	return nil
}

func writeQuantizedGGUFTensorStream(ctx context.Context, file *core.OSFile, ref safetensorTensorRef, format GGUFQuantizeFormat, chunkElements int) (uint64, error) {
	reader, err := openSafetensorTensorReader(ref)
	if err != nil {
		return 0, err
	}
	defer reader.close()
	var written uint64
	for offset := 0; offset < ref.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		count := min(chunkElements, ref.Elements-offset)
		values, err := reader.readFloat32Chunk(offset, count)
		if err != nil {
			return written, err
		}
		data, err := quantizeGGUFValues(format, values)
		if err != nil {
			return written, err
		}
		if _, err := file.Write(data); err != nil {
			return written, err
		}
		written += uint64(len(data))
	}
	return written, nil
}

func quantizeGGUFValues(format GGUFQuantizeFormat, values []float32) ([]byte, error) {
	switch format {
	case GGUFQuantizeQ8_0:
		return quantizeQ8_0(values), nil
	case GGUFQuantizeQ4_0:
		return quantizeQ4_0(values), nil
	default:
		return nil, core.NewError("mlx: unsupported resolved GGUF format: " + string(format))
	}
}

func assignGGUFTensorOffsets(tensors []ggufQuantizedTensor, alignment uint64) {
	var offset uint64
	for i := range tensors {
		offset += alignPadding(offset, alignment)
		tensors[i].Offset = offset
		offset += ggufQuantizedTensorDataSize(tensors[i])
	}
}

func ggufQuantizedTensorDataSize(tensor ggufQuantizedTensor) uint64 {
	if tensor.Size > 0 {
		return tensor.Size
	}
	return uint64(len(tensor.Data))
}

func writeGGUFMetadataEntry(file *core.OSFile, entry ggufMetadataEntry) error {
	if err := writeGGUFStringValue(file, entry.Key); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, entry.ValueType); err != nil {
		return err
	}
	return writeGGUFMetadataValue(file, entry.ValueType, entry.Value)
}

func writeGGUFMetadataValue(file *core.OSFile, valueType uint32, value any) error {
	switch valueType {
	case gguf.ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			return core.NewError("mlx: GGUF metadata value is not a string")
		}
		return writeGGUFStringValue(file, stringValue)
	case gguf.ValueTypeUint32:
		switch concrete := value.(type) {
		case uint32:
			return binary.Write(file, binary.LittleEndian, concrete)
		case int:
			return binary.Write(file, binary.LittleEndian, uint32(concrete))
		default:
			return core.NewError("mlx: GGUF metadata value is not uint32")
		}
	default:
		return core.NewError(core.Sprintf("mlx: unsupported GGUF metadata write type %d", valueType))
	}
}

func writeGGUFTensorInfo(file *core.OSFile, tensor ggufQuantizedTensor) error {
	if err := writeGGUFStringValue(file, tensor.Name); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(len(tensor.Shape))); err != nil {
		return err
	}
	for _, dim := range tensor.Shape {
		if err := binary.Write(file, binary.LittleEndian, dim); err != nil {
			return err
		}
	}
	if err := binary.Write(file, binary.LittleEndian, tensor.Type); err != nil {
		return err
	}
	return binary.Write(file, binary.LittleEndian, tensor.Offset)
}

func writeGGUFStringValue(file *core.OSFile, value string) error {
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		return err
	}
	_, err := file.Write([]byte(value))
	return err
}

func writePadding(file *core.OSFile, n uint64) error {
	const chunkSize = 32 * 1024
	var zeros [chunkSize]byte
	for n > 0 {
		size := uint64(chunkSize)
		if n < size {
			size = n
		}
		if _, err := file.Write(zeros[:size]); err != nil {
			return err
		}
		n -= size
	}
	return nil
}

func alignPadding(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return 0
	}
	return (alignment - (offset % alignment)) % alignment
}

func maxAbsFloat32(values []float32) float32 {
	var maxAbs float32
	for _, value := range values {
		abs := float32(math.Abs(float64(value)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

func appendUint16LE(out []byte, value uint16) []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], value)
	return append(out, buf[:]...)
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func float16ToFloat32(value uint16) float32 {
	sign := uint32(value>>15) & 0x1
	exp := int((value >> 10) & 0x1f)
	frac := uint32(value & 0x03ff)
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03ff
	} else if exp == 31 {
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}

func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 255 {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(frac >> shift)
		if (frac>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(frac>>13)
	if frac&0x00001000 != 0 {
		half++
	}
	return half
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
