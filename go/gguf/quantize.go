// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"sync"

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
	// Delegate header parsing to the shared safetensors walker (W8-I + W8-K).
	// It hand-rolls the JSON parse, interns canonical dtype strings, and
	// carves all Shape slices out of one slab so per-tensor cost lands at
	// ~1 alloc once the arena is in scope — replacing the reflection-driven
	// map[string]HeaderEntry decode that previously dominated this path's
	// allocations. dataStart is the absolute offset of the first payload
	// byte in `data` (i.e. headerEnd), which is what ParseHeaderRefs uses
	// as the base for each TensorRef.DataStart.
	index, err := safetensors.ParseHeaderRefs(path, data[headerStart:headerEnd], int64(headerEnd))
	if err != nil {
		return nil, err
	}
	tensors := make([]denseSafetensor, 0, len(index.Tensors))
	for _, name := range index.Names {
		tensor, err := decodeDenseSafetensorRef(index.Tensors[name], data)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, tensor)
	}
	return tensors, nil
}

// decodeDenseSafetensorRef is the TensorRef-shaped sibling of
// decodeDenseSafetensor. The shared safetensors walker emits one
// TensorRef per tensor with Shape pre-validated and DType pre-uppercased,
// so this path skips the per-entry validation that the HeaderEntry
// variant has to do (handled inside ParseHeaderRefs / refFromHeaderSlab).
// data is the whole-file byte slice; the payload window is sliced via
// the TensorRef's absolute DataStart + ByteLen.
func decodeDenseSafetensorRef(ref safetensors.TensorRef, data []byte) (denseSafetensor, error) {
	end := ref.DataStart + ref.ByteLen
	if ref.DataStart < 0 || end < ref.DataStart || end > int64(len(data)) {
		return denseSafetensor{}, core.NewError("mlx: safetensors tensor offsets exceed payload: " + ref.Name)
	}
	raw := data[ref.DataStart:end]
	values, err := safetensors.DecodeFloatData(ref.DType, raw, ref.Elements)
	if err != nil {
		return denseSafetensor{}, core.E("QuantizeModelPack", "decode "+ref.Path+" tensor "+ref.Name, err)
	}
	return denseSafetensor{Name: ref.Name, Shape: ref.Shape, Data: values}, nil
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
		tensors = append(tensors, ggufQuantizedTensor{
			Name:  ref.Name,
			Type:  tensorType,
			Shape: core.SliceClone(ref.Shape),
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
		return ggufTensorTypeQ8K, 256, 274, nil
	case QuantizeQ3_K:
		return ggufTensorTypeQ3K, 256, 110, nil
	case QuantizeQ2_K:
		return ggufTensorTypeQ2K, 256, 82, nil
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
		// Inline AppendUint16: skip the appendUint16LE func-call + its
		// [2]byte temp. binary.LittleEndian.AppendUint16 lowers to a
		// direct two-byte append.
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		// Stack-allocated pack buffer + single append at end of block —
		// replaces 32 individual `out = append(out, byte)` calls (each
		// with its own bounds check + length update) with one bulk
		// memcpy. Matches the pattern Q4_0 already uses.
		var packed [32]byte
		if scale == 0 {
			// Zero-block fast path: invScale would be zero so every q
			// is 0; skip the per-element work. `packed` already zeroed
			// by the var declaration.
			out = append(out, packed[:]...)
			continue
		}
		invScale := 1 / scale
		// Hoist the invScale==0 branch out of the inner loop — saves
		// 32 branch evaluations per block.
		for i, value := range block {
			// Multiply by 1/scale instead of dividing — single FMUL
			// vs FDIV per element (32x per block, millions per tensor).
			// Round-half-away-from-zero in float32 directly; skips the
			// float32→float64→math.Round→int round-trip and the call
			// overhead of math.Round (which handles edge cases
			// irrelevant to a clamped-to-127 quantiser).
			scaled := value * invScale
			var q int
			if scaled >= 0 {
				q = int(scaled + 0.5)
			} else {
				q = int(scaled - 0.5)
			}
			// Inline clampInt — avoids the func-call boundary on a
			// 2-branch primitive. The compiler will most likely inline
			// already, but doing it explicitly keeps the hot path
			// dependency-light.
			if q < -127 {
				q = -127
			} else if q > 127 {
				q = 127
			}
			packed[i] = byte(int8(q))
		}
		out = append(out, packed[:]...)
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
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		// Stack-allocated pack buffer instead of make([]byte, 16) per
		// block — saves one heap alloc per 32 input floats.
		var packed [16]byte
		if scale == 0 {
			// Zero-block fast path: q=0 → q+8=8 (Q4_0 stores
			// (q+8) ∈ [0,15] unsigned). Both nibbles of each packed
			// byte are 8, so the byte value is 0x88. Skips the
			// per-element multiply + round + branch work.
			for i := range packed {
				packed[i] = 0x88
			}
			out = append(out, packed[:]...)
			continue
		}
		invScale := 1 / scale
		// Split the i<16 branch out of the inner loop — two clean
		// 16-iter loops let the back-end keep the lower-nibble writes
		// (packed[i] = q) and upper-nibble OR-writes (packed[i-16] |=
		// q<<4) on independent memory dependencies. Same total work,
		// less branch overhead and a cleaner dep chain.
		for i := range 16 {
			value := block[i]
			scaled := value * invScale
			var q int
			// Round-half-away-from-zero in float32 — same optimisation
			// as quantizeQ8_0. The +8 bias re-centres the signed
			// quantised range into the [0,15] unsigned range Q4_0
			// stores.
			if scaled >= 0 {
				q = int(scaled+0.5) + 8
			} else {
				q = int(scaled-0.5) + 8
			}
			if q < 0 {
				q = 0
			} else if q > 15 {
				q = 15
			}
			packed[i] = byte(q)
		}
		for i := 16; i < 32; i++ {
			value := block[i]
			scaled := value * invScale
			var q int
			if scaled >= 0 {
				q = int(scaled+0.5) + 8
			} else {
				q = int(scaled-0.5) + 8
			}
			if q < 0 {
				q = 0
			} else if q > 15 {
				q = 15
			}
			packed[i-16] |= byte(q << 4)
		}
		out = append(out, packed[:]...)
	}
	return out
}

func quantizeQ5_0(values []float32) []byte {
	out := make([]byte, 0, len(values)/32*24)
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		minVal := minFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = (maxAbs - minVal) / 31
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(minVal))

		var packed [20]byte
		if scale == 0 {
			for i := range packed {
				packed[i] = 0x44 // 0b01000100 → each 5-bit nibble is 4 (midpoint)
			}
		} else {
			invScale := 1 / scale
			bitBuf := uint64(0)
			bitCount := 0
			byteIdx := 0
			for _, value := range block {
				scaled := (value - minVal) * invScale
				var q int
				if scaled >= 0 {
					q = int(scaled + 0.5)
				} else {
					q = int(scaled - 0.5)
				}
				if q < 0 {
					q = 0
				} else if q > 31 {
					q = 31
				}
				bitBuf |= uint64(q) << bitCount
				bitCount += 5
				for bitCount >= 8 {
					packed[byteIdx] = byte(bitBuf & 0xFF)
					bitBuf >>= 8
					bitCount -= 8
					byteIdx++
				}
			}
		}
		out = append(out, packed[:]...)
	}
	return out
}

const qkBlockSize = 256
const qkSubBlocks = 16
const qkSubBlockSize = qkBlockSize / qkSubBlocks

type qkScratch struct {
	minBlock     float32
	maxBlock     float32
	subMin       [qkSubBlocks]float32
	subMax       [qkSubBlocks]float32
	scales       [qkSubBlocks]float32
	scalesPacked [12]byte
}

var qkScratchPool = sync.Pool{New: func() any { return &qkScratch{} }}

func quantizeQ4_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*144)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)

	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 15
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))

		var quants [qkBlockSize / 2]byte
		if d == 0 {
			for i := range quants {
				quants[i] = 0x88
			}
		} else {
			invD := 1 / d
			for sb := range qkSubBlocks {
				subStart := sb * qkSubBlockSize
				scratch.subMin[sb] = block[subStart]
				scratch.subMax[sb] = block[subStart]
				for j := 1; j < qkSubBlockSize; j++ {
					v := block[subStart+j]
					if v < scratch.subMin[sb] {
						scratch.subMin[sb] = v
					}
					if v > scratch.subMax[sb] {
						scratch.subMax[sb] = v
					}
				}
				if scratch.subMax[sb] > scratch.subMin[sb] {
					scratch.scales[sb] = (scratch.subMax[sb] - scratch.subMin[sb]) / 63
				} else {
					scratch.scales[sb] = 0
				}
			}
			for sb := range qkSubBlocks {
				subStart := sb * qkSubBlockSize
				for j := range qkSubBlockSize {
					scaled := (block[subStart+j] - dmin) * invD
					q := clampInt(int(scaled+0.5), 0, 15)
					if j%2 == 0 {
						quants[(subStart+j)/2] = byte(q)
					} else {
						quants[(subStart+j)/2] |= byte(q << 4)
					}
				}
			}
		}
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func packKScales(scales []float32, packed *[12]byte) {
	var scMin, scMax float32 = scales[0], scales[0]
	for _, s := range scales[1:] {
		if s < scMin {
			scMin = s
		}
		if s > scMax {
			scMax = s
		}
	}
	if scMax <= scMin {
		return
	}
	dScale := (scMax - scMin) / 63
	invDScale := 1 / dScale
	bitBuf := uint64(0)
	bitCount := 0
	byteIdx := 0
	for _, s := range scales {
		scaled := (s - scMin) * invDScale
		q := clampInt(int(scaled+0.5), 0, 63)
		bitBuf |= uint64(q) << bitCount
		bitCount += 6
		for bitCount >= 8 && byteIdx < 12 {
			packed[byteIdx] = byte(bitBuf & 0xFF)
			bitBuf >>= 8
			bitCount -= 8
			byteIdx++
		}
	}
}

func quantizeKBlock(values []float32, quants []byte, bits int, d, dmin float32, scratch *qkScratch) {
	if d == 0 {
		return
	}
	invD := 1 / d
	bitBuf := uint64(0)
	bitCount := 0
	byteIdx := 0
	for idx, value := range values {
		if idx%qkSubBlockSize == 0 {
			sb := idx / qkSubBlockSize
			scratch.subMin[sb] = value
			scratch.subMax[sb] = value
			for j := 1; j < qkSubBlockSize && idx+j < len(values); j++ {
				v := values[idx+j]
				if v < scratch.subMin[sb] {
					scratch.subMin[sb] = v
				}
				if v > scratch.subMax[sb] {
					scratch.subMax[sb] = v
				}
			}
			if scratch.subMax[sb] > scratch.subMin[sb] {
				scratch.scales[sb] = (scratch.subMax[sb] - scratch.subMin[sb]) / 63
			} else {
				scratch.scales[sb] = 0
			}
		}
		scaled := (value - dmin) * invD
		q := clampInt(int(scaled+0.5), 0, (1<<bits)-1)
		bitBuf |= uint64(q) << bitCount
		bitCount += bits
		for bitCount >= 8 && byteIdx < len(quants) {
			quants[byteIdx] = byte(bitBuf & 0xFF)
			bitBuf >>= 8
			bitCount -= 8
			byteIdx++
		}
	}
}

func quantizeQ5_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*176)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 31
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize * 5 / 8]byte
		quantizeKBlock(block, quants[:], 5, d, dmin, scratch)
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func quantizeQ6_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*210)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 63
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize * 6 / 8]byte
		quantizeKBlock(block, quants[:], 6, d, dmin, scratch)
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func quantizeQ3_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*110)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 7
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize * 3 / 8]byte
		quantizeKBlock(block, quants[:], 3, d, dmin, scratch)
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func quantizeQ2_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*82)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 3
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize / 4]byte
		quantizeKBlock(block, quants[:], 2, d, dmin, scratch)
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func quantizeQ8_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	out := make([]byte, 0, nBlocks*274)
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 255
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize]byte
		if d > 0 {
			invD := 1 / d
			for sb := range qkSubBlocks {
				subStart := sb * qkSubBlockSize
				scratch.subMin[sb] = block[subStart]
				scratch.subMax[sb] = block[subStart]
				for j := 1; j < qkSubBlockSize; j++ {
					v := block[subStart+j]
					if v < scratch.subMin[sb] {
						scratch.subMin[sb] = v
					}
					if v > scratch.subMax[sb] {
						scratch.subMax[sb] = v
					}
				}
				if scratch.subMax[sb] > scratch.subMin[sb] {
					scratch.scales[sb] = (scratch.subMax[sb] - scratch.subMin[sb]) / 63
				} else {
					scratch.scales[sb] = 0
				}
			}
			for i, value := range block {
				scaled := (value - dmin) * invD
				quants[i] = byte(clampInt(int(scaled+0.5), 0, 255))
			}
		}
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func ggufQuantizeMetadata(source mp.ModelPack, format QuantizeFormat, labels map[string]string) []ggufMetadataEntry {
	fileType := uint32(7)
	quantizationType := string(QuantizeQ8_0)
	if format == QuantizeQ4_0 {
		fileType = 2
		quantizationType = string(QuantizeQ4_0)
	} else if format == QuantizeQ5_0 {
		fileType = 12
		quantizationType = string(QuantizeQ5_0)
	} else if format == QuantizeQ4_K {
		fileType = 15
		quantizationType = string(QuantizeQ4_K_M)
	} else if format == QuantizeQ5_K {
		fileType = 16
		quantizationType = "q5_k_m"
	} else if format == QuantizeQ6_K {
		fileType = 17
		quantizationType = "q6_k"
	} else if format == QuantizeQ8_K {
		fileType = 18
		quantizationType = "q8_k"
	} else if format == QuantizeQ3_K {
		fileType = 12
		quantizationType = "q3_k"
	} else if format == QuantizeQ2_K {
		fileType = 10
		quantizationType = "q2_k"
	}
	architecture := source.Architecture
	metadata := []ggufMetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: architecture},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: fileType},
		{Key: "general.quantization_version", ValueType: ValueTypeUint32, Value: uint32(2)},
		{Key: "general.quantization_type", ValueType: ValueTypeString, Value: quantizationType},
		{Key: "general.alignment", ValueType: ValueTypeUint32, Value: uint32(32)},
	}
	if source.VocabSize > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".vocab_size", ValueType: ValueTypeUint32, Value: uint32(source.VocabSize)})
	}
	if source.HiddenSize > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".embedding_length", ValueType: ValueTypeUint32, Value: uint32(source.HiddenSize)})
	}
	if source.NumLayers > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".block_count", ValueType: ValueTypeUint32, Value: uint32(source.NumLayers)})
	}
	if source.ContextLength > 0 {
		metadata = append(metadata, ggufMetadataEntry{Key: architecture + ".context_length", ValueType: ValueTypeUint32, Value: uint32(source.ContextLength)})
	}
	if len(labels) > 0 {
		keys := make([]string, 0, len(labels))
		for key := range labels {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			metadata = append(metadata, ggufMetadataEntry{Key: "go_mlx.label." + key, ValueType: ValueTypeString, Value: labels[key]})
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

func writeQuantizedGGUFStream(ctx context.Context, path string, metadata []ggufMetadataEntry, tensors []ggufQuantizedTensor, refs []safetensors.TensorRef, format QuantizeFormat, chunkElements int) error {
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
		expected := ggufQuantizedTensorDataSize(tensor)
		if dataSize != expected {
			return core.NewError("mlx: streamed GGUF tensor " + tensor.Name + " wrote " + strconv.FormatUint(dataSize, 10) + " bytes, want " + strconv.FormatUint(expected, 10))
		}
		written = tensor.Offset + expected
	}
	return nil
}

func writeQuantizedGGUFHeader(file *core.OSFile, metadata []ggufMetadataEntry, tensors []ggufQuantizedTensor) error {
	// Single 24-byte header: magic(4) + version(4) + tensorCount(8) + metadataCount(8).
	// One write call replaces 4 reflect.Write calls.
	var header [24]byte
	copy(header[:4], "GGUF")
	binary.LittleEndian.PutUint32(header[4:8], 3)
	binary.LittleEndian.PutUint64(header[8:16], uint64(len(tensors)))
	binary.LittleEndian.PutUint64(header[16:24], uint64(len(metadata)))
	if _, err := file.Write(header[:]); err != nil {
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

func writeQuantizedGGUFTensorStream(ctx context.Context, file *core.OSFile, ref safetensors.TensorRef, format QuantizeFormat, chunkElements int) (uint64, error) {
	// Resolve the quantiser once outside the chunk loop — saves a
	// switch per chunk (millions of chunks per multi-GB tensor).
	var quantise func([]float32) []byte
	switch format {
	case QuantizeQ8_0:
		quantise = quantizeQ8_0
	case QuantizeQ4_0:
		quantise = quantizeQ4_0
	case QuantizeQ5_0:
		quantise = quantizeQ5_0
	case QuantizeQ4_K:
		quantise = quantizeQ4_K
	case QuantizeQ5_K:
		quantise = quantizeQ5_K
	case QuantizeQ6_K:
		quantise = quantizeQ6_K
	case QuantizeQ8_K:
		quantise = quantizeQ8_K
	case QuantizeQ3_K:
		quantise = quantizeQ3_K
	case QuantizeQ2_K:
		quantise = quantizeQ2_K
	default:
		return 0, core.NewError("mlx: unsupported resolved GGUF format: " + string(format))
	}

	reader, err := safetensors.OpenReader(ref)
	if err != nil {
		return 0, err
	}
	defer reader.Close()
	var written uint64
	for offset := 0; offset < ref.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		count := min(chunkElements, ref.Elements-offset)
		values, err := reader.ReadFloat32Chunk(offset, count)
		if err != nil {
			return written, err
		}
		data := quantise(values)
		if _, err := file.Write(data); err != nil {
			return written, err
		}
		written += uint64(len(data))
	}
	return written, nil
}

func quantizeGGUFValues(format QuantizeFormat, values []float32) ([]byte, error) {
	switch format {
	case QuantizeQ8_0:
		return quantizeQ8_0(values), nil
	case QuantizeQ4_0:
		return quantizeQ4_0(values), nil
	case QuantizeQ5_0:
		return quantizeQ5_0(values), nil
	case QuantizeQ4_K:
		return quantizeQ4_K(values), nil
	case QuantizeQ5_K:
		return quantizeQ5_K(values), nil
	case QuantizeQ6_K:
		return quantizeQ6_K(values), nil
	case QuantizeQ8_K:
		return quantizeQ8_K(values), nil
	case QuantizeQ3_K:
		return quantizeQ3_K(values), nil
	case QuantizeQ2_K:
		return quantizeQ2_K(values), nil
	default:
		return nil, core.NewError("mlx: unsupported resolved GGUF format: " + string(format))
	}
}

func assignGGUFTensorOffsets(tensors []ggufQuantizedTensor, alignment uint64) {
	var offset uint64
	for i := range tensors {
		offset += alignPadding(offset, alignment)
		tensors[i].Offset = offset
		// Inline the data-size computation rather than passing the struct
		// by value to ggufQuantizedTensorDataSize (which would copy the
		// whole ggufQuantizedTensor including the Shape/Data slice
		// headers on every iteration).
		if tensors[i].Size > 0 {
			offset += tensors[i].Size
		} else {
			offset += uint64(len(tensors[i].Data))
		}
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
	// valueType(4) — direct LE encoding skips reflect dispatch.
	var typeBuf [4]byte
	binary.LittleEndian.PutUint32(typeBuf[:], entry.ValueType)
	if _, err := file.Write(typeBuf[:]); err != nil {
		return err
	}
	return writeGGUFMetadataValue(file, entry.ValueType, entry.Value)
}

func writeGGUFMetadataValue(file *core.OSFile, valueType uint32, value any) error {
	switch valueType {
	case ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			return core.NewError("mlx: GGUF metadata value is not a string")
		}
		return writeGGUFStringValue(file, stringValue)
	case ValueTypeUint32:
		var v uint32
		switch concrete := value.(type) {
		case uint32:
			v = concrete
		case int:
			v = uint32(concrete)
		default:
			return core.NewError("mlx: GGUF metadata value is not uint32")
		}
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], v)
		_, err := file.Write(buf[:])
		return err
	default:
		return core.NewError("mlx: unsupported GGUF metadata write type " + strconv.FormatUint(uint64(valueType), 10))
	}
}

func writeGGUFTensorInfo(file *core.OSFile, tensor ggufQuantizedTensor) error {
	if err := writeGGUFStringValue(file, tensor.Name); err != nil {
		return err
	}
	// Pack ndim(4) + all dim(8 each) + tensorType(4) + offset(8) into
	// one batched write — avoids one binary.Write reflect call per
	// dimension (typically 2-4 per tensor).
	dims := tensor.Shape
	bufLen := 4 + len(dims)*8 + 4 + 8
	// Small scratch on stack for the common 2-4 dim case; fall back to
	// heap for higher rank tensors (rare in real GGUF files).
	var stack [64]byte
	var buf []byte
	if bufLen <= len(stack) {
		buf = stack[:bufLen]
	} else {
		buf = make([]byte, bufLen)
	}
	binary.LittleEndian.PutUint32(buf[:4], uint32(len(dims)))
	pos := 4
	for _, dim := range dims {
		binary.LittleEndian.PutUint64(buf[pos:pos+8], dim)
		pos += 8
	}
	binary.LittleEndian.PutUint32(buf[pos:pos+4], tensor.Type)
	pos += 4
	binary.LittleEndian.PutUint64(buf[pos:pos+8], tensor.Offset)
	_, err := file.Write(buf)
	return err
}

func writeGGUFStringValue(file *core.OSFile, value string) error {
	// Length-prefix in one batched write with the value bytes when the
	// value is small enough to fit on stack. For the common metadata-
	// key case (32-200 bytes) this skips one syscall + one Write call.
	var stack [256]byte
	if len(value)+8 <= len(stack) {
		buf := stack[:8+len(value)]
		binary.LittleEndian.PutUint64(buf[:8], uint64(len(value)))
		copy(buf[8:], value)
		_, err := file.Write(buf)
		return err
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(value)))
	if _, err := file.Write(lenBuf[:]); err != nil {
		return err
	}
	_, err := file.Write(core.AsBytes(value))
	return err
}

// ggufPaddingZeros — package-level read-only zero buffer for writePadding.
// 32 KiB chunk matches the original on-stack size; living at package scope
// avoids a 32 KiB stack-frame allocation per writePadding call.
var ggufPaddingZeros [32 * 1024]byte

func writePadding(file *core.OSFile, n uint64) error {
	for n > 0 {
		size := min(n, uint64(len(ggufPaddingZeros)))
		if _, err := file.Write(ggufPaddingZeros[:size]); err != nil {
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

// maxAbsFloat32 returns max(|v|) over values. The inner loop avoids
// math.Abs (which round-trips float32→float64→float32 per element); a
// direct bit-clear of the float32 sign bit lowers to ARM64 FABS in one
// instruction. The 4-way unroll (W8-A2 lever) lets the M-series pipeline
// keep four FABS+FCMP chains independent so per-iteration latency hides
// behind instruction-level parallelism. Block-sized inputs (32 / 256
// elements) hit the unrolled path; the scalar tail handles the
// remainder.
func maxAbsFloat32(values []float32) float32 {
	const mask = 0x7fffffff
	var m0, m1, m2, m3 float32
	i := 0
	n := len(values)
	for ; i+4 <= n; i += 4 {
		a0 := math.Float32frombits(math.Float32bits(values[i]) & mask)
		a1 := math.Float32frombits(math.Float32bits(values[i+1]) & mask)
		a2 := math.Float32frombits(math.Float32bits(values[i+2]) & mask)
		a3 := math.Float32frombits(math.Float32bits(values[i+3]) & mask)
		if a0 > m0 {
			m0 = a0
		}
		if a1 > m1 {
			m1 = a1
		}
		if a2 > m2 {
			m2 = a2
		}
		if a3 > m3 {
			m3 = a3
		}
	}
	maxAbs := m0
	if m1 > maxAbs {
		maxAbs = m1
	}
	if m2 > maxAbs {
		maxAbs = m2
	}
	if m3 > maxAbs {
		maxAbs = m3
	}
	for ; i < n; i++ {
		abs := math.Float32frombits(math.Float32bits(values[i]) & mask)
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

func minFloat32(values []float32) float32 {
	minVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] < minVal {
			minVal = values[i]
		}
	}
	return minVal
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

func copyLocalFile(sourcePath, destinationPath string) error {
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return quantizeGGUFResultError(read)
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return quantizeGGUFResultError(result)
	}
	return nil
}
