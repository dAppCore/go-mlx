// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"encoding/binary"
	stdio "io"
	"math"

	core "dappco.re/go"
)

// HeaderEntry is one tensor entry in the safetensors JSON header.
type HeaderEntry struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

type Index struct {
	Path    string
	Tensors map[string]TensorRef
	Names   []string
}

type TensorRef struct {
	Name      string
	Path      string
	DType     string
	Shape     []uint64
	Elements  int
	DataStart int64
	ByteLen   int64
}

type TensorReader struct {
	ref             TensorRef
	file            *core.OSFile
	bytesPerElement int
}

func IndexFiles(paths []string) (Index, error) {
	index := Index{Tensors: map[string]TensorRef{}}
	for _, path := range paths {
		shard, err := ReadIndex(path)
		if err != nil {
			return Index{}, err
		}
		if cap(index.Names) < len(index.Names)+len(shard.Names) {
			grown := make([]string, len(index.Names), len(index.Names)+len(shard.Names))
			copy(grown, index.Names)
			index.Names = grown
		}
		for _, name := range shard.Names {
			if _, ok := index.Tensors[name]; ok {
				return Index{}, core.NewError("mlx: duplicate tensor in safetensors shards: " + name)
			}
			index.Tensors[name] = shard.Tensors[name]
			index.Names = append(index.Names, name)
		}
	}
	core.SliceSort(index.Names)
	return index, nil
}

func ReadIndex(path string) (Index, error) {
	opened := core.Open(path)
	if !opened.OK {
		return Index{}, resultError(opened)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	var headerLenBuf [8]byte
	if _, err := stdio.ReadFull(file, headerLenBuf[:]); err != nil {
		return Index{}, err
	}
	headerLen := binary.LittleEndian.Uint64(headerLenBuf[:])
	headerBytes := make([]byte, int(headerLen))
	if _, err := stdio.ReadFull(file, headerBytes); err != nil {
		return Index{}, err
	}
	var header map[string]HeaderEntry
	if result := core.JSONUnmarshal(headerBytes, &header); !result.OK {
		return Index{}, resultError(result)
	}

	index := Index{
		Path:    path,
		Tensors: make(map[string]TensorRef, len(header)),
		Names:   make([]string, 0, len(header)),
	}
	dataStart := int64(8 + headerLen)
	// Pre-scan to size a single uint64 slab covering every per-tensor
	// Shape slice. Replaces N small allocs (one per tensor) with one,
	// matching the writeSubset slab pattern. RefFromHeader stays a
	// public allocator for callers outside the index path.
	totalDims := 0
	for name, entry := range header {
		if name == "__metadata__" {
			continue
		}
		totalDims += len(entry.Shape)
	}
	shapeSlab := make([]uint64, 0, totalDims)
	for name, entry := range header {
		if name == "__metadata__" {
			continue
		}
		ref, err := refFromHeaderSlab(path, name, entry, dataStart, &shapeSlab)
		if err != nil {
			return Index{}, err
		}
		index.Tensors[name] = ref
		index.Names = append(index.Names, name)
	}
	core.SliceSort(index.Names)
	return index, nil
}

// refFromHeaderSlab is the index-local variant of RefFromHeader that
// carves each tensor's Shape slice out of a shared uint64 slab. Callers
// guarantee the slab has enough capacity (sized by the prior header
// scan). Public RefFromHeader retains its standalone allocation form.
func refFromHeaderSlab(path, name string, entry HeaderEntry, dataStart int64, slab *[]uint64) (TensorRef, error) {
	if len(entry.DataOffsets) != 2 {
		return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid data_offsets: " + name)
	}
	begin := entry.DataOffsets[0]
	end := entry.DataOffsets[1]
	if begin < 0 || end < begin {
		return TensorRef{}, core.NewError("mlx: safetensors tensor offsets are invalid: " + name)
	}
	start := len(*slab)
	*slab = (*slab)[: start+len(entry.Shape) : cap(*slab)]
	shape := (*slab)[start : start+len(entry.Shape) : start+len(entry.Shape)]
	elements := 1
	for i, dim := range entry.Shape {
		if dim <= 0 {
			return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid shape: " + name)
		}
		shape[i] = uint64(dim)
		elements *= int(dim)
	}
	return TensorRef{
		Name:      name,
		Path:      path,
		DType:     core.Upper(entry.DType),
		Shape:     shape,
		Elements:  elements,
		DataStart: dataStart + begin,
		ByteLen:   end - begin,
	}, nil
}

func RefFromHeader(path, name string, entry HeaderEntry, dataStart int64) (TensorRef, error) {
	if len(entry.DataOffsets) != 2 {
		return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid data_offsets: " + name)
	}
	begin := entry.DataOffsets[0]
	end := entry.DataOffsets[1]
	if begin < 0 || end < begin {
		return TensorRef{}, core.NewError("mlx: safetensors tensor offsets are invalid: " + name)
	}
	shape := make([]uint64, len(entry.Shape))
	elements := 1
	for i, dim := range entry.Shape {
		if dim <= 0 {
			return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid shape: " + name)
		}
		shape[i] = uint64(dim)
		elements *= int(dim)
	}
	return TensorRef{
		Name:      name,
		Path:      path,
		DType:     core.Upper(entry.DType),
		Shape:     shape,
		Elements:  elements,
		DataStart: dataStart + begin,
		ByteLen:   end - begin,
	}, nil
}

func ReadRefValues(ref TensorRef) ([]float32, error) {
	opened := core.Open(ref.Path)
	if !opened.OK {
		return nil, resultError(opened)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	return DecodeFloatData(ref.DType, raw, ref.Elements)
}

func WriteRefFloat32Chunks(ctx context.Context, file *core.OSFile, ref TensorRef, chunkElements int) error {
	if chunkElements <= 0 {
		chunkElements = defaultChunkElements
	}
	reader, err := OpenReader(ref)
	if err != nil {
		return err
	}
	defer reader.Close()
	for offset := 0; offset < ref.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(chunkElements, ref.Elements-offset)
		values, err := reader.ReadFloat32Chunk(offset, count)
		if err != nil {
			return err
		}
		if err := writeFloat32Values(file, values); err != nil {
			return err
		}
	}
	return nil
}

func ReadRefFloat32Chunk(ref TensorRef, offset, count int) ([]float32, error) {
	reader, err := OpenReader(ref)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	return reader.ReadFloat32Chunk(offset, count)
}

func OpenReaders(refs []TensorRef) ([]TensorReader, error) {
	readers := make([]TensorReader, 0, len(refs))
	for _, ref := range refs {
		reader, err := OpenReader(ref)
		if err != nil {
			CloseReaders(readers)
			return nil, err
		}
		readers = append(readers, reader)
	}
	return readers, nil
}

func OpenReader(ref TensorRef) (TensorReader, error) {
	bytesPerElement, err := DTypeByteSize(ref.DType)
	if err != nil {
		return TensorReader{}, err
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return TensorReader{}, resultError(opened)
	}
	return TensorReader{
		ref:             ref,
		file:            opened.Value.(*core.OSFile),
		bytesPerElement: bytesPerElement,
	}, nil
}

func CloseReaders(readers []TensorReader) {
	for _, reader := range readers {
		reader.Close()
	}
}

func (r TensorReader) Close() {
	if r.file != nil {
		_ = r.file.Close()
	}
}

func (r TensorReader) ReadFloat32Chunk(offset, count int) ([]float32, error) {
	if offset < 0 || count < 0 || offset+count > r.ref.Elements {
		return nil, core.NewError("mlx: safetensors tensor chunk exceeds tensor bounds")
	}
	raw := make([]byte, count*r.bytesPerElement)
	start := r.ref.DataStart + int64(offset*r.bytesPerElement)
	n, err := r.file.ReadAt(raw, start)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, core.NewError("mlx: safetensors tensor chunk is truncated")
	}
	return DecodeFloatData(r.ref.DType, raw, count)
}

func DTypeByteSize(dtype string) (int, error) {
	switch dtype {
	case "F16", "BF16":
		return 2, nil
	case "F32":
		return 4, nil
	case "F64":
		return 8, nil
	}
	switch core.Upper(dtype) {
	case "F16", "BF16":
		return 2, nil
	case "F32":
		return 4, nil
	case "F64":
		return 8, nil
	default:
		return 0, core.NewError("unsupported dense safetensors dtype: " + dtype)
	}
}

func maxIntValue() int { return int(^uint(0) >> 1) }

func ReadRefRaw(ref TensorRef) ([]byte, error) {
	if ref.ByteLen < 0 || ref.ByteLen > int64(maxIntValue()) {
		return nil, core.NewError("mlx: safetensors tensor byte length is invalid: " + ref.Name)
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return nil, resultError(opened)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, core.NewError("mlx: safetensors tensor payload is truncated: " + ref.Name)
	}
	return raw, nil
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

const defaultChunkElements = 1 << 20

func writeFloat32Values(file *core.OSFile, values []float32) error {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	_, err := file.Write(raw)
	return err
}

func DecodeFloatData(dtype string, raw []byte, elements int) ([]float32, error) {
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
			values[i] = Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
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

func Float16ToFloat32(value uint16) float32 {
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
