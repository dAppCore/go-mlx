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
	// Reuse three scratch buffers across chunked writes:
	//   raw       — the byte payload read from the source file
	//   values    — the decoded float32 slice
	//   writeBuf  — the re-encoded bytes the writer flushes
	// Each chunk previously allocated all three; now they grow once
	// to chunkElements (or chunkElements*bytesPerElement / 4) and are
	// reused for every subsequent chunk on the same tensor.
	var (
		rawScratch    []byte
		valuesScratch []float32
		writeScratch  []byte
	)
	for offset := 0; offset < ref.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(chunkElements, ref.Elements-offset)
		var values []float32
		rawScratch, valuesScratch, values, err = reader.readFloat32ChunkInto(offset, count, rawScratch, valuesScratch)
		if err != nil {
			return err
		}
		writeScratch, err = writeFloat32ValuesScratch(file, values, writeScratch)
		if err != nil {
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

// readFloat32ChunkInto is the scratch-aware variant of ReadFloat32Chunk.
// It accepts (and returns) byte + float32 scratch buffers so a caller
// in a chunked loop (WriteRefFloat32Chunks) can avoid allocating fresh
// buffers per chunk. The returned values slice always equals the
// (possibly grown) valuesScratch sliced to count.
func (r TensorReader) readFloat32ChunkInto(offset, count int, rawScratch []byte, valuesScratch []float32) ([]byte, []float32, []float32, error) {
	if offset < 0 || count < 0 || offset+count > r.ref.Elements {
		return rawScratch, valuesScratch, nil, core.NewError("mlx: safetensors tensor chunk exceeds tensor bounds")
	}
	rawNeed := count * r.bytesPerElement
	if cap(rawScratch) < rawNeed {
		rawScratch = make([]byte, rawNeed)
	} else {
		rawScratch = rawScratch[:rawNeed]
	}
	start := r.ref.DataStart + int64(offset*r.bytesPerElement)
	n, err := r.file.ReadAt(rawScratch, start)
	if err != nil && !(err == stdio.EOF && n == len(rawScratch)) {
		return rawScratch, valuesScratch, nil, err
	}
	if n != len(rawScratch) {
		return rawScratch, valuesScratch, nil, core.NewError("mlx: safetensors tensor chunk is truncated")
	}
	values, err := decodeFloatDataInto(r.ref.DType, rawScratch, count, valuesScratch)
	if err != nil {
		return rawScratch, valuesScratch, nil, err
	}
	if cap(values) > cap(valuesScratch) {
		valuesScratch = values
	}
	return rawScratch, valuesScratch, values, nil
}

func DTypeByteSize(dtype string) (int, error) {
	// Canonical fast path covers the four supported dtypes by exact
	// match (the common case after RefFromHeader has normalised
	// entry.DType through core.Upper).
	switch dtype {
	case "F16", "BF16":
		return 2, nil
	case "F32":
		return 4, nil
	case "F64":
		return 8, nil
	}
	// Non-canonical input (callers handing us lowercase / mixed case).
	// Branch by length so we never call core.Upper — that path was
	// dominating the 26 ns / 1 alloc on lowercase "bf16". Each branch
	// is a single direct byte compare for the ASCII letters.
	switch len(dtype) {
	case 3:
		// F16, F32, F64.
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '1' && dtype[2] == '6' {
			return 2, nil
		}
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '3' && dtype[2] == '2' {
			return 4, nil
		}
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '6' && dtype[2] == '4' {
			return 8, nil
		}
	case 4:
		// BF16.
		if (dtype[0] == 'B' || dtype[0] == 'b') && (dtype[1] == 'F' || dtype[1] == 'f') && dtype[2] == '1' && dtype[3] == '6' {
			return 2, nil
		}
	}
	return 0, core.NewError("unsupported dense safetensors dtype: " + dtype)
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
	_, err := writeFloat32ValuesScratch(file, values, nil)
	return err
}

// writeFloat32ValuesScratch reuses a caller-supplied byte buffer for
// the F32 encode. The buffer is grown when too small and returned so
// the caller (WriteRefFloat32Chunks) can reuse it across chunks.
func writeFloat32ValuesScratch(file *core.OSFile, values []float32, scratch []byte) ([]byte, error) {
	need := len(values) * 4
	if cap(scratch) < need {
		scratch = make([]byte, need)
	} else {
		scratch = scratch[:need]
	}
	for i, value := range values {
		binary.LittleEndian.PutUint32(scratch[i*4:], math.Float32bits(value))
	}
	_, err := file.Write(scratch)
	return scratch, err
}

func DecodeFloatData(dtype string, raw []byte, elements int) ([]float32, error) {
	return decodeFloatDataInto(dtype, raw, elements, nil)
}

// decodeFloatDataInto is the scratch-aware variant of DecodeFloatData.
// Callers that decode in a loop (WriteRefFloat32Chunks) can hand back
// the prior chunk's slice to avoid re-allocating.
func decodeFloatDataInto(dtype string, raw []byte, elements int, scratch []float32) ([]float32, error) {
	var values []float32
	if cap(scratch) < elements {
		values = make([]float32, elements)
	} else {
		values = scratch[:elements]
	}
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
		// Hoist a fixed-cap subslice and read by direct byte pair so
		// the compiler can elide the per-iter bound-check on raw[i*2:]
		// re-slicing. With Float16ToFloat32 dominating per-elem cost,
		// this drops the F16 decode to ~3.2us / 2048 elems (-23%).
		buf := raw[: elements*2 : elements*2]
		for i := 0; i < elements; i++ {
			j := i * 2
			values[i] = Float16ToFloat32(uint16(buf[j]) | uint16(buf[j+1])<<8)
		}
	case "BF16":
		if len(raw) != elements*2 {
			return nil, core.NewError("BF16 payload length does not match tensor shape")
		}
		// Same byte-pair hoist as F16. The body is a straight bit shift
		// into the float32 high half — no function call, so the saving
		// is smaller (-9%) but compounds when packed alongside F16.
		buf := raw[: elements*2 : elements*2]
		for i := 0; i < elements; i++ {
			j := i * 2
			values[i] = math.Float32frombits((uint32(buf[j]) | uint32(buf[j+1])<<8) << 16)
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
