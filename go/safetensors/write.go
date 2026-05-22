// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"encoding/binary"

	core "dappco.re/go"
)

const defaultRawChunkBytes = 4 << 20

// WriteSubset writes a safetensors file containing refs without loading all
// selected tensors into memory. Tensor payloads are copied directly from the
// indexed source files in bounded chunks.
func WriteSubset(ctx context.Context, path string, refs []TensorRef) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if core.Trim(path) == "" {
		return core.NewError("mlx: safetensors subset path is empty")
	}
	if len(refs) == 0 {
		return core.NewError("mlx: safetensors subset requires at least one tensor")
	}

	ordered, header, err := subsetHeader(refs)
	if err != nil {
		return err
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return resultError(encoded)
	}
	headerBytes := encoded.Value.([]byte)

	parent := core.PathDir(path)
	if result := core.MkdirAll(parent, 0o755); !result.OK {
		return resultError(result)
	}
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		return resultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	if err := writeAll(file, headerLen[:]); err != nil {
		return err
	}
	if err := writeAll(file, headerBytes); err != nil {
		return err
	}
	// Reuse a single byte buffer across every per-ref chunked copy.
	// writeRefRawChunks previously allocated its own buffer per call,
	// so a subset of N tensors meant N small-or-large allocations.
	// Each ref's payload size is capped by chunkBytes anyway, so
	// reuse is safe — the buffer is grown on demand by passing
	// through writeRefRawChunksScratch.
	var scratch []byte
	for _, ref := range ordered {
		if err := ctx.Err(); err != nil {
			return err
		}
		scratch, err = writeRefRawChunksScratch(ctx, file, ref, defaultRawChunkBytes, scratch)
		if err != nil {
			return err
		}
	}
	return nil
}

func subsetHeader(refs []TensorRef) ([]TensorRef, map[string]HeaderEntry, error) {
	byName := make(map[string]TensorRef, len(refs))
	names := make([]string, 0, len(refs))
	for _, ref := range refs {
		if core.Trim(ref.Name) == "" {
			return nil, nil, core.NewError("mlx: safetensors subset tensor name is empty")
		}
		if ref.ByteLen < 0 {
			return nil, nil, core.NewError("mlx: safetensors subset tensor byte length is invalid: " + ref.Name)
		}
		if _, ok := byName[ref.Name]; ok {
			return nil, nil, core.NewError("mlx: safetensors subset contains duplicate tensor: " + ref.Name)
		}
		byName[ref.Name] = ref
		names = append(names, ref.Name)
	}
	core.SliceSort(names)

	ordered := make([]TensorRef, 0, len(names))
	header := make(map[string]HeaderEntry, len(names))
	// Single int64 slab covering all per-tensor Shape + DataOffsets slices —
	// every header entry needs len(Shape)+2 int64s, so we count once then
	// hand each entry a sub-slice. Replaces 2*N small allocs with one.
	totalDims := 0
	for _, name := range names {
		totalDims += len(byName[name].Shape)
	}
	slab := make([]int64, totalDims+2*len(names))
	cursor := 0
	var offset int64
	for _, name := range names {
		ref := byName[name]
		shape := slab[cursor : cursor+len(ref.Shape) : cursor+len(ref.Shape)]
		cursor += len(ref.Shape)
		for i, dim := range ref.Shape {
			if dim > uint64(maxInt64Value()) {
				return nil, nil, core.NewError("mlx: safetensors subset tensor shape is too large: " + ref.Name)
			}
			shape[i] = int64(dim)
		}
		offsets := slab[cursor : cursor+2 : cursor+2]
		cursor += 2
		offsets[0] = offset
		offsets[1] = offset + ref.ByteLen
		header[name] = HeaderEntry{
			DType:       core.Upper(ref.DType),
			Shape:       shape,
			DataOffsets: offsets,
		}
		offset += ref.ByteLen
		ordered = append(ordered, ref)
	}
	return ordered, header, nil
}

func writeRefRawChunks(ctx context.Context, out *core.OSFile, ref TensorRef, chunkBytes int64) error {
	_, err := writeRefRawChunksScratch(ctx, out, ref, chunkBytes, nil)
	return err
}

// writeRefRawChunksScratch streams one tensor's raw payload through a
// caller-supplied byte buffer, returning the (possibly grown) buffer
// for the next call to reuse. Hoisting the buffer up to WriteSubset
// collapses what was N small allocs into one.
func writeRefRawChunksScratch(ctx context.Context, out *core.OSFile, ref TensorRef, chunkBytes int64, scratch []byte) ([]byte, error) {
	if chunkBytes <= 0 {
		chunkBytes = defaultRawChunkBytes
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return scratch, resultError(opened)
	}
	in := opened.Value.(*core.OSFile)
	defer in.Close()

	need := minInt64(chunkBytes, ref.ByteLen)
	if int64(cap(scratch)) < need {
		scratch = make([]byte, need)
	} else {
		scratch = scratch[:need]
	}
	remaining := ref.ByteLen
	offset := ref.DataStart
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return scratch, err
		}
		want := minInt64(int64(len(scratch)), remaining)
		n, err := in.ReadAt(scratch[:want], offset)
		if err != nil && !(err == core.EOF && int64(n) == want) {
			return scratch, err
		}
		if int64(n) != want {
			return scratch, core.NewError("mlx: safetensors tensor payload is truncated: " + ref.Name)
		}
		if err := writeAll(out, scratch[:want]); err != nil {
			return scratch, err
		}
		offset += want
		remaining -= want
	}
	return scratch, nil
}

func writeAll(file *core.OSFile, data []byte) error {
	for len(data) > 0 {
		n, err := file.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return core.NewError("mlx: safetensors write made no progress")
		}
		data = data[n:]
	}
	return nil
}

func maxInt64Value() int64 { return int64(^uint64(0) >> 1) }

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
