// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"encoding/binary"
	"math"
	"sort"
	"unicode/utf8"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

func writeMergedSafetensors(ctx context.Context, path string, indexes []safetensors.Index, method Method, t float64, sources []Source, allowMismatch bool) (int, int, []string, error) {
	header := buildMergedHeader(indexes[0])
	created := core.Create(path)
	if !created.OK {
		return 0, 0, nil, resultError(created)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	// marshalMergedHeader is the hand-rolled, byte-identical replacement for
	// core.JSONMarshal(header). encoding/json reflects over the
	// map[string]HeaderEntry — building per-entry interface boxes and a
	// growable internal buffer — which was the single biggest remaining
	// Packs allocator (reflect.unsafe_New ~20% of alloc_objects at 32
	// tensors). The emitter walks the map's sorted keys once into a
	// pre-sized buffer with no reflection. Output is bit-exact identical to
	// json.Marshal for the same map (see TestMarshalMergedHeaderParity).
	headerBytes := marshalMergedHeader(header)
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

// marshalMergedHeader emits the safetensors JSON header for header with no
// reflection, byte-for-byte identical to core.JSONMarshal(header). Keys are
// emitted in sorted order (encoding/json sorts map keys), each entry's
// fields in struct-declaration order (dtype, shape, data_offsets), integers
// in base-10 with no leading zeros, and strings with encoding/json's
// HTML-safe default escaping (`<` `>` `&` → < > &, the
// \b\f\n\r\t mnemonics, \u00XX otherwise). A nil slice emits null and a
// non-nil empty slice emits [], matching json.Marshal exactly. The byte
// parity is locked by TestMarshalMergedHeaderParity against core.JSONMarshal
// over adversarial fixtures (HTML-meta names, escapes, scalar/nil shapes,
// file-order-vs-alphabetical, large offsets).
func marshalMergedHeader(header map[string]safetensors.HeaderEntry) []byte {
	// encoding/json marshals a nil map as null (a non-nil empty map is {}).
	// buildMergedHeader always returns a non-nil make(...) map, so this is
	// unreachable from writeMergedSafetensors — it keeps the emitter exactly
	// equivalent to core.JSONMarshal for any input should it ever be reused.
	if header == nil {
		return []byte("null")
	}
	names := make([]string, 0, len(header))
	for name := range header {
		names = append(names, name)
	}
	sort.Strings(names)

	// Size the buffer up-front. Per entry: the quoted+escaped name, the
	// fixed {"dtype":"","shape":[],"data_offsets":[]} scaffold (~44 bytes),
	// the dtype, and the integer widths (~20 bytes per dim/offset). Over- or
	// under-estimating only changes one append-grow, never the bytes.
	estBytes := 2 // {}
	for _, name := range names {
		entry := header[name]
		estBytes += len(name) + len(entry.DType) + 44 + 20*(len(entry.Shape)+len(entry.DataOffsets))
	}
	out := make([]byte, 0, estBytes)
	out = append(out, '{')
	for i, name := range names {
		entry := header[name]
		if i > 0 {
			out = append(out, ',')
		}
		out = appendHeaderJSONString(out, name)
		out = append(out, ':', '{')
		out = append(out, '"', 'd', 't', 'y', 'p', 'e', '"', ':')
		out = appendHeaderJSONString(out, entry.DType)
		out = append(out, ',', '"', 's', 'h', 'a', 'p', 'e', '"', ':')
		out = appendInt64Array(out, entry.Shape)
		out = append(out, ',', '"', 'd', 'a', 't', 'a', '_', 'o', 'f', 'f', 's', 'e', 't', 's', '"', ':')
		out = appendInt64Array(out, entry.DataOffsets)
		out = append(out, '}')
	}
	out = append(out, '}')
	return out
}

// appendInt64Array emits a JSON array of int64s, or null for a nil slice
// (encoding/json marshals a nil slice as null, a non-nil empty slice as []).
func appendInt64Array(dst []byte, values []int64) []byte {
	if values == nil {
		return append(dst, 'n', 'u', 'l', 'l')
	}
	dst = append(dst, '[')
	for i, v := range values {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendInt64(dst, v)
	}
	return append(dst, ']')
}

// appendInt64 emits v in base-10 with no leading zeros, matching
// encoding/json / strconv.FormatInt. The digits land in a fixed stack
// buffer so no heap allocation occurs regardless of magnitude.
func appendInt64(dst []byte, v int64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var buf [20]byte
	i := len(buf)
	neg := v < 0
	var uv uint64
	if neg {
		uv = uint64(-v)
	} else {
		uv = uint64(v)
	}
	for uv > 0 {
		i--
		buf[i] = byte('0' + uv%10)
		uv /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return append(dst, buf[i:]...)
}

// appendHeaderJSONString appends s as a JSON string literal byte-identical
// to encoding/json's default Marshal of a Go string — the same HTML-safe
// escaping (`<` `>` `&`), control-byte mnemonics (\b\f\n\r\t), \u00XX for
// the rest, \u202X for U+2028/U+2029, and � for invalid UTF-8. Mirrors
// the fuzz-locked appendJSONStringHTML in go/openai/openai.go; merge keeps
// its own copy rather than importing across the package boundary (AX-8).
func appendHeaderJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if headerJSONSafe(b) {
				i++
				continue
			}
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			switch b {
			case '\\', '"':
				dst = append(dst, '\\', b)
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			case '\b':
				dst = append(dst, '\\', 'b')
			case '\f':
				dst = append(dst, '\\', 'f')
			default:
				dst = append(dst, '\\', 'u', '0', '0', headerHexNibble(b>>4), headerHexNibble(b&0xF))
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			dst = append(dst, '\\', 'u', 'f', 'f', 'f', 'd')
			i += size
			start = i
			continue
		}
		if c == ' ' || c == ' ' {
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			dst = append(dst, '\\', 'u', '2', '0', '2', headerHexNibble(byte(c&0xF)))
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		dst = append(dst, s[start:]...)
	}
	return append(dst, '"')
}

// headerJSONSafe reports whether ASCII byte b passes through a JSON string
// body unescaped under encoding/json's HTML-safe default.
func headerJSONSafe(b byte) bool {
	if b < 0x20 {
		return false
	}
	switch b {
	case '"', '\\', '<', '>', '&':
		return false
	}
	return true
}

// headerHexNibble returns the lowercase ASCII hex digit for the low nibble
// of v — the \u00XX / \u202X escape branches of appendHeaderJSONString.
func headerHexNibble(v byte) byte {
	const hex = "0123456789abcdef"
	return hex[v&0xF]
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
	// slerpRawA/B + slerpValuesA/B back the SLERP dot/norm scan's two
	// simultaneously-live chunk reads. The scan ran before writeLinearChunksUsing
	// (which uses rawRead/valuesRead) and previously allocated its four
	// buffers fresh per tensor — the largest SLERP-merge alloc count after
	// the weight-pair fix (~94% of the chunk-read allocs at -alloc_objects).
	// They are owned by the pass-level scratch and grown to the largest
	// tensor once; the scan completes (producing the weights) before
	// writeLinearChunksUsing touches its own buffers, so the two sets never
	// alias mid-use. A and B stay separate because both chunks are compared
	// in the same iteration.
	slerpRawA    []byte
	slerpRawB    []byte
	slerpValuesA []float32
	slerpValuesB []float32
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
	weights, err := slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, scratch.slerpWeights[:], scratch)
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
	weights, err := slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, nil, nil)
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
	return slerpChunkedWeightsFromReaders(ctx, readers, refs[0].Elements, t, chunkElements, nil, nil)
}

// slerpChunkedWeightsFromReaders is the readers-already-open variant
// for the SLERP dot/norm scan. Lets writeSLERPChunks share readers
// across the SLERP weight scan and the writeLinearChunks pass.
//
// out is an optional two-element scratch the result is written into and
// returned as out[:2] — the cached per-tensor path passes a writeScratch
// field so the two-element weight slice isn't heap-allocated once per
// tensor. Pass nil (single-call sites) to get a freshly-allocated slice.
//
// scratch is an optional pass-level buffer set for the dot/norm scan's two
// chunk reads; the cached per-tensor path passes its writeScratch so the
// scan buffers grow once for the whole merge instead of once per tensor.
// Pass nil (single-call sites) to allocate locally.
func slerpChunkedWeightsFromReaders(ctx context.Context, readers []safetensors.TensorReader, elements int, t float64, chunkElements int, out []float64, scratch *writeScratch) ([]float64, error) {
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
	// the b read clobber a's values mid-scan. Reused across chunks (and,
	// when scratch is non-nil, across every tensor of the pass), this drops
	// the two fresh []byte + two fresh []float32 ReadFloat32Chunk allocated
	// per chunk down to a one-time grow.
	var rawA, rawB []byte
	var valuesA, valuesB []float32
	if scratch != nil {
		rawA, rawB = scratch.slerpRawA, scratch.slerpRawB
		valuesA, valuesB = scratch.slerpValuesA, scratch.slerpValuesB
	}
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
	// Carry the grown scan buffers back to the pass scratch so the next
	// tensor reuses them. On an error return above the merge aborts, so
	// dropping the partially-grown buffers there is harmless.
	if scratch != nil {
		scratch.slerpRawA, scratch.slerpRawB = rawA, rawB
		scratch.slerpValuesA, scratch.slerpValuesB = valuesA, valuesB
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
