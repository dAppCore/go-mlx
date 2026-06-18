// SPDX-Licence-Identifier: EUPL-1.2

// Package safetensors is a pure-Go, all-platforms reader for the safetensors checkpoint
// format — name → raw bytes, the lingua franca the byte-native backends consume. It is
// the format reader half of the weight loader; mapping the parsed tensors onto a gemma4
// model (the weight structs + embed/head tables) is a separate slice on top. (pkg/metal
// has its own safetensors loader, but it materialises into cgo *metal.Array; this one
// stays at bytes so the no-cgo native path and go-rocm can share it.)
package safetensors

import (
	"sort"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Tensor is one safetensors entry: its dtype name (e.g. "BF16", "F32", "U8"), shape, and
// raw little-endian bytes — Data sub-slices the source blob (no copy).
type Tensor struct {
	Dtype string
	Shape []int
	Data  []byte
}

// Mapping is a memory-mapped safetensors file (see LoadMmap). Data is the WHOLE file mapped
// page-aligned, and every entry in Tensors is a view into it (no heap copy of the weights).
// The page-aligned base is what makes the zero-copy GPU path possible: a backend wraps Data
// in ONE no-copy buffer (Metal's bytesNoCopy needs page alignment) and binds each tensor at
// its byte offset into Data — so a multi-GB checkpoint is never duplicated in heap or GPU
// memory. Close unmaps; the Mapping MUST outlive every Tensor view and every GPU buffer
// taken over Data.
type Mapping struct {
	Data    []byte // the whole file, mapped page-aligned (the no-copy buffer's backing)
	Tensors map[string]Tensor
}

// dtypeBytes is the element byte size of the safetensors dtypes gemma4 checkpoints use:
// bf16/f32 weights, and the 4-bit-quant companions (u8/u32 packed codes + bf16 scales).
var dtypeBytes = map[string]int{
	"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
	"I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1,
}

// Parse reads a safetensors blob: an 8-byte little-endian header length, then that many
// bytes of JSON ({name:{dtype,shape,data_offsets:[start,end]}, optional "__metadata__"}),
// then the tensor data. data_offsets are relative to the END of the header. Returns
// name→Tensor with Data sub-slicing blob (no copy). Validates the header length, each
// entry's dtype/shape/offsets, and that the byte span equals dtype × ∏shape.
func Parse(blob []byte) (map[string]Tensor, error) {
	if len(blob) < 8 {
		return nil, core.NewError("safetensors.Parse: blob shorter than the 8-byte header length")
	}
	var hdrLen uint64
	for i := 0; i < 8; i++ {
		hdrLen |= uint64(blob[i]) << (8 * uint(i))
	}
	dataStart := 8 + int(hdrLen)
	if hdrLen == 0 || dataStart < 8 || dataStart > len(blob) {
		return nil, core.NewError("safetensors.Parse: header length out of range")
	}
	var hdr map[string]map[string]any
	if r := core.JSONUnmarshal(blob[8:dataStart], &hdr); !r.OK {
		return nil, core.NewError("safetensors.Parse: header JSON parse failed")
	}

	out := make(map[string]Tensor, len(hdr))
	for name, e := range hdr {
		if name == "__metadata__" { // the one reserved non-tensor key
			continue
		}
		dt, ok := e["dtype"].(string)
		if !ok {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " missing dtype")
		}
		elem, known := dtypeBytes[dt]
		if !known {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " unsupported dtype " + dt)
		}
		shapeRaw, ok := e["shape"].([]any)
		if !ok {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " missing shape")
		}
		shape := make([]int, len(shapeRaw))
		count := 1
		for i, s := range shapeRaw {
			f, ok := s.(float64) // JSON numbers decode as float64
			if !ok || f < 0 {
				return nil, core.NewError("safetensors.Parse: tensor " + name + " bad shape entry")
			}
			shape[i] = int(f)
			count *= shape[i]
		}
		offRaw, ok := e["data_offsets"].([]any)
		if !ok || len(offRaw) != 2 {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " data_offsets must be [start,end]")
		}
		sf, ok1 := offRaw[0].(float64)
		ef, ok2 := offRaw[1].(float64)
		if !ok1 || !ok2 {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " non-numeric data_offsets")
		}
		start, end := int(sf), int(ef)
		if start < 0 || end < start || dataStart+end > len(blob) {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " data_offsets out of range")
		}
		if end-start != count*elem {
			return nil, core.NewError("safetensors.Parse: tensor " + name + " byte span != dtype × shape")
		}
		out[name] = Tensor{Dtype: dt, Shape: shape, Data: blob[dataStart+start : dataStart+end]}
	}
	return out, nil
}

// Encode writes tensors to a safetensors blob — the inverse of Parse: the 8-byte
// little-endian header length, the JSON header ({name:{dtype,shape,data_offsets}}), then
// the tensor data laid out in sorted-name order (deterministic). Validates each tensor's
// dtype + that its bytes match dtype × ∏shape. Parse(Encode(x)) round-trips x.
func Encode(tensors map[string]Tensor) ([]byte, error) {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		if n == "__metadata__" {
			return nil, core.NewError("safetensors.Encode: __metadata__ is reserved")
		}
		names = append(names, n)
	}
	sort.Strings(names) // deterministic layout

	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	hdr := make(map[string]entry, len(names))
	var data []byte
	off := 0
	for _, n := range names {
		t := tensors[n]
		elem, ok := dtypeBytes[t.Dtype]
		if !ok {
			return nil, core.NewError("safetensors.Encode: tensor " + n + " unsupported dtype " + t.Dtype)
		}
		count := 1
		for _, d := range t.Shape {
			if d < 0 {
				return nil, core.NewError("safetensors.Encode: tensor " + n + " negative shape")
			}
			count *= d
		}
		if len(t.Data) != count*elem {
			return nil, core.NewError("safetensors.Encode: tensor " + n + " byte span != dtype × shape")
		}
		shape := t.Shape
		if shape == nil {
			shape = []int{}
		}
		hdr[n] = entry{Dtype: t.Dtype, Shape: shape, DataOffsets: [2]int{off, off + len(t.Data)}}
		data = append(data, t.Data...)
		off += len(t.Data)
	}

	hj := core.JSONMarshal(hdr)
	if !hj.OK {
		return nil, core.NewError("safetensors.Encode: header marshal failed")
	}
	hdrBytes := hj.Value.([]byte)
	out := make([]byte, 8+len(hdrBytes)+len(data))
	n := uint64(len(hdrBytes))
	for i := 0; i < 8; i++ {
		out[i] = byte(n >> (8 * uint(i)))
	}
	copy(out[8:], hdrBytes)
	copy(out[8+len(hdrBytes):], data)
	return out, nil
}

// Load reads a safetensors file and Parses it. NOTE: it reads the whole file into memory
// (the per-tensor Data then sub-slices it); an mmap variant for multi-GB checkpoints is a
// later optimisation, and loading a real model is a deliberate, memory-heavy operation.
func Load(path string) (map[string]Tensor, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("safetensors.Load", "read "+path, err)
	}
	return Parse([]byte(str))
}
