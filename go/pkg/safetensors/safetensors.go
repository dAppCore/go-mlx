// SPDX-Licence-Identifier: EUPL-1.2

// Package safetensors is a pure-Go, all-platforms reader for the safetensors checkpoint
// format — name → raw bytes, the lingua franca the byte-native backends consume. It is
// the format reader half of the weight loader; mapping the parsed tensors onto a gemma4
// model (the weight structs + embed/head tables) is a separate slice on top. (pkg/metal
// has its own safetensors loader, but it materialises into cgo *metal.Array; this one
// stays at bytes so the no-cgo native path and go-rocm can share it.)
package safetensors

import (
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
