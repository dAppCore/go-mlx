// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"testing"
	"unsafe"

	core "dappco.re/go"
)

func TestLoadTensorsDenseQ4_0AndQ8_0(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "model.gguf")
	f32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32[0:4], 0x3f800000)
	binary.LittleEndian.PutUint32(f32[4:8], 0xc0000000)
	bf16 := []byte{0x80, 0x3f, 0x00, 0xc0}
	q8 := make([]byte, 34)
	binary.LittleEndian.PutUint16(q8[0:2], float32ToFloat16(0.5))
	q8[2] = byte(int8(2))
	q8[3] = 0xfe
	q4 := make([]byte, 18)
	binary.LittleEndian.PutUint16(q4[0:2], float32ToFloat16(0.5))
	q4[2] = 0x6a

	writeTestGGUFPayload(t, path,
		[]ggufMetaSpec{{Key: "general.alignment", ValueType: ValueTypeUint32, Value: uint32(32)}},
		[]ggufTensorPayloadSpec{
			{Name: "f32.weight", Type: ggufTensorTypeF32, Dims: []uint64{2}, Data: f32},
			{Name: "bf16.weight", Type: ggufTensorTypeBF16, Dims: []uint64{2}, Data: bf16},
			{Name: "q4.weight", Type: TensorTypeQ4_0, Dims: []uint64{32}, Data: q4},
			{Name: "q8.weight", Type: TensorTypeQ8_0, Dims: []uint64{32}, Data: q8},
		})

	mapping, err := LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	defer mapping.Close()

	if got := mapping.Tensors["f32.weight"]; got.Dtype != "F32" || len(got.Shape) != 1 || got.Shape[0] != 2 || !bytesEqual(got.Data, f32) {
		t.Fatalf("f32 tensor = %+v, want F32 [2] payload", got)
	}
	if got := mapping.Tensors["bf16.weight"]; got.Dtype != "BF16" || len(got.Shape) != 1 || got.Shape[0] != 2 || !bytesEqual(got.Data, bf16) {
		t.Fatalf("bf16 tensor = %+v, want BF16 [2] payload", got)
	}
	assertTensorDataViewsMapping(t, mapping, "f32.weight")
	assertTensorDataViewsMapping(t, mapping, "bf16.weight")
	gotQ8 := mapping.Tensors["q8.weight"]
	if gotQ8.Dtype != "F16" || len(gotQ8.Shape) != 1 || gotQ8.Shape[0] != 32 {
		t.Fatalf("q8 tensor = %+v, want F16 [32]", gotQ8)
	}
	gotQ4 := mapping.Tensors["q4.weight"]
	if gotQ4.Dtype != "F16" || len(gotQ4.Shape) != 1 || gotQ4.Shape[0] != 32 {
		t.Fatalf("q4 tensor = %+v, want F16 [32]", gotQ4)
	}
	if binary.LittleEndian.Uint16(gotQ4.Data[0:2]) != float32ToFloat16(1) ||
		binary.LittleEndian.Uint16(gotQ4.Data[32:34]) != float32ToFloat16(-1) {
		t.Fatalf("q4 dequant selected values = %#x %#x, want +/-1 in f16",
			binary.LittleEndian.Uint16(gotQ4.Data[0:2]),
			binary.LittleEndian.Uint16(gotQ4.Data[32:34]))
	}
	if binary.LittleEndian.Uint16(gotQ8.Data[0:2]) != float32ToFloat16(1) ||
		binary.LittleEndian.Uint16(gotQ8.Data[2:4]) != float32ToFloat16(-1) {
		t.Fatalf("q8 dequant first values = %#x %#x, want +/-1 in f16",
			binary.LittleEndian.Uint16(gotQ8.Data[0:2]),
			binary.LittleEndian.Uint16(gotQ8.Data[2:4]))
	}
}

func assertTensorDataViewsMapping(t *testing.T, mapping *TensorMapping, name string) {
	t.Helper()
	tensor := mapping.Tensors[name]
	if len(mapping.Data) == 0 || len(tensor.Data) == 0 {
		t.Fatalf("%s has empty mapping or tensor data", name)
	}
	base := uintptr(unsafe.Pointer(&mapping.Data[0]))
	end := base + uintptr(len(mapping.Data))
	ptr := uintptr(unsafe.Pointer(&tensor.Data[0]))
	if ptr < base || ptr >= end {
		t.Fatalf("%s Data does not view the GGUF mapping", name)
	}
}

type ggufTensorPayloadSpec struct {
	Name string
	Type uint32
	Dims []uint64
	Data []byte
}

func writeTestGGUFPayload(t *testing.T, path string, metadata []ggufMetaSpec, tensors []ggufTensorPayloadSpec) {
	t.Helper()
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	write := func(value any) {
		t.Helper()
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			t.Fatalf("binary write failed: %v", err)
		}
	}
	if _, err := file.Write([]byte("GGUF")); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(len(metadata)))
	for _, entry := range metadata {
		writeGGUFString(t, file, entry.Key)
		write(entry.ValueType)
		writeGGUFValue(t, file, entry.ValueType, entry.Value)
	}
	var offset uint64
	offsets := make([]uint64, len(tensors))
	for i, tensor := range tensors {
		offset += alignPadding(offset, 32)
		offsets[i] = offset
		offset += uint64(len(tensor.Data))
	}
	for i, tensor := range tensors {
		writeGGUFString(t, file, tensor.Name)
		write(uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			write(dim)
		}
		write(tensor.Type)
		write(offsets[i])
	}
	position, err := file.Seek(0, 1)
	if err != nil {
		t.Fatalf("seek gguf header end: %v", err)
	}
	if err := writePadding(file, alignPadding(uint64(position), 32)); err != nil {
		t.Fatalf("write data padding: %v", err)
	}
	var written uint64
	for i, tensor := range tensors {
		if err := writePadding(file, offsets[i]-written); err != nil {
			t.Fatalf("write tensor padding: %v", err)
		}
		if _, err := file.Write(tensor.Data); err != nil {
			t.Fatalf("write tensor payload: %v", err)
		}
		written = offsets[i] + uint64(len(tensor.Data))
	}
}

func bytesEqual(a, b []byte) bool {
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
