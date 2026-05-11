// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/safetensors"
)

const (
	ggufValueTypeBool   = 7
	ggufValueTypeUint64 = 10
	ggufValueTypeArray  = 9
	ggufTensorTypeQ4K   = 12
)

type ggufMetaSpec struct {
	Key       string
	ValueType uint32
	Value     any
}

type ggufArraySpec struct {
	ElementType uint32
	Values      []any
}

type ggufTensorSpec struct {
	Name string
	Type uint32
	Dims []uint64
}

func writeTestGGUF(t *testing.T, path string, metadata []ggufMetaSpec, tensors []ggufTensorSpec) {
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

	for _, tensor := range tensors {
		writeGGUFString(t, file, tensor.Name)
		write(uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			write(dim)
		}
		write(tensor.Type)
		write(uint64(0))
	}
}

func writeGGUFString(t *testing.T, file *core.OSFile, value string) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatalf("write string length: %v", err)
	}
	if _, err := file.Write([]byte(value)); err != nil {
		t.Fatalf("write string bytes: %v", err)
	}
}

func writeGGUFValue(t *testing.T, file *core.OSFile, valueType uint32, value any) {
	t.Helper()
	switch valueType {
	case ggufValueTypeBool:
		boolValue, ok := value.(bool)
		if !ok {
			t.Fatalf("write bool: got %T, want bool", value)
		}
		var encoded uint8
		if boolValue {
			encoded = 1
		}
		if err := binary.Write(file, binary.LittleEndian, encoded); err != nil {
			t.Fatalf("write bool: %v", err)
		}
	case gguf.ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			t.Fatalf("write string: got %T, want string", value)
		}
		writeGGUFString(t, file, stringValue)
	case gguf.ValueTypeUint32:
		uint32Value, ok := value.(uint32)
		if !ok {
			t.Fatalf("write uint32: got %T, want uint32", value)
		}
		if err := binary.Write(file, binary.LittleEndian, uint32Value); err != nil {
			t.Fatalf("write uint32: %v", err)
		}
	case ggufValueTypeUint64:
		uint64Value, ok := value.(uint64)
		if !ok {
			t.Fatalf("write uint64: got %T, want uint64", value)
		}
		if err := binary.Write(file, binary.LittleEndian, uint64Value); err != nil {
			t.Fatalf("write uint64: %v", err)
		}
	case ggufValueTypeArray:
		arrayValue, ok := value.(ggufArraySpec)
		if !ok {
			t.Fatalf("write array: got %T, want ggufArraySpec", value)
		}
		if err := binary.Write(file, binary.LittleEndian, arrayValue.ElementType); err != nil {
			t.Fatalf("write array element type: %v", err)
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(arrayValue.Values))); err != nil {
			t.Fatalf("write array length: %v", err)
		}
		for _, item := range arrayValue.Values {
			writeGGUFValue(t, file, arrayValue.ElementType, item)
		}
	default:
		t.Fatalf("unsupported test gguf value type %d", valueType)
	}
}

// math.Float32bits-based helpers used by mlx-root tests that produce
// binary test fixtures (kv_snapshot_*_test.go, api_test.go).

type denseSafetensor struct {
	Name  string
	Shape []uint64
	Data  []float32
}

func appendUint16LE(out []byte, value uint16) []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], value)
	return append(out, buf[:]...)
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
type safetensorTestTensor struct {
	Name  string
	Shape []int
	Data  []float32
}

func writeDenseSafetensorsPack(t *testing.T, modelType string, tensors []safetensorTestTensor) string {
	t.Helper()
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), core.Sprintf(`{
		"model_type": %q,
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960
	}`, modelType))
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestSafetensorsF32(t, core.PathJoin(dir, "model.safetensors"), tensors)
	return dir
}

func writeTestSafetensorsF32(t *testing.T, path string, tensors []safetensorTestTensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		buf := make([]byte, len(tensor.Data)*4)
		for i, value := range tensor.Data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
		}
		data = append(data, buf...)
		header[tensor.Name] = entry{
			DType:       "F32",
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
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
		return nil, testResultError(read)
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
	var header map[string]safetensors.HeaderEntry
	if result := core.JSONUnmarshal(data[headerStart:headerEnd], &header); !result.OK {
		return nil, testResultError(result)
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

func decodeDenseSafetensor(path, name string, entry safetensors.HeaderEntry, payload []byte) (denseSafetensor, error) {
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
	values, err := safetensors.DecodeFloatData(core.Upper(entry.DType), raw, int(elements))
	if err != nil {
		return denseSafetensor{}, core.E("decodeDenseSafetensor", "decode "+path+" tensor "+name, err)
	}
	return denseSafetensor{Name: name, Shape: shape, Data: values}, nil
}

func testResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
