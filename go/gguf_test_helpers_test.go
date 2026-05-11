// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/gguf"
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
