// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"

	core "dappco.re/go"
	sharedsafetensors "dappco.re/go/inference/safetensors"
	mp "dappco.re/go/inference/modelpack"
)

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

// writeTestSafetensorsF32 writes tensors as an F32 safetensors file via the
// shared dappco.re/go/inference/safetensors codec (WriteSafetensors +
// EncodeFloat32) instead of hand-rolling the header JSON and byte layout —
// go-mlx's chunked merge engine and go-inference's ported one read/write the
// same on-disk format, so fixtures are built through that same codec. Takes
// testing.TB so *testing.B fixture builders (merge_bench_test.go,
// compare_bench_test.go) can call it directly instead of keeping their own
// copy.
func writeTestSafetensorsF32(tb testing.TB, path string, tensors []safetensorTestTensor) {
	tb.Helper()
	infos := make(map[string]sharedsafetensors.SafetensorsTensorInfo, len(tensors))
	data := make(map[string][]byte, len(tensors))
	for _, tensor := range tensors {
		infos[tensor.Name] = sharedsafetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: tensor.Shape}
		data[tensor.Name] = sharedsafetensors.EncodeFloat32(tensor.Data)
	}
	if result := sharedsafetensors.WriteSafetensors(path, infos, data); !result.OK {
		tb.Fatalf("write safetensors: %v", result.Value)
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

// readDenseSafetensors reads path via the shared
// dappco.re/go/inference/safetensors codec (ReadSafetensors + GetTensorData +
// DecodeFloat32) instead of hand-parsing the header/offset layout. This
// helper only ever reads back a file Packs itself just wrote, so the
// malformed-input rejections the hand-rolled version carried (truncated
// header, non-2-length data_offsets, non-positive shape dims) never fire in
// practice — Packs/ComparePacks error-path coverage lives in
// merge_coverage*_test.go against the real chunked reader instead.
func readDenseSafetensors(path string) ([]denseSafetensor, error) {
	read := sharedsafetensors.ReadSafetensors(path)
	if !read.OK {
		return nil, testResultError(read)
	}
	data := read.Value.(sharedsafetensors.SafetensorsData)
	tensors := make([]denseSafetensor, 0, len(data.Tensors))
	for name, info := range data.Tensors {
		tensor, err := decodeDenseSafetensor(path, name, info, data.Data)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, tensor)
	}
	return tensors, nil
}

func decodeDenseSafetensor(path, name string, info sharedsafetensors.SafetensorsTensorInfo, allData []byte) (denseSafetensor, error) {
	shape := make([]uint64, len(info.Shape))
	elements := 1
	for i, dim := range info.Shape {
		shape[i] = uint64(dim)
		elements *= dim
	}
	raw := sharedsafetensors.GetTensorData(info, allData)
	values, err := sharedsafetensors.DecodeFloat32(info.Dtype, raw, elements)
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

func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

// writeF16SafetensorsPack builds a single-tensor F16 model pack from float32
// source values, encoding each value with the float32ToFloat16 helper. It is
// the F16 counterpart of writeDenseSafetensorsPack — used to exercise the
// dtype-mismatch comparison branch (F16 tuned vs F32 base at the same shape).
func writeF16SafetensorsPack(t *testing.T, modelType string, tensors []safetensorTestTensor) string {
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
	writeTestSafetensorsF16(t, core.PathJoin(dir, "model.safetensors"), tensors)
	return dir
}

// writeTestSafetensorsF16 is the F16 counterpart of writeTestSafetensorsF32.
// go-inference's safetensors codec only exposes a decoder for F16 (models
// are authored upstream in whatever dtype they ship; merge output is always
// F32 — see mergedInfo in merge_write.go), so the F16 byte-encode stays
// local and only the write (header + on-disk layout) delegates to
// WriteSafetensors.
func writeTestSafetensorsF16(tb testing.TB, path string, tensors []safetensorTestTensor) {
	tb.Helper()
	infos := make(map[string]sharedsafetensors.SafetensorsTensorInfo, len(tensors))
	data := make(map[string][]byte, len(tensors))
	for _, tensor := range tensors {
		var buf []byte
		for _, value := range tensor.Data {
			buf = appendUint16LE(buf, float32ToFloat16(value))
		}
		infos[tensor.Name] = sharedsafetensors.SafetensorsTensorInfo{Dtype: "F16", Shape: tensor.Shape}
		data[tensor.Name] = buf
	}
	if result := sharedsafetensors.WriteSafetensors(path, infos, data); !result.OK {
		tb.Fatalf("write safetensors: %v", result.Value)
	}
}

const modelPackTokenizerJSON = `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`

func testPack(dir string) mp.ModelPack {
	return testPackArch(dir, "qwen3")
}

func testPackArch(dir, architecture string) mp.ModelPack {
	return mp.ModelPack{
		Root:          dir,
		Path:          dir,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{core.PathJoin(dir, "model.safetensors")},
		TokenizerPath: core.PathJoin(dir, "tokenizer.json"),
		Architecture:  architecture,
	}
}
