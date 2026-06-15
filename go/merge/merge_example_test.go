// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// ExamplePacks merges two single-tensor safetensors model packs with a linear
// 0.25/0.75 weighting and prints the resulting tensor — each element is
// 0.25*left + 0.75*right. The merge writes a new model-pack directory plus a
// provenance file; only the merged values are printed for a stable output.
func ExamplePacks() {
	left := exampleWritePack("qwen3", 0, 2, 4, 6)
	right := exampleWritePack("qwen3", 10, 12, 14, 16)

	out := core.PathJoin(core.MkdirTemp("", "merge-example-*").Value.(string), "merged")
	result, err := Packs(context.Background(), Options{
		OutputPath: out,
		Method:     MethodLinear,
		Sources: []Source{
			{Pack: examplePack(left), Weight: 0.25},
			{Pack: examplePack(right), Weight: 0.75},
		},
	})
	if err != nil {
		core.Println("error:", err)
		return
	}

	values := exampleReadTensor(result.WeightPath)
	core.Println(result.Method, values)
	// Output: linear [7.5 9.5 11.5 13.5]
}

// exampleWritePack writes a minimal single-tensor F32 safetensors pack and
// returns its directory.
func exampleWritePack(modelType string, data ...float32) string {
	dir := core.MkdirTemp("", "merge-example-src-*").Value.(string)
	core.WriteFile(core.PathJoin(dir, "config.json"), []byte(core.Sprintf(`{"model_type":%q,"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960}`, modelType)), 0o644)
	core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`), 0o644)

	name := "model.norm.weight"
	header := core.Sprintf(`{%q:{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, name, len(data), len(data)*4)
	body := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(body[i*4:], math.Float32bits(v))
	}
	out := make([]byte, 8+len(header)+len(body))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(header)))
	copy(out[8:], header)
	copy(out[8+len(header):], body)
	core.WriteFile(core.PathJoin(dir, "model.safetensors"), out, 0o644)
	return dir
}

func examplePack(dir string) mp.ModelPack {
	return mp.ModelPack{
		Root:          dir,
		Path:          dir,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{core.PathJoin(dir, "model.safetensors")},
		TokenizerPath: core.PathJoin(dir, "tokenizer.json"),
		Architecture:  "qwen3",
	}
}

func exampleReadTensor(path string) []float32 {
	data := core.ReadFile(path).Value.([]byte)
	headerLen := binary.LittleEndian.Uint64(data[:8])
	payload := data[8+headerLen:]
	values := make([]float32, len(payload)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
	}
	return values
}
