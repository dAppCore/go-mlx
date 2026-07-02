// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"

	core "dappco.re/go"
	sharedsafetensors "dappco.re/go/inference/safetensors"
	mp "dappco.re/go/mlx/pack"
)

// ExamplePacks merges two single-tensor safetensors model packs with a linear
// 0.25/0.75 weighting and prints the resulting tensor — each element is
// 0.25*left + 0.75*right. The merge writes a new model-pack directory plus a
// provenance file; only the merged values are printed for a stable output.
func ExamplePacks() {
	left := exampleWritePack("qwen3", 0, 2, 4, 6)
	right := exampleWritePack("qwen3", 10, 12, 14, 16)
	defer core.RemoveAll(left)
	defer core.RemoveAll(right)

	outRoot := core.MkdirTemp("", "merge-example-*").Value.(string)
	defer core.RemoveAll(outRoot)
	out := core.PathJoin(outRoot, "merged")
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

// ExamplePacks_slerp spherically interpolates (SLERP) two orthogonal
// single-tensor packs at the midpoint t = 0.5. Unit vectors [1,0] and [0,1]
// are 90 degrees apart, so the half-way point on the unit arc is
// [sqrt(0.5), sqrt(0.5)] — both components equal, unlike a linear blend which
// would also give [0.5, 0.5] here but shrinks the norm. SLERP requires exactly
// two sources.
func ExamplePacks_slerp() {
	left := exampleWritePack("qwen3", 1, 0)
	right := exampleWritePack("qwen3", 0, 1)
	defer core.RemoveAll(left)
	defer core.RemoveAll(right)

	outRoot := core.MkdirTemp("", "merge-example-slerp-*").Value.(string)
	defer core.RemoveAll(outRoot)
	out := core.PathJoin(outRoot, "merged")
	result, err := Packs(context.Background(), Options{
		OutputPath: out,
		Method:     MethodSLERP,
		T:          0.5,
		Sources: []Source{
			{Pack: examplePack(left)},
			{Pack: examplePack(right)},
		},
	})
	if err != nil {
		core.Println("error:", err)
		return
	}

	values := exampleReadTensor(result.WeightPath)
	core.Println(result.Method, values)
	// Output: slerp [0.70710677 0.70710677]
}

// exampleWritePack writes a minimal single-tensor F32 safetensors pack via
// the shared dappco.re/go/inference/safetensors codec and returns its
// directory.
func exampleWritePack(modelType string, data ...float32) string {
	dir := core.MkdirTemp("", "merge-example-src-*").Value.(string)
	core.WriteFile(core.PathJoin(dir, "config.json"), []byte(core.Sprintf(`{"model_type":%q,"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960}`, modelType)), 0o644)
	core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`), 0o644)

	name := "model.norm.weight"
	infos := map[string]sharedsafetensors.SafetensorsTensorInfo{name: {Dtype: "F32", Shape: []int{len(data)}}}
	tensorData := map[string][]byte{name: sharedsafetensors.EncodeFloat32(data)}
	sharedsafetensors.WriteSafetensors(core.PathJoin(dir, "model.safetensors"), infos, tensorData)
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
	read := sharedsafetensors.ReadSafetensors(path)
	data := read.Value.(sharedsafetensors.SafetensorsData)
	info := data.Tensors["model.norm.weight"]
	raw := sharedsafetensors.GetTensorData(info, data.Data)
	values, _ := sharedsafetensors.DecodeFloat32(info.Dtype, raw, len(raw)/4)
	return values
}
