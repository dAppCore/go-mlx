// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleLoadSafetensors() {
	path, cleanup := mustExampleSafetensorsFile()
	defer cleanup()

	loaded := map[string]*Array{}
	for name, arr := range LoadSafetensors(path) {
		loaded[name] = arr
	}
	defer freeExampleSafetensors(loaded)

	names := exampleSafetensorsNames(loaded)
	first := loaded[names[0]]
	Materialize(first)

	core.Println(names)
	core.Println(first.Shape(), first.Floats())
	// Output:
	// [model.layers.0.self_attn.q_proj.lora_A.weight model.layers.0.self_attn.q_proj.lora_B.weight]
	// [2 2] [1 2 3 4]
}

func ExampleLoadAllSafetensors() {
	path, cleanup := mustExampleSafetensorsFile()
	defer cleanup()

	loaded, err := LoadAllSafetensors(path)
	if err != nil {
		panic(err)
	}
	defer freeExampleSafetensors(loaded)

	down := loaded["model.layers.0.self_attn.q_proj.lora_B.weight"]
	Materialize(down)

	core.Println(len(loaded), down.Shape(), down.Floats()[3])
	// Output: 2 [2 2] 8
}

func mustExampleSafetensorsFile() (string, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-metal-safetensors-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "adapter.safetensors")
	tensors, freeTensors := mustExampleSafetensorsTensors()
	defer freeTensors()

	if err := SaveSafetensors(path, tensors); err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return path, func() { core.RemoveAll(dir) }
}

func mustExampleSafetensorsTensors() (map[string]*Array, func()) {
	up := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	down := FromValues([]float32{5, 6, 7, 8}, 2, 2)
	Materialize(up, down)

	return map[string]*Array{
		"model.layers.0.self_attn.q_proj.lora_A.weight": up,
		"model.layers.0.self_attn.q_proj.lora_B.weight": down,
	}, func() { Free(up, down) }
}

func exampleSafetensorsNames(tensors map[string]*Array) []string {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	return names
}

func freeExampleSafetensors(tensors map[string]*Array) {
	for _, tensor := range tensors {
		Free(tensor)
	}
}
