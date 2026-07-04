// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleLoadSafetensorsFromReader() {
	data := mustExampleSafetensorsBytes()
	reader := newBytesRWS(data)

	loaded := map[string]*Array{}
	for name, arr := range LoadSafetensorsFromReader(reader, int64(len(data)), "memory-adapter") {
		loaded[name] = arr
	}
	defer freeExampleSafetensors(loaded)

	names := exampleSafetensorsNames(loaded)
	up := loaded[names[0]]
	Materialize(up)

	core.Println(names[0])
	core.Println(up.Shape(), up.Floats()[1])
	// Output:
	// model.layers.0.self_attn.q_proj.lora_A.weight
	// [2 2] 2
}

func ExampleLoadAllSafetensorsFromReader() {
	data := mustExampleSafetensorsBytes()
	reader := newBytesRWS(data)

	loaded, err := LoadAllSafetensorsFromReader(reader, int64(len(data)), "memory-adapter")
	if err != nil {
		panic(err)
	}
	defer freeExampleSafetensors(loaded)

	down := loaded["model.layers.0.self_attn.q_proj.lora_B.weight"]
	Materialize(down)

	core.Println(len(loaded), down.Shape(), down.Floats()[0])
	// Output: 2 [2 2] 5
}

func ExampleSaveSafetensorsToWriter() {
	tensors, freeTensors := mustExampleSafetensorsTensors()
	defer freeTensors()

	writer := newBytesRWSSize(8192)
	if err := SaveSafetensorsToWriter(writer, 8192, "memory-adapter", tensors, map[string]string{"format": "pt"}); err != nil {
		panic(err)
	}

	data := writer.Bytes()
	loaded, err := LoadAllSafetensorsFromReader(newBytesRWS(data), int64(len(data)), "memory-adapter")
	if err != nil {
		panic(err)
	}
	defer freeExampleSafetensors(loaded)

	core.Println(len(data) > 0, exampleSafetensorsNames(loaded)[1])
	// Output: true model.layers.0.self_attn.q_proj.lora_B.weight
}

func mustExampleSafetensorsBytes() []byte {
	tensors, freeTensors := mustExampleSafetensorsTensors()
	defer freeTensors()

	writer := newBytesRWSSize(8192)
	if err := SaveSafetensorsToWriter(writer, 8192, "memory-adapter", tensors, nil); err != nil {
		panic(err)
	}
	return append([]byte(nil), writer.Bytes()...)
}
