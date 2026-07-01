// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleMetalAvailable() {
	core.Println(Available() == MetalAvailable())
	// Output: true
}

func ExampleAvailable() {
	if Available() {
		core.Println("metal")
	}
}

func ExampleSetCacheLimit() {
	previous := SetCacheLimit(4 << 30)
	_ = SetCacheLimit(previous)
}

func ExampleSetMemoryLimit() {
	previous := SetMemoryLimit(32 << 30)
	_ = SetMemoryLimit(previous)
}

func ExampleGetActiveMemory() {
	active := GetActiveMemory()
	_ = active
}

func ExampleGetPeakMemory() {
	peak := GetPeakMemory()
	_ = peak
}

func ExampleClearCache() {
	ClearCache()
}

func ExampleGetCacheMemory() {
	cache := GetCacheMemory()
	_ = cache
}

func ExampleResetPeakMemory() {
	ResetPeakMemory()
}

func ExampleSetWiredLimit() {
	previous := SetWiredLimit(8 << 30)
	_ = SetWiredLimit(previous)
}

func ExampleGetDeviceInfo() {
	info := GetDeviceInfo()
	_ = info
}

func Example_metalbackendName() {
	backend := &metalbackend{}
	core.Println(backend.Name())
	// Output: metal
}

func Example_metalbackendAvailable() {
	backend := &metalbackend{}
	core.Println(backend.Available() == MetalAvailable())
	// Output: true
}

func Example_metalbackendLoadModel() {
	backend := &metalbackend{}
	model, err := castTextModel(backend.LoadModel("/models/gemma4"))
	if err != nil {
		return
	}
	defer model.Close()
}

func Example_metaladapterGenerate() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	for token := range model.Generate(context.Background(), "Write a short training note.") {
		_ = token
	}
}

func Example_metaladapterChat() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	messages := []inference.Message{{Role: "user", Content: "Write a short training note."}}
	for token := range model.Chat(context.Background(), messages) {
		_ = token
	}
}

func Example_metaladapterClassify() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	_, _ = castClassify(model.Classify(context.Background(), []string{"adapter quality improved"}))
}

func Example_metaladapterBatchGenerate() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	_, _ = castBatch(model.BatchGenerate(context.Background(), []string{
		"Summarise the adapter change:",
		"Write a regression note:",
	}))
}

func Example_metaladapterMetrics() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	metrics := model.Metrics()
	_ = metrics
}

func Example_metaladapterModelType() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	modelType := model.ModelType()
	_ = modelType
}

func Example_metaladapterInfo() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	info := model.Info()
	_ = info
}

func Example_metaladapterInspectAttention() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	inspector, ok := model.(inference.AttentionInspector)
	if !ok {
		return
	}
	_, _ = inspector.InspectAttention(context.Background(), "adapter attention")
}

func Example_metaladapterErr() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	_ = model.Err()
}

func Example_metaladapterClose() {
	model, err := LoadModelAsTextModel("/models/gemma4")
	if err != nil {
		return
	}
	_ = model.Close()
}
