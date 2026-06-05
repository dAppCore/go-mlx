// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// Examples for file-aware public API coverage.
func ExampleLoadModel() {
	model, err := LoadModel("/models/gemma4")
	if err != nil {
		return
	}
	defer model.Close()

	_ = model.Info()
}

func ExampleModel_Generate() {
	model, native := exampleRootModel("ok")

	text, err := model.Generate("prompt")

	core.Println(text, err == nil, native.lastGeneratePrompt)
	// Output: ok true prompt
}

func ExampleModel_Chat() {
	model, native := exampleRootModel("chat-ok")

	text, err := model.Chat([]inference.Message{{Role: "user", Content: "hello"}})

	core.Println(text, err == nil, native.lastChatMessages[0].Role)
	// Output: chat-ok true user
}

func ExampleModel_GenerateStream() {
	model, _ := exampleRootModel("stream", "-ok")

	text := ""
	for token := range model.GenerateStream(nil, "prompt") {
		text += token.Text
	}

	core.Println(text)
	// Output: stream-ok
}

func ExampleModel_ChatStream() {
	model, native := exampleRootModel("chat", "-stream")

	text := ""
	for token := range model.ChatStream(nil, []inference.Message{{Role: "user", Content: "hello"}}) {
		text += token.Text
	}

	core.Println(text, native.lastChatMessages[0].Content)
	// Output: chat-stream hello
}

func ExampleModel_Classify() {
	native := &fakeNativeModel{
		classifyResults: []metal.ClassifyResult{{Token: metal.Token{ID: 7, Text: "yes"}}},
	}
	model := &Model{model: native}

	results, err := model.Classify([]string{"approve?"}, WithReturnLogits())

	core.Println(results[0].Token.Text, err == nil, native.classifyReturnLogits)
	// Output: yes true true
}

func ExampleModel_BatchGenerate() {
	native := &fakeNativeModel{
		batchResults: []metal.BatchResult{{Tokens: []metal.Token{{ID: 1, Text: "first"}}}},
	}
	model := &Model{model: native}

	results, err := model.BatchGenerate([]string{"one"})

	core.Println(results[0].Tokens[0].Text, err == nil)
	// Output: first true
}

func ExampleModel_Err() {
	model := &Model{model: &fakeNativeModel{err: core.NewError("example failure")}}

	core.Println(model.Err() != nil)
	// Output: true
}

func ExampleModel_Metrics() {
	model := &Model{model: &fakeNativeModel{
		metrics: metal.Metrics{
			GeneratedTokens: 2,
			Adapter:         metal.AdapterInfo{Name: "demo-lora"},
		},
	}}

	metrics := model.Metrics()

	core.Println(metrics.GeneratedTokens, metrics.Adapter.Name)
	// Output: 2 demo-lora
}

func ExampleModel_ModelType() {
	model, _ := exampleRootModel()

	core.Println(model.ModelType())
	// Output: gemma4_text
}

func ExampleModel_Info() {
	model, _ := exampleRootModel()

	info := model.Info()

	core.Println(info.Architecture, info.ContextLength, info.Adapter.Name)
	// Output: gemma4_text 262144 demo-lora
}

func ExampleModel_InspectAttention() {
	model := &Model{model: &fakeNativeModel{
		attention: &metal.AttentionResult{
			Architecture: "gemma4_text",
			NumLayers:    2,
			NumHeads:     4,
		},
	}}

	snapshot, err := model.InspectAttention("prompt")

	core.Println(snapshot.Architecture, snapshot.NumLayers, snapshot.NumHeads, err == nil)
	// Output: gemma4_text 2 4 true
}

func ExampleModel_CaptureKV() {
	model := &Model{model: &fakeNativeModel{
		kvSnapshot: &metal.KVSnapshot{
			Architecture: "gemma4_text",
			Tokens:       []int32{1, 2, 3},
			NumLayers:    2,
		},
	}}

	snapshot, err := model.CaptureKV("prompt")

	core.Println(snapshot.Architecture, len(snapshot.Tokens), snapshot.NumLayers, err == nil)
	// Output: gemma4_text 3 2 true
}

func ExampleModel_ClearPromptCache() {
	model, native := exampleRootModel()

	err := model.ClearPromptCache()

	core.Println(native.clearPromptCacheCalls, err == nil)
	// Output: 1 true
}

func ExampleModel_Tokenizer() {
	model := &Model{tok: &Tokenizer{}}

	core.Println(model.Tokenizer() != nil)
	// Output: true
}

func ExampleModel_Close() {
	model, native := exampleRootModel()

	err := model.Close()

	core.Println(native.closeCalls, model.model == nil, err == nil)
	// Output: 1 true true
}

func ExampleNewLoRA() {
	model, native := exampleRootModel()

	adapter := NewLoRA(model, &LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj", "o_proj"},
		DType:      DTypeBFloat16,
	})

	core.Println(adapter == nil, native.lastLoRAConfig.Rank, native.lastLoRAConfig.TargetKeys[2])
	// Output: true 8 o_proj
}

func ExampleModel_MergeLoRA() {
	model, _ := exampleRootModel()

	merged := model.MergeLoRA(nil)

	core.Println(merged == model)
	// Output: true
}

func ExampleMatMul() {
	var a, b *Array
	_, _, _ = a, b, MatMul
}

func ExampleAdd() {
	var a, b *Array
	_, _, _ = a, b, Add
}

func ExampleMul() {
	var a, b *Array
	_, _, _ = a, b, Mul
}

func ExampleSoftmax() {
	var logits *Array
	_, _ = logits, Softmax
}

func ExampleSlice() {
	var values *Array
	_, _ = values, Slice
}

func ExampleReshape() {
	var values *Array
	_, _ = values, Reshape
}

func ExampleVJP() {
	_ = VJP
}

func ExampleJVP() {
	_ = JVP
}

func exampleRootModel(text ...string) (*Model, *fakeNativeModel) {
	native := &fakeNativeModel{
		info: metal.ModelInfo{
			Architecture:  "gemma4_text",
			ContextLength: 262144,
			Adapter: metal.AdapterInfo{
				Name:       "demo-lora",
				TargetKeys: []string{"q_proj", "v_proj", "o_proj"},
			},
		},
		modelType: "gemma4_text",
	}
	for i, token := range text {
		native.tokens = append(native.tokens, metal.Token{ID: int32(i + 1), Text: token})
	}
	return &Model{model: native}, native
}
