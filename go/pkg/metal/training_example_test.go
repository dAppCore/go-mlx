// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleModel_ApplyLoRA() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	adapter := model.ApplyLoRA(LoRAConfig{
		Rank:       4,
		Alpha:      8,
		TargetKeys: []string{"q_proj", "o_proj"},
	})
	info := model.Adapter()

	core.Println(adapter.Config.Rank, adapter.Config.Scale, adapter.Config.TargetKeys, info.Rank, info.Scale, model.adapter == adapter)
	// Output: 4 2 [q_proj o_proj] 4 2 true
}

func ExampleModel_Encode() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	core.Println(model.Encode("hello"))
	// Output: [100 4 5 6 3]
}

func ExampleModel_Decode() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	core.Println(model.Decode([]int32{100, 4, 5, 6, 3}))
	// Output: hello
}

func ExampleModel_Tokenizer() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	core.Println(model.Tokenizer() != nil, model.Tokenizer().HasBOSToken())
	// Output: true true
}

func ExampleModel_NumLayers() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	core.Println(model.NumLayers())
	// Output: 3
}

func ExampleModel_Internal() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	internal := model.Internal()
	core.Println(internal.ModelType(), internal.NumLayers(), internal.Tokenizer() == model.Tokenizer())
	// Output: gemma4_text 3 true
}

func ExampleInternalModel_Forward() {
	model := exampleTrainingInternal()

	core.Println(model.Forward(nil, nil) == nil, model.forwardCalls)
	// Output: true 1
}

func ExampleInternalModel_ForwardMasked() {
	model := exampleTrainingInternal()

	core.Println(model.ForwardMasked(nil, nil, nil) == nil, model.maskedCalls)
	// Output: true 1
}

func ExampleInternalModel_NewCache() {
	model := exampleTrainingInternal()
	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.RotatingKVCache
}

func ExampleInternalModel_NumLayers() {
	model := exampleTrainingInternal()

	core.Println(model.NumLayers())
	// Output: 3
}

func ExampleInternalModel_Tokenizer() {
	model, _, cleanup := exampleTrainingModel()
	defer cleanup()

	core.Println(model.model.Tokenizer() == model.Tokenizer())
	// Output: true
}

func ExampleInternalModel_ModelType() {
	model := exampleTrainingInternal()

	core.Println(model.ModelType())
	// Output: gemma4_text
}

func ExampleInternalModel_ApplyLoRA() {
	model := exampleTrainingInternal()

	adapter := model.ApplyLoRA(LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	})

	core.Println(adapter.Config.Rank, adapter.Config.Scale, adapter.Config.TargetKeys, model.lora == adapter)
	// Output: 8 2 [q_proj v_proj] true
}

func exampleTrainingModel() (*Model, *exampleTrainingInternalModel, func()) {
	tok, cleanup := mustExampleTokenizer()
	internal := &exampleTrainingInternalModel{
		modelType: "gemma4_text",
		layers:    3,
		tokenizer: tok,
	}
	model := &Model{
		model:     internal,
		tokenizer: tok,
		modelType: "gemma4_text",
		device:    DeviceCPU,
	}
	return model, internal, cleanup
}

func exampleTrainingInternal() *exampleTrainingInternalModel {
	return &exampleTrainingInternalModel{
		modelType: "gemma4_text",
		layers:    3,
	}
}

type exampleTrainingInternalModel struct {
	modelType    string
	layers       int
	tokenizer    *Tokenizer
	forwardCalls int
	maskedCalls  int
	lora         *LoRAAdapter
}

func (m *exampleTrainingInternalModel) Forward(_ *Array, _ []Cache) *Array {
	m.forwardCalls++
	return nil
}

func (m *exampleTrainingInternalModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array {
	m.maskedCalls++
	return nil
}

func (m *exampleTrainingInternalModel) NewCache() []Cache {
	return []Cache{NewKVCache(), NewRotatingKVCache(64)}
}

func (m *exampleTrainingInternalModel) NumLayers() int {
	return m.layers
}

func (m *exampleTrainingInternalModel) Tokenizer() *Tokenizer {
	return m.tokenizer
}

func (m *exampleTrainingInternalModel) ModelType() string {
	return m.modelType
}

func (m *exampleTrainingInternalModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	cfg = normalizeLoRAConfig(cfg)
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{},
		Config: cfg,
		Model:  m,
	}
	m.lora = adapter
	return adapter
}

// minimalTokenizerJSON + mustExampleTokenizer are a self-contained fixture for the training
// examples here: the shared copy lives with the tokenizer's own example tests in
// pkg/tokenizer (test helpers don't cross packages), so the metal training example carries
// its own. Loads a tiny BPE tokenizer through the metal.LoadTokenizer alias.
const minimalTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "▁": 4, "he": 5, "ll": 6, "▁h": 7},
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

func mustExampleTokenizer() (*Tokenizer, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-metal-tokenizer-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(minimalTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		panic(result.Value)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return tok, func() { core.RemoveAll(dir) }
}
