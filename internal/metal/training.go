//go:build darwin && arm64

package metal

// ApplyLoRA injects LoRA adapters into the model's projection layers.
//
//	adapter := m.ApplyLoRA(metal.LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"}})
func (m *Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	return m.model.ApplyLoRA(cfg)
}

// Encode tokenises text into token IDs.
//
//	ids := m.Encode("Hello world") // → []int32{2, 9906, 1917}
func (m *Model) Encode(text string) []int32 {
	return m.tokenizer.Encode(text)
}

// Decode converts token IDs back to text.
//
//	text := m.Decode([]int32{9906, 1917}) // → "Hello world"
func (m *Model) Decode(ids []int32) string {
	return m.tokenizer.Decode(ids)
}

// NumLayers returns the number of transformer layers in the model.
//
//	fmt.Printf("model has %d layers\n", m.NumLayers()) // e.g. 28 for Gemma3-7B
func (m *Model) NumLayers() int {
	return m.model.NumLayers()
}

// Internal returns the underlying InternalModel for direct forward pass access.
//
//	im := m.Internal()
//	logits := im.Forward(tokens, caches)
func (m *Model) Internal() InternalModel {
	return m.model
}

// ArrayElement is the exported type constraint for FromValues.
type ArrayElement interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}
