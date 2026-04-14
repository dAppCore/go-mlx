//go:build darwin && arm64 && !nomlx

package metal

import "dappco.re/go/core"

// ApplyLoRA injects LoRA adapters into the model's projection layers.
//
//	adapter := m.ApplyLoRA(metal.LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"}})
func (m *Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	var adapter *LoRAAdapter
	if err := m.withDevice(func() {
		adapter = m.model.ApplyLoRA(cfg)
	}); err != nil {
		core.Error("mlx: apply lora", "error", err)
	}
	return adapter
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

// Tokenizer returns the loaded tokenizer for direct encode/decode access.
func (m *Model) Tokenizer() *Tokenizer {
	return m.tokenizer
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
	return &deviceInternalModel{device: m.modelDevice(), inner: m.model}
}

type deviceInternalModel struct {
	device DeviceType
	inner  InternalModel
}

func (m *deviceInternalModel) Forward(tokens *Array, caches []Cache) *Array {
	var out *Array
	if err := withDefaultDevice(m.device, func() {
		out = m.inner.Forward(tokens, caches)
	}); err != nil {
		core.Error("mlx: internal forward", "error", err)
	}
	return out
}

func (m *deviceInternalModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	var out *Array
	if err := withDefaultDevice(m.device, func() {
		out = m.inner.ForwardMasked(tokens, mask, caches)
	}); err != nil {
		core.Error("mlx: internal masked forward", "error", err)
	}
	return out
}

func (m *deviceInternalModel) NewCache() []Cache {
	return m.inner.NewCache()
}

func (m *deviceInternalModel) NumLayers() int {
	return m.inner.NumLayers()
}

func (m *deviceInternalModel) Tokenizer() *Tokenizer {
	return m.inner.Tokenizer()
}

func (m *deviceInternalModel) ModelType() string {
	return m.inner.ModelType()
}

func (m *deviceInternalModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	var adapter *LoRAAdapter
	if err := withDefaultDevice(m.device, func() {
		adapter = m.inner.ApplyLoRA(cfg)
	}); err != nil {
		core.Error("mlx: internal apply lora", "error", err)
	}
	return adapter
}

// ArrayElement is the exported type constraint for FromValues.
type ArrayElement interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}
