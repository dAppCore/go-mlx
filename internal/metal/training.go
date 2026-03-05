//go:build darwin && arm64

package metal

// Training bridge methods for Model.
// These expose the InternalModel's training surface through the public Model wrapper,
// enabling go-mlx/training.go to satisfy inference.TrainableModel.

// ApplyLoRA injects LoRA adapters into the underlying model's projection layers.
func (m *Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	return m.model.ApplyLoRA(cfg)
}

// Encode tokenises text into token IDs using the model's tokeniser.
func (m *Model) Encode(text string) []int32 {
	return m.tokenizer.Encode(text)
}

// Decode converts token IDs back to text.
func (m *Model) Decode(ids []int32) string {
	return m.tokenizer.Decode(ids)
}

// NumLayers returns the number of transformer layers.
func (m *Model) NumLayers() int {
	return m.model.NumLayers()
}

// Internal returns the underlying InternalModel for direct forward pass access.
// Used by training loops that need Forward() and NewCache().
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
