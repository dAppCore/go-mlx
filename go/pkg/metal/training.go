// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

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
	if adapter != nil {
		m.clearPromptCache()
		m.adapter = adapter
		m.adapterInfo = adapterInfoFromLoRA("", adapter)
	}
	return adapter
}

// LoadLoRA injects a saved adapter package into the loaded model and returns it.
func (m *Model) LoadLoRA(path string) (*LoRAAdapter, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	var (
		adapter *LoRAAdapter
		loadErr error
	)
	if err := m.withDevice(func() {
		if m.adapter != nil {
			m.adapter.Unload()
			m.adapter = nil
			m.adapterInfo = AdapterInfo{}
			m.clearPromptCache()
		}
		adapter, loadErr = loadLoRAAdapter(m.model, path)
	}); err != nil {
		return nil, core.E("mlx.LoadLoRA", "select device", err)
	}
	if loadErr != nil {
		return nil, loadErr
	}
	m.clearPromptCache()
	m.adapter = adapter
	m.adapterInfo = adapterInfoFromLoRA(path, adapter)
	return adapter, nil
}

// UnloadLoRA removes the active adapter from projection layers.
func (m *Model) UnloadLoRA() error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	if m.adapter == nil {
		return nil
	}
	if err := m.withDevice(func() {
		m.adapter.Unload()
		m.adapter = nil
		m.adapterInfo = AdapterInfo{}
		m.clearPromptCache()
	}); err != nil {
		return core.E("mlx.UnloadLoRA", "select device", err)
	}
	return nil
}

// Adapter returns the active adapter identity.
func (m *Model) Adapter() AdapterInfo {
	if m == nil {
		return AdapterInfo{}
	}
	return cloneMetalAdapterInfo(m.adapterInfo)
}

func adapterInfoFromLoRA(path string, adapter *LoRAAdapter) AdapterInfo {
	if adapter == nil {
		return AdapterInfo{}
	}
	cfg := normalizeLoRAConfig(adapter.Config)
	info := AdapterInfo{
		Name:       core.PathBase(path),
		Path:       path,
		Rank:       cfg.Rank,
		Alpha:      cfg.Alpha,
		Scale:      cfg.Scale,
		TargetKeys: append([]string(nil), cfg.TargetKeys...),
	}
	info.Hash = core.SHA256HexString(core.Join("\n", info.Name, info.Path, core.Sprintf("%d", info.Rank), core.Sprintf("%f", info.Alpha), core.Sprintf("%f", info.Scale), core.Join(",", info.TargetKeys...)))
	if path == "" {
		info.Hash = core.SHA256HexString(core.Join("\n", core.Sprintf("%d", info.Rank), core.Sprintf("%f", info.Alpha), core.Sprintf("%f", info.Scale), core.Join(",", info.TargetKeys...)))
	}
	return info
}

func cloneMetalAdapterInfo(info AdapterInfo) AdapterInfo {
	info.TargetKeys = append([]string(nil), info.TargetKeys...)
	return info
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

func (m *deviceInternalModel) ForwardLastTokenLogits(tokens *Array, mask *Array, caches []Cache) *Array {
	lastModel, ok := m.inner.(LastTokenLogitsModel)
	if !ok {
		return m.ForwardMasked(tokens, mask, caches)
	}
	var out *Array
	if err := withDefaultDevice(m.device, func() {
		out = lastModel.ForwardLastTokenLogits(tokens, mask, caches)
	}); err != nil {
		core.Error("mlx: internal last-token forward", "error", err)
	}
	return out
}

func (m *deviceInternalModel) ForwardGreedyToken(tokens *Array, mask *Array, caches []Cache) *Array {
	greedyModel, ok := m.inner.(GreedyTokenModel)
	if !ok {
		logits := m.ForwardMasked(tokens, mask, caches)
		token := Argmax(logits, -1, false)
		Free(logits)
		return token
	}
	var out *Array
	if err := withDefaultDevice(m.device, func() {
		out = greedyModel.ForwardGreedyToken(tokens, mask, caches)
	}); err != nil {
		core.Error("mlx: internal Greedy-token forward", "error", err)
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
