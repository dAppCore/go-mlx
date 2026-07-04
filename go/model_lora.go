// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// model_lora.go: Model-level LoRA adapter management — apply / load / swap / unload
// / merge of LoRA adapters on a loaded model, plus adapter-info reconciliation.

// NewLoRA applies a LoRA adapter to a loaded model. A nil config leaves LoRA
// defaults to the native model normaliser; pass DefaultLoRAConfig explicitly
// when the generic q/v target set is required.
func NewLoRA(model *Model, cfg *LoRAConfig) *LoRAAdapter {
	if model == nil || model.model == nil {
		return nil
	}
	var mcfg LoRAConfig
	if cfg != nil {
		mcfg = *cfg
	}
	adapter := model.model.ApplyLoRA(spine.ToMetalLoRAConfig(mcfg))
	// ApplyLoRA mutates the native model's adapter identity — refresh the
	// cached parserHint so the next Generate / Chat picks up the new
	// adapter name in its parser dispatch without re-reading m.model.Info()
	// per call.
	model.refreshParserHint()
	return adapter
}

// LoadLoRA loads a saved adapter package into a loaded model and returns it.
func (m *Model) LoadLoRA(path string) (*LoRAAdapter, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	info, err := lora.InspectAdapter(path)
	if err != nil {
		return nil, err
	}
	loader, ok := m.model.(nativeLoRALoader)
	if !ok {
		return nil, errMLXLoRALoadUnsupp
	}
	adapter, err := loader.LoadLoRA(path)
	if err != nil {
		return nil, err
	}
	m.adapterInfo = mergeLoadedAdapterInfo(info, adapterInfoFromLoadedLoRA(adapter))
	m.cfg.AdapterPath = path
	// Adapter identity changed — refresh the cached parserHint so the next
	// Generate / Chat picks up the new adapter name without paying for an
	// m.model.Info() fan-out per call.
	m.refreshParserHint()
	return adapter, nil
}

func adapterInfoFromLoadedLoRA(adapter *metal.LoRAAdapter) lora.AdapterInfo {
	if adapter == nil {
		return lora.AdapterInfo{}
	}
	targetKeys := adapter.Config.TargetKeys
	if len(targetKeys) == 0 {
		targetKeys = adapter.Config.TargetLayers
	}
	return lora.AdapterInfo{
		Rank:       adapter.Config.Rank,
		Alpha:      adapter.Config.Alpha,
		Scale:      adapter.Config.Scale,
		TargetKeys: core.SliceClone(targetKeys),
	}
}

func mergeLoadedAdapterInfo(inspected, loaded lora.AdapterInfo) lora.AdapterInfo {
	if inspected.IsEmpty() {
		return loaded
	}
	if loaded.IsEmpty() {
		return inspected
	}
	out := inspected
	if out.Name == "" {
		out.Name = loaded.Name
	}
	if out.Path == "" {
		out.Path = loaded.Path
	}
	if out.Hash == "" {
		out.Hash = loaded.Hash
	}
	if loaded.Rank != 0 {
		out.Rank = loaded.Rank
	}
	if loaded.Alpha != 0 {
		out.Alpha = loaded.Alpha
	}
	if loaded.Scale != 0 {
		out.Scale = loaded.Scale
	}
	if len(loaded.TargetKeys) > 0 {
		out.TargetKeys = core.SliceClone(loaded.TargetKeys)
	}
	return out
}

// UnloadLoRA removes the active inference adapter when the backend supports it.
func (m *Model) UnloadLoRA() error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if m.adapterInfo.IsEmpty() {
		return nil
	}
	unloader, ok := m.model.(nativeLoRAUnloader)
	if !ok {
		return errMLXLoRAUnloadUnsupp
	}
	if err := unloader.UnloadLoRA(); err != nil {
		return err
	}
	m.adapterInfo = lora.AdapterInfo{}
	m.cfg.AdapterPath = ""
	// Adapter cleared — refresh the cached parserHint so the next Generate
	// / Chat reads the post-unload adapter name (may fall back to the
	// native model's AdapterInfo.Name) without re-entering m.model.Info()
	// per call.
	m.refreshParserHint()
	return nil
}

// SwapLoRA replaces the active inference adapter with another adapter package.
func (m *Model) SwapLoRA(path string) (*LoRAAdapter, error) {
	if err := m.UnloadLoRA(); err != nil {
		return nil, err
	}
	return m.LoadLoRA(path)
}

// MergeLoRA returns the current model with the adapter applied in-place.
func (m *Model) MergeLoRA(adapter *LoRAAdapter) *Model {
	if adapter == nil {
		return m
	}
	adapter.Merge()
	return m
}
