// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
)

// LoadAdapter fuses a LoRA adapter into a temporary native-loadable pack and hot-swaps the
// no-cgo token model to that fused pack. This mirrors the metal adapter lifecycle at the
// inference.TextModel surface while keeping go/pkg/native free of cgo and adapter-pack parsing.
func (m *nativeTextModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if m == nil {
		return inference.AdapterIdentity{}, errMLXModelNil
	}
	if path == "" {
		return inference.AdapterIdentity{}, core.NewError("mlx.native.LoadAdapter: adapter path is required")
	}
	if m.modelPath == "" {
		return inference.AdapterIdentity{}, core.NewError("mlx.native.LoadAdapter: model path is not available")
	}
	if err := m.UnloadAdapter(); err != nil {
		return inference.AdapterIdentity{}, err
	}

	tmp := core.MkdirTemp("", "go-mlx-native-lora-*")
	if !tmp.OK {
		return inference.AdapterIdentity{}, tmp.Value.(error)
	}
	tmpRoot := tmp.Value.(string)
	output := core.PathJoin(tmpRoot, "fused")
	result, err := FuseLoRAIntoModelPack(context.Background(), FuseLoRAOptions{
		ModelPath:   m.modelPath,
		AdapterPath: path,
		OutputPath:  output,
		Labels:      map[string]string{"runtime": "native"},
	})
	if err != nil {
		_ = nativeRemoveAll(tmpRoot)
		return inference.AdapterIdentity{}, err
	}

	tm, err := native.LoadTokenModelDir(result.OutputPath, m.maxLen)
	if err != nil {
		_ = nativeRemoveAll(tmpRoot)
		return inference.AdapterIdentity{}, err
	}
	id := toInferenceRootAdapterIdentity(result.Adapter)
	oldTM, oldPack := m.swapNativeTokenModel(tm, id, tmpRoot)
	_ = closeNativeTokenModel(oldTM)
	_ = nativeRemoveAll(oldPack)
	return id, nil
}

// UnloadAdapter restores the source model path after LoadAdapter swapped in a fused pack.
func (m *nativeTextModel) UnloadAdapter() error {
	if m == nil {
		return errMLXModelNil
	}
	if m.ActiveAdapter().Path == "" && m.ActiveAdapter().Hash == "" {
		return nil
	}
	if m.modelPath == "" {
		return core.NewError("mlx.native.UnloadAdapter: model path is not available")
	}
	tm, err := native.LoadTokenModelDir(m.modelPath, m.maxLen)
	if err != nil {
		return err
	}
	oldTM, oldPack := m.swapNativeTokenModel(tm, inference.AdapterIdentity{}, "")
	_ = closeNativeTokenModel(oldTM)
	return nativeRemoveAll(oldPack)
}

func (m *nativeTextModel) ActiveAdapter() inference.AdapterIdentity {
	if m == nil {
		return inference.AdapterIdentity{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return cloneNativeAdapterIdentity(m.adapter)
}

func (m *nativeTextModel) swapNativeTokenModel(tm model.TokenModel, adapter inference.AdapterIdentity, adapterPack string) (model.TokenModel, string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	oldTM, oldPack := m.tm, m.adapterPack
	m.clearNativePromptCacheLocked()
	m.tm = tm
	m.adapter = cloneNativeAdapterIdentity(adapter)
	m.adapterPack = adapterPack
	return oldTM, oldPack
}

func (m *nativeTextModel) clearNativePromptCacheLocked() {
	if m.cacheSess != nil {
		m.cacheSess.ClearPromptCache()
		if c, ok := m.cacheSess.(interface{ Close() error }); ok {
			_ = c.Close()
		}
		m.cacheSess = nil
	}
	m.cacheBlocks = nil
}

func closeNativeTokenModel(tm any) error {
	if c, ok := tm.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func nativeRemoveAll(path string) error {
	if path == "" {
		return nil
	}
	if result := core.RemoveAll(path); !result.OK {
		return result.Value.(error)
	}
	return nil
}

func cloneNativeAdapterIdentity(id inference.AdapterIdentity) inference.AdapterIdentity {
	id.TargetKeys = core.SliceClone(id.TargetKeys)
	if id.Labels != nil {
		labels := make(map[string]string, len(id.Labels))
		for k, v := range id.Labels {
			labels[k] = v
		}
		id.Labels = labels
	}
	return id
}
