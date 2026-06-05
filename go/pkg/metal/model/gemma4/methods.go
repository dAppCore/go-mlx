// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func (m *Gemma4Model) NewCache() []metal.Cache {
	m.ensureCacheLayout()

	numCaches := 0
	for _, cacheIdx := range m.CacheIndexByLayer {
		if cacheIdx >= 0 {
			numCaches++
		}
	}
	caches := make([]metal.Cache, numCaches)
	for layerIdx, cacheIdx := range m.CacheIndexByLayer {
		if cacheIdx < 0 {
			continue
		}
		if m.Layers[layerIdx].LayerType == "full_attention" {
			caches[cacheIdx] = metal.NewKVCache()
		} else {
			caches[cacheIdx] = metal.NewRotatingKVCache(int(m.Cfg.SlidingWindow))
		}
	}
	return caches
}

// EngineFeatures declares the engine fast-path kernels Gemma 4 activates. Today
// this is the accepted validated set (identical greedy token hash vs the generic
// path); gemma4-specific axes (cache / attention algo selection) are declared
// here as they graduate from the diagnostic runtime gates. backend.LoadAndInit
// applies this when the model loads, so serve and the benchmarks run it alike.
func (m *Gemma4Model) EngineFeatures() metal.EngineFeatures {
	return metal.DefaultEngineFeatures()
}

// Compile-time proof Gemma4Model satisfies the engine-feature declaration
// capability the loader (backend.LoadAndInit) dispatches on.
var _ metal.EngineFeaturesModel = (*Gemma4Model)(nil)

// NumLayers returns the number of transformer layers.
func (m *Gemma4Model) NumLayers() int { return len(m.Layers) }

// NumQueryHeads reports the attention query-head count for KV/attention
// extraction (QueryHeadCounter). Zero when the config is unavailable.
func (m *Gemma4Model) NumQueryHeads() int {
	if m.Cfg != nil {
		return int(m.Cfg.NumAttentionHeads)
	}
	return 0
}

// ResolveLoRALinear resolves a LoRA-targetable projection by path
// (LoRALinearResolver). Returns nil for an unknown layer or path.
func (m *Gemma4Model) ResolveLoRALinear(layerIdx int, projPath string) *metal.Linear {
	if m == nil || layerIdx < 0 || layerIdx >= len(m.Layers) {
		return nil
	}
	layer := m.Layers[layerIdx]
	if layer == nil {
		return nil
	}
	switch projPath {
	case "self_attn.q_proj":
		if layer.Attention == nil {
			return nil
		}
		return layer.Attention.QProj
	case "self_attn.k_proj":
		if layer.Attention == nil {
			return nil
		}
		return layer.Attention.KProj
	case "self_attn.v_proj":
		if layer.Attention == nil {
			return nil
		}
		return layer.Attention.VProj
	case "self_attn.o_proj":
		if layer.Attention == nil {
			return nil
		}
		return layer.Attention.OProj
	case "mlp.gate_proj":
		if layer.MLP == nil {
			return nil
		}
		return layer.MLP.GateProj
	case "mlp.up_proj":
		if layer.MLP == nil {
			return nil
		}
		return layer.MLP.UpProj
	case "mlp.down_proj":
		if layer.MLP == nil {
			return nil
		}
		return layer.MLP.DownProj
	case "per_layer_input_gate":
		return layer.PerLayerInputGate
	case "per_layer_projection":
		return layer.PerLayerProjection
	case "router.proj":
		if layer.Router != nil {
			return layer.Router.Proj
		}
	}
	return nil
}

// RecordCacheTopology records Gemma 4's local/global sliding-window KV-cache
// layout into the profile, on top of the generic per-cache pass
// (CacheTopologyRecorder).
func (m *Gemma4Model) RecordCacheTopology(profile *metal.CacheProfile, caches []metal.Cache) {
	if profile == nil || m == nil || m.Cfg == nil {
		return
	}
	m.ensureCacheLayout()
	profile.LocalWindowTokens = int(m.Cfg.SlidingWindow)
	for layerIdx, cacheIdx := range m.CacheIndexByLayer {
		if cacheIdx < 0 {
			profile.SharedLayers++
			continue
		}
		if int(cacheIdx) >= len(caches) || layerIdx >= len(m.Layers) {
			continue
		}
		cache := caches[cacheIdx]
		tokens := metal.CacheLen(cache)
		capacity, bounded := metal.CacheCapacity(cache)
		if m.Layers[layerIdx].LayerType == "full_attention" {
			profile.GlobalCaches++
			profile.MaxGlobalTokens = max(profile.MaxGlobalTokens, tokens)
			profile.MaxGlobalCapacity = max(profile.MaxGlobalCapacity, capacity)
			continue
		}
		profile.LocalCaches++
		profile.MaxLocalTokens = max(profile.MaxLocalTokens, tokens)
		profile.MaxLocalCapacity = max(profile.MaxLocalCapacity, capacity)
		if profile.LocalWindowTokens > 0 && (tokens > profile.LocalWindowTokens || capacity > profile.LocalWindowTokens || !bounded) {
			profile.LocalWindowLeaked = true
		}
	}
}

// AttentionCacheLayout maps each layer to its KV-cache index following Gemma 4's
// shared local/global cache layout (AttentionCacheLayouter); -1 for a layer with
// no own cache.
func (m *Gemma4Model) AttentionCacheLayout(numLayers, numCaches int) []int {
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	m.ensureCacheLayout()
	for layerIdx := 0; layerIdx < numLayers && layerIdx < len(m.PreviousKVs); layerIdx++ {
		ownerIdx := int(m.PreviousKVs[layerIdx])
		if ownerIdx < 0 || ownerIdx >= len(m.CacheIndexByLayer) {
			continue
		}
		cacheIdx := int(m.CacheIndexByLayer[ownerIdx])
		if cacheIdx < 0 || cacheIdx >= numCaches {
			continue
		}
		cacheIndexByLayer[layerIdx] = cacheIdx
	}
	return cacheIndexByLayer
}

// FixedSlidingPrefillChunkLimit reports the largest safe prefill chunk for Gemma
// 4's fixed-size sliding-window caches, or 0 when there is no sliding window
// (FixedSlidingPrefillLimiter).
func (m *Gemma4Model) FixedSlidingPrefillChunkLimit(caches []metal.Cache) int {
	if m == nil || m.Cfg == nil || m.Cfg.SlidingWindow <= 0 {
		return 0
	}
	limit := int(m.Cfg.SlidingWindow)
	for _, cache := range caches {
		fixed, ok := cache.(*metal.FixedKVCache)
		if !ok || fixed == nil || fixed.MaxSize() <= 0 {
			continue
		}
		if limit <= 0 || fixed.MaxSize() < limit {
			limit = fixed.MaxSize()
		}
	}
	return limit
}

// Tokenizer returns the model's tokenizer.
func (m *Gemma4Model) Tokenizer() *metal.Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *Gemma4Model) ModelType() string { return m.modelType }

// ApplyLoRA wraps target projection layers with LoRA adapters for training.
func (m *Gemma4Model) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	cfg = metal.NormalizeGemma4LoRAConfig(cfg)
	adapter := &metal.LoRAAdapter{
		Layers: make(map[string]*metal.LoRALinear),
		Config: cfg,
		Model:  m,
	}

	if m == nil {
		return adapter
	}
	for i, layer := range m.Layers {
		if layer == nil {
			continue
		}
		for _, target := range cfg.TargetKeys {
			projPath, ok := metal.Gemma4LoRATargetPath(target)
			if !ok {
				continue
			}
			proj := m.ResolveLoRALinear(i, projPath)
			if proj != nil {
				lora := metal.NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
				proj.LoRA = lora
				adapter.Layers[core.Sprintf("model.layers.%d.%s", i, projPath)] = lora
			}
		}
	}

	return adapter
}

func (v *Gemma4Model) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.NumHeads = int(v.Cfg.NumAttentionHeads)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	info.Gemma4SlidingWindow = int(v.Cfg.SlidingWindow)
	info.DefaultOutputLength = int(v.Cfg.DefaultOutputLength)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}
