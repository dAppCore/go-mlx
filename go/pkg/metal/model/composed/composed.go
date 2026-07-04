// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package composed is the generic config-driven transformer the engine composes
// from a checkpoint config when no hand-written model package claims the family.
// It is the consumer the mixer registry was built for: a standard pre-norm SwiGLU
// transformer whose per-layer sequence mixer is resolved at load time by the kind
// the config declares (model_type for a uniform model, layer_types for a hybrid),
// through metal.MixerLoaderFor — softmax attention, GLA, RetNet, Mamba-2, RWKV-7,
// … all build the same way, with no central switch and no edit to any model file.
//
// gemma4 stays the hand-written softmax model; this is the path a Mamba/RWKV or a
// softmax+linear-attention hybrid runs. Cut 1 composes the pre-norm SwiGLU block
// (norm→mixer→add, norm→MLP→add); families whose block topology differs (pure
// Mamba with no MLP, RWKV's time/channel-mix) keep their own model package.
package composed

import (
	"slices"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	metal "dappco.re/go/mlx/pkg/metal"
)

// ComposedModel is a pre-norm SwiGLU transformer whose attention slot is a
// config-resolved sequence mixer per layer. It composes the SDK's neutral pieces
// (Embedding, RMSNormModule, SiLUMLP, the registry mixers) and satisfies
// metal.InternalModel, so the generate path drives it exactly like a hand-written
// model.
type ComposedModel struct {
	EmbedTokens *metal.Embedding
	Layers      []*composedLayer
	Norm        *metal.RMSNormModule
	Output      *metal.Linear

	Tok       *metal.Tokenizer
	Cfg       *metal.DenseConfig
	modelType string
}

// composedLayer is one pre-norm block: norm → mixer → residual, norm → MLP →
// residual. Mixer is whatever the config declared for this layer; Kind is kept
// for cache typing and diagnostics.
type composedLayer struct {
	InputNorm    *metal.RMSNormModule
	Mixer        metal.MixerCompute
	Kind         string
	PostAttnNorm *metal.RMSNormModule
	MLP          *metal.SiLUMLP
}

func init() {
	loader := func(modelPath string, _ []byte) (metal.InternalModel, error) {
		model, err := LoadComposed(modelPath)
		if err != nil {
			return nil, core.E("model.loadModel", "load composed model", err)
		}
		return model, nil
	}
	// A config opts into the generic path by declaring model_type "composed" or
	// "hybrid" with layer_types. A concrete family (a real Mamba/RWKV/hybrid
	// checkpoint with its own model_type) is wired to this loader when its
	// weight-naming + numerics are validated against a checkpoint — until then
	// the path is reachable by config without hijacking a family id.
	metal.RegisterModelLoader("composed", loader)
	metal.RegisterModelLoader("hybrid", loader)
}

// LoadComposed loads a config-composed model from a safetensors directory: parse
// the dense config, resolve each layer's mixer kind, and compose the block stack.
func LoadComposed(modelPath string) (*ComposedModel, error) {
	const op = "composed.LoadComposed"
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E(op, "load config", err)
	}
	data := []byte(str)

	cfg, err := metal.ParseDenseConfig(data)
	if err != nil {
		return nil, core.E(op, "parse config", err)
	}

	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E(op, "load tokenizer", err)
	}

	weights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E(op, "load weights", err)
	}

	m, err := buildComposed(cfg, weights, tok)
	if err != nil {
		return nil, core.E(op, "compose model", err)
	}

	allArrays := make([]*metal.Array, 0, len(weights))
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	metal.Materialize(allArrays...)
	core.Info("composed model loaded",
		"arch", m.modelType, "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads, "vocab", cfg.VocabSize,
	)
	return m, nil
}

// buildComposed assembles the model from a parsed config and a weights map. It is
// the disk-free core of LoadComposed so the composition (kind resolution, the
// registry dispatch, the cache layout, the loud refusals) is unit-testable from
// synthetic weights without loading a model.
func buildComposed(cfg *metal.DenseConfig, weights map[string]*metal.Array, tok *metal.Tokenizer) (*ComposedModel, error) {
	const op = "composed.build"

	kinds, err := resolveLayerKinds(cfg)
	if err != nil {
		return nil, core.E(op, "resolve layer kinds", err)
	}

	w := func(name string) *metal.Array { return metal.ResolveWeight(weights, name) }

	// vocab_size is a dimension — derive it from the embedding rows when the
	// config omits it, the same way the dense loaders do.
	if cfg.VocabSize == 0 {
		if ew := w("model.embed_tokens.weight"); ew != nil {
			if s := ew.Shape(); len(s) > 0 && s[0] > 0 {
				cfg.VocabSize = s[0]
			}
		}
	}

	embed := &metal.Embedding{Weight: w("model.embed_tokens.weight")}
	if embed.Weight == nil {
		return nil, core.E(op, "missing model.embed_tokens.weight", nil)
	}
	if embedScales := w("model.embed_tokens.scales"); embedScales != nil {
		embed.Scales = embedScales
		embed.Biases = w("model.embed_tokens.biases")
		if cfg.Quantization != nil {
			embed.GroupSize = cfg.Quantization.GroupSize
			embed.Bits = cfg.Quantization.Bits
		}
	}

	m := &ComposedModel{
		EmbedTokens: embed,
		Layers:      make([]*composedLayer, cfg.NumHiddenLayers),
		Norm:        &metal.RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   cfg.ModelType,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		kind := kinds[i]
		load, ok := metal.MixerLoaderFor(kind)
		if !ok {
			return nil, core.E(op, core.Sprintf("layer %d: unregistered mixer kind %q", i, kind), nil)
		}
		prefix := core.Sprintf("model.layers.%d", i)
		// The mixer's weights live under a model-specific sublayer (self_attn,
		// mixer, attn, mamba, …) discovered from the checkpoint — the mixer loaders
		// ask for bare leaves and the composed model owns the layout, so a hybrid
		// nesting it differently per family still resolves.
		mixerSub, err := discoverMixerSubpath(weights, i)
		if err != nil {
			return nil, core.E(op, core.Sprintf("layer %d", i), err)
		}
		mixerPrefix := prefix
		if mixerSub != "" {
			mixerPrefix = prefix + "." + mixerSub
		}
		bctx := metal.MixerBuildCtx{
			Linear: func(proj string) *metal.Linear {
				return metal.LoadLinear(weights, mixerPrefix+"."+proj, cfg.Quantization)
			},
			Weight:   func(name string) *metal.Array { return metal.ResolveWeight(weights, mixerPrefix+"."+name) },
			Cfg:      cfg.TransformerConfig,
			LayerIdx: i,
			Extra:    cfg, // softmax needs rope_theta + scale from the family config
		}
		mixer, err := load(bctx)
		if err != nil {
			return nil, core.E(op, core.Sprintf("layer %d: build mixer %q", i, kind), err)
		}
		m.Layers[i] = &composedLayer{
			InputNorm:    &metal.RMSNormModule{Weight: w(prefix + ".input_layernorm.weight")},
			Mixer:        mixer,
			Kind:         kind,
			PostAttnNorm: &metal.RMSNormModule{Weight: w(prefix + ".post_attention_layernorm.weight")},
			MLP: &metal.SiLUMLP{
				GateProj: metal.LoadLinear(weights, prefix+".mlp.gate_proj", cfg.Quantization),
				UpProj:   metal.LoadLinear(weights, prefix+".mlp.up_proj", cfg.Quantization),
				DownProj: metal.LoadLinear(weights, prefix+".mlp.down_proj", cfg.Quantization),
			},
		}
	}

	// lm_head when present; otherwise the tied embedding (the dense-family default).
	if lmHead := w("lm_head.weight"); lmHead != nil {
		if lmScales := w("lm_head.scales"); lmScales != nil && cfg.Quantization != nil {
			m.Output = metal.NewQuantizedLinearWithMode(lmHead, lmScales, w("lm_head.biases"), nil, cfg.Quantization.GroupSize, cfg.Quantization.Bits, cfg.Quantization.Mode)
		} else {
			m.Output = metal.NewLinear(lmHead, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}

	return m, nil
}

// discoverMixerSubpath finds the sublayer a layer's mixer weights nest under
// (self_attn, mixer, attn, mamba, …) by reading the checkpoint, rather than
// guessing a per-family name — a real hybrid nests the mixer model-specifically.
// The mixer and the MLP nest sub-projections (model.layers.N.SUB.proj.weight, ≥3
// components past the layer); the norms do not. Excluding the MLP leaves the
// mixer. Exactly one such sublayer → use it; none → the weights are bare (no
// nesting) and the caller resolves leaves directly; two or more → refuse, because
// picking one would be a silent random choice (map order is unspecified) — the
// silent wrong-geometry trap. The model./language_model. alias is resolved at
// load (ResolveWeight); a language_model.-prefixed key misses this raw-key scan
// and falls to the bare path, failing loud on the missing weight, not silently.
func discoverMixerSubpath(weights map[string]*metal.Array, layer int32) (string, error) {
	layerPrefix := core.Sprintf("model.layers.%d.", layer)
	subs := map[string]struct{}{}
	for key := range weights {
		if !core.HasPrefix(key, layerPrefix) {
			continue
		}
		parts := core.Split(core.TrimPrefix(key, layerPrefix), ".")
		if len(parts) >= 3 && parts[0] != "mlp" {
			subs[parts[0]] = struct{}{}
		}
	}
	switch len(subs) {
	case 0:
		return "", nil
	case 1:
		for s := range subs {
			return s, nil
		}
	}
	names := make([]string, 0, len(subs))
	for s := range subs {
		names = append(names, s)
	}
	slices.Sort(names)
	return "", core.E("composed.discoverMixerSubpath", core.Sprintf("ambiguous mixer subpath %v (exactly one non-mlp sublayer expected)", names), nil)
}

// resolveLayerKinds maps each layer to its declared mixer kind: layer_types when
// it lists one entry per layer (a hybrid), otherwise the model_type as a uniform
// kind for every layer. A "composed"/"hybrid" model_type with no usable
// layer_types is ambiguous and refused — the config must say what each layer is.
func resolveLayerKinds(cfg *metal.DenseConfig) ([]string, error) {
	const op = "composed.resolveLayerKinds"
	n := int(cfg.NumHiddenLayers)
	if n <= 0 {
		return nil, core.E(op, "num_hidden_layers must be > 0", nil)
	}
	kinds := make([]string, n)
	if len(cfg.LayerTypes) == n {
		for i, lt := range cfg.LayerTypes {
			kinds[i] = metal.NormalizeDenseLayerType(lt)
		}
		return kinds, nil
	}
	if len(cfg.LayerTypes) != 0 {
		return nil, core.E(op, core.Sprintf("layer_types length %d != num_hidden_layers %d", len(cfg.LayerTypes), n), nil)
	}
	uniform := metal.NormalizeDenseLayerType(cfg.ModelType)
	if uniform == "" || uniform == "composed" || uniform == "hybrid" {
		return nil, core.E(op, "needs per-layer layer_types or a mixer model_type", nil)
	}
	for i := range kinds {
		kinds[i] = uniform
	}
	return kinds, nil
}

// Forward runs the composed forward pass with standard causal attention.
func (m *ComposedModel) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

// ForwardMasked runs the forward pass; a nil mask is standard causal attention.
//
// The per-layer mixer context is built once and reused — only its Cache changes
// per layer (every other field is constant across the trunk for one forward).
// The mixers read MixerCtx during their Forward call and never retain the
// pointer, so a single mutated context is sound and saves one heap allocation
// per layer (the &MixerCtx{} literal escapes through the MixerCompute interface).
func (m *ComposedModel) ForwardMasked(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	mctx := metal.MixerCtx{B: B, L: L, Mask: mask}
	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		mctx.Cache = caches[i]
		hNext := layer.forward(h, &mctx, m.Cfg)
		metal.Free(h)
		h = hNext
	}
	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	return out
}

func (l *composedLayer) forward(x *metal.Array, mctx *metal.MixerCtx, cfg *metal.DenseConfig) *metal.Array {
	residual := x
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	mixOut, _ := l.Mixer.Forward(normed, mctx)
	metal.Free(normed)
	h := metal.Add(residual, mixOut)
	metal.Free(mixOut)

	normed2 := l.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.Forward(normed2)
	metal.Free(normed2)
	out := metal.Add(h, mlpOut)
	metal.Free(h, mlpOut)
	return out
}

// NewCache creates one cache per layer, typed by the layer mixer's declared
// state: a growing K/V cache for softmax attention (StateKVCache), the fixed-size
// recurrent holder for an SSM / linear-attention mixer (StateRecurrent). The
// caches stay 1:1 with layers, so Forward hands layer i its caches[i].
func (m *ComposedModel) NewCache() []metal.Cache {
	caches := make([]metal.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if layer == nil || layer.Mixer == nil {
			caches[i] = metal.NewKVCache()
			continue
		}
		// The KV factory types each layer by its mixer's declared shape — the
		// recurrent holder for SSM / linear-attn mixers, MLA's latent store for
		// MLA, the growing KV cache otherwise — instead of hand-picking here.
		caches[i] = metal.NewCacheForMixer(layer.Mixer, metal.CacheParams{})
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *ComposedModel) NumLayers() int { return len(m.Layers) }

// NumQueryHeads reports the attention query-head count (QueryHeadCounter).
func (m *ComposedModel) NumQueryHeads() int {
	if m.Cfg != nil {
		return int(m.Cfg.NumAttentionHeads)
	}
	return 0
}

// Tokenizer returns the model's tokenizer.
func (m *ComposedModel) Tokenizer() *metal.Tokenizer { return m.Tok }

// ModelType returns the architecture identifier the config declared.
func (m *ComposedModel) ModelType() string { return m.modelType }

// ApplyLoRA wraps the MLP projections this model owns directly. The per-layer
// mixer is opaque (a MixerCompute), so attention-target LoRA on a composed mixer
// is a later refinement (a LoRA-target capability on the mixer contract); the
// MLP path is the same SwiGLU every layer carries and is wired now.
func (m *ComposedModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	cfg = metal.NormalizeLoRAConfig(cfg)
	adapter := &metal.LoRAAdapter{
		Layers: make(map[string]*metal.LoRALinear),
		Config: cfg,
		Model:  m,
	}
	for i, layer := range m.Layers {
		if layer == nil || layer.MLP == nil {
			continue
		}
		for _, target := range cfg.TargetKeys {
			var proj *metal.Linear
			switch target {
			case "gate_proj":
				proj = layer.MLP.GateProj
			case "up_proj":
				proj = layer.MLP.UpProj
			case "down_proj":
				proj = layer.MLP.DownProj
			}
			if proj != nil {
				lora := metal.NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
				proj.LoRA = lora
				adapter.Layers[core.Sprintf("model.layers.%d.mlp.%s", i, target)] = lora
			}
		}
	}
	return adapter
}

// FillModelInfo fills architecture metadata (ModelInfoReporter).
func (m *ComposedModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(m.Cfg.VocabSize)
	info.HiddenSize = int(m.Cfg.HiddenSize)
	info.ContextLength = int(m.Cfg.MaxPositionEmbeddings)
	if m.Cfg.Quantization != nil {
		info.QuantBits = m.Cfg.Quantization.Bits
		info.QuantGroup = m.Cfg.Quantization.GroupSize
	}
}

// CloseModel releases the model's Metal arrays (ModelCloser): the embedding, the
// final norm, an untied output, and per layer the norms, the MLP, and the mixer's
// own weights when it implements MixerCloser.
func (m *ComposedModel) CloseModel() {
	if m == nil {
		return
	}
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeRMSNorm(m.Norm)
	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		metal.FreeLinear(m.Output)
	}
	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		metal.FreeRMSNorm(layer.InputNorm)
		metal.FreeRMSNorm(layer.PostAttnNorm)
		if closer, ok := layer.Mixer.(metal.MixerCloser); ok {
			closer.CloseMixer()
		}
		if layer.MLP != nil {
			metal.FreeLinear(layer.MLP.GateProj)
			metal.FreeLinear(layer.MLP.UpProj)
			metal.FreeLinear(layer.MLP.DownProj)
		}
	}
}

// Compile-time proof the composed model satisfies the engine interface and the
// capabilities the loader dispatches on.
var (
	_ metal.InternalModel     = (*ComposedModel)(nil)
	_ metal.ModelCloser       = (*ComposedModel)(nil)
	_ metal.QueryHeadCounter  = (*ComposedModel)(nil)
	_ metal.ModelInfoReporter = (*ComposedModel)(nil)
)
