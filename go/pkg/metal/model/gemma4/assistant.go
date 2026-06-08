// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Gemma4AssistantConfig holds the metadata that makes a Gemma 4 assistant
// checkpoint different from a standalone Gemma 4 text model.
type Gemma4AssistantConfig struct {
	ModelType                string
	BackboneHiddenSize       int32
	NumCentroids             int32
	CentroidIntermediateTopK int32
	UseOrderedEmbeddings     bool
	TextConfig               *Gemma4TextConfig
}

// Gemma4AssistantModel is the attached Gemma 4 MTP drafter. It is not an
// InternalModel because it borrows target-model hidden state and K/V caches.
type Gemma4AssistantModel struct {
	EmbedTokens     *metal.Embedding
	Layers          []*Gemma4AssistantLayer
	Norm            *metal.RMSNormModule
	PreProjection   *metal.Linear
	PostProjection  *metal.Linear
	MaskedCentroids *metal.Linear
	TokenOrdering   *metal.Array

	Tok *metal.Tokenizer
	Cfg *Gemma4TextConfig

	BackboneHiddenSize       int32
	NumCentroids             int32
	CentroidIntermediateTopK int32
	UseOrderedEmbeddings     bool
}

// Gemma4AssistantLayer is one MTP drafter block. Its attention owns Q/O only;
// K/V are supplied by the target model's matching cache stream.
type Gemma4AssistantLayer struct {
	InputNorm    *metal.RMSNormModule
	Attention    *Gemma4AssistantAttention
	PostAttnNorm *metal.RMSNormModule
	PreFFNorm    *metal.RMSNormModule
	MLP          *metal.MLP
	PostFFNorm   *metal.RMSNormModule
	LayerScalar  *metal.Array
	LayerType    string
	IsSliding    bool
	LayerIdx     int32
}

// Gemma4AssistantAttention is the assistant-side Q projection and output
// projection used with target-side K/V cache tensors.
type Gemma4AssistantAttention struct {
	QProj *metal.Linear
	OProj *metal.Linear
	QNorm *metal.RMSNormModule

	HeadDim        int32
	NHeads         int32
	Scale          float32
	RopeBase       float32
	RopeRotatedDim int32
	RopeFreqs      *metal.Array
}

func parseGemma4AssistantConfig(data []byte) (*Gemma4AssistantConfig, error) {
	var wrapper struct {
		ModelType                string `json:"model_type"`
		BackboneHiddenSize       int32  `json:"backbone_hidden_size"`
		NumCentroids             int32  `json:"num_centroids"`
		CentroidIntermediateTopK int32  `json:"centroid_intermediate_top_k"`
		UseOrderedEmbeddings     bool   `json:"use_ordered_embeddings"`
	}
	if result := core.JSONUnmarshal(data, &wrapper); !result.OK {
		return nil, core.E("gemma4.assistant.parseConfig", "parse assistant config", nil)
	}
	textCfg, err := parseGemma4Config(data)
	if err != nil {
		return nil, core.E("gemma4.assistant.parseConfig", "parse text config", err)
	}
	cfg := &Gemma4AssistantConfig{
		ModelType:                wrapper.ModelType,
		BackboneHiddenSize:       wrapper.BackboneHiddenSize,
		NumCentroids:             wrapper.NumCentroids,
		CentroidIntermediateTopK: wrapper.CentroidIntermediateTopK,
		UseOrderedEmbeddings:     wrapper.UseOrderedEmbeddings,
		TextConfig:               textCfg,
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_assistant"
	}
	// The larger QAT drafters ship as gemma4_unified_assistant (unified-text
	// variant: K=V, multi-head KV) but are the same 4-layer MTP drafter that
	// borrows the target's K/V — the unified-ness lives in the target it attaches
	// to, not the drafter's own forward. Accept it as an assistant; the text
	// config keeps its unified model_type so per-layer RoPE/heads parse correctly.
	unified := cfg.ModelType == "gemma4_unified_assistant"
	if cfg.TextConfig != nil {
		if unified {
			cfg.TextConfig.ModelType = "gemma4_unified_text"
		} else {
			cfg.TextConfig.ModelType = "gemma4_assistant"
		}
	}
	if err := validateGemma4AssistantConfig(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func validateGemma4AssistantConfig(cfg *Gemma4AssistantConfig) error {
	if cfg == nil || cfg.TextConfig == nil {
		return core.NewError("gemma4.assistant config is nil")
	}
	if cfg.ModelType != "gemma4_assistant" && cfg.ModelType != "gemma4_unified_assistant" {
		return core.NewError("gemma4.assistant config has unsupported model_type: " + cfg.ModelType)
	}
	if cfg.BackboneHiddenSize <= 0 {
		return core.NewError("gemma4.assistant config has invalid backbone_hidden_size")
	}
	if cfg.TextConfig.HiddenSize <= 0 {
		return core.NewError("gemma4.assistant config has invalid hidden_size")
	}
	if cfg.TextConfig.NumHiddenLayers <= 0 {
		return core.NewError("gemma4.assistant config has invalid num_hidden_layers")
	}
	if cfg.TextConfig.NumAttentionHeads <= 0 {
		return core.NewError("gemma4.assistant config has invalid num_attention_heads")
	}
	if cfg.TextConfig.HeadDim <= 0 {
		return core.NewError("gemma4.assistant config has invalid head_dim")
	}
	if cfg.UseOrderedEmbeddings && cfg.NumCentroids <= 0 {
		return core.NewError("gemma4.assistant ordered embeddings require num_centroids")
	}
	return nil
}

// LoadGemma4Assistant loads and validates a Gemma 4 assistant drafter
// checkpoint. The returned value is intended to be attached to a target Gemma 4
// model; standalone text generation remains unsupported for this architecture.
func LoadGemma4Assistant(modelPath string) (*Gemma4AssistantModel, error) {
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gemma4.assistant.Load", "load config", err)
	}
	cfg, err := parseGemma4AssistantConfig([]byte(str))
	if err != nil {
		return nil, core.E("gemma4.assistant.Load", "parse config", err)
	}
	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma4.assistant.Load", "load tokenizer", err)
	}
	rawWeights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma4.assistant.Load", "load weights", err)
	}
	weights := sanitizeGemma4Weights(rawWeights)
	m := buildGemma4AssistantFromWeights(cfg, weights, tok)

	loadSucceeded := false
	defer func() {
		if loadSucceeded {
			return
		}
		retained := gemma4AssistantRetainedWeights(m)
		gemma4FreeUnusedWeights(weights, retained)
		closeGemma4Assistant(m)
		metal.ClearCache()
	}()

	if err := validateGemma4AssistantModel(m); err != nil {
		return nil, core.E("gemma4.assistant.Load", "validate tensors", err)
	}
	retained := gemma4AssistantRetainedWeights(m)
	gemma4FreeUnusedWeights(weights, retained)
	gemma4MaterializeRetainedWeights(retained, nil)
	loadSucceeded = true
	return m, nil
}

func buildGemma4AssistantFromWeights(cfg *Gemma4AssistantConfig, weights map[string]*metal.Array, tok *metal.Tokenizer) *Gemma4AssistantModel {
	text := cfg.TextConfig
	m := &Gemma4AssistantModel{
		EmbedTokens:              gemma4Embedding(weights, "model.embed_tokens", text.Quantization),
		Layers:                   make([]*Gemma4AssistantLayer, text.NumHiddenLayers),
		Norm:                     &metal.RMSNormModule{Weight: gemma4WeightAny(weights, "model.norm.weight")},
		PreProjection:            gemma4Linear(weights, "pre_projection", text.Quantization),
		PostProjection:           gemma4Linear(weights, "post_projection", text.Quantization),
		Tok:                      tok,
		Cfg:                      text,
		BackboneHiddenSize:       cfg.BackboneHiddenSize,
		NumCentroids:             cfg.NumCentroids,
		CentroidIntermediateTopK: cfg.CentroidIntermediateTopK,
		UseOrderedEmbeddings:     cfg.UseOrderedEmbeddings,
	}
	if cfg.UseOrderedEmbeddings {
		m.MaskedCentroids = gemma4Linear(weights, "masked_embedding.centroids", text.Quantization)
		m.TokenOrdering = gemma4WeightAny(weights, "masked_embedding.token_ordering")
		m.TokenOrdering = normalizeGemma4AssistantTokenOrdering(m.TokenOrdering, cfg.NumCentroids, text.VocabSize)
	}

	for i := int32(0); i < text.NumHiddenLayers; i++ {
		prefix := core.Sprintf("model.layers.%d", i)
		layerType := text.LayerTypes[i]
		isSliding := layerType == "sliding_attention"
		headDim := text.HeadDim
		if !isSliding && text.GlobalHeadDim > 0 {
			headDim = text.GlobalHeadDim
		}
		ropeParams := text.RopeParameters[layerType]
		rotatedDims := gemma4RotatedDims(headDim, ropeParams)
		var ropeFreqs *metal.Array
		if ropeParams.RopeType == "proportional" {
			factor := ropeParams.Factor
			if factor == 0 {
				factor = 1
			}
			ropeFreqs = gemma4ProportionalFreqs(headDim, rotatedDims, float32(ropeParams.RopeTheta), factor)
		}
		layer := &Gemma4AssistantLayer{
			InputNorm:    &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".input_layernorm.weight")},
			PostAttnNorm: &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_attention_layernorm.weight")},
			PreFFNorm:    &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm.weight")},
			Attention: &Gemma4AssistantAttention{
				QProj:          gemma4Linear(weights, prefix+".self_attn.q_proj", text.Quantization),
				OProj:          gemma4Linear(weights, prefix+".self_attn.o_proj", text.Quantization),
				QNorm:          &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".self_attn.q_norm.weight")},
				HeadDim:        headDim,
				NHeads:         text.NumAttentionHeads,
				Scale:          gemma4AttentionScale(headDim),
				RopeBase:       float32(ropeParams.RopeTheta),
				RopeRotatedDim: rotatedDims,
				RopeFreqs:      ropeFreqs,
			},
			MLP: &metal.MLP{
				GateProj: gemma4Linear(weights, prefix+".mlp.gate_proj", text.Quantization),
				UpProj:   gemma4Linear(weights, prefix+".mlp.up_proj", text.Quantization),
				DownProj: gemma4Linear(weights, prefix+".mlp.down_proj", text.Quantization),
			},
			LayerScalar: gemma4WeightAny(weights, prefix+".layer_scalar", prefix+".layer_scalar.weight"),
			LayerType:   layerType,
			IsSliding:   isSliding,
			LayerIdx:    i,
		}
		m.Layers[i] = layer
	}
	return m
}

func normalizeGemma4AssistantTokenOrdering(ordering *metal.Array, numCentroids, vocabSize int32) *metal.Array {
	if ordering == nil || !ordering.Valid() || numCentroids <= 0 || vocabSize <= 0 || vocabSize%numCentroids != 0 {
		return ordering
	}
	var shapeBuf [metal.MaxTensorRank]int32
	shape := ordering.ShapeInto(shapeBuf[:0])
	if len(shape) == 1 && shape[0] == vocabSize {
		return metal.Reshape2(ordering, numCentroids, vocabSize/numCentroids)
	}
	return ordering
}

func validateGemma4AssistantModel(m *Gemma4AssistantModel) error {
	var missing []string
	addMissing := func(name string, arr *metal.Array) {
		if arr == nil || !arr.Valid() {
			missing = append(missing, name)
		}
	}
	addLinearMissing := func(name string, linear *metal.Linear) {
		if linear == nil {
			missing = append(missing, name+".weight")
			return
		}
		addMissing(name+".weight", linear.Weight)
	}
	addNormMissing := func(name string, norm *metal.RMSNormModule) {
		if norm == nil {
			missing = append(missing, name+".weight")
			return
		}
		addMissing(name+".weight", norm.Weight)
	}

	if m == nil || m.Cfg == nil {
		return core.NewError("gemma4.assistant model is nil")
	}
	if m.BackboneHiddenSize <= 0 {
		return core.NewError("gemma4.assistant backbone_hidden_size is invalid")
	}
	addMissing("model.embed_tokens.weight", embeddingWeight(m.EmbedTokens))
	addNormMissing("model.norm", m.Norm)
	addLinearMissing("pre_projection", m.PreProjection)
	addLinearMissing("post_projection", m.PostProjection)
	if m.UseOrderedEmbeddings {
		addLinearMissing("masked_embedding.centroids", m.MaskedCentroids)
		addMissing("masked_embedding.token_ordering", m.TokenOrdering)
	}

	for i, layer := range m.Layers {
		prefix := core.Sprintf("model.layers.%d", i)
		if layer == nil {
			missing = append(missing, prefix)
			continue
		}
		addNormMissing(prefix+".input_layernorm", layer.InputNorm)
		addNormMissing(prefix+".post_attention_layernorm", layer.PostAttnNorm)
		addNormMissing(prefix+".pre_feedforward_layernorm", layer.PreFFNorm)
		addNormMissing(prefix+".post_feedforward_layernorm", layer.PostFFNorm)
		addMissing(prefix+".layer_scalar", layer.LayerScalar)
		if layer.Attention == nil {
			missing = append(missing, prefix+".self_attn")
		} else {
			addLinearMissing(prefix+".self_attn.q_proj", layer.Attention.QProj)
			addLinearMissing(prefix+".self_attn.o_proj", layer.Attention.OProj)
			addNormMissing(prefix+".self_attn.q_norm", layer.Attention.QNorm)
			if layer.Attention.HeadDim <= 0 {
				missing = append(missing, prefix+".self_attn.head_dim")
			}
			if layer.Attention.NHeads <= 0 {
				missing = append(missing, prefix+".self_attn.num_attention_heads")
			}
		}
		if layer.MLP == nil {
			missing = append(missing, prefix+".mlp")
		} else {
			addLinearMissing(prefix+".mlp.gate_proj", layer.MLP.GateProj)
			addLinearMissing(prefix+".mlp.up_proj", layer.MLP.UpProj)
			addLinearMissing(prefix+".mlp.down_proj", layer.MLP.DownProj)
		}
	}
	if len(missing) > 0 {
		return core.NewError("missing required tensors: " + core.Join(", ", missing...))
	}
	if err := validateGemma4AssistantProjectionShapes(m); err != nil {
		return err
	}
	if err := validateGemma4AssistantOrderedEmbeddingShape(m); err != nil {
		return err
	}
	return nil
}

func embeddingWeight(embedding *metal.Embedding) *metal.Array {
	if embedding == nil {
		return nil
	}
	return embedding.Weight
}

func validateGemma4AssistantProjectionShapes(m *Gemma4AssistantModel) error {
	if m == nil || m.Cfg == nil {
		return nil
	}
	if err := validateGemma4AssistantLinearShape("pre_projection", m.PreProjection, m.Cfg.HiddenSize, m.BackboneHiddenSize*2); err != nil {
		return err
	}
	if err := validateGemma4AssistantLinearShape("post_projection", m.PostProjection, m.BackboneHiddenSize, m.Cfg.HiddenSize); err != nil {
		return err
	}
	if m.UseOrderedEmbeddings {
		if err := validateGemma4AssistantLinearShape("masked_embedding.centroids", m.MaskedCentroids, m.NumCentroids, m.Cfg.HiddenSize); err != nil {
			return err
		}
	}
	return nil
}

func validateGemma4AssistantOrderedEmbeddingShape(m *Gemma4AssistantModel) error {
	if m == nil || m.Cfg == nil || !m.UseOrderedEmbeddings || m.TokenOrdering == nil || !m.TokenOrdering.Valid() {
		return nil
	}
	switch m.TokenOrdering.Dtype() {
	case metal.DTypeInt32, metal.DTypeInt64:
	default:
		return core.NewError(core.Sprintf("masked_embedding.token_ordering dtype = %s, want int32 or int64", m.TokenOrdering.Dtype()))
	}
	vocabSize := m.Cfg.VocabSize
	numCentroids := m.NumCentroids
	if vocabSize <= 0 || numCentroids <= 0 || vocabSize%numCentroids != 0 {
		return core.NewError("masked_embedding.token_ordering requires vocab_size divisible by num_centroids")
	}
	var shapeBuf [metal.MaxTensorRank]int32
	shape := m.TokenOrdering.ShapeInto(shapeBuf[:0])
	tokensPerCentroid := vocabSize / numCentroids
	if len(shape) == 1 && shape[0] == vocabSize {
		return nil
	}
	if len(shape) == 2 && shape[0] == numCentroids && shape[1] == tokensPerCentroid {
		return nil
	}
	return core.NewError(core.Sprintf("masked_embedding.token_ordering shape = %v, want [%d] or [%d %d]", shape, vocabSize, numCentroids, tokensPerCentroid))
}

func validateGemma4AssistantLinearShape(name string, linear *metal.Linear, out, in int32) error {
	if linear == nil || linear.Weight == nil || !linear.Weight.Valid() {
		return nil
	}
	shape := linear.Weight.Shape()
	if len(shape) < 2 {
		return core.NewError(name + ".weight has invalid rank")
	}
	gotOut := shape[len(shape)-2]
	gotIn := shape[len(shape)-1]
	if out > 0 && gotOut != out {
		return core.NewError(core.Sprintf("%s.weight output dim = %d, want %d", name, gotOut, out))
	}
	if in > 0 && !gemma4AssistantLinearInputMatches(linear, gotIn, in) {
		return core.NewError(core.Sprintf("%s.weight input dim = %d, want %d", name, gotIn, in))
	}
	return nil
}

// gemma4AssistantLinearInputMatches reports whether a linear's stored weight
// input dim is consistent with the expected unpacked input dim. A quantized
// weight packs its input dim into uint32 words, so the stored dim is smaller
// than the logical one — a 4-bit QAT drafter stores 10752/8 = 1344. Accept the
// unpacked dim (bf16) or either packing scheme go-mlx emits (legacy pack-factor
// or bitstream).
func gemma4AssistantLinearInputMatches(linear *metal.Linear, gotIn, wantIn int32) bool {
	if gotIn == wantIn {
		return true
	}
	if linear.Scales == nil || !linear.Scales.Valid() || linear.Bits <= 0 {
		return false
	}
	packFactor := int32(32 / linear.Bits)
	if packFactor > 0 && wantIn%packFactor == 0 && gotIn == wantIn/packFactor {
		return true // legacy packing: packedIn = inDim / (32/bits)
	}
	return gotIn == (wantIn*int32(linear.Bits)+31)/32 // bitstream packing
}

func gemma4AssistantRetainedWeights(m *Gemma4AssistantModel) map[*metal.Array]struct{} {
	retained := make(map[*metal.Array]struct{})
	if m == nil {
		return retained
	}
	gemma4TrackEmbedding(retained, m.EmbedTokens)
	gemma4TrackLinear(retained, m.PreProjection)
	gemma4TrackLinear(retained, m.PostProjection)
	gemma4TrackLinear(retained, m.MaskedCentroids)
	gemma4TrackArrays(retained, m.TokenOrdering)
	if m.Norm != nil {
		gemma4TrackArrays(retained, m.Norm.Weight)
	}
	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		if layer.InputNorm != nil {
			gemma4TrackArrays(retained, layer.InputNorm.Weight)
		}
		if layer.PostAttnNorm != nil {
			gemma4TrackArrays(retained, layer.PostAttnNorm.Weight)
		}
		if layer.PreFFNorm != nil {
			gemma4TrackArrays(retained, layer.PreFFNorm.Weight)
		}
		if layer.PostFFNorm != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm.Weight)
		}
		gemma4TrackArrays(retained, layer.LayerScalar)
		if layer.Attention != nil {
			gemma4TrackLinear(retained, layer.Attention.QProj)
			gemma4TrackLinear(retained, layer.Attention.OProj)
			if layer.Attention.QNorm != nil {
				gemma4TrackArrays(retained, layer.Attention.QNorm.Weight)
			}
			gemma4TrackArrays(retained, layer.Attention.RopeFreqs)
		}
		if layer.MLP != nil {
			gemma4TrackLinear(retained, layer.MLP.GateProj)
			gemma4TrackLinear(retained, layer.MLP.UpProj)
			gemma4TrackLinear(retained, layer.MLP.DownProj)
		}
	}
	return retained
}

func closeGemma4Assistant(m *Gemma4AssistantModel) {
	if m == nil {
		return
	}
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeLinear(m.PreProjection)
	metal.FreeLinear(m.PostProjection)
	metal.FreeLinear(m.MaskedCentroids)
	metal.Free(m.TokenOrdering)
	metal.FreeRMSNorm(m.Norm)
	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		metal.FreeRMSNorm(layer.InputNorm)
		metal.FreeRMSNorm(layer.PostAttnNorm)
		metal.FreeRMSNorm(layer.PreFFNorm)
		metal.FreeRMSNorm(layer.PostFFNorm)
		metal.Free(layer.LayerScalar)
		if layer.Attention != nil {
			metal.FreeLinear(layer.Attention.QProj)
			metal.FreeLinear(layer.Attention.OProj)
			metal.FreeRMSNorm(layer.Attention.QNorm)
			metal.Free(layer.Attention.RopeFreqs)
		}
		if layer.MLP != nil {
			metal.FreeLinear(layer.MLP.GateProj)
			metal.FreeLinear(layer.MLP.UpProj)
			metal.FreeLinear(layer.MLP.DownProj)
		}
	}
}

func (m *Gemma4AssistantModel) Close() error {
	closeGemma4Assistant(m)
	metal.ClearCache()
	return nil
}

func (m *Gemma4AssistantModel) NumLayers() int {
	if m == nil {
		return 0
	}
	return len(m.Layers)
}

func (m *Gemma4AssistantModel) Tokenizer() *metal.Tokenizer {
	if m == nil {
		return nil
	}
	return m.Tok
}

func (m *Gemma4AssistantModel) ModelType() string {
	return "gemma4_assistant"
}
