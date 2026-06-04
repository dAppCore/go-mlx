// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Gemma4VisionRopeParameters holds the 2-D RoPE settings for the vision tower.
type Gemma4VisionRopeParameters struct {
	RopeType  string  `json:"rope_type"`
	RopeTheta float32 `json:"rope_theta"`
}

// Gemma4VisionConfig holds the Gemma 4 SigLIP-derived vision tower configuration.
type Gemma4VisionConfig struct {
	ModelType             string                     `json:"model_type"`
	ImageSize             int32                      `json:"image_size"`
	PatchSize             int32                      `json:"patch_size"`
	NumChannels           int32                      `json:"num_channels"`
	HiddenSize            int32                      `json:"hidden_size"`
	IntermediateSize      int32                      `json:"intermediate_size"`
	NumHiddenLayers       int32                      `json:"num_hidden_layers"`
	NumAttentionHeads     int32                      `json:"num_attention_heads"`
	NumKeyValueHeads      int32                      `json:"num_key_value_heads"`
	HeadDim               int32                      `json:"head_dim"`
	HiddenActivation      string                     `json:"hidden_activation"`
	LayerNormEps          float32                    `json:"layer_norm_eps"`
	RMSNormEps            float32                    `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32                      `json:"max_position_embeddings"`
	AttentionBias         bool                       `json:"attention_bias"`
	AttentionDropout      float32                    `json:"attention_dropout"`
	RopeParameters        Gemma4VisionRopeParameters `json:"rope_parameters"`
	PoolingKernelSize     int32                      `json:"pooling_kernel_size"`
	PositionEmbeddingSize int32                      `json:"position_embedding_size"`
	UseClippedLinears     bool                       `json:"use_clipped_linears"`
	Standardize           bool                       `json:"standardize"`
	InitializerRange      float32                    `json:"initializer_range"`
}

// Gemma4VisionModel is the Gemma 4 vision encoder.
type Gemma4VisionModel struct {
	PatchEmbedder *Gemma4VisionPatchEmbedder
	Encoder       *Gemma4VisionEncoder
	Pooler        *Gemma4VisionPooler
	PostLayernorm *metal.RMSNormModule

	PatchEmbedding     *metal.Linear
	PositionEmbeddings *metal.Array
	EncoderLayers      []*Gemma4VisionLayer

	StdBias  *metal.Array
	StdScale *metal.Array
	Cfg      *Gemma4VisionConfig
}

// Gemma4VisionPatchEmbedder projects patch pixels and adds learned 2-D positions.
type Gemma4VisionPatchEmbedder struct {
	InputProj              *metal.Linear
	PatchConvWeight        *metal.Array
	PositionEmbeddingTable *metal.Array
	PatchSize              int32
	NumChannels            int32
	PoolingKernelSize      int32
	PositionEmbeddingSize  int32
	HiddenSize             int32
}

// Gemma4VisionEncoder is the stack of bidirectional vision transformer layers.
type Gemma4VisionEncoder struct {
	Layers []*Gemma4VisionEncoderLayer
	Cfg    *Gemma4VisionConfig
}

// Gemma4VisionEncoderLayer is a pre-norm vision transformer block.
type Gemma4VisionEncoderLayer struct {
	InputNorm    *metal.RMSNormModule
	Attention    *Gemma4VisionAttention
	PostAttnNorm *metal.RMSNormModule
	PreFFNorm    *metal.RMSNormModule
	MLP          *Gemma4VisionMLP
	PostFFNorm   *metal.RMSNormModule
}

// Gemma4VisionAttention is bidirectional MHA/GQA with Q/K/V normalization.
type Gemma4VisionAttention struct {
	QProj *metal.Linear
	KProj *metal.Linear
	VProj *metal.Linear
	OProj *metal.Linear
	QNorm *metal.RMSNormModule
	KNorm *metal.RMSNormModule

	HeadDim   int32
	NHeads    int32
	NKVHeads  int32
	RopeBase  float32
	Attention float32
}

// Gemma4VisionMLP is the gated feed-forward block used by Gemma 4 vision layers.
type Gemma4VisionMLP struct {
	GateProj *metal.Linear
	UpProj   *metal.Linear
	DownProj *metal.Linear
}

// Gemma4VisionPooler converts patch encodings into the configured soft-token budget.
type Gemma4VisionPooler struct {
	HiddenSize        int32
	PoolingKernelSize int32
	EmbeddingScale    float32 // Computed: sqrt(HiddenSize); cached to skip per-token math.Sqrt
}

// Gemma4VisionLayer is the public Phase 4 layer name for the vision encoder.
type Gemma4VisionLayer = Gemma4VisionEncoderLayer

// Gemma4MultiModalProjector maps vision soft tokens into the text hidden size.
type Gemma4MultiModalProjector struct {
	Projection *metal.Linear
	Linear1    *metal.Linear
	Linear2    *metal.Linear
	Eps        float32
}

// MultiModalProjector is the RFC name for the Gemma 4 vision-to-text projector.
type MultiModalProjector = Gemma4MultiModalProjector

func defaultGemma4VisionConfig() *Gemma4VisionConfig {
	return &Gemma4VisionConfig{
		ModelType:             "gemma4_vision",
		ImageSize:             896,
		PatchSize:             16,
		NumChannels:           3,
		HiddenSize:            768,
		IntermediateSize:      3072,
		NumHiddenLayers:       16,
		NumAttentionHeads:     12,
		NumKeyValueHeads:      12,
		HeadDim:               64,
		HiddenActivation:      "gelu_pytorch_tanh",
		LayerNormEps:          1e-6,
		RMSNormEps:            1e-6,
		MaxPositionEmbeddings: 131072,
		RopeParameters: Gemma4VisionRopeParameters{
			RopeType:  "default",
			RopeTheta: 100,
		},
		PoolingKernelSize:     3,
		PositionEmbeddingSize: 10 * 1024,
		InitializerRange:      0.02,
	}
}

func normalizeGemma4VisionConfig(cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		return nil
	}
	defaults := defaultGemma4VisionConfig()
	if cfg.ModelType == "" {
		cfg.ModelType = defaults.ModelType
	}
	if cfg.ImageSize == 0 {
		cfg.ImageSize = defaults.ImageSize
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = defaults.PatchSize
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = defaults.NumChannels
	}
	if cfg.HiddenSize == 0 {
		cfg.HiddenSize = defaults.HiddenSize
	}
	if cfg.IntermediateSize == 0 {
		cfg.IntermediateSize = defaults.IntermediateSize
	}
	if cfg.NumHiddenLayers == 0 {
		cfg.NumHiddenLayers = defaults.NumHiddenLayers
	}
	if cfg.NumAttentionHeads == 0 {
		cfg.NumAttentionHeads = defaults.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = defaults.HeadDim
	}
	if cfg.HiddenActivation == "" {
		cfg.HiddenActivation = defaults.HiddenActivation
	}
	if cfg.LayerNormEps == 0 && cfg.RMSNormEps != 0 {
		cfg.LayerNormEps = cfg.RMSNormEps
	}
	if cfg.RMSNormEps == 0 && cfg.LayerNormEps != 0 {
		cfg.RMSNormEps = cfg.LayerNormEps
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = defaults.LayerNormEps
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = defaults.RMSNormEps
	}
	if cfg.MaxPositionEmbeddings == 0 {
		cfg.MaxPositionEmbeddings = defaults.MaxPositionEmbeddings
	}
	if cfg.RopeParameters.RopeType == "" {
		cfg.RopeParameters.RopeType = defaults.RopeParameters.RopeType
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = defaults.RopeParameters.RopeTheta
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = defaults.PoolingKernelSize
	}
	if cfg.PositionEmbeddingSize == 0 {
		cfg.PositionEmbeddingSize = defaults.PositionEmbeddingSize
	}
	if cfg.InitializerRange == 0 {
		cfg.InitializerRange = defaults.InitializerRange
	}
	return cfg
}

func sanitizeGemma4VisionWeights(raw map[string]*metal.Array) map[string]*metal.Array {
	vision := make(map[string]*metal.Array)
	for name, arr := range raw {
		canonical, ok := canonicalGemma4VisionWeightName(name)
		if !ok {
			continue
		}
		if prev, exists := vision[canonical]; exists && prev != arr {
			metal.Free(prev)
		}
		vision[canonical] = arr
		delete(raw, name)
	}
	return vision
}

func canonicalGemma4VisionWeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := trimGemma4WrapperPrefix(trimmed)
		if !changed {
			break
		}
		trimmed = next
	}

	for _, prefix := range []string{
		"vision_tower.",
		"vision_model.",
	} {
		if core.HasPrefix(trimmed, prefix) {
			return core.TrimPrefix(trimmed, prefix), true
		}
	}
	for _, prefix := range []string{
		"multi_modal_projector.",
		"embed_vision.",
	} {
		if core.HasPrefix(trimmed, prefix) {
			return trimmed, true
		}
	}
	return "", false
}

func hasGemma4VisionTowerWeights(weights map[string]*metal.Array) bool {
	return gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	) != nil
}

func buildGemma4VisionComponents(cfg *Gemma4TextConfig, weights map[string]*metal.Array) (*Gemma4VisionModel, *Gemma4MultiModalProjector, error) {
	if !hasGemma4VisionTowerWeights(weights) {
		gemma4FreeUnusedWeights(weights, map[*metal.Array]struct{}{})
		return nil, nil, nil
	}

	visionCfg := cfg.VisionConfig
	if visionCfg == nil {
		visionCfg = defaultGemma4VisionConfig()
	}
	visionCfg = inferGemma4VisionConfig(weights, normalizeGemma4VisionConfig(visionCfg))

	vision, err := buildGemma4VisionModel(visionCfg, weights)
	if err != nil {
		gemma4FreeUnusedWeights(weights, map[*metal.Array]struct{}{})
		return nil, nil, err
	}
	projector := buildGemma4MultiModalProjector(cfg, visionCfg, weights)

	retained := gemma4VisionRetainedWeights(vision, projector)
	gemma4FreeUnusedWeights(weights, retained)
	gemma4MaterializeRetainedWeights(retained, nil)
	return vision, projector, nil
}

func inferGemma4VisionConfig(weights map[string]*metal.Array, cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		cfg = defaultGemma4VisionConfig()
	}
	if w := gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	); w != nil {
		shape := w.Shape()
		if len(shape) > 0 && shape[0] > 0 {
			cfg.HiddenSize = shape[0]
		}
		patchDim := int32(0)
		switch len(shape) {
		case 2:
			patchDim = shape[1]
		case 4:
			patchDim = shape[1] * shape[2] * shape[3]
		}
		channels := cfg.NumChannels
		if channels <= 0 {
			channels = 3
		}
		if patchDim > 0 && patchDim%channels == 0 {
			patch := int32(math.Round(math.Sqrt(float64(patchDim / channels))))
			if patch > 0 && channels*patch*patch == patchDim {
				cfg.PatchSize = patch
			}
		}
	}
	if cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	for i := int32(0); ; i++ {
		prefix := core.Sprintf("encoder.layers.%d", i)
		if gemma4VisionWeightAny(weights,
			prefix+".self_attn.q_proj.weight",
			prefix+".self_attn.q_proj.linear.weight",
			prefix+".attention.q_proj.weight",
			prefix+".attention.q_proj.linear.weight",
		) == nil {
			if i > 0 {
				cfg.NumHiddenLayers = i
			}
			break
		}
	}
	return normalizeGemma4VisionConfig(cfg)
}

func gemma4VisionWeightAny(weights map[string]*metal.Array, names ...string) *metal.Array {
	for _, name := range names {
		if arr := weights[name]; arr != nil {
			return arr
		}
	}
	return nil
}

func gemma4VisionLinear(weights map[string]*metal.Array, prefixes ...string) *metal.Linear {
	for _, prefix := range prefixes {
		weight := gemma4VisionWeightAny(weights, prefix+".weight", prefix+".linear.weight")
		if weight == nil {
			continue
		}
		bias := gemma4VisionWeightAny(weights, prefix+".bias", prefix+".linear.bias")
		return metal.NewLinear(weight, bias)
	}
	return nil
}

func gemma4VisionNorm(weights map[string]*metal.Array, hiddenSize int32, names ...string) *metal.RMSNormModule {
	if weight := gemma4VisionWeightAny(weights, names...); weight != nil {
		return &metal.RMSNormModule{Weight: weight}
	}
	return &metal.RMSNormModule{Weight: gemma4Ones([]int32{hiddenSize})}
}

func normalizeGemma4PatchProjection(weight *metal.Array, cfg *Gemma4VisionConfig) (*metal.Array, *metal.Array, bool) {
	if weight == nil {
		return nil, nil, false
	}
	channels := cfg.NumChannels
	if channels <= 0 {
		channels = 3
	}
	shape := weight.Shape()
	if len(shape) == 2 {
		conv := metal.Reshape(weight, shape[0], cfg.PatchSize, cfg.PatchSize, channels)
		return weight, conv, true
	}
	if len(shape) != 4 {
		return weight, nil, true
	}
	var conv *metal.Array
	if shape[3] == channels {
		conv = weight
	} else if shape[1] == channels {
		conv = metal.Transpose(weight, 0, 2, 3, 1)
	} else {
		conv = weight
	}
	linear := metal.Reshape(conv, shape[0], shape[1]*shape[2]*shape[3])
	return linear, conv, true
}

func buildGemma4VisionModel(cfg *Gemma4VisionConfig, weights map[string]*metal.Array) (*Gemma4VisionModel, error) {
	patchWeight := gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	)
	inputWeight, convWeight, ok := normalizeGemma4PatchProjection(patchWeight, cfg)
	if !ok || inputWeight == nil {
		return nil, core.E("gemma4.vision", "missing patch embedding weight", nil)
	}

	var postLayernorm *metal.RMSNormModule
	if weight := gemma4VisionWeightAny(weights,
		"post_layernorm.weight",
		"post_layer_norm.weight",
		"encoder.post_layernorm.weight",
		"vision_model.post_layernorm.weight",
	); weight != nil {
		postLayernorm = &metal.RMSNormModule{Weight: weight}
	}

	vision := &Gemma4VisionModel{
		PatchEmbedder: &Gemma4VisionPatchEmbedder{
			InputProj:              metal.NewLinear(inputWeight, nil),
			PatchConvWeight:        convWeight,
			PositionEmbeddingTable: gemma4VisionWeightAny(weights, "patch_embedder.position_embedding_table", "embeddings.position_embedding.weight"),
			PatchSize:              cfg.PatchSize,
			NumChannels:            cfg.NumChannels,
			PoolingKernelSize:      cfg.PoolingKernelSize,
			PositionEmbeddingSize:  cfg.PositionEmbeddingSize,
			HiddenSize:             cfg.HiddenSize,
		},
		Encoder: &Gemma4VisionEncoder{
			Layers: make([]*Gemma4VisionEncoderLayer, cfg.NumHiddenLayers),
			Cfg:    cfg,
		},
		Pooler: &Gemma4VisionPooler{
			HiddenSize:        cfg.HiddenSize,
			PoolingKernelSize: cfg.PoolingKernelSize,
			EmbeddingScale:    float32(math.Sqrt(float64(cfg.HiddenSize))),
		},
		PostLayernorm: postLayernorm,
		StdBias:       gemma4VisionWeightAny(weights, "std_bias"),
		StdScale:      gemma4VisionWeightAny(weights, "std_scale"),
		Cfg:           cfg,
	}
	vision.PatchEmbedding = vision.PatchEmbedder.InputProj
	vision.PositionEmbeddings = vision.PatchEmbedder.PositionEmbeddingTable
	vision.EncoderLayers = vision.Encoder.Layers

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("encoder.layers.%d", i)
		layer := &Gemma4VisionEncoderLayer{
			InputNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".input_layernorm.weight",
				prefix+".layer_norm1.weight",
			),
			PostAttnNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".post_attention_layernorm.weight",
				prefix+".post_attention_layernorm.linear.weight",
			),
			PreFFNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".pre_feedforward_layernorm.weight",
				prefix+".layer_norm2.weight",
			),
			PostFFNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".post_feedforward_layernorm.weight",
				prefix+".post_feedforward_layernorm.linear.weight",
			),
			Attention: &Gemma4VisionAttention{
				QProj: gemma4VisionLinear(weights,
					prefix+".self_attn.q_proj",
					prefix+".attention.q_proj",
				),
				KProj: gemma4VisionLinear(weights,
					prefix+".self_attn.k_proj",
					prefix+".attention.k_proj",
				),
				VProj: gemma4VisionLinear(weights,
					prefix+".self_attn.v_proj",
					prefix+".attention.v_proj",
				),
				OProj: gemma4VisionLinear(weights,
					prefix+".self_attn.o_proj",
					prefix+".attention.out_proj",
					prefix+".attention.o_proj",
				),
				QNorm: gemma4VisionNorm(weights, cfg.HeadDim, prefix+".self_attn.q_norm.weight"),
				KNorm: gemma4VisionNorm(weights, cfg.HeadDim, prefix+".self_attn.k_norm.weight"),

				HeadDim:   cfg.HeadDim,
				NHeads:    cfg.NumAttentionHeads,
				NKVHeads:  cfg.NumKeyValueHeads,
				RopeBase:  cfg.RopeParameters.RopeTheta,
				Attention: 1.0,
			},
			MLP: &Gemma4VisionMLP{
				GateProj: gemma4VisionLinear(weights, prefix+".mlp.gate_proj", prefix+".mlp.fc1"),
				UpProj:   gemma4VisionLinear(weights, prefix+".mlp.up_proj"),
				DownProj: gemma4VisionLinear(weights, prefix+".mlp.down_proj", prefix+".mlp.fc2"),
			},
		}
		if err := validateGemma4VisionEncoderLayer(layer, i); err != nil {
			return nil, err
		}
		vision.Encoder.Layers[i] = layer
	}

	return vision, nil
}

func validateGemma4VisionLinear(linear *metal.Linear, name string) error {
	if linear == nil || linear.Weight == nil {
		return core.E("gemma4.vision", "missing "+name, nil)
	}
	return nil
}

func validateGemma4VisionNorm(norm *metal.RMSNormModule, name string) error {
	if norm == nil || norm.Weight == nil {
		return core.E("gemma4.vision", "missing "+name, nil)
	}
	return nil
}

func validateGemma4VisionEncoderLayer(layer *Gemma4VisionEncoderLayer, idx int32) error {
	prefix := core.Sprintf("encoder layer %d ", idx)
	if err := validateGemma4VisionNorm(layer.InputNorm, prefix+"input norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PostAttnNorm, prefix+"post-attention norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PreFFNorm, prefix+"pre-feedforward norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PostFFNorm, prefix+"post-feedforward norm"); err != nil {
		return err
	}
	if layer.Attention == nil {
		return core.E("gemma4.vision", "missing "+prefix+"attention", nil)
	}
	if err := validateGemma4VisionLinear(layer.Attention.QProj, prefix+"q projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.KProj, prefix+"k projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.VProj, prefix+"v projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.OProj, prefix+"output projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.Attention.QNorm, prefix+"q norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.Attention.KNorm, prefix+"k norm"); err != nil {
		return err
	}
	if layer.MLP == nil {
		return core.E("gemma4.vision", "missing "+prefix+"mlp", nil)
	}
	if err := validateGemma4VisionLinear(layer.MLP.GateProj, prefix+"gate projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.MLP.UpProj, prefix+"up projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.MLP.DownProj, prefix+"down projection"); err != nil {
		return err
	}
	return nil
}

func buildGemma4MultiModalProjector(textCfg *Gemma4TextConfig, visionCfg *Gemma4VisionConfig, weights map[string]*metal.Array) *Gemma4MultiModalProjector {
	projector := &Gemma4MultiModalProjector{
		Projection: gemma4VisionLinear(weights,
			"embed_vision.embedding_projection",
			"multi_modal_projector.embedding_projection",
			"multi_modal_projector.proj",
			"multi_modal_projector",
		),
		Linear1: gemma4VisionLinear(weights,
			"multi_modal_projector.linear_1",
			"multi_modal_projector.fc1",
		),
		Linear2: gemma4VisionLinear(weights,
			"multi_modal_projector.linear_2",
			"multi_modal_projector.fc2",
		),
		Eps: visionCfg.RMSNormEps,
	}
	ready := projector.Projection != nil || (projector.Linear1 != nil && projector.Linear2 != nil)
	if visionCfg.HiddenSize != textCfg.HiddenSize && !ready {
		return nil
	}
	return projector
}

func (m *Gemma4Model) ForwardMultiModal(tokens *metal.Array, imagePixels []*metal.Array, caches []metal.Cache) *metal.Array {
	if len(imagePixels) == 0 || m.VisionTower == nil {
		return m.Forward(tokens, caches)
	}

	// Stack-allocated shape scratch — multimodal forward-pass entrypoint.
	// Reused as the tokenShape argument to injectGemma4ImageFeatures.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	if len(shape) != 2 {
		return m.Forward(tokens, caches)
	}

	tokenIDs := tokens.DataInt32()
	imageTokenCount := 0
	for _, id := range tokenIDs {
		if id == m.Cfg.ImageTokenID {
			imageTokenCount++
		}
	}
	if imageTokenCount == 0 {
		return m.Forward(tokens, caches)
	}

	h := m.EmbedTokens.Forward(tokens)
	scaledH := metal.MulScalar(h, m.Cfg.EmbeddingScale)
	metal.Free(h)
	h = scaledH

	imageFeatures := m.encodeGemma4Images(imagePixels)
	if imageFeatures == nil || !imageFeatures.Valid() {
		metal.Free(h)
		return m.Forward(tokens, caches)
	}
	defer metal.Free(imageFeatures)

	h = m.injectGemma4ImageFeatures(h, tokenIDs, shape, imageFeatures)
	return m.forwardGemma4EmbeddingsMasked(tokens, h, nil, caches)
}

func (m *Gemma4Model) encodeGemma4Images(imagePixels []*metal.Array) *metal.Array {
	features := make([]*metal.Array, 0, len(imagePixels))
	for _, image := range imagePixels {
		if image == nil || !image.Valid() {
			continue
		}
		encoded := m.VisionTower.Forward(image)
		if encoded == nil || !encoded.Valid() {
			continue
		}
		projected := encoded
		if m.MultiModalProjector != nil {
			projected = m.MultiModalProjector.Forward(encoded)
			metal.Free(encoded)
		}
		features = append(features, projected)
	}
	if len(features) == 0 {
		return nil
	}
	if len(features) == 1 {
		return features[0]
	}
	combined := metal.Concatenate(features, 0)
	metal.Free(features...)
	return combined
}

func (m *Gemma4Model) injectGemma4ImageFeatures(h *metal.Array, tokenIDs []int32, tokenShape []int32, features *metal.Array) *metal.Array {
	featureRows := features
	if features.NumDims() == 3 {
		// Stack-allocated shape scratch — image-feature reshape called per
		// multimodal forward pass.
		var shapeBuf [metal.MaxTensorRank]int32
		shape := features.ShapeInto(shapeBuf[:0])
		featureRows = metal.Reshape(features, shape[0]*shape[1], shape[2])
		defer metal.Free(featureRows)
	}
	if featureRows.NumDims() != 2 {
		return h
	}

	// h.Shape()[2] previously allocated; use Dim(2) instead.
	B, L, H := tokenShape[0], tokenShape[1], int32(h.Dim(2))
	if int32(featureRows.Dim(1)) != H {
		core.Error("gemma4: image features hidden size mismatch", "features", featureRows.Dim(1), "hidden", H)
		return h
	}
	nFeatures := int32(featureRows.Dim(0))
	imageSlots := int32(0)
	for _, id := range tokenIDs {
		if id == m.Cfg.ImageTokenID {
			imageSlots++
		}
	}
	if nFeatures != imageSlots {
		core.Error("gemma4: image feature count mismatch", "features", nFeatures, "tokens", imageSlots)
	}
	featureIdx := int32(0)
	for flatIdx, id := range tokenIDs {
		if id != m.Cfg.ImageTokenID {
			continue
		}
		if featureIdx >= nFeatures {
			break
		}
		b := int32(flatIdx) / L
		pos := int32(flatIdx) % L
		if b >= B {
			break
		}

		row := metal.SliceAxis(featureRows, 0, featureIdx, featureIdx+1)
		update := metal.Reshape(row, 1, 1, H)
		next := metal.SliceUpdateInplace(h, update, []int32{b, pos, 0}, []int32{b + 1, pos + 1, H})
		metal.Free(h, row, update)
		h = next
		featureIdx++
	}
	return h
}

func (m *Gemma4Model) forwardGemma4EmbeddingsMasked(tokens *metal.Array, h *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	m.ensureCacheLayout()

	// Stack-allocated shape scratch — per-forward-pass hot path.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	perLayerInputs := m.computePerLayerInputs(tokens, h)
	defer metal.Free(perLayerInputs...)

	var ownedMasks []*metal.Array
	fullMask := mask
	slidingMask := mask
	if mask == nil {
		if L > 1 && m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
			slidingMask = buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
			ownedMasks = append(ownedMasks, slidingMask)
		}
	} else if m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
		windowMask := buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
		combined := gemma4CombineMasks(mask, windowMask)
		metal.Free(windowMask)
		slidingMask = combined
		ownedMasks = append(ownedMasks, combined)
	}
	defer metal.Free(ownedMasks...)

	intermediates := make([]sharedKV, len(m.Layers))
	sharedSources := make([]bool, len(m.Layers))
	for i, prevIdx := range m.PreviousKVs {
		if i >= len(sharedSources) {
			break
		}
		if prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(sharedSources)) {
			sharedSources[prevIdx] = true
		}
	}
	for i, layer := range m.Layers {
		var prev sharedKV
		if prevIdx := m.PreviousKVs[i]; prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(intermediates)) {
			prev = intermediates[prevIdx]
		}

		var cache metal.Cache
		if m.PreviousKVs[i] == int32(i) && i < len(m.CacheIndexByLayer) {
			if cacheIdx := m.CacheIndexByLayer[i]; cacheIdx >= 0 && int(cacheIdx) < len(caches) {
				cache = caches[cacheIdx]
			}
		}

		layerMask := fullMask
		if layer.IsSliding {
			layerMask = slidingMask
		}

		var pli *metal.Array
		if len(perLayerInputs) > i {
			pli = perLayerInputs[i]
		}

		materializePagedKVForReuse := m.PreviousKVs[i] == int32(i) && sharedSources[i]
		nextH, kv := layer.forward(h, cache, B, L, layerMask, pli, prev, m.Cfg, nil, nil, materializePagedKVForReuse)
		metal.Free(h)
		h = nextH
		intermediates[i] = kv
	}
	defer func() {
		for i, kv := range intermediates {
			if m.PreviousKVs[i] != int32(i) {
				continue
			}
			metal.Free(kv.Keys, kv.Values)
		}
	}()

	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out
}

func (v *Gemma4VisionModel) Forward(pixelValues *metal.Array) *metal.Array {
	if v == nil || v.PatchEmbedder == nil {
		return nil
	}
	h, gridH, gridW := v.PatchEmbedder.Forward(pixelValues)
	if h == nil || !h.Valid() {
		return nil
	}

	encoded := v.Encoder.Forward(h, gridH, gridW)
	metal.Free(h)
	if v.PostLayernorm != nil && v.PostLayernorm.Weight != nil && v.PostLayernorm.Weight.Valid() {
		normed := metal.RMSNorm(encoded, v.PostLayernorm.Weight, v.Cfg.RMSNormEps)
		metal.Free(encoded)
		encoded = normed
	}
	pooled := v.Pooler.Forward(encoded, gridH, gridW)
	metal.Free(encoded)

	if v.Cfg.Standardize && v.StdBias != nil && v.StdScale != nil {
		centered := metal.Subtract(pooled, v.StdBias)
		metal.Free(pooled)
		pooled = metal.Mul(centered, v.StdScale)
		metal.Free(centered)
	}
	return pooled
}

func (p *Gemma4VisionPatchEmbedder) Forward(pixelValues *metal.Array) (*metal.Array, int32, int32) {
	patches, projected, gridH, gridW := p.prepare(pixelValues)
	if patches == nil || !patches.Valid() {
		return nil, 0, 0
	}

	hidden := patches
	if !projected {
		shifted := metal.AddScalar(patches, -0.5)
		scaled := metal.MulScalar(shifted, 2.0)
		metal.Free(shifted)
		if scaled != patches {
			metal.Free(patches)
		}
		hidden = p.InputProj.Forward(scaled)
		metal.Free(scaled)
	}

	if p.PositionEmbeddingTable != nil && p.PositionEmbeddingTable.Valid() {
		// hidden.Shape()[0] previously allocated; Dim(0) is one C call zero allocs.
		pos := p.positionEmbeddings(int32(hidden.Dim(0)), gridH, gridW)
		if pos != nil && pos.Valid() {
			next := metal.Add(hidden, pos)
			metal.Free(hidden, pos)
			hidden = next
		}
	}
	return hidden, gridH, gridW
}

func (p *Gemma4VisionPatchEmbedder) prepare(pixelValues *metal.Array) (*metal.Array, bool, int32, int32) {
	// Stack-allocated shape scratch — vision patch embed prepare; per-image
	// hot path. The Transpose(0,2,3,1) on the rank-4 branches is rank-4 by
	// case-construction, so Transpose4 applies.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := pixelValues.ShapeInto(shapeBuf[:0])
	channels := p.NumChannels
	if channels <= 0 {
		channels = 3
	}
	patchDim := channels * p.PatchSize * p.PatchSize
	switch len(shape) {
	case 2:
		gridH, gridW := gemma4VisionGridForPatchCount(shape[0], p.poolKernel())
		return metal.Reshape(pixelValues, 1, shape[0], shape[1]), false, gridH, gridW
	case 3:
		if shape[2] == patchDim {
			gridH, gridW := gemma4VisionGridForPatchCount(shape[1], p.poolKernel())
			return pixelValues.Clone(), false, gridH, gridW
		}
		if shape[2] == channels {
			expanded := metal.ExpandDims(pixelValues, 0)
			return p.prepareRawNHWC(expanded, true)
		}
		if shape[0] == channels {
			expanded := metal.ExpandDims(pixelValues, 0)
			transposed := metal.Transpose4(expanded, 0, 2, 3, 1)
			metal.Free(expanded)
			return p.prepareRawNHWC(transposed, true)
		}
	case 4:
		if shape[3] == channels {
			return p.prepareRawNHWC(pixelValues.Clone(), true)
		}
		if shape[1] == channels {
			transposed := metal.Transpose4(pixelValues, 0, 2, 3, 1)
			return p.prepareRawNHWC(transposed, true)
		}
	}
	return nil, false, 0, 0
}

func (p *Gemma4VisionPatchEmbedder) prepareRawNHWC(nhwc *metal.Array, owned bool) (*metal.Array, bool, int32, int32) {
	// Stack-allocated shape scratch — per-image patch-embed convolution
	// path. Both nhwc and conv are rank-4 NHWC tensors.
	var shapeBuf, convShapeBuf [metal.MaxTensorRank]int32
	shape := nhwc.ShapeInto(shapeBuf[:0])
	if len(shape) != 4 || p.PatchConvWeight == nil || !p.PatchConvWeight.Valid() {
		if owned {
			metal.Free(nhwc)
		}
		return nil, false, 0, 0
	}
	gridH := shape[1] / p.PatchSize
	gridW := shape[2] / p.PatchSize

	shifted := metal.AddScalar(nhwc, -0.5)
	scaled := metal.MulScalar(shifted, 2.0)
	metal.Free(shifted)
	if owned {
		metal.Free(nhwc)
	}

	conv := metal.Conv2d(scaled, p.PatchConvWeight, int(p.PatchSize), int(p.PatchSize), 0, 0, 1, 1, 1)
	metal.Free(scaled)
	convShape := conv.ShapeInto(convShapeBuf[:0])
	patches := metal.Reshape(conv, convShape[0], convShape[1]*convShape[2], convShape[3])
	metal.Free(conv)
	return patches, true, gridH, gridW
}

func (p *Gemma4VisionPatchEmbedder) poolKernel() int32 {
	if p == nil {
		return 1
	}
	if p.PoolingKernelSize <= 0 {
		return 1
	}
	return p.PoolingKernelSize
}

func (p *Gemma4VisionPatchEmbedder) positionEmbeddings(batch, gridH, gridW int32) *metal.Array {
	table := p.PositionEmbeddingTable
	// Stack-allocated shape scratch — per-vision-pass position embedding.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := table.ShapeInto(shapeBuf[:0])
	if len(shape) < 2 {
		return nil
	}

	count := int(batch * gridH * gridW)
	xIDs := make([]int32, count)
	yIDs := make([]int32, count)
	for b := range batch {
		base := int(b * gridH * gridW)
		for y := range gridH {
			for x := range gridW {
				idx := base + int(y*gridW+x)
				xIDs[idx] = x
				yIDs[idx] = y
			}
		}
	}
	xIdx := metal.FromValues(xIDs, int(batch), int(gridH*gridW))
	yIdx := metal.FromValues(yIDs, int(batch), int(gridH*gridW))
	defer metal.Free(xIdx, yIdx)

	if len(shape) == 3 && shape[0] >= 2 {
		xTableSlice := metal.SliceAxis(table, 0, 0, 1)
		xTable := metal.Squeeze(xTableSlice, 0)
		yTableSlice := metal.SliceAxis(table, 0, 1, 2)
		yTable := metal.Squeeze(yTableSlice, 0)
		xEmb := metal.Take(xTable, xIdx, 0)
		yEmb := metal.Take(yTable, yIdx, 0)
		pos := metal.Add(xEmb, yEmb)
		metal.Free(xTableSlice, xTable, yTableSlice, yTable, xEmb, yEmb)
		return pos
	}

	flatIDs := make([]int32, count)
	for i := range flatIDs {
		flatIDs[i] = int32(i) % (gridH * gridW)
	}
	flatIdx := metal.FromValues(flatIDs, int(batch), int(gridH*gridW))
	pos := metal.Take(table, flatIdx, 0)
	metal.Free(flatIdx)
	return pos
}

func (e *Gemma4VisionEncoder) Forward(x *metal.Array, grid ...int32) *metal.Array {
	gridH, gridW := int32(0), int32(0)
	if len(grid) >= 2 {
		gridH, gridW = grid[0], grid[1]
	}
	if (gridH <= 0 || gridW <= 0) && x != nil && x.NumDims() >= 2 {
		gridH, gridW = gemma4VisionGridForPatchCount(int32(x.Dim(1)), 1)
	}
	h := x
	cfg := e.Cfg
	if cfg == nil {
		cfg = normalizeGemma4VisionConfig(defaultGemma4VisionConfig())
	}
	for _, layer := range e.Layers {
		next := layer.Forward(h, gridH, gridW, cfg)
		if h != x {
			metal.Free(h)
		}
		h = next
	}
	return h
}

func (l *Gemma4VisionEncoderLayer) Forward(x *metal.Array, gridH, gridW int32, cfg *Gemma4VisionConfig) *metal.Array {
	residual := x
	normed := metal.RMSNorm(x, l.InputNorm.Weight, cfg.RMSNormEps)
	attnOut := l.Attention.Forward(normed, gridH, gridW, cfg)
	metal.Free(normed)
	attnNormed := metal.RMSNorm(attnOut, l.PostAttnNorm.Weight, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(residual, attnNormed)
	metal.Free(attnNormed)

	residual = h
	ffIn := metal.RMSNorm(h, l.PreFFNorm.Weight, cfg.RMSNormEps)
	ff := l.MLP.Forward(ffIn)
	metal.Free(ffIn)
	ffNormed := metal.RMSNorm(ff, l.PostFFNorm.Weight, cfg.RMSNormEps)
	metal.Free(ff)
	out := metal.Add(residual, ffNormed)
	metal.Free(h, ffNormed)
	return out
}

func (a *Gemma4VisionAttention) Forward(x *metal.Array, gridH, gridW int32, cfg *Gemma4VisionConfig) *metal.Array {
	// Stack-allocated shape scratch — per-vision-attention-layer hot path.
	// All rank-4 Transposes on the V and out paths use the scalar-pass
	// Transpose4 (axes 0,2,1,3 — rank-4 by construction).
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	qProj := a.QProj.Forward(x)
	q := metal.Reshape(qProj, B, L, a.NHeads, a.HeadDim)
	metal.Free(qProj)
	qNorm := metal.RMSNorm(q, a.QNorm.Weight, cfg.RMSNormEps)
	metal.Free(q)
	q = gemma4VisionRoPEAndTranspose(qNorm, gridH, gridW, a.RopeBase, a.HeadDim)
	metal.Free(qNorm)

	kProj := a.KProj.Forward(x)
	k := metal.Reshape(kProj, B, L, a.NKVHeads, a.HeadDim)
	metal.Free(kProj)
	kNorm := metal.RMSNorm(k, a.KNorm.Weight, cfg.RMSNormEps)
	metal.Free(k)
	k = gemma4VisionRoPEAndTranspose(kNorm, gridH, gridW, a.RopeBase, a.HeadDim)
	metal.Free(kNorm)

	vProj := a.VProj.Forward(x)
	v := metal.Reshape(vProj, B, L, a.NKVHeads, a.HeadDim)
	metal.Free(vProj)
	vNorm := metal.RMSNormNoScale(v, cfg.RMSNormEps)
	metal.Free(v)
	v = metal.Transpose4(vNorm, 0, 2, 1, 3)
	metal.Free(vNorm)

	repeatFactor := a.NHeads / a.NKVHeads
	kAttn, vAttn := k, v
	repeated := false
	if repeatFactor > 1 {
		kAttn = metal.RepeatKV(k, repeatFactor)
		vAttn = metal.RepeatKV(v, repeatFactor)
		repeated = true
	}

	out := metal.ScaledDotProductAttention(q, kAttn, vAttn, a.Attention, false)
	metal.Free(q, k, v)
	if repeated {
		metal.Free(kAttn, vAttn)
	}

	transposed := metal.Transpose4(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, a.NHeads*a.HeadDim)
	metal.Free(transposed)
	result := a.OProj.Forward(reshaped)
	metal.Free(reshaped)
	return result
}

func gemma4VisionRoPEAndTranspose(x *metal.Array, gridH, gridW int32, base float32, headDim int32) *metal.Array {
	// Rank-4 transposes (axes 0,2,1,3) — substrate Transpose4 form.
	if rotated := gemma4VisionApply2DRoPE(x, gridH, gridW, base); rotated != nil {
		transposed := metal.Transpose4(rotated, 0, 2, 1, 3)
		metal.Free(rotated)
		return transposed
	}
	transposed := metal.Transpose4(x, 0, 2, 1, 3)
	out := metal.RoPE(transposed, int(headDim), false, base, 1.0, 0)
	metal.Free(transposed)
	return out
}

func gemma4VisionApply2DRoPE(x *metal.Array, gridH, gridW int32, base float32) *metal.Array {
	// Stack-allocated shape scratch — per-vision-layer 2D RoPE; rank-4 by
	// guard. The three rank-4 Slice calls use the scalar-pass Slice4 form.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	if len(shape) != 4 || base == 0 {
		return nil
	}
	B, L, N, D := shape[0], shape[1], shape[2], shape[3]
	if D < 4 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || gridH*gridW != L {
		gridH, gridW = gemma4VisionGridForPatchCount(L, 1)
	}
	if gridH <= 0 || gridW <= 0 || gridH*gridW != L {
		return nil
	}

	rotatedPerDim := 2 * (D / 4)
	if rotatedPerDim <= 0 || rotatedPerDim%2 != 0 {
		return nil
	}
	rotatedTotal := rotatedPerDim * 2

	cosX, sinX, cosY, sinY := gemma4Vision2DRoPETables(B, L, gridH, gridW, rotatedPerDim, base)
	defer metal.Free(cosX, sinX, cosY, sinY)

	xPart := metal.Slice4(x, 0, 0, 0, 0, B, L, N, rotatedPerDim)
	yPart := metal.Slice4(x, 0, 0, 0, rotatedPerDim, B, L, N, rotatedTotal)
	xRot := gemma4VisionRotatePart(xPart, cosX, sinX)
	yRot := gemma4VisionRotatePart(yPart, cosY, sinY)
	metal.Free(xPart, yPart)

	parts := []*metal.Array{xRot, yRot}
	if rotatedTotal < D {
		rest := metal.Slice4(x, 0, 0, 0, rotatedTotal, B, L, N, D)
		parts = append(parts, rest)
	}
	out := metal.Concatenate(parts, 3)
	metal.Free(parts...)
	return out
}

func gemma4Vision2DRoPETables(batch, seqLen, gridH, gridW, dim int32, base float32) (*metal.Array, *metal.Array, *metal.Array, *metal.Array) {
	freqCount := dim / 2
	invFreq := make([]float64, int(freqCount))
	for i := range freqCount {
		invFreq[int(i)] = 1.0 / math.Pow(float64(base), float64(2*i)/float64(dim))
	}

	size := int(batch * seqLen * dim)
	cosX := make([]float32, size)
	sinX := make([]float32, size)
	cosY := make([]float32, size)
	sinY := make([]float32, size)
	for b := range batch {
		for pos := range seqLen {
			x := float64(pos % gridW)
			y := float64(pos / gridW)
			baseIdx := int((b*seqLen + pos) * dim)
			for d := range dim {
				freq := invFreq[int(d%freqCount)]
				cx := x * freq
				cy := y * freq
				idx := baseIdx + int(d)
				cosX[idx] = float32(math.Cos(cx))
				sinX[idx] = float32(math.Sin(cx))
				cosY[idx] = float32(math.Cos(cy))
				sinY[idx] = float32(math.Sin(cy))
			}
		}
	}

	shape := []int{int(batch), int(seqLen), 1, int(dim)}
	return metal.FromValues(cosX, shape...), metal.FromValues(sinX, shape...), metal.FromValues(cosY, shape...), metal.FromValues(sinY, shape...)
}

func gemma4VisionRotatePart(x, cos, sin *metal.Array) *metal.Array {
	// Stack-allocated shape scratch — per-vision-layer rotate half;
	// x is always rank-4 [B,L,N,D] by caller (gemma4VisionApply2DRoPE).
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	D := shape[3]
	half := D / 2
	first := metal.Slice4(x, 0, 0, 0, 0, shape[0], shape[1], shape[2], half)
	second := metal.Slice4(x, 0, 0, 0, half, shape[0], shape[1], shape[2], D)
	negativeSecond := metal.Negative(second)
	rotated := metal.Concatenate2(negativeSecond, first, 3)
	scaled := metal.Mul(x, cos)
	rotatedScaled := metal.Mul(rotated, sin)
	out := metal.Add(scaled, rotatedScaled)
	metal.Free(first, second, negativeSecond, rotated, scaled, rotatedScaled)
	return out
}

func (m *Gemma4VisionMLP) Forward(x *metal.Array) *metal.Array {
	gate := m.GateProj.Forward(x)
	activated := metal.GeluActivation(gate)
	metal.Free(gate)
	var hidden *metal.Array
	if m.UpProj != nil {
		up := m.UpProj.Forward(x)
		hidden = metal.Mul(activated, up)
		metal.Free(activated, up)
	} else {
		hidden = activated
	}
	out := m.DownProj.Forward(hidden)
	metal.Free(hidden)
	return out
}

func (p *Gemma4VisionPooler) Forward(hidden *metal.Array, gridH, gridW int32) *metal.Array {
	// Stack-allocated shape scratch — per-vision-pass pooler entrypoint.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := hidden.ShapeInto(shapeBuf[:0])
	B, L, H := shape[0], shape[1], shape[2]
	k := p.PoolingKernelSize
	var pooled *metal.Array

	if k > 1 && gridH > 0 && gridW > 0 && gridH%k == 0 && gridW%k == 0 && gridH*gridW == L {
		pooled = p.poolByGrid(hidden, B, gridH, gridW, H, k)
	} else if k > 1 && L%(k*k) == 0 {
		outLen := L / (k * k)
		grouped := metal.Reshape(hidden, B, outLen, k*k, H)
		mean := metal.Mean(grouped, 2, false)
		metal.Free(grouped)
		pooled = metal.Reshape(mean, B*outLen, H)
		metal.Free(mean)
	} else {
		pooled = metal.Reshape(hidden, B*L, H)
	}

	scaled := metal.MulScalar(pooled, p.EmbeddingScale)
	metal.Free(pooled)
	return scaled
}

func (p *Gemma4VisionPooler) poolByGrid(hidden *metal.Array, B, gridH, gridW, H, k int32) *metal.Array {
	rows := gridH / k
	cols := gridW / k
	groups := make([]*metal.Array, 0, rows*cols)
	for y := range rows {
		for x := range cols {
			indices := make([]int32, 0, k*k)
			for dy := range k {
				for dx := range k {
					indices = append(indices, (y*k+dy)*gridW+(x*k+dx))
				}
			}
			idx := metal.FromValues(indices, len(indices))
			patches := metal.Take(hidden, idx, 1)
			mean := metal.Mean(patches, 1, false)
			expanded := metal.ExpandDims(mean, 1)
			metal.Free(idx, patches, mean)
			groups = append(groups, expanded)
		}
	}
	combined := metal.Concatenate(groups, 1)
	metal.Free(groups...)
	flat := metal.Reshape(combined, B*rows*cols, H)
	metal.Free(combined)
	return flat
}

func (p *Gemma4MultiModalProjector) Forward(x *metal.Array) *metal.Array {
	if p == nil {
		return x.Clone()
	}
	normed := metal.RMSNormNoScale(x, p.Eps)
	if p.Projection != nil {
		out := p.Projection.Forward(normed)
		metal.Free(normed)
		return out
	}
	if p.Linear1 != nil && p.Linear2 != nil {
		hidden := p.Linear1.Forward(normed)
		activated := metal.GeluActivation(hidden)
		metal.Free(hidden, normed)
		out := p.Linear2.Forward(activated)
		metal.Free(activated)
		return out
	}
	return normed
}

func gemma4VisionGridForPatchCount(patches, poolKernel int32) (int32, int32) {
	if patches <= 0 {
		return 0, 0
	}
	bestH, bestW := int32(1), patches
	bestDelta := patches
	for h := int32(1); h*h <= patches; h++ {
		if patches%h != 0 {
			continue
		}
		w := patches / h
		if poolKernel > 1 && (h%poolKernel != 0 || w%poolKernel != 0) {
			continue
		}
		delta := w - h
		if delta < 0 {
			delta = -delta
		}
		if delta < bestDelta {
			bestH, bestW = h, w
			bestDelta = delta
		}
	}
	return bestH, bestW
}

func gemma4VisionTrackRMSNorm(retained map[*metal.Array]struct{}, norm *metal.RMSNormModule) {
	if norm == nil {
		return
	}
	gemma4TrackArrays(retained, norm.Weight)
}

func gemma4VisionRetainedWeights(vision *Gemma4VisionModel, projector *Gemma4MultiModalProjector) map[*metal.Array]struct{} {
	retained := make(map[*metal.Array]struct{})
	if vision != nil {
		if vision.PatchEmbedder != nil {
			gemma4TrackLinear(retained, vision.PatchEmbedder.InputProj)
			gemma4TrackArrays(retained, vision.PatchEmbedder.PatchConvWeight, vision.PatchEmbedder.PositionEmbeddingTable)
		}
		gemma4VisionTrackRMSNorm(retained, vision.PostLayernorm)
		gemma4TrackArrays(retained, vision.StdBias, vision.StdScale)
		if vision.Encoder != nil {
			for _, layer := range vision.Encoder.Layers {
				if layer == nil {
					continue
				}
				gemma4VisionTrackRMSNorm(retained, layer.InputNorm)
				gemma4VisionTrackRMSNorm(retained, layer.PostAttnNorm)
				gemma4VisionTrackRMSNorm(retained, layer.PreFFNorm)
				gemma4VisionTrackRMSNorm(retained, layer.PostFFNorm)
				if attn := layer.Attention; attn != nil {
					gemma4TrackLinear(retained, attn.QProj)
					gemma4TrackLinear(retained, attn.KProj)
					gemma4TrackLinear(retained, attn.VProj)
					gemma4TrackLinear(retained, attn.OProj)
					gemma4VisionTrackRMSNorm(retained, attn.QNorm)
					gemma4VisionTrackRMSNorm(retained, attn.KNorm)
				}
				if mlp := layer.MLP; mlp != nil {
					gemma4TrackLinear(retained, mlp.GateProj)
					gemma4TrackLinear(retained, mlp.UpProj)
					gemma4TrackLinear(retained, mlp.DownProj)
				}
			}
		}
	}
	if projector != nil {
		gemma4TrackLinear(retained, projector.Projection)
		gemma4TrackLinear(retained, projector.Linear1)
		gemma4TrackLinear(retained, projector.Linear2)
	}
	return retained
}

func closeGemma4Vision(vision *Gemma4VisionModel, projector *Gemma4MultiModalProjector) {
	if vision != nil {
		if vision.PatchEmbedder != nil {
			metal.FreeLinear(vision.PatchEmbedder.InputProj)
			metal.Free(vision.PatchEmbedder.PatchConvWeight, vision.PatchEmbedder.PositionEmbeddingTable)
		}
		metal.FreeRMSNorm(vision.PostLayernorm)
		metal.Free(vision.StdBias, vision.StdScale)
		if vision.Encoder != nil {
			for _, layer := range vision.Encoder.Layers {
				if layer == nil {
					continue
				}
				metal.FreeRMSNorm(layer.InputNorm)
				metal.FreeRMSNorm(layer.PostAttnNorm)
				metal.FreeRMSNorm(layer.PreFFNorm)
				metal.FreeRMSNorm(layer.PostFFNorm)
				if attn := layer.Attention; attn != nil {
					metal.FreeLinear(attn.QProj)
					metal.FreeLinear(attn.KProj)
					metal.FreeLinear(attn.VProj)
					metal.FreeLinear(attn.OProj)
					metal.FreeRMSNorm(attn.QNorm)
					metal.FreeRMSNorm(attn.KNorm)
				}
				if mlp := layer.MLP; mlp != nil {
					metal.FreeLinear(mlp.GateProj)
					metal.FreeLinear(mlp.UpProj)
					metal.FreeLinear(mlp.DownProj)
				}
			}
		}
	}
	if projector != nil {
		metal.FreeLinear(projector.Projection)
		metal.FreeLinear(projector.Linear1)
		metal.FreeLinear(projector.Linear2)
	}
}
