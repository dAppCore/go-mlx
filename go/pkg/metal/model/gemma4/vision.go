// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "dappco.re/go/mlx/pkg/metal"

// Gemma4VisionRopeParameters holds the 2-D RoPE settings for the vision tower.
type Gemma4VisionRopeParameters struct {
	RopeType  string  `json:"rope_type"`
	RopeTheta float32 `json:"rope_theta"`
}

// Gemma4VisionConfig holds the Gemma 4 SigLIP-derived vision tower configuration.
type Gemma4VisionConfig struct {
	// Embedded neutral core — promotes ModelType/HiddenSize/IntermediateSize/
	// NumHiddenLayers/NumAttentionHeads/NumKeyValueHeads/HeadDim/RMSNormEps/
	// MaxPositionEmbeddings (the vision tower is a transformer; VocabSize is
	// carried by the core but unused here).
	metal.TransformerConfig

	ImageSize             int32                      `json:"image_size"`
	PatchSize             int32                      `json:"patch_size"`
	NumChannels           int32                      `json:"num_channels"`
	HiddenActivation      string                     `json:"hidden_activation"`
	LayerNormEps          float32                    `json:"layer_norm_eps"`
	MMEmbedDim            int32                      `json:"mm_embed_dim"`
	MMPosembSize          int32                      `json:"mm_posemb_size"`
	ModelPatchSize        int32                      `json:"model_patch_size"`
	NumSoftTokens         int32                      `json:"num_soft_tokens"`
	OutputProjDims        int32                      `json:"output_proj_dims"`
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
		TransformerConfig: metal.TransformerConfig{
			ModelType:             "gemma4_vision",
			HiddenSize:            768,
			IntermediateSize:      3072,
			NumHiddenLayers:       16,
			NumAttentionHeads:     12,
			NumKeyValueHeads:      12,
			HeadDim:               64,
			RMSNormEps:            1e-6,
			MaxPositionEmbeddings: 131072,
		},
		ImageSize:        896,
		PatchSize:        16,
		NumChannels:      3,
		HiddenActivation: "gelu_pytorch_tanh",
		LayerNormEps:     1e-6,
		MMEmbedDim:       768,
		MMPosembSize:     1120,
		ModelPatchSize:   48,
		NumSoftTokens:    280,
		OutputProjDims:   2048,
		RopeParameters: Gemma4VisionRopeParameters{
			RopeType:  "default",
			RopeTheta: 100,
		},
		PoolingKernelSize:     3,
		PositionEmbeddingSize: 10 * 1024,
		InitializerRange:      0.02,
	}
}

// normalizeGemma4VisionConfig fills only the scheme/constant fields that are the
// same across every Gemma 4 vision tower (RGB channels, the activation, the
// norm epsilon, the rope scheme, the pooling kernel) and the values that DERIVE
// from declared dims (head_dim = hidden/heads, kv-heads = heads, the
// layer/rms-norm-eps cross-fill). Every per-model DIMENSION — hidden_size,
// intermediate_size, layer/head counts, image/patch size, the multimodal
// projection dims, soft-token count, position-embedding size — is left exactly
// as the model declared it (config.json) or as inferGemma4VisionConfig derives
// it from the loaded tensors. No model's dimensions are guessed from another
// model's defaults.
func normalizeGemma4VisionConfig(cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		return nil
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_vision"
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 3 // RGB — physical, not a tuned guess
	}
	if cfg.HiddenActivation == "" {
		cfg.HiddenActivation = "gelu_pytorch_tanh"
	}
	// RMS/Layer-norm epsilon: cross-fill the two names, then the Gemma constant.
	if cfg.LayerNormEps == 0 && cfg.RMSNormEps != 0 {
		cfg.LayerNormEps = cfg.RMSNormEps
	}
	if cfg.RMSNormEps == 0 && cfg.LayerNormEps != 0 {
		cfg.RMSNormEps = cfg.LayerNormEps
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeParameters.RopeType == "" {
		cfg.RopeParameters.RopeType = "default"
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = 100
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	// Derivations from the model's own declared dims — not cross-model guesses.
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	return cfg
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
