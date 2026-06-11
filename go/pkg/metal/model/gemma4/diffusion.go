// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// DiffusionGemma — the block-diffusion gemma4 (docs/RFC.diffusion-gemma.md).
//
// The checkpoint's weight-tied trunk (model.decoder.*) loads as a standard
// Gemma4Model through the diffusion_gemma architecture profile; this file
// carries the diffusion extras on top: the self-conditioning block, the
// encoder-role layer scalars, and the model_type registration. The canvas
// denoising sampler is the next unit — until it lands, the trunk serves the
// causal (encoder-mode) forward, which is exactly the append-to-cache shape
// the sampler reuses per accepted canvas.
type DiffusionGemmaModel struct {
	*Gemma4Model

	// SelfCondPreNorm + SelfCondMLP implement the reference SelfConditioning
	// block: RMSNorm_noscale(canvas_embed + MLP(RMSNorm_scaled(sc_signal))).
	// The post-norm carries no weight by design (with_scale=False upstream),
	// so only the pre-norm scale loads.
	SelfCondPreNorm *metal.RMSNormModule
	SelfCondMLP     *metal.MLP

	// EncoderLayerScalars are the encoder-role per-layer hidden multipliers
	// (model.encoder.language_model.layers.N.layer_scalar). The decoder-role
	// scalars load on the trunk layers as LayerScalar — one trunk, two roles,
	// distinguished by which scalar set a forward applies.
	EncoderLayerScalars []*metal.Array

	// CanvasLength is the checkpoint's declared denoising block size.
	CanvasLength int32
	// EOSTokens are the checkpoint's declared end-of-sequence ids.
	EOSTokens []int32
}

// LoadDiffusionGemma loads a DiffusionGemma checkpoint: the trunk via the
// shared gemma4 builder, the diffusion extras collected alongside.
//
//	m, err := gemma4.LoadDiffusionGemma(path)
func LoadDiffusionGemma(modelPath string) (*DiffusionGemmaModel, error) {
	const op = "gemma4.LoadDiffusionGemma"
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E(op, "load config", err)
	}
	cfg, err := parseGemma4Config([]byte(str))
	if err != nil {
		return nil, core.E(op, "parse config", err)
	}
	canvasLength, eosTokens := parseDiffusionConfigExtras([]byte(str))
	if err := validateGemma4QuantizationConfig(cfg.Quantization); err != nil {
		return nil, core.E(op, "validate quantization", err)
	}

	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E(op, "load tokenizer", err)
	}

	rawWeights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E(op, "load weights", err)
	}

	// The encoder-role scalars must leave the raw map BEFORE the trunk
	// sanitize: its skip rules discard (and free) everything under
	// encoder.*, including these. Extract-and-delete, the vision pattern.
	encoderScalars := extractDiffusionEncoderScalars(rawWeights, cfg.NumHiddenLayers)

	weights := sanitizeGemma4WeightsAs("diffusion_gemma", rawWeights)

	// The self-conditioning block unwraps to a bare self_conditioning.*
	// root on purpose (no model. re-rooting) — collect it before the build
	// so its arrays join the retained set.
	preNormWeight := gemma4WeightAny(weights, "self_conditioning.pre_norm.weight", "self_conditioning.pre_norm")
	scMLP := &metal.MLP{
		GateProj: gemma4Linear(weights, "self_conditioning.gate_proj", cfg.Quantization),
		UpProj:   gemma4Linear(weights, "self_conditioning.up_proj", cfg.Quantization),
		DownProj: gemma4Linear(weights, "self_conditioning.down_proj", cfg.Quantization),
	}
	if preNormWeight == nil || scMLP.GateProj == nil || scMLP.UpProj == nil || scMLP.DownProj == nil {
		return nil, core.E(op, "self-conditioning block incomplete in checkpoint", nil)
	}
	if int32(len(encoderScalars)) != cfg.NumHiddenLayers {
		return nil, core.E(op, core.Sprintf("encoder layer scalars: %d of %d", len(encoderScalars), cfg.NumHiddenLayers), nil)
	}

	retainExtra := append([]*metal.Array{preNormWeight}, encoderScalars...)
	for _, linear := range []*metal.Linear{scMLP.GateProj, scMLP.UpProj, scMLP.DownProj} {
		retainExtra = append(retainExtra, linear.Weight, linear.Scales, linear.Biases)
	}

	trunk, err := buildGemma4FromWeights(op, cfg, tok, weights, nil, nil, retainExtra)
	if err != nil {
		return nil, err
	}

	return &DiffusionGemmaModel{
		Gemma4Model:         trunk,
		SelfCondPreNorm:     &metal.RMSNormModule{Weight: preNormWeight},
		SelfCondMLP:         scMLP,
		EncoderLayerScalars: encoderScalars,
		CanvasLength:        canvasLength,
		EOSTokens:           eosTokens,
	}, nil
}

// parseDiffusionConfigExtras reads the diffusion-specific top-level config
// fields: canvas_length (the denoising block size) and eos_token_id (int or
// list — HF ships both shapes).
func parseDiffusionConfigExtras(data []byte) (int32, []int32) {
	var wrapper struct {
		CanvasLength int32 `json:"canvas_length"`
		EOSTokenID   any   `json:"eos_token_id"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return 0, nil
	}
	var eos []int32
	switch v := wrapper.EOSTokenID.(type) {
	case float64:
		eos = []int32{int32(v)}
	case []any:
		for _, e := range v {
			if f, ok := e.(float64); ok {
				eos = append(eos, int32(f))
			}
		}
	}
	return wrapper.CanvasLength, eos
}

// extractDiffusionEncoderScalars pulls the encoder-role layer scalars out of
// the raw weight map (deleting them so the trunk sanitize cannot free them)
// and returns them indexed by layer.
func extractDiffusionEncoderScalars(raw map[string]*metal.Array, numLayers int32) []*metal.Array {
	scalars := make([]*metal.Array, numLayers)
	for i := int32(0); i < numLayers; i++ {
		base := core.Sprintf("model.encoder.language_model.layers.%d.layer_scalar", i)
		for _, name := range []string{base, base + ".weight"} {
			if arr, ok := raw[name]; ok && arr != nil {
				scalars[i] = arr
				delete(raw, name)
				break
			}
		}
	}
	out := scalars[:0]
	for _, arr := range scalars {
		if arr != nil {
			out = append(out, arr)
		}
	}
	return out
}

// Close releases the trunk weights, the diffusion extras, and the MLX
// allocator cache.
func (m *DiffusionGemmaModel) Close() error {
	if m == nil {
		return nil
	}
	metal.Free(m.EncoderLayerScalars...)
	m.EncoderLayerScalars = nil
	if m.SelfCondPreNorm != nil {
		metal.Free(m.SelfCondPreNorm.Weight)
		m.SelfCondPreNorm = nil
	}
	if m.SelfCondMLP != nil {
		for _, l := range []*metal.Linear{m.SelfCondMLP.GateProj, m.SelfCondMLP.UpProj, m.SelfCondMLP.DownProj} {
			if l != nil {
				metal.Free(l.Weight, l.Scales, l.Biases)
			}
		}
		m.SelfCondMLP = nil
	}
	closeGemma4(m.Gemma4Model)
	metal.ClearCache()
	return nil
}

func init() {
	metal.RegisterModelLoader("diffusion_gemma", func(p string, _ []byte) (metal.InternalModel, error) {
		return LoadDiffusionGemma(p)
	})
}
