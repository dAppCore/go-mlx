// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

const gemma4AudioTowerPrefix = "audio_tower."

// gemma4AudioEncoderPresent reports whether the sanitized audio weight map
// carries Conformer tower tensors. 12B-unified ships only embed_audio (the
// projector) and stays encoder-free.
func gemma4AudioEncoderPresent(weights map[string]*metal.Array) bool {
	return gemma4WeightAny(weights, gemma4AudioTowerPrefix+"subsample_conv_projection.input_proj_linear.weight") != nil
}

// buildGemma4AudioEncoder assembles the Conformer audio tower from the
// sanitized audio weight map. Returns (nil, nil) when the checkpoint has no
// tower (projector-only variants); errors loud when the tower is present but
// the config or weight map is incomplete — a guessed encoder produces
// garbage audio silently.
func buildGemma4AudioEncoder(cfg *Gemma4TextConfig, weights map[string]*metal.Array) (*Gemma4AudioEncoder, error) {
	if !gemma4AudioEncoderPresent(weights) {
		return nil, nil
	}
	if cfg == nil || cfg.AudioConfig == nil {
		return nil, core.NewError("gemma4: audio tower weights present but config declares no audio_config")
	}
	audioCfg := normalizeGemma4AudioConfig(cfg.AudioConfig)
	if err := validateGemma4AudioEncoderConfig(audioCfg); err != nil {
		return nil, err
	}
	var quant *metal.QuantizationConfig
	if cfg != nil {
		quant = cfg.Quantization
	}

	headDim := audioCfg.HiddenSize / audioCfg.NumAttentionHeads
	enc := &Gemma4AudioEncoder{
		Cfg:   audioCfg,
		GCMin: metal.FromValue(-audioCfg.GradientClipping),
		GCMax: metal.FromValue(audioCfg.GradientClipping),
		// One sinusoid table and one folded per-layer q-scale dtype shared by
		// every layer's attention; positions are [ContextLeft-1 .. 0].
		PosEmbed: gemma4AudioPositionTable(audioCfg.AttentionContextLeft, audioCfg.HiddenSize),
	}

	sub, err := buildGemma4AudioSubsample(audioCfg, weights, quant)
	if err != nil {
		closeGemma4AudioEncoder(enc)
		return nil, err
	}
	enc.Subsample = sub

	enc.Layers = make([]*Gemma4AudioLayer, audioCfg.NumHiddenLayers)
	for i := range enc.Layers {
		layer, layerErr := buildGemma4AudioLayer(audioCfg, weights, quant, enc, int32(i), headDim)
		if layerErr != nil {
			closeGemma4AudioEncoder(enc)
			return nil, layerErr
		}
		enc.Layers[i] = layer
	}

	enc.OutputProj = gemma4Linear(weights, gemma4AudioTowerPrefix+"output_proj", quant)
	if enc.OutputProj == nil {
		closeGemma4AudioEncoder(enc)
		return nil, core.NewError("gemma4: audio tower missing output_proj")
	}
	return enc, nil
}

func validateGemma4AudioEncoderConfig(cfg *Gemma4AudioConfig) error {
	switch {
	case cfg.HiddenSize <= 0,
		cfg.NumHiddenLayers <= 0,
		cfg.NumAttentionHeads <= 0,
		cfg.AttentionChunkSize <= 0,
		cfg.AttentionContextLeft <= 0,
		cfg.ConvKernelSize <= 0,
		len(cfg.SubsamplingConvChannels) != 2,
		cfg.OutputProjDims <= 0,
		cfg.ResidualWeight == 0,
		cfg.AttentionLogitCap == 0:
		return core.E("gemma4.audio", core.Sprintf(
			"audio_config incomplete for the Conformer encoder: hidden=%d layers=%d heads=%d chunk=%d left=%d kernel=%d channels=%v proj=%d residual=%v cap=%v",
			cfg.HiddenSize, cfg.NumHiddenLayers, cfg.NumAttentionHeads,
			cfg.AttentionChunkSize, cfg.AttentionContextLeft, cfg.ConvKernelSize,
			cfg.SubsamplingConvChannels, cfg.OutputProjDims, cfg.ResidualWeight, cfg.AttentionLogitCap), nil)
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return core.E("gemma4.audio", core.Sprintf("hidden_size %d not divisible by heads %d", cfg.HiddenSize, cfg.NumAttentionHeads), nil)
	}
	return nil
}

func buildGemma4AudioSubsample(cfg *Gemma4AudioConfig, weights map[string]*metal.Array, quant *metal.QuantizationConfig) (*Gemma4AudioSubSampleConvProjection, error) {
	layer := func(idx int) (*Gemma4AudioSubSampleConvLayer, error) {
		base := core.Sprintf("%ssubsample_conv_projection.layer%d", gemma4AudioTowerPrefix, idx)
		conv := gemma4WeightAny(weights, base+".conv.weight")
		norm := gemma4WeightAny(weights, base+".norm.weight")
		if conv == nil || norm == nil {
			return nil, core.E("gemma4.audio", core.Sprintf("subsample layer%d conv/norm weights missing", idx), nil)
		}
		outC := conv.Dim(0)
		return &Gemma4AudioSubSampleConvLayer{
			ConvWeight: gemma4AudioConvToNHWC(conv),
			NormWeight: norm,
			NormBias:   metal.Zeros([]int32{int32(outC)}, norm.Dtype()),
			Eps:        cfg.RMSNormEps,
		}, nil
	}
	l0, err := layer(0)
	if err != nil {
		return nil, err
	}
	l1, err := layer(1)
	if err != nil {
		freeGemma4AudioSubsampleLayer(l0)
		return nil, err
	}
	proj := gemma4Linear(weights, gemma4AudioTowerPrefix+"subsample_conv_projection.input_proj_linear", quant)
	if proj == nil {
		freeGemma4AudioSubsampleLayer(l0)
		freeGemma4AudioSubsampleLayer(l1)
		return nil, core.NewError("gemma4: audio subsample input_proj_linear missing")
	}
	return &Gemma4AudioSubSampleConvProjection{Layer0: l0, Layer1: l1, InputProj: proj}, nil
}

func buildGemma4AudioLayer(cfg *Gemma4AudioConfig, weights map[string]*metal.Array, quant *metal.QuantizationConfig, enc *Gemma4AudioEncoder, idx, headDim int32) (*Gemma4AudioLayer, error) {
	base := core.Sprintf("%slayers.%d.", gemma4AudioTowerPrefix, idx)
	norm := func(name string) *metal.Array { return gemma4WeightAny(weights, base+name+".weight") }

	ff := func(name string) (*Gemma4AudioFeedForward, error) {
		ffw1 := gemma4AudioClippable(weights, base+name+".ffw_layer_1", quant)
		ffw2 := gemma4AudioClippable(weights, base+name+".ffw_layer_2", quant)
		pre := norm(name + ".pre_layer_norm")
		post := norm(name + ".post_layer_norm")
		if ffw1 == nil || ffw2 == nil || pre == nil || post == nil {
			return nil, core.E("gemma4.audio", core.Sprintf("layer %d %s incomplete", idx, name), nil)
		}
		return &Gemma4AudioFeedForward{
			FFW1: ffw1, FFW2: ffw2, PreNorm: pre, PostNorm: post,
			Eps: cfg.RMSNormEps, Residual: cfg.ResidualWeight, Act: cfg.HiddenAct,
			ClipMin: enc.GCMin, ClipMax: enc.GCMax,
		}, nil
	}
	ff1, err := ff("feed_forward1")
	if err != nil {
		return nil, err
	}
	ff2, err := ff("feed_forward2")
	if err != nil {
		return nil, err
	}

	perDim := gemma4WeightAny(weights, base+"self_attn.per_dim_scale")
	relK := gemma4Linear(weights, base+"self_attn.relative_k_proj", quant)
	attn := &Gemma4AudioAttention{
		QProj:         gemma4AudioClippable(weights, base+"self_attn.q_proj", quant),
		KProj:         gemma4AudioClippable(weights, base+"self_attn.k_proj", quant),
		VProj:         gemma4AudioClippable(weights, base+"self_attn.v_proj", quant),
		Post:          gemma4AudioClippable(weights, base+"self_attn.post", quant),
		RelativeKProj: relK,
		PosEmbed:      enc.PosEmbed,
		NumHeads:      cfg.NumAttentionHeads,
		HeadDim:       headDim,
		ChunkSize:     cfg.AttentionChunkSize,
		PastHorizon:   cfg.AttentionContextLeft - 1,
		FutureHorizon: cfg.AttentionContextRight,
		KScale:        float32(math.Log(1+math.E) / math.Ln2),
		LogitCap:      cfg.AttentionLogitCap,
		InvalidLogit:  cfg.AttentionInvalidLogitsValue,
	}
	if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.Post == nil || relK == nil || perDim == nil {
		return nil, core.E("gemma4.audio", core.Sprintf("layer %d self_attn incomplete", idx), nil)
	}
	attn.QScalePerDim = gemma4AudioFoldPerDimScale(perDim, headDim)

	lconv := &Gemma4AudioLightConv{
		LinearStart: gemma4AudioClippable(weights, base+"lconv1d.linear_start", quant),
		LinearEnd:   gemma4AudioClippable(weights, base+"lconv1d.linear_end", quant),
		PreNorm:     norm("lconv1d.pre_layer_norm"),
		ConvNorm:    norm("lconv1d.conv_norm"),
		Eps:         cfg.RMSNormEps,
		KernelSize:  cfg.ConvKernelSize,
		Channels:    cfg.HiddenSize,
		Act:         cfg.HiddenAct,
		ClipMin:     enc.GCMin,
		ClipMax:     enc.GCMax,
	}
	depthwise := gemma4WeightAny(weights, base+"lconv1d.depthwise_conv1d.weight")
	if lconv.LinearStart == nil || lconv.LinearEnd == nil || lconv.PreNorm == nil || lconv.ConvNorm == nil || depthwise == nil {
		return nil, core.E("gemma4.audio", core.Sprintf("layer %d lconv1d incomplete", idx), nil)
	}
	lconv.DepthwiseWeight = gemma4AudioDepthwiseToNLC(depthwise)

	layer := &Gemma4AudioLayer{
		FeedForward1: ff1,
		FeedForward2: ff2,
		SelfAttn:     attn,
		LConv:        lconv,
		NormPreAttn:  norm("norm_pre_attn"),
		NormPostAttn: norm("norm_post_attn"),
		NormOut:      norm("norm_out"),
		Eps:          cfg.RMSNormEps,
		ClipMin:      enc.GCMin,
		ClipMax:      enc.GCMax,
	}
	if layer.NormPreAttn == nil || layer.NormPostAttn == nil || layer.NormOut == nil {
		return nil, core.E("gemma4.audio", core.Sprintf("layer %d block norms incomplete", idx), nil)
	}
	return layer, nil
}

// gemma4AudioClippable loads a checkpoint ClippableLinear: the wrapped linear
// at prefix.linear plus the recorded input/output clamp scalars.
func gemma4AudioClippable(weights map[string]*metal.Array, prefix string, quant *metal.QuantizationConfig) *Gemma4AudioClippableLinear {
	lin := gemma4Linear(weights, prefix+".linear", quant)
	if lin == nil {
		return nil
	}
	return &Gemma4AudioClippableLinear{
		Linear:    lin,
		InputMin:  gemma4WeightAny(weights, prefix+".input_min"),
		InputMax:  gemma4WeightAny(weights, prefix+".input_max"),
		OutputMin: gemma4WeightAny(weights, prefix+".output_min"),
		OutputMax: gemma4WeightAny(weights, prefix+".output_max"),
	}
}

// gemma4AudioConvToNHWC transposes a torch conv2d weight
// [out, in, kH, kW] to MLX's [out, kH, kW, in].
func gemma4AudioConvToNHWC(w *metal.Array) *metal.Array {
	t := metal.Transpose(w, 0, 2, 3, 1)
	out := metal.Contiguous(t)
	metal.Free(t)
	return out
}

// gemma4AudioDepthwiseToNLC transposes a torch depthwise conv1d weight
// [channels, 1, kernel] to MLX's [channels, kernel, 1].
func gemma4AudioDepthwiseToNLC(w *metal.Array) *metal.Array {
	t := metal.Transpose(w, 0, 2, 1)
	out := metal.Contiguous(t)
	metal.Free(t)
	return out
}

// gemma4AudioFoldPerDimScale precomputes q_scale * softplus(per_dim_scale)
// as a [1,1,1,headDim] broadcast factor — the reference applies both to the
// query every forward; folding them is a load-time constant.
func gemma4AudioFoldPerDimScale(perDim *metal.Array, headDim int32) *metal.Array {
	f32 := metal.AsType(perDim, metal.DTypeFloat32)
	e := metal.Exp(f32)
	metal.Free(f32)
	plus1 := metal.AddScalar(e, 1)
	metal.Free(e)
	softplus := metal.Log(plus1)
	metal.Free(plus1)
	qScale := float32(1 / (math.Sqrt(float64(headDim)) * math.Ln2))
	scaled := metal.MulScalar(softplus, qScale)
	metal.Free(softplus)
	out := metal.Reshape(scaled, 1, 1, 1, headDim)
	metal.Free(scaled)
	return out
}

// gemma4TrackAudioEncoder marks every array the audio encoder holds as
// retained — checkpoint weights it kept, plus the load-derived constants
// (transposed convs, folded per-dim scale, sinusoid table, clamp scalars).
func gemma4TrackAudioEncoder(retained map[*metal.Array]struct{}, e *Gemma4AudioEncoder) {
	if e == nil {
		return
	}
	clippable := func(l *Gemma4AudioClippableLinear) {
		if l == nil {
			return
		}
		gemma4TrackLinear(retained, l.Linear)
		gemma4TrackArrays(retained, l.InputMin, l.InputMax, l.OutputMin, l.OutputMax)
	}
	if s := e.Subsample; s != nil {
		for _, layer := range []*Gemma4AudioSubSampleConvLayer{s.Layer0, s.Layer1} {
			if layer != nil {
				gemma4TrackArrays(retained, layer.ConvWeight, layer.NormWeight, layer.NormBias)
			}
		}
		gemma4TrackLinear(retained, s.InputProj)
	}
	for _, layer := range e.Layers {
		if layer == nil {
			continue
		}
		for _, ff := range []*Gemma4AudioFeedForward{layer.FeedForward1, layer.FeedForward2} {
			if ff == nil {
				continue
			}
			clippable(ff.FFW1)
			clippable(ff.FFW2)
			gemma4TrackArrays(retained, ff.PreNorm, ff.PostNorm)
		}
		if attn := layer.SelfAttn; attn != nil {
			clippable(attn.QProj)
			clippable(attn.KProj)
			clippable(attn.VProj)
			clippable(attn.Post)
			gemma4TrackLinear(retained, attn.RelativeKProj)
			gemma4TrackArrays(retained, attn.QScalePerDim)
		}
		if lc := layer.LConv; lc != nil {
			clippable(lc.LinearStart)
			clippable(lc.LinearEnd)
			gemma4TrackArrays(retained, lc.DepthwiseWeight, lc.PreNorm, lc.ConvNorm)
		}
		gemma4TrackArrays(retained, layer.NormPreAttn, layer.NormPostAttn, layer.NormOut)
	}
	gemma4TrackLinear(retained, e.OutputProj)
	gemma4TrackArrays(retained, e.PosEmbed, e.GCMin, e.GCMax)
}

func freeGemma4AudioSubsampleLayer(l *Gemma4AudioSubSampleConvLayer) {
	if l == nil {
		return
	}
	metal.Free(l.ConvWeight, l.NormBias)
}

func closeGemma4AudioClippable(l *Gemma4AudioClippableLinear) {
	if l == nil {
		return
	}
	metal.FreeLinear(l.Linear)
}

func closeGemma4AudioEncoder(e *Gemma4AudioEncoder) {
	if e == nil {
		return
	}
	if e.Subsample != nil {
		freeGemma4AudioSubsampleLayer(e.Subsample.Layer0)
		freeGemma4AudioSubsampleLayer(e.Subsample.Layer1)
		metal.FreeLinear(e.Subsample.InputProj)
	}
	for _, layer := range e.Layers {
		if layer == nil {
			continue
		}
		for _, ff := range []*Gemma4AudioFeedForward{layer.FeedForward1, layer.FeedForward2} {
			if ff == nil {
				continue
			}
			closeGemma4AudioClippable(ff.FFW1)
			closeGemma4AudioClippable(ff.FFW2)
		}
		if attn := layer.SelfAttn; attn != nil {
			closeGemma4AudioClippable(attn.QProj)
			closeGemma4AudioClippable(attn.KProj)
			closeGemma4AudioClippable(attn.VProj)
			closeGemma4AudioClippable(attn.Post)
			metal.FreeLinear(attn.RelativeKProj)
			metal.Free(attn.QScalePerDim)
		}
		if lc := layer.LConv; lc != nil {
			closeGemma4AudioClippable(lc.LinearStart)
			closeGemma4AudioClippable(lc.LinearEnd)
			metal.Free(lc.DepthwiseWeight)
		}
	}
	metal.FreeLinear(e.OutputProj)
	metal.Free(e.PosEmbed, e.GCMin, e.GCMax)
}
