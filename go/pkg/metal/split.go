// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"

	core "dappco.re/go"
)

// SplitState is the Metal-side state retained across split-inference calls.
type SplitState struct {
	Tokens      []int32
	Hidden      []float32
	HiddenShape []int32
	Layers      int

	caches []Cache
}

// Close releases the KV cache state held by the split state.
func (state *SplitState) Close() {
	if state == nil {
		return
	}
	FreeCaches(state.caches)
	state.caches = nil
}

// SplitAttentionRequest asks the local runtime to run one attention layer.
type SplitAttentionRequest struct {
	Layer       int
	Hidden      []float32
	HiddenShape []int32
}

// SplitAttentionResult is the hidden state after local attention.
type SplitAttentionResult struct {
	Hidden      []float32
	HiddenShape []int32
}

// SplitSampleRequest asks the local runtime to project logits and sample.
type SplitSampleRequest struct {
	Tokens      []int32
	Hidden      []float32
	HiddenShape []int32
	Config      GenerateConfig
}

// SplitSampleResult carries the sampled token and the next-token embedding.
type SplitSampleResult struct {
	TokenID     int32
	Hidden      []float32
	HiddenShape []int32
}

// SplitPrefill tokenises prompt and prepares the first local hidden state.
func (m *Model) SplitPrefill(ctx context.Context, prompt string) (*SplitState, error) {
	if m == nil || m.tokenizer == nil {
		return nil, core.NewError("mlx: split prefill tokenizer is nil")
	}
	return m.SplitPrefillTokens(ctx, m.tokenizer.Encode(prompt))
}

// SplitPrefillTokens prepares local split state from already-tokenised input.
func (m *Model) SplitPrefillTokens(ctx context.Context, tokens []int32) (*SplitState, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

	var (
		state    *SplitState
		splitErr error
	)
	if deviceErr := m.withDevice(func() {
		state, splitErr = m.splitPrefillTokensLocked(ctx, tokens)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return state, splitErr
}

func (m *Model) splitPrefillTokensLocked(ctx context.Context, tokens []int32) (*SplitState, error) {
	if len(tokens) == 0 {
		return nil, core.NewError("mlx: split prefill tokens are empty")
	}
	switch dense := m.model.(type) {
	case DenseSplitParts:
		caches := m.newCaches()
		state, err := splitPrefillDenseTokens(ctx, dense, tokens, caches)
		if err != nil {
			FreeCaches(caches)
			return nil, err
		}
		return state, nil
	default:
		return nil, core.Errorf("mlx: split prefill supports qwen2/qwen3 local attention, got %s", m.ModelType())
	}
}

func splitPrefillDenseTokens(ctx context.Context, dense DenseSplitParts, tokens []int32, caches []Cache) (*SplitState, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if dense == nil || dense.SplitEmbedding() == nil {
		return nil, core.NewError("mlx: dense split prefill missing embeddings")
	}
	vInput := FromValues(tokens, len(tokens))
	input := Reshape2(vInput, 1, int32(len(tokens)))
	Free(vInput)
	hidden := dense.SplitEmbedding().Forward(input)
	Free(input)
	if hidden == nil {
		return nil, core.NewError("mlx: dense split prefill returned nil hidden state")
	}
	if err := Eval(hidden); err != nil {
		Free(hidden)
		return nil, err
	}
	Detach(hidden)
	shape := hidden.Shape()
	state := &SplitState{
		Tokens:      append([]int32(nil), tokens...),
		Hidden:      hidden.Floats(),
		HiddenShape: append([]int32(nil), shape...),
		Layers:      len(dense.SplitDecoderLayers()),
		caches:      caches,
	}
	Free(hidden)
	return state, nil
}

// SplitForwardAttention runs one dense model local attention layer.
func (m *Model) SplitForwardAttention(ctx context.Context, state *SplitState, req SplitAttentionRequest) (SplitAttentionResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SplitAttentionResult{}, err
	}
	if m == nil || m.model == nil {
		return SplitAttentionResult{}, core.NewError("mlx: model is nil")
	}
	if state == nil {
		return SplitAttentionResult{}, core.NewError("mlx: split state is nil")
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return SplitAttentionResult{}, err
	}
	defer release()

	var (
		result   SplitAttentionResult
		splitErr error
	)
	if deviceErr := m.withDevice(func() {
		result, splitErr = m.splitForwardAttentionLocked(ctx, state, req)
	}); deviceErr != nil {
		return SplitAttentionResult{}, deviceErr
	}
	return result, splitErr
}

func (m *Model) splitForwardAttentionLocked(ctx context.Context, state *SplitState, req SplitAttentionRequest) (SplitAttentionResult, error) {
	switch dense := m.model.(type) {
	case DenseSplitParts:
		return splitForwardDenseAttention(ctx, dense, state, req)
	default:
		return SplitAttentionResult{}, core.Errorf("mlx: split attention supports qwen2/qwen3, got %s", m.ModelType())
	}
}

func splitForwardDenseAttention(ctx context.Context, dense DenseSplitParts, state *SplitState, req SplitAttentionRequest) (SplitAttentionResult, error) {
	select {
	case <-ctx.Done():
		return SplitAttentionResult{}, ctx.Err()
	default:
	}
	if dense == nil || dense.SplitConfig() == nil {
		return SplitAttentionResult{}, core.NewError("mlx: dense split attention missing config")
	}
	layers := dense.SplitDecoderLayers()
	if req.Layer < 0 || req.Layer >= len(layers) {
		return SplitAttentionResult{}, core.Errorf("mlx: dense split attention layer %d out of range", req.Layer)
	}
	if req.Layer >= len(state.caches) || state.caches[req.Layer] == nil {
		return SplitAttentionResult{}, core.Errorf("mlx: dense split attention cache %d unavailable", req.Layer)
	}
	layer := layers[req.Layer]
	if layer == nil || layer.InputNorm == nil || layer.Attention == nil {
		return SplitAttentionResult{}, core.Errorf("mlx: dense split attention layer %d is incomplete", req.Layer)
	}
	hidden := req.Hidden
	if len(hidden) == 0 {
		hidden = state.Hidden
	}
	shape := req.HiddenShape
	if len(shape) == 0 {
		shape = state.HiddenShape
	}
	if len(hidden) == 0 || len(shape) != 3 {
		return SplitAttentionResult{}, core.NewError("mlx: dense split attention requires rank-3 hidden state")
	}
	input := FromValues(hidden, splitShapeInts(shape)...)
	cfg := dense.SplitConfig()
	normed := layer.InputNorm.Forward(input, cfg.RMSNormEps)
	attnOut := layer.Attention.Forward(normed, state.caches[req.Layer], shape[0], shape[1], nil, cfg)
	Free(normed)
	out := Add(input, attnOut)
	Free(input, attnOut)
	if err := Eval(out); err != nil {
		Free(out)
		return SplitAttentionResult{}, err
	}
	Detach(out)
	resultShape := out.Shape()
	result := SplitAttentionResult{
		Hidden:      out.Floats(),
		HiddenShape: append([]int32(nil), resultShape...),
	}
	state.Hidden = append([]float32(nil), result.Hidden...)
	state.HiddenShape = append([]int32(nil), result.HiddenShape...)
	Free(out)
	return result, nil
}

// SplitSample projects the final hidden state to logits and samples one token.
func (m *Model) SplitSample(ctx context.Context, state *SplitState, req SplitSampleRequest) (SplitSampleResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SplitSampleResult{}, err
	}
	if m == nil || m.model == nil {
		return SplitSampleResult{}, core.NewError("mlx: model is nil")
	}
	if state == nil {
		return SplitSampleResult{}, core.NewError("mlx: split state is nil")
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return SplitSampleResult{}, err
	}
	defer release()

	var (
		result   SplitSampleResult
		splitErr error
	)
	if deviceErr := m.withDevice(func() {
		result, splitErr = m.splitSampleLocked(ctx, state, req)
	}); deviceErr != nil {
		return SplitSampleResult{}, deviceErr
	}
	return result, splitErr
}

func (m *Model) splitSampleLocked(ctx context.Context, state *SplitState, req SplitSampleRequest) (SplitSampleResult, error) {
	switch dense := m.model.(type) {
	case DenseSplitParts:
		return splitSampleDense(ctx, dense, state, req)
	default:
		return SplitSampleResult{}, core.Errorf("mlx: split sample supports qwen2/qwen3, got %s", m.ModelType())
	}
}

func splitSampleDense(ctx context.Context, dense DenseSplitParts, state *SplitState, req SplitSampleRequest) (SplitSampleResult, error) {
	select {
	case <-ctx.Done():
		return SplitSampleResult{}, ctx.Err()
	default:
	}
	if dense == nil || dense.SplitConfig() == nil {
		return SplitSampleResult{}, core.NewError("mlx: dense split sample missing config")
	}
	if dense.SplitNorm() == nil || dense.SplitNorm().Weight == nil || dense.SplitOutput() == nil {
		return SplitSampleResult{}, core.NewError("mlx: dense split sample missing output projection")
	}
	hidden := req.Hidden
	if len(hidden) == 0 {
		hidden = state.Hidden
	}
	shape := req.HiddenShape
	if len(shape) == 0 {
		shape = state.HiddenShape
	}
	if len(hidden) == 0 || len(shape) != 3 {
		return SplitSampleResult{}, core.NewError("mlx: dense split sample requires rank-3 hidden state")
	}
	input := FromValues(hidden, splitShapeInts(shape)...)
	normed := dense.SplitNorm().Forward(input, dense.SplitConfig().RMSNormEps)
	logits := dense.SplitOutput().Forward(normed)
	Free(input, normed)

	lastPos, err := materializeLastTokenLogits(logits)
	if err != nil {
		return SplitSampleResult{}, err
	}
	if req.Config.RepeatPenalty > 1.0 && len(req.Tokens) > 0 {
		oldLastPos := lastPos
		lastPos = applyRepeatPenalty(lastPos, req.Tokens, req.Config.RepeatPenalty)
		Free(oldLastPos)
	}
	sampler := newSampler(req.Config.Temperature, req.Config.TopP, req.Config.MinP, req.Config.TopK)
	next := sampler.Sample(lastPos)
	if err := Eval(next); err != nil {
		Free(lastPos, next)
		return SplitSampleResult{}, err
	}
	id := int32(next.Int())
	Free(lastPos, next)

	nextHidden, nextShape, err := splitDenseEmbedNextToken(ctx, dense, id)
	if err != nil {
		return SplitSampleResult{}, err
	}
	state.Tokens = append(state.Tokens, id)
	state.Hidden = append([]float32(nil), nextHidden...)
	state.HiddenShape = append([]int32(nil), nextShape...)
	return SplitSampleResult{
		TokenID:     id,
		Hidden:      nextHidden,
		HiddenShape: nextShape,
	}, nil
}

func splitDenseEmbedNextToken(ctx context.Context, dense DenseSplitParts, id int32) ([]float32, []int32, error) {
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}
	if dense == nil || dense.SplitEmbedding() == nil {
		return nil, nil, core.NewError("mlx: dense split sample missing embeddings")
	}
	input := FromSingleInt32Matrix(id)
	hidden := dense.SplitEmbedding().Forward(input)
	Free(input)
	if hidden == nil {
		return nil, nil, core.NewError("mlx: dense split sample returned nil next hidden state")
	}
	if err := Eval(hidden); err != nil {
		Free(hidden)
		return nil, nil, err
	}
	Detach(hidden)
	shape := hidden.Shape()
	values := hidden.Floats()
	Free(hidden)
	return values, append([]int32(nil), shape...), nil
}

func splitShapeInts(shape []int32) []int {
	out := make([]int, len(shape))
	for i, dim := range shape {
		out[i] = int(dim)
	}
	return out
}
