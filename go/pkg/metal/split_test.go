// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"math"
	"testing"
)

type splitDenseTestModel struct {
	embed     *Embedding
	layers    []*DenseDecoderLayer
	norm      *RMSNormModule
	output    *Linear
	cfg       *DenseConfig
	modelType string
}

func (m *splitDenseTestModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (m *splitDenseTestModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (m *splitDenseTestModel) NewCache() []Cache {
	caches := make([]Cache, len(m.layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}
func (m *splitDenseTestModel) NumLayers() int                           { return len(m.layers) }
func (m *splitDenseTestModel) Tokenizer() *Tokenizer                    { return nil }
func (m *splitDenseTestModel) ModelType() string                        { return m.modelType }
func (m *splitDenseTestModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter      { return nil }
func (m *splitDenseTestModel) SplitEmbedding() *Embedding               { return m.embed }
func (m *splitDenseTestModel) SplitDecoderLayers() []*DenseDecoderLayer { return m.layers }
func (m *splitDenseTestModel) SplitNorm() *RMSNormModule                { return m.norm }
func (m *splitDenseTestModel) SplitOutput() *Linear                     { return m.output }
func (m *splitDenseTestModel) SplitConfig() *DenseConfig                { return m.cfg }

func TestSplit_Qwen3SplitPrefillAndAttention_Good(t *testing.T) {
	model := newSplitQwen3TestModel()
	defer model.Close()

	state, err := model.SplitPrefillTokens(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("SplitPrefillTokens: %v", err)
	}
	defer state.Close()

	if state.Layers != 1 {
		t.Fatalf("layers = %d, want 1", state.Layers)
	}
	if !equalSplitInt32Slices(state.HiddenShape, []int32{1, 1, 2}) {
		t.Fatalf("prefill hidden shape = %v, want [1 1 2]", state.HiddenShape)
	}
	if len(state.Hidden) != 2 {
		t.Fatalf("prefill hidden len = %d, want 2", len(state.Hidden))
	}

	result, err := model.SplitForwardAttention(context.Background(), state, SplitAttentionRequest{
		Layer:       0,
		Hidden:      state.Hidden,
		HiddenShape: state.HiddenShape,
	})
	if err != nil {
		t.Fatalf("SplitForwardAttention: %v", err)
	}
	if !equalSplitInt32Slices(result.HiddenShape, []int32{1, 1, 2}) {
		t.Fatalf("attention hidden shape = %v, want [1 1 2]", result.HiddenShape)
	}
	if len(result.Hidden) != 2 {
		t.Fatalf("attention hidden len = %d, want 2", len(result.Hidden))
	}
	if state.caches[0].Offset() != 1 {
		t.Fatalf("cache offset = %d, want 1", state.caches[0].Offset())
	}

	sample, err := model.SplitSample(context.Background(), state, SplitSampleRequest{
		Hidden:      result.Hidden,
		HiddenShape: result.HiddenShape,
		Config:      GenerateConfig{Temperature: 0},
	})
	if err != nil {
		t.Fatalf("SplitSample: %v", err)
	}
	if sample.TokenID != 1 {
		t.Fatalf("sample token = %d, want 1", sample.TokenID)
	}
	if !equalSplitInt32Slices(sample.HiddenShape, []int32{1, 1, 2}) {
		t.Fatalf("sample hidden shape = %v, want [1 1 2]", sample.HiddenShape)
	}
	if len(sample.Hidden) != 2 {
		t.Fatalf("sample hidden len = %d, want 2", len(sample.Hidden))
	}
}

func newSplitQwen3TestModel() *Model {
	embedW := FromValues([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	inNormW := FromValues([]float32{1, 1}, 2)
	qW := FromValues([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	kW := FromValues([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	vW := FromValues([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	oW := FromValues([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	finalNormW := FromValues([]float32{1, 1}, 2)
	outputW := FromValues([]float32{
		0, 1,
		2, 0,
	}, 2, 2)
	Materialize(embedW, inNormW, qW, kW, vW, oW, finalNormW, outputW)
	qwen := &splitDenseTestModel{
		embed: &Embedding{Weight: embedW},
		layers: []*DenseDecoderLayer{{
			InputNorm: &RMSNormModule{Weight: inNormW},
			Attention: &GQAAttention{
				QProj: NewLinear(qW, nil),
				KProj: NewLinear(kW, nil),
				VProj: NewLinear(vW, nil),
				OProj: NewLinear(oW, nil),
			},
		}},
		norm:   &RMSNormModule{Weight: finalNormW},
		output: NewLinear(outputW, nil),
		cfg: &DenseConfig{
			HiddenSize:        2,
			NumHiddenLayers:   1,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			HeadDim:           2,
			RMSNormEps:        1e-6,
			RopeTheta:         10000,
			Scale:             float32(1 / math.Sqrt(2)),
		},
		modelType: "qwen2",
	}
	return &Model{
		model:     qwen,
		modelType: "qwen2",
		device:    DeviceGPU,
	}
}

func equalSplitInt32Slices(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
