// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// withEncoderScalars swaps the encoder-role layer scalars in for the
// callback and restores the decoder set after — the one parametric
// difference between the two roles of the weight-tied trunk.
func TestWithEncoderScalars_SwapsAndRestores_Good(t *testing.T) {
	decoder0 := metal.FromValues([]float32{1}, 1)
	decoder1 := metal.FromValues([]float32{2}, 1)
	encoder0 := metal.FromValues([]float32{3}, 1)
	encoder1 := metal.FromValues([]float32{4}, 1)
	defer metal.Free(decoder0, decoder1, encoder0, encoder1)

	m := &DiffusionGemmaModel{
		Gemma4Model: &Gemma4Model{
			Layers: []*Gemma4DecoderLayer{
				{LayerScalar: decoder0},
				{LayerScalar: decoder1},
			},
		},
		EncoderLayerScalars: []*metal.Array{encoder0, encoder1},
	}

	called := false
	m.withEncoderScalars(func() {
		called = true
		if m.Layers[0].LayerScalar != encoder0 || m.Layers[1].LayerScalar != encoder1 {
			t.Fatal("encoder scalars not swapped in for the callback")
		}
		if m.EncoderLayerScalars[0] != decoder0 || m.EncoderLayerScalars[1] != decoder1 {
			t.Fatal("decoder scalars not parked on the model during the callback")
		}
	})
	if !called {
		t.Fatal("callback not invoked")
	}
	if m.Layers[0].LayerScalar != decoder0 || m.Layers[1].LayerScalar != decoder1 {
		t.Fatal("decoder scalars not restored after the callback")
	}
	if m.EncoderLayerScalars[0] != encoder0 || m.EncoderLayerScalars[1] != encoder1 {
		t.Fatal("encoder set not restored after the callback")
	}
}

func TestWithEncoderScalars_CountMismatchRunsUnswapped_Ugly(t *testing.T) {
	decoder0 := metal.FromValues([]float32{1}, 1)
	defer metal.Free(decoder0)
	m := &DiffusionGemmaModel{
		Gemma4Model:         &Gemma4Model{Layers: []*Gemma4DecoderLayer{{LayerScalar: decoder0}}},
		EncoderLayerScalars: nil,
	}

	m.withEncoderScalars(func() {
		if m.Layers[0].LayerScalar != decoder0 {
			t.Fatal("mismatched scalar set must run the callback unswapped")
		}
	})
}

func TestInt32SlicesEqual_Good(t *testing.T) {
	if !int32SlicesEqual([]int32{1, 2}, []int32{1, 2}) {
		t.Fatal("equal slices reported unequal")
	}
	if int32SlicesEqual([]int32{1, 2}, []int32{1, 3}) || int32SlicesEqual([]int32{1}, []int32{1, 2}) {
		t.Fatal("unequal slices reported equal")
	}
}

func TestTokenInSet_Good(t *testing.T) {
	if !tokenInSet(106, []int32{1, 106}) {
		t.Fatal("member not found")
	}
	if tokenInSet(7, []int32{1, 106}) || tokenInSet(7, nil) {
		t.Fatal("non-member reported found")
	}
}
