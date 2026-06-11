// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// The splice path (#1839): mel input_features → Conformer tower →
// embed_audio projector → soft-token rows replacing AudioTokenID
// placeholder embeddings. These tests drive the real encodeGemma4Audio /
// injectGemma4TokenFeatures seams on the synthetic tower.

func audioSpliceTestModel(t *testing.T) *Gemma4Model {
	t.Helper()
	enc := buildAudioTestEncoder(t)
	t.Cleanup(func() { closeGemma4AudioEncoder(enc) })
	projWeight := audioTestArray(t, 77, audioTestProj, audioTestProj)
	projector := &Gemma4AudioProjector{Projection: metal.NewLinear(projWeight, nil), Eps: 1e-6}
	t.Cleanup(func() { closeGemma4AudioProjector(projector) })
	return &Gemma4Model{AudioEncoder: enc, AudioProjector: projector}
}

func TestGemma4_AudioSplice_MelRouting_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := audioSpliceTestModel(t)

	const frames = 19
	mel := audioTestArray(t, 3, frames, audioTestMelBins) // 2-D clip
	defer metal.Free(mel)

	rows := m.encodeGemma4Audio([]*metal.Array{mel})
	if rows == nil || !rows.Valid() {
		t.Fatal("encodeGemma4Audio returned nil for valid mel input")
	}
	defer metal.Free(rows)
	if err := metal.Eval(rows); err != nil {
		t.Fatalf("encode eval: %v", err)
	}

	wantTokens := m.AudioEncoder.SoftTokens(frames)
	if rows.NumDims() != 2 || rows.Dim(0) != wantTokens || rows.Dim(1) != audioTestProj {
		t.Fatalf("soft-token rows = %dD %d×%d, want 2D %d×%d",
			rows.NumDims(), rows.Dim(0), rows.Dim(1), wantTokens, audioTestProj)
	}
}

func TestGemma4_AudioSplice_WrongMelWidth_Bad(t *testing.T) {
	requireMetalRuntime(t)
	m := audioSpliceTestModel(t)

	wrong := audioTestArray(t, 4, 19, audioTestMelBins*2)
	defer metal.Free(wrong)
	if rows := m.encodeGemma4Audio([]*metal.Array{wrong}); rows != nil {
		metal.Free(rows)
		t.Fatal("encoder accepted mel input of the wrong width")
	}
}

func TestGemma4_AudioSplice_InjectsAtPlaceholders_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := audioSpliceTestModel(t)

	const frames = 8 // SoftTokens(8) = 2 placeholder rows
	mel := audioTestArray(t, 5, frames, audioTestMelBins)
	defer metal.Free(mel)
	rows := m.encodeGemma4Audio([]*metal.Array{mel})
	if rows == nil {
		t.Fatal("encode returned nil")
	}
	defer metal.Free(rows)
	if err := metal.Eval(rows); err != nil {
		t.Fatalf("encode eval: %v", err)
	}
	wantRows := rows.Floats()
	softTokens := rows.Dim(0)
	if softTokens != 2 {
		t.Fatalf("soft tokens = %d, want 2", softTokens)
	}

	// Sequence of 5: [text, audio, audio, text, text].
	const audioID = int32(777)
	tokenIDs := []int32{11, audioID, audioID, 12, 13}
	h := audioTestArray(t, 6, 1, len(tokenIDs), audioTestProj)
	before := append([]float32(nil), h.Floats()...)

	spliced := m.injectGemma4TokenFeatures(h, tokenIDs, []int32{1, int32(len(tokenIDs))}, rows, audioID, "audio")
	defer metal.Free(spliced)
	if err := metal.Eval(spliced); err != nil {
		t.Fatalf("splice eval: %v", err)
	}
	after := spliced.Floats()

	for pos := range tokenIDs {
		rowStart := pos * audioTestProj
		for d := 0; d < audioTestProj; d++ {
			got := after[rowStart+d]
			if tokenIDs[pos] == audioID {
				slot := 0
				if pos == 2 {
					slot = 1
				}
				if want := wantRows[slot*audioTestProj+d]; got != want {
					t.Fatalf("position %d dim %d = %v, want spliced soft token %v", pos, d, got, want)
				}
			} else if got != before[rowStart+d] {
				t.Fatalf("position %d dim %d changed (%v → %v) — splice touched a text embedding", pos, d, before[rowStart+d], got)
			}
		}
	}
}

func TestGemma4_AudioInputFeatures_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := audioSpliceTestModel(t)
	cfg := audioFeatureTestConfig()
	cfg.FeatureSize = audioTestMelBins // synthetic tower eats 8 mel bins
	extractor, err := NewGemma4AudioFeatureExtractor(cfg)
	if err != nil {
		t.Fatalf("NewGemma4AudioFeatureExtractor: %v", err)
	}
	m.AudioFeatures = extractor

	samples := make([]float32, 1600)
	for i := range samples {
		samples[i] = float32(i%7) * 0.01
	}
	mel, softTokens, err := m.AudioInputFeatures(samples)
	if err != nil {
		t.Fatalf("AudioInputFeatures: %v", err)
	}
	defer metal.Free(mel)
	// 1600 samples → 1664 padded → 10 mel frames → ceil-halved twice = 3.
	if mel.NumDims() != 3 || mel.Dim(0) != 1 || mel.Dim(1) != 10 || mel.Dim(2) != audioTestMelBins {
		t.Fatalf("mel shape = %d×%d×%d, want 1×10×%d", mel.Dim(0), mel.Dim(1), mel.Dim(2), audioTestMelBins)
	}
	if softTokens != 3 {
		t.Fatalf("soft tokens = %d, want 3", softTokens)
	}

	// The returned mel must round-trip the splice path end to end.
	rows := m.encodeGemma4Audio([]*metal.Array{mel})
	if rows == nil {
		t.Fatal("waveform mel did not encode")
	}
	defer metal.Free(rows)
	if rows.Dim(0) != softTokens {
		t.Fatalf("encoded rows = %d, want the reported %d soft tokens", rows.Dim(0), softTokens)
	}
}

func TestGemma4_AudioInputFeatures_Bad(t *testing.T) {
	var nilModel *Gemma4Model
	if _, _, err := nilModel.AudioInputFeatures([]float32{0}); err == nil {
		t.Fatal("nil model produced features")
	}
	if _, _, err := (&Gemma4Model{}).AudioInputFeatures([]float32{0}); err == nil {
		t.Fatal("encoder-free model produced features")
	}
	m := &Gemma4Model{AudioEncoder: &Gemma4AudioEncoder{}}
	if _, _, err := m.AudioInputFeatures([]float32{0}); err == nil {
		t.Fatal("extractor-free model produced features")
	}
}
