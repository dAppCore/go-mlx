// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Runnable usage-in-situ examples for the Gemma 4 audio lane, built on the
// synthetic Conformer fixture (audioTestWeights / buildAudioTestEncoder) — no
// real multi-GB checkpoint, no live model load. They are tagged metal_runtime
// because an Example cannot skip (it has no *testing.T), so it must compile out
// of an untagged `go test` rather than run Metal ops where the runtime may be
// unavailable; under `-tags metal_runtime` they run and document the contract.
//
// Each Example prints structural facts (shapes / counts) rather than floats —
// the maths is deterministic, but exact float formatting is not a stable golden.

// ExampleGemma4AudioEncoder_Forward shows the audio-tower contract: a log-mel
// clip [1, frames, melBins] runs through the strided subsampler + Conformer
// layers + output projection, emerging as soft-token rows
// [1, ceil(frames/4), OutputProjDims]. 19 mel frames pool 19 -> 10 -> 5 across
// the two stride-2 pad-1 convs, so the tower yields 5 soft tokens.
func ExampleGemma4AudioEncoder_Forward() {
	// The synthetic tower is the test fixture's Conformer (2 layers, 2 heads,
	// chunked attention) — the same geometry buildGemma4AudioEncoder produces
	// from a checkpoint, just at toy dimensions.
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), audioExampleWeights())
	if err != nil || enc == nil {
		core.Println("encoder unavailable")
		return
	}
	defer closeGemma4AudioEncoder(enc)

	mel := metal.Zeros([]int32{1, 19, audioTestMelBins}, metal.DTypeFloat32)
	defer metal.Free(mel)
	soft := enc.Forward(mel)
	defer metal.Free(soft)
	_ = metal.Eval(soft)

	shape := soft.Shape()
	core.Println(core.Sprintf("soft tokens %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: soft tokens 1x5x24
}

// ExampleGemma4AudioEncoder_SoftTokens shows the placeholder-count contract the
// processor uses: SoftTokens reports how many AudioTokenID placeholders one clip
// needs (two ceil-halvings of the mel frame count), so the splice has exactly one
// soft-token row per placeholder. 19 -> 10 -> 5 matches the encoder's row count.
func ExampleGemma4AudioEncoder_SoftTokens() {
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), audioExampleWeights())
	if err != nil || enc == nil {
		core.Println("encoder unavailable")
		return
	}
	defer closeGemma4AudioEncoder(enc)

	core.Println(core.Sprintf("placeholders for 19 frames = %d", enc.SoftTokens(19)))
	// Output: placeholders for 19 frames = 5
}

// ExampleGemma4Model_encodeGemma4Audio shows the full audio lane in situ: a
// log-mel clip flows through the Conformer tower (encodeGemma4AudioMel) and then
// the audio projector (embed_audio.embedding_projection), landing as feature rows
// in the decoder embedding width. This is the encode half of the audio branch of
// ForwardUnifiedMultiModal — one clip of 24 mel frames -> 6 soft tokens ->
// projected to the synthetic text hidden width.
func ExampleGemma4Model_encodeGemma4Audio() {
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), audioExampleWeights())
	if err != nil || enc == nil {
		core.Println("encoder unavailable")
		return
	}
	proj := &Gemma4AudioProjector{
		Projection: synthVisionLinear("embed_audio.embedding_projection", gemma4VisionTextHid, audioTestProj),
		Eps:        1e-6,
	}
	model := &Gemma4Model{AudioEncoder: enc, AudioProjector: proj}
	defer func() {
		closeGemma4AudioEncoder(enc)
		closeGemma4AudioProjector(proj)
	}()

	mel := metal.Zeros([]int32{24, audioTestMelBins}, metal.DTypeFloat32) // 2-D clip
	defer metal.Free(mel)
	features := model.encodeGemma4Audio([]*metal.Array{mel})
	defer metal.Free(features)
	_ = metal.Eval(features)

	shape := features.Shape()
	core.Println(core.Sprintf("audio features %dx%d", shape[0], shape[1]))
	// Output: audio features 6x8
}
