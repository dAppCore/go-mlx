// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Branch-level coverage for the vision/audio projector + pooler + patch-prepare
// seams whose remaining arms a loaded checkpoint cannot reach. The package-local
// test sees every unexported field, so the projector/pooler structs are built
// directly in the exact shapes the loader never produces (e.g. a projector with
// Linear1/Linear2 but no Projection, or a pooler driven with a non-grid sequence
// length). gemma4Linear with a one-entry weight map is the same synthetic-Linear
// idiom the audio loader uses, so no real metal constructor plumbing is needed.

// synthVisionLinear builds a real *metal.Linear [out, in] from a seqArray weight
// through the production gemma4Linear path — the loader's own constructor.
func synthVisionLinear(prefix string, out, in int) *metal.Linear {
	return gemma4Linear(map[string]*metal.Array{
		prefix + ".weight": seqArray(0.05, out, in),
	}, prefix, nil)
}

// TestVisionAudioBranch_MultiModalProjector_Linears_Good drives the Linear1/Linear2
// projector arm of Gemma4MultiModalProjector.Forward — the GELU-MLP projection a
// loaded checkpoint never builds (it always resolves to a single embedding_projection
// Linear or nothing). Maps a 16-wide pooled row down through hidden 12 to text 8.
func TestVisionAudioBranch_MultiModalProjector_Linears_Good(t *testing.T) {
	requireMetalRuntime(t)

	proj := &Gemma4MultiModalProjector{
		Linear1: synthVisionLinear("l1", 12, gemma4VisionHidden),
		Linear2: synthVisionLinear("l2", gemma4VisionTextHid, 12),
		Eps:     1e-6,
	}
	defer closeGemma4Vision(nil, proj)

	x := metal.Zeros([]int32{4, gemma4VisionHidden}, metal.DTypeFloat32)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("MultiModalProjector.Forward (Linear1/Linear2) returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 4 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("Linear1/Linear2 projection shape = %v, want [4 %d]", shape, gemma4VisionTextHid)
	}
	if out == x {
		t.Fatal("Linear1/Linear2 arm returned the input unchanged, want a new projected array")
	}
}

// TestVisionAudioBranch_MultiModalProjector_NilReceiver_Bad pins the nil-projector
// clone guard: a nil *Gemma4MultiModalProjector clones its input rather than
// dereferencing. The loader wires a nil projector when a checkpoint ships no
// multi_modal_projector, so encodeGemma4Images must survive calling Forward on it.
func TestVisionAudioBranch_MultiModalProjector_NilReceiver_Bad(t *testing.T) {
	requireMetalRuntime(t)

	var proj *Gemma4MultiModalProjector
	x := seqArray(0.3, 4, gemma4VisionHidden)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("nil-projector Forward returned nil, want a clone")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	if out == x {
		t.Fatal("nil-projector Forward returned the same array, want an independent clone")
	}
	got, want := out.Shape(), x.Shape()
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("clone shape = %v, want %v", got, want)
	}
}

// TestVisionAudioBranch_MultiModalProjector_NormOnlyPassthrough_Ugly drives the
// final fallthrough: a projector with neither Projection nor a Linear1/Linear2
// pair returns the RMS-normed input directly. This is the degenerate config arm
// (no projection weights at all) that the build path skips but Forward must not
// panic on.
func TestVisionAudioBranch_MultiModalProjector_NormOnlyPassthrough_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	proj := &Gemma4MultiModalProjector{Eps: 1e-6} // no Projection, no Linears
	x := seqArray(0.4, 3, gemma4VisionHidden)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("norm-only projector Forward returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	// RMSNormNoScale preserves shape; the passthrough returns that normed array.
	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != gemma4VisionHidden {
		t.Fatalf("norm-only passthrough shape = %v, want [3 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionAudioBranch_AudioProjector_Forward_Good drives the real projection arm
// of Gemma4AudioProjector.Forward: RMS-norm then the embedding_projection linear,
// mapping encoder soft-token rows into the text hidden width.
func TestVisionAudioBranch_AudioProjector_Forward_Good(t *testing.T) {
	requireMetalRuntime(t)

	proj := &Gemma4AudioProjector{
		Projection: synthVisionLinear("embed_audio.embedding_projection", gemma4VisionTextHid, audioTestProj),
		Eps:        1e-6,
	}
	defer closeGemma4AudioProjector(proj)

	x := seqArray(0.2, 5, audioTestProj)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("AudioProjector.Forward returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 5 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("audio projection shape = %v, want [5 %d]", shape, gemma4VisionTextHid)
	}
	if out == x {
		t.Fatal("AudioProjector.Forward returned the input unchanged, want a projected array")
	}
}

// TestVisionAudioBranch_AudioProjector_NilReceiver_Bad pins the nil-projector clone
// guard. buildGemma4AudioProjector returns nil whenever the projection weight is
// absent (audio.go), so the model can carry a nil AudioProjector; Forward on it
// must clone, not dereference.
func TestVisionAudioBranch_AudioProjector_NilReceiver_Bad(t *testing.T) {
	requireMetalRuntime(t)

	var proj *Gemma4AudioProjector
	x := seqArray(0.6, 3, audioTestProj)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("nil audio projector Forward returned nil, want a clone")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	if out == x {
		t.Fatal("nil audio projector Forward returned the same array, want a clone")
	}
}

// TestVisionAudioBranch_AudioProjector_NoProjection_Ugly drives the degenerate arm
// where a non-nil projector has a nil Projection linear: Forward returns the
// RMS-normed input. The build path never produces this (it returns a nil projector
// instead), so only direct construction reaches it — a fail-safe, not a live config.
func TestVisionAudioBranch_AudioProjector_NoProjection_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	proj := &Gemma4AudioProjector{Eps: 1e-6} // non-nil struct, nil Projection
	x := seqArray(0.7, 4, audioTestProj)
	out := proj.Forward(x)
	if out == nil || !out.Valid() {
		t.Fatal("no-Projection audio projector Forward returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(x, out)

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 4 || shape[1] != audioTestProj {
		t.Fatalf("no-Projection passthrough shape = %v, want [4 %d] (norm preserves width)", shape, audioTestProj)
	}
}

// TestVisionAudioBranch_Pooler_FlatGroupReshape_Good drives the middle pooler arm:
// no usable grid (gridH=gridW=0) but L divisible by k*k, so Forward reshapes into
// [B, L/(k*k), k*k, H], means the group axis, and flattens. k=2 over L=8 → 2 pooled
// rows. This is the path poolByGrid (the grid-aligned arm) deliberately bypasses.
func TestVisionAudioBranch_Pooler_FlatGroupReshape_Good(t *testing.T) {
	requireMetalRuntime(t)

	const k, B, L, H = 2, 1, 8, gemma4VisionHidden
	pooler := &Gemma4VisionPooler{
		HiddenSize:        H,
		PoolingKernelSize: k,
		EmbeddingScale:    1.0,
	}
	hidden := seqArray(0.01, B, L, H)
	// gridH=gridW=0 fails the poolByGrid guard; L%(k*k)==0 takes the group arm.
	pooled := pooler.Forward(hidden, 0, 0)
	if pooled == nil || !pooled.Valid() {
		t.Fatal("Pooler.Forward (group-reshape arm) returned nil")
	}
	if err := metal.Eval(pooled); err != nil {
		t.Fatalf("Eval pooled: %v", err)
	}
	defer metal.Free(hidden, pooled)

	shape := pooled.Shape()
	// B*(L/(k*k)) = 1*(8/4) = 2 rows, each hidden-wide.
	if len(shape) != 2 || shape[0] != 2 || shape[1] != H {
		t.Fatalf("group-reshape pooled shape = %v, want [2 %d]", shape, H)
	}
}

// TestVisionAudioBranch_Pooler_FlatFallback_Ugly drives the final pooler arm: no
// grid and no kernel pooling (k=1), so Forward flattens [B, L, H] straight to
// [B*L, H] with no grouping. This is the identity-pool fallback for towers that
// declare no pooling kernel.
func TestVisionAudioBranch_Pooler_FlatFallback_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	const B, L, H = 1, 5, gemma4VisionHidden
	pooler := &Gemma4VisionPooler{
		HiddenSize:        H,
		PoolingKernelSize: 1, // k>1 false → neither grid nor group arm
		EmbeddingScale:    1.0,
	}
	hidden := seqArray(0.02, B, L, H)
	pooled := pooler.Forward(hidden, 0, 0)
	if pooled == nil || !pooled.Valid() {
		t.Fatal("Pooler.Forward (flat fallback) returned nil")
	}
	if err := metal.Eval(pooled); err != nil {
		t.Fatalf("Eval pooled: %v", err)
	}
	defer metal.Free(hidden, pooled)

	shape := pooled.Shape()
	if len(shape) != 2 || shape[0] != B*L || shape[1] != H {
		t.Fatalf("flat-fallback pooled shape = %v, want [%d %d]", shape, B*L, H)
	}
}

// TestVisionAudioBranch_Prepare_Rank2_Good drives prepare's case-2 arm: a rank-2
// [patches, patchDim] tensor is reshaped to [1, patches, patchDim] and marked
// pre-projected (projected=false → caller runs InputProj). The grid derives from
// the patch count under the pooling kernel.
func TestVisionAudioBranch_Prepare_Rank2_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)
	embedder := model.VisionTower.PatchEmbedder

	// [16 patches, patchDim] — rank-2, the per-row patch-vector form.
	in := metal.Zeros([]int32{16, gemma4VisionPatchDim}, metal.DTypeFloat32)
	patches, projected, gridH, gridW := embedder.prepare(in)
	if patches == nil || !patches.Valid() {
		t.Fatal("prepare(rank-2) returned nil patches")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(in, patches)

	if projected {
		t.Fatal("rank-2 prepare marked projected=true, want false (caller runs InputProj)")
	}
	if gridH != 4 || gridW != 4 {
		t.Fatalf("rank-2 grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 16 || shape[2] != gemma4VisionPatchDim {
		t.Fatalf("rank-2 prepared shape = %v, want [1 16 %d]", shape, gemma4VisionPatchDim)
	}
}

// TestVisionAudioBranch_Prepare_Rank3NHWC_Good drives prepare's case-3 raw-NHWC
// arm: a rank-3 [H, W, channels] image is expanded to [1, H, W, C] and run through
// the conv subsampler (prepareRawNHWC). 32/patch 8 → a 4x4 grid → [1, 16, hidden].
func TestVisionAudioBranch_Prepare_Rank3NHWC_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)
	embedder := model.VisionTower.PatchEmbedder

	in := metal.Zeros([]int32{32, 32, 3}, metal.DTypeFloat32) // HWC, channels last
	patches, projected, gridH, gridW := embedder.prepare(in)
	if patches == nil || !patches.Valid() {
		t.Fatal("prepare(rank-3 NHWC) returned nil patches")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(in, patches)

	if !projected {
		t.Fatal("rank-3 NHWC prepare marked projected=false, want true (conv already projected)")
	}
	if gridH != 4 || gridW != 4 {
		t.Fatalf("rank-3 NHWC grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[1] != 16 || shape[2] != gemma4VisionHidden {
		t.Fatalf("rank-3 NHWC prepared shape = %v, want [1 16 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionAudioBranch_Prepare_Rank3CHW_Good drives prepare's case-3 CHW arm:
// a rank-3 [channels, H, W] image (channels first) is expanded then Transpose4'd to
// NHWC before the conv. Covers the shape[0]==channels predicate the NHWC test skips.
func TestVisionAudioBranch_Prepare_Rank3CHW_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)
	embedder := model.VisionTower.PatchEmbedder

	in := metal.Zeros([]int32{3, 32, 32}, metal.DTypeFloat32) // CHW, channels first
	patches, projected, gridH, gridW := embedder.prepare(in)
	if patches == nil || !patches.Valid() {
		t.Fatal("prepare(rank-3 CHW) returned nil patches")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(in, patches)

	if !projected {
		t.Fatal("rank-3 CHW prepare marked projected=false, want true (conv already projected)")
	}
	if gridH != 4 || gridW != 4 {
		t.Fatalf("rank-3 CHW grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[1] != 16 || shape[2] != gemma4VisionHidden {
		t.Fatalf("rank-3 CHW prepared shape = %v, want [1 16 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionAudioBranch_Prepare_Rank4NCHW_Good drives prepare's case-4 NCHW arm:
// a rank-4 [1, channels, H, W] image is Transpose4'd to NHWC before the conv.
// Covers the shape[1]==channels predicate (pass-2 covered only shape[3]==channels).
func TestVisionAudioBranch_Prepare_Rank4NCHW_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)
	embedder := model.VisionTower.PatchEmbedder

	in := metal.Zeros([]int32{1, 3, 32, 32}, metal.DTypeFloat32) // NCHW
	patches, projected, gridH, gridW := embedder.prepare(in)
	if patches == nil || !patches.Valid() {
		t.Fatal("prepare(rank-4 NCHW) returned nil patches")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(in, patches)

	if !projected {
		t.Fatal("rank-4 NCHW prepare marked projected=false, want true (conv already projected)")
	}
	if gridH != 4 || gridW != 4 {
		t.Fatalf("rank-4 NCHW grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[1] != 16 || shape[2] != gemma4VisionHidden {
		t.Fatalf("rank-4 NCHW prepared shape = %v, want [1 16 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionAudioBranch_Prepare_NoMatch_Bad drives prepare's fall-through: a shape
// matching no channel layout (rank-3 with neither last nor first axis == channels)
// returns nil patches, which PatchEmbedder.Forward surfaces as a nil tower output.
func TestVisionAudioBranch_Prepare_NoMatch_Bad(t *testing.T) {
	model := loadGemma4VisionTestModel(t)
	embedder := model.VisionTower.PatchEmbedder

	// rank-3, no axis == 3 channels and shape[2] != patchDim → no case matches.
	in := metal.Zeros([]int32{5, 7, 9}, metal.DTypeFloat32)
	patches, projected, gridH, gridW := embedder.prepare(in)
	defer metal.Free(in)
	if patches != nil {
		metal.Free(patches)
		t.Fatal("prepare(no-match) returned non-nil patches, want nil")
	}
	if projected || gridH != 0 || gridW != 0 {
		t.Fatalf("no-match prepare = (projected=%v, grid %dx%d), want (false, 0x0)", projected, gridH, gridW)
	}
}

// buildAudioBranchModel builds a bare Gemma4Model carrying only the synthetic
// Conformer encoder + a real audio projector — the minimum encodeGemma4Audio
// reads (it touches m.AudioEncoder/m.AudioProjector, never m.Cfg). The projector
// maps the encoder's OutputProjDims rows into a distinct text width.
func buildAudioBranchModel(t *testing.T) *Gemma4Model {
	t.Helper()
	enc := buildAudioTestEncoder(t)
	proj := &Gemma4AudioProjector{
		Projection: synthVisionLinear("embed_audio.embedding_projection", gemma4VisionTextHid, audioTestProj),
		Eps:        1e-6,
	}
	m := &Gemma4Model{AudioEncoder: enc, AudioProjector: proj}
	t.Cleanup(func() {
		closeGemma4AudioEncoder(enc)
		closeGemma4AudioProjector(proj)
	})
	return m
}

// audioBranchMel returns a synthetic log-mel clip [frames, melBins] — the 2-D
// input encodeGemma4AudioMel reshapes to [1, frames, melBins] before the tower.
func audioBranchMel(frames int) *metal.Array {
	return metal.Zeros([]int32{int32(frames), audioTestMelBins}, metal.DTypeFloat32)
}

// TestVisionAudioBranch_EncodeAudio_Single_Good drives encodeGemma4Audio's single-
// clip arm end to end: one mel clip → Conformer tower → audio projector → one
// feature block. 24 mel frames pool to ceil(24/4)=6 soft-token rows, projected to
// the text width.
func TestVisionAudioBranch_EncodeAudio_Single_Good(t *testing.T) {
	model := buildAudioBranchModel(t)

	mel := audioBranchMel(24)
	out := model.encodeGemma4Audio([]*metal.Array{mel})
	if out == nil || !out.Valid() {
		t.Fatal("encodeGemma4Audio(single) returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(mel, out)

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 6 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("single-clip audio features shape = %v, want [6 %d]", shape, gemma4VisionTextHid)
	}
}

// TestVisionAudioBranch_EncodeAudio_MultiClipConcat_Good drives the len(features)>1
// concat arm: two mel clips encode + project independently, then concatenate on
// axis 0. A nil entry between them exercises the invalid-skip guard in the same
// call. 24 frames → 6 rows each → 12 concatenated rows.
func TestVisionAudioBranch_EncodeAudio_MultiClipConcat_Good(t *testing.T) {
	model := buildAudioBranchModel(t)

	a := audioBranchMel(24)
	b := audioBranchMel(24)
	out := model.encodeGemma4Audio([]*metal.Array{a, nil, b}) // nil skipped
	if out == nil || !out.Valid() {
		t.Fatal("encodeGemma4Audio(multi) returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(a, b, out)

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 12 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("multi-clip audio features shape = %v, want [12 %d]", shape, gemma4VisionTextHid)
	}
}

// TestVisionAudioBranch_EncodeAudio_AllInvalid_Bad pins the empty-result guard:
// every clip nil/invalid → no features → nil return (no projector dereference).
func TestVisionAudioBranch_EncodeAudio_AllInvalid_Bad(t *testing.T) {
	model := buildAudioBranchModel(t)

	if got := model.encodeGemma4Audio([]*metal.Array{nil, nil}); got != nil {
		metal.Free(got)
		t.Fatal("encodeGemma4Audio(all-invalid) returned non-nil, want nil")
	}
}

// TestVisionAudioBranch_EncodeAudio_NoEncoderClone_Ugly drives the no-encoder /
// no-projector arm: a model with neither tower nor projector clones each input
// clip unchanged (projected==feature → Clone), so the rows pass through as-is.
// This is the encoder-free, projector-free degenerate path.
func TestVisionAudioBranch_EncodeAudio_NoEncoderClone_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	model := &Gemma4Model{} // no AudioEncoder, no AudioProjector
	// Already-shaped soft-token rows: encode is a clone passthrough.
	feat := seqArray(0.3, 6, gemma4VisionTextHid)
	out := model.encodeGemma4Audio([]*metal.Array{feat})
	if out == nil || !out.Valid() {
		t.Fatal("encodeGemma4Audio(no encoder/projector) returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	defer metal.Free(feat, out)

	if out == feat {
		t.Fatal("no-encoder arm returned the input array, want a clone")
	}
	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 6 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("clone-passthrough shape = %v, want [6 %d]", shape, gemma4VisionTextHid)
	}
}

// ExampleGemma4MultiModalProjector_Forward_linears shows the two-linear projection
// arm: a projector with Linear1/Linear2 maps pooled vision rows through a GELU MLP
// into the text hidden width (the single-Projection arm is shown elsewhere).
func ExampleGemma4MultiModalProjector_Forward_linears() {
	proj := &Gemma4MultiModalProjector{
		Linear1: synthVisionLinear("l1", 12, gemma4VisionHidden),
		Linear2: synthVisionLinear("l2", gemma4VisionTextHid, 12),
		Eps:     1e-6,
	}
	defer closeGemma4Vision(nil, proj)

	pooled := metal.Zeros([]int32{2, gemma4VisionHidden}, metal.DTypeFloat32)
	defer metal.Free(pooled)
	out := proj.Forward(pooled)
	defer metal.Free(out)
	_ = metal.Eval(out)

	shape := out.Shape()
	core.Println(core.Sprintf("projected %dx%d", shape[0], shape[1]))
	// Output: projected 2x8
}
