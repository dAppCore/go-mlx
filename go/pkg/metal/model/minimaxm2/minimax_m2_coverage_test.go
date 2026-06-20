// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package minimaxm2

import (
	"encoding/binary"
	"math"
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// TestMiniMaxM2_StagedStubMethods_Good executes the staged-model stub method
// bodies directly (Forward, ForwardMasked, ApplyLoRA) against a non-nil model.
// The example tests guard these with `model == nil` so the bodies never run;
// invoking them here covers the nil-returning statement arms.
func TestMiniMaxM2_StagedStubMethods_Good(t *testing.T) {
	model := &miniMaxM2StagedModel{}
	if got := model.Forward(nil, nil); got != nil {
		t.Fatalf("Forward() = %v, want nil staged stub", got)
	}
	if got := model.ForwardMasked(nil, nil, nil); got != nil {
		t.Fatalf("ForwardMasked() = %v, want nil staged stub", got)
	}
	if got := model.ApplyLoRA(metal.LoRAConfig{Rank: 2, Alpha: 4}); got != nil {
		t.Fatalf("ApplyLoRA() = %v, want nil staged stub", got)
	}
}

// TestMiniMaxM2_FindTensorRef_NotFound_Bad covers the not-found fall-through
// return of findMiniMaxM2NativeTensorRef when none of the candidates match.
func TestMiniMaxM2_FindTensorRef_NotFound_Bad(t *testing.T) {
	tensors := map[string]miniMaxM2SafetensorTensorRef{"present": {Name: "present"}}
	if _, ok := findMiniMaxM2NativeTensorRef(tensors, []string{"absent.a", "absent.b"}); ok {
		t.Fatal("findMiniMaxM2NativeTensorRef(no match) = true, want false")
	}
	if ref, ok := findMiniMaxM2NativeTensorRef(tensors, []string{"absent", "present"}); !ok || ref.Name != "present" {
		t.Fatalf("findMiniMaxM2NativeTensorRef(second candidate) = %+v,%v, want present,true", ref, ok)
	}
}

// TestMiniMaxM2_UniqueExpertIDs_Dedup_Good covers the seen-id skip arm of
// miniMaxM2NativeUniqueExpertIDs (duplicate ids collapse, output is sorted).
func TestMiniMaxM2_UniqueExpertIDs_Dedup_Good(t *testing.T) {
	got := miniMaxM2NativeUniqueExpertIDs([]int{3, 1, 3, 2, 1, 2})
	want := []int{1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("unique ids = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unique ids = %v, want %v", got, want)
		}
	}
}

// TestMiniMaxM2_PackedTensorSpec_EmptyBaseSkipped_Good drives
// miniMaxM2NativePackedTensorSpec with an empty-string alias so the
// `if base == "" { continue }` arm executes while the real name still
// contributes .packed/.qweight candidates.
func TestMiniMaxM2_PackedTensorSpec_EmptyBaseSkipped_Good(t *testing.T) {
	spec := miniMaxM2NativePackedTensorSpec("real.weight", []string{""}, "role", []uint64{2, 2}, 2)
	var sawPacked, sawEmptyDerived bool
	for _, candidate := range spec.Candidates {
		if candidate == "real.weight.packed" {
			sawPacked = true
		}
		if candidate == ".packed" || candidate == ".qweight" {
			sawEmptyDerived = true
		}
	}
	if !sawPacked {
		t.Fatalf("candidates = %v, want real.weight.packed present", spec.Candidates)
	}
	if sawEmptyDerived {
		t.Fatalf("candidates = %v, want empty alias to produce no .packed/.qweight", spec.Candidates)
	}
}

// TestMiniMaxM2_ReadJANGLoadConfig_Missing_Good covers the read-failure arm of
// readMiniMaxM2JANGLoadConfig: when jang_config.json is absent the zero config
// is returned without error.
func TestMiniMaxM2_ReadJANGLoadConfig_Missing_Good(t *testing.T) {
	dir := t.TempDir() // no jang_config.json written
	cfg := readMiniMaxM2JANGLoadConfig(dir)
	if cfg.WeightFormat != "" || cfg.Profile != "" || cfg.Quantization.GroupSize != 0 {
		t.Fatalf("readMiniMaxM2JANGLoadConfig(missing) = %+v, want zero config", cfg)
	}
}

// TestMiniMaxM2_ResolvePayloadSidecarRef_Missing_Bad covers the not-found
// return of resolvePayloadSidecarRef when no sidecar candidate resolves.
func TestMiniMaxM2_ResolvePayloadSidecarRef_Missing_Bad(t *testing.T) {
	plan := miniMaxM2NativeLoadPlan{TensorRefs: map[string]miniMaxM2SafetensorTensorRef{}}
	if _, err := plan.resolvePayloadSidecarRef("model.layers.0.experts.0.gate_proj.weight", "scales"); err == nil {
		t.Fatal("resolvePayloadSidecarRef(missing) = nil, want missing-sidecar error")
	}
}

// TestMiniMaxM2_ReadSafetensorRaw_Bad covers the raw-read guards:
// invalid byte length, missing file, and a truncated payload.
func TestMiniMaxM2_ReadSafetensorRaw_Bad(t *testing.T) {
	if _, err := readMiniMaxM2SafetensorRaw("/nonexistent", 0, -1); err == nil {
		t.Fatal("readMiniMaxM2SafetensorRaw(negative len) = nil, want invalid-length error")
	}
	if _, err := readMiniMaxM2SafetensorRaw("/nonexistent/path.safetensors", 0, 4); err == nil {
		t.Fatal("readMiniMaxM2SafetensorRaw(missing file) = nil, want open error")
	}
	dir := t.TempDir()
	path := core.JoinPath(dir, "tiny.bin")
	if result := core.WriteFile(path, []byte{1, 2, 3}, 0o644); !result.OK {
		t.Fatalf("write tiny file: %v", result.Value)
	}
	// Ask for 8 bytes starting at offset 0 of a 3-byte file → ReadAt returns EOF
	// with n < len, so the truncated guard fires.
	if _, err := readMiniMaxM2SafetensorRaw(path, 0, 8); err == nil {
		t.Fatal("readMiniMaxM2SafetensorRaw(truncated) = nil, want truncation error")
	}
}

// TestMiniMaxM2_ReadSafetensorFloat32_DTypeLenGuards_Bad walks the per-dtype
// byte-length guards (BF16, F32, F64) and the unsupported-dtype default arm of
// readMiniMaxM2SafetensorFloat32 by writing tensors whose declared element
// count disagrees with the raw byte length, plus an I-something dtype that
// passes the float-dtype gate only via a crafted ref.
func TestMiniMaxM2_ReadSafetensorFloat32_DTypeLenGuards_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "lenguards.safetensors")
	tensors := []miniMaxM2TinyTensor{
		// BF16 declares 4 elements but supplies 2 → len guard.
		{Name: "bf16short", DType: "BF16", Shape: []int64{4}, Raw: miniMaxM2BF16Bytes(1.0, 2.0)},
		// F32 declares 4 elements but supplies 2 → len guard.
		{Name: "f32short", DType: "F32", Shape: []int64{4}, Raw: miniMaxM2TinyF32Raw(1.0, 2.0)},
		// F64 declares 4 elements but supplies 2 → len guard.
		{Name: "f64short", DType: "F64", Shape: []int64{4}, Raw: miniMaxM2F64Bytes(1.0, 2.0)},
	}
	writeMiniMaxM2TinySafetensors(t, path, tensors)
	refs, err := readMiniMaxM2SafetensorHeaderRefs(path)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorHeaderRefs() error = %v", err)
	}
	for _, name := range []string{"bf16short", "f32short", "f64short"} {
		if _, err := readMiniMaxM2SafetensorFloat32(refs[name]); err == nil {
			t.Fatalf("readMiniMaxM2SafetensorFloat32(%s) = nil, want byte-length error", name)
		}
	}

	// Unsupported-but-float-gated dtype: build a ref whose DType passes
	// miniMaxM2NativeFloatDType only if listed there — it is not, so to hit the
	// switch default we forge a ref that the float gate accepts. The float gate
	// accepts F16/BF16/F32/F64 only, so the switch default is unreachable via a
	// gate-passing dtype; instead cover the raw-read-failure arm with a ref that
	// points at a missing path.
	missing := miniMaxM2SafetensorTensorRef{Name: "gone", DType: "F32", Path: "/nonexistent/x.safetensors", Elements: 1, ByteLen: 4}
	if _, err := readMiniMaxM2SafetensorFloat32(missing); err == nil {
		t.Fatal("readMiniMaxM2SafetensorFloat32(missing path) = nil, want raw-read error")
	}
}

// TestMiniMaxM2_ValidatePackedPayload_Bad walks the three reject arms of
// validateMiniMaxM2NativePackedPayload: packed length mismatch, scale-count
// mismatch, and bias-count mismatch.
func TestMiniMaxM2_ValidatePackedPayload_Bad(t *testing.T) {
	ref := miniMaxM2NativePackedTensorPayloadRef{
		Name:         "expert",
		LogicalShape: []uint64{2, 2}, // 4 elements
		PackedBytes:  1,
	}
	// Packed length mismatch (2 != 1).
	if err := validateMiniMaxM2NativePackedPayload(ref, []byte{0, 0}, []float32{0}, []float32{0}, 4); err == nil {
		t.Fatal("validatePackedPayload(packed len) = nil, want error")
	}
	// 4 elements / group 4 = 1 expected group; supply 2 scales → mismatch.
	if err := validateMiniMaxM2NativePackedPayload(ref, []byte{0}, []float32{0, 0}, []float32{0}, 4); err == nil {
		t.Fatal("validatePackedPayload(scale count) = nil, want error")
	}
	// Correct scale count, wrong bias count.
	if err := validateMiniMaxM2NativePackedPayload(ref, []byte{0}, []float32{0}, []float32{0, 0}, 4); err == nil {
		t.Fatal("validatePackedPayload(bias count) = nil, want error")
	}
	// Good case: 1 group, matching counts.
	if err := validateMiniMaxM2NativePackedPayload(ref, []byte{0}, []float32{0}, []float32{0}, 4); err != nil {
		t.Fatalf("validatePackedPayload(good) = %v, want nil", err)
	}
}

// miniMaxM2TinyF32Raw encodes float32 values into a little-endian byte slice
// (the raw payload form used by the dtype-length guard fixtures).
func miniMaxM2TinyF32Raw(values ...float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}

// writeMiniMaxM2CoverConfig writes a config.json into dir and returns the bytes.
func writeMiniMaxM2CoverConfig(t *testing.T, dir, config string) []byte {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	return []byte(config)
}

// TestMiniMaxM2_ResolveSkeletonTensor_Bad walks every reject arm of
// resolveMiniMaxM2NativeSkeletonTensor: missing tensor, packed spec with a
// non-U8 dtype, packed spec with a byte-count mismatch, float spec with a
// non-float dtype, float spec with a shape mismatch, and float spec with a
// byte-length mismatch.
func TestMiniMaxM2_ResolveSkeletonTensor_Bad(t *testing.T) {
	packedSpec := miniMaxM2NativeTensorSpec{
		Name:        "w",
		Candidates:  []string{"w"},
		Role:        "expert.gate_proj",
		Shape:       []uint64{2, 2},
		Packed:      true,
		PackedBytes: 1,
	}
	floatSpec := miniMaxM2NativeTensorSpec{
		Name:       "g",
		Candidates: []string{"g"},
		Role:       "router.gate",
		Shape:      []uint64{2, 2},
	}

	// Missing tensor.
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(map[string]miniMaxM2SafetensorTensorRef{}, packedSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(missing) = nil, want error")
	}

	// Packed spec, dtype not U8.
	notU8 := map[string]miniMaxM2SafetensorTensorRef{"w": {Name: "w", DType: "F32", Elements: 1, ByteLen: 1}}
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(notU8, packedSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(packed not U8) = nil, want error")
	}

	// Packed spec, U8 but byte count mismatches PackedBytes (want 1, supply 2).
	badPackedBytes := map[string]miniMaxM2SafetensorTensorRef{"w": {Name: "w", DType: "U8", Elements: 2, ByteLen: 2}}
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(badPackedBytes, packedSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(packed byte mismatch) = nil, want error")
	}

	// Float spec, dtype not float (U8).
	floatNotFloat := map[string]miniMaxM2SafetensorTensorRef{"g": {Name: "g", DType: "U8", Shape: []uint64{2, 2}, Elements: 4, ByteLen: 4}}
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(floatNotFloat, floatSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(float spec, U8 dtype) = nil, want error")
	}

	// Float spec, shape mismatch (want 2x2, supply 2x3).
	floatBadShape := map[string]miniMaxM2SafetensorTensorRef{"g": {Name: "g", DType: "F32", Shape: []uint64{2, 3}, Elements: 6, ByteLen: 24}}
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(floatBadShape, floatSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(float shape mismatch) = nil, want error")
	}

	// Float spec, correct shape but byte length mismatch (F32 4 elems = 16
	// bytes, supply 99).
	floatBadBytes := map[string]miniMaxM2SafetensorTensorRef{"g": {Name: "g", DType: "F32", Shape: []uint64{2, 2}, Elements: 4, ByteLen: 99}}
	if _, err := resolveMiniMaxM2NativeSkeletonTensor(floatBadBytes, floatSpec); err == nil {
		t.Fatal("resolveSkeletonTensor(float byte mismatch) = nil, want error")
	}
}

// TestMiniMaxM2_ResolveSkeletonTensor_PackedGood_Good covers the packed
// success return (resolved.PackedBytes set, byte/elem counts match).
func TestMiniMaxM2_ResolveSkeletonTensor_PackedGood_Good(t *testing.T) {
	spec := miniMaxM2NativeTensorSpec{
		Name:        "w",
		Candidates:  []string{"w"},
		Role:        "expert.gate_proj",
		Shape:       []uint64{2, 2},
		Packed:      true,
		PackedBytes: 1,
	}
	tensors := map[string]miniMaxM2SafetensorTensorRef{"w": {Name: "w", DType: "U8", Shape: []uint64{1}, Elements: 1, ByteLen: 1}}
	resolved, err := resolveMiniMaxM2NativeSkeletonTensor(tensors, spec)
	if err != nil {
		t.Fatalf("resolveSkeletonTensor(packed good) error = %v", err)
	}
	if resolved.PackedBytes != 1 || resolved.Role != "expert.gate_proj" {
		t.Fatalf("resolved = %+v, want PackedBytes 1 / role expert.gate_proj", resolved)
	}
}

// TestMiniMaxM2_ResolvePackedPayloadRef_Bad walks every reject arm of
// resolveMiniMaxM2NativePackedPayloadRef: non-packed spec, missing tensor,
// non-U8 dtype, and packed-byte mismatch.
func TestMiniMaxM2_ResolvePackedPayloadRef_Bad(t *testing.T) {
	// Non-packed spec is rejected outright.
	unpacked := miniMaxM2NativeTensorSpec{Name: "u", Candidates: []string{"u"}, Packed: false}
	if _, err := resolveMiniMaxM2NativePackedPayloadRef(map[string]miniMaxM2SafetensorTensorRef{}, unpacked); err == nil {
		t.Fatal("resolvePackedPayloadRef(non-packed spec) = nil, want error")
	}

	packed := miniMaxM2NativeTensorSpec{
		Name:        "w",
		Candidates:  []string{"w"},
		Role:        "expert.gate_proj",
		Shape:       []uint64{2, 2},
		Packed:      true,
		PackedBytes: 1,
	}
	// Missing tensor.
	if _, err := resolveMiniMaxM2NativePackedPayloadRef(map[string]miniMaxM2SafetensorTensorRef{}, packed); err == nil {
		t.Fatal("resolvePackedPayloadRef(missing) = nil, want error")
	}
	// Non-U8 dtype.
	notU8 := map[string]miniMaxM2SafetensorTensorRef{"w": {Name: "w", DType: "F32", Elements: 1, ByteLen: 1}}
	if _, err := resolveMiniMaxM2NativePackedPayloadRef(notU8, packed); err == nil {
		t.Fatal("resolvePackedPayloadRef(non-U8) = nil, want error")
	}
	// Packed-byte mismatch (want 1, supply 2).
	badBytes := map[string]miniMaxM2SafetensorTensorRef{"w": {Name: "w", DType: "U8", Elements: 2, ByteLen: 2}}
	if _, err := resolveMiniMaxM2NativePackedPayloadRef(badBytes, packed); err == nil {
		t.Fatal("resolvePackedPayloadRef(packed byte mismatch) = nil, want error")
	}
}

// TestMiniMaxM2_BuildLayerSkeleton_Bad walks the four error arms of
// buildMiniMaxM2NativeLayerSkeleton: out-of-range layer, attention-resolve
// failure (no tensors), router-gate-resolve failure, and router-bias-resolve
// failure.
func TestMiniMaxM2_BuildLayerSkeleton_Bad(t *testing.T) {
	cfg := miniMaxM2LoadConfig{
		HiddenSize:        2,
		IntermediateSize:  2,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           2,
		NumLocalExperts:   2,
		NumExpertsPerToken: 1,
	}
	jang := miniMaxM2JANGLoadConfig{}

	// Out-of-range layer (>= NumHiddenLayers and negative).
	if _, err := buildMiniMaxM2NativeLayerSkeleton(cfg, jang, map[string]miniMaxM2SafetensorTensorRef{}, 5); err == nil {
		t.Fatal("buildLayerSkeleton(out of range) = nil, want error")
	}
	if _, err := buildMiniMaxM2NativeLayerSkeleton(cfg, jang, map[string]miniMaxM2SafetensorTensorRef{}, -1); err == nil {
		t.Fatal("buildLayerSkeleton(negative) = nil, want error")
	}

	// Empty tensor map → attention resolve fails first.
	if _, err := buildMiniMaxM2NativeLayerSkeleton(cfg, jang, map[string]miniMaxM2SafetensorTensorRef{}, 0); err == nil {
		t.Fatal("buildLayerSkeleton(no attention) = nil, want error")
	}

	// Attention present (U8 packed, 4 bytes each) but router gate missing.
	attnOnly := miniMaxM2SkeletonAttentionTensors(cfg, jang, 0)
	if _, err := buildMiniMaxM2NativeLayerSkeleton(cfg, jang, attnOnly, 0); err == nil {
		t.Fatal("buildLayerSkeleton(no router gate) = nil, want error")
	}

	// Attention + router gate present, UseRoutingBias set, bias missing.
	biasCfg := cfg
	biasCfg.UseRoutingBias = true
	attnGate := miniMaxM2SkeletonAttentionTensors(biasCfg, jang, 0)
	gateSpec := miniMaxM2NativeRouterGateSpec(biasCfg, 0)
	attnGate[gateSpec.Name] = miniMaxM2SafetensorTensorRef{
		Name:     gateSpec.Name,
		DType:    "F32",
		Shape:    gateSpec.Shape,
		Elements: int64(biasCfg.NumLocalExperts * biasCfg.HiddenSize),
		ByteLen:  int64(biasCfg.NumLocalExperts * biasCfg.HiddenSize * 4),
	}
	if _, err := buildMiniMaxM2NativeLayerSkeleton(biasCfg, jang, attnGate, 0); err == nil {
		t.Fatal("buildLayerSkeleton(no router bias) = nil, want error")
	}
}

// miniMaxM2SkeletonAttentionTensors builds the four packed attention tensor
// refs the skeleton resolver expects for a layer, sized to the spec's
// PackedBytes so the resolver accepts them.
func miniMaxM2SkeletonAttentionTensors(cfg miniMaxM2LoadConfig, jang miniMaxM2JANGLoadConfig, layer int) map[string]miniMaxM2SafetensorTensorRef {
	tensors := map[string]miniMaxM2SafetensorTensorRef{}
	for _, spec := range miniMaxM2NativeAttentionSpecs(cfg, jang, layer) {
		tensors[spec.Name] = miniMaxM2SafetensorTensorRef{
			Name:     spec.Name,
			DType:    "U8",
			Shape:    []uint64{uint64(spec.PackedBytes)},
			Elements: spec.PackedBytes,
			ByteLen:  spec.PackedBytes,
		}
	}
	return tensors
}

// TestMiniMaxM2_RouteTokens_EqualScoresTieBreak_Good drives the equal-scores
// branch of the stable-sort comparator in routeMiniMaxM2NativeTokens: when two
// experts share a score the lower index ranks first.
func TestMiniMaxM2_RouteTokens_EqualScoresTieBreak_Good(t *testing.T) {
	cfg := miniMaxM2LoadConfig{NumLocalExperts: 4, NumExpertsPerToken: 2}
	// experts 1 and 3 tie at 5; the comparator's equal-score arm keeps index
	// order so the selected top-2 is deterministic.
	decisions, selected, err := routeMiniMaxM2NativeTokens(cfg, [][]float32{{5, 5, 5, 5}})
	if err != nil {
		t.Fatalf("routeMiniMaxM2NativeTokens(all-equal) error = %v", err)
	}
	if len(decisions) != 1 || len(decisions[0].ExpertIDs) != 2 {
		t.Fatalf("decisions = %+v, want one token with two experts", decisions)
	}
	// With every score equal, the tie-break keeps ascending index order: 0,1.
	if decisions[0].ExpertIDs[0] != 0 || decisions[0].ExpertIDs[1] != 1 {
		t.Fatalf("expert ids = %v, want [0 1] from index tie-break", decisions[0].ExpertIDs)
	}
	if len(selected) != 2 {
		t.Fatalf("selected = %v, want two unique experts", selected)
	}
}

// TestMiniMaxM2_RunProjection_BadShape_Bad covers the shape-conversion error
// arm of runMiniMaxM2NativeProjection (empty LogicalShape → int32-shape error).
func TestMiniMaxM2_RunProjection_BadShape_Bad(t *testing.T) {
	requireMetalRuntime(t)
	input := metal.FromValues([]float32{1, 2}, 1, 2)
	defer metal.Free(input)
	payload := miniMaxM2NativePackedProjectionPayload{
		Ref:       miniMaxM2NativePackedTensorPayloadRef{LogicalShape: nil}, // empty → error
		Packed:    []byte{0},
		Scales:    []float32{1},
		Biases:    []float32{0},
		GroupSize: 4,
		Bits:      2,
	}
	if _, err := runMiniMaxM2NativeProjection(input, payload); err == nil {
		t.Fatal("runMiniMaxM2NativeProjection(empty shape) = nil, want shape error")
	}
}

// TestMiniMaxM2_ForwardExpertPayload_GateBadShape_Bad covers the gate_proj
// error wrap of forwardMiniMaxM2NativeExpertPayload (and, transitively, the
// runMiniMaxM2NativeProjection shape arm) by giving the gate an empty logical
// shape.
func TestMiniMaxM2_ForwardExpertPayload_GateBadShape_Bad(t *testing.T) {
	requireMetalRuntime(t)
	payload := miniMaxM2NativeExpertPayload{
		ExpertID: 0,
		GateProj: miniMaxM2NativePackedProjectionPayload{
			Ref:       miniMaxM2NativePackedTensorPayloadRef{LogicalShape: nil},
			Packed:    []byte{0}, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 4, Bits: 2,
		},
	}
	if _, err := forwardMiniMaxM2NativeExpertPayload([]float32{1, 2}, payload); err == nil {
		t.Fatal("forwardExpertPayload(bad gate shape) = nil, want gate_proj error")
	}
}

// TestMiniMaxM2_ForwardExpertPayload_UpDownBadShape_Bad covers the up_proj and
// down_proj error-wrap arms of forwardMiniMaxM2NativeExpertPayload. The gate
// uses a valid 2x2 identity payload so it succeeds, then the up (resp. down)
// projection carries an empty logical shape so the matching wrap fires.
func TestMiniMaxM2_ForwardExpertPayload_UpDownBadShape_Bad(t *testing.T) {
	requireMetalRuntime(t)
	identity := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1})
	goodProj := miniMaxM2NativePackedProjectionPayload{
		Ref:       miniMaxM2NativePackedTensorPayloadRef{LogicalShape: []uint64{2, 2}},
		Packed:    identity, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 4, Bits: 2,
	}
	badProj := miniMaxM2NativePackedProjectionPayload{
		Ref:       miniMaxM2NativePackedTensorPayloadRef{LogicalShape: nil},
		Packed:    []byte{0}, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 4, Bits: 2,
	}

	// up_proj fails (gate good, up bad).
	upBad := miniMaxM2NativeExpertPayload{GateProj: goodProj, UpProj: badProj, DownProj: goodProj}
	if _, err := forwardMiniMaxM2NativeExpertPayload([]float32{1, 2}, upBad); err == nil {
		t.Fatal("forwardExpertPayload(bad up shape) = nil, want up_proj error")
	}

	// down_proj fails (gate + up good, down bad).
	downBad := miniMaxM2NativeExpertPayload{GateProj: goodProj, UpProj: goodProj, DownProj: badProj}
	if _, err := forwardMiniMaxM2NativeExpertPayload([]float32{1, 2}, downBad); err == nil {
		t.Fatal("forwardExpertPayload(bad down shape) = nil, want down_proj error")
	}
}

// TestMiniMaxM2_DispatchExperts_ForwardError_Bad covers the expert-forward
// error-wrap arm (the core.E wrap after forwardMiniMaxM2NativeExpertPayload
// fails) by handing dispatch a payload whose gate has an empty logical shape.
func TestMiniMaxM2_DispatchExperts_ForwardError_Bad(t *testing.T) {
	requireMetalRuntime(t)
	hidden := [][]float32{{1, 2}}
	decisions := []miniMaxM2NativeRouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	payloads := map[int]miniMaxM2NativeExpertPayload{
		0: {GateProj: miniMaxM2NativePackedProjectionPayload{
			Ref:    miniMaxM2NativePackedTensorPayloadRef{LogicalShape: nil},
			Packed: []byte{0}, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 4, Bits: 2,
		}},
	}
	if _, err := dispatchMiniMaxM2NativeExperts(hidden, decisions, payloads); err == nil {
		t.Fatal("dispatchExperts(forward error) = nil, want wrapped expert error")
	}
}

// TestMiniMaxM2_DispatchExperts_OutputWidthMismatch_Bad covers the
// output-width guard: a down_proj that projects to a width different from the
// input vector makes forwardMiniMaxM2NativeExpertPayload return a slice whose
// length disagrees with tokenOutput.
func TestMiniMaxM2_DispatchExperts_OutputWidthMismatch_Bad(t *testing.T) {
	requireMetalRuntime(t)
	// hidden width 2; gate/up are 2x2 (output width 2 feeds down), down is 3x2
	// so the expert output width is 3 != 2 → width-mismatch guard.
	identity := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1})
	down3x2 := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1, 0, 0}) // 3 rows x 2 cols
	gate := miniMaxM2NativePackedProjectionPayload{
		Ref:    miniMaxM2NativePackedTensorPayloadRef{LogicalShape: []uint64{2, 2}},
		Packed: identity, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 4, Bits: 2,
	}
	down := miniMaxM2NativePackedProjectionPayload{
		Ref:    miniMaxM2NativePackedTensorPayloadRef{LogicalShape: []uint64{3, 2}},
		Packed: down3x2, Scales: []float32{1}, Biases: []float32{0}, GroupSize: 6, Bits: 2,
	}
	hidden := [][]float32{{1, 2}}
	decisions := []miniMaxM2NativeRouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	payloads := map[int]miniMaxM2NativeExpertPayload{0: {GateProj: gate, UpProj: gate, DownProj: down}}
	if _, err := dispatchMiniMaxM2NativeExperts(hidden, decisions, payloads); err == nil {
		t.Fatal("dispatchExperts(width mismatch) = nil, want output-width error")
	}
}
