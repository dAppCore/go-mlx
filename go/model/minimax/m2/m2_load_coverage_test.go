// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/safetensors"
)

// miniMaxM2IndexFromFileForCover builds a real safetensors index (refs carry
// file paths + byte offsets, unlike the synthetic miniMaxM2Index) so the
// loadPackedProjection read legs exercise genuine ReadRefRaw/ReadRefValues IO.
func miniMaxM2IndexFromFileForCover(t *testing.T, path string) (safetensors.Index, error) {
	t.Helper()
	return safetensors.IndexFiles([]string{path})
}

// These tests close the remaining statement-coverage gaps in m2_load.go. They
// are device-free: every assertion exercises a metadata/error leg that returns
// before any kernel runs (AX-11). Fixtures reuse the package-scoped helpers in
// m2_test.go (writeMiniMaxM2RawSafetensors, miniMaxM2Index, etc.).

// --- LoadLazyExpertsForHidden error legs (post-router) ------------------

func TestM2LoadCover_LoadLazyExpertsForHidden_RouteTokensError(t *testing.T) {
	// The router loads and scores fine, but the plan's top-k exceeds the local
	// expert pool only at routing time when fed scores whose width matches the
	// router yet the config rejects: build a plan whose config has a valid
	// router shape but a routing contract that RouteTokens rejects. Here the
	// router gate is 3x2 (3 experts) while the plan config claims 0 local
	// experts for routing — RouteTokens fails after the router loads.
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	// Corrupt only the routing top-k after the plan/fixtures are built so the
	// router still loads (gate shape [3,2] matches the unchanged 3 experts) and
	// scores fine, but RouteTokens rejects a top-k that now exceeds the pool.
	routePlan := plan
	routePlan.Config.NumExpertsPerToken = routePlan.Config.NumLocalExperts + 1
	if _, err := LoadLazyExpertsForHidden(routePlan, []string{weights}, 0, [][]float32{{1, 0}}, nil, nil); err == nil || !core.Contains(err.Error(), "top-k exceeds expert count") {
		t.Fatalf("error = %v, want RouteTokens top-k diagnostic", err)
	}
}

func TestM2LoadCover_LoadLazyExpertsForHidden_LoadExpertsError(t *testing.T) {
	// Router + routing succeed, but the routed expert's packed projection
	// tensors are absent from the safetensors file, so the expert load fails.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	// Only the router gate is present; expert 2 (the routed pick for hidden
	// {1,0} under this fixture's gate) has no projection tensors.
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			0, 0,
			-1, 0,
			3, 0,
		}, 3, 2),
	})
	if _, err := LoadLazyExpertsForHidden(plan, []string{weights}, 0, [][]float32{{1, 0}}, nil, nil); err == nil || !core.Contains(err.Error(), "gate_proj") {
		t.Fatalf("error = %v, want expert gate_proj load failure", err)
	}
}

// --- LoadPackedExperts index + projection error legs --------------------

func TestM2LoadCover_LoadPackedExperts_IndexError(t *testing.T) {
	// A weight-file path that does not exist makes safetensors.IndexFiles fail,
	// surfacing the index error leg distinct from the empty-files leg.
	plan := miniMaxM2SmallJANGTQPlan(t)
	if _, err := LoadPackedExperts(plan, []string{"/no/such/dir/model.safetensors"}, 0, []int{0}); err == nil || !core.Contains(err.Error(), "index safetensors") {
		t.Fatalf("error = %v, want index safetensors diagnostic", err)
	}
}

func TestM2LoadCover_LoadPackedExperts_UpDownProjErrors(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	prefix := "model.layers.0.block_sparse_moe.experts.0"
	gate := miniMaxM2PackedRawTensor(t, prefix+".gate_proj.weight", []uint8{0, 1, 2, 3})

	// up_proj weight tensor absent → the up projection load fails (down present
	// but unreached). Gate fully resolves first.
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		gate,
		miniMaxM2F32RawTensor(gate.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(gate.Name+".biases", []float32{0}),
	})
	if _, err := LoadPackedExperts(plan, []string{weights}, 0, []int{0}); err == nil || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want up_proj load failure", err)
	}

	// up_proj present + valid, down_proj weight absent → down projection fails.
	up := miniMaxM2PackedRawTensor(t, prefix+".up_proj.weight", []uint8{0, 1, 2, 3})
	weights2 := core.PathJoin(dir, "model2.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights2, []miniMaxM2RawSafetensor{
		gate,
		miniMaxM2F32RawTensor(gate.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(gate.Name+".biases", []float32{0}),
		up,
		miniMaxM2F32RawTensor(up.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(up.Name+".biases", []float32{0}),
	})
	if _, err := LoadPackedExperts(plan, []string{weights2}, 0, []int{0}); err == nil || !core.Contains(err.Error(), "down_proj") {
		t.Fatalf("error = %v, want down_proj load failure", err)
	}
}

// --- DequantizedExperts up/down error legs ------------------------------

func TestM2LoadCover_DequantizedExperts_UpDownErrors(t *testing.T) {
	// A valid gate projection but a malformed up projection: the dequant of the
	// up projection fails, exercising the up_proj error leg (down unreached).
	goodGate := miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3})
	badProj := JANGPackedProjectionTensor{
		Descriptor: jang.PackedTensorDescriptor{
			Name: "experts.0.up_proj.weight", Shape: []uint64{2, 2}, Elements: 4,
			Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1,
		},
	}
	upBad := LazyExpertLoad{Experts: map[int]PackedExpertWeights{
		0: {GateProj: goodGate, UpProj: badProj},
	}}
	if _, err := upBad.DequantizedExperts(); err == nil || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want up_proj dequant failure", err)
	}

	// Valid gate + up, malformed down projection: the down_proj error leg.
	goodUp := miniMaxM2PackedProjectionFixture(t, "up_proj", []uint8{0, 1, 2, 3})
	badDown := badProj
	badDown.Descriptor.Name = "experts.0.down_proj.weight"
	downBad := LazyExpertLoad{Experts: map[int]PackedExpertWeights{
		0: {GateProj: goodGate, UpProj: goodUp, DownProj: badDown},
	}}
	if _, err := downBad.DequantizedExperts(); err == nil || !core.Contains(err.Error(), "down_proj") {
		t.Fatalf("error = %v, want down_proj dequant failure", err)
	}
}

// --- LoadRouter index / gate / read error legs --------------------------

func TestM2LoadCover_LoadRouter_IndexError(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	if _, err := LoadRouter(plan, []string{"/no/such/dir/model.safetensors"}, 0); err == nil || !core.Contains(err.Error(), "index safetensors") {
		t.Fatalf("error = %v, want index safetensors diagnostic", err)
	}
}

func TestM2LoadCover_LoadRouter_MissingGate(t *testing.T) {
	// A valid safetensors file that does not contain the router gate tensor:
	// LoadRouter reports the missing gate rather than reading a stray tensor.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("some.unrelated.tensor", []float32{1, 2}),
	})
	if _, err := LoadRouter(plan, []string{weights}, 0); err == nil || !core.Contains(err.Error(), "missing gate tensor") {
		t.Fatalf("error = %v, want missing gate diagnostic", err)
	}
}

func TestM2LoadCover_LoadRouter_GateReadError(t *testing.T) {
	// The router gate is declared in the header with a shape/dtype whose byte
	// span exceeds the file payload, so reading its values fails after the
	// gate is found.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	// Declare a 3x2 f32 gate (24 bytes) but supply no payload bytes for it.
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		{Name: "model.layers.0.block_sparse_moe.gate.weight", DType: "F32", Shape: []int{3, 2}, Raw: nil},
	})
	if _, err := LoadRouter(plan, []string{weights}, 0); err == nil {
		t.Fatalf("error = nil, want gate read failure for truncated payload")
	}
}

func TestM2LoadCover_LoadRouter_BiasLegs(t *testing.T) {
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 2,
		UseRoutingBias: true,
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	// Bias tensor present but the wrong length (2 vs 3 experts): the
	// bias-length mismatch leg fires after the bias is read.
	dir := t.TempDir()
	wrongLen := core.PathJoin(dir, "wronglen.safetensors")
	writeMiniMaxM2RawSafetensors(t, wrongLen, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0, 0, 1, 1, 1}, 3, 2),
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.5}, 2),
	})
	if _, err := LoadRouter(plan, []string{wrongLen}, 0); err == nil || !core.Contains(err.Error(), "bias length") {
		t.Fatalf("error = %v, want router bias length diagnostic", err)
	}

	// UseRoutingBias set but no correction-bias tensor at all: the
	// missing-correction-bias leg fires.
	missing := core.PathJoin(dir, "nobias.safetensors")
	writeMiniMaxM2RawSafetensors(t, missing, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0, 0, 1, 1, 1}, 3, 2),
	})
	if _, err := LoadRouter(plan, []string{missing}, 0); err == nil || !core.Contains(err.Error(), "missing correction bias") {
		t.Fatalf("error = %v, want missing correction bias diagnostic", err)
	}
}

func TestM2LoadCover_LoadRouter_BiasReadError(t *testing.T) {
	// The correction bias tensor is declared but its payload span is truncated,
	// so reading the bias values fails after the gate loads cleanly.
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 2,
		UseRoutingBias: true,
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	// Gate has a full payload; the bias declares 3 f32 (12 bytes) but is the
	// last tensor with no bytes behind it.
	gate := miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0, 0, 1, 1, 1}, 3, 2)
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		gate,
		{Name: "model.layers.0.block_sparse_moe.e_score_correction_bias", DType: "F32", Shape: []int{3}, Raw: nil},
	})
	if _, err := LoadRouter(plan, []string{weights}, 0); err == nil {
		t.Fatalf("error = nil, want correction-bias read failure for truncated payload")
	}
}

// --- BuildLayerForwardSkeleton error legs -------------------------------

func TestM2LoadCover_BuildLayerForwardSkeleton_SpecError(t *testing.T) {
	// An out-of-range layer makes LayerTensorSpecs fail before any IO, even
	// with a valid weight file present.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	if _, err := BuildLayerForwardSkeleton(plan, []string{weights}, 99); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want layer out of range", err)
	}
}

func TestM2LoadCover_BuildLayerForwardSkeleton_IndexError(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	if _, err := BuildLayerForwardSkeleton(plan, []string{"/no/such/dir/model.safetensors"}, 0); err == nil || !core.Contains(err.Error(), "index safetensors") {
		t.Fatalf("error = %v, want index safetensors diagnostic", err)
	}
}

func TestM2LoadCover_BuildLayerForwardSkeleton_MissingRouterGate(t *testing.T) {
	// All four attention projections resolve, but the router gate tensor is
	// absent, so the router-gate resolve leg fails.
	plan := miniMaxM2SmallJANGTQPlan(t)
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	var tensors []miniMaxM2RawSafetensor
	for _, role := range []TensorRole{TensorRoleAttentionQ, TensorRoleAttentionK, TensorRoleAttentionV, TensorRoleAttentionO} {
		spec := findMiniMaxM2Spec(specs, role)
		tensors = append(tensors, miniMaxM2RawSafetensor{
			Name: spec.Name, DType: "U8", Shape: []int{spec.Packed.PackedBytes}, Raw: make([]byte, spec.Packed.PackedBytes),
		})
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, tensors)
	if _, err := BuildLayerForwardSkeleton(plan, []string{weights}, 0); err == nil || !core.Contains(err.Error(), "missing tensor") {
		t.Fatalf("error = %v, want missing router gate diagnostic", err)
	}
}

func TestM2LoadCover_BuildLayerForwardSkeleton_MissingRouterBias(t *testing.T) {
	// UseRoutingBias is set; attention + router gate resolve but the bias
	// tensor is absent, so the router-bias resolve leg fails.
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 4, IntermediateSize: 4,
		NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 2,
		UseRoutingBias: true,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 4, BitsDefault: 2, AttentionBits: 8, RoutedExpertBits: 2})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	// miniMaxM2SkeletonRawTensors with UseRoutingBias adds the bias; drop it by
	// re-deriving the attention + gate tensors only.
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	var tensors []miniMaxM2RawSafetensor
	for _, role := range []TensorRole{TensorRoleAttentionQ, TensorRoleAttentionK, TensorRoleAttentionV, TensorRoleAttentionO} {
		spec := findMiniMaxM2Spec(specs, role)
		tensors = append(tensors, miniMaxM2RawSafetensor{
			Name: spec.Name, DType: "U8", Shape: []int{spec.Packed.PackedBytes}, Raw: make([]byte, spec.Packed.PackedBytes),
		})
	}
	tensors = append(tensors, miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", make([]float32, 12), 3, 4))
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, tensors)
	if _, err := BuildLayerForwardSkeleton(plan, []string{weights}, 0); err == nil || !core.Contains(err.Error(), "missing tensor") {
		t.Fatalf("error = %v, want missing router bias diagnostic", err)
	}
}

// --- loadPackedProjection guard legs (white-box) ------------------------

func TestM2LoadCover_loadPackedProjection_NilDescriptor(t *testing.T) {
	// A spec with no packed descriptor is rejected before any index lookup.
	if _, err := loadPackedProjection(miniMaxM2Index(), &TensorSpec{Name: "experts.0.gate_proj.weight"}); err == nil || !core.Contains(err.Error(), "missing descriptor") {
		t.Fatalf("error = %v, want missing descriptor diagnostic", err)
	}
}

func TestM2LoadCover_loadPackedProjection_MissingWeight(t *testing.T) {
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	if _, err := loadPackedProjection(miniMaxM2Index("unrelated"), spec); err == nil || !core.Contains(err.Error(), "missing weight tensor") {
		t.Fatalf("error = %v, want missing weight diagnostic", err)
	}
}

func TestM2LoadCover_loadPackedProjection_WrongDType(t *testing.T) {
	// The weight tensor is present but declared as a float dtype, not U8 —
	// the packed-dtype guard rejects it.
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight", []float32{1}),
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "is not U8") {
		t.Fatalf("error = %v, want non-U8 dtype diagnostic", err)
	}
}

func TestM2LoadCover_loadPackedProjection_RawReadError(t *testing.T) {
	// Weight tensor is U8 and found with a non-zero declared byte span, but the
	// file on disk is shrunk after indexing so the raw read of the packed
	// payload is truncated.
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	// The packed weight is the only tensor, so its payload sits at the file
	// tail; loadPackedProjection reads it (ReadRefRaw) before any sidecar.
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{packed})
	// Index reflects the full (valid) header + offsets (declared ByteLen = 1).
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	// Drop the trailing payload byte so the declared 1-byte span overshoots the
	// file: ReadRefRaw's ReadAt returns 0 of 1 → truncated-payload error.
	read := core.ReadFile(weights)
	if !read.OK {
		t.Fatalf("read back error = %v", read.Value)
	}
	full := read.Value.([]byte)
	if result := core.WriteFile(weights, full[:len(full)-1], 0o644); !result.OK {
		t.Fatalf("shrink error = %v", result.Value)
	}
	if _, err := loadPackedProjection(index, spec); err == nil {
		t.Fatalf("error = nil, want packed raw read failure for truncated payload")
	}
}

func TestM2LoadCover_loadPackedProjection_ScalesReadError(t *testing.T) {
	// Weight + scales tensors are found, but the scales payload is truncated,
	// so the scales-values read fails (the read leg after the sidecar resolves).
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		packed,
		{Name: "experts.0.gate_proj.weight.scales", DType: "F32", Shape: []int{1}, Raw: nil},
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "read scales") {
		t.Fatalf("error = %v, want scales read failure", err)
	}
}

func TestM2LoadCover_loadPackedProjection_MissingBiases(t *testing.T) {
	// Weight + scales present and valid, but the biases sidecar is absent.
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		packed,
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.scales", []float32{1}),
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "missing biases") {
		t.Fatalf("error = %v, want missing biases diagnostic", err)
	}
}

func TestM2LoadCover_loadPackedProjection_BiasesReadError(t *testing.T) {
	// Weight + scales + biases all found, but the biases payload is truncated.
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		packed,
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.scales", []float32{1}),
		{Name: "experts.0.gate_proj.weight.biases", DType: "F32", Shape: []int{1}, Raw: nil},
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "read biases") {
		t.Fatalf("error = %v, want biases read failure", err)
	}
}

func TestM2LoadCover_loadPackedProjection_ProjectionBiasReadError(t *testing.T) {
	// Weight + scales + biases valid, and a projection bias tensor is present
	// but truncated, so reading it fails (the projection-bias read leg).
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		packed,
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.scales", []float32{1}),
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.biases", []float32{0}),
		// trim(weightName)+".bias" = "experts.0.gate_proj.bias" is the first
		// projection-bias probe; declare it truncated.
		{Name: "experts.0.gate_proj.bias", DType: "F32", Shape: []int{2}, Raw: nil},
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "projection bias") {
		t.Fatalf("error = %v, want projection bias read failure", err)
	}
}

func TestM2LoadCover_loadPackedProjection_ValidateError(t *testing.T) {
	// Weight (1 packed byte) + scales (1) resolve, but the biases sidecar holds
	// two values while the descriptor's BiasCount is 1, so the final
	// ValidatePackedTensor rejects the bias-count mismatch.
	desc := jang.PackedTensorDescriptor{Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4, Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1}
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Packed: &desc}
	packed := miniMaxM2PackedRawTensor(t, "experts.0.gate_proj.weight", []uint8{0, 1, 2, 3})
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		packed,
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.scales", []float32{1}),
		miniMaxM2F32RawTensor("experts.0.gate_proj.weight.biases", []float32{0, 0}),
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := loadPackedProjection(index, spec); err == nil || !core.Contains(err.Error(), "bias count") {
		t.Fatalf("error = %v, want ValidatePackedTensor bias-count failure", err)
	}
}

// --- resolveSkeletonTensor guard legs (white-box) -----------------------

func TestM2LoadCover_resolveSkeletonTensor_EmptySpec(t *testing.T) {
	if _, err := resolveSkeletonTensor(miniMaxM2Index(), TensorSpec{}, packedWeightCandidates); err == nil || !core.Contains(err.Error(), "empty tensor spec") {
		t.Fatalf("error = %v, want empty spec diagnostic", err)
	}
}

func TestM2LoadCover_resolveSkeletonTensor_Missing(t *testing.T) {
	spec := TensorSpec{Name: "experts.0.gate_proj.weight"}
	if _, err := resolveSkeletonTensor(miniMaxM2Index("unrelated"), spec, packedWeightCandidates); err == nil || !core.Contains(err.Error(), "missing tensor") {
		t.Fatalf("error = %v, want missing tensor diagnostic", err)
	}
}

func TestM2LoadCover_resolveSkeletonTensor_PackedDTypeMismatch(t *testing.T) {
	// A packed spec resolves to a tensor whose dtype is not U8 → packed-dtype
	// guard fires.
	desc := jang.PackedTensorDescriptor{Name: "w", Shape: []uint64{2, 2}, PackedBytes: 1}
	spec := TensorSpec{Name: "w", Packed: &desc, Shape: []uint64{2, 2}}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("w", []float32{1, 2, 3, 4}, 2, 2),
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := resolveSkeletonTensor(index, spec, packedWeightCandidates); err == nil || !core.Contains(err.Error(), "not packed U8") {
		t.Fatalf("error = %v, want packed dtype diagnostic", err)
	}
}

func TestM2LoadCover_resolveSkeletonTensor_FloatDTypeMismatch(t *testing.T) {
	// A dense (non-packed) spec resolves to a U8 tensor → float-dtype guard.
	spec := TensorSpec{Name: "w", Shape: []uint64{2, 2}}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		{Name: "w", DType: "U8", Shape: []int{2, 2}, Raw: []byte{0, 0, 0, 0}},
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := resolveSkeletonTensor(index, spec, packedWeightCandidates); err == nil || !core.Contains(err.Error(), "not floating point") {
		t.Fatalf("error = %v, want float dtype diagnostic", err)
	}
}

func TestM2LoadCover_resolveSkeletonTensor_ShapeMismatch(t *testing.T) {
	// A dense spec whose logical shape disagrees with the on-disk shape.
	spec := TensorSpec{Name: "w", Shape: []uint64{3, 4}}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("w", []float32{1, 2, 3, 4}, 2, 2),
	})
	index, err := miniMaxM2IndexFromFileForCover(t, weights)
	if err != nil {
		t.Fatalf("index error = %v", err)
	}
	if _, err := resolveSkeletonTensor(index, spec, packedWeightCandidates); err == nil || !core.Contains(err.Error(), "shape") {
		t.Fatalf("error = %v, want shape mismatch diagnostic", err)
	}
}

// --- candidate-resolver fall-through legs (white-box) -------------------

func TestM2LoadCover_routerBiasCandidates_AliasLeg(t *testing.T) {
	// A spec with aliases drives the alias-append loop in routerBiasCandidates.
	spec := &TensorSpec{Name: "primary.bias", Aliases: []string{"legacy.bias", "older.bias"}}
	got := routerBiasCandidates(spec, 0)
	foundLegacy, foundOlder := false, false
	for _, c := range got {
		if c == "legacy.bias" {
			foundLegacy = true
		}
		if c == "older.bias" {
			foundOlder = true
		}
	}
	if !foundLegacy || !foundOlder {
		t.Fatalf("candidates = %+v, want both aliases appended", got)
	}
}

func TestM2LoadCover_findProjectionBiasRef_SpecNameLeg(t *testing.T) {
	// weightName fully misses every projection-bias probe, but spec.Name (which
	// differs from weightName) resolves a ".bias" — the spec.Name fall-through.
	// weightName has a ".weight" suffix so its trim probes are well-formed yet
	// absent from the index; spec.Name is a distinct base that carries .bias.
	spec := &TensorSpec{Name: "experts.0.up_proj"}
	index := miniMaxM2Index("experts.0.up_proj.bias")
	ref, name, ok := findProjectionBiasRef(index, spec, "experts.0.gate_proj.weight")
	if !ok || name != "experts.0.up_proj.bias" || ref.Name != name {
		t.Fatalf("findProjectionBiasRef() = (%+v,%q,%v), want spec.Name .bias hit", ref, name, ok)
	}
}

func TestM2LoadCover_tryProjectionBiasName_ProjBiasLeg(t *testing.T) {
	// A name with no ".weight" suffix (trim==name): trim(name)+".bias" misses,
	// then name+".proj_bias" hits — the second probe.
	ref, name, ok := tryProjectionBiasName(miniMaxM2Index("experts.0.down_proj.proj_bias"), "experts.0.down_proj")
	if !ok || name != "experts.0.down_proj.proj_bias" || ref.Name != name {
		t.Fatalf("tryProjectionBiasName() = (%+v,%q,%v), want name+.proj_bias hit", ref, name, ok)
	}
}

func TestM2LoadCover_findSidecarRef_SpecNameReturnLeg(t *testing.T) {
	// weightName and its packed-trim both miss; spec.Name (distinct, no trim
	// needed) resolves the dotted sidecar — the spec.Name return leg in
	// findSidecarRef distinct from the weightName/trim/alias legs.
	spec := &TensorSpec{Name: "gate_proj.weight"}
	index := miniMaxM2Index("gate_proj.weight.scales")
	ref, name, ok := findSidecarRef(index, spec, "missing.weight", "scales")
	if !ok || name != "gate_proj.weight.scales" || ref.Name != name {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want spec.Name dotted scales hit", ref, name, ok)
	}
}
