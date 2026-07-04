// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
)

// TestCompare_ComparePacks_Good is the canonical AX-7 happy-path triplet leg
// for the public ComparePacks symbol: a base pack and a fine-tuned pack that
// differ in exactly one tensor are compared, and ComparePacks reports one
// changed tensor, one unchanged tensor, and the aggregate delta statistics —
// without ever loading either model through the runtime.
func TestCompare_ComparePacks_Good(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{3}, Data: []float32{1, 2, 3}},
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 1}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{3}, Data: []float32{1, 4, 1}},
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 1}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:             testPack(base),
		FineTuned:        testPack(tuned),
		IncludeUnchanged: true,
	})
	if err != nil {
		t.Fatalf("ComparePacks() error = %v", err)
	}
	if report.ComparedTensors != 2 || report.ChangedTensors != 1 || report.UnchangedTensors != 1 {
		t.Fatalf("compare counts = %+v, want 2 compared / 1 changed / 1 unchanged", report)
	}
	if report.TensorCount != 2 || report.ElementsCompared != 5 {
		t.Fatalf("tensor/elements = %d/%d, want 2/5", report.TensorCount, report.ElementsCompared)
	}
	assertClose(t, report.MaxAbsDelta, 2)
	deltas := tensorDeltaByName(report.Tensors)
	if deltas["model.layers.0.self_attn.q_proj.weight"].Status != CompareStatusChanged {
		t.Fatalf("q_proj status = %v, want changed", deltas["model.layers.0.self_attn.q_proj.weight"].Status)
	}
	if deltas["model.norm.weight"].Status != CompareStatusUnchanged {
		t.Fatalf("norm status = %v, want unchanged", deltas["model.norm.weight"].Status)
	}
}

// TestCompare_ComparePacks_Bad is the canonical AX-7 invalid-input triplet leg:
// ComparePacks must reject a comparison whose base pack carries the wrong weight
// format (gguf instead of safetensors), returning an error and no report rather
// than attempting to index non-safetensors weights.
func TestCompare_ComparePacks_Bad(t *testing.T) {
	good := testPack(writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}},
	}))

	// Empty options: no base/fine-tuned packs at all.
	if report, err := ComparePacks(context.Background(), CompareOptions{}); err == nil {
		t.Fatalf("ComparePacks(empty) error = nil, report = %+v", report)
	}

	// Right roots and weight files, wrong format on the base pack.
	wrongFormat := good
	wrongFormat.Format = "gguf"
	if report, err := ComparePacks(context.Background(), CompareOptions{Base: wrongFormat, FineTuned: good}); err == nil {
		t.Fatalf("ComparePacks(non-safetensors base) error = nil, report = %+v", report)
	}
}

// TestCompare_ComparePacks_Ugly is the canonical AX-7 edge-case triplet leg: the
// two packs share a tensor name but disagree on its shape. ComparePacks does not
// error — it classifies the tensor as a shape mismatch, increments the
// ShapeMismatches counter, and records the delta with CompareStatusShapeMismatch
// without reading any chunk data.
func TestCompare_ComparePacks_Ugly(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{1, 2, 3}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:      testPack(base),
		FineTuned: testPack(tuned),
	})
	if err != nil {
		t.Fatalf("ComparePacks(shape mismatch) error = %v", err)
	}
	if report.ShapeMismatches != 1 || report.ComparedTensors != 0 || report.TensorCount != 1 {
		t.Fatalf("report = %+v, want exactly one shape mismatch", report)
	}
	if len(report.Tensors) != 1 || report.Tensors[0].Status != CompareStatusShapeMismatch {
		t.Fatalf("tensor deltas = %+v, want shape-mismatch status", report.Tensors)
	}
}

func TestComparePacks_BaseFineTunedSafetensorsGood(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{3}, Data: []float32{1, 2, 3}},
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 1}},
		{Name: "model.base_only.weight", Shape: []int{1}, Data: []float32{9}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{3}, Data: []float32{1, 4, 1}},
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 1}},
		{Name: "model.tuned_only.weight", Shape: []int{1}, Data: []float32{5}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:             testPack(base),
		FineTuned:        testPack(tuned),
		IncludeUnchanged: true,
		Labels:           map[string]string{"experiment": "delta"},
	})

	if err != nil {
		t.Fatalf("ComparePacks() error = %v", err)
	}
	if report.ComparedTensors != 2 || report.ChangedTensors != 1 || report.UnchangedTensors != 1 || report.MissingInFineTuned != 1 || report.ExtraInFineTuned != 1 {
		t.Fatalf("report counts = %+v", report)
	}
	if report.TensorCount != 4 || report.ElementsCompared != 5 {
		t.Fatalf("tensor/elements = %d/%d, want 4/5", report.TensorCount, report.ElementsCompared)
	}
	assertClose(t, report.MeanAbsDelta, 0.8)
	assertClose(t, report.RMSDelta, math.Sqrt(8.0/5.0))
	assertClose(t, report.MaxAbsDelta, 2)
	if report.Labels["experiment"] != "delta" {
		t.Fatalf("labels = %+v, want experiment label", report.Labels)
	}

	deltas := tensorDeltaByName(report.Tensors)
	changed := deltas["model.layers.0.self_attn.q_proj.weight"]
	if changed.Status != CompareStatusChanged || changed.Elements != 3 {
		t.Fatalf("changed delta = %+v", changed)
	}
	assertClose(t, changed.MeanAbsDelta, 4.0/3.0)
	assertClose(t, changed.RMSDelta, math.Sqrt(8.0/3.0))
	assertClose(t, changed.L2Delta, math.Sqrt(8.0))
	if deltas["model.norm.weight"].Status != CompareStatusUnchanged {
		t.Fatalf("norm delta = %+v, want unchanged", deltas["model.norm.weight"])
	}
	if deltas["model.base_only.weight"].Status != CompareStatusMissingInTuned {
		t.Fatalf("base-only delta = %+v, want missing", deltas["model.base_only.weight"])
	}
	if deltas["model.tuned_only.weight"].Status != CompareStatusExtraInTuned {
		t.Fatalf("tuned-only delta = %+v, want extra", deltas["model.tuned_only.weight"])
	}
}

func TestComparePacks_RequiresSafetensorsPacksBad(t *testing.T) {
	if _, err := ComparePacks(context.Background(), CompareOptions{}); err == nil {
		t.Fatal("ComparePacks(empty) error = nil")
	}

	pack := testPack(writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}},
	}))
	unsupported := pack
	unsupported.Format = "gguf"
	if _, err := ComparePacks(context.Background(), CompareOptions{Base: unsupported, FineTuned: pack}); err == nil {
		t.Fatal("ComparePacks(non-safetensors) error = nil")
	}
}

func TestComparePacks_ReportsShapeMismatchUgly(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{3}, Data: []float32{1, 2, 3}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:      testPack(base),
		FineTuned: testPack(tuned),
	})

	if err != nil {
		t.Fatalf("ComparePacks(shape mismatch) error = %v", err)
	}
	if report.ShapeMismatches != 1 || report.ComparedTensors != 0 || report.TensorCount != 1 {
		t.Fatalf("report = %+v, want one shape mismatch", report)
	}
	if len(report.Tensors) != 1 || report.Tensors[0].Status != CompareStatusShapeMismatch {
		t.Fatalf("tensor deltas = %+v, want shape mismatch", report.Tensors)
	}
}

// TestComparePacks_ReportsDTypeMismatch_Ugly compares an F32 base tensor
// against the same-shape tensor stored as F16 in the fine-tuned pack. The
// shapes match but the dtypes differ, so compareTensorRefs short-circuits to
// the dtype-mismatch status without reading any chunk data.
func TestComparePacks_ReportsDTypeMismatchUgly(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})
	tuned := writeF16SafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{1, 2}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:      testPack(base),
		FineTuned: testPack(tuned),
	})
	if err != nil {
		t.Fatalf("ComparePacks(dtype mismatch) error = %v", err)
	}
	if report.DTypeMismatches != 1 || report.ComparedTensors != 0 || report.TensorCount != 1 {
		t.Fatalf("report = %+v, want one dtype mismatch", report)
	}
	if len(report.Tensors) != 1 || report.Tensors[0].Status != CompareStatusDTypeMismatch {
		t.Fatalf("tensor deltas = %+v, want dtype mismatch", report.Tensors)
	}
	if report.Tensors[0].BaseDType != "F32" || report.Tensors[0].FineTunedDType != "F16" {
		t.Fatalf("delta dtypes = %q/%q, want F32/F16", report.Tensors[0].BaseDType, report.Tensors[0].FineTunedDType)
	}
}

// TestComparePacks_ZeroNormCosine_Ugly drives compareCosine's zero-norm legs
// through the public API: a base tensor that is all-zero (baseNorm == 0) gives
// cosine 0, while two all-zero tensors (both norms zero) give cosine 1.
func TestComparePacks_ZeroNormCosineUgly(t *testing.T) {
	// baseNorm == 0, tunedNorm != 0 -> cosine 0, status changed.
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{0, 0}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{3, 4}},
	})
	report, err := ComparePacks(context.Background(), CompareOptions{Base: testPack(base), FineTuned: testPack(tuned)})
	if err != nil {
		t.Fatalf("ComparePacks(one zero) error = %v", err)
	}
	if len(report.Tensors) != 1 || report.Tensors[0].Cosine != 0 {
		t.Fatalf("cosine = %v, want 0 (one zero-norm tensor)", report.Tensors)
	}

	// Both all-zero -> maxAbs 0 so status unchanged, cosine 1 (both norms zero).
	zeroBase := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{0, 0}},
	})
	zeroTuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{2}, Data: []float32{0, 0}},
	})
	report, err = ComparePacks(context.Background(), CompareOptions{
		Base:             testPack(zeroBase),
		FineTuned:        testPack(zeroTuned),
		IncludeUnchanged: true,
	})
	if err != nil {
		t.Fatalf("ComparePacks(both zero) error = %v", err)
	}
	if len(report.Tensors) != 1 || report.Tensors[0].Cosine != 1 || report.Tensors[0].Status != CompareStatusUnchanged {
		t.Fatalf("delta = %+v, want cosine 1 and unchanged", report.Tensors)
	}
}

// TestComparePacks_MaxTensorReportsCap_Good confirms MaxTensorReports caps the
// number of per-tensor deltas recorded while the aggregate counts still reflect
// every tensor — exercising appendTensorDelta's cap branch.
func TestComparePacks_MaxTensorReportsCapGood(t *testing.T) {
	base := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.weight", Shape: []int{2}, Data: []float32{1, 1}},
		{Name: "model.layers.1.weight", Shape: []int{2}, Data: []float32{1, 1}},
		{Name: "model.layers.2.weight", Shape: []int{2}, Data: []float32{1, 1}},
	})
	tuned := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.weight", Shape: []int{2}, Data: []float32{2, 2}},
		{Name: "model.layers.1.weight", Shape: []int{2}, Data: []float32{3, 3}},
		{Name: "model.layers.2.weight", Shape: []int{2}, Data: []float32{4, 4}},
	})

	report, err := ComparePacks(context.Background(), CompareOptions{
		Base:             testPack(base),
		FineTuned:        testPack(tuned),
		MaxTensorReports: 2,
	})
	if err != nil {
		t.Fatalf("ComparePacks(capped) error = %v", err)
	}
	if report.ChangedTensors != 3 || report.ComparedTensors != 3 {
		t.Fatalf("aggregate counts = %+v, want all three compared", report)
	}
	if len(report.Tensors) != 2 {
		t.Fatalf("recorded deltas = %d, want 2 (capped)", len(report.Tensors))
	}
}

// TestModelMerge_DualShapeClone_Good covers all four branches of
// dualShapeClone: both empty, base-only, tuned-only, and the shared-arena
// case where both clones carve from one backing array.
func TestModelMerge_DualShapeCloneGood(t *testing.T) {
	if b, tn := dualShapeClone(nil, nil); b != nil || tn != nil {
		t.Fatalf("dualShapeClone(nil, nil) = %v/%v, want nil/nil", b, tn)
	}

	base := []uint64{4096, 11008}
	if b, tn := dualShapeClone(base, nil); tn != nil || !sameUint64Slice(b, base) {
		t.Fatalf("dualShapeClone(base, nil) = %v/%v, want %v/nil", b, tn, base)
	}

	tuned := []uint64{2048}
	if b, tn := dualShapeClone(nil, tuned); b != nil || !sameUint64Slice(tn, tuned) {
		t.Fatalf("dualShapeClone(nil, tuned) = %v/%v, want nil/%v", b, tn, tuned)
	}

	gotBase, gotTuned := dualShapeClone(base, tuned)
	if !sameUint64Slice(gotBase, base) || !sameUint64Slice(gotTuned, tuned) {
		t.Fatalf("dualShapeClone = %v/%v, want %v/%v", gotBase, gotTuned, base, tuned)
	}
	// Clones are independent copies (cap == len so an append re-allocs and
	// cannot corrupt the neighbour) — mutating one must not touch the source.
	gotBase[0] = 0
	if base[0] != 4096 {
		t.Fatal("dualShapeClone returned an aliasing slice; mutation leaked to source")
	}
}

// TestComparePacks_RequiresWeightFiles_Bad drives validateComparePack's
// no-weight-files leg (a safetensors pack with the right root + format but an
// empty WeightFiles list) for both the base and the fine-tuned position.
func TestComparePacks_RequiresWeightFilesBad(t *testing.T) {
	good := testPack(writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}},
	}))
	noWeights := good
	noWeights.WeightFiles = nil

	if _, err := ComparePacks(context.Background(), CompareOptions{Base: noWeights, FineTuned: good}); err == nil {
		t.Fatal("ComparePacks(base without weight files) error = nil")
	}
	if _, err := ComparePacks(context.Background(), CompareOptions{Base: good, FineTuned: noWeights}); err == nil {
		t.Fatal("ComparePacks(fine-tuned without weight files) error = nil")
	}
}

// TestComparePacks_UnindexableWeights_Bad points a pack's WeightFiles at a
// path that is not a readable safetensors shard, so safetensors.IndexFiles
// fails — exercising the base-index and fine-tuned-index error legs of
// ComparePacks that the happy-path tests never reach.
func TestComparePacks_UnindexableWeightsBad(t *testing.T) {
	good := testPack(writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.norm.weight", Shape: []int{1}, Data: []float32{1}},
	}))
	missing := good
	missing.WeightFiles = []string{core.PathJoin(t.TempDir(), "absent.safetensors")}

	if _, err := ComparePacks(context.Background(), CompareOptions{Base: missing, FineTuned: good}); err == nil {
		t.Fatal("ComparePacks(unindexable base weights) error = nil")
	}
	if _, err := ComparePacks(context.Background(), CompareOptions{Base: good, FineTuned: missing}); err == nil {
		t.Fatal("ComparePacks(unindexable fine-tuned weights) error = nil")
	}
}

func tensorDeltaByName(deltas []TensorDelta) map[string]TensorDelta {
	out := make(map[string]TensorDelta, len(deltas))
	for _, delta := range deltas {
		out[delta.Name] = delta
	}
	return out
}

func assertClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("value = %.9f, want %.9f", got, want)
	}
}
