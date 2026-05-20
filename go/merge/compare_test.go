// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"testing"
)

func TestComparePacks_BaseFineTunedSafetensors_Good(t *testing.T) {
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

func TestComparePacks_RequiresSafetensorsPacks_Bad(t *testing.T) {
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

func TestComparePacks_ReportsShapeMismatch_Ugly(t *testing.T) {
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
