// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestProbeSummarizeLogits_Good(t *testing.T) {
	logits := FromValues([]float32{0, 1, 2, 3}, 1, 4)
	defer Free(logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}

	summary, entropy, ok, err := summarizeProbeLogits(logits, 3)
	if err != nil {
		t.Fatalf("summarizeProbeLogits: %v", err)
	}
	if !ok {
		t.Fatal("summarizeProbeLogits ok = false, want true")
	}
	if summary.VocabSize != 4 {
		t.Fatalf("VocabSize = %d, want 4", summary.VocabSize)
	}
	if summary.MaxTokenID != 3 {
		t.Fatalf("MaxTokenID = %d, want 3", summary.MaxTokenID)
	}
	if len(summary.Top) != 3 {
		t.Fatalf("Top len = %d, want 3", len(summary.Top))
	}
	if summary.Top[0].TokenID != 3 || summary.Top[1].TokenID != 2 || summary.Top[2].TokenID != 1 {
		t.Fatalf("Top logits = %+v, want token IDs [3 2 1]", summary.Top)
	}
	if summary.Meta["cpu_transfer"] != "compact_topk" {
		t.Fatalf("cpu_transfer = %q, want compact_topk", summary.Meta["cpu_transfer"])
	}
	if entropy.Value <= 0 || entropy.Value >= math.Log(4) {
		t.Fatalf("Entropy = %f, want in (0, log(4))", entropy.Value)
	}
}

func TestProbeSummarizeLogits_LastRowCompact_Good(t *testing.T) {
	logits := FromValues([]float32{
		100, 99, 98, 97,
		0, 5, 2, 7,
	}, 1, 2, 4)
	defer Free(logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}

	summary, _, ok, err := summarizeProbeLogits(logits, 2)
	if err != nil {
		t.Fatalf("summarizeProbeLogits: %v", err)
	}
	if !ok {
		t.Fatal("summarizeProbeLogits ok = false, want true")
	}
	if summary.MaxTokenID != 3 {
		t.Fatalf("MaxTokenID = %d, want 3 from final row", summary.MaxTokenID)
	}
	if len(summary.Top) != 2 {
		t.Fatalf("Top len = %d, want 2", len(summary.Top))
	}
	if summary.Top[0].TokenID != 3 || summary.Top[1].TokenID != 1 {
		t.Fatalf("Top logits = %+v, want token IDs [3 1]", summary.Top)
	}
	if summary.Meta["cpu_transfer"] != "compact_topk" {
		t.Fatalf("cpu_transfer = %q, want compact_topk", summary.Meta["cpu_transfer"])
	}
}

type probeTestCache struct {
	length int
	offset int
}

func (c probeTestCache) Update(k, v *Array, seqLen int) (*Array, *Array) { return k, v }
func (c probeTestCache) Offset() int                                     { return c.offset }
func (c probeTestCache) Len() int                                        { return c.length }
func (c probeTestCache) State() []*Array                                 { return nil }
func (c probeTestCache) Reset()                                          {}
func (c probeTestCache) Detach()                                         {}

func TestProbeCachePressure_Good(t *testing.T) {
	event := probeCachePressure(ProbePhaseDecode, 4, 2, 1, []Cache{
		probeTestCache{length: 8, offset: 12},
		probeTestCache{length: 6, offset: 12},
	})

	if event.Kind != ProbeEventCachePressure {
		t.Fatalf("Kind = %q, want %q", event.Kind, ProbeEventCachePressure)
	}
	if event.Cache.LayerCount != 2 {
		t.Fatalf("LayerCount = %d, want 2", event.Cache.LayerCount)
	}
	if event.Cache.CacheTokens != 8 {
		t.Fatalf("CacheTokens = %d, want 8", event.Cache.CacheTokens)
	}
	if event.Cache.ProcessedTokens != 12 {
		t.Fatalf("ProcessedTokens = %d, want 12", event.Cache.ProcessedTokens)
	}
}
