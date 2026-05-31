// SPDX-Licence-Identifier: EUPL-1.2

package memory

import (
	"strings"
	"testing"

	mp "dappco.re/go/mlx/pack"
)

func hasNote(plan Plan, fragment string) bool {
	for _, note := range plan.Notes {
		if strings.Contains(note, fragment) {
			return true
		}
	}
	return false
}

func TestNewPlan_M1Class16GB_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 * GiB,
			MaxRecommendedWorkingSetSize: 14 * GiB,
		},
	})
	if plan.MachineClass != ClassApple16GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple16GB)
	}
	if plan.ContextLength != 8192 || plan.CachePolicy != KVCacheRotating || plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("plan shape = %+v", plan)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 512 {
		t.Fatalf("batch/prefill = %d/%d, want 1/512", plan.BatchSize, plan.PrefillChunkSize)
	}
	if plan.PromptCache {
		t.Fatal("PromptCache = true, want false on 16GB class")
	}
	if plan.PreferredQuantization != 4 {
		t.Fatalf("PreferredQuantization = %d, want 4", plan.PreferredQuantization)
	}
	if plan.MemoryLimitBytes == 0 || plan.CacheLimitBytes == 0 || plan.WiredLimitBytes == 0 {
		t.Fatalf("allocator limits unset: %+v", plan)
	}
}

func TestNewPlan_M3Ultra96GB_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * GiB,
			MaxRecommendedWorkingSetSize: 90 * GiB,
		},
	})
	if plan.MachineClass != ClassApple96GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple96GB)
	}
	if plan.ContextLength != 131072 || plan.CacheMode != KVCacheModePaged {
		t.Fatalf("shape = ctx:%d mode:%q", plan.ContextLength, plan.CacheMode)
	}
	if plan.BatchSize != 4 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 2 {
		t.Fatalf("shape = batch %d prefill %d slots %d", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if !plan.PromptCache || plan.PreferredQuantization != 8 {
		t.Fatalf("prompt-cache/quant = %v/%d", plan.PromptCache, plan.PreferredQuantization)
	}
}

func TestNewPlan_Gemma4SmallDefaultQuantizationPolicy_Good(t *testing.T) {
	pack := mp.ModelPack{Architecture: "gemma4_text", ContextLength: 32768, NumLayers: 34, HiddenSize: 2304, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if plan.PreferredQuantization != 6 || plan.QualityQuantization != 8 || plan.FallbackQuantization != 4 {
		t.Fatalf("quantisation policy = preferred:%d quality:%d fallback:%d, want 6/8/4", plan.PreferredQuantization, plan.QualityQuantization, plan.FallbackQuantization)
	}
	if plan.QuantizationPolicy != quantizationPolicyGemma4SmallDefault {
		t.Fatalf("QuantizationPolicy = %q, want %q", plan.QuantizationPolicy, quantizationPolicyGemma4SmallDefault)
	}
	if !hasNote(plan, "defaults to q6") || !hasNote(plan, "model quantization is below machine-class preference") {
		t.Fatalf("Notes = %+v, want Gemma 4 q6 policy plus q4 warning", plan.Notes)
	}
	if len(plan.QuantizationCandidates) != 3 {
		t.Fatalf("QuantizationCandidates = %+v, want q8/q6/q4 ladder", plan.QuantizationCandidates)
	}
	q8, ok := quantizationCandidateByBits(plan, 8)
	if !ok || q8.Role != QuantizationRoleQuality || !q8.RequiresHeadroom || q8.Selected {
		t.Fatalf("q8 candidate = %+v, want quality/headroom candidate not selected by default", q8)
	}
	q6, ok := quantizationCandidateByBits(plan, 6)
	if !ok || q6.Role != QuantizationRoleDefault || !q6.Selected {
		t.Fatalf("q6 candidate = %+v, want selected normal default", q6)
	}
	q4, ok := quantizationCandidateByBits(plan, 4)
	if !ok || q4.Role != QuantizationRoleFallback || q4.Selected {
		t.Fatalf("q4 candidate = %+v, want unselected constrained fallback", q4)
	}
}

func TestNewPlan_Gemma4SmallConstrainedQuantizationPolicy_Good(t *testing.T) {
	pack := mp.ModelPack{Architecture: "gemma4_text", ContextLength: 8192, NumLayers: 34, HiddenSize: 2304}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
		Pack:   &pack,
	})
	if plan.PreferredQuantization != 4 || plan.QualityQuantization != 0 || plan.FallbackQuantization != 4 {
		t.Fatalf("quantisation policy = preferred:%d quality:%d fallback:%d, want 4/0/4", plan.PreferredQuantization, plan.QualityQuantization, plan.FallbackQuantization)
	}
	if plan.QuantizationPolicy != quantizationPolicyGemma4SmallConstrained {
		t.Fatalf("QuantizationPolicy = %q, want %q", plan.QuantizationPolicy, quantizationPolicyGemma4SmallConstrained)
	}
	if !hasNote(plan, "constrained-memory fallback") {
		t.Fatalf("Notes = %+v, want constrained fallback note", plan.Notes)
	}
	q4, ok := quantizationCandidateByBits(plan, 4)
	if !ok || q4.Role != QuantizationRoleFallback || !q4.Selected {
		t.Fatalf("q4 candidate = %+v, want selected constrained fallback", q4)
	}
	q6, ok := quantizationCandidateByBits(plan, 6)
	if !ok || q6.Role != QuantizationRoleDefault || q6.Selected || q6.MinimumMachineClass != ClassApple64GB {
		t.Fatalf("q6 candidate = %+v, want normal default gated behind 64GB class", q6)
	}
}

func quantizationCandidateByBits(plan Plan, bits int) (QuantizationCandidate, bool) {
	for _, candidate := range plan.QuantizationCandidates {
		if candidate.Bits == bits {
			return candidate, true
		}
	}
	return QuantizationCandidate{}, false
}

func TestNewPlan_Apple64GBUsesWidePrefill_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   64 * GiB,
			MaxRecommendedWorkingSetSize: 60 * GiB,
		},
	})
	if plan.MachineClass != ClassApple64GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple64GB)
	}
	if plan.BatchSize != 2 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 1 {
		t.Fatalf("shape = batch %d prefill %d slots %d, want 2/4096/1", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if plan.CacheMode != KVCacheModePaged || !plan.PromptCache {
		t.Fatalf("cache = mode %q prompt %t, want paged prompt cache", plan.CacheMode, plan.PromptCache)
	}
}

func TestNewPlan_CapsContextToModelPack_Good(t *testing.T) {
	pack := mp.ModelPack{ContextLength: 40960, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 40960 {
		t.Fatalf("ContextLength = %d, want model cap 40960", plan.ContextLength)
	}
	if plan.ModelQuantization != 4 || plan.PreferredQuantization != 8 {
		t.Fatalf("quantization = model %d preferred %d", plan.ModelQuantization, plan.PreferredQuantization)
	}
}

func TestNewPlan_QwenMoEHints_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "qwen3_moe", ContextLength: 32768,
		NumLayers: 48, HiddenSize: 4096, QuantBits: 4,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 16 * GiB, MaxRecommendedWorkingSetSize: 13 * GiB},
		Pack:   &pack,
	})
	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q for Qwen3-MoE on 16GB", plan.CacheMode, KVCacheModeKQ8VQ4)
	}
	if !hasNote(plan, "Qwen3-MoE") || !hasNote(plan, "expert") {
		t.Fatalf("Notes = %+v", plan.Notes)
	}
}

func TestNewPlan_MiniMaxArchitectureHintsAndCaps_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 196608,
		NumLayers:     62, HiddenSize: 3072,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 32768 || plan.BatchSize != 1 {
		t.Fatalf("MiniMax shape = ctx:%d batch:%d, want 32768/1", plan.ContextLength, plan.BatchSize)
	}
	if !hasNote(plan, "MiniMax M2") {
		t.Fatalf("Notes = %+v, want MiniMax hint", plan.Notes)
	}
}

func TestNewPlan_BertEmbeddingDisablesGenerationCache_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "bert", ContextLength: 512,
		NumLayers: 12, HiddenSize: 768,
		Embedding:   &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
		WeightBytes: 420 * 1024 * 1024,
		QuantBits:   16, QuantType: "fp16", QuantFamily: "dense",
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 16 * GiB, MaxRecommendedWorkingSetSize: 13 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 512 {
		t.Fatalf("ContextLength = %d, want BERT max 512", plan.ContextLength)
	}
	if plan.CachePolicy != KVCacheDefault || plan.CacheMode != KVCacheModeDefault || plan.PromptCache {
		t.Fatalf("cache policy = %+v, want disabled generation cache", plan)
	}
	if plan.EstimatedKVCacheBytes != 0 || plan.EstimatedKVCacheModeBytes != 0 {
		t.Fatalf("KV estimates = fp:%d mode:%d, want zero for encoder", plan.EstimatedKVCacheBytes, plan.EstimatedKVCacheModeBytes)
	}
	if plan.BatchSize < 4 || !hasNote(plan, "embedding encoder") {
		t.Fatalf("plan = %+v, want embedding throughput hint", plan)
	}
}

func TestNewPlan_FallbackOnZeroMemory_Bad(t *testing.T) {
	plan := NewPlan(Input{})
	if plan.MachineClass != ClassUnknown {
		t.Fatalf("MachineClass = %q, want unknown", plan.MachineClass)
	}
	if plan.ContextLength != defaultLocalContextLength || plan.BatchSize != 1 {
		t.Fatalf("fallback = %+v", plan)
	}
}

func TestNewPlan_ModelMetadataCapsContext_Ugly(t *testing.T) {
	plan := NewPlan(Input{
		Device:    DeviceInfo{MemorySize: 24 * GiB},
		ModelInfo: &ModelInfo{ContextLength: 4096, QuantBits: 2},
	})
	if plan.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want metadata cap 4096", plan.ContextLength)
	}
	if len(plan.Notes) == 0 {
		t.Fatal("expected notes for constrained model metadata")
	}
}

func TestNewPlan_KVCacheQ8ForMiddleClass_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
	})
	if plan.CacheMode != KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, KVCacheModeQ8)
	}
	if plan.EstimatedKVCacheBytes == 0 || plan.EstimatedKVCacheModeBytes == 0 {
		t.Fatalf("KV estimates unset: %+v", plan)
	}
	if plan.EstimatedKVCacheModeBytes >= plan.EstimatedKVCacheBytes {
		t.Fatalf("mode bytes %d >= fp bytes %d", plan.EstimatedKVCacheModeBytes, plan.EstimatedKVCacheBytes)
	}
}

func TestNewPlan_TurboQuantKVCacheEstimate_ResearchMode_Good(t *testing.T) {
	const elements uint64 = 32

	got := scaleKVElements(elements, KVCacheModeTurboQuant)

	if got != 14 {
		t.Fatalf("TurboQuant bytes = %d, want 14 for 32 KV elements at 3.5 bits/element", got)
	}
}

func TestNewPlan_TurboQuantIsNeverDefault_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
	})

	if plan.CacheMode == KVCacheModeTurboQuant {
		t.Fatal("CacheMode = turboquant, want opt-in research mode only")
	}
}

func TestNewPlan_GenericMoEResidencyEnabled_Good(t *testing.T) {
	// MoE architecture without MiniMax-specific tensor plan should still get
	// generic lazy residency from the architecture profile.
	pack := mp.ModelPack{Architecture: "qwen3_moe", NumLayers: 48, HiddenSize: 4096}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if !plan.ExpertResidency.Enabled || plan.ExpertResidency.Mode != ExpertResidencyModeLazy {
		t.Fatalf("ExpertResidency = %+v, want lazy residency for MoE", plan.ExpertResidency)
	}
	if plan.ExpertResidency.EvictionPolicy != ExpertEvictionLRU {
		t.Fatalf("EvictionPolicy = %q, want LRU", plan.ExpertResidency.EvictionPolicy)
	}
}

func TestClassForBytes_BoundariesAndDefaults_Good(t *testing.T) {
	cases := []struct {
		bytes uint64
		want  Class
	}{
		{0, ClassUnknown},
		{16 * GiB, ClassApple16GB},
		{24 * GiB, ClassApple24GB},
		{32 * GiB, ClassApple32GB},
		{64 * GiB, ClassApple64GB},
		{96 * GiB, ClassApple96GB},
		{128 * GiB, ClassApple128GB},
	}
	for _, c := range cases {
		if got := ClassForBytes(c.bytes); got != c.want {
			t.Fatalf("ClassForBytes(%d) = %q, want %q", c.bytes, got, c.want)
		}
	}
}

func TestMinPositive_FavoursPositive_Good(t *testing.T) {
	if minPositive(0, 5) != 5 {
		t.Fatal("minPositive(0,5) != 5")
	}
	if minPositive(5, 0) != 5 {
		t.Fatal("minPositive(5,0) != 5")
	}
	if minPositive(3, 7) != 3 {
		t.Fatal("minPositive(3,7) != 3")
	}
	if minPositive(0, 0) != 0 {
		t.Fatal("minPositive(0,0) != 0")
	}
}

func TestPercentBytes_GuardsAgainstZero_Ugly(t *testing.T) {
	if percentBytes(0, 50) != 0 {
		t.Fatal("percentBytes(0,50) != 0")
	}
	if percentBytes(100, 25) != 25 {
		t.Fatal("percentBytes(100,25) != 25")
	}
}

func TestNormalizeKnownArchitecture_KnownAliases_Good(t *testing.T) {
	cases := map[string]string{
		"qwen3_5":            "qwen3_6",
		"qwen3.6":            "qwen3_6",
		"qwen3_5_text":       "qwen3_6",
		"qwen3_5_moe":        "qwen3_6_moe",
		"qwen2.5":            "qwen2",
		"MiniMax-M2":         "minimax_m2",
		"  bert ":            "bert",
		"bert_cross_encoder": "bert_rerank",
		"phi3":               "phi",
		"unknown-arch":       "unknown_arch",
	}
	for in, want := range cases {
		if got := normalizeKnownArchitecture(in); got != want {
			t.Fatalf("normalizeKnownArchitecture(%q) = %q, want %q", in, got, want)
		}
	}
}
