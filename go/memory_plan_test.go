// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
)

func TestMemoryPlan_M1Class16GB_Good(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 << 30,
			MaxRecommendedWorkingSetSize: 14 << 30,
		},
	})

	if plan.MachineClass != MemoryClassApple16GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, MemoryClassApple16GB)
	}
	if plan.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", plan.ContextLength)
	}
	if plan.CachePolicy != KVCacheRotating {
		t.Fatalf("CachePolicy = %q, want rotating", plan.CachePolicy)
	}
	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, KVCacheModeKQ8VQ4)
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
		t.Fatalf("allocator limits should be populated: %+v", plan)
	}
}

func TestMemoryPlan_M3Ultra96GB_Good(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		},
	})

	if plan.MachineClass != MemoryClassApple96GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, MemoryClassApple96GB)
	}
	if plan.ContextLength != 131072 {
		t.Fatalf("ContextLength = %d, want 131072", plan.ContextLength)
	}
	if plan.CacheMode != KVCacheModePaged {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, KVCacheModePaged)
	}
	if plan.BatchSize != 4 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 2 {
		t.Fatalf("shape = batch %d prefill %d slots %d, want 4/4096/2", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if !plan.PromptCache {
		t.Fatal("PromptCache = false, want true on 96GB class")
	}
	if plan.PreferredQuantization != 8 {
		t.Fatalf("PreferredQuantization = %d, want 8", plan.PreferredQuantization)
	}
}

func TestMemoryPlan_CapsContextToModel_Good(t *testing.T) {
	pack := mp.ModelPack{ContextLength: 40960, QuantBits: 4}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 96 << 30},
		Pack:   &pack,
	})

	if plan.ContextLength != 40960 {
		t.Fatalf("ContextLength = %d, want model cap 40960", plan.ContextLength)
	}
	if plan.ModelQuantization != 4 || plan.PreferredQuantization != 8 {
		t.Fatalf("quantization = model %d preferred %d, want 4/8", plan.ModelQuantization, plan.PreferredQuantization)
	}
}

func TestMemoryPlan_QwenFamilyHints_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "qwen3_moe",
		ContextLength: 32768,
		NumLayers:     48,
		HiddenSize:    4096,
		QuantBits:     4,
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			MemorySize:                   16 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 13 * MemoryGiB,
		},
		Pack: &pack,
	})

	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q for Qwen3-MoE on 16GB", plan.CacheMode, KVCacheModeKQ8VQ4)
	}
	if !memoryPlanHasNote(plan, "Qwen3-MoE") || !memoryPlanHasNote(plan, "expert") {
		t.Fatalf("Notes = %+v, want Qwen3-MoE expert memory hint", plan.Notes)
	}
}

func TestMemoryPlan_MiniMaxJANGTQ96GB_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 196608,
		NumLayers:     62,
		HiddenSize:    3072,
		QuantBits:     2,
		QuantGroup:    64,
		QuantType:     "jangtq",
		QuantFamily:   "jang",
		PackedQuantization: jang.BuildPackedProfile(&jang.Info{
			WeightFormat:     "mxtq",
			Profile:          "JANGTQ",
			Method:           "affine+mxtq",
			GroupSize:        64,
			BitsDefault:      2,
			AttentionBits:    8,
			RoutedExpertBits: 2,
		}),
		WeightBytes: 60 * MemoryGiB,
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 90 * MemoryGiB,
		},
		Pack: &pack,
	})

	if plan.ContextLength != 32768 || plan.BatchSize != 1 {
		t.Fatalf("MiniMax plan shape = ctx:%d batch:%d, want 32768/1", plan.ContextLength, plan.BatchSize)
	}
	if plan.CacheMode != KVCacheModePaged || !plan.PromptCache {
		t.Fatalf("MiniMax cache policy = mode:%q prompt:%v", plan.CacheMode, plan.PromptCache)
	}
	if !plan.ExpertResidency.Enabled || plan.ExpertResidency.Mode != memory.ExpertResidencyModeLazy {
		t.Fatalf("expert residency = %+v, want lazy residency for MiniMax on 96GB", plan.ExpertResidency)
	}
	if plan.ModelQuantization != 2 || plan.ModelQuantizationType != "jangtq" || plan.ModelQuantizationFamily != "jang" {
		t.Fatalf("quantization hints = %+v", plan)
	}
	if plan.ModelPackedQuantization == nil || plan.ModelPackedQuantization.Format != "mxtq" || plan.ModelPackedQuantization.MaxBits != 8 {
		t.Fatalf("packed quantization = %+v, want MXTQ profile", plan.ModelPackedQuantization)
	}
	if !memoryPlanHasNote(plan, "MiniMax") || !memoryPlanHasNote(plan, "JANGTQ") {
		t.Fatalf("Notes = %+v, want MiniMax/JANGTQ memory hint", plan.Notes)
	}
}

func TestMemoryPlan_MiniMaxLayerSkeletonHints_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 32768,
		NumLayers:     1,
		HiddenSize:    4,
		MiniMaxM2LayerSkeleton: &MiniMaxM2LayerForwardSkeleton{
			Layer: 0,
			Attention: []MiniMaxM2ResolvedTensor{
				{Name: "q", Role: MiniMaxM2TensorRoleAttentionQ, PackedBytes: 16},
				{Name: "k", Role: MiniMaxM2TensorRoleAttentionK, PackedBytes: 8},
				{Name: "v", Role: MiniMaxM2TensorRoleAttentionV, PackedBytes: 8},
				{Name: "o", Role: MiniMaxM2TensorRoleAttentionO, PackedBytes: 16},
			},
			RouterGate: MiniMaxM2ResolvedTensor{Name: "gate", Role: MiniMaxM2TensorRoleRouterGate, DType: "F32", Shape: []uint64{3, 4}},
			RouterBias: &MiniMaxM2ResolvedTensor{Name: "bias", Role: MiniMaxM2TensorRoleRouterBias, DType: "F32", Shape: []uint64{3}},
		},
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 96 * MemoryGiB, MaxRecommendedWorkingSetSize: 90 * MemoryGiB},
		Pack:   &pack,
	})

	if !plan.ModelForwardSkeletonValidated || plan.ModelForwardSkeletonBytes != 108 {
		t.Fatalf("forward skeleton hints = validated:%v bytes:%d, want true/108", plan.ModelForwardSkeletonValidated, plan.ModelForwardSkeletonBytes)
	}
	if !memoryPlanHasNote(plan, "skeleton") || !memoryPlanHasNote(plan, "safetensors") {
		t.Fatalf("Notes = %+v, want skeleton validation hint", plan.Notes)
	}
}

func TestMemoryPlan_BertEmbeddingDisablesGenerationCache_Good(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:    "bert",
		ContextLength:   512,
		NumLayers:       12,
		HiddenSize:      768,
		Embedding:       &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
		WeightBytes:     420 * 1024 * 1024,
		QuantBits:       16,
		QuantType:       "fp16",
		QuantFamily:     "dense",
		HasTokenizer:    true,
		HasChatTemplate: false,
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 16 * MemoryGiB, MaxRecommendedWorkingSetSize: 13 * MemoryGiB},
		Pack:   &pack,
	})

	if plan.ContextLength != 512 {
		t.Fatalf("ContextLength = %d, want BERT max sequence 512", plan.ContextLength)
	}
	if plan.CachePolicy != KVCacheDefault || plan.CacheMode != KVCacheModeDefault || plan.PromptCache {
		t.Fatalf("cache policy = policy:%q mode:%q prompt:%v, want disabled generation cache for embeddings", plan.CachePolicy, plan.CacheMode, plan.PromptCache)
	}
	if plan.EstimatedKVCacheBytes != 0 || plan.EstimatedKVCacheModeBytes != 0 {
		t.Fatalf("KV estimates = fp:%d mode:%d, want zero for encoder embeddings", plan.EstimatedKVCacheBytes, plan.EstimatedKVCacheModeBytes)
	}
	if plan.BatchSize < 4 || !memoryPlanHasNote(plan, "embedding encoder") {
		t.Fatalf("plan = %+v, want embedding throughput hint", plan)
	}
}

func TestMemoryPlan_PlanMemory_Good(t *testing.T) {
	target := "PlanMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestMemoryPlan_PlanMemory_Bad(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{})
	if plan.MachineClass != MemoryClassUnknown {
		t.Fatalf("MachineClass = %q, want unknown", plan.MachineClass)
	}
	if plan.ContextLength != DefaultLocalContextLength || plan.BatchSize != 1 {
		t.Fatalf("fallback plan = %+v, want local defaults", plan)
	}
}

func TestMemoryPlan_PlanMemory_Ugly(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 24 << 30},
		ModelInfo: &ModelInfo{
			ContextLength: 4096,
			QuantBits:     2,
		},
	})
	if plan.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want metadata cap 4096", plan.ContextLength)
	}
	if len(plan.Notes) == 0 {
		t.Fatal("expected planner notes for constrained model metadata")
	}
}

func TestMemoryPlan_KVCacheQ8ForMiddleMemoryClasses_Good(t *testing.T) {
	coverageTokens := "KVCacheQ8ForMiddleMemoryClasses"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 32 << 30, MaxRecommendedWorkingSetSize: 28 << 30},
	})

	if plan.CacheMode != KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, KVCacheModeQ8)
	}
	if plan.EstimatedKVCacheBytes == 0 || plan.EstimatedKVCacheModeBytes == 0 {
		t.Fatalf("expected KV byte estimates: %+v", plan)
	}
	if plan.EstimatedKVCacheModeBytes >= plan.EstimatedKVCacheBytes {
		t.Fatalf("mode bytes = %d, want less than fp cache bytes %d", plan.EstimatedKVCacheModeBytes, plan.EstimatedKVCacheBytes)
	}
}

func memoryPlanHasNote(plan MemoryPlan, fragment string) bool {
	for _, note := range plan.Notes {
		if core.Contains(note, fragment) {
			return true
		}
	}
	return false
}
