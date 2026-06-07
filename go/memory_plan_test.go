// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model/minimax/m2"
	mp "dappco.re/go/mlx/pack"
)

func TestMemoryPlan_M1Class16GB_Good(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 << 30,
			MaxRecommendedWorkingSetSize: 14 << 30,
		},
	})

	if plan.MachineClass != memory.ClassApple16GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, memory.ClassApple16GB)
	}
	if plan.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", plan.ContextLength)
	}
	if plan.CachePolicy != memory.KVCacheRotating {
		t.Fatalf("CachePolicy = %q, want rotating", plan.CachePolicy)
	}
	if plan.CacheMode != memory.KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, memory.KVCacheModeKQ8VQ4)
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

	if plan.MachineClass != memory.ClassApple96GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, memory.ClassApple96GB)
	}
	if plan.ContextLength != 131072 {
		t.Fatalf("ContextLength = %d, want 131072", plan.ContextLength)
	}
	if plan.CacheMode != memory.KVCacheModeDefault {
		t.Fatalf("CacheMode = %q, want default (bounded) cache — the planner must not select the broken paged cache", plan.CacheMode)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 1 {
		t.Fatalf("cold-start shape = batch %d prefill %d slots %d, want 1/4096/1 (no model → honest local default; concurrency capacity is derived once a model is known)", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if !plan.PromptCache {
		t.Fatal("PromptCache = false, want true on 96GB class")
	}
	if plan.PreferredQuantization != 8 {
		t.Fatalf("PreferredQuantization = %d, want 8", plan.PreferredQuantization)
	}
}

func TestMemoryPlan_Gemma4SmallDefaultQuantizationPolicy_Good(t *testing.T) {
	pack := mp.ModelPack{Architecture: "gemma4_text", ContextLength: 32768, NumLayers: 34, HiddenSize: 2304}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		},
		Pack: &pack,
	})
	if plan.PreferredQuantization != 6 || plan.QualityQuantization != 8 || plan.FallbackQuantization != 4 {
		t.Fatalf("Gemma 4 quantisation policy = preferred:%d quality:%d fallback:%d, want 6/8/4", plan.PreferredQuantization, plan.QualityQuantization, plan.FallbackQuantization)
	}
	if len(plan.QuantizationCandidates) != 3 {
		t.Fatalf("Gemma 4 quantisation candidates = %+v, want machine-readable q8/q6/q4 ladder", plan.QuantizationCandidates)
	}

	cfg := applyLoadOptions([]LoadOption{WithMemoryPlan(plan)})
	got := applyMemoryPlanToLoadConfig("", cfg)
	if got.ExpectedQuantization != 6 {
		t.Fatalf("ExpectedQuantization = %d, want planner default q6", got.ExpectedQuantization)
	}
}

func TestMemoryPlan_AutoPlanOfficialGemma4SourceDoesNotExpectQ6_Good(t *testing.T) {
	dir := t.TempDir()
	writeMemoryPlanFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 1536,
			"num_hidden_layers": 35,
			"max_position_embeddings": 131072
		}
	}`)
	writeMemoryPlanFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	originalDeviceInfo := memoryPlannerDeviceInfo
	t.Cleanup(func() { memoryPlannerDeviceInfo = originalDeviceInfo })
	memoryPlannerDeviceInfo = func() DeviceInfo {
		return DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	cfg := applyLoadOptions([]LoadOption{WithAutoMemoryPlan(true)})

	got := applyMemoryPlanToLoadConfig(dir, cfg)

	if got.ExpectedQuantization != 0 {
		t.Fatalf("ExpectedQuantization = %d, want 0 for unquantised official source pack", got.ExpectedQuantization)
	}
	if got.MemoryPlan == nil {
		t.Fatal("MemoryPlan = nil, want auto-planned Gemma 4 source pack")
	}
	if got.MemoryPlan.PreferredQuantization != 6 {
		t.Fatalf("PreferredQuantization = %d, want q6 product policy preserved", got.MemoryPlan.PreferredQuantization)
	}
	if got.MemoryPlan.ModelQuantization != 0 {
		t.Fatalf("ModelQuantization = %d, want 0 for source pack without quantisation metadata", got.MemoryPlan.ModelQuantization)
	}
}

func TestMemoryPlan_AutoPlanQuantizedGemma4PackExpectsModelBits_Good(t *testing.T) {
	dir := t.TempDir()
	writeMemoryPlanFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 262144,
		"hidden_size": 1536,
		"num_hidden_layers": 35,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeMemoryPlanFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	originalDeviceInfo := memoryPlannerDeviceInfo
	t.Cleanup(func() { memoryPlannerDeviceInfo = originalDeviceInfo })
	memoryPlannerDeviceInfo = func() DeviceInfo {
		return DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	cfg := applyLoadOptions([]LoadOption{WithAutoMemoryPlan(true)})

	got := applyMemoryPlanToLoadConfig(dir, cfg)

	if got.ExpectedQuantization != 6 {
		t.Fatalf("ExpectedQuantization = %d, want inspected model q6", got.ExpectedQuantization)
	}
	if got.MemoryPlan == nil || got.MemoryPlan.ModelQuantization != 6 {
		t.Fatalf("MemoryPlan = %+v, want model quantisation q6", got.MemoryPlan)
	}
}

func TestMemoryPlan_ExplicitDefaultContextSurvivesPlannerClamp_Good(t *testing.T) {
	plan := memory.Plan{ContextLength: 32768}
	cfg := applyLoadOptions([]LoadOption{
		WithContextLength(DefaultLocalContextLength),
		WithMemoryPlan(plan),
	})

	got := applyMemoryPlanToLoadConfig("", cfg)

	if got.ContextLength != DefaultLocalContextLength {
		t.Fatalf("ContextLength = %d, want explicit default-length context %d", got.ContextLength, DefaultLocalContextLength)
	}
}

func TestMemoryPlan_ImplicitDefaultContextCanUsePlannerClamp_Good(t *testing.T) {
	plan := memory.Plan{ContextLength: 32768}
	cfg := applyLoadOptions([]LoadOption{
		WithMemoryPlan(plan),
	})

	got := applyMemoryPlanToLoadConfig("", cfg)

	if got.ContextLength != 32768 {
		t.Fatalf("ContextLength = %d, want implicit default clamped by planner", got.ContextLength)
	}
}

func TestMemoryPlan_Apple64GBUsesWidePrefill_Good(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   64 * memory.GiB,
			MaxRecommendedWorkingSetSize: 60 * memory.GiB,
		},
	})

	if plan.MachineClass != memory.ClassApple64GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, memory.ClassApple64GB)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 1 {
		t.Fatalf("cold-start shape = batch %d prefill %d slots %d, want 1/4096/1 (no model → honest local default)", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if plan.CacheMode != memory.KVCacheModeDefault || !plan.PromptCache {
		t.Fatalf("cache = mode %q prompt %t, want default (bounded) prompt cache", plan.CacheMode, plan.PromptCache)
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
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 13 * memory.GiB,
		},
		Pack: &pack,
	})

	if plan.CacheMode != memory.KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q for Qwen3-MoE on 16GB", plan.CacheMode, memory.KVCacheModeKQ8VQ4)
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
		WeightBytes: 60 * memory.GiB,
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Pack: &pack,
	})

	// MiniMax is an other-model arch not yet updated to declare its KV dims, so
	// its context derives via the hidden-size KV fallback — a 60GB pack on a
	// 96GB box lands below the 32768 arch cap. Assert the cap as the ceiling and
	// a positive derived context, not a fixed number that assumes memory it does
	// not have; the exact value firms up when MiniMax declares its real KV shape.
	if plan.ContextLength <= 0 || plan.ContextLength > 32768 || plan.BatchSize != 1 {
		t.Fatalf("MiniMax plan shape = ctx:%d batch:%d, want 0<ctx<=32768 and batch 1", plan.ContextLength, plan.BatchSize)
	}
	if plan.CacheMode != memory.KVCacheModeDefault || !plan.PromptCache {
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
		MiniMaxM2LayerSkeleton: &m2.LayerForwardSkeleton{
			Layer: 0,
			Attention: []m2.ResolvedTensor{
				{Name: "q", Role: m2.TensorRoleAttentionQ, PackedBytes: 16},
				{Name: "k", Role: m2.TensorRoleAttentionK, PackedBytes: 8},
				{Name: "v", Role: m2.TensorRoleAttentionV, PackedBytes: 8},
				{Name: "o", Role: m2.TensorRoleAttentionO, PackedBytes: 16},
			},
			RouterGate: m2.ResolvedTensor{Name: "gate", Role: m2.TensorRoleRouterGate, DType: "F32", Shape: []uint64{3, 4}},
			RouterBias: &m2.ResolvedTensor{Name: "bias", Role: m2.TensorRoleRouterBias, DType: "F32", Shape: []uint64{3}},
		},
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 90 * memory.GiB},
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
		Device: DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
		Pack:   &pack,
	})

	if plan.ContextLength != 512 {
		t.Fatalf("ContextLength = %d, want BERT max sequence 512", plan.ContextLength)
	}
	if plan.CachePolicy != memory.KVCacheDefault || plan.CacheMode != memory.KVCacheModeDefault || plan.PromptCache {
		t.Fatalf("cache policy = policy:%q mode:%q prompt:%v, want disabled generation cache for embeddings", plan.CachePolicy, plan.CacheMode, plan.PromptCache)
	}
	if plan.EstimatedKVCacheBytes != 0 || plan.EstimatedKVCacheModeBytes != 0 {
		t.Fatalf("KV estimates = fp:%d mode:%d, want zero for encoder embeddings", plan.EstimatedKVCacheBytes, plan.EstimatedKVCacheModeBytes)
	}
	if plan.BatchSize < 4 || !memoryPlanHasNote(plan, "embedding encoder") {
		t.Fatalf("plan = %+v, want embedding throughput hint", plan)
	}
}

func TestMemoryPlan_PlanMemory_Bad(t *testing.T) {
	plan := PlanMemory(MemoryPlanInput{})
	if plan.MachineClass != memory.ClassUnknown {
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
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 32 << 30, MaxRecommendedWorkingSetSize: 28 << 30},
	})

	if plan.CacheMode != memory.KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, memory.KVCacheModeQ8)
	}
	if plan.EstimatedKVCacheBytes == 0 || plan.EstimatedKVCacheModeBytes == 0 {
		t.Fatalf("expected KV byte estimates: %+v", plan)
	}
	if plan.EstimatedKVCacheModeBytes >= plan.EstimatedKVCacheBytes {
		t.Fatalf("mode bytes = %d, want less than fp cache bytes %d", plan.EstimatedKVCacheModeBytes, plan.EstimatedKVCacheBytes)
	}
}

func memoryPlanHasNote(plan memory.Plan, fragment string) bool {
	for _, note := range plan.Notes {
		if core.Contains(note, fragment) {
			return true
		}
	}
	return false
}

func writeMemoryPlanFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}
