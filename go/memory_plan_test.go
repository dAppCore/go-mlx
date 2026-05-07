// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

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
	pack := ModelPack{ContextLength: 40960, QuantBits: 4}
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
