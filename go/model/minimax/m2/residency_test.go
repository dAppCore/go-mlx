// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/probe"
)

func TestExpertResidency_PlanMiniMaxM2ChoosesLazyHotSetFor96GB_Good(t *testing.T) {
	tensorPlan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    16,
		NumExpertsPerToken: 2,
	}, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	plan := PlanResidency(tensorPlan, memory.Plan{
		MachineClass:          memory.ClassApple96GB,
		MemoryLimitBytes:      76 * memory.GiB,
		CacheLimitBytes:       7 * memory.GiB,
		ModelWeightBytes:      60 * memory.GiB,
		ContextLength:         32768,
		CacheMode:             memory.KVCacheModePaged,
		ParallelSlots:         1,
		PrefillChunkSize:      2048,
		ModelQuantization:     2,
		ModelQuantizationType: "jangtq",
	}, []int{5, 3, 5, 1, 9})

	if !plan.Enabled || plan.Mode != memory.ExpertResidencyModeLazy {
		t.Fatalf("residency mode = enabled:%v mode:%q, want lazy enabled", plan.Enabled, plan.Mode)
	}
	if plan.TotalExperts != 16 || plan.ExpertsPerToken != 2 {
		t.Fatalf("expert shape = total:%d per-token:%d, want 16/2", plan.TotalExperts, plan.ExpertsPerToken)
	}
	if plan.MaxResidentExperts != 8 {
		t.Fatalf("MaxResidentExperts = %d, want 8 for tiny 96GB MiniMax plan", plan.MaxResidentExperts)
	}
	if !sameIntSlice(plan.StartupExpertIDs, []int{1, 3, 5, 9}) {
		t.Fatalf("StartupExpertIDs = %+v, want sorted unique hot experts", plan.StartupExpertIDs)
	}
	if plan.EstimatedExpertBytes == 0 || plan.EstimatedResidentBytes == 0 {
		t.Fatalf("estimated bytes = expert:%d resident:%d, want non-zero", plan.EstimatedExpertBytes, plan.EstimatedResidentBytes)
	}
}

func TestExpertResidency_ManagerStartsHotPagesColdAndEvicts_Good(t *testing.T) {
	var loaded []int
	recorder := probe.NewRecorder()
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{1},
			MaxResidentExperts: 2,
			EvictionPolicy:     memory.ExpertEvictionLRU,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			loaded = append(loaded, expertID)
			return tinyResidencyExpert(expertID), nil
		},
		ProbeSink: recorder,
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if !sameIntSlice(loaded, []int{1}) {
		t.Fatalf("startup loads = %+v, want hot expert 1", loaded)
	}

	experts, stats, err := manager.EnsureExperts(context.Background(), []int{1, 2})
	if err != nil {
		t.Fatalf("EnsureExperts([1 2]) error = %v", err)
	}
	if len(experts) != 2 || stats.PageIns != 2 || stats.ColdLoads != 1 || stats.HotLoads != 1 {
		t.Fatalf("first stats = %+v experts=%d, want startup hot plus one cold page-in", stats, len(experts))
	}

	_, stats, err = manager.EnsureExperts(context.Background(), []int{3})
	if err != nil {
		t.Fatalf("EnsureExperts([3]) error = %v", err)
	}
	if !sameIntSlice(manager.ResidentExpertIDs(), []int{1, 3}) {
		t.Fatalf("resident experts = %+v, want hot expert 1 pinned and cold expert 3 resident", manager.ResidentExpertIDs())
	}
	if stats.PageOuts != 1 || stats.ColdLoads != 2 || stats.FirstUseLatency <= 0 {
		t.Fatalf("second stats = %+v, want one eviction, two cold loads, and first-use latency", stats)
	}

	events := recorder.Events()
	if len(events) < 3 {
		t.Fatalf("events = %+v, want startup/page-in/evict probes", events)
	}
	if events[0].Kind != probe.KindExpertResidency || events[0].ExpertResidency.Action != probe.ExpertResidencyActionStartup {
		t.Fatalf("first event = %+v, want startup expert residency event", events[0])
	}
	if !hasExpertResidencyAction(events, probe.ExpertResidencyActionEvict) || !hasExpertResidencyAction(events, probe.ExpertResidencyActionPageIn) {
		t.Fatalf("events = %+v, want page-in and evict actions", events)
	}
}

func TestExpertResidency_ManagerRequiresLoaderForEnabledPolicy_Bad(t *testing.T) {
	_, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Policy: memory.ExpertResidencyPlan{Enabled: true, Mode: memory.ExpertResidencyModeLazy, StartupExpertIDs: []int{1}},
	})
	if err == nil || !core.Contains(err.Error(), "loader") {
		t.Fatalf("error = %v, want loader diagnostic", err)
	}
}

func TestExpertResidency_PlanResidencyPinsTinyExpertPool_Ugly(t *testing.T) {
	// A tiny expert pool fits entirely in a large machine's resident budget, so
	// the planner switches to pinned mode and synthesises a default hot set
	// (exercises defaultHotExpertIDs + minPositive) instead of staying lazy.
	tensorPlan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    2,
		NumExpertsPerToken: 1,
	}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	plan := PlanResidency(tensorPlan, memory.Plan{MachineClass: memory.ClassApple128GB}, nil)

	if !plan.Enabled || plan.Mode != memory.ExpertResidencyModePinned {
		t.Fatalf("residency mode = enabled:%v mode:%q, want pinned for a fully-resident pool", plan.Enabled, plan.Mode)
	}
	if plan.MaxResidentExperts < plan.TotalExperts {
		t.Fatalf("MaxResidentExperts = %d, want >= total experts %d when pinned", plan.MaxResidentExperts, plan.TotalExperts)
	}
	// Pinned mode with no caller-supplied hints fills a default hot set capped
	// at the hot limit; every id is a valid expert index.
	for _, id := range plan.HotExpertIDs {
		if id < 0 || id >= plan.TotalExperts {
			t.Fatalf("hot expert id %d out of range for %d experts", id, plan.TotalExperts)
		}
	}
}

func TestExpertResidency_PlanResidencyDisabledWithoutExpertCounts_Bad(t *testing.T) {
	// A plan missing expert counts cannot be made resident, so the planner
	// returns a disabled policy with an explanatory note rather than panicking.
	plan := PlanResidency(TensorPlan{Config: Config{ModelType: "minimax_m2"}}, memory.Plan{MachineClass: memory.ClassApple96GB}, []int{1, 2})
	if plan.Enabled {
		t.Fatalf("plan = %+v, want disabled residency for missing expert counts", plan)
	}
	if len(plan.Notes) == 0 || !core.Contains(plan.Notes[0], "disabled") {
		t.Fatalf("notes = %+v, want a disabled-residency note", plan.Notes)
	}
	if plan.Architecture != "minimax_m2" {
		t.Fatalf("architecture = %q, want minimax_m2 even when disabled", plan.Architecture)
	}
}

func tinyResidencyExpert(expertID int) PackedExpertWeights {
	packed := []byte{byte(expertID)}
	return PackedExpertWeights{
		GateProj: JANGPackedProjectionTensor{Packed: packed},
		UpProj:   JANGPackedProjectionTensor{Packed: packed},
		DownProj: JANGPackedProjectionTensor{Packed: packed},
	}
}

func hasExpertResidencyAction(events []probe.Event, action probe.ExpertResidencyAction) bool {
	for _, event := range events {
		if event.ExpertResidency != nil && event.ExpertResidency.Action == action {
			return true
		}
	}
	return false
}

func sameIntSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
