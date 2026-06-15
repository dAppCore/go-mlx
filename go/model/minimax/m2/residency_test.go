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

// specDenseBytes estimates a dense tensor's footprint at 2 bytes/element
// (the bf16/f16 working assumption used when a packed descriptor is
// absent). It is reached by the residency byte-accounting path only when a
// spec lacks packed bytes; these units pin the shape math directly with
// synthetic specs (no device, no model — AX-11).

func TestExpertResidency_SpecDenseBytes_Good(t *testing.T) {
	// 2x3 dense elements * 2 bytes = 12.
	if got := specDenseBytes(&TensorSpec{Shape: []uint64{2, 3}}); got != 12 {
		t.Fatalf("specDenseBytes([2 3]) = %d, want 12", got)
	}
}

func TestExpertResidency_SpecDenseBytes_Bad(t *testing.T) {
	// No shape at all → 0 (nothing to account for).
	if got := specDenseBytes(&TensorSpec{}); got != 0 {
		t.Fatalf("specDenseBytes(no shape) = %d, want 0", got)
	}
}

func TestExpertResidency_SpecDenseBytes_Ugly(t *testing.T) {
	// A zero dimension makes the product meaningless → 0, not a silent
	// under-count from multiplying by zero mid-loop.
	if got := specDenseBytes(&TensorSpec{Shape: []uint64{4, 0, 8}}); got != 0 {
		t.Fatalf("specDenseBytes([4 0 8]) = %d, want 0 on zero dim", got)
	}
}

// residentExpertLimit / hotExpertLimit are pure machine-class policy
// switches. The Good residency tests exercise the 96 GB class; this
// table walks every memory.Class so each switch arm and the
// total/residentLimit floors are pinned (no device, no model — AX-11).
func TestExpertResidency_ResidentAndHotLimitsByClass(t *testing.T) {
	const perToken, total = 8, 256
	cases := []struct {
		class        memory.Class
		wantResident int
		wantHot      int
	}{
		{memory.ClassApple16GB, 16, 0},
		{memory.ClassApple24GB, 16, 0},
		{memory.ClassApple32GB, 24, 8},
		{memory.ClassApple64GB, 32, 16},
		{memory.ClassApple96GB, 32, 16},
		{memory.ClassApple128GB, 48, 32},
		{memory.ClassUnknown, 16, 8}, // default arm: resident 2x, hot keeps perToken
	}
	for _, tc := range cases {
		gotResident := residentExpertLimit(tc.class, total, perToken)
		if gotResident != tc.wantResident {
			t.Fatalf("residentExpertLimit(%s) = %d, want %d", tc.class, gotResident, tc.wantResident)
		}
		if gotHot := hotExpertLimit(tc.class, total, perToken, gotResident); gotHot != tc.wantHot {
			t.Fatalf("hotExpertLimit(%s) = %d, want %d", tc.class, gotHot, tc.wantHot)
		}
	}
}

func TestExpertResidency_LimitsDegenerate(t *testing.T) {
	// No experts at all → both limits are 0 (nothing to keep resident).
	if got := residentExpertLimit(memory.ClassApple96GB, 0, 8); got != 0 {
		t.Fatalf("residentExpertLimit(total=0) = %d, want 0", got)
	}
	// A zero resident limit forces zero hot experts regardless of class.
	if got := hotExpertLimit(memory.ClassApple128GB, 256, 8, 0); got != 0 {
		t.Fatalf("hotExpertLimit(residentLimit=0) = %d, want 0", got)
	}
	// total smaller than the class base clamps resident down to total.
	if got := residentExpertLimit(memory.ClassApple128GB, 3, 8); got != 3 {
		t.Fatalf("residentExpertLimit(total=3) = %d, want clamp to 3", got)
	}
	// hot base exceeding total clamps to total.
	if got := hotExpertLimit(memory.ClassApple128GB, 5, 8, 64); got != 5 {
		t.Fatalf("hotExpertLimit(total=5) = %d, want clamp to 5", got)
	}
}
