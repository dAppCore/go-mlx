// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/inference/memory"
	"dappco.re/go/inference/probe"
)

func TestResidency_PlanResidency_Good(t *testing.T) {
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

func TestResidency_ResidencyManager_EnsureExperts_Good(t *testing.T) {
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

func TestResidency_NewResidencyManager_Bad(t *testing.T) {
	_, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Policy: memory.ExpertResidencyPlan{Enabled: true, Mode: memory.ExpertResidencyModeLazy, StartupExpertIDs: []int{1}},
	})
	if err == nil || !core.Contains(err.Error(), "loader") {
		t.Fatalf("error = %v, want loader diagnostic", err)
	}
}

func TestResidency_PlanResidency_Ugly(t *testing.T) {
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

func TestResidency_PlanResidency_Bad(t *testing.T) {
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

func TestResidency_TensorPlan_EstimatedPackedExpertBytes_Good(t *testing.T) {
	// A JANGTQ plan resolves packed descriptors for the three expert
	// projections; EstimatedPackedExpertBytes sums their packed byte counts.
	plan := miniMaxM2SmallJANGTQPlan(t)
	if got := plan.EstimatedPackedExpertBytes(); got == 0 {
		t.Fatalf("EstimatedPackedExpertBytes() = %d, want non-zero packed estimate", got)
	}
}

func TestResidency_TensorPlan_EstimatedPackedExpertBytes_Bad(t *testing.T) {
	// A plan with zero layers makes the internal LayerTensorSpecs(0,0) fail, so
	// EstimatedPackedExpertBytes swallows the range error and reports 0 rather
	// than panicking.
	plan := TensorPlan{Config: Config{ModelType: "minimax_m2", NumHiddenLayers: 0, NumLocalExperts: 4}}
	if got := plan.EstimatedPackedExpertBytes(); got != 0 {
		t.Fatalf("EstimatedPackedExpertBytes(zero layers) = %d, want 0 on spec error", got)
	}
}

func TestResidency_TensorPlan_EstimatedPackedExpertBytes_Ugly(t *testing.T) {
	// Dense (non-packed) expert specs: with a nil JANG info the expert specs
	// still carry a packed descriptor, but the estimate must remain positive and
	// descriptor-derived for a degenerate-tiny expert shape (the smallest plan
	// the planner will ever see).
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    2,
		NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	if got := plan.EstimatedPackedExpertBytes(); got == 0 {
		t.Fatalf("EstimatedPackedExpertBytes() = %d, want non-zero for a tiny expert plan", got)
	}
}

func TestResidency_NewResidencyManager_Good(t *testing.T) {
	// A disabled policy needs no loader; NewResidencyManager still returns a
	// usable manager with no resident experts.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer:  0,
		Policy: memory.ExpertResidencyPlan{Enabled: false},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if manager == nil {
		t.Fatal("NewResidencyManager() returned nil manager for a disabled policy")
	}
	if len(manager.ResidentExpertIDs()) != 0 {
		t.Fatalf("resident experts = %+v, want none before any load", manager.ResidentExpertIDs())
	}
	// An enabled policy with a loader pre-loads the startup experts immediately.
	withLoader, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{4, 2},
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager(enabled) error = %v", err)
	}
	if !sameIntSlice(withLoader.ResidentExpertIDs(), []int{2, 4}) {
		t.Fatalf("startup residents = %+v, want sorted hot experts 2 and 4", withLoader.ResidentExpertIDs())
	}
}

func TestResidency_NewResidencyManager_Ugly(t *testing.T) {
	// A startup loader that fails surfaces its error out of NewResidencyManager
	// rather than returning a half-built manager.
	_, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:          true,
			Mode:             memory.ExpertResidencyModeLazy,
			StartupExpertIDs: []int{1},
		},
		Loader: func(_ context.Context, _ int, _ int) (PackedExpertWeights, error) {
			return PackedExpertWeights{}, core.NewError("synthetic startup load failure")
		},
	})
	if err == nil || !core.Contains(err.Error(), "synthetic startup load failure") {
		t.Fatalf("error = %v, want startup loader failure propagated", err)
	}
	// A nil context is tolerated: NewResidencyManager falls back to Background.
	manager, err := NewResidencyManager(nil, ResidencyConfig{Policy: memory.ExpertResidencyPlan{Enabled: false}})
	if err != nil {
		t.Fatalf("NewResidencyManager(nil ctx) error = %v", err)
	}
	if manager == nil {
		t.Fatal("NewResidencyManager(nil ctx) returned nil manager")
	}
}

func TestResidency_ResidencyManager_EnsureExperts_Bad(t *testing.T) {
	// EnsureExperts on a nil manager returns a diagnostic, not a panic.
	var manager *ResidencyManager
	if _, _, err := manager.EnsureExperts(context.Background(), []int{1}); err == nil || !core.Contains(err.Error(), "nil") {
		t.Fatalf("error = %v, want nil-manager diagnostic", err)
	}
}

func TestResidency_ResidencyManager_EnsureExperts_Ugly(t *testing.T) {
	// A bounded resident budget of 1 with two distinct experts requested in one
	// call: both are protected (same request) and neither is hot, so no eviction
	// is possible — EnsureExperts surfaces the no-evictable diagnostic rather
	// than over-committing memory.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			MaxResidentExperts: 1,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if _, _, err := manager.EnsureExperts(context.Background(), []int{1, 2}); err == nil || !core.Contains(err.Error(), "evictable") {
		t.Fatalf("error = %v, want no-evictable-cold-expert diagnostic", err)
	}
}

func TestResidency_ResidencyManager_ResidentExpertIDs_Good(t *testing.T) {
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{7, 1, 4},
			MaxResidentExperts: 8,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	// ResidentExpertIDs returns a sorted snapshot regardless of load order.
	if !sameIntSlice(manager.ResidentExpertIDs(), []int{1, 4, 7}) {
		t.Fatalf("ResidentExpertIDs() = %+v, want sorted [1 4 7]", manager.ResidentExpertIDs())
	}
}

func TestResidency_ResidencyManager_ResidentExpertIDs_Bad(t *testing.T) {
	// A nil manager yields a nil slice, not a panic.
	var manager *ResidencyManager
	if ids := manager.ResidentExpertIDs(); ids != nil {
		t.Fatalf("ResidentExpertIDs(nil manager) = %+v, want nil", ids)
	}
}

func TestResidency_ResidencyManager_ResidentExpertIDs_Ugly(t *testing.T) {
	// A freshly built manager with a disabled policy holds no experts, so
	// ResidentExpertIDs returns an empty (non-nil) sorted slice.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Policy: memory.ExpertResidencyPlan{Enabled: false},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if ids := manager.ResidentExpertIDs(); len(ids) != 0 {
		t.Fatalf("ResidentExpertIDs() = %+v, want empty for a manager with no loads", ids)
	}
}

func TestResidency_NormalisePlan_Good(t *testing.T) {
	// NormalisePlan sorts+dedups hot/startup IDs, fills the LRU eviction policy,
	// derives MaxResidentExperts from the startup set, and sets the page-in
	// batch size from the per-token count.
	got := NormalisePlan(memory.ExpertResidencyPlan{
		Enabled:          true,
		HotExpertIDs:     []int{5, 1, 5, 3},
		StartupExpertIDs: []int{2, 2, 9},
		ExpertsPerToken:  4,
	})
	if !sameIntSlice(got.HotExpertIDs, []int{1, 3, 5}) {
		t.Fatalf("HotExpertIDs = %+v, want sorted unique [1 3 5]", got.HotExpertIDs)
	}
	if !sameIntSlice(got.StartupExpertIDs, []int{2, 9}) {
		t.Fatalf("StartupExpertIDs = %+v, want sorted unique [2 9]", got.StartupExpertIDs)
	}
	if got.MaxResidentExperts != 2 {
		t.Fatalf("MaxResidentExperts = %d, want len(startup)=2", got.MaxResidentExperts)
	}
	if got.EvictionPolicy != memory.ExpertEvictionLRU || got.PageInBatchSize != 4 {
		t.Fatalf("policy = %+v, want LRU eviction and page-in batch 4", got)
	}
}

func TestResidency_NormalisePlan_Bad(t *testing.T) {
	// An enabled plan left in the Off mode is repaired to Lazy so an enabled
	// policy is never silently a no-op.
	got := NormalisePlan(memory.ExpertResidencyPlan{
		Enabled: true,
		Mode:    memory.ExpertResidencyModeOff,
	})
	if got.Mode != memory.ExpertResidencyModeLazy {
		t.Fatalf("Mode = %q, want Off repaired to Lazy when enabled", got.Mode)
	}
}

func TestResidency_NormalisePlan_Ugly(t *testing.T) {
	// A fully-zero plan: no startup IDs means MaxResidentExperts stays 0, the
	// page-in batch floors at 1 (maxPositive guard), and a disabled+Off plan is
	// left in Off (the repair only fires when Enabled).
	got := NormalisePlan(memory.ExpertResidencyPlan{})
	if got.MaxResidentExperts != 0 {
		t.Fatalf("MaxResidentExperts = %d, want 0 with no startup experts", got.MaxResidentExperts)
	}
	if got.PageInBatchSize != 1 {
		t.Fatalf("PageInBatchSize = %d, want floored to 1", got.PageInBatchSize)
	}
	if got.Mode != memory.ExpertResidencyModeOff {
		t.Fatalf("Mode = %q, want Off left untouched for a disabled plan", got.Mode)
	}
	if got.EvictionPolicy != memory.ExpertEvictionLRU {
		t.Fatalf("EvictionPolicy = %q, want LRU default", got.EvictionPolicy)
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

func TestResidency_specDenseBytes_Good(t *testing.T) {
	// 2x3 dense elements * 2 bytes = 12.
	if got := specDenseBytes(&TensorSpec{Shape: []uint64{2, 3}}); got != 12 {
		t.Fatalf("specDenseBytes([2 3]) = %d, want 12", got)
	}
}

func TestResidency_specDenseBytes_Bad(t *testing.T) {
	// No shape at all → 0 (nothing to account for).
	if got := specDenseBytes(&TensorSpec{}); got != 0 {
		t.Fatalf("specDenseBytes(no shape) = %d, want 0", got)
	}
}

func TestResidency_specDenseBytes_Ugly(t *testing.T) {
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
