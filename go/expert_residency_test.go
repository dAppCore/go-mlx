// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
)

func TestExpertResidency_PlanMiniMaxM2ChoosesLazyHotSetFor96GB_Good(t *testing.T) {
	tensorPlan, err := BuildMiniMaxM2TensorPlan(MiniMaxM2Config{
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
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
	}

	plan := PlanMiniMaxM2ExpertResidency(tensorPlan, MemoryPlan{
		MachineClass:          MemoryClassApple96GB,
		MemoryLimitBytes:      76 * MemoryGiB,
		CacheLimitBytes:       7 * MemoryGiB,
		ModelWeightBytes:      60 * MemoryGiB,
		ContextLength:         32768,
		CacheMode:             KVCacheModePaged,
		ParallelSlots:         1,
		PrefillChunkSize:      2048,
		ModelQuantization:     2,
		ModelQuantizationType: "jangtq",
	}, []int{5, 3, 5, 1, 9})

	if !plan.Enabled || plan.Mode != ExpertResidencyModeLazy {
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
	recorder := NewProbeRecorder()
	manager, err := NewMiniMaxM2ExpertResidencyManager(context.Background(), MiniMaxM2ExpertResidencyConfig{
		Layer: 0,
		Policy: ExpertResidencyPlan{
			Enabled:            true,
			Mode:               ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{1},
			MaxResidentExperts: 2,
			EvictionPolicy:     ExpertEvictionLRU,
		},
		Loader: func(_ context.Context, _ int, expertID int) (MiniMaxM2PackedExpertWeights, error) {
			loaded = append(loaded, expertID)
			return tinyResidencyExpert(expertID), nil
		},
		ProbeSink: recorder,
	})
	if err != nil {
		t.Fatalf("NewMiniMaxM2ExpertResidencyManager() error = %v", err)
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
	if events[0].Kind != ProbeEventExpertResidency || events[0].ExpertResidency.Action != ExpertResidencyActionStartup {
		t.Fatalf("first event = %+v, want startup expert residency event", events[0])
	}
	if !hasExpertResidencyAction(events, ExpertResidencyActionEvict) || !hasExpertResidencyAction(events, ExpertResidencyActionPageIn) {
		t.Fatalf("events = %+v, want page-in and evict actions", events)
	}
}

func TestExpertResidency_ManagerRequiresLoaderForEnabledPolicy_Bad(t *testing.T) {
	_, err := NewMiniMaxM2ExpertResidencyManager(context.Background(), MiniMaxM2ExpertResidencyConfig{
		Policy: ExpertResidencyPlan{Enabled: true, Mode: ExpertResidencyModeLazy, StartupExpertIDs: []int{1}},
	})
	if err == nil || !core.Contains(err.Error(), "loader") {
		t.Fatalf("error = %v, want loader diagnostic", err)
	}
}

func tinyResidencyExpert(expertID int) MiniMaxM2PackedExpertWeights {
	packed := []byte{byte(expertID)}
	return MiniMaxM2PackedExpertWeights{
		GateProj: JANGPackedProjectionTensor{Packed: packed},
		UpProj:   JANGPackedProjectionTensor{Packed: packed},
		DownProj: JANGPackedProjectionTensor{Packed: packed},
	}
}

func hasExpertResidencyAction(events []ProbeEvent, action ExpertResidencyAction) bool {
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
