// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
)

// These tests close the remaining statement-coverage gaps in residency.go.
// They are device-free: pure policy maths and in-memory residency bookkeeping
// with a synthetic loader (AX-11). Helpers (tinyResidencyExpert, sameIntSlice)
// are package-scoped in residency_test.go.

// --- PlanResidency hot-trim leg -----------------------------------------

func TestResidencyCover_PlanResidency_HotTrim(t *testing.T) {
	// A 32 GB class with per-token 1 yields hotLimit 1 but resident limit 3, so
	// the pool stays lazy (total 8 > 3) while more supplied hot IDs than the hot
	// limit allows are trimmed down to the limit.
	tensorPlan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    8,
		NumExpertsPerToken: 1,
	}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	plan := PlanResidency(tensorPlan, memory.Plan{MachineClass: memory.ClassApple32GB}, []int{7, 5, 3, 1})
	if plan.Mode != memory.ExpertResidencyModeLazy {
		t.Fatalf("mode = %q, want lazy for a pool larger than the resident budget", plan.Mode)
	}
	if len(plan.HotExpertIDs) > plan.HotExperts {
		t.Fatalf("hot ids = %+v (limit %d), want trimmed to the hot limit", plan.HotExpertIDs, plan.HotExperts)
	}
	if plan.HotExperts != 1 || len(plan.HotExpertIDs) != 1 {
		t.Fatalf("hot = ids:%+v limit:%d, want exactly one hot expert after trim", plan.HotExpertIDs, plan.HotExperts)
	}
}

// --- EstimatedPackedExpertBytes dense (non-packed) leg ------------------

func TestResidencyCover_EstimatedPackedExpertBytes_DenseLeg(t *testing.T) {
	// A plan whose JANG info has an invalid group size makes every expert spec's
	// packed descriptor construction fail, so the specs carry a shape but no
	// packed descriptor. EstimatedPackedExpertBytes then falls back to the dense
	// byte estimate (2 bytes/element) rather than packed bytes.
	plan := TensorPlan{
		Config: Config{
			ModelType:          "minimax_m2",
			HiddenSize:         2,
			IntermediateSize:   2,
			NumHiddenLayers:    1,
			NumAttentionHeads:  1,
			NumKeyValueHeads:   1,
			HeadDim:            2,
			NumLocalExperts:    2,
			NumExpertsPerToken: 1,
		},
		// GroupSize 0 forces jang.NewPackedTensorDescriptor to error, so
		// expertSpec leaves Packed nil.
		JANG: &jang.Info{Profile: "JANGTQ", GroupSize: 0, BitsDefault: 2, RoutedExpertBits: 2},
	}
	// Sanity: confirm the expert specs really lack a packed descriptor so the
	// dense leg is the one under test.
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	for _, role := range []TensorRole{TensorRoleExpertGate, TensorRoleExpertUp, TensorRoleExpertDown} {
		if spec := findMiniMaxM2Spec(specs, role); spec.Packed != nil {
			t.Fatalf("expert role %s unexpectedly has a packed descriptor; dense leg not exercised", role)
		}
	}
	// 3 projections × (2×2 elements) × 2 bytes = 24.
	if got := plan.EstimatedPackedExpertBytes(); got != 24 {
		t.Fatalf("EstimatedPackedExpertBytes() = %d, want 24 (dense 2B/elem fallback)", got)
	}
}

// --- EnsureExperts nil-ctx + page-in error legs -------------------------

func TestResidencyCover_EnsureExperts_NilContext(t *testing.T) {
	// A nil context is tolerated: EnsureExperts falls back to Background and
	// loads the requested cold expert.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	//nolint:staticcheck // intentionally passing a nil context to cover the guard.
	experts, _, err := manager.EnsureExperts(nil, []int{1})
	if err != nil {
		t.Fatalf("EnsureExperts(nil ctx) error = %v", err)
	}
	if len(experts) != 1 {
		t.Fatalf("experts = %+v, want the requested cold expert loaded under Background", experts)
	}
}

func TestResidencyCover_EnsureExperts_PageInError(t *testing.T) {
	// The startup expert loads fine, but a later cold page-in fails; EnsureExperts
	// surfaces that loader error (the page-in loadExpert error leg).
	failExpert := 2
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{1},
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			if expertID == failExpert {
				return PackedExpertWeights{}, core.NewError("synthetic page-in failure")
			}
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if _, _, err := manager.EnsureExperts(context.Background(), []int{failExpert}); err == nil || !core.Contains(err.Error(), "synthetic page-in failure") {
		t.Fatalf("error = %v, want page-in loader failure propagated", err)
	}
}

// --- loadExpert ctx + nil-loader guard legs -----------------------------

func TestResidencyCover_EnsureExperts_CancelledContext(t *testing.T) {
	// A cancelled context short-circuits the cold load inside loadExpert before
	// the loader runs.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return tinyResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := manager.EnsureExperts(ctx, []int{3}); err == nil {
		t.Fatalf("error = nil, want cancelled-context error from the cold load")
	}
}

func TestResidencyCover_EnsureExperts_NilLoaderDisabledPolicy(t *testing.T) {
	// A disabled policy needs no loader, so NewResidencyManager builds a manager
	// with a nil loader and an unbounded (0) resident limit. Requesting a cold
	// expert then runs ensureCapacityFor (limit<=0 → no eviction) and reaches
	// loadExpert, which rejects the nil loader.
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer:  0,
		Policy: memory.ExpertResidencyPlan{Enabled: false},
	})
	if err != nil {
		t.Fatalf("NewResidencyManager() error = %v", err)
	}
	if _, _, err := manager.EnsureExperts(context.Background(), []int{1}); err == nil || !core.Contains(err.Error(), "loader is nil") {
		t.Fatalf("error = %v, want nil-loader diagnostic from loadExpert", err)
	}
}

// --- residentExpertLimit degenerate clamps (white-box) ------------------

func TestResidencyCover_ResidentExpertLimit_NegativePerToken(t *testing.T) {
	// A negative per-token count makes base (perToken*2) drop below perToken,
	// triggering the base<perToken clamp, then the base<1 clamp floors it to 1.
	// The default arm is taken via ClassUnknown so the maths is unambiguous.
	if got := residentExpertLimit(memory.ClassUnknown, 8, -1); got != 1 {
		t.Fatalf("residentExpertLimit(perToken=-1) = %d, want floored to 1", got)
	}
}

// --- defaultHotExpertIDs guard + clamp legs (white-box) -----------------

func TestResidencyCover_DefaultHotExpertIDs_Guards(t *testing.T) {
	// Non-positive count or total → nil (nothing to seed).
	if ids := defaultHotExpertIDs(8, 0); ids != nil {
		t.Fatalf("defaultHotExpertIDs(count=0) = %+v, want nil", ids)
	}
	if ids := defaultHotExpertIDs(0, 4); ids != nil {
		t.Fatalf("defaultHotExpertIDs(total=0) = %+v, want nil", ids)
	}
	// count exceeding total is clamped to total.
	if ids := defaultHotExpertIDs(3, 9); !sameIntSlice(ids, []int{0, 1, 2}) {
		t.Fatalf("defaultHotExpertIDs(total=3,count=9) = %+v, want clamped [0 1 2]", ids)
	}
}
