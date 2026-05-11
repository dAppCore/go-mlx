// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model/minimax/m2"
	"dappco.re/go/mlx/probe"
)

// ExpertResidencyMode names how routed MoE experts are kept resident.
// Aliased from dappco.re/go/mlx/memory/.
type ExpertResidencyMode = memory.ExpertResidencyMode

const (
	ExpertResidencyModeOff    = memory.ExpertResidencyModeOff
	ExpertResidencyModePinned = memory.ExpertResidencyModePinned
	ExpertResidencyModeLazy   = memory.ExpertResidencyModeLazy
)

// ExpertEvictionPolicy names the cold-expert eviction strategy.
// Aliased from dappco.re/go/mlx/memory/.
type ExpertEvictionPolicy = memory.ExpertEvictionPolicy

const (
	ExpertEvictionLRU = memory.ExpertEvictionLRU
)

// ExpertResidencyAction names probe-visible expert residency transitions.
// Aliased from dappco.re/go/mlx/probe/.
type ExpertResidencyAction = probe.ExpertResidencyAction

const (
	ExpertResidencyActionStartup = probe.ExpertResidencyActionStartup
	ExpertResidencyActionPageIn  = probe.ExpertResidencyActionPageIn
	ExpertResidencyActionEvict   = probe.ExpertResidencyActionEvict
	ExpertResidencyActionHit     = probe.ExpertResidencyActionHit
)

// ExpertResidencyPlan is a backend-neutral MoE residency policy.
// Aliased from dappco.re/go/mlx/memory/.
type ExpertResidencyPlan = memory.ExpertResidencyPlan

// ExpertResidencyStats records measured hot-load, page-in, and eviction
// behaviour. Aliased from dappco.re/go/mlx/memory/.
type ExpertResidencyStats = memory.ExpertResidencyStats

// MiniMaxM2ExpertResidencyLoader loads one packed routed expert for a layer.
// Aliased from dappco.re/go/mlx/model/minimax/m2/.
type MiniMaxM2ExpertResidencyLoader = m2.ResidencyLoader

// MiniMaxM2ExpertResidencyConfig configures a lazy resident expert set.
// Aliased from dappco.re/go/mlx/model/minimax/m2/.
type MiniMaxM2ExpertResidencyConfig = m2.ResidencyConfig

// MiniMaxM2ExpertResidencyManager keeps a bounded set of routed experts.
// Aliased from dappco.re/go/mlx/model/minimax/m2/.
type MiniMaxM2ExpertResidencyManager = m2.ResidencyManager

// PlanMiniMaxM2ExpertResidency derives a lazy expert policy for MiniMax M2.
//
//	plan := mlx.PlanMiniMaxM2ExpertResidency(tensorPlan, memoryPlan, hotIDs)
func PlanMiniMaxM2ExpertResidency(plan MiniMaxM2TensorPlan, memoryPlan MemoryPlan, hotExpertIDs []int) ExpertResidencyPlan {
	return m2.PlanResidency(plan, memoryPlan, hotExpertIDs)
}

// NewMiniMaxM2ExpertResidencyManager creates a resident expert set and
// loads configured startup experts immediately.
//
//	mgr, err := mlx.NewMiniMaxM2ExpertResidencyManager(ctx, cfg)
func NewMiniMaxM2ExpertResidencyManager(ctx context.Context, cfg MiniMaxM2ExpertResidencyConfig) (*MiniMaxM2ExpertResidencyManager, error) {
	return m2.NewResidencyManager(ctx, cfg)
}

// normaliseExpertResidencyPlan fills missing fields on a residency plan
// (page-in batch size, eviction policy, max-resident expert count).
// Retained as a private mlx-root helper for workload_bench.go.
func normaliseExpertResidencyPlan(plan ExpertResidencyPlan) ExpertResidencyPlan {
	return m2.NormalisePlan(plan)
}
