// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"context"
	"sort"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/probe"
)

// ResidencyLoader loads one packed routed expert for a layer.
type ResidencyLoader func(context.Context, int, int) (PackedExpertWeights, error)

// ResidencyConfig configures a lazy resident expert set.
type ResidencyConfig struct {
	Plan      TensorPlan                 `json:"plan"`
	Layer     int                        `json:"layer,omitempty"`
	Policy    memory.ExpertResidencyPlan `json:"policy"`
	Loader    ResidencyLoader            `json:"-"`
	ProbeSink probe.Sink                 `json:"-"`
	now       func() time.Time
}

// ResidencyManager keeps a bounded set of routed experts in
// memory. It is deterministic and backend-neutral; native MLX/HIP loaders can
// supply the Loader hook without changing scheduler or bench contracts.
type ResidencyManager struct {
	layer     int
	policy    memory.ExpertResidencyPlan
	loader    ResidencyLoader
	probeSink probe.Sink
	now       func() time.Time
	resident  map[int]PackedExpertWeights
	lastUsed  map[int]int
	hot       map[int]bool
	clock     int
	stats     memory.ExpertResidencyStats
}

// PlanResidency derives a lazy expert policy for MiniMax M2 from
// the current memory plan. Hot IDs are optional observed/router-prior experts;
// the planner sorts and deduplicates them for reproducible state bundles.
func PlanResidency(plan TensorPlan, memPlan memory.Plan, hotExpertIDs []int) memory.ExpertResidencyPlan {
	total := plan.Config.NumLocalExperts
	perToken := plan.Config.NumExpertsPerToken
	if total <= 0 || perToken <= 0 {
		return memory.ExpertResidencyPlan{
			Architecture: "minimax_m2",
			Notes:        []string{"MiniMax M2 expert residency disabled because expert counts are missing"},
		}
	}
	estimatedExpertBytes := plan.EstimatedPackedExpertBytes()
	residentLimit := residentExpertLimit(memPlan.MachineClass, total, perToken)
	hotLimit := hotExpertLimit(memPlan.MachineClass, total, perToken, residentLimit)
	hot := uniqueExpertIDs(hotExpertIDs)
	if len(hot) > hotLimit {
		hot = hot[:hotLimit]
	}
	mode := memory.ExpertResidencyModeLazy
	if residentLimit >= total {
		mode = memory.ExpertResidencyModePinned
		hot = defaultHotExpertIDs(total, minPositive(hotLimit, total))
	}
	startup := core.SliceClone(hot)
	return memory.ExpertResidencyPlan{
		Enabled:                 true,
		Mode:                    mode,
		Architecture:            "minimax_m2",
		TotalExperts:            total,
		ExpertsPerToken:         perToken,
		HotExpertIDs:            core.SliceClone(hot),
		StartupExpertIDs:        startup,
		HotExperts:              hotLimit,
		MaxResidentExperts:      residentLimit,
		PageInBatchSize:         maxPositive(perToken, 1),
		EvictionPolicy:          memory.ExpertEvictionLRU,
		EstimatedExpertBytes:    estimatedExpertBytes,
		EstimatedResidentBytes:  estimatedExpertBytes * uint64(residentLimit),
		MaxResidentBytes:        estimatedExpertBytes * uint64(residentLimit),
		FirstUseLatencyExpected: mode == memory.ExpertResidencyModeLazy,
		Notes: []string{
			"MiniMax M2 routed experts use lazy residency so cold experts are paged on first use instead of loading every expert at startup",
		},
	}
}

// EstimatedPackedExpertBytes estimates one routed expert's packed payload from
// tensor descriptors. It intentionally excludes scale/bias sidecars until native
// loaders expose measured sidecar bytes.
func (plan TensorPlan) EstimatedPackedExpertBytes() uint64 {
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		return 0
	}
	total := uint64(0)
	for _, spec := range specs {
		switch spec.Role {
		case TensorRoleExpertGate, TensorRoleExpertUp, TensorRoleExpertDown:
			if spec.Packed != nil && spec.Packed.PackedBytes > 0 {
				total += uint64(spec.Packed.PackedBytes)
			} else {
				total += specDenseBytes(spec)
			}
		}
	}
	return total
}

// NewResidencyManager creates a resident expert set and loads
// configured startup experts immediately.
func NewResidencyManager(ctx context.Context, cfg ResidencyConfig) (*ResidencyManager, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	policy := NormalisePlan(cfg.Policy)
	if policy.Enabled && cfg.Loader == nil {
		return nil, core.NewError("mlx: expert residency requires loader for enabled policy")
	}
	residentHint := policy.MaxResidentExperts
	if residentHint <= 0 {
		residentHint = len(policy.StartupExpertIDs)
	}
	manager := &ResidencyManager{
		layer:     cfg.Layer,
		policy:    policy,
		loader:    cfg.Loader,
		probeSink: cfg.ProbeSink,
		now:       cfg.now,
		resident:  make(map[int]PackedExpertWeights, residentHint),
		lastUsed:  make(map[int]int, residentHint),
		hot:       make(map[int]bool, len(policy.StartupExpertIDs)),
	}
	if manager.now == nil {
		manager.now = time.Now
	}
	for _, expertID := range policy.StartupExpertIDs {
		manager.hot[expertID] = true
	}
	for _, expertID := range policy.StartupExpertIDs {
		if err := manager.loadExpert(ctx, expertID, probe.ExpertResidencyActionStartup); err != nil {
			return nil, err
		}
	}
	return manager, nil
}

// EnsureExperts returns a map containing all requested experts, loading cold
// experts and evicting non-hot residents as required.
func (manager *ResidencyManager) EnsureExperts(ctx context.Context, expertIDs []int) (map[int]PackedExpertWeights, memory.ExpertResidencyStats, error) {
	if manager == nil {
		return nil, memory.ExpertResidencyStats{}, core.NewError("mlx: expert residency manager is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	requested := uniqueExpertIDs(expertIDs)
	for _, expertID := range requested {
		if _, ok := manager.resident[expertID]; ok {
			manager.touch(expertID)
			manager.stats.Hits++
			manager.emitExpertResidencyProbe(probe.ExpertResidencyActionHit, []int{expertID}, 0, 0, 0)
			continue
		}
		if err := manager.ensureCapacityFor(expertID, requested); err != nil {
			return nil, manager.snapshotStats(), err
		}
		if err := manager.loadExpert(ctx, expertID, probe.ExpertResidencyActionPageIn); err != nil {
			return nil, manager.snapshotStats(), err
		}
	}
	out := make(map[int]PackedExpertWeights, len(requested))
	for _, expertID := range requested {
		expert, ok := manager.resident[expertID]
		if !ok {
			return nil, manager.snapshotStats(), core.NewError(core.Sprintf("mlx: expert %d is not resident after load", expertID))
		}
		out[expertID] = expert
	}
	return out, manager.snapshotStats(), nil
}

// ResidentExpertIDs returns sorted resident expert IDs.
func (manager *ResidencyManager) ResidentExpertIDs() []int {
	if manager == nil {
		return nil
	}
	ids := make([]int, 0, len(manager.resident))
	for expertID := range manager.resident {
		ids = append(ids, expertID)
	}
	sort.Ints(ids)
	return ids
}

func (manager *ResidencyManager) loadExpert(ctx context.Context, expertID int, action probe.ExpertResidencyAction) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if manager.loader == nil {
		return core.NewError("mlx: expert residency loader is nil")
	}
	start := manager.now()
	expert, err := manager.loader(ctx, manager.layer, expertID)
	duration := nonZeroDuration(manager.now().Sub(start))
	if err != nil {
		return err
	}
	loadedBytes := packedExpertBytes(expert)
	manager.resident[expertID] = expert
	manager.touch(expertID)
	manager.stats.PageIns++
	manager.stats.LoadedBytes += loadedBytes
	manager.stats.TotalLoadDuration += duration
	if manager.stats.FirstUseLatency == 0 && action == probe.ExpertResidencyActionPageIn {
		manager.stats.FirstUseLatency = duration
	}
	if action == probe.ExpertResidencyActionStartup {
		manager.stats.HotLoads++
	} else {
		manager.stats.ColdLoads++
	}
	manager.updateResidentStats()
	manager.emitExpertResidencyProbe(action, []int{expertID}, loadedBytes, 0, duration)
	return nil
}

func (manager *ResidencyManager) ensureCapacityFor(incoming int, requested []int) error {
	limit := manager.policy.MaxResidentExperts
	if limit <= 0 {
		return nil
	}
	protected := map[int]bool{incoming: true}
	for _, expertID := range requested {
		if _, ok := manager.resident[expertID]; ok {
			protected[expertID] = true
		}
	}
	for len(manager.resident)+1 > limit {
		victim, ok := manager.evictableExpert(protected)
		if !ok {
			return core.NewError("mlx: expert residency has no evictable cold expert")
		}
		manager.evictExpert(victim)
	}
	return nil
}

func (manager *ResidencyManager) evictableExpert(protected map[int]bool) (int, bool) {
	var victim int
	var victimUse int
	found := false
	for expertID := range manager.resident {
		if protected[expertID] || manager.hot[expertID] {
			continue
		}
		used := manager.lastUsed[expertID]
		if !found || used < victimUse {
			victim = expertID
			victimUse = used
			found = true
		}
	}
	return victim, found
}

func (manager *ResidencyManager) evictExpert(expertID int) {
	expert := manager.resident[expertID]
	evictedBytes := packedExpertBytes(expert)
	delete(manager.resident, expertID)
	delete(manager.lastUsed, expertID)
	manager.stats.PageOuts++
	manager.stats.EvictedBytes += evictedBytes
	manager.updateResidentStats()
	manager.emitExpertResidencyProbe(probe.ExpertResidencyActionEvict, []int{expertID}, 0, evictedBytes, 0)
}

func (manager *ResidencyManager) touch(expertID int) {
	manager.clock++
	manager.lastUsed[expertID] = manager.clock
}

func (manager *ResidencyManager) updateResidentStats() {
	manager.stats.ResidentExperts = len(manager.resident)
	if manager.stats.ResidentExperts > manager.stats.PeakResidentExperts {
		manager.stats.PeakResidentExperts = manager.stats.ResidentExperts
	}
}

func (manager *ResidencyManager) snapshotStats() memory.ExpertResidencyStats {
	stats := manager.stats
	stats.ResidentExperts = len(manager.resident)
	return stats
}

func (manager *ResidencyManager) emitExpertResidencyProbe(action probe.ExpertResidencyAction, expertIDs []int, loadedBytes, evictedBytes uint64, duration time.Duration) {
	if manager.probeSink == nil {
		return
	}
	manager.probeSink.EmitProbe(probe.Event{
		Kind:  probe.KindExpertResidency,
		Phase: probe.PhasePrefill,
		Step:  manager.layer,
		ExpertResidency: &probe.ExpertResidency{
			Action:             action,
			Layer:              manager.layer,
			ExpertIDs:          core.SliceClone(expertIDs),
			ResidentExperts:    len(manager.resident),
			MaxResidentExperts: manager.policy.MaxResidentExperts,
			LoadedBytes:        loadedBytes,
			EvictedBytes:       evictedBytes,
			Duration:           int64(duration),
		},
		Meta: map[string]string{"architecture": "minimax_m2"},
	})
}

func NormalisePlan(plan memory.ExpertResidencyPlan) memory.ExpertResidencyPlan {
	plan.HotExpertIDs = uniqueExpertIDs(plan.HotExpertIDs)
	plan.StartupExpertIDs = uniqueExpertIDs(plan.StartupExpertIDs)
	if plan.Mode == memory.ExpertResidencyModeOff && plan.Enabled {
		plan.Mode = memory.ExpertResidencyModeLazy
	}
	if plan.EvictionPolicy == "" {
		plan.EvictionPolicy = memory.ExpertEvictionLRU
	}
	if plan.MaxResidentExperts <= 0 && len(plan.StartupExpertIDs) > 0 {
		plan.MaxResidentExperts = len(plan.StartupExpertIDs)
	}
	if plan.PageInBatchSize <= 0 {
		plan.PageInBatchSize = maxPositive(plan.ExpertsPerToken, 1)
	}
	return plan
}

func residentExpertLimit(class memory.Class, total, perToken int) int {
	if total <= 0 {
		return 0
	}
	base := perToken * 2
	switch class {
	case memory.ClassApple16GB, memory.ClassApple24GB:
		base = perToken * 2
	case memory.ClassApple32GB:
		base = perToken * 3
	case memory.ClassApple64GB:
		base = perToken * 4
	case memory.ClassApple96GB:
		base = perToken * 4
	case memory.ClassApple128GB:
		base = perToken * 6
	default:
		base = perToken * 2
	}
	if base < perToken {
		base = perToken
	}
	if base < 1 {
		base = 1
	}
	if base > total {
		return total
	}
	return base
}

func hotExpertLimit(class memory.Class, total, perToken, residentLimit int) int {
	if residentLimit <= 0 {
		return 0
	}
	base := perToken
	switch class {
	case memory.ClassApple16GB, memory.ClassApple24GB:
		base = 0
	case memory.ClassApple32GB:
		base = perToken
	case memory.ClassApple64GB, memory.ClassApple96GB:
		base = perToken * 2
	case memory.ClassApple128GB:
		base = perToken * 4
	}
	if base > residentLimit {
		base = residentLimit
	}
	if base > total {
		return total
	}
	return base
}

func defaultHotExpertIDs(total, count int) []int {
	if count <= 0 || total <= 0 {
		return nil
	}
	if count > total {
		count = total
	}
	ids := make([]int, count)
	for i := range ids {
		ids[i] = i
	}
	return ids
}

func specDenseBytes(spec TensorSpec) uint64 {
	if len(spec.Shape) == 0 {
		return 0
	}
	elements := uint64(1)
	for _, dim := range spec.Shape {
		if dim == 0 {
			return 0
		}
		elements *= dim
	}
	return elements * 2
}

func packedExpertBytes(expert PackedExpertWeights) uint64 {
	return uint64(len(expert.GateProj.Packed) + len(expert.UpProj.Packed) + len(expert.DownProj.Packed))
}
