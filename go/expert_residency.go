// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"sort"
	"time"

	core "dappco.re/go"
)

// ExpertResidencyMode names how routed MoE experts are kept resident.
type ExpertResidencyMode string

const (
	ExpertResidencyModeOff    ExpertResidencyMode = ""
	ExpertResidencyModePinned ExpertResidencyMode = "pinned"
	ExpertResidencyModeLazy   ExpertResidencyMode = "lazy"
)

// ExpertEvictionPolicy names the cold-expert eviction strategy.
type ExpertEvictionPolicy string

const (
	ExpertEvictionLRU ExpertEvictionPolicy = "lru"
)

// ExpertResidencyAction names probe-visible expert residency transitions.
type ExpertResidencyAction string

const (
	ExpertResidencyActionStartup ExpertResidencyAction = "startup"
	ExpertResidencyActionPageIn  ExpertResidencyAction = "page_in"
	ExpertResidencyActionEvict   ExpertResidencyAction = "evict"
	ExpertResidencyActionHit     ExpertResidencyAction = "hit"
)

// ExpertResidencyPlan is a backend-neutral MoE residency policy. It is small
// enough for memory planners and benchmark reports while still explicit about
// hot experts, resident limits, and expected first-use pressure.
type ExpertResidencyPlan struct {
	Enabled                 bool                 `json:"enabled"`
	Mode                    ExpertResidencyMode  `json:"mode,omitempty"`
	Architecture            string               `json:"architecture,omitempty"`
	TotalExperts            int                  `json:"total_experts,omitempty"`
	ExpertsPerToken         int                  `json:"experts_per_token,omitempty"`
	HotExpertIDs            []int                `json:"hot_expert_ids,omitempty"`
	StartupExpertIDs        []int                `json:"startup_expert_ids,omitempty"`
	HotExperts              int                  `json:"hot_experts,omitempty"`
	MaxResidentExperts      int                  `json:"max_resident_experts,omitempty"`
	PageInBatchSize         int                  `json:"page_in_batch_size,omitempty"`
	EvictionPolicy          ExpertEvictionPolicy `json:"eviction_policy,omitempty"`
	EstimatedExpertBytes    uint64               `json:"estimated_expert_bytes,omitempty"`
	EstimatedResidentBytes  uint64               `json:"estimated_resident_bytes,omitempty"`
	MaxResidentBytes        uint64               `json:"max_resident_bytes,omitempty"`
	FirstUseLatencyExpected bool                 `json:"first_use_latency_expected,omitempty"`
	Notes                   []string             `json:"notes,omitempty"`
}

// ExpertResidencyStats records measured hot-load, page-in, and eviction
// behaviour. Backends can feed this directly into workload bench reports.
type ExpertResidencyStats struct {
	ResidentExperts     int           `json:"resident_experts,omitempty"`
	PeakResidentExperts int           `json:"peak_resident_experts,omitempty"`
	HotLoads            int           `json:"hot_loads,omitempty"`
	ColdLoads           int           `json:"cold_loads,omitempty"`
	PageIns             int           `json:"page_ins,omitempty"`
	PageOuts            int           `json:"page_outs,omitempty"`
	Hits                int           `json:"hits,omitempty"`
	LoadedBytes         uint64        `json:"loaded_bytes,omitempty"`
	EvictedBytes        uint64        `json:"evicted_bytes,omitempty"`
	FirstUseLatency     time.Duration `json:"first_use_latency,omitempty"`
	TotalLoadDuration   time.Duration `json:"total_load_duration,omitempty"`
}

// MiniMaxM2ExpertResidencyLoader loads one packed routed expert for a layer.
type MiniMaxM2ExpertResidencyLoader func(context.Context, int, int) (MiniMaxM2PackedExpertWeights, error)

// MiniMaxM2ExpertResidencyConfig configures a lazy resident expert set.
type MiniMaxM2ExpertResidencyConfig struct {
	Plan      MiniMaxM2TensorPlan            `json:"plan"`
	Layer     int                            `json:"layer,omitempty"`
	Policy    ExpertResidencyPlan            `json:"policy"`
	Loader    MiniMaxM2ExpertResidencyLoader `json:"-"`
	ProbeSink ProbeSink                      `json:"-"`
	now       func() time.Time
}

// MiniMaxM2ExpertResidencyManager keeps a bounded set of routed experts in
// memory. It is deterministic and backend-neutral; native MLX/HIP loaders can
// supply the Loader hook without changing scheduler or bench contracts.
type MiniMaxM2ExpertResidencyManager struct {
	layer     int
	policy    ExpertResidencyPlan
	loader    MiniMaxM2ExpertResidencyLoader
	probeSink ProbeSink
	now       func() time.Time
	resident  map[int]MiniMaxM2PackedExpertWeights
	lastUsed  map[int]int
	hot       map[int]bool
	clock     int
	stats     ExpertResidencyStats
}

// PlanMiniMaxM2ExpertResidency derives a lazy expert policy for MiniMax M2 from
// the current memory plan. Hot IDs are optional observed/router-prior experts;
// the planner sorts and deduplicates them for reproducible state bundles.
func PlanMiniMaxM2ExpertResidency(plan MiniMaxM2TensorPlan, memory MemoryPlan, hotExpertIDs []int) ExpertResidencyPlan {
	total := plan.Config.NumLocalExperts
	perToken := plan.Config.NumExpertsPerToken
	if total <= 0 || perToken <= 0 {
		return ExpertResidencyPlan{
			Architecture: "minimax_m2",
			Notes:        []string{"MiniMax M2 expert residency disabled because expert counts are missing"},
		}
	}
	estimatedExpertBytes := plan.EstimatedPackedExpertBytes()
	residentLimit := miniMaxM2ResidentExpertLimit(memory.MachineClass, total, perToken)
	hotLimit := miniMaxM2HotExpertLimit(memory.MachineClass, total, perToken, residentLimit)
	hot := miniMaxM2UniqueExpertIDs(hotExpertIDs)
	if len(hot) > hotLimit {
		hot = hot[:hotLimit]
	}
	mode := ExpertResidencyModeLazy
	if residentLimit >= total {
		mode = ExpertResidencyModePinned
		hot = miniMaxM2DefaultHotExpertIDs(total, minPositive(hotLimit, total))
	}
	startup := append([]int(nil), hot...)
	return ExpertResidencyPlan{
		Enabled:                 true,
		Mode:                    mode,
		Architecture:            "minimax_m2",
		TotalExperts:            total,
		ExpertsPerToken:         perToken,
		HotExpertIDs:            append([]int(nil), hot...),
		StartupExpertIDs:        startup,
		HotExperts:              hotLimit,
		MaxResidentExperts:      residentLimit,
		PageInBatchSize:         maxPositive(perToken, 1),
		EvictionPolicy:          ExpertEvictionLRU,
		EstimatedExpertBytes:    estimatedExpertBytes,
		EstimatedResidentBytes:  estimatedExpertBytes * uint64(residentLimit),
		MaxResidentBytes:        estimatedExpertBytes * uint64(residentLimit),
		FirstUseLatencyExpected: mode == ExpertResidencyModeLazy,
		Notes: []string{
			"MiniMax M2 routed experts use lazy residency so cold experts are paged on first use instead of loading every expert at startup",
		},
	}
}

// EstimatedPackedExpertBytes estimates one routed expert's packed payload from
// tensor descriptors. It intentionally excludes scale/bias sidecars until native
// loaders expose measured sidecar bytes.
func (plan MiniMaxM2TensorPlan) EstimatedPackedExpertBytes() uint64 {
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		return 0
	}
	total := uint64(0)
	for _, spec := range specs {
		switch spec.Role {
		case MiniMaxM2TensorRoleExpertGate, MiniMaxM2TensorRoleExpertUp, MiniMaxM2TensorRoleExpertDown:
			if spec.Packed != nil && spec.Packed.PackedBytes > 0 {
				total += uint64(spec.Packed.PackedBytes)
			} else {
				total += miniMaxM2SpecDenseBytes(spec)
			}
		}
	}
	return total
}

// NewMiniMaxM2ExpertResidencyManager creates a resident expert set and loads
// configured startup experts immediately.
func NewMiniMaxM2ExpertResidencyManager(ctx context.Context, cfg MiniMaxM2ExpertResidencyConfig) (*MiniMaxM2ExpertResidencyManager, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	policy := normaliseExpertResidencyPlan(cfg.Policy)
	if policy.Enabled && cfg.Loader == nil {
		return nil, core.NewError("mlx: expert residency requires loader for enabled policy")
	}
	manager := &MiniMaxM2ExpertResidencyManager{
		layer:     cfg.Layer,
		policy:    policy,
		loader:    cfg.Loader,
		probeSink: cfg.ProbeSink,
		now:       cfg.now,
		resident:  map[int]MiniMaxM2PackedExpertWeights{},
		lastUsed:  map[int]int{},
		hot:       map[int]bool{},
	}
	if manager.now == nil {
		manager.now = time.Now
	}
	for _, expertID := range policy.StartupExpertIDs {
		manager.hot[expertID] = true
	}
	for _, expertID := range policy.StartupExpertIDs {
		if err := manager.loadExpert(ctx, expertID, ExpertResidencyActionStartup); err != nil {
			return nil, err
		}
	}
	return manager, nil
}

// EnsureExperts returns a map containing all requested experts, loading cold
// experts and evicting non-hot residents as required.
func (manager *MiniMaxM2ExpertResidencyManager) EnsureExperts(ctx context.Context, expertIDs []int) (map[int]MiniMaxM2PackedExpertWeights, ExpertResidencyStats, error) {
	if manager == nil {
		return nil, ExpertResidencyStats{}, core.NewError("mlx: expert residency manager is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	requested := miniMaxM2UniqueExpertIDs(expertIDs)
	for _, expertID := range requested {
		if _, ok := manager.resident[expertID]; ok {
			manager.touch(expertID)
			manager.stats.Hits++
			manager.emitExpertResidencyProbe(ExpertResidencyActionHit, []int{expertID}, 0, 0, 0)
			continue
		}
		if err := manager.ensureCapacityFor(expertID, requested); err != nil {
			return nil, manager.snapshotStats(), err
		}
		if err := manager.loadExpert(ctx, expertID, ExpertResidencyActionPageIn); err != nil {
			return nil, manager.snapshotStats(), err
		}
	}
	out := make(map[int]MiniMaxM2PackedExpertWeights, len(requested))
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
func (manager *MiniMaxM2ExpertResidencyManager) ResidentExpertIDs() []int {
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

func (manager *MiniMaxM2ExpertResidencyManager) loadExpert(ctx context.Context, expertID int, action ExpertResidencyAction) error {
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
	loadedBytes := miniMaxM2PackedExpertBytes(expert)
	manager.resident[expertID] = expert
	manager.touch(expertID)
	manager.stats.PageIns++
	manager.stats.LoadedBytes += loadedBytes
	manager.stats.TotalLoadDuration += duration
	if manager.stats.FirstUseLatency == 0 && action == ExpertResidencyActionPageIn {
		manager.stats.FirstUseLatency = duration
	}
	if action == ExpertResidencyActionStartup {
		manager.stats.HotLoads++
	} else {
		manager.stats.ColdLoads++
	}
	manager.updateResidentStats()
	manager.emitExpertResidencyProbe(action, []int{expertID}, loadedBytes, 0, duration)
	return nil
}

func (manager *MiniMaxM2ExpertResidencyManager) ensureCapacityFor(incoming int, requested []int) error {
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

func (manager *MiniMaxM2ExpertResidencyManager) evictableExpert(protected map[int]bool) (int, bool) {
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

func (manager *MiniMaxM2ExpertResidencyManager) evictExpert(expertID int) {
	expert := manager.resident[expertID]
	evictedBytes := miniMaxM2PackedExpertBytes(expert)
	delete(manager.resident, expertID)
	delete(manager.lastUsed, expertID)
	manager.stats.PageOuts++
	manager.stats.EvictedBytes += evictedBytes
	manager.updateResidentStats()
	manager.emitExpertResidencyProbe(ExpertResidencyActionEvict, []int{expertID}, 0, evictedBytes, 0)
}

func (manager *MiniMaxM2ExpertResidencyManager) touch(expertID int) {
	manager.clock++
	manager.lastUsed[expertID] = manager.clock
}

func (manager *MiniMaxM2ExpertResidencyManager) updateResidentStats() {
	manager.stats.ResidentExperts = len(manager.resident)
	if manager.stats.ResidentExperts > manager.stats.PeakResidentExperts {
		manager.stats.PeakResidentExperts = manager.stats.ResidentExperts
	}
}

func (manager *MiniMaxM2ExpertResidencyManager) snapshotStats() ExpertResidencyStats {
	stats := manager.stats
	stats.ResidentExperts = len(manager.resident)
	return stats
}

func (manager *MiniMaxM2ExpertResidencyManager) emitExpertResidencyProbe(action ExpertResidencyAction, expertIDs []int, loadedBytes, evictedBytes uint64, duration time.Duration) {
	if manager.probeSink == nil {
		return
	}
	manager.probeSink.EmitProbe(ProbeEvent{
		Kind:  ProbeEventExpertResidency,
		Phase: ProbePhasePrefill,
		Step:  manager.layer,
		ExpertResidency: &ProbeExpertResidency{
			Action:             action,
			Layer:              manager.layer,
			ExpertIDs:          append([]int(nil), expertIDs...),
			ResidentExperts:    len(manager.resident),
			MaxResidentExperts: manager.policy.MaxResidentExperts,
			LoadedBytes:        loadedBytes,
			EvictedBytes:       evictedBytes,
			Duration:           int64(duration),
		},
		Meta: map[string]string{"architecture": "minimax_m2"},
	})
}

func normaliseExpertResidencyPlan(plan ExpertResidencyPlan) ExpertResidencyPlan {
	plan.HotExpertIDs = miniMaxM2UniqueExpertIDs(plan.HotExpertIDs)
	plan.StartupExpertIDs = miniMaxM2UniqueExpertIDs(plan.StartupExpertIDs)
	if plan.Mode == ExpertResidencyModeOff && plan.Enabled {
		plan.Mode = ExpertResidencyModeLazy
	}
	if plan.EvictionPolicy == "" {
		plan.EvictionPolicy = ExpertEvictionLRU
	}
	if plan.MaxResidentExperts <= 0 && len(plan.StartupExpertIDs) > 0 {
		plan.MaxResidentExperts = len(plan.StartupExpertIDs)
	}
	if plan.PageInBatchSize <= 0 {
		plan.PageInBatchSize = maxPositive(plan.ExpertsPerToken, 1)
	}
	return plan
}

func miniMaxM2ResidentExpertLimit(class MemoryClass, total, perToken int) int {
	if total <= 0 {
		return 0
	}
	base := perToken * 2
	switch class {
	case MemoryClassApple16GB, MemoryClassApple24GB:
		base = perToken * 2
	case MemoryClassApple32GB:
		base = perToken * 3
	case MemoryClassApple64GB:
		base = perToken * 4
	case MemoryClassApple96GB:
		base = perToken * 4
	case MemoryClassApple128GB:
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

func miniMaxM2HotExpertLimit(class MemoryClass, total, perToken, residentLimit int) int {
	if residentLimit <= 0 {
		return 0
	}
	base := perToken
	switch class {
	case MemoryClassApple16GB, MemoryClassApple24GB:
		base = 0
	case MemoryClassApple32GB:
		base = perToken
	case MemoryClassApple64GB, MemoryClassApple96GB:
		base = perToken * 2
	case MemoryClassApple128GB:
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

func miniMaxM2DefaultHotExpertIDs(total, count int) []int {
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

func miniMaxM2SpecDenseBytes(spec MiniMaxM2TensorSpec) uint64 {
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

func miniMaxM2PackedExpertBytes(expert MiniMaxM2PackedExpertWeights) uint64 {
	return uint64(len(expert.GateProj.Packed) + len(expert.UpProj.Packed) + len(expert.DownProj.Packed))
}

func maxPositive(a, b int) int {
	if a > b {
		return a
	}
	return b
}
