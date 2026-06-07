// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"strconv"
	"sync"
	"sync/atomic"

	"dappco.re/go/mlx/memory"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/profile"
)

// metal_capabilities.go: the backend capability report plus the device/runtime
// label strings — what the Metal backend advertises to the inference layer.

var metalCapabilityDeviceInfo = func(available bool) DeviceInfo {
	if !available {
		return DeviceInfo{}
	}
	return safeRuntimeDeviceInfo()
}

// metalDeviceLabel cache — the device probe returns the same
// (MemorySize, MaxRecommendedWorkingSetSize) tuple for the whole process
// lifetime (host RAM doesn't grow between calls). A single-slot lookup
// matches the singleton-device pattern; tests that swap the
// metalCapabilityDeviceInfo hook with synthetic device shapes still
// re-format on the first call with the new tuple.
//
// The cache stores an immutable *metalDeviceLabelEntry behind an
// atomic.Pointer so the hot read path is lock-free. Cache misses (new
// device or first call) take the rare-path mutex to populate; misses
// during test hook swaps are bounded by the number of distinct device
// shapes exercised in a single run.
type metalDeviceLabelEntry struct {
	memorySize     uint64
	workingSetSize uint64
	memoryStr      string
	workingSetStr  string
}

var (
	metalDeviceLabelCache atomic.Pointer[metalDeviceLabelEntry]
	metalDeviceLabelMu    sync.Mutex
)

// metalRuntimeLabelsEntry caches the per-call runtimeLabels map for a
// given device shape AND loadReady value. The map header itself (~80 B)
// would otherwise allocate per call — the singleton-device contract +
// boolLabel's two-string output means ≤ 2 distinct maps fit the entire
// process lifetime. atomic.Pointer keeps the read path lock-free.
type metalRuntimeLabelsEntry struct {
	memorySize     uint64
	workingSetSize uint64
	loadReady      bool
	labels         map[string]string
}

// metalRuntimeLabelsCache stores both the loadReady=true and loadReady=false
// shapes side-by-side — at most one of each. Tests that swap the
// metalCapabilityDeviceInfo hook with synthetic device shapes invalidate
// both slots on the next call with the new tuple.
type metalRuntimeLabelsCachePair struct {
	loadReadyTrue  *metalRuntimeLabelsEntry
	loadReadyFalse *metalRuntimeLabelsEntry
}

var (
	metalRuntimeLabelsCache atomic.Pointer[metalRuntimeLabelsCachePair]
	metalRuntimeLabelsMu    sync.Mutex
)

// metalDeviceLabelStrings returns the strconv.FormatUint outputs for
// (memorySize, workingSetSize). The atomic single-slot cache hits on
// every subsequent call with the same tuple — lock-free read path,
// rare-path mutex only on miss. Returns "" for any zero-size input
// (so callers can branch on the empty string instead of duplicating
// the > 0 check).
func metalDeviceLabelStrings(memorySize, workingSetSize uint64) (string, string) {
	if memorySize == 0 && workingSetSize == 0 {
		return "", ""
	}
	if entry := metalDeviceLabelCache.Load(); entry != nil &&
		entry.memorySize == memorySize && entry.workingSetSize == workingSetSize {
		return entry.memoryStr, entry.workingSetStr
	}
	return metalDeviceLabelStringsSlow(memorySize, workingSetSize)
}

// metalDeviceLabelStringsSlow is the cache-miss path — populates the
// shared cache under the mutex. Split out so the fast atomic load path
// stays inlineable.
func metalDeviceLabelStringsSlow(memorySize, workingSetSize uint64) (string, string) {
	metalDeviceLabelMu.Lock()
	defer metalDeviceLabelMu.Unlock()
	// Double-check under the lock — another goroutine may have populated
	// the cache while we were waiting.
	if entry := metalDeviceLabelCache.Load(); entry != nil &&
		entry.memorySize == memorySize && entry.workingSetSize == workingSetSize {
		return entry.memoryStr, entry.workingSetStr
	}
	entry := &metalDeviceLabelEntry{
		memorySize:     memorySize,
		workingSetSize: workingSetSize,
	}
	if memorySize > 0 {
		entry.memoryStr = strconv.FormatUint(memorySize, 10)
	}
	if workingSetSize > 0 {
		entry.workingSetStr = strconv.FormatUint(workingSetSize, 10)
	}
	metalDeviceLabelCache.Store(entry)
	return entry.memoryStr, entry.workingSetStr
}

// metalRuntimeLabels returns the per-Capability-Report Runtime.Labels map
// for (memorySize, workingSetSize, loadReady). The result is a shared
// singleton — consumers (go-ml fallback, go-ai providers) treat the field
// as read-only so a shared map is safe. Lock-free atomic read on the hot
// path; rare-path mutex only on miss.
func metalRuntimeLabels(memoryBytesStr, workingSetBytesStr string, memorySize, workingSetSize uint64, loadReady bool) map[string]string {
	if pair := metalRuntimeLabelsCache.Load(); pair != nil {
		slot := pair.loadReadyTrue
		if !loadReady {
			slot = pair.loadReadyFalse
		}
		if slot != nil && slot.memorySize == memorySize && slot.workingSetSize == workingSetSize {
			return slot.labels
		}
	}
	return metalRuntimeLabelsSlow(memoryBytesStr, workingSetBytesStr, memorySize, workingSetSize, loadReady)
}

// metalRuntimeLabelsSlow is the cache-miss path. Builds the map under the
// mutex; preserves the OTHER loadReady slot when present + still device-
// matched, so a single (true) + single (false) call doesn't churn each
// other out.
func metalRuntimeLabelsSlow(memoryBytesStr, workingSetBytesStr string, memorySize, workingSetSize uint64, loadReady bool) map[string]string {
	metalRuntimeLabelsMu.Lock()
	defer metalRuntimeLabelsMu.Unlock()
	if pair := metalRuntimeLabelsCache.Load(); pair != nil {
		slot := pair.loadReadyTrue
		if !loadReady {
			slot = pair.loadReadyFalse
		}
		if slot != nil && slot.memorySize == memorySize && slot.workingSetSize == workingSetSize {
			return slot.labels
		}
	}
	labels := make(map[string]string, 3)
	if memoryBytesStr != "" {
		labels["memory_bytes"] = memoryBytesStr
	}
	if workingSetBytesStr != "" {
		labels["working_set_bytes"] = workingSetBytesStr
	}
	labels["load_available"] = boolLabel(loadReady)
	entry := &metalRuntimeLabelsEntry{
		memorySize:     memorySize,
		workingSetSize: workingSetSize,
		loadReady:      loadReady,
		labels:         labels,
	}
	// Preserve the other-loadReady slot if it still matches the same
	// device — only invalidate when the device shape itself shifts.
	pair := &metalRuntimeLabelsCachePair{}
	if existing := metalRuntimeLabelsCache.Load(); existing != nil {
		if loadReady {
			pair.loadReadyFalse = existing.loadReadyFalse
		} else {
			pair.loadReadyTrue = existing.loadReadyTrue
		}
		// Drop the preserved slot if the device shape no longer matches.
		if loadReady && pair.loadReadyFalse != nil &&
			(pair.loadReadyFalse.memorySize != memorySize || pair.loadReadyFalse.workingSetSize != workingSetSize) {
			pair.loadReadyFalse = nil
		}
		if !loadReady && pair.loadReadyTrue != nil &&
			(pair.loadReadyTrue.memorySize != memorySize || pair.loadReadyTrue.workingSetSize != workingSetSize) {
			pair.loadReadyTrue = nil
		}
	}
	if loadReady {
		pair.loadReadyTrue = entry
	} else {
		pair.loadReadyFalse = entry
	}
	metalRuntimeLabelsCache.Store(pair)
	return labels
}

func metalCapabilityReport(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool) inference.CapabilityReport {
	return metalCapabilityReportWithLoadReady(model, adapter, available, available)
}

func metalCapabilityReportWithLoadReady(model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool, loadReady bool) inference.CapabilityReport {
	device := metalCapabilityDeviceInfo(available)
	// Cache the per-DeviceInfo formatted strings — the device probe
	// returns the same (MemorySize, WorkingSet) tuple for the whole
	// process lifetime (the host doesn't grow RAM between calls). The
	// shared cache hits on every subsequent call and reuses the
	// previously formatted strings, dropping 2 strconv allocs per
	// CapabilityReport invocation when the cache hits.
	memoryBytesStr, workingSetBytesStr := metalDeviceLabelStrings(device.MemorySize, device.MaxRecommendedWorkingSetSize)
	// Cache the whole runtimeLabels map per (device, loadReady) shape.
	// Real callers see only 2 distinct shapes per process (loadReady=true
	// and loadReady=false against the same singleton device), so the map
	// header allocation (~80 B per call) collapses to a single one-time
	// cost. metalRuntimeLabels is read-only — consumers don't mutate.
	runtimeLabels := metalRuntimeLabels(memoryBytesStr, workingSetBytesStr, device.MemorySize, device.MaxRecommendedWorkingSetSize, loadReady)
	// Full pre-built capability list — see metalCapabilityFixedFull /
	// metalCapabilityFixedFullMarked. Both forms (head + fixed tail) are
	// merged once at package init; the !loadReady tail has already been
	// passed through markMetalUnavailableCapabilities once at init.
	// Per call we just hand back the singleton — same Wave-5+ shared-
	// read-only-singleton pattern Architectures / Quantizations /
	// CacheModes / Labels adopted above. Drops the per-call
	// make([]inference.Capability, 39) alloc (~4 KB / 1 alloc) and the
	// copy() body that followed it; the only meaningful per-call cost
	// is now the CapabilityReport struct itself (returned by value).
	capabilities := metalCapabilityFixedFull
	if !loadReady {
		capabilities = metalCapabilityFixedFullMarked
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "metal",
			Device:        device.Architecture,
			NativeRuntime: true,
			Labels:        runtimeLabels,
		},
		Model:     model,
		Adapter:   adapter,
		Available: available,
		// Architectures / Quantizations / CacheModes share the package-init
		// singletons directly. The consumer surface is read-only — the only
		// callers that ever stored these into another struct (local_tuning
		// MachineDiscoveryReport, go-ml/go-ai display paths) clone defensively
		// at their own boundary, and no code in go-ml / go-ai / lem / cmd
		// mutates a CapabilityReport.{Architectures,Quantizations,CacheModes}
		// slice. Drops 3 clone allocs (~256 B) per CapabilityReport call.
		Architectures: metalCapabilityArchitectures,
		Quantizations: metalCapabilityQuantizations,
		CacheModes:    metalCapabilityCacheModes,
		Capabilities:  capabilities,
		// Single shared singleton — the value is the same constant on every
		// call ({"library": "go-mlx"}) and consumers treat report.Labels as
		// read-only (go-ml / go-ai never mutate it). Skips one map make +
		// one map-bucket alloc per CapabilityReport (~80 B + 1 alloc).
		Labels: metalCapabilityReportLabels,
	}
}

// metalLoadBlockedCapabilities is the immutable lookup table of
// capability IDs that get marked unsupported when the Metal runtime
// is unavailable. Hoisted to package-level so markMetalUnavailable-
// Capabilities doesn't rebuild a 26-entry hash map on every call.
var metalLoadBlockedCapabilities = map[inference.CapabilityID]bool{
	inference.CapabilityModelLoad:      true,
	inference.CapabilityAutoTuning:     true,
	inference.CapabilityEvaluation:     true,
	inference.CapabilityGenerate:       true,
	inference.CapabilityChat:           true,
	inference.CapabilityClassify:       true,
	inference.CapabilityBatchGenerate:  true,
	inference.CapabilityLoRAInference:  true,
	inference.CapabilityStateBundle:    true,
	inference.CapabilityKVSnapshot:     true,
	inference.CapabilityPromptCache:    true,
	inference.CapabilityAgentMemory:    true,
	inference.CapabilityStateWake:      true,
	inference.CapabilityStateSleep:     true,
	inference.CapabilityStateFork:      true,
	inference.CapabilityLoRATraining:   true,
	inference.CapabilityDistillation:   true,
	inference.CapabilityGRPO:           true,
	inference.CapabilityProbeEvents:    true,
	inference.CapabilityAttentionProbe: true,
	inference.CapabilityLogitProbe:     true,
	inference.CapabilityScheduler:      true,
	inference.CapabilityRequestCancel:  true,
	inference.CapabilityCacheBlocks:    true,
	inference.CapabilityCacheWarm:      true,
}

func markMetalUnavailableCapabilities(capabilities []inference.Capability) []inference.Capability {
	const detail = "native Metal runtime is unavailable; no usable Metal device is visible for model loading"
	for i := range capabilities {
		if !metalLoadBlockedCapabilities[capabilities[i].ID] {
			continue
		}
		capabilities[i].Status = inference.CapabilityStatusUnsupported
		if core.Contains(capabilities[i].Detail, "native Metal runtime is unavailable") {
			continue
		}
		if capabilities[i].Detail == "" {
			capabilities[i].Detail = detail
		} else {
			capabilities[i].Detail = detail + "; " + capabilities[i].Detail
		}
	}
	return capabilities
}

// metalCapabilityFixedCount is the number of always-present capability
// entries in metalCapabilityReportWithLoadReady's literal — used to
// pre-size the capabilities slice in one allocation so the AlgorithmCapabilities
// append doesn't need to grow. Update this if the literal entry count
// changes (the test in inference_contract_test.go counts the slice
// after build and asserts the expected total).
const metalCapabilityFixedCount = 39

// metalModelLoadAvailable / metalModelLoadUnavailable are the two
// possible shapes of the capabilities[0] entry built per call from
// loadReady. inference.SupportedCapability / UnsupportedCapability
// each allocate (constructor + labels map) — caching the two
// outcomes once at package init drops 1–2 allocs per call.
var (
	metalModelLoadAvailable   = inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime)
	metalModelLoadUnavailable = inference.UnsupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime, "native Metal runtime is unavailable; no usable Metal device is visible for model loading")
)

// metalCapabilityFixedTail / metalCapabilityFixedTailMarked are the two
// pre-built shapes of the tail (38 static entries + AlgorithmCapabilities
// from profile). One mirrors the loadReady=true form, the other has
// already been passed through markMetalUnavailableCapabilities once at
// package init. They're folded into metalCapabilityFixedFull /
// metalCapabilityFixedFullMarked below (head + tail) — the per-call
// path now reads only the full forms directly.
//
// This drops the per-call markMetalUnavailableCapabilities scan (a 39+N
// element loop + ~4 string concat allocs per call when the populated-
// Detail entries got rewritten). Sharing the underlying Labels-map header
// is safe because markMetalUnavailableCapabilities only writes Status and
// Detail value fields, never touches Labels.
//
// Initialised via init() so we run after the profile package's own init
// has populated builtinAlgorithmProfilesData.
var (
	metalCapabilityFixedTail       []inference.Capability
	metalCapabilityFixedTailMarked []inference.Capability
	// metalCapabilityFixedFull / metalCapabilityFixedFullMarked are the
	// full per-call slices — head (metalModelLoadAvailable /
	// metalModelLoadUnavailable) plus the corresponding tail, pre-built
	// once at init. Consumers (go-ml / go-ai / local_tuning) treat the
	// Capabilities slice as read-only, mirroring the same convention
	// Architectures / Quantizations / CacheModes / Labels rely on. This
	// folds the per-call make([]inference.Capability, 39) (~4 KB / 1
	// alloc) into a one-time init cost. The two slices are independent
	// backings so a hypothetical-but-unsupported consumer mutation in
	// one branch cannot bleed into the other.
	metalCapabilityFixedFull       []inference.Capability
	metalCapabilityFixedFullMarked []inference.Capability
)

func init() {
	algorithmCaps := profile.AlgorithmCapabilities()
	metalCapabilityFixedTail = make([]inference.Capability, 0, len(metalCapabilityStaticTail)+len(algorithmCaps))
	metalCapabilityFixedTail = append(metalCapabilityFixedTail, metalCapabilityStaticTail...)
	metalCapabilityFixedTail = append(metalCapabilityFixedTail, algorithmCaps...)
	// Pre-mark the !loadReady variant once. We deep-copy first so the
	// loadReady path keeps its un-rewritten Status/Detail entries.
	metalCapabilityFixedTailMarked = make([]inference.Capability, len(metalCapabilityFixedTail))
	copy(metalCapabilityFixedTailMarked, metalCapabilityFixedTail)
	metalCapabilityFixedTailMarked = markMetalUnavailableCapabilities(metalCapabilityFixedTailMarked)
	// Build the head-prepended full forms once. Independent backings so
	// either branch can be exposed without aliasing the other.
	metalCapabilityFixedFull = make([]inference.Capability, 1+len(metalCapabilityFixedTail))
	metalCapabilityFixedFull[0] = metalModelLoadAvailable
	copy(metalCapabilityFixedFull[1:], metalCapabilityFixedTail)
	metalCapabilityFixedFullMarked = make([]inference.Capability, 1+len(metalCapabilityFixedTailMarked))
	metalCapabilityFixedFullMarked[0] = metalModelLoadUnavailable
	copy(metalCapabilityFixedFullMarked[1:], metalCapabilityFixedTailMarked)
}

// metalCapabilityStaticTail is the 38-entry portion of the capability
// list that does NOT vary with loadReady. metalCapabilityReportWithLoad-
// Ready prepends the per-call modelLoadCapability (entry 0 — varies
// because it switches between Supported and Unsupported based on
// loadReady) and appends the per-call algorithmCaps tail (varies in
// length); the middle is identical on every call. Pre-building once at
// package init replaces 38 SupportedCapability/Experimental/Planned
// calls + 38 boxed append args with one bulk slice copy. Keep in sync
// with metalCapabilityFixedCount (38 entries here + 1 modelLoadCapability
// at index 0 = 39).
var metalCapabilityStaticTail = []inference.Capability{
	inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityRuntimeDiscovery, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAutoTuning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelReplace, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelSlice, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityQuantization, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityModelMerge, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityClassify, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityTokenizer, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityLoRAInference, inference.CapabilityGroupModel),
	inference.SupportedCapability(inference.CapabilityStateBundle, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityKVSnapshot, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityPromptCache, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAgentMemory, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateWake, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateSleep, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityStateFork, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityLoRATraining, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityDistillation, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityGRPO, inference.CapabilityGroupTraining),
	inference.SupportedCapability(inference.CapabilityProbeEvents, inference.CapabilityGroupProbe),
	inference.SupportedCapability(inference.CapabilityAttentionProbe, inference.CapabilityGroupProbe),
	inference.SupportedCapability(inference.CapabilityLogitProbe, inference.CapabilityGroupProbe),
	inference.ExperimentalCapability(inference.CapabilitySplitInference, inference.CapabilityGroupModel, "local dense Qwen split execution supports Metal attention/logits plus CPU FFN; remote FFN/expert execution is not wired yet"),
	inference.PlannedCapability(inference.CapabilityDifferentialLoad, inference.CapabilityGroupRuntime, "base/fine-tune differential loading belongs in go-ai/go-ml orchestration"),
	inference.PlannedCapability(inference.CapabilityVIndex, inference.CapabilityGroupProbe, "LarQL-style vindex extraction is planned for research queries"),
	inference.SupportedCapability(inference.CapabilityResponsesAPI, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityAnthropicMessages, inference.CapabilityGroupRuntime),
	inference.SupportedCapability(inference.CapabilityOllamaCompat, inference.CapabilityGroupRuntime),
}

var (
	metalCapabilityArchitectures = profile.ArchitectureIDs()
	metalCapabilityQuantizations = []string{
		"bf16",
		"fp16",
		"jang",
		"jangtq",
		"codebook",
		"vq",
		"mxtq",
		"q4_0",
		"q4_k_m",
		"q5",
		"q8_0",
		"iq",
		"mxfp4",
		"nvfp4",
	}
	metalCapabilityCacheModes = []string{
		string(memory.KVCacheModeFP16),
		string(memory.KVCacheModeQ8),
		string(memory.KVCacheModeKQ8VQ4),
		string(memory.KVCacheModePaged),
		string(memory.KVCacheModeTurboQuant),
	}
	// metalCapabilityReportLabels is the shared CapabilityReport.Labels
	// payload — the value is the same constant on every call and
	// downstream consumers (go-ml / go-ai) only read this field, so the
	// single-allocation literal that used to fire per call now lives at
	// package init. Saves ~80 B + 1 alloc per metalCapabilityReport call.
	metalCapabilityReportLabels = map[string]string{"library": "go-mlx"}
)
