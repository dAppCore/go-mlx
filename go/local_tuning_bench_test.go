// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU-only side of local_tuning.go — candidate
// construction, load-option projection, measurement aggregation, and
// the per-machine identity hash. Per AX-11 — TuningCandidateLoadOptions
// runs on every candidate switch a UI offers; tuningCandidateForWorkload
// runs N times during PlanLocalTuning where N = workload count;
// tuningMachineHash runs once per discovery report. Local-tuning UIs
// can re-plan dozens of times per session.
//
// Functions that need device probing (DiscoverLocalRuntime,
// safeRuntimeDeviceInfo, PlanMemory) reach into metal/cgo and are
// intentionally OUT of scope.
//
// Run:    go test -bench='BenchmarkLocalTuning' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/memory"
)

// Sinks defeat compiler DCE. Distinct from other bench files in this package.
var (
	localTuningBenchOpts          []LoadOption
	localTuningBenchString        string
	localTuningBenchCandidate     inference.TuningCandidate
	localTuningBenchDeviceInfo    DeviceInfo
	localTuningBenchMachineInfo   inference.MachineDeviceInfo
	localTuningBenchModelInfo     ModelInfo
	localTuningBenchModelIdentity inference.ModelIdentity
	localTuningBenchWorkloads     []inference.TuningWorkload
	localTuningBenchLabels        map[string]string
	localTuningBenchRuntime       inference.RuntimeIdentity
	localTuningBenchWarning       string
)

// localTuningBenchDevice returns a representative M3 Ultra device fixture
// — close to Snider's measured topology so the bench reflects real prod
// shape rather than zero-sized defaults.
func localTuningBenchDevice() DeviceInfo {
	return DeviceInfo{
		Name:                         "Apple M3 Ultra",
		Architecture:                 "apple9",
		MaxBufferLength:              64 * memory.GiB,
		MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		MemorySize:                   96 * memory.GiB,
	}
}

// localTuningBenchModelIdentityFixture mirrors a qwen3-class model
// loaded for chat tuning.
func localTuningBenchModelIdentityFixture() inference.ModelIdentity {
	return inference.ModelIdentity{
		ID:            "qwen3-coder",
		Path:          "/models/qwen3-coder-3b-4bit",
		Architecture:  "qwen3",
		Hash:          "sha256:abcdef0123456789",
		QuantBits:     4,
		QuantGroup:    64,
		QuantType:     "Q4_0",
		ContextLength: 131072,
		NumLayers:     28,
		HiddenSize:    2048,
		VocabSize:     151936,
		Labels:        map[string]string{"profile": "chat"},
	}
}

// localTuningBenchAdapterIdentity — typical attached adapter shape.
func localTuningBenchAdapterIdentity() inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:          "/models/adapters/qwen3-coder-lora",
		Hash:          "sha256:0123456789abcdef",
		Format:        "lora",
		Rank:          16,
		Alpha:         32,
		TargetKeys:    []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		BaseModelHash: "sha256:abcdef0123456789",
	}
}

// localTuningBenchRuntimeFixture — representative metal runtime identity.
func localTuningBenchRuntimeFixture() inference.RuntimeIdentity {
	return inference.RuntimeIdentity{
		Backend:       "metal",
		Device:        "apple9",
		Version:       "go-mlx-2026.05",
		CacheMode:     string(memory.KVCacheModeFP16),
		NativeRuntime: true,
		Labels:        map[string]string{"runtime": "go-mlx"},
	}
}

// localTuningBenchMemoryPlan — representative memory.Plan output
// localTuning consumes from PlanMemory.
func localTuningBenchMemoryPlan() memory.Plan {
	return memory.Plan{
		ContextLength:        131072,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: 2048,
		BatchSize:            1,
		PrefillChunkSize:     512,
		CachePolicy:          memory.KVCacheFull,
		CacheMode:            memory.KVCacheModeFP16,
		MemoryLimitBytes:     48 * memory.GiB,
		CacheLimitBytes:      4 * memory.GiB,
		WiredLimitBytes:      24 * memory.GiB,
		Notes:                []string{"chat profile", "long-context capable"},
	}
}

// localTuningBenchCandidateFixture — populated candidate the UI saves.
func localTuningBenchCandidateFixture() inference.TuningCandidate {
	return inference.TuningCandidate{
		ID:                   "chat:fp16:131072:1",
		Workload:             inference.TuningWorkloadChat,
		Model:                localTuningBenchModelIdentityFixture(),
		Adapter:              localTuningBenchAdapterIdentity(),
		Runtime:              localTuningBenchRuntimeFixture(),
		ContextLength:        131072,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: 2048,
		CachePolicy:          string(memory.KVCacheFull),
		CacheMode:            string(memory.KVCacheModeFP16),
		BatchSize:            1,
		PrefillChunkSize:     512,
		ExpectedQuantization: 4,
		MemoryLimitBytes:     48 * memory.GiB,
		CacheLimitBytes:      4 * memory.GiB,
		WiredLimitBytes:      24 * memory.GiB,
		Reasons:              []string{"chat profile"},
		Labels:               map[string]string{"machine_class": "workstation"},
	}
}

// --- TuningCandidateLoadOptions — per-candidate option-projection ---

func BenchmarkLocalTuning_TuningCandidateLoadOptions_Populated(b *testing.B) {
	candidate := localTuningBenchCandidateFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchOpts = TuningCandidateLoadOptions(candidate)
	}
}

// Sparse candidate — most fields zero, exercises the early-out branches.
func BenchmarkLocalTuning_TuningCandidateLoadOptions_Sparse(b *testing.B) {
	candidate := inference.TuningCandidate{
		Workload:    inference.TuningWorkloadChat,
		PromptCache: true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchOpts = TuningCandidateLoadOptions(candidate)
	}
}

// --- tuningCandidateForWorkload — per-workload candidate builder ---

func BenchmarkLocalTuning_TuningCandidateForWorkload_Chat(b *testing.B) {
	modelIdentity := localTuningBenchModelIdentityFixture()
	adapter := localTuningBenchAdapterIdentity()
	runtime := localTuningBenchRuntimeFixture()
	plan := localTuningBenchMemoryPlan()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchCandidate = tuningCandidateForWorkload(inference.TuningWorkloadChat, modelIdentity, adapter, runtime, plan)
	}
}

func BenchmarkLocalTuning_TuningCandidateForWorkload_LowLatency(b *testing.B) {
	modelIdentity := localTuningBenchModelIdentityFixture()
	adapter := localTuningBenchAdapterIdentity()
	runtime := localTuningBenchRuntimeFixture()
	plan := localTuningBenchMemoryPlan()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchCandidate = tuningCandidateForWorkload(inference.TuningWorkloadLowLatency, modelIdentity, adapter, runtime, plan)
	}
}

func BenchmarkLocalTuning_TuningCandidateForWorkload_LongContext(b *testing.B) {
	modelIdentity := localTuningBenchModelIdentityFixture()
	adapter := localTuningBenchAdapterIdentity()
	runtime := localTuningBenchRuntimeFixture()
	plan := localTuningBenchMemoryPlan()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchCandidate = tuningCandidateForWorkload(inference.TuningWorkloadLongContext, modelIdentity, adapter, runtime, plan)
	}
}

func BenchmarkLocalTuning_TuningCandidateForWorkload_AgentState(b *testing.B) {
	modelIdentity := localTuningBenchModelIdentityFixture()
	adapter := localTuningBenchAdapterIdentity()
	runtime := localTuningBenchRuntimeFixture()
	plan := localTuningBenchMemoryPlan()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchCandidate = tuningCandidateForWorkload(inference.TuningWorkloadAgentState, modelIdentity, adapter, runtime, plan)
	}
}

// --- tuningRequestDevice (with populated device — skips cgo fallback) ---

func BenchmarkLocalTuning_TuningRequestDevice_Populated(b *testing.B) {
	device := inference.MachineDeviceInfo{
		Name:                         "Apple M3 Ultra",
		Architecture:                 "apple9",
		MaxBufferLength:              64 * memory.GiB,
		MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		MemorySize:                   96 * memory.GiB,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchDeviceInfo = tuningRequestDevice(device)
	}
}

// --- tuningDeviceInfo — DeviceInfo → MachineDeviceInfo ---

func BenchmarkLocalTuning_TuningDeviceInfo(b *testing.B) {
	device := localTuningBenchDevice()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchMachineInfo = tuningDeviceInfo(device)
	}
}

// --- tuningMachineHash — JSON-marshal + SHA256 per discovery report ---

func BenchmarkLocalTuning_TuningMachineHash_Populated(b *testing.B) {
	device := localTuningBenchDevice()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchString = tuningMachineHash(device)
	}
}

func BenchmarkLocalTuning_TuningMachineHash_Empty(b *testing.B) {
	device := DeviceInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchString = tuningMachineHash(device)
	}
}

// --- tuningModelInfo — ModelIdentity → ModelInfo ---

func BenchmarkLocalTuning_TuningModelInfo(b *testing.B) {
	identity := localTuningBenchModelIdentityFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchModelInfo = tuningModelInfo(identity)
	}
}

// --- discoveredModelIdentity — DiscoveredModel → ModelIdentity ---

func BenchmarkLocalTuning_DiscoveredModelIdentity(b *testing.B) {
	model := inference.DiscoveredModel{
		Path:       "/models/qwen3-coder-3b-4bit",
		ModelType:  "qwen3",
		QuantBits:  4,
		QuantGroup: 64,
		QuantType:  "Q4_0",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchModelIdentity = discoveredModelIdentity(model)
	}
}

// --- tuningWorkloadsOrDefault ---

func BenchmarkLocalTuning_TuningWorkloadsOrDefault_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchWorkloads = tuningWorkloadsOrDefault(nil)
	}
}

func BenchmarkLocalTuning_TuningWorkloadsOrDefault_Populated(b *testing.B) {
	workloads := []inference.TuningWorkload{
		inference.TuningWorkloadChat,
		inference.TuningWorkloadCoding,
		inference.TuningWorkloadLongContext,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchWorkloads = tuningWorkloadsOrDefault(workloads)
	}
}

// --- cloneTuningLabels / withTuningMachineHash ---

func BenchmarkLocalTuning_CloneTuningLabels_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchLabels = cloneTuningLabels(nil)
	}
}

func BenchmarkLocalTuning_CloneTuningLabels_4Entries(b *testing.B) {
	labels := map[string]string{
		"profile":       "chat",
		"runtime":       "go-mlx",
		"machine_class": "workstation",
		"region":        "local",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchLabels = cloneTuningLabels(labels)
	}
}

func BenchmarkLocalTuning_WithTuningMachineHash_AddsHash(b *testing.B) {
	labels := map[string]string{
		"profile": "chat",
		"runtime": "go-mlx",
	}
	hash := "sha256:0123456789abcdef0123456789abcdef0123456789abcdef"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchLabels = withTuningMachineHash(labels, hash)
	}
}

// --- boolLabel — trivial branch label ---

func BenchmarkLocalTuning_BoolLabel_True(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchString = boolLabel(true)
	}
}

// --- tuningRuntimeForArchitecture — profile.LookupArchitectureProfile ---

func BenchmarkLocalTuning_TuningRuntimeForArchitecture_KnownArch(b *testing.B) {
	runtime := localTuningBenchRuntimeFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchRuntime, localTuningBenchWarning = tuningRuntimeForArchitecture(runtime, "qwen3")
	}
}

func BenchmarkLocalTuning_TuningRuntimeForArchitecture_UnknownArch(b *testing.B) {
	runtime := localTuningBenchRuntimeFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		localTuningBenchRuntime, localTuningBenchWarning = tuningRuntimeForArchitecture(runtime, "unknown-arch")
	}
}
