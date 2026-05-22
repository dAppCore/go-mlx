// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the profile package — BuiltinAlgorithmProfiles,
// LookupAlgorithmProfile, AlgorithmCapabilities (the algorithm side),
// plus BuiltinArchitectureProfiles, LookupArchitectureProfile,
// ArchitectureID, ArchitectureIDs (the architecture side).
//
// Per AX-11 — these surfaces are touched on every CapabilityReport()
// call (algorithm capabilities is appended), on every model-load
// architecture-resolution path (LookupArchitectureProfile /
// ArchitectureID), and on every profile clone/list. Cold-start latency
// budget flows through them.
//
// Run:    go test -bench='BenchmarkProfile' -benchmem -run='^$' ./go/profile

package profile_test

import (
	"testing"

	"dappco.re/go/inference"
	prof "dappco.re/go/mlx/profile"
)

// Sinks defeat compiler DCE.
var (
	profileBenchSinkAlgorithms     []prof.AlgorithmProfile
	profileBenchSinkAlgorithm      prof.AlgorithmProfile
	profileBenchSinkAlgorithmOK    bool
	profileBenchSinkCapabilities   []inference.Capability
	profileBenchSinkArchitectures  []prof.ModelArchitectureProfile
	profileBenchSinkArchitecture   prof.ModelArchitectureProfile
	profileBenchSinkArchitectureRP *prof.ModelArchitectureProfile
	profileBenchSinkArchOK         bool
	profileBenchSinkArchIDs        []string
	profileBenchSinkArchID         string
)

// --- BuiltinAlgorithmProfiles ---
// Full-list clone of the 14-entry built-in algorithm matrix. Fires
// once per CapabilityReport via AlgorithmCapabilities.

func BenchmarkProfile_BuiltinAlgorithmProfiles(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithms = prof.BuiltinAlgorithmProfiles()
	}
}

// --- LookupAlgorithmProfile ---
// Linear scan over the built-in list — hits early (first entry),
// late (deep in list), and miss-path.

func BenchmarkProfile_LookupAlgorithmProfile_EarlyHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityScheduler)
	}
}

func BenchmarkProfile_LookupAlgorithmProfile_LateHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityCacheDisk)
	}
}

func BenchmarkProfile_LookupAlgorithmProfile_Miss(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkAlgorithm, profileBenchSinkAlgorithmOK = prof.LookupAlgorithmProfile(inference.CapabilityID("not-a-real-cap"))
	}
}

// --- AlgorithmCapabilities ---
// Fires on every CapabilityReport — produces the inference.Capability
// slice consumed by the metalCapabilityReport.

func BenchmarkProfile_AlgorithmCapabilities(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkCapabilities = prof.AlgorithmCapabilities()
	}
}

// --- BuiltinArchitectureProfiles ---
// Deep clone of the architecture matrix.

func BenchmarkProfile_BuiltinArchitectureProfiles(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectures = prof.BuiltinArchitectureProfiles()
	}
}

// --- LookupArchitectureProfile ---

func BenchmarkProfile_LookupArchitectureProfile_Native(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("qwen3")
	}
}

// Transformers-name path — exercises architectureFromTransformersName.
func BenchmarkProfile_LookupArchitectureProfile_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("Qwen3ForCausalLM")
	}
}

// Alias path — exercises the second-pass alias scan.
func BenchmarkProfile_LookupArchitectureProfile_Alias(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("MiniMaxM2ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfile_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitecture, profileBenchSinkArchOK = prof.LookupArchitectureProfile("")
	}
}

// --- LookupArchitectureProfileRef ---
// Pointer-into-static-table form used by read-only callers (planFit,
// archSupported, archNativeRuntime, tuningRuntimeForArchitecture,
// memory.NewPlan, model.pack inspectors). Should be zero-alloc.

func BenchmarkProfile_LookupArchitectureProfileRef_Native(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("qwen3")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("Qwen3ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_Alias(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("MiniMaxM2ForCausalLM")
	}
}

func BenchmarkProfile_LookupArchitectureProfileRef_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchitectureRP, profileBenchSinkArchOK = prof.LookupArchitectureProfileRef("")
	}
}

// --- ArchitectureID ---
// Hot path during model-load — resolves Transformers names back to
// internal architecture IDs.

func BenchmarkProfile_ArchitectureID_TransformersName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("Gemma4ForConditionalGeneration")
	}
}

func BenchmarkProfile_ArchitectureID_Direct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("qwen3")
	}
}

func BenchmarkProfile_ArchitectureID_Normalised(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("qwen-3.5")
	}
}

func BenchmarkProfile_ArchitectureID_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchID = prof.ArchitectureID("")
	}
}

// --- ArchitectureIDs ---
// Slice clone of the full architecture-ID list.

func BenchmarkProfile_ArchitectureIDs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		profileBenchSinkArchIDs = prof.ArchitectureIDs()
	}
}
