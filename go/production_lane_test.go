// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/profile"
)

func TestProductionLane_DefaultGemma4E2B_Good(t *testing.T) {
	lane := DefaultProductionLane()

	if lane.ModelID != "mlx-community/gemma-4-e2b-it-6bit" {
		t.Fatalf("ModelID = %q, want Gemma 4 E2B q6 default", lane.ModelID)
	}
	if lane.Architecture != "gemma4_text" || lane.ChatTemplate != "gemma4" || lane.QuantBits != 6 {
		t.Fatalf("lane identity = %+v, want Gemma 4 text q6 with Gemma chat template", lane)
	}
	if ProductionLaneProductDefaultQuantBits != 6 || ProductionLaneQualityQuantBits != 8 || ProductionLaneConstrainedQuantBits != 4 {
		t.Fatalf("quant constants = default:%d quality:%d constrained:%d, want 6/8/4", ProductionLaneProductDefaultQuantBits, ProductionLaneQualityQuantBits, ProductionLaneConstrainedQuantBits)
	}
	if lane.ContextLength != 4096 || lane.MaxTokens != 128 || lane.Runs != 3 {
		t.Fatalf("profile shape = context:%d tokens:%d runs:%d, want GOAL.md target shape", lane.ContextLength, lane.MaxTokens, lane.Runs)
	}
	if ProductionLaneLongContextLength != 32768 || ProductionLaneHyperLongContextLength != 131072 || ProductionLaneLongFormMaxTokens != 8192 || ProductionLaneLongContextPrefillChunkSize != 512 || ProductionLaneLongContextPromptChunkBytes != 4096 || ProductionLanePagedKVPageSize != 2048 || ProductionLaneRetainedKVCacheDType != "fp16" {
		t.Fatalf("long context shape = context:%d hyper:%d tokens:%d prefill:%d prompt:%d page:%d dtype:%s, want retained-state defaults", ProductionLaneLongContextLength, ProductionLaneHyperLongContextLength, ProductionLaneLongFormMaxTokens, ProductionLaneLongContextPrefillChunkSize, ProductionLaneLongContextPromptChunkBytes, ProductionLanePagedKVPageSize, ProductionLaneRetainedKVCacheDType)
	}
	if lane.IncludeOutput || !lane.TraceTokenPhases {
		t.Fatalf("profile reporting = include_output:%v trace:%v, want hidden output plus token phase trace", lane.IncludeOutput, lane.TraceTokenPhases)
	}
	if lane.Prompt != DefaultNewSessionText || !core.Contains(lane.Prompt, "Lemma") {
		t.Fatalf("Prompt = %q, want Lemma new-session default", lane.Prompt)
	}
}

func TestProductionLane_DefaultProductionQuantizationPolicy_Good(t *testing.T) {
	policy := DefaultProductionQuantizationPolicy()

	if policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID || policy.AssistantModelID != OfficialGemma4E2BAssistantLock().ModelID || policy.ArchivedBaseline != ProductionLaneArchivedBaselineModelID {
		t.Fatalf("policy identity = %+v, want official target+assistant plus archived q4 baseline", policy)
	}
	if policy.DefaultBits != 6 || policy.QualityBits != 8 || policy.ConstrainedBits != 4 {
		t.Fatalf("policy bits = default:%d quality:%d constrained:%d, want 6/8/4", policy.DefaultBits, policy.QualityBits, policy.ConstrainedBits)
	}
	if policy.ActiveParameterEstimate != ProductionLaneActiveParameterEstimate || !core.Contains(policy.DecodeThroughputEstimate, "memory bandwidth") {
		t.Fatalf("throughput estimate = params:%d formula:%q, want active-weight-read model", policy.ActiveParameterEstimate, policy.DecodeThroughputEstimate)
	}
	for _, metric := range []string{
		"load_duration",
		"peak_memory_bytes",
		"retained_restore_duration",
		"raw_decode_tokens_per_sec",
		"active_weight_read_bytes_per_token",
		"memory_bandwidth_bytes_per_sec",
		"long_output_quality_flags",
		"step_down_working_set_bytes",
		"context_length",
	} {
		if !stringSliceContains(policy.RequiredBenchmarkMetrics, metric) {
			t.Fatalf("RequiredBenchmarkMetrics = %v, missing %q", policy.RequiredBenchmarkMetrics, metric)
		}
	}
	if len(policy.Tiers) != 3 {
		t.Fatalf("tiers = %+v, want quality/default/constrained", policy.Tiers)
	}
	if policy.Tiers[0].Bits != 8 || policy.Tiers[0].ModelID != "mlx-community/gemma-4-e2b-it-8bit" || !policy.Tiers[0].QualityFirst || policy.Tiers[0].StepDownToBits != 6 {
		t.Fatalf("quality tier = %+v, want q8 quality-first", policy.Tiers[0])
	}
	if policy.Tiers[0].ActiveWeightReadBytesPerToken != 2300000000 {
		t.Fatalf("quality tier active read = %d, want q8 active-weight-read estimate", policy.Tiers[0].ActiveWeightReadBytesPerToken)
	}
	if policy.Tiers[1].Bits != 6 || policy.Tiers[1].ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !policy.Tiers[1].ProductDefault || policy.Tiers[1].StepDownToBits != 4 {
		t.Fatalf("default tier = %+v, want q6 product default", policy.Tiers[1])
	}
	if policy.Tiers[1].ActiveWeightReadBytesPerToken != 1725000000 {
		t.Fatalf("default tier active read = %d, want q6 active-weight-read estimate", policy.Tiers[1].ActiveWeightReadBytesPerToken)
	}
	if policy.Tiers[2].Bits != 4 || policy.Tiers[2].ModelID != "mlx-community/gemma-4-e2b-it-4bit" || !policy.Tiers[2].ConstrainedOnly || !policy.Tiers[2].ArchivedControl {
		t.Fatalf("constrained tier = %+v, want q4 constrained archived control", policy.Tiers[2])
	}
	if policy.Tiers[2].ActiveWeightReadBytesPerToken != 1150000000 {
		t.Fatalf("constrained tier active read = %d, want q4 active-weight-read estimate", policy.Tiers[2].ActiveWeightReadBytesPerToken)
	}

	if len(policy.SupportedPacks) != 7 {
		t.Fatalf("supported packs = %+v, want mlx-community mxfp4/mxfp8/4bit/5bit/6bit/8bit/bf16", policy.SupportedPacks)
	}
	byName := make(map[string]ProductionQuantizationPackSupport, len(policy.SupportedPacks))
	for _, pack := range policy.SupportedPacks {
		byName[pack.Name] = pack
		if !pack.Supported || pack.ModelID == "" || pack.QuantMode == "" {
			t.Fatalf("supported pack = %+v, want explicit supported model/mode", pack)
		}
	}
	for name, want := range map[string]struct {
		modelID    string
		bits       int
		mode       string
		group      int
		role       string
		bench      bool
		nativeOnly bool
	}{
		"mxfp4": {"mlx-community/gemma-4-e2b-it-mxfp4", 4, "mxfp4", 32, "research", true, false},
		"mxfp8": {"mlx-community/gemma-4-e2b-it-mxfp8", 8, "mxfp8", 32, "research", true, false},
		"4bit":  {"mlx-community/gemma-4-e2b-it-4bit", 4, "affine", 64, "constrained", false, false},
		"5bit":  {"mlx-community/gemma-4-e2b-it-5bit", 5, "affine", 64, "bench", true, false},
		"6bit":  {"mlx-community/gemma-4-e2b-it-6bit", 6, "affine", 64, "default", false, false},
		"8bit":  {"mlx-community/gemma-4-e2b-it-8bit", 8, "affine", 64, "quality", false, false},
		"bf16":  {"mlx-community/gemma-4-e2b-it-bf16", 16, "bf16", 0, "quality-control", true, true},
	} {
		got, ok := byName[name]
		if !ok {
			t.Fatalf("supported packs missing %q", name)
		}
		if got.ModelID != want.modelID || got.Bits != want.bits || got.QuantMode != want.mode ||
			got.QuantGroup != want.group || got.ProductRole != want.role || got.RequiresBench != want.bench ||
			got.RequiresNative != want.nativeOnly {
			t.Fatalf("supported pack %q = %+v, want %+v", name, got, want)
		}
	}
}

func TestProductionLane_DefaultPoliciesReturnDefensiveCopies_Good(t *testing.T) {
	quant := DefaultProductionQuantizationPolicy()
	quant.RequiredBenchmarkMetrics[0] = "mutated"
	quant.Tiers[0].Bits = 99
	quant.SupportedPacks[0].Name = "mutated"
	if next := DefaultProductionQuantizationPolicy(); next.RequiredBenchmarkMetrics[0] == "mutated" || next.Tiers[0].Bits == 99 || next.SupportedPacks[0].Name == "mutated" {
		t.Fatalf("DefaultProductionQuantizationPolicy leaked mutable slices: %+v", next)
	}
	packs := DefaultProductionQuantizationPackSupport()
	packs[0].Name = "mutated"
	if next := DefaultProductionQuantizationPackSupport(); next[0].Name == "mutated" {
		t.Fatalf("DefaultProductionQuantizationPackSupport leaked mutable slice: %+v", next)
	}

	gates := DefaultGemma4FastRuntimeGates()
	gates[0] = "mutated"
	if next := DefaultGemma4FastRuntimeGates(); next[0] == "mutated" {
		t.Fatalf("DefaultGemma4FastRuntimeGates leaked mutable slice: %v", next)
	}

	mtp := DefaultProductionMTPPolicy()
	mtp.RequiredDraftTokenSweeps[0] = 99
	mtp.RequiredMetrics[0] = "mutated"
	if next := DefaultProductionMTPPolicy(); next.RequiredDraftTokenSweeps[0] == 99 || next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionMTPPolicy leaked mutable slices: %+v", next)
	}

	turbo := DefaultProductionTurboQuantPolicy()
	turbo.CompareAgainstCacheModes[0] = memory.KVCacheModeTurboQuant
	turbo.RequiredMetrics[0] = "mutated"
	if next := DefaultProductionTurboQuantPolicy(); next.CompareAgainstCacheModes[0] == memory.KVCacheModeTurboQuant || next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionTurboQuantPolicy leaked mutable slices: %+v", next)
	}

	combined := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	combined.RequiredMetrics[0] = "mutated"
	if next := DefaultProductionCombinedMTPAndTurboQuantPolicy(); next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionCombinedMTPAndTurboQuantPolicy leaked mutable slice: %+v", next)
	}
}

func TestProductionLane_ProductionQuantizationPackByName_Good(t *testing.T) {
	q5, ok := ProductionQuantizationPackByName("5BIT")
	if !ok {
		t.Fatal("ProductionQuantizationPackByName(5BIT) = false, want q5 bench pack")
	}
	if q5.ModelID != "mlx-community/gemma-4-e2b-it-5bit" || q5.Bits != 5 || q5.QuantMode != "affine" || !q5.RequiresBench {
		t.Fatalf("q5 pack = %+v, want affine q5 bench pack", q5)
	}

	mxfp8, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-e2b-it-mxfp8")
	if !ok {
		t.Fatal("ProductionQuantizationPackByName(model id) = false, want mxfp8 pack")
	}
	if mxfp8.Name != "mxfp8" || mxfp8.Bits != 8 || mxfp8.QuantMode != "mxfp8" || mxfp8.QuantGroup != 32 {
		t.Fatalf("mxfp8 pack = %+v, want mxfp8/g32 support", mxfp8)
	}

	if _, ok := ProductionQuantizationPackByName("q7"); ok {
		t.Fatal("ProductionQuantizationPackByName(q7) = true, want unsupported")
	}
}

func TestProductionLane_DefaultProductionArchitectureStatus_Good(t *testing.T) {
	status := DefaultProductionArchitectureStatus()

	if status.TotalArchitectures != 25 || status.NativeArchitectures != 18 || status.MetadataOnlyArchitectures != 7 {
		t.Fatalf("status counts = total:%d native:%d metadata:%d, want 25/18/7", status.TotalArchitectures, status.NativeArchitectures, status.MetadataOnlyArchitectures)
	}
	if status.RemovePythonFallbackReady {
		t.Fatal("RemovePythonFallbackReady = true, want false until metadata-only gaps are native")
	}
	for _, id := range []string{"gemma4", "gemma4_assistant", "minimax_m2", "granite", "bert", "bert_rerank"} {
		if !stringSliceContains(status.NativeIDs, id) {
			t.Fatalf("NativeIDs = %v, missing %q", status.NativeIDs, id)
		}
	}
	for _, id := range []string{"qwen3_6", "qwen3_6_moe", "qwen3_moe", "mixtral", "deepseek", "gpt_oss", "kimi"} {
		if !stringSliceContains(status.MetadataOnlyIDs, id) {
			t.Fatalf("MetadataOnlyIDs = %v, missing %q", status.MetadataOnlyIDs, id)
		}
	}

	gaps := make(map[string]ProductionArchitectureGap, len(status.RemainingGaps))
	for _, gap := range status.RemainingGaps {
		gaps[gap.ID] = gap
	}
	qwen36 := gaps["qwen3_6"]
	if qwen36.MissingNative != "hybrid linear attention" || !stringSliceContains(qwen36.NextWork, "linear_attention_kernel") || qwen36.MoE {
		t.Fatalf("qwen3_6 gap = %+v, want dense hybrid linear-attention work", qwen36)
	}
	deepseek := gaps["deepseek"]
	if deepseek.MissingNative != "MoE router plus MLA attention variants" || !deepseek.MoE || !stringSliceContains(deepseek.NextWork, "mla_attention_variant") {
		t.Fatalf("deepseek gap = %+v, want MoE+MLA work", deepseek)
	}
	if _, ok := gaps["bert"]; ok {
		t.Fatalf("bert gap still reported after staged native loader: %+v", gaps["bert"])
	}
	if _, ok := gaps["bert_rerank"]; ok {
		t.Fatalf("bert_rerank gap still reported after staged native loader: %+v", gaps["bert_rerank"])
	}
}

func TestProductionLane_DefaultQuantizationPackLocks_Good(t *testing.T) {
	locks := DefaultProductionQuantizationPackLocks()
	if len(locks) != 3 {
		t.Fatalf("DefaultProductionQuantizationPackLocks() = %d locks, want q8 quality plus q6 default plus q4 constrained fallback", len(locks))
	}
	byBits := map[int]ProductionQuantizationPackLock{}
	for _, lock := range locks {
		byBits[lock.QuantBits] = lock
		if lock.BaseModelID != OfficialGemma4E2BTargetLock().ModelID || lock.SourceCheckedAt != "2026-05-31" {
			t.Fatalf("lock provenance = %+v, want official Google E2B source checked on 2026-05-31", lock)
		}
		if lock.BaseRevision != OfficialGemma4E2BTargetLock().Revision || lock.ConversionCommand == "" || lock.AccuracySmoke == "" {
			t.Fatalf("lock conversion record = %+v, want official base revision, conversion command, and accuracy-smoke status", lock)
		}
		if lock.Licence != "apache-2.0" || lock.LicenceURL != "https://ai.google.dev/gemma/docs/gemma_4_license" {
			t.Fatalf("lock licence = %+v, want Apache-2.0 Gemma 4 licence metadata", lock)
		}
		if lock.ConfigSHA256 == "" || lock.TokenizerSHA256 == "" || lock.TokenizerConfigSHA256 == "" || lock.SafetensorsIndexSHA256 == "" {
			t.Fatalf("lock hashes incomplete: %+v", lock)
		}
		if !lock.SafetensorsIndexPresent || len(lock.WeightFiles) == 0 {
			t.Fatalf("lock safetensors = present:%v files:%d, want indexed MLX quant pack", lock.SafetensorsIndexPresent, len(lock.WeightFiles))
		}
	}

	q8 := byBits[ProductionLaneQualityQuantBits]
	if q8.ModelID != "mlx-community/gemma-4-e2b-it-8bit" || q8.Revision != "48ef0737faea4e72556670e49da0ba421027a545" {
		t.Fatalf("q8 lock identity = %+v", q8)
	}
	if len(q8.WeightFiles) != 2 || q8.WeightFiles[0].Name != "model-00001-of-00002.safetensors" || q8.WeightFiles[1].Name != "model-00002-of-00002.safetensors" {
		t.Fatalf("q8 weights = %+v, want two locked shards", q8.WeightFiles)
	}

	q6 := byBits[ProductionLaneProductDefaultQuantBits]
	if q6.ModelID != ProductionLaneModelID || q6.Revision != "40d43b05f94ee798c0e40fe19fcd9ef49928486b" {
		t.Fatalf("q6 lock identity = %+v", q6)
	}
	if len(q6.WeightFiles) != 1 || q6.WeightFiles[0].Name != "model.safetensors" {
		t.Fatalf("q6 weights = %+v, want one locked safetensors file", q6.WeightFiles)
	}

	q4 := byBits[ProductionLaneConstrainedQuantBits]
	if q4.Name != "constrained" || q4.ModelID != ProductionLaneArchivedBaselineModelID || q4.Revision != "99d9a53ff828d365a8ecae538e45f80a08d612cd" {
		t.Fatalf("q4 lock identity = %+v", q4)
	}
	if q4.QuantGroup != 64 || q4.QuantMode != "affine" {
		t.Fatalf("q4 quantisation = group:%d mode:%q, want affine g64", q4.QuantGroup, q4.QuantMode)
	}
	if len(q4.WeightFiles) != 1 || q4.WeightFiles[0].Name != "model.safetensors" || q4.WeightFiles[0].Bytes != 3581101896 {
		t.Fatalf("q4 weights = %+v, want one locked safetensors fallback file", q4.WeightFiles)
	}
}

func TestProductionLane_SelectProductionQuantizationTier_Good(t *testing.T) {
	wide := memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 90 * memory.GiB}
	choice := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
	})
	if choice.Tier.Bits != 6 || choice.Tier.ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !choice.Fits {
		t.Fatalf("default wide choice = %+v, want fitting q6", choice)
	}

	quality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if quality.Tier.Bits != 8 || quality.Tier.ModelID != "mlx-community/gemma-4-e2b-it-8bit" || !quality.Fits {
		t.Fatalf("quality wide choice = %+v, want fitting q8", quality)
	}
	if quality.RequestedBits != 8 || quality.StepDownFromBits != 0 || quality.StepDownWorkingSetBytes != 0 || quality.StepDownRequiredWorkingSet != 0 {
		t.Fatalf("quality step-down evidence = %+v, want requested q8 with no step-down", quality)
	}

	qualityStepDown := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        memory.DeviceInfo{MemorySize: 64 * memory.GiB, MaxRecommendedWorkingSetSize: 48 * memory.GiB},
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if qualityStepDown.Tier.Bits != 6 || qualityStepDown.Tier.ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !qualityStepDown.Fits {
		t.Fatalf("quality step-down choice = %+v, want fitting q6", qualityStepDown)
	}
	if qualityStepDown.RequestedBits != 8 || qualityStepDown.StepDownFromBits != 8 || qualityStepDown.StepDownWorkingSetBytes != 48*memory.GiB || qualityStepDown.StepDownRequiredWorkingSet != 64*memory.GiB {
		t.Fatalf("quality step-down evidence = %+v, want q8 required 64GiB stepping down at 48GiB working set", qualityStepDown)
	}

	unknownQuality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		QualityFirst: true,
	})
	if unknownQuality.Tier.Bits != 6 || unknownQuality.RequestedBits != 8 || unknownQuality.StepDownFromBits != 8 || !unknownQuality.Fits {
		t.Fatalf("unknown-memory quality choice = %+v, want q6 default until q8 headroom is measured", unknownQuality)
	}
	if !core.Contains(unknownQuality.Reason, "measured memory headroom") {
		t.Fatalf("unknown-memory quality reason = %q, want measured-headroom explanation", unknownQuality.Reason)
	}

	constrained := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
		ContextLength: ProductionLaneLongContextLength,
	})
	if constrained.Tier.Bits != 4 || constrained.Tier.ModelID != "mlx-community/gemma-4-e2b-it-4bit" || !constrained.Fits {
		t.Fatalf("constrained long-context choice = %+v, want fitting q4 fallback", constrained)
	}
	if constrained.RequestedBits != 6 || constrained.StepDownFromBits != 6 || constrained.StepDownWorkingSetBytes != 13*memory.GiB || constrained.StepDownRequiredWorkingSet != 24*memory.GiB {
		t.Fatalf("constrained step-down evidence = %+v, want q6 required 24GiB stepping down at 13GiB working set", constrained)
	}

	forced := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:              wide,
		ContextLength:       ProductionLaneContextLength,
		ConstrainedFallback: true,
	})
	if forced.Tier.Bits != 4 || !forced.Tier.ConstrainedOnly {
		t.Fatalf("forced constrained choice = %+v, want q4 fallback", forced)
	}
	if forced.RequestedBits != 4 || forced.StepDownFromBits != 0 {
		t.Fatalf("forced constrained evidence = %+v, want requested q4 without step-down", forced)
	}
}

func TestProductionLane_ArchitectureProfileNative_Good(t *testing.T) {
	lane := DefaultProductionLane()
	prof, ok := profile.LookupArchitectureProfile(lane.Architecture)

	if !ok {
		t.Fatalf("profile.LookupArchitectureProfile(%q) = false", lane.Architecture)
	}
	if !prof.NativeRuntime || !prof.Generation || !prof.Chat {
		t.Fatalf("architecture profile = %+v, want native chat/generation runtime", prof)
	}
	if prof.ChatTemplate != lane.ChatTemplate {
		t.Fatalf("ChatTemplate = %q, want lane template %q", prof.ChatTemplate, lane.ChatTemplate)
	}
}

func TestProductionLane_DefaultGemma4FastRuntimeGates_Good(t *testing.T) {
	gates := DefaultGemma4FastRuntimeGates()
	seen := map[string]bool{}
	for _, gate := range gates {
		seen[gate] = true
	}

	if len(gates) != 1 || !seen[Gemma4FastRuntimeGateDirectGreedyToken] {
		t.Fatalf("DefaultGemma4FastRuntimeGates() = %v, want direct greedy promoted", gates)
	}
	if count := DefaultGemma4FastRuntimeGateCount(); count != len(gates) {
		t.Fatalf("DefaultGemma4FastRuntimeGateCount() = %d, want %d", count, len(gates))
	}
	for i, want := range gates {
		got, ok := DefaultGemma4FastRuntimeGate(i)
		if !ok || got != want {
			t.Fatalf("DefaultGemma4FastRuntimeGate(%d) = %q, %t; want %q, true", i, got, ok, want)
		}
	}
	if got, ok := DefaultGemma4FastRuntimeGate(-1); ok || got != "" {
		t.Fatalf("DefaultGemma4FastRuntimeGate(-1) = %q, %t; want empty false", got, ok)
	}
	if got, ok := DefaultGemma4FastRuntimeGate(len(gates)); ok || got != "" {
		t.Fatalf("DefaultGemma4FastRuntimeGate(len) = %q, %t; want empty false", got, ok)
	}
	for _, rejected := range []string{
		Gemma4FastRuntimeGateExpertIDMatVec,
		Gemma4FastRuntimeGateExpertIDFused,
		Gemma4FastRuntimeGateSortedExpertPrefill,
		Gemma4FastRuntimeGateNativeMLPMatVec,
		Gemma4FastRuntimeGateNativeLinearMatVec,
		Gemma4FastRuntimeGateNativeRouterMatVec,
		Gemma4FastRuntimeGateNativeRouterTopK,
		Gemma4FastRuntimeGateGenerationStream,
		Gemma4FastRuntimeGateFixedGemma4SharedMask,
		"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		Gemma4FastRuntimeGatePagedDecodeFastConcat,
		Gemma4FastRuntimeGateNativePagedAttention,
		Gemma4FastRuntimeGateFixedGemma4Cache,
		Gemma4FastRuntimeGateFixedGemma4Sliding,
		Gemma4FastRuntimeGateNativeFixedSliding,
		Gemma4FastRuntimeGateAsyncDecodePrefetch,
	} {
		if seen[rejected] {
			t.Fatalf("DefaultGemma4FastRuntimeGates() = %v, should exclude rejected gate %s", gates, rejected)
		}
	}
}

func TestProductionLane_DefaultMTPPolicy_OptInUntilRetainedBenchmarkWin_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	if policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID || policy.AssistantModelID != OfficialGemma4E2BAssistantLock().ModelID {
		t.Fatalf("policy identity = %+v, want official target+assistant IDs", policy)
	}
	if policy.EnabledByDefault {
		t.Fatalf("EnabledByDefault = true, want MTP opt-in until retained benchmark promotion")
	}
	if policy.DefaultDraftTokens != 2 || policy.MinimumRetainedTurns != 10 {
		t.Fatalf("policy defaults = draft:%d turns:%d, want draft=2 and retained 10-turn evidence", policy.DefaultDraftTokens, policy.MinimumRetainedTurns)
	}
	if !intSliceEqual(policy.RequiredDraftTokenSweeps, []int{1, 2, 4}) {
		t.Fatalf("RequiredDraftTokenSweeps = %v, want 1/2/4 sweep evidence", policy.RequiredDraftTokenSweeps)
	}
	if !policy.RequiresGreedyParity || !policy.RequiresRetainedWorkflow || policy.RequiresSideBySideBenchmark == false {
		t.Fatalf("policy requirements = %+v, want side-by-side retained greedy-parity benchmark", policy)
	}
	for _, metric := range []string{
		"target_only_visible_tokens_per_sec",
		"mtp_visible_tokens_per_sec",
		"target_only_input_output_tokens_per_sec",
		"mtp_input_output_tokens_per_sec",
		"mtp_target_tokens_per_sec",
		"mtp_warm_decode_tokens_per_sec",
		"target_only_wall_duration",
		"mtp_wall_duration",
		"target_only_first_token_duration",
		"mtp_first_token_duration",
		"target_only_restore_duration",
		"mtp_restore_duration",
		"target_only_peak_memory_bytes",
		"mtp_peak_memory_bytes",
		"target_only_active_plus_cache_memory_bytes",
		"mtp_active_plus_cache_memory_bytes",
		"target_only_energy_joules",
		"mtp_energy_joules",
		"estimated_power_watts",
		"same_load_policy",
		"target_only_cache_policy",
		"mtp_cache_policy",
		"target_only_cache_mode",
		"mtp_cache_mode",
		"target_only_context_length",
		"mtp_context_length",
		"mtp_observed_draft_token_sweeps",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"quality_flags",
		"assistant_architecture",
		"assistant_ordered_embeddings",
		"assistant_centroids",
		"assistant_centroid_intermediate_top_k",
		"assistant_four_layer_drafter",
		"assistant_token_ordering_dtype",
		"assistant_token_ordering_shape",
		"official_pair_verified",
		"official_target_model_id",
		"official_target_revision",
		"official_assistant_model_id",
		"official_assistant_revision",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsSlowerOrUnproven_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               95,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      11 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   110 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   4096,
		TargetOnlyActivePlusCacheMemoryBytes: 3072,
		MTPActivePlusCacheMemoryBytes:        3072,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      1000,
		EstimatedPowerWatts:                  100,
		MTPTargetTokensPerSec:                90,
		MTPWarmDecodeTokensPerSec:            94,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP rejected when slower than target-only", decision)
	}
	if !core.Contains(decision.Reason, "faster") {
		t.Fatalf("decision reason = %q, want faster-than-target-only failure", decision.Reason)
	}

	unproven := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              false,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        120,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     118,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if unproven.EnableByDefault || !core.Contains(unproven.Reason, "retained") {
		t.Fatalf("unproven decision = %+v, want retained-workflow gate", unproven)
	}

	missingOperationalEvidence := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        125,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     123,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPAcceptedTokens:             30,
		MTPRejectedTokens:             10,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if missingOperationalEvidence.EnableByDefault || !core.Contains(missingOperationalEvidence.Reason, "restore, memory, and energy") {
		t.Fatalf("missing operational evidence decision = %+v, want restore/memory/energy gate", missingOperationalEvidence)
	}

	missingActiveCacheEvidence := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        125,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     123,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		TargetOnlyRestoreDuration:     100 * time.Millisecond,
		MTPRestoreDuration:            80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:     4096,
		MTPPeakMemoryBytes:            3584,
		TargetOnlyEnergyJoules:        1000,
		MTPEnergyJoules:               760,
		EstimatedPowerWatts:           100,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPAcceptedTokens:             30,
		MTPRejectedTokens:             10,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if missingActiveCacheEvidence.EnableByDefault || !core.Contains(missingActiveCacheEvidence.Reason, "active+cache") {
		t.Fatalf("missing active+cache decision = %+v, want active+cache memory gate", missingActiveCacheEvidence)
	}

	missingDraftIdentity := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyFirstTokenDuration:         120 * time.Millisecond,
		MTPFirstTokenDuration:                90 * time.Millisecond,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingDraftIdentity.EnableByDefault || !core.Contains(missingDraftIdentity.Reason, "draft model") {
		t.Fatalf("missing draft identity decision = %+v, want draft model/schedule gate", missingDraftIdentity)
	}

	missingDraftSweep := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyFirstTokenDuration:         120 * time.Millisecond,
		MTPFirstTokenDuration:                90 * time.Millisecond,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{2},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingDraftSweep.EnableByDefault || !core.Contains(missingDraftSweep.Reason, "draft-token sweep") {
		t.Fatalf("missing draft-token sweep decision = %+v, want required 1/2/4 sweep gate", missingDraftSweep)
	}

	missingThroughputBreakdown := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingThroughputBreakdown.EnableByDefault || !core.Contains(missingThroughputBreakdown.Reason, "target-verify and warm-decode") {
		t.Fatalf("missing throughput breakdown decision = %+v, want target-verify/warm-decode gate", missingThroughputBreakdown)
	}

	missingAcceptanceAccounting := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyFirstTokenDuration:         120 * time.Millisecond,
		MTPFirstTokenDuration:                90 * time.Millisecond,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingAcceptanceAccounting.EnableByDefault || !core.Contains(missingAcceptanceAccounting.Reason, "accepted/rejected") {
		t.Fatalf("missing acceptance accounting decision = %+v, want accepted/rejected counter gate", missingAcceptanceAccounting)
	}

	missingDraftCalls := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
	})
	if missingDraftCalls.EnableByDefault || !core.Contains(missingDraftCalls.Reason, "draft-call") {
		t.Fatalf("missing draft-call decision = %+v, want draft-call counter gate", missingDraftCalls)
	}

	noAcceptedDraftTokens := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPRejectedTokens:                    40,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if noAcceptedDraftTokens.EnableByDefault || !core.Contains(noAcceptedDraftTokens.Reason, "accepted draft tokens") {
		t.Fatalf("zero accepted draft decision = %+v, want accepted-token gate", noAcceptedDraftTokens)
	}
}

func TestProductionLane_EvaluateMTPPromotion_AcceptsFasterGreedyParityEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyFirstTokenDuration:         120 * time.Millisecond,
		MTPFirstTokenDuration:                90 * time.Millisecond,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		AssistantFourLayerDrafter:            true,
		AssistantTokenOrderingDType:          "int64",
		AssistantTokenOrderingShape:          []int{2048, 128},
		OfficialPairVerified:                 true,
		OfficialTargetModelID:                OfficialGemma4E2BTargetLock().ModelID,
		OfficialTargetRevision:               OfficialGemma4E2BTargetLock().Revision,
		OfficialAssistantModelID:             OfficialGemma4E2BAssistantLock().ModelID,
		OfficialAssistantRevision:            OfficialGemma4E2BAssistantLock().Revision,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP promotion when retained wall and visible speed both win", decision)
	}
	if decision.WallSpeedup <= 1 || decision.VisibleSpeedup <= 1 {
		t.Fatalf("speedups = wall:%f visible:%f, want both > 1", decision.WallSpeedup, decision.VisibleSpeedup)
	}
	if decision.RestoreSpeedup <= 1 || decision.EnergySavings <= 0 {
		t.Fatalf("operational ratios = restore:%f energy:%f, want restore speedup and energy savings recorded", decision.RestoreSpeedup, decision.EnergySavings)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingFirstTokenEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()
	evidence := productionCombinedMTPPassEvidence(memory.KVCacheModePaged)
	evidence.TargetOnlyFirstTokenDuration = 0
	evidence.MTPFirstTokenDuration = 0

	decision := EvaluateProductionMTPPromotion(policy, evidence)

	if decision.EnableByDefault || !core.Contains(decision.Reason, "first-token") {
		t.Fatalf("decision = %+v, want first-token latency evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsUnverifiedOfficialPairEvidence_Bad(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		AssistantFourLayerDrafter:            true,
		AssistantTokenOrderingDType:          "int64",
		AssistantTokenOrderingShape:          []int{2048, 128},
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "official Gemma 4 target+assistant pair") {
		t.Fatalf("decision = %+v, want verified official pair evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingAssistantTokenOrderingEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		AssistantFourLayerDrafter:            true,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "token-ordering") {
		t.Fatalf("decision = %+v, want assistant token-ordering evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsWrongOfficialAssistantTokenOrderingEvidence_Bad(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	wrongShape := productionCombinedMTPPassEvidence(memory.KVCacheModePaged)
	wrongShape.AssistantTokenOrderingShape = []int{2048, 64}

	shapeDecision := EvaluateProductionMTPPromotion(policy, wrongShape)
	if shapeDecision.EnableByDefault || !core.Contains(shapeDecision.Reason, "token-ordering") {
		t.Fatalf("wrong shape decision = %+v, want official token-ordering shape gate", shapeDecision)
	}

	wrongDType := productionCombinedMTPPassEvidence(memory.KVCacheModePaged)
	wrongDType.AssistantTokenOrderingDType = "int32"

	dtypeDecision := EvaluateProductionMTPPromotion(policy, wrongDType)
	if dtypeDecision.EnableByDefault || !core.Contains(dtypeDecision.Reason, "token-ordering") {
		t.Fatalf("wrong dtype decision = %+v, want official token-ordering dtype gate", dtypeDecision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingAssistantLayoutEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "ordered-embedding evidence") {
		t.Fatalf("decision = %+v, want official assistant ordered-embedding evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingFourLayerAssistantEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "four-layer") {
		t.Fatalf("decision = %+v, want official assistant four-layer drafter evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingInputOutputEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "input+output") {
		t.Fatalf("decision = %+v, want input+output throughput evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingLoadPolicyEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "load policy") {
		t.Fatalf("decision = %+v, want load-policy evidence gate", decision)
	}
}

func TestProductionLane_DefaultTurboQuantPolicy_ResearchOptIn_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	if policy.CacheMode != memory.KVCacheModeTurboQuant || policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID {
		t.Fatalf("policy identity = %+v, want official target plus turboquant cache mode", policy)
	}
	if policy.EnabledByDefault {
		t.Fatalf("EnabledByDefault = true, want TurboQuant opt-in until retained workflow validation")
	}
	if policy.TargetEffectiveBitsMilli != 3500 {
		t.Fatalf("TargetEffectiveBitsMilli = %d, want 3500 for 3.5 bits/channel research target", policy.TargetEffectiveBitsMilli)
	}
	if policy.RequiredLayoutVersion != ProductionTurboQuantKVLayoutVersion ||
		policy.RequiredKeyAlgorithm != ProductionTurboQuantKeyAlgorithm ||
		policy.RequiredValueAlgorithm != ProductionTurboQuantValueAlgorithm ||
		policy.RequiredOutlierPolicy != ProductionTurboQuantOutlierPolicy {
		t.Fatalf("policy layout requirements = %+v, want TurboQuant KV v1 production layout", policy)
	}
	if !policy.RequiresQJLResidual || !policy.RequiresMetadataAccounting {
		t.Fatalf("policy QJL/metadata requirements = qjl:%v metadata:%v, want both required", policy.RequiresQJLResidual, policy.RequiresMetadataAccounting)
	}
	if !policy.RequiresExplicitOptIn || !policy.RequiresRetainedWorkflow || !policy.RequiresQualityParity ||
		!policy.RequiresSideBySideBenchmark || !policy.RequiresNormalContextValidation || !policy.RequiresStressContextValidation {
		t.Fatalf("policy requirements = %+v, want explicit retained-workflow quality-gated research mode", policy)
	}
	for _, mode := range []memory.KVCacheMode{
		memory.KVCacheModeFP16,
		memory.KVCacheModePaged,
		memory.KVCacheModeQ8,
		memory.KVCacheModeKQ8VQ4,
	} {
		if !kvCacheModeSliceContains(policy.CompareAgainstCacheModes, mode) {
			t.Fatalf("CompareAgainstCacheModes = %v, missing %q", policy.CompareAgainstCacheModes, mode)
		}
	}
	for _, metric := range []string{
		"baseline_cache_mode",
		"candidate_cache_mode",
		"candidate_layout_version",
		"candidate_key_algorithm",
		"candidate_value_algorithm",
		"candidate_outlier_policy",
		"candidate_effective_bits_milli",
		"candidate_qjl_residual",
		"candidate_metadata_bytes",
		"same_load_policy",
		"baseline_cache_policy",
		"candidate_cache_policy",
		"baseline_context_length",
		"candidate_context_length",
		"normal_context_validated",
		"stress_context_validated",
		"candidate_peak_memory_bytes",
		"baseline_peak_memory_bytes",
		"candidate_wall_duration",
		"baseline_wall_duration",
		"candidate_restore_duration",
		"baseline_restore_duration",
		"candidate_visible_tokens_per_sec",
		"baseline_visible_tokens_per_sec",
		"candidate_input_output_tokens_per_sec",
		"baseline_input_output_tokens_per_sec",
		"candidate_energy_joules",
		"baseline_energy_joules",
		"estimated_power_watts",
		"quality_flags",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsIncompleteValidation_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:             true,
		Turns:                        ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:               true,
		BaselineCacheMode:            memory.KVCacheModePaged,
		CandidateCacheMode:           memory.KVCacheModeTurboQuant,
		ComparedCacheModes:           policy.CompareAgainstCacheModes,
		NormalContextValidated:       true,
		StressContextValidated:       false,
		BaselineWallDuration:         10 * time.Second,
		CandidateWallDuration:        8 * time.Second,
		BaselinePeakMemoryBytes:      10 * memory.GiB,
		CandidatePeakMemoryBytes:     7 * memory.GiB,
		BaselineEnergyJoules:         1000,
		CandidateEnergyJoules:        800,
		EstimatedPowerWatts:          100,
		BaselineRestoreDuration:      100 * time.Millisecond,
		CandidateRestoreDuration:     80 * time.Millisecond,
		BaselineVisibleTokensPerSec:  80,
		CandidateVisibleTokensPerSec: 80,
	})

	if decision.ProductionCandidate {
		t.Fatalf("decision = %+v, want rejection until 100k stress lane is validated", decision)
	}
	if !core.Contains(decision.Reason, "stress") {
		t.Fatalf("decision reason = %q, want stress-context validation failure", decision.Reason)
	}

	missingBaselineMode := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:             true,
		Turns:                        ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:               true,
		CandidateCacheMode:           memory.KVCacheModeTurboQuant,
		ComparedCacheModes:           policy.CompareAgainstCacheModes,
		NormalContextValidated:       true,
		StressContextValidated:       true,
		BaselineWallDuration:         10 * time.Second,
		CandidateWallDuration:        8 * time.Second,
		BaselinePeakMemoryBytes:      10 * memory.GiB,
		CandidatePeakMemoryBytes:     7 * memory.GiB,
		BaselineEnergyJoules:         1000,
		CandidateEnergyJoules:        800,
		EstimatedPowerWatts:          100,
		BaselineRestoreDuration:      100 * time.Millisecond,
		CandidateRestoreDuration:     80 * time.Millisecond,
		BaselineVisibleTokensPerSec:  80,
		CandidateVisibleTokensPerSec: 80,
	})
	if missingBaselineMode.ProductionCandidate || !core.Contains(missingBaselineMode.Reason, "baseline cache mode") {
		t.Fatalf("missing baseline mode decision = %+v, want baseline cache mode gate", missingBaselineMode)
	}

	missingVisibleThroughput := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
	})
	if missingVisibleThroughput.ProductionCandidate || !core.Contains(missingVisibleThroughput.Reason, "visible throughput") {
		t.Fatalf("missing visible throughput decision = %+v, want visible-throughput gate", missingVisibleThroughput)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_AllowsMeasuredCandidate_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, productionTurboQuantMeasuredCandidateEvidence(policy))

	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v, want TurboQuant production candidate after full retained validation", decision)
	}
	if decision.EnableByDefault {
		t.Fatalf("EnableByDefault = true, want TurboQuant still explicit/non-default after candidate promotion")
	}
	if decision.WallSpeedup <= 1 || decision.MemorySavingsRatio <= 0 || decision.EnergySavingsRatio <= 0 {
		t.Fatalf("decision metrics = %+v, want wall, memory, and energy savings recorded", decision)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsMissingLayoutEvidence_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()
	evidence := productionTurboQuantMeasuredCandidateEvidence(policy)
	evidence.CandidateLayoutVersion = ""

	decision := EvaluateProductionTurboQuantPromotion(policy, evidence)

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "layout version evidence") {
		t.Fatalf("decision = %+v, want TurboQuant layout evidence gate", decision)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsNoActiveCacheMemoryWin_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  5 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 6 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
		BaselineInputOutputTokensPerSec:     33000,
		CandidateInputOutputTokensPerSec:    36000,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "active+cache memory savings") {
		t.Fatalf("decision = %+v, want active+cache memory-savings gate", decision)
	}
	if decision.MemorySavingsRatio != 0 {
		t.Fatalf("memory savings ratio = %f, want no active+cache savings recorded", decision.MemorySavingsRatio)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsMissingInputOutputEvidence_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "input+output") {
		t.Fatalf("decision = %+v, want input+output throughput evidence gate", decision)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsMissingLoadPolicyEvidence_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "load policy") {
		t.Fatalf("decision = %+v, want load-policy evidence gate", decision)
	}
}

func productionTurboQuantMeasuredCandidateEvidence(policy ProductionTurboQuantPolicy) ProductionTurboQuantPromotionEvidence {
	return ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		CandidateLayoutVersion:              policy.RequiredLayoutVersion,
		CandidateKeyAlgorithm:               policy.RequiredKeyAlgorithm,
		CandidateValueAlgorithm:             policy.RequiredValueAlgorithm,
		CandidateOutlierPolicy:              policy.RequiredOutlierPolicy,
		CandidateEffectiveBitsMilli:         policy.TargetEffectiveBitsMilli,
		CandidateQJLResidual:                true,
		CandidateMetadataBytes:              64 * 1024,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
		BaselineInputOutputTokensPerSec:     33000,
		CandidateInputOutputTokensPerSec:    36000,
	}
}

func kvCacheModeSliceContains(values []memory.KVCacheMode, needle memory.KVCacheMode) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func intSliceEqual(values, want []int) bool {
	if len(values) != len(want) {
		return false
	}
	for i, value := range values {
		if value != want[i] {
			return false
		}
	}
	return true
}
