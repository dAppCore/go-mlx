// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

func TestApiCommon_KVSnapshot_Head_Good(t *testing.T) {
	snapshot := &kv.Snapshot{
		Layers: []kv.LayerSnapshot{{
			Layer: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	head, ok := snapshot.Head(0, 0)
	if !ok {
		t.Fatal("Head() ok = false, want true")
	}
	if len(head.Key) != 2 || head.Key[0] != 1 || head.Value[1] != 4 {
		t.Fatalf("Head() = %+v, want copied key/value data", head)
	}
	head.Key[0] = 99
	if snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Head() returned aliased key data")
	}
}

func TestApiCommon_KVSnapshot_Head_Bad(t *testing.T) {
	snapshot := &kv.Snapshot{}

	_, ok := snapshot.Head(0, 0)

	if ok {
		t.Fatal("Head() ok = true, want false for missing layer")
	}
}

func TestApiCommon_KVSnapshot_SaveLoad_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "sample.kvbin")
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{10, 20, 30},
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4, 5, 6},
				Value: []float32{7, 8, 9, 10, 11, 12},
			}},
		}},
	}

	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}

	if loaded.Architecture != "gemma4_text" || loaded.SeqLen != 3 || loaded.HeadDim != 2 {
		t.Fatalf("loaded metadata = %+v", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head() ok = false, want true")
	}
	if len(head.Key) != 6 || head.Key[5] != 6 || head.Value[0] != 7 {
		t.Fatalf("loaded head = %+v", head)
	}
}

func TestApiCommon_DefaultLoadConfig_LocalRunnerDefaults_Good(t *testing.T) {
	cfg := DefaultLoadConfig()
	if cfg.ContextLength != 0 {
		t.Fatalf("ContextLength = %d, want model-native default 0", cfg.ContextLength)
	}
	if cfg.ParallelSlots != DefaultLocalParallelSlots {
		t.Fatalf("ParallelSlots = %d, want %d", cfg.ParallelSlots, DefaultLocalParallelSlots)
	}
	if !cfg.PromptCache {
		t.Fatal("PromptCache = false, want true")
	}
	if cfg.PromptCacheMinTokens != DefaultPromptCacheMinTokens {
		t.Fatalf("PromptCacheMinTokens = %d, want %d", cfg.PromptCacheMinTokens, DefaultPromptCacheMinTokens)
	}
}

func TestApiCommon_WithParallelSlots_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithParallelSlots(4)})
	if cfg.ParallelSlots != 4 {
		t.Fatalf("ParallelSlots = %d, want 4", cfg.ParallelSlots)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativeParallelSlots_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{ParallelSlots: -1})
	if err == nil {
		t.Fatal("expected negative parallel slots error")
	}
}

func TestApiCommon_WithPromptCache_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPromptCache(false)})
	if cfg.PromptCache {
		t.Fatal("PromptCache = true, want false")
	}
}

func TestApiCommon_WithPromptCacheMinTokens_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPromptCacheMinTokens(8192)})
	if cfg.PromptCacheMinTokens != 8192 {
		t.Fatalf("PromptCacheMinTokens = %d, want 8192", cfg.PromptCacheMinTokens)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativePromptCacheMinTokens_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{PromptCacheMinTokens: -1})
	if err == nil {
		t.Fatal("expected negative prompt cache min tokens error")
	}
}

func TestApiCommon_WithMemoryPlannerLoadOptions_Good(t *testing.T) {
	plan := memory.Plan{ContextLength: 8192, CachePolicy: memory.KVCacheRotating, CacheMode: memory.KVCacheModeQ8}
	split := inference.SplitInferencePlan{
		Mode:       inference.SplitInferenceModeLocal,
		LocalSlice: inference.ModelSlicePlan{Preset: inference.ModelSlicePresetFull},
	}
	cfg := applyLoadOptions([]LoadOption{
		WithAutoMemoryPlan(false),
		WithMemoryPlan(plan),
		WithCachePolicy(memory.KVCacheFull),
		WithKVCacheMode(memory.KVCacheModeKQ8VQ4),
		WithBatchSize(3),
		WithPrefillChunkSize(256),
		WithAllocatorLimits(10, 3, 7),
		WithSplitInference(split),
	})
	if cfg.AutoMemoryPlan {
		t.Fatal("AutoMemoryPlan = true, want false")
	}
	if cfg.MemoryPlan == nil || cfg.MemoryPlan.ContextLength != 8192 {
		t.Fatalf("memory.Plan = %+v, want explicit plan", cfg.MemoryPlan)
	}
	if cfg.CachePolicy != memory.KVCacheFull || cfg.CacheMode != memory.KVCacheModeKQ8VQ4 || cfg.BatchSize != 3 || cfg.PrefillChunkSize != 256 {
		t.Fatalf("planner shape = policy %q mode %q batch %d prefill %d", cfg.CachePolicy, cfg.CacheMode, cfg.BatchSize, cfg.PrefillChunkSize)
	}
	if cfg.MemoryLimitBytes != 10 || cfg.CacheLimitBytes != 3 || cfg.WiredLimitBytes != 7 {
		t.Fatalf("limits = %d/%d/%d, want 10/3/7", cfg.MemoryLimitBytes, cfg.CacheLimitBytes, cfg.WiredLimitBytes)
	}
	if cfg.SplitInference == nil || cfg.SplitInference.Mode != inference.SplitInferenceModeLocal {
		t.Fatalf("SplitInference = %+v, want cloned local plan", cfg.SplitInference)
	}
	split.Mode = inference.SplitInferenceModeRemoteFFN
	if cfg.SplitInference.Mode != inference.SplitInferenceModeLocal {
		t.Fatalf("WithSplitInference leaked caller mutation: %+v", cfg.SplitInference)
	}
}

func TestApiCommon_WithKVCacheMode_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithKVCacheMode(memory.KVCacheModeQ8)})
	if cfg.CacheMode != memory.KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", cfg.CacheMode, memory.KVCacheModeQ8)
	}
}

func TestApiCommon_WithKVCacheStorageDType_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithKVCacheStorageDType("fp16")})
	if cfg.KVCacheStorageDType != "fp16" {
		t.Fatalf("KVCacheStorageDType = %q, want fp16", cfg.KVCacheStorageDType)
	}
}

func TestApiCommon_WithPagedKVPageSize_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPagedKVPageSize(1024)})
	if cfg.PagedKVPageSize != 1024 {
		t.Fatalf("PagedKVPageSize = %d, want 1024", cfg.PagedKVPageSize)
	}
}

func TestApiCommon_WithPagedKVPrealloc_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPagedKVPrealloc(true)})
	if !cfg.PagedKVPrealloc {
		t.Fatal("PagedKVPrealloc = false, want true")
	}
}

func TestApiCommon_WithFixedSlidingCacheSize_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithFixedSlidingCacheSize(2048)})
	if cfg.FixedSlidingCacheSize != 2048 {
		t.Fatalf("FixedSlidingCacheSize = %d, want 2048", cfg.FixedSlidingCacheSize)
	}
}

func TestApiCommon_NormalizeLoadConfig_AcceptsTurboQuantResearchMode_Good(t *testing.T) {
	cfg, err := normalizeLoadConfig(LoadConfig{CacheMode: memory.KVCacheModeTurboQuant})
	if err != nil {
		t.Fatalf("normalizeLoadConfig(turboquant) error = %v, want nil", err)
	}
	if cfg.CacheMode != memory.KVCacheModeTurboQuant {
		t.Fatalf("CacheMode = %q, want turboquant", cfg.CacheMode)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativePlannerShape_Bad(t *testing.T) {
	if _, err := normalizeLoadConfig(LoadConfig{BatchSize: -1}); err == nil {
		t.Fatal("expected negative batch size error")
	}
	if _, err := normalizeLoadConfig(LoadConfig{PrefillChunkSize: -1}); err == nil {
		t.Fatal("expected negative prefill chunk size error")
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsRemoteSplit_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{
		SplitInference: &inference.SplitInferencePlan{
			Mode: inference.SplitInferenceModeRemoteFFN,
			LocalSlice: inference.ModelSlicePlan{
				Preset:     inference.ModelSlicePresetClient,
				Components: []inference.ModelComponent{inference.ModelComponentAttention},
			},
			Endpoints: []inference.SplitEndpoint{{
				ID:   "ffn-0",
				Role: inference.SplitEndpointRoleFFN,
			}},
		},
	})
	if err == nil {
		t.Fatal("expected remote split execution error")
	}
	if !core.Contains(err.Error(), "split inference execution is planned") {
		t.Fatalf("error = %v, want split execution planned message", err)
	}
}

func TestApiCommon_WithMemoryPlan_ClonesPlan_Ugly(t *testing.T) {
	plan := memory.Plan{ContextLength: 8192}
	cfg := applyLoadOptions([]LoadOption{WithMemoryPlan(plan)})
	plan.ContextLength = 4096
	if cfg.MemoryPlan == nil || cfg.MemoryPlan.ContextLength != 8192 {
		t.Fatalf("memory.Plan = %+v, want cloned 8192 plan", cfg.MemoryPlan)
	}
}
func TestAPIGenerateOptions_Good(t *testing.T) {
	cfg := spine.ApplyGenerateOptions([]GenerateOption{
		WithMaxTokens(64),
		WithTemperature(0.7),
		WithTopK(20),
		WithTopP(0.9),
		WithMinP(0.05),
		WithSeed(42),
		WithLogits(),
		WithReturnLogits(),
		WithStopTokens(1, 2),
		WithMinTokensBeforeStop(1),
		WithRepeatPenalty(1.1),
		WithGenerationClearCache(),
		WithGenerationClearCacheInterval(64),
		WithTokenPhaseTrace(),
		WithTokenPhaseTraceText(),
	})
	if cfg.MaxTokens != 64 || cfg.Temperature != 0.7 || cfg.TopK != 20 || cfg.TopP != 0.9 || cfg.MinP != 0.05 {
		t.Fatalf("unexpected generate config: %+v", cfg)
	}
	if !cfg.SeedSet || cfg.Seed != 42 {
		t.Fatalf("seed config = %d/%v, want 42/true", cfg.Seed, cfg.SeedSet)
	}
	if !cfg.ReturnLogits {
		t.Fatal("ReturnLogits = false, want true")
	}
	if !reflect.DeepEqual(cfg.StopTokens, []int32{1, 2}) {
		t.Fatalf("stop tokens = %v", cfg.StopTokens)
	}
	if cfg.MinTokensBeforeStop != 1 {
		t.Fatalf("MinTokensBeforeStop = %d, want 1", cfg.MinTokensBeforeStop)
	}
	if cfg.RepeatPenalty != 1.1 {
		t.Fatalf("repeat penalty = %f, want 1.1", cfg.RepeatPenalty)
	}
	if !cfg.GenerationClearCache || cfg.GenerationClearCacheInterval != 64 {
		t.Fatalf("GenerationClearCache = %v/%d, want true/64", cfg.GenerationClearCache, cfg.GenerationClearCacheInterval)
	}
	if !cfg.TraceTokenPhases {
		t.Fatal("TraceTokenPhases = false, want true")
	}
	if !cfg.TraceTokenText {
		t.Fatal("TraceTokenText = false, want true")
	}
}

func TestAPILoadOptions_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{
		WithContextLength(8192),
		WithParallelSlots(4),
		WithPromptCache(false),
		WithPromptCacheMinTokens(4096),
		WithQuantization(4),
		WithExpectedQuantization(4),
		WithDevice("cpu"),
		WithAdapterPath("/models/lora/demo"),
	})
	if cfg.ContextLength != 8192 || cfg.ParallelSlots != 4 || cfg.PromptCache || cfg.PromptCacheMinTokens != 4096 || cfg.Quantization != 4 || cfg.ExpectedQuantization != 4 || cfg.Device != "cpu" || cfg.AdapterPath != "/models/lora/demo" {
		t.Fatalf("unexpected load config: %+v", cfg)
	}
}

func TestAPIKVHeadDTypeAndChunkStringHelpers_Good(t *testing.T) {
	if kvconv.RootKVHeadDType(metal.DTypeFloat16, []byte{1}) != "float16" {
		t.Fatal("kvconv.RootKVHeadDType(float16) did not preserve dtype")
	}
	if kvconv.RootKVHeadDType(metal.DTypeFloat32, nil) != "" || kvconv.RootKVHeadDType(metal.DTypeInt8, []byte{1}) != "" {
		t.Fatal("kvconv.RootKVHeadDType should reject empty raw data and unsupported dtype")
	}
	if kvconv.MetalKVHeadDType("F32", []byte{1}) != metal.DTypeFloat32 || kvconv.MetalKVHeadDType("BF16", []byte{1}) != metal.DTypeBFloat16 {
		t.Fatal("kvconv.MetalKVHeadDType aliases did not map to metal dtypes")
	}
	if kvconv.MetalKVHeadDType("bad", []byte{1}) != 0 || kvconv.MetalKVHeadDType("float16", nil) != 0 {
		t.Fatal("kvconv.MetalKVHeadDType should reject empty raw data and unsupported dtype")
	}
}
