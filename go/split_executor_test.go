// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
)

func TestSplitExecutor_LoadSplitExecutor_GoodClientRequiresFFN(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	executor, err := LoadSplitExecutor(context.Background(), slicePath)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	plan := executor.Placement()
	if plan.Ready {
		t.Fatalf("placement = %+v, want not ready without FFN executor", plan)
	}
	if !plan.Requires(inference.ModelComponentFFN) {
		t.Fatalf("placement = %+v, want FFN requirement", plan)
	}
	if plan.LocalTensorBytes != 16 || plan.OffloadTensorBytes != 8 {
		t.Fatalf("placement bytes = local:%d offload:%d, want 16/8", plan.LocalTensorBytes, plan.OffloadTensorBytes)
	}

	_, err = executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 1})
	if err == nil || !core.Contains(err.Error(), "requires an FFN executor") {
		t.Fatalf("Generate error = %v, want FFN executor requirement", err)
	}
}

func TestSplitExecutor_LoadSplitExecutor_GoodClientWithFFNPlacementReady(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	executor, err := LoadSplitExecutor(context.Background(), slicePath, WithSplitFFNExecutor(splitExecutorTestFFN{}))
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	plan := executor.Placement()
	if !plan.Ready {
		t.Fatalf("placement = %+v, want ready with FFN executor", plan)
	}
	if !plan.Requires(inference.ModelComponentFFN) {
		t.Fatalf("placement = %+v, want FFN requirement to remain visible", plan)
	}

	_, err = executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 1})
	if err == nil || !core.Contains(err.Error(), "local attention execution is not wired") {
		t.Fatalf("Generate error = %v, want local-attention boundary", err)
	}
}

func TestSplitExecutor_Generate_GoodRoutesAttentionAndFFNPerLayer(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	local := &splitExecutorTestLocalRuntime{
		prefill: SplitPrefillResult{
			Tokens: []int32{11, 12},
			Hidden: []float32{1},
			Layers: 2,
		},
		samples: []SplitSampleResult{{TokenID: 42}},
		text:    map[int32]string{42: " answer"},
	}
	ffn := &splitExecutorRecordingFFN{}
	executor, err := LoadSplitExecutor(
		context.Background(),
		slicePath,
		WithSplitLocalRuntime(local),
		WithSplitFFNExecutor(ffn),
	)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	got, err := executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 1})

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != " answer" {
		t.Fatalf("Generate = %q, want token text", got)
	}
	if len(local.prefillPrompts) != 1 || local.prefillPrompts[0] != "hi" {
		t.Fatalf("prefill prompts = %v, want hi", local.prefillPrompts)
	}
	if !equalIntSlices(local.attentionLayers, []int{0, 1}) {
		t.Fatalf("attention layers = %v, want [0 1]", local.attentionLayers)
	}
	if !equalIntSlices(ffn.layers, []int{0, 1}) {
		t.Fatalf("ffn layers = %v, want [0 1]", ffn.layers)
	}
	if len(local.sampleHidden) != 1 || local.sampleHidden[0] != 23 {
		t.Fatalf("sample hidden = %v, want final FFN hidden [23]", local.sampleHidden)
	}
}

func TestSplitExecutor_Generate_GoodUsesSampleHiddenForNextStep(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	local := &splitExecutorTestLocalRuntime{
		prefill: SplitPrefillResult{
			Tokens: []int32{11},
			Hidden: []float32{1},
			Layers: 1,
		},
		samples: []SplitSampleResult{
			{TokenID: 42, Hidden: []float32{100}},
			{TokenID: 43},
		},
		text: map[int32]string{42: " first", 43: " second"},
	}
	ffn := &splitExecutorRecordingFFN{}
	executor, err := LoadSplitExecutor(
		context.Background(),
		slicePath,
		WithSplitLocalRuntime(local),
		WithSplitFFNExecutor(ffn),
	)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	got, err := executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 2})

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != " first second" {
		t.Fatalf("Generate = %q, want both decoded tokens", got)
	}
	if len(local.sampleHidden) != 1 || local.sampleHidden[0] != 111 {
		t.Fatalf("second sample hidden = %v, want next-token hidden to feed step 1", local.sampleHidden)
	}
}

func TestSplitExecutor_Generate_GoodRecordsMetricsMemoryAndPower(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	local := &splitExecutorTestLocalRuntime{
		prefill: SplitPrefillResult{
			Tokens: []int32{11, 12},
			Hidden: []float32{1},
			Layers: 1,
		},
		samples: []SplitSampleResult{
			{TokenID: 42},
			{TokenID: 43},
		},
		text: map[int32]string{42: " answer", 43: " done"},
	}
	ffn := &splitExecutorMetricsFFN{
		report: CPUSplitFFNMemoryReport{
			LoadedLayers:      1,
			ResidentBytes:     1024,
			PeakResidentBytes: 2048,
		},
	}
	power := &splitExecutorTestPowerMeter{watts: []float64{1, 2, 4, 3}}
	executor, err := LoadSplitExecutor(
		context.Background(),
		slicePath,
		WithSplitLocalRuntime(local),
		WithSplitFFNExecutor(ffn),
		WithSplitPowerMeter(power),
	)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	got, err := executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 2})

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != " answer done" {
		t.Fatalf("Generate = %q, want two decoded tokens", got)
	}
	metrics := executor.Metrics()
	if metrics.PromptTokens != 2 || metrics.GeneratedTokens != 2 {
		t.Fatalf("Metrics tokens = %+v, want prompt=2 generated=2", metrics)
	}
	if metrics.PrefillDuration <= 0 || metrics.DecodeDuration <= 0 || metrics.TotalDuration <= 0 || metrics.FirstTokenDuration <= 0 {
		t.Fatalf("Metrics durations = %+v, want non-zero timings", metrics)
	}
	if metrics.PrefillTokensPerSec <= 0 || metrics.DecodeTokensPerSec <= 0 {
		t.Fatalf("Metrics throughput = %+v, want tok/s values", metrics)
	}
	if metrics.CPUFFNMemory == nil || metrics.CPUFFNMemory.PeakResidentBytes != 2048 {
		t.Fatalf("Metrics CPU FFN memory = %+v, want peak resident bytes", metrics.CPUFFNMemory)
	}
	if !metrics.Power.Available || metrics.Power.SampleCount != 4 || metrics.Power.PeakWatts != 4 {
		t.Fatalf("Metrics power = %+v, want four samples with 4W peak", metrics.Power)
	}
	if !equalSplitStringSlices(power.phases, []string{"start", "prefill", "first_token", "complete"}) {
		t.Fatalf("power phases = %v, want start/prefill/first_token/complete", power.phases)
	}
}

func TestSplitExecutor_LoadSplitExecutor_GoodNativeLocalRuntimeOptionLoadsSlice(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	originalLoadNativeSplitLocalRuntime := loadNativeSplitLocalRuntime
	t.Cleanup(func() { loadNativeSplitLocalRuntime = originalLoadNativeSplitLocalRuntime })
	var gotPath string
	local := &splitExecutorTestLocalRuntime{
		prefill: SplitPrefillResult{
			Tokens: []int32{1},
			Hidden: []float32{1},
			Layers: 1,
		},
		samples: []SplitSampleResult{{TokenID: 7}},
		text:    map[int32]string{7: " native"},
	}
	loadNativeSplitLocalRuntime = func(_ context.Context, path string, cfg LoadConfig) (SplitLocalRuntime, error) {
		gotPath = path
		if cfg.ContextLength != 64 {
			t.Fatalf("native local runtime config = %+v, want context length 64", cfg)
		}
		return local, nil
	}

	executor, err := LoadSplitExecutor(
		context.Background(),
		slicePath,
		WithNativeSplitLocalRuntime(WithContextLength(64)),
		WithSplitFFNExecutor(splitExecutorTestFFN{}),
	)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}
	got, err := executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 1})

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if gotPath != slicePath {
		t.Fatalf("native local runtime path = %q, want %q", gotPath, slicePath)
	}
	if got != " native" {
		t.Fatalf("Generate = %q, want native token text", got)
	}
}

func TestNativeSplitLocalRuntime_DecodeTokenGood(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	runtime, err := LoadNativeSplitLocalRuntime(context.Background(), slicePath, LoadConfig{ContextLength: 32})
	if err != nil {
		t.Fatalf("LoadNativeSplitLocalRuntime: %v", err)
	}

	text, err := runtime.DecodeToken(context.Background(), 0)
	if err != nil {
		t.Fatalf("DecodeToken: %v", err)
	}
	if text != "a" {
		t.Fatalf("DecodeToken = %q, want tokenizer text", text)
	}
}

func TestNativeSplitLocalRuntime_PrefillGoodUsesNativeSplitModel(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	originalLoadNativeSplitModel := loadNativeSplitModel
	t.Cleanup(func() { loadNativeSplitModel = originalLoadNativeSplitModel })
	model := &splitNativeTestModel{
		prefill: &metal.SplitState{
			Tokens:      []int32{0},
			Hidden:      []float32{1, 2},
			HiddenShape: []int32{1, 1, 2},
			Layers:      1,
		},
	}
	loadNativeSplitModel = func(path string, cfg metal.LoadConfig) (nativeSplitModel, error) {
		if path != slicePath {
			t.Fatalf("load path = %q, want %q", path, slicePath)
		}
		if cfg.ContextLen != 32 {
			t.Fatalf("load config = %+v, want context length 32", cfg)
		}
		return model, nil
	}
	runtime, err := LoadNativeSplitLocalRuntime(context.Background(), slicePath, LoadConfig{ContextLength: 32})
	if err != nil {
		t.Fatalf("LoadNativeSplitLocalRuntime: %v", err)
	}

	state, err := runtime.Prefill(context.Background(), SplitPrefillRequest{Prompt: "a"})

	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if len(model.prefillPrompts) != 1 || model.prefillPrompts[0] != "a" {
		t.Fatalf("prefill prompts = %v, want [a]", model.prefillPrompts)
	}
	if state.Layers != 1 || len(state.Hidden) != 2 || state.Hidden[0] != 1 || state.Hidden[1] != 2 {
		t.Fatalf("prefill state = %+v, want native hidden", state)
	}
}

func TestNativeSplitLocalRuntime_SampleGoodUsesNativeSplitModel(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	originalLoadNativeSplitModel := loadNativeSplitModel
	t.Cleanup(func() { loadNativeSplitModel = originalLoadNativeSplitModel })
	model := &splitNativeTestModel{
		prefill: &metal.SplitState{
			Tokens:      []int32{0},
			Hidden:      []float32{1, 2},
			HiddenShape: []int32{1, 1, 2},
			Layers:      1,
		},
		sample: metal.SplitSampleResult{
			TokenID:     1,
			Hidden:      []float32{3, 4},
			HiddenShape: []int32{1, 1, 2},
		},
	}
	loadNativeSplitModel = func(string, metal.LoadConfig) (nativeSplitModel, error) {
		return model, nil
	}
	runtime, err := LoadNativeSplitLocalRuntime(context.Background(), slicePath, LoadConfig{ContextLength: 32})
	if err != nil {
		t.Fatalf("LoadNativeSplitLocalRuntime: %v", err)
	}
	if _, err := runtime.Prefill(context.Background(), SplitPrefillRequest{Prompt: "a"}); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	sample, err := runtime.Sample(context.Background(), SplitSampleRequest{
		Step:   0,
		Tokens: []int32{0},
		Hidden: []float32{9, 8},
		Config: GenerateConfig{Temperature: 0, TopK: 1},
	})

	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	if sample.TokenID != 1 || len(sample.Hidden) != 2 || sample.Hidden[0] != 3 || sample.Hidden[1] != 4 {
		t.Fatalf("sample = %+v, want native token and next hidden", sample)
	}
	if len(model.sampleRequests) != 1 {
		t.Fatalf("sample requests = %d, want 1", len(model.sampleRequests))
	}
	req := model.sampleRequests[0]
	if req.Config.TopK != 1 || req.Config.Temperature != 0 {
		t.Fatalf("sample config = %+v, want root config mapped", req.Config)
	}
	if !equalSplitFloat32Slices(req.Hidden, []float32{9, 8}) {
		t.Fatalf("sample hidden = %v, want request hidden", req.Hidden)
	}
}

type splitExecutorTestFFN struct{}

func (splitExecutorTestFFN) ForwardFFN(_ context.Context, req SplitFFNRequest) (SplitFFNResult, error) {
	return SplitFFNResult{Hidden: append([]float32(nil), req.Hidden...)}, nil
}

type splitExecutorRecordingFFN struct {
	layers []int
}

func (ffn *splitExecutorRecordingFFN) ForwardFFN(_ context.Context, req SplitFFNRequest) (SplitFFNResult, error) {
	ffn.layers = append(ffn.layers, req.Layer)
	return SplitFFNResult{Hidden: []float32{req.Hidden[0] + 10}}, nil
}

type splitExecutorMetricsFFN struct {
	layers []int
	report CPUSplitFFNMemoryReport
}

func (ffn *splitExecutorMetricsFFN) ForwardFFN(_ context.Context, req SplitFFNRequest) (SplitFFNResult, error) {
	ffn.layers = append(ffn.layers, req.Layer)
	return SplitFFNResult{Hidden: []float32{req.Hidden[0] + 10}}, nil
}

func (ffn *splitExecutorMetricsFFN) MemoryReport() CPUSplitFFNMemoryReport {
	report := ffn.report
	report.LayerLoads = len(ffn.layers)
	return report
}

type splitExecutorTestPowerMeter struct {
	watts  []float64
	phases []string
	index  int
}

func (meter *splitExecutorTestPowerMeter) SampleSplitPower(_ context.Context, phase string) (SplitPowerSample, error) {
	meter.phases = append(meter.phases, phase)
	watts := float64(1)
	if meter.index < len(meter.watts) {
		watts = meter.watts[meter.index]
	}
	meter.index++
	return SplitPowerSample{Watts: watts, Source: "test"}, nil
}

type splitExecutorTestLocalRuntime struct {
	prefill         SplitPrefillResult
	samples         []SplitSampleResult
	text            map[int32]string
	prefillPrompts  []string
	attentionLayers []int
	sampleHidden    []float32
}

func (runtime *splitExecutorTestLocalRuntime) Prefill(_ context.Context, req SplitPrefillRequest) (SplitPrefillResult, error) {
	runtime.prefillPrompts = append(runtime.prefillPrompts, req.Prompt)
	return runtime.prefill, nil
}

func (runtime *splitExecutorTestLocalRuntime) ForwardAttention(_ context.Context, req SplitAttentionRequest) (SplitAttentionResult, error) {
	runtime.attentionLayers = append(runtime.attentionLayers, req.Layer)
	return SplitAttentionResult{Hidden: []float32{req.Hidden[0] + 1}}, nil
}

func (runtime *splitExecutorTestLocalRuntime) Sample(_ context.Context, req SplitSampleRequest) (SplitSampleResult, error) {
	runtime.sampleHidden = append([]float32(nil), req.Hidden...)
	return runtime.samples[req.Step], nil
}

func (runtime *splitExecutorTestLocalRuntime) DecodeToken(_ context.Context, id int32) (string, error) {
	return runtime.text[id], nil
}

type splitNativeTestModel struct {
	prefill        *metal.SplitState
	sample         metal.SplitSampleResult
	prefillPrompts []string
	sampleRequests []metal.SplitSampleRequest
}

func (model *splitNativeTestModel) SplitPrefill(_ context.Context, prompt string) (*metal.SplitState, error) {
	model.prefillPrompts = append(model.prefillPrompts, prompt)
	return model.prefill, nil
}

func (model *splitNativeTestModel) SplitForwardAttention(context.Context, *metal.SplitState, metal.SplitAttentionRequest) (metal.SplitAttentionResult, error) {
	return metal.SplitAttentionResult{}, nil
}

func (model *splitNativeTestModel) SplitSample(_ context.Context, _ *metal.SplitState, req metal.SplitSampleRequest) (metal.SplitSampleResult, error) {
	model.sampleRequests = append(model.sampleRequests, req)
	return model.sample, nil
}

func (model *splitNativeTestModel) Close() error { return nil }

func equalSplitFloat32Slices(a, b []float32) bool {
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

func equalSplitStringSlices(a, b []string) bool {
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
