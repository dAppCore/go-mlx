// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/safetensors"
)

const cliTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h":0,"e":1,"l":2,"o":3,"▁":4,"he":5,"ll":6},
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

const cliGemma4TokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h":0,"e":1,"l":2,"o":3,"▁":4,"he":5,"ll":6},
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 2, "content": "<bos>", "special": true},
    {"id": 3, "content": "<unk>", "special": true},
    {"id": 4, "content": "<mask>", "special": true},
    {"id": 50, "content": "<|tool_response>", "special": true},
    {"id": 105, "content": "<|turn>", "special": true},
    {"id": 106, "content": "<turn|>", "special": true}
  ]
}`

func writeCLIPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

func TestRunCommand_PackJSON_Good(t *testing.T) {
	dir := t.TempDir()
	writeCLIPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"max_position_embeddings": 32768,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeCLIPackFile(t, core.PathJoin(dir, "tokenizer.json"), cliTokenizerJSON)
	writeCLIPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"pack", "-json", "-quantization", "4", "-max-context", "131072", dir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), `"valid":true`) || !core.Contains(stdout.String(), `"architecture":"qwen3"`) {
		t.Fatalf("stdout = %q, want JSON pack report", stdout.String())
	}
}

func TestRunCommand_PackInvalid_Bad(t *testing.T) {
	dir := t.TempDir()
	writeCLIPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"unknown"}`)
	writeCLIPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"pack", dir}, stdout, stderr)
	if code == 0 {
		t.Fatalf("exit code = %d, want non-zero", code)
	}
	if !core.Contains(stderr.String(), "unsupported_architecture") || !core.Contains(stderr.String(), "missing_tokenizer") {
		t.Fatalf("stderr = %q, want validation issues", stderr.String())
	}
}

func TestRunCommand_SSDRecipesJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"ssd-recipes", "-json"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"kind": "simple-self-distillation-recipes"`,
		`"SimpleSD-4B-instruct"`,
		`"apple/SimpleSD-4B-thinking"`,
		`"LiveCodeBench-v6"`,
		`"n_repeat": 20`,
		`"filter_shortest_percent": 10`,
		`"repetition_penalty": 1`,
		`"no_python": true`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
}

func TestRunCommand_AutoRoundJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"auto-round", "-json"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"kind": "auto-round-profiles"`,
		`"no_python": true`,
		`"source": "https://github.com/intel/auto-round"`,
		`"id": "auto-round"`,
		`"id": "auto-round-best"`,
		`"id": "auto-round-light"`,
		`"scheme": "W4A16"`,
		`"scheme": "GGUF:Q4_K_M"`,
		`"pack_sidecars":`,
		`"auto_round_config.json"`,
		`"quantization_config.json"`,
		`"calibration_default":`,
		`"algorithm": "auto-round"`,
		`"weight_rounding.signround"`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
}

func TestRunCommand_AutoRoundProfileJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"auto-round", "-json", "-profile", "auto-round-best"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"id": "auto-round-best"`,
		`"iters": 1000`,
		`"nsamples": 512`,
		`"group_size": 32`,
		`"bits": 2`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
	if core.Contains(out, `"id": "auto-round-light"`) {
		t.Fatalf("stdout = %q, want only selected profile", out)
	}
}

func TestRunCommand_AutoRoundValidation_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"auto-round", "-profile", "missing"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), `unknown profile "missing"`) {
		t.Fatalf("stderr = %q, want unknown profile", stderr.String())
	}
}

func TestRunCommand_SSDEvalJSON_Good(t *testing.T) {
	dir := t.TempDir()
	samplesPath := core.PathJoin(dir, "lcb.jsonl")
	outputPath := core.PathJoin(dir, "reports", "lcb-report.json")
	if result := core.WriteFile(samplesPath, []byte(
		`{"id":"old","prompt":"old","contest_date":"2025-01-31"}`+"\n"+
			`{"id":"v6","prompt":"Write add.","contest_date":"2025-03-15","difficulty":"easy","tests":["assert add(1,2)==3"]}`+"\n"), 0o644); !result.OK {
		t.Fatalf("WriteFile(samples) error = %v", result.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"ssd-eval",
		"-json",
		"-samples", samplesPath,
		"-output", outputPath,
		"-n-repeat", "10",
		"-sampling-params", "temperature=0.9,top_p=0.8,top_k=20,max_tokens=65536",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"kind": "simple-self-distillation-eval-plan"`,
		`"no_python": true`,
		`"livecodebench_v6": true`,
		`"samples": 1`,
		`"output_path": "` + outputPath + `"`,
		`"n_repeat": 10`,
		`"max_tokens": 65536`,
		`"temperature": 0.9`,
		`"top_p": 0.8`,
		`"top_k": 20`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
}

func TestRunCommand_SSDEvalValidation_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"ssd-eval", "-json"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), "samples path is required") {
		t.Fatalf("stderr = %q, want missing samples path", stderr.String())
	}
}

func TestRunCommand_MemoryPretrainBuildJSON_Good(t *testing.T) {
	dir := t.TempDir()
	corpusPath := core.PathJoin(dir, "corpus.jsonl")
	routerPath := core.PathJoin(dir, "router.json")
	ffnPath := core.PathJoin(dir, "ffn.json")
	clusterInput := core.PathJoin(dir, "tasks.jsonl")
	clusterOutput := core.PathJoin(dir, "clustered.jsonl")
	if result := core.WriteFile(corpusPath, []byte(
		`{"id":"go-1","text":"Go memory planning","meta":{"source":"docs"}}`+"\n"+
			`{"id":"go-2","text":"Go cgo bridge","meta":{"source":"docs"}}`+"\n"+
			`{"id":"poem-1","text":"winter proof poem","meta":{"source":"creative"}}`+"\n"+
			`{"id":"poem-2","text":"autumn prayer","meta":{"source":"creative"}}`+"\n"), 0o644); !result.OK {
		t.Fatalf("WriteFile(corpus) error = %v", result.Value)
	}
	if result := core.WriteFile(clusterInput, []byte(`{"context":"Go memory planning"}`+"\n"), 0o644); !result.OK {
		t.Fatalf("WriteFile(cluster input) error = %v", result.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"memory-pretrain-build",
		"-json",
		"-corpus", corpusPath,
		"-router", routerPath,
		"-ffn-memory", ffnPath,
		"-hidden-size", "8",
		"-layers", "2",
		"-levels", "1",
		"-tokens", "2",
		"-branching", "2",
		"-depth", "1",
		"-min-cluster-size", "1",
		"-kmeans-iters", "4",
		"-cluster-input", clusterInput,
		"-cluster-output", clusterOutput,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"kind": "memory-pretraining-artifacts"`,
		`"no_python": true`,
		`"corpus_records": 4`,
		`"ffn_memory_layers": 2`,
		`"learned_rows": 1`,
		`"embedding": "text-hash"`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
	for _, path := range []string{routerPath, ffnPath, clusterOutput} {
		if result := core.ReadFile(path); !result.OK {
			t.Fatalf("ReadFile(%s) error = %v", path, result.Value)
		}
	}
	readClustered := core.ReadFile(clusterOutput)
	if !core.Contains(core.AsString(readClustered.Value.([]byte)), `"cluster_ids"`) {
		t.Fatalf("cluster output = %q, want cluster_ids", core.AsString(readClustered.Value.([]byte)))
	}
}

func TestRunCommand_MemoryPretrainBuildValidation_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"memory-pretrain-build", "-json"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), "corpus path is required") {
		t.Fatalf("stderr = %q, want missing corpus path", stderr.String())
	}
}

func countInt32(values []int32, needle int32) int {
	count := 0
	for _, value := range values {
		if value == needle {
			count++
		}
	}
	return count
}

type fakeDriverProfileModel struct {
	generateCalls     int
	chunkCalls        int
	chatChunkCalls    int
	chatCalls         int
	chunks            []string
	chatChunkBytes    int
	chatChunkMessages []inference.Message
	metrics           mlx.Metrics
	streamTokens      []mlx.Token
	delayedMetrics    mlx.Metrics
	metricsReady      chan struct{}
	metricsClosed     bool
	lastConfig        mlx.GenerateConfig
}

func TestRunCommand_SliceJSON_Good(t *testing.T) {
	source := writeCLISlicePack(t)
	output := core.PathJoin(t.TempDir(), "client-slice")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"slice", "-json", "-preset", "client", "-output", output, source}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), `"output_path":`) || !core.Contains(stdout.String(), `"selected_tensor_bytes": "12"`) {
		t.Fatalf("stdout = %q, want slice JSON report with byte labels", stdout.String())
	}
	if result := core.Stat(core.PathJoin(output, "model.safetensors")); !result.OK {
		t.Fatalf("slice model.safetensors not written: %v", result.Value)
	}
}

func TestRunCommand_FFNEstimateJSON_Good(t *testing.T) {
	originalEstimate := runCPUFFNMemoryEstimate
	t.Cleanup(func() { runCPUFFNMemoryEstimate = originalEstimate })
	var gotPath string
	var gotCache int
	runCPUFFNMemoryEstimate = func(_ context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
		gotPath = sourcePath
		gotCache = cpuFFNCache
		return &mlx.CPUSplitFFNMemoryReport{
			Estimated:            true,
			TotalLayers:          4,
			LoadedLayers:         2,
			LayerLoads:           4,
			EvictedLayers:        2,
			CacheLimit:           2,
			ResidentBytes:        128,
			PeakResidentBytes:    256,
			DenseEquivalentBytes: 512,
			SavedBytes:           384,
			ResidentRatio:        0.25,
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"ffn-estimate", "-json", "-cpu-ffn-cache", "2", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotPath != "/models/qwen" || gotCache != 2 {
		t.Fatalf("estimate args path=%q cache=%d", gotPath, gotCache)
	}
	for _, want := range []string{`"source_path": "/models/qwen"`, `"cpu_ffn_cache": 2`, `"cpu_ffn_memory_estimate"`, `"estimated": true`, `"total_layers": 4`, `"peak_resident_bytes": 256`, `"saved_bytes": 384`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DiscoverJSON_Good(t *testing.T) {
	originalDiscover := runDiscoverLocalRuntime
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() {
		runDiscoverLocalRuntime = originalDiscover
		runGetDeviceInfo = originalDeviceInfo
	})
	var gotCfg mlx.LocalDiscoveryConfig
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	runDiscoverLocalRuntime = func(_ context.Context, cfg mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		gotCfg = cfg
		return inference.MachineDiscoveryReport{
			Runtime:    inference.RuntimeIdentity{Backend: "metal", Device: "apple9"},
			Available:  true,
			Device:     inference.MachineDeviceInfo{Architecture: "apple9", MemorySize: 96 << 30},
			Workloads:  []inference.TuningWorkload{inference.TuningWorkloadCoding},
			CacheModes: []string{"paged"},
			Capabilities: []inference.Capability{
				inference.SupportedCapability(inference.CapabilityRuntimeDiscovery, inference.CapabilityGroupRuntime),
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"discover", "-json", "-probe-device", "-model-dir", "/models", "-include-models", "-include-candidates", "-max-models", "3", "-workload", "coding"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if len(gotCfg.ModelDirs) != 1 || gotCfg.ModelDirs[0] != "/models" || !gotCfg.IncludeModels || !gotCfg.IncludeCandidates || gotCfg.MaxModels != 3 {
		t.Fatalf("discovery cfg = %+v", gotCfg)
	}
	if len(gotCfg.Workloads) != 1 || gotCfg.Workloads[0] != inference.TuningWorkloadCoding {
		t.Fatalf("workloads = %+v, want coding", gotCfg.Workloads)
	}
	if gotCfg.Device.Architecture != "apple9" || gotCfg.Device.MemorySize != 96<<30 {
		t.Fatalf("device = %+v, want probed apple9 device", gotCfg.Device)
	}
	for _, want := range []string{`"backend": "metal"`, `"available": true`, `"architecture": "apple9"`, `"cache_modes":`, `"runtime.discovery"`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_TunePlanJSON_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	t.Cleanup(func() { runPlanLocalTuning = originalPlan })
	var gotReq inference.TuningPlanRequest
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		gotReq = req
		return inference.TuningPlan{
			Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:   inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3"},
			Workloads: []inference.TuningWorkload{
				inference.TuningWorkloadAgentState,
			},
			Candidates: []inference.TuningCandidate{
				{
					ID:            "agent_state:paged:ctx32768:batch1",
					Workload:      inference.TuningWorkloadAgentState,
					ContextLength: 32768,
					BatchSize:     1,
					CacheMode:     "paged",
				},
			},
			Recommended: map[inference.TuningWorkload]string{
				inference.TuningWorkloadAgentState: "agent_state:paged:ctx32768:batch1",
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-plan", "-json", "-workload", "agent_state", "-max-candidates", "2", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotReq.Model.Path != "/models/qwen" || gotReq.Budget.MaxCandidates != 2 {
		t.Fatalf("plan req = %+v", gotReq)
	}
	if len(gotReq.Workloads) != 1 || gotReq.Workloads[0] != inference.TuningWorkloadAgentState {
		t.Fatalf("workloads = %+v, want agent_state", gotReq.Workloads)
	}
	for _, want := range []string{`"model":`, `"path": "/models/qwen"`, `"candidates"`, `"agent_state:paged:ctx32768:batch1"`, `"recommended"`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_TunePlanSplitFFNJSON_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	originalEstimate := runCPUFFNMemoryEstimate
	t.Cleanup(func() {
		runPlanLocalTuning = originalPlan
		runCPUFFNMemoryEstimate = originalEstimate
	})
	var estimatePath string
	var estimateCaches []int
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		return inference.TuningPlan{
			Runtime:   inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:     inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3"},
			Workloads: req.Workloads,
			Candidates: []inference.TuningCandidate{
				{
					ID:            "coding:paged:ctx32768:batch1",
					Workload:      inference.TuningWorkloadCoding,
					ContextLength: 32768,
					BatchSize:     1,
					CacheMode:     "paged",
				},
			},
			Recommended: map[inference.TuningWorkload]string{
				inference.TuningWorkloadCoding: "coding:paged:ctx32768:batch1",
			},
		}, nil
	}
	runCPUFFNMemoryEstimate = func(_ context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
		estimatePath = sourcePath
		estimateCaches = append(estimateCaches, cpuFFNCache)
		report := &mlx.CPUSplitFFNMemoryReport{
			Estimated:            true,
			TotalLayers:          4,
			LoadedLayers:         1,
			LayerLoads:           4,
			EvictedLayers:        3,
			CacheLimit:           cpuFFNCache,
			ResidentBytes:        64,
			PeakResidentBytes:    64,
			DenseEquivalentBytes: 512,
			SavedBytes:           448,
		}
		if cpuFFNCache == 0 {
			report.LoadedLayers = 4
			report.LayerLoads = 4
			report.EvictedLayers = 0
			report.ResidentBytes = 256
			report.PeakResidentBytes = 256
			report.SavedBytes = 256
		}
		return report, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-plan", "-json", "-workload", "coding", "-split-ffn-caches", "0,1", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if estimatePath != "/models/qwen" || len(estimateCaches) != 2 || estimateCaches[0] != 0 || estimateCaches[1] != 1 {
		t.Fatalf("estimate path=%q caches=%v, want /models/qwen [0 1]", estimatePath, estimateCaches)
	}
	for _, want := range []string{
		`"coding:split_cpu_ffn:cache1"`,
		`"coding:split_cpu_ffn:cache0"`,
		`"split": "cpu_ffn"`,
		`"cpu_ffn_cache_layers": "1"`,
		`"cpu_ffn_cache_layers": "0"`,
		`"cpu_ffn_peak_resident_bytes": "64"`,
		`"cpu_ffn_peak_resident_bytes": "256"`,
		`"rank": "1"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_TuneRunJSONL_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	originalRun := runLocalTuning
	t.Cleanup(func() {
		runPlanLocalTuning = originalPlan
		runLocalTuning = originalRun
	})
	candidate := inference.TuningCandidate{
		ID:            "coding:paged:ctx32768:batch1",
		Workload:      inference.TuningWorkloadCoding,
		ContextLength: 32768,
		BatchSize:     1,
		CacheMode:     "paged",
	}
	var gotReq inference.TuningPlanRequest
	var gotCfg mlx.LocalTuningRunConfig
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		gotReq = req
		return inference.TuningPlan{
			Runtime:     inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:       inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3"},
			Workloads:   req.Workloads,
			Candidates:  []inference.TuningCandidate{candidate},
			Recommended: map[inference.TuningWorkload]string{inference.TuningWorkloadCoding: candidate.ID},
		}, nil
	}
	runLocalTuning = func(_ context.Context, cfg mlx.LocalTuningRunConfig) ([]inference.TuningResult, error) {
		gotCfg = cfg
		if cfg.Emit != nil {
			cfg.Emit(inference.TuningEvent{Kind: inference.TuningEventCandidate, Candidate: candidate})
		}
		result := inference.TuningResult{
			Candidate: candidate,
			Measurements: inference.TuningMeasurements{
				DecodeTokensPerSec: 42,
				PeakMemoryBytes:    2048,
			},
			Score: inference.TuningScore{
				Workload:           inference.TuningWorkloadCoding,
				Score:              42,
				DecodeTokensPerSec: 42,
			},
		}
		if cfg.Emit != nil {
			cfg.Emit(inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: candidate, Result: &result})
		}
		return []inference.TuningResult{result}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-run", "-jsonl", "-workload", "coding", "-max-candidates", "1", "-prompt", "smoke", "-max-tokens", "4", "-runs", "2", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotReq.Model.Path != "/models/qwen" || gotReq.Budget.MaxCandidates != 1 {
		t.Fatalf("plan req = %+v", gotReq)
	}
	if len(gotReq.Workloads) != 1 || gotReq.Workloads[0] != inference.TuningWorkloadCoding {
		t.Fatalf("workloads = %+v, want coding", gotReq.Workloads)
	}
	if gotCfg.ModelPath != "/models/qwen" || gotCfg.Workload != inference.TuningWorkloadCoding || len(gotCfg.Candidates) != 1 {
		t.Fatalf("tune cfg = %+v", gotCfg)
	}
	if gotCfg.Bench.Prompt != "smoke" || gotCfg.Bench.MaxTokens != 4 || gotCfg.Bench.Runs != 2 {
		t.Fatalf("bench cfg = %+v, want smoke/4/2", gotCfg.Bench)
	}
	for _, want := range []string{
		`"kind":"candidate"`,
		`"kind":"result"`,
		`"decode_tokens_per_sec":42`,
		`"score":42`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_TuneRunProfileOutput_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	originalRun := runLocalTuning
	t.Cleanup(func() {
		runPlanLocalTuning = originalPlan
		runLocalTuning = originalRun
	})
	slow := inference.TuningCandidate{
		ID:       "coding:paged:slow",
		Workload: inference.TuningWorkloadCoding,
		Model:    inference.ModelIdentity{Path: "/models/qwen", Architecture: "qwen3"},
		Runtime:  inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
	}
	fast := inference.TuningCandidate{
		ID:       "coding:paged:fast",
		Workload: inference.TuningWorkloadCoding,
		Model:    inference.ModelIdentity{Path: "/models/qwen", Architecture: "qwen3"},
		Runtime:  inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
	}
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		return inference.TuningPlan{
			Runtime:    inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:      inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3"},
			Workloads:  req.Workloads,
			Candidates: []inference.TuningCandidate{slow, fast},
		}, nil
	}
	runLocalTuning = func(_ context.Context, cfg mlx.LocalTuningRunConfig) ([]inference.TuningResult, error) {
		results := []inference.TuningResult{
			{
				Candidate:    slow,
				Measurements: inference.TuningMeasurements{LoadMilliseconds: 90, FirstTokenMilliseconds: 40, DecodeTokensPerSec: 12, KVRestoreMilliseconds: 8, PeakMemoryBytes: 4096, CorrectnessSmokeResult: "passed", CorrectnessSmokeChecks: 2},
				Score:        inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 12, DecodeTokensPerSec: 12},
			},
			{
				Candidate:    fast,
				Measurements: inference.TuningMeasurements{LoadMilliseconds: 70, FirstTokenMilliseconds: 25, DecodeTokensPerSec: 42, KVRestoreMilliseconds: 3, PeakMemoryBytes: 2048, CorrectnessSmokeResult: "passed", CorrectnessSmokeChecks: 2},
				Score:        inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42, DecodeTokensPerSec: 42},
			},
		}
		for _, result := range results {
			if cfg.Emit != nil {
				cfg.Emit(inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: result.Candidate, Result: &result})
			}
		}
		return results, nil
	}
	profilePath := core.PathJoin(t.TempDir(), "coding-profile.json")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-run", "-jsonl", "-workload", "coding", "-profile-output", profilePath, "-machine-hash", "apple9-96gb", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"kind":"selected"`) || !core.Contains(stdout.String(), `"profile_output":"`+profilePath+`"`) || !core.Contains(stdout.String(), `"selection_policy":"highest_successful_score"`) {
		t.Fatalf("stdout = %q, want selected event with profile output", stdout.String())
	}
	read := core.ReadFile(profilePath)
	if !read.OK {
		t.Fatalf("read profile: %v", read.Value)
	}
	var profile inference.TuningProfile
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		t.Fatalf("unmarshal profile: %v", result.Value)
	}
	if profile.Candidate.ID != fast.ID || profile.Score.Score != 42 {
		t.Fatalf("profile = %+v, want fast candidate", profile)
	}
	if profile.Key.MachineHash != "apple9-96gb" || profile.Key.Workload != inference.TuningWorkloadCoding {
		t.Fatalf("profile key = %+v, want machine/workload", profile.Key)
	}
	if profile.CreatedAtUnix == 0 {
		t.Fatalf("profile CreatedAtUnix = 0, want timestamp")
	}
	if profile.Labels["selection_policy"] != "highest_successful_score" || profile.Labels["selected_candidate_id"] != fast.ID || profile.Labels["successful_candidates"] != "2" {
		t.Fatalf("profile labels = %+v, want persisted selection policy and candidate count", profile.Labels)
	}
	if profile.Labels["selected_decode_tokens_per_sec"] != "42.000000" || profile.Labels["selection_score_delta"] != "30.000000" {
		t.Fatalf("profile labels = %+v, want measured winner reason", profile.Labels)
	}
	if profile.Measurements.LoadMilliseconds != 70 || profile.Measurements.FirstTokenMilliseconds != 25 || profile.Measurements.KVRestoreMilliseconds != 3 || profile.Measurements.CorrectnessSmokeResult != "passed" {
		t.Fatalf("profile measurements = %+v, want non-expert trust counters", profile.Measurements)
	}
	if profile.Labels["selected_load_milliseconds"] != "70.000000" || profile.Labels["selected_first_token_milliseconds"] != "25.000000" || profile.Labels["selected_restore_milliseconds"] != "3.000000" || profile.Labels["selected_correctness_smoke_result"] != "passed" {
		t.Fatalf("profile labels = %+v, want trust summary labels", profile.Labels)
	}
}

func TestRunCommand_TuneRunCurrentMachineProfileOutput_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	originalRun := runLocalTuning
	originalDiscover := runDiscoverLocalRuntime
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() {
		runPlanLocalTuning = originalPlan
		runLocalTuning = originalRun
		runDiscoverLocalRuntime = originalDiscover
		runGetDeviceInfo = originalDeviceInfo
	})
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	var gotDiscoveryCfg mlx.LocalDiscoveryConfig
	runDiscoverLocalRuntime = func(_ context.Context, cfg mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		gotDiscoveryCfg = cfg
		return inference.MachineDiscoveryReport{
			Labels: map[string]string{"machine_hash": "apple9-96gb"},
		}, nil
	}
	candidate := inference.TuningCandidate{
		ID:       "coding:paged:fast",
		Workload: inference.TuningWorkloadCoding,
		Model:    inference.ModelIdentity{Path: "/models/qwen", Architecture: "qwen3"},
		Runtime:  inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
	}
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		return inference.TuningPlan{
			Runtime:    inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:      inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3"},
			Workloads:  req.Workloads,
			Candidates: []inference.TuningCandidate{candidate},
		}, nil
	}
	runLocalTuning = func(_ context.Context, cfg mlx.LocalTuningRunConfig) ([]inference.TuningResult, error) {
		result := inference.TuningResult{
			Candidate:    candidate,
			Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 42},
			Score:        inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42, DecodeTokensPerSec: 42},
		}
		if cfg.Emit != nil {
			cfg.Emit(inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: candidate, Result: &result})
		}
		return []inference.TuningResult{result}, nil
	}
	profilePath := core.PathJoin(t.TempDir(), "coding-profile.json")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-run", "-jsonl", "-workload", "coding", "-profile-output", profilePath, "-current-machine", "/models/qwen"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotDiscoveryCfg.Device.Architecture != "apple9" || gotDiscoveryCfg.Device.MemorySize != 96<<30 {
		t.Fatalf("discovery cfg device = %+v, want current machine probe", gotDiscoveryCfg.Device)
	}
	if !core.Contains(stdout.String(), `"kind":"selected"`) || !core.Contains(stdout.String(), `"machine_hash":"apple9-96gb"`) {
		t.Fatalf("stdout = %q, want selected event with current machine hash", stdout.String())
	}
	read := core.ReadFile(profilePath)
	if !read.OK {
		t.Fatalf("read profile: %v", read.Value)
	}
	var profile inference.TuningProfile
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		t.Fatalf("unmarshal profile: %v", result.Value)
	}
	if profile.Key.MachineHash != "apple9-96gb" {
		t.Fatalf("profile key = %+v, want current machine hash", profile.Key)
	}
}

func TestRunCommand_TuneRunProfileDir_Good(t *testing.T) {
	originalPlan := runPlanLocalTuning
	originalRun := runLocalTuning
	t.Cleanup(func() {
		runPlanLocalTuning = originalPlan
		runLocalTuning = originalRun
	})
	candidate := inference.TuningCandidate{
		ID:       "coding:paged:fast",
		Workload: inference.TuningWorkloadCoding,
		Model:    inference.ModelIdentity{Path: "/models/qwen3.6", Architecture: "qwen3_6"},
		Runtime:  inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
	}
	runPlanLocalTuning = func(_ context.Context, req inference.TuningPlanRequest) (inference.TuningPlan, error) {
		return inference.TuningPlan{
			Runtime:    inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:      inference.ModelIdentity{Path: req.Model.Path, Architecture: "qwen3_6"},
			Workloads:  req.Workloads,
			Candidates: []inference.TuningCandidate{candidate},
		}, nil
	}
	runLocalTuning = func(_ context.Context, cfg mlx.LocalTuningRunConfig) ([]inference.TuningResult, error) {
		result := inference.TuningResult{
			Candidate:    candidate,
			Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 42},
			Score:        inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42, DecodeTokensPerSec: 42},
		}
		if cfg.Emit != nil {
			cfg.Emit(inference.TuningEvent{Kind: inference.TuningEventResult, Candidate: candidate, Result: &result})
		}
		return []inference.TuningResult{result}, nil
	}
	dir := t.TempDir()
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-run", "-jsonl", "-workload", "coding", "-profile-dir", dir, "-machine-hash", "sha256:abcdef1234567890", "/models/qwen3.6"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	profiles := core.PathGlob(core.PathJoin(dir, "*.json"))
	if len(profiles) != 1 {
		t.Fatalf("profiles = %+v, want one generated profile", profiles)
	}
	expectedPath := core.PathJoin(dir, "coding-abcdef123456-qwen3-6-coding-paged-fast.json")
	if profiles[0] != expectedPath {
		t.Fatalf("profile path = %q, want %q", profiles[0], expectedPath)
	}
	if !core.Contains(stdout.String(), `"profile_output":"`+expectedPath+`"`) {
		t.Fatalf("stdout = %q, want generated profile_output", stdout.String())
	}
	var profile inference.TuningProfile
	read := core.ReadFile(expectedPath)
	if !read.OK {
		t.Fatalf("read profile: %v", read.Value)
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		t.Fatalf("unmarshal profile: %v", result.Value)
	}
	if profile.Key.MachineHash != "sha256:abcdef1234567890" || profile.Candidate.ID != candidate.ID {
		t.Fatalf("profile = %+v, want stored key and candidate", profile)
	}
}

func TestRunCommand_TuneProfileJSON_Good(t *testing.T) {
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Runtime:     inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			Model:       inference.ModelIdentity{Path: "/models/qwen", Architecture: "qwen3"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate: inference.TuningCandidate{
			ID:                   "coding:paged:ctx32768:batch1",
			Workload:             inference.TuningWorkloadCoding,
			Model:                inference.ModelIdentity{Path: "/models/qwen", Architecture: "qwen3"},
			Runtime:              inference.RuntimeIdentity{Backend: "metal", Device: "apple9", CacheMode: "paged"},
			ContextLength:        32768,
			ParallelSlots:        2,
			PromptCache:          true,
			PromptCacheMinTokens: 512,
			CachePolicy:          "full",
			CacheMode:            "paged",
			BatchSize:            1,
			PrefillChunkSize:     1024,
			ExpectedQuantization: 4,
			MemoryLimitBytes:     8 << 30,
			CacheLimitBytes:      2 << 30,
			WiredLimitBytes:      1 << 30,
			Adapter:              inference.AdapterIdentity{Path: "/models/qwen/adapter"},
		},
		Score: inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42, DecodeTokensPerSec: 42},
	}
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		t.Fatalf("marshal profile: %v", data.Value)
	}
	profilePath := core.PathJoin(t.TempDir(), "coding-profile.json")
	if result := core.WriteFile(profilePath, data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("write profile: %v", result.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"tune-profile", "-json", profilePath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"profile_path": "` + profilePath + `"`,
		`"model_path": "/models/qwen"`,
		`"workload": "coding"`,
		`"candidate_id": "coding:paged:ctx32768:batch1"`,
		`"context_length": 32768`,
		`"parallel_slots": 2`,
		`"prompt_cache": true`,
		`"prompt_cache_min_tokens": 512`,
		`"cache_policy": "full"`,
		`"cache_mode": "paged"`,
		`"batch_size": 1`,
		`"prefill_chunk_size": 1024`,
		`"expected_quantization": 4`,
		`"adapter_path": "/models/qwen/adapter"`,
		`"score": 42`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProfileSelectJSON_Good(t *testing.T) {
	dir := t.TempDir()
	slowPath := core.PathJoin(dir, "slow.json")
	fastPath := core.PathJoin(dir, "fast.json")
	otherPath := core.PathJoin(dir, "other.json")
	baseProfile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate: inference.TuningCandidate{
			Workload:      inference.TuningWorkloadCoding,
			Model:         inference.ModelIdentity{Path: "/models/qwen"},
			ContextLength: 32768,
			CacheMode:     "paged",
		},
	}
	slow := baseProfile
	slow.Candidate.ID = "slow"
	slow.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 12}
	fast := baseProfile
	fast.Candidate.ID = "fast"
	fast.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42}
	other := baseProfile
	other.Key.MachineHash = "other-machine"
	other.Candidate.ID = "other"
	other.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 100}
	writeCLIProfile(t, slowPath, slow)
	writeCLIProfile(t, fastPath, fast)
	writeCLIProfile(t, otherPath, other)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-select", "-json", "-machine-hash", "apple9-96gb", "-workload", "coding", "-model-path", "/models/qwen", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"profile_dir": "` + dir + `"`,
		`"profile_path": "` + fastPath + `"`,
		`"matched_profiles": 2`,
		`"candidate_id": "fast"`,
		`"model_path": "/models/qwen"`,
		`"workload": "coding"`,
		`"machine_hash": "apple9-96gb"`,
		`"score": 42`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProfileListJSON_Good(t *testing.T) {
	dir := t.TempDir()
	slowPath := core.PathJoin(dir, "slow.json")
	fastPath := core.PathJoin(dir, "fast.json")
	otherPath := core.PathJoin(dir, "other.json")
	baseProfile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate: inference.TuningCandidate{
			Workload: inference.TuningWorkloadCoding,
			Model:    inference.ModelIdentity{Path: "/models/qwen"},
		},
	}
	slow := baseProfile
	slow.Candidate.ID = "slow"
	slow.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 12}
	fast := baseProfile
	fast.Candidate.ID = "fast"
	fast.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42}
	other := baseProfile
	other.Key.MachineHash = "other-machine"
	other.Candidate.ID = "other"
	other.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 100}
	writeCLIProfile(t, slowPath, slow)
	writeCLIProfile(t, fastPath, fast)
	writeCLIProfile(t, otherPath, other)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-list", "-json", "-machine-hash", "apple9-96gb", "-workload", "coding", "-model-path", "/models/qwen", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"profile_dir": "` + dir + `"`,
		`"profile_count": 2`,
		`"profile_path": "` + fastPath + `"`,
		`"profile_path": "` + slowPath + `"`,
		`"candidate_id": "fast"`,
		`"candidate_id": "slow"`,
		`"machine_hash": "apple9-96gb"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), otherPath) || core.Contains(stdout.String(), `"candidate_id": "other"`) {
		t.Fatalf("stdout = %q, want other-machine profile filtered out", stdout.String())
	}
}

func TestRunCommand_ProfileListOmitsFullProfilesByDefault_Good(t *testing.T) {
	dir := t.TempDir()
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate:     inference.TuningCandidate{ID: "fast", Workload: inference.TuningWorkloadCoding, Model: inference.ModelIdentity{Path: "/models/qwen"}},
		Score:         inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42},
		CreatedAtUnix: 1710000000,
	}
	writeCLIProfile(t, core.PathJoin(dir, "fast.json"), profile)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-list", "-json", "-machine-hash", "apple9-96gb", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if core.Contains(stdout.String(), `"profile": {`) {
		t.Fatalf("stdout = %q, want lightweight list without nested profile", stdout.String())
	}
	if !core.Contains(stdout.String(), `"candidate_id": "fast"`) {
		t.Fatalf("stdout = %q, want profile summary", stdout.String())
	}
}

func TestRunCommand_ProfileListIncludeProfileJSON_Good(t *testing.T) {
	dir := t.TempDir()
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate:     inference.TuningCandidate{ID: "fast", Workload: inference.TuningWorkloadCoding, Model: inference.ModelIdentity{Path: "/models/qwen"}},
		Score:         inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42},
		CreatedAtUnix: 1710000000,
	}
	writeCLIProfile(t, core.PathJoin(dir, "fast.json"), profile)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-list", "-json", "-include-profile", "-machine-hash", "apple9-96gb", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"profile": {`) || !core.Contains(stdout.String(), `"created_at_unix": 1710000000`) {
		t.Fatalf("stdout = %q, want nested profile when requested", stdout.String())
	}
}

func TestRunCommand_ProfileListBestPerWorkloadJSON_Good(t *testing.T) {
	dir := t.TempDir()
	baseProfile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
		},
		Candidate: inference.TuningCandidate{
			Model: inference.ModelIdentity{Path: "/models/qwen"},
		},
	}
	slowCoding := baseProfile
	slowCoding.Key.Workload = inference.TuningWorkloadCoding
	slowCoding.Candidate.ID = "coding-slow"
	slowCoding.Candidate.Workload = inference.TuningWorkloadCoding
	slowCoding.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 12}
	fastCoding := baseProfile
	fastCoding.Key.Workload = inference.TuningWorkloadCoding
	fastCoding.Candidate.ID = "coding-fast"
	fastCoding.Candidate.Workload = inference.TuningWorkloadCoding
	fastCoding.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42}
	agentState := baseProfile
	agentState.Key.Workload = inference.TuningWorkloadAgentState
	agentState.Candidate.ID = "agent-state"
	agentState.Candidate.Workload = inference.TuningWorkloadAgentState
	agentState.Score = inference.TuningScore{Workload: inference.TuningWorkloadAgentState, Score: 30}
	writeCLIProfile(t, core.PathJoin(dir, "coding-slow.json"), slowCoding)
	writeCLIProfile(t, core.PathJoin(dir, "coding-fast.json"), fastCoding)
	writeCLIProfile(t, core.PathJoin(dir, "agent-state.json"), agentState)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-list", "-json", "-best-per-workload", "-machine-hash", "apple9-96gb", "-model-path", "/models/qwen", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{`"profile_count": 2`, `"candidate_id": "coding-fast"`, `"candidate_id": "agent-state"`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), `"candidate_id": "coding-slow"`) {
		t.Fatalf("stdout = %q, want slower coding profile removed", stdout.String())
	}
}

func TestRunCommand_ProfileSelectCurrentMachineJSON_Good(t *testing.T) {
	originalDiscover := runDiscoverLocalRuntime
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() {
		runDiscoverLocalRuntime = originalDiscover
		runGetDeviceInfo = originalDeviceInfo
	})
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	var gotCfg mlx.LocalDiscoveryConfig
	runDiscoverLocalRuntime = func(_ context.Context, cfg mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		gotCfg = cfg
		return inference.MachineDiscoveryReport{
			Device: inference.MachineDeviceInfo{
				Architecture: "apple9",
				Labels:       map[string]string{"machine_hash": "apple9-96gb"},
			},
			Labels: map[string]string{"machine_hash": "apple9-96gb"},
		}, nil
	}
	dir := t.TempDir()
	fastPath := core.PathJoin(dir, "fast.json")
	otherPath := core.PathJoin(dir, "other.json")
	fast := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "apple9-96gb",
			Model:       inference.ModelIdentity{Path: "/models/qwen"},
			Workload:    inference.TuningWorkloadCoding,
		},
		Candidate: inference.TuningCandidate{
			ID:       "fast",
			Workload: inference.TuningWorkloadCoding,
			Model:    inference.ModelIdentity{Path: "/models/qwen"},
		},
		Score: inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 42},
	}
	other := fast
	other.Key.MachineHash = "other-machine"
	other.Candidate.ID = "other"
	other.Score = inference.TuningScore{Workload: inference.TuningWorkloadCoding, Score: 100}
	writeCLIProfile(t, fastPath, fast)
	writeCLIProfile(t, otherPath, other)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"profile-select", "-json", "-current-machine", "-workload", "coding", "-model-path", "/models/qwen", dir}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Device.Architecture != "apple9" || gotCfg.Device.MemorySize != 96<<30 {
		t.Fatalf("discovery cfg device = %+v, want current machine probe", gotCfg.Device)
	}
	for _, want := range []string{
		`"profile_path": "` + fastPath + `"`,
		`"matched_profiles": 1`,
		`"candidate_id": "fast"`,
		`"machine_hash": "apple9-96gb"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ReplacePlanProfilesJSON_Good(t *testing.T) {
	dir := t.TempDir()
	currentPath := core.PathJoin(dir, "current-profile.json")
	nextPath := core.PathJoin(dir, "next-profile.json")
	current := inference.TuningProfile{
		Key: inference.TuningProfileKey{MachineHash: "apple9-96gb", Workload: inference.TuningWorkloadCoding},
		Candidate: inference.TuningCandidate{
			ID:      "current",
			Model:   inference.ModelIdentity{Path: "/models/qwen", QuantBits: 4},
			Adapter: inference.AdapterIdentity{Path: "/models/qwen/adapter"},
			Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "gpu", CacheMode: "paged"},
		},
	}
	next := inference.TuningProfile{
		Key: inference.TuningProfileKey{MachineHash: "apple9-96gb", Workload: inference.TuningWorkloadCoding},
		Candidate: inference.TuningCandidate{
			ID:      "next",
			Model:   inference.ModelIdentity{Path: "/models/qwen", QuantBits: 4},
			Adapter: inference.AdapterIdentity{Path: "/models/qwen/adapter"},
			Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "gpu", CacheMode: "q8"},
		},
	}
	writeCLIProfile(t, currentPath, current)
	writeCLIProfile(t, nextPath, next)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"replace-plan", "-json", "-current-profile", currentPath, "-next-profile", nextPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"current_profile_path": "` + currentPath + `"`,
		`"next_profile_path": "` + nextPath + `"`,
		`"action": "checkpoint_state"`,
		`"compatible": true`,
		`"runtime or cache settings changed"`,
		`"cache_mode": "paged"`,
		`"cache_mode": "q8"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func writeCLIProfile(t *testing.T, path string, profile inference.TuningProfile) {
	t.Helper()
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		t.Fatalf("marshal profile: %v", data.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("write profile: %v", result.Value)
	}
}

func writeCLISlicePack(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeCLIPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen2",
		"vocab_size": 16,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 32
	}`)
	writeCLIPackFile(t, core.PathJoin(dir, "tokenizer.json"), cliTokenizerJSON)
	writeCLISliceSafetensors(t, core.PathJoin(dir, "model.safetensors"), map[string][]byte{
		"model.embed_tokens.weight":              {1, 2, 3, 4},
		"model.layers.0.self_attn.q_proj.weight": {5, 6, 7, 8},
		"model.layers.0.mlp.down_proj.weight":    {9, 10, 11, 12},
		"lm_head.weight":                         {13, 14, 15, 16},
	})
	return dir
}

func writeCLISliceSafetensors(t *testing.T, path string, tensors map[string][]byte) {
	t.Helper()
	header := map[string]safetensors.HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		raw := tensors[name]
		header[name] = safetensors.HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(len(raw))},
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

func TestRunCommand_UsesBinaryNameForUsage_Good(t *testing.T) {
	previous := commandName
	commandName = "lthn-mlx"
	t.Cleanup(func() { commandName = previous })
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"help"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "Usage: lthn-mlx <command> [flags]") {
		t.Fatalf("stdout = %q, want lthn-mlx usage", stdout.String())
	}
}
