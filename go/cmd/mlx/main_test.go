// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/inference/safetensors"
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

// An unknown command prints the unknown-command notice + usage and exits 2.
func TestRunCommand_UnknownCommand_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"definitely-not-a-command"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 for an unknown command", code)
	}
	if !core.Contains(stderr.String(), "unknown command") {
		t.Fatalf("stderr = %q, want the unknown-command notice", stderr.String())
	}
}

// printDiscoverySummary renders the human-readable discovery report, including
// the metallib line when the label is present.
func TestPrintDiscoverySummary_Good(t *testing.T) {
	stdout := core.NewBuffer()
	printDiscoverySummary(stdout, inference.MachineDiscoveryReport{
		Runtime:   inference.RuntimeIdentity{Backend: "metal"},
		Available: true,
		Device:    inference.MachineDeviceInfo{Architecture: "apple9", MemorySize: 96 << 30},
		Labels: map[string]string{
			"metallib_kernel": "abc123",
			"metallib_source": "embedded",
			"metallib_path":   "/x/mlx.metallib",
		},
	})
	out := stdout.String()
	for _, want := range []string{"runtime discovery: metal", "available: true", "apple9", "metallib: embedded", "kernel=abc123"} {
		if !core.Contains(out, want) {
			t.Fatalf("summary = %q, missing %q", out, want)
		}
	}
}

// modelIdentityFromProfile overlays the candidate's non-zero fields on top of
// the key identity (last-wins per populated field).
func TestModelIdentityFromProfile_CandidateOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Model = inference.ModelIdentity{Path: "/key/path", Architecture: "gemma3", QuantBits: 4, ContextLength: 4096}
	profile.Candidate.Model = inference.ModelIdentity{Path: "/candidate/path", QuantBits: 8}

	id := modelIdentityFromProfile(profile)
	if id.Path != "/candidate/path" {
		t.Fatalf("Path = %q, want the candidate override", id.Path)
	}
	if id.QuantBits != 8 {
		t.Fatalf("QuantBits = %d, want the candidate override 8", id.QuantBits)
	}
	// Fields the candidate leaves zero keep the key's value.
	if id.Architecture != "gemma3" {
		t.Fatalf("Architecture = %q, want the key value retained", id.Architecture)
	}
	if id.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want the key value retained", id.ContextLength)
	}
}

// runtimeIdentityFromProfile overlays the candidate runtime fields.
func TestRuntimeIdentityFromProfile_CandidateOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Runtime = inference.RuntimeIdentity{Backend: "metal", Device: "key-dev"}
	profile.Candidate.Runtime = inference.RuntimeIdentity{Device: "cand-dev", NativeRuntime: true}

	id := runtimeIdentityFromProfile(profile)
	if id.Device != "cand-dev" {
		t.Fatalf("Device = %q, want candidate override", id.Device)
	}
	if !id.NativeRuntime {
		t.Fatal("NativeRuntime = false, want candidate true")
	}
	if id.Backend != "metal" {
		t.Fatalf("Backend = %q, want key value retained", id.Backend)
	}
}

// adapterIdentityFromProfile overlays the candidate adapter fields.
func TestAdapterIdentityFromProfile_CandidateOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Adapter = inference.AdapterIdentity{Path: "/key/adapter", Rank: 8}
	profile.Candidate.Adapter = inference.AdapterIdentity{Path: "/cand/adapter", Alpha: 16}

	id := adapterIdentityFromProfile(profile)
	if id.Path != "/cand/adapter" {
		t.Fatalf("Path = %q, want candidate override", id.Path)
	}
	if id.Alpha != 16 {
		t.Fatalf("Alpha = %v, want candidate 16", id.Alpha)
	}
	if id.Rank != 8 {
		t.Fatalf("Rank = %d, want key value retained", id.Rank)
	}
}
