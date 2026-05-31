// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/memory"
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

func TestRunCommand_BenchJSON_Good(t *testing.T) {
	originalLoad := loadBenchModel
	originalRun := runBenchReport
	t.Cleanup(func() {
		loadBenchModel = originalLoad
		runBenchReport = originalRun
	})

	var gotPath string
	var gotCfg bench.Config
	loadBenchModel = func(path string, opts ...mlx.LoadOption) (*mlx.Model, error) {
		gotPath = path
		return &mlx.Model{}, nil
	}
	runBenchReport = func(ctx context.Context, model *mlx.Model, cfg bench.Config) (*bench.Report, error) {
		gotCfg = cfg
		return &bench.Report{
			Version:   bench.ReportVersion,
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Generation: bench.GenerationSummary{
				DecodeTokensPerSec: 42,
				PeakMemoryBytes:    2048,
			},
		}, nil
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"bench", "-json", "-prompt", "hi", "-max-tokens", "7", "-runs", "2", "/models/demo"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if gotPath != "/models/demo" || gotCfg.Prompt != "hi" || gotCfg.MaxTokens != 7 || gotCfg.Runs != 2 {
		t.Fatalf("bench args path=%q cfg=%+v", gotPath, gotCfg)
	}
	if !core.Contains(stdout.String(), `"decode_tokens_per_sec": 42`) || !core.Contains(stdout.String(), `"model_path": "/models/demo"`) {
		t.Fatalf("stdout = %q, want JSON bench report", stdout.String())
	}
}

func TestRunCommand_BenchPromptFileStateKVWarm_Good(t *testing.T) {
	originalLoad := loadBenchModel
	originalRun := runBenchReport
	t.Cleanup(func() {
		loadBenchModel = originalLoad
		runBenchReport = originalRun
	})

	dir := t.TempDir()
	promptPath := core.PathJoin(dir, "prompt.txt")
	suffixPath := core.PathJoin(dir, "suffix.txt")
	writeCLIPackFile(t, promptPath, "alpha")
	writeCLIPackFile(t, suffixPath, "omega")

	var gotCfg bench.Config
	loadBenchModel = func(string, ...mlx.LoadOption) (*mlx.Model, error) {
		return &mlx.Model{}, nil
	}
	runBenchReport = func(_ context.Context, _ *mlx.Model, cfg bench.Config) (*bench.Report, error) {
		gotCfg = cfg
		return &bench.Report{
			Version: bench.ReportVersion,
			Config:  cfg,
			StateKVBlockWarm: bench.StateKVBlockWarmReport{
				Attempted: true,
				BlockSize: 512,
			},
		}, nil
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-prompt-file", promptPath,
		"-prompt-repeat", "2",
		"-prompt-suffix-file", suffixPath,
		"-state-kv-warm",
		"-state-kv-block-size", "512",
		"-state-kv-prefix-tokens", "1024",
		"-state-kv-store", "/tmp/bench.mvlog",
		"/models/demo",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Prompt != "alpha\n\nalpha\n\nomega" {
		t.Fatalf("bench prompt = %q, want repeated prompt plus suffix", gotCfg.Prompt)
	}
	if !gotCfg.IncludeStateKVBlockWarm || gotCfg.StateKVBlockSize != 512 || gotCfg.StateKVPrefixTokens != 1024 || gotCfg.StateKVBlockStorePath != "/tmp/bench.mvlog" {
		t.Fatalf("State bench cfg = %+v, want explicit KV block warm settings", gotCfg)
	}
	if !core.Contains(stdout.String(), `"include_state_kv_block_warm": true`) || !core.Contains(stdout.String(), `"state_kv_block_size": 512`) {
		t.Fatalf("stdout = %q, want State bench config", stdout.String())
	}
}

func TestRunCommand_BenchSpeculativeDraftModel_Good(t *testing.T) {
	originalLoadPair := loadSpeculativePair
	originalRunDraft := runBenchReportWithDraft
	originalRun := runBenchReport
	t.Cleanup(func() {
		loadSpeculativePair = originalLoadPair
		runBenchReportWithDraft = originalRunDraft
		runBenchReport = originalRun
	})

	var gotTargetPath, gotDraftPath string
	var gotCfg bench.Config
	loadSpeculativePair = func(targetPath, draftPath string, cfg mlx.SpeculativePairConfig) (*mlx.SpeculativePair, error) {
		gotTargetPath = targetPath
		gotDraftPath = draftPath
		if len(cfg.TargetOptions) == 0 || len(cfg.DraftOptions) == 0 {
			t.Fatalf("speculative load options = %+v, want target and draft options", cfg)
		}
		return &mlx.SpeculativePair{Target: &mlx.Model{}, Draft: &mlx.Model{}}, nil
	}
	runBenchReport = func(context.Context, *mlx.Model, bench.Config) (*bench.Report, error) {
		t.Fatal("runBenchReport called for speculative pair; want draft-aware runner")
		return nil, nil
	}
	runBenchReportWithDraft = func(_ context.Context, target, draft *mlx.Model, cfg bench.Config) (*bench.Report, error) {
		if target == nil || draft == nil {
			t.Fatalf("target/draft = %v/%v, want both models", target, draft)
		}
		gotCfg = cfg
		return &bench.Report{
			Version:   bench.ReportVersion,
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Config:    cfg,
			SpeculativeDecode: bench.DecodeOptimisationReport{
				Attempted: true,
				Metrics: bench.DecodeOptimisationMetrics{
					AcceptedTokens:      1,
					RejectedTokens:      1,
					AcceptanceRate:      0.5,
					VisibleTokensPerSec: 12.5,
				},
			},
		}, nil
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-context", "4096",
		"-speculative-draft-model", "/models/target-assistant",
		"-speculative-draft-tokens", "2",
		"/models/target",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotTargetPath != "/models/target" || gotDraftPath != "/models/target-assistant" {
		t.Fatalf("speculative paths target=%q draft=%q", gotTargetPath, gotDraftPath)
	}
	if !gotCfg.IncludeSpeculativeDecode || gotCfg.SpeculativeDraftModelPath != "/models/target-assistant" || gotCfg.SpeculativeDraftTokens != 2 {
		t.Fatalf("bench config = %+v, want speculative draft config", gotCfg)
	}
	if !core.Contains(stdout.String(), `"speculative_draft_model_path": "/models/target-assistant"`) ||
		!core.Contains(stdout.String(), `"visible_tokens_per_sec": 12.5`) {
		t.Fatalf("stdout = %q, want speculative config and metrics", stdout.String())
	}
}

func TestRunCommand_BenchSpeculativeDraftTokens_Bad(t *testing.T) {
	originalLoadPair := loadSpeculativePair
	t.Cleanup(func() { loadSpeculativePair = originalLoadPair })
	loadSpeculativePair = func(string, string, mlx.SpeculativePairConfig) (*mlx.SpeculativePair, error) {
		t.Fatal("loadSpeculativePair called for invalid draft token count")
		return nil, nil
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-speculative-draft-model", "/models/target-assistant",
		"-speculative-draft-tokens", "-1",
		"/models/target",
	}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "speculative draft tokens must be >= 0") {
		t.Fatalf("stderr = %q, want validation error", stderr.String())
	}
}

func TestRunCommand_BenchProfileJSON_Good(t *testing.T) {
	originalLoad := loadBenchModel
	originalRun := runBenchReport
	t.Cleanup(func() {
		loadBenchModel = originalLoad
		runBenchReport = originalRun
	})
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			Model:    inference.ModelIdentity{Path: "/models/qwen"},
			Workload: inference.TuningWorkloadCoding,
		},
		Candidate: inference.TuningCandidate{
			ID:                   "coding:paged:ctx32768:batch1",
			Workload:             inference.TuningWorkloadCoding,
			Model:                inference.ModelIdentity{Path: "/models/qwen"},
			ContextLength:        32768,
			ParallelSlots:        2,
			PromptCache:          true,
			PromptCacheMinTokens: 512,
			CachePolicy:          string(memory.KVCacheFull),
			CacheMode:            string(memory.KVCacheModeKQ8VQ4),
			BatchSize:            1,
			PrefillChunkSize:     1024,
			ExpectedQuantization: 4,
			MemoryLimitBytes:     8 << 30,
			CacheLimitBytes:      2 << 30,
			WiredLimitBytes:      1 << 30,
			Adapter:              inference.AdapterIdentity{Path: "/models/qwen/adapter"},
		},
	}
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		t.Fatalf("marshal profile: %v", data.Value)
	}
	profilePath := core.PathJoin(t.TempDir(), "coding-profile.json")
	if result := core.WriteFile(profilePath, data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("write profile: %v", result.Value)
	}

	var gotPath string
	var gotLoad mlx.LoadConfig
	var gotCfg bench.Config
	loadBenchModel = func(path string, opts ...mlx.LoadOption) (*mlx.Model, error) {
		gotPath = path
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		return &mlx.Model{}, nil
	}
	runBenchReport = func(_ context.Context, _ *mlx.Model, cfg bench.Config) (*bench.Report, error) {
		gotCfg = cfg
		return &bench.Report{
			Version:   bench.ReportVersion,
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Generation: bench.GenerationSummary{
				DecodeTokensPerSec: 42,
				PeakMemoryBytes:    2048,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"bench", "-json", "-profile", profilePath, "-prompt", "hi", "-max-tokens", "7", "-runs", "2"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotPath != "/models/qwen" || gotCfg.ModelPath != "/models/qwen" || gotCfg.Prompt != "hi" || gotCfg.MaxTokens != 7 || gotCfg.Runs != 2 {
		t.Fatalf("bench path=%q cfg=%+v", gotPath, gotCfg)
	}
	if gotLoad.ContextLength != 32768 || gotLoad.ParallelSlots != 2 || !gotLoad.PromptCache || gotLoad.PromptCacheMinTokens != 512 {
		t.Fatalf("profile prompt/context load = %+v", gotLoad)
	}
	if gotLoad.CachePolicy != memory.KVCacheFull || gotLoad.CacheMode != memory.KVCacheModeKQ8VQ4 || gotLoad.BatchSize != 1 || gotLoad.PrefillChunkSize != 1024 {
		t.Fatalf("profile cache/batch load = %+v", gotLoad)
	}
	if gotLoad.ExpectedQuantization != 4 || gotLoad.MemoryLimitBytes != 8<<30 || gotLoad.CacheLimitBytes != 2<<30 || gotLoad.WiredLimitBytes != 1<<30 {
		t.Fatalf("profile memory load = %+v", gotLoad)
	}
	if gotLoad.AdapterPath != "/models/qwen/adapter" || gotLoad.AutoMemoryPlan {
		t.Fatalf("profile adapter/planner load = %+v", gotLoad)
	}
	if !core.Contains(stdout.String(), `"decode_tokens_per_sec": 42`) || !core.Contains(stdout.String(), `"model_path": "/models/qwen"`) {
		t.Fatalf("stdout = %q, want JSON bench report", stdout.String())
	}
}

func TestRunCommand_DriverProfileProfileJSON_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			Model:    inference.ModelIdentity{Path: "/models/qwen"},
			Workload: inference.TuningWorkloadAgentState,
		},
		Candidate: inference.TuningCandidate{
			ID:                   "agent_state:paged:ctx32768:batch1",
			Workload:             inference.TuningWorkloadAgentState,
			Model:                inference.ModelIdentity{Path: "/models/qwen"},
			ContextLength:        32768,
			ParallelSlots:        2,
			PromptCache:          true,
			PromptCacheMinTokens: 512,
			CachePolicy:          string(memory.KVCacheFull),
			CacheMode:            string(memory.KVCacheModeKQ8VQ4),
			BatchSize:            1,
			PrefillChunkSize:     1024,
			ExpectedQuantization: 4,
			MemoryLimitBytes:     8 << 30,
			CacheLimitBytes:      2 << 30,
			WiredLimitBytes:      1 << 30,
		},
	}
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		t.Fatalf("marshal profile: %v", data.Value)
	}
	profilePath := core.PathJoin(t.TempDir(), "agent-profile.json")
	if result := core.WriteFile(profilePath, data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("write profile: %v", result.Value)
	}
	var gotPath string
	var gotLoad mlx.LoadConfig
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, loadOptions []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotPath = modelPath
		gotCfg = cfg
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range loadOptions {
			opt(&gotLoad)
		}
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Runs: []driverProfileRun{
				{
					Index:              1,
					Duration:           80 * time.Millisecond,
					RestoreDuration:    5 * time.Millisecond,
					FirstTokenDuration: 12 * time.Millisecond,
					StreamDuration:     68 * time.Millisecond,
					Output:             "Because retained state avoids replay.",
					Metrics: mlx.Metrics{
						PromptTokens:               17,
						GeneratedTokens:            8,
						PrefillDuration:            20 * time.Millisecond,
						DecodeDuration:             60 * time.Millisecond,
						TotalDuration:              80 * time.Millisecond,
						PromptCacheRestoreDuration: 5 * time.Millisecond,
						PrefillTokensPerSec:        850,
						DecodeTokensPerSec:         133.3,
						PeakMemoryBytes:            2048,
						ActiveMemoryBytes:          1024,
					},
				},
			},
			Summary: driverProfileSummary{
				SuccessfulRuns:            1,
				GeneratedTokens:           8,
				RestoreAvgDuration:        5 * time.Millisecond,
				RestoreMinDuration:        5 * time.Millisecond,
				RestoreMaxDuration:        5 * time.Millisecond,
				FirstTokenAvgDuration:     12 * time.Millisecond,
				DecodeTokensPerSecAverage: 133.3,
				PeakMemoryBytes:           2048,
				ActiveMemoryBytes:         1024,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-profile", profilePath, "-prompt", "Why does retained state matter?", "-max-tokens", "8", "-runs", "1", "-include-output"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotPath != "/models/qwen" || gotCfg.Prompt != "Why does retained state matter?" || gotCfg.MaxTokens != 8 || gotCfg.Runs != 1 || !gotCfg.IncludeOutput || !gotCfg.Chat {
		t.Fatalf("driver profile args path=%q cfg=%+v", gotPath, gotCfg)
	}
	if gotLoad.ContextLength != 32768 || gotLoad.ParallelSlots != 2 || !gotLoad.PromptCache || gotLoad.PromptCacheMinTokens != 512 {
		t.Fatalf("profile prompt/context load = %+v", gotLoad)
	}
	if gotLoad.CachePolicy != memory.KVCacheFull || gotLoad.CacheMode != memory.KVCacheModeKQ8VQ4 || gotLoad.BatchSize != 1 || gotLoad.PrefillChunkSize != 1024 {
		t.Fatalf("profile cache/batch load = %+v", gotLoad)
	}
	for _, want := range []string{
		`"model_path": "/models/qwen"`,
		`"prompt_bytes": 31`,
		`"restore_duration": 5000000`,
		`"restore_duration_average": 5000000`,
		`"first_token_duration": 12000000`,
		`"decode_tokens_per_sec": 133.3`,
		`"output": "Because retained state avoids replay."`,
		`"successful_runs": 1`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileReportFile_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Runs: []driverProfileRun{
				{
					Index:         1,
					Duration:      100 * time.Millisecond,
					VisibleTokens: 4,
					Metrics: mlx.Metrics{
						PromptTokens:        11,
						GeneratedTokens:     4,
						PrefillDuration:     10 * time.Millisecond,
						DecodeDuration:      90 * time.Millisecond,
						TotalDuration:       100 * time.Millisecond,
						PrefillTokensPerSec: 1100,
						DecodeTokensPerSec:  44.4,
					},
				},
			},
			Summary: driverProfileSummary{
				SuccessfulRuns:             1,
				GeneratedTokens:            4,
				VisibleTokens:              4,
				TotalDuration:              100 * time.Millisecond,
				PrefillTokensPerSecAverage: 1100,
				DecodeTokensPerSecAverage:  44.4,
			},
		}, nil
	}
	reportPath := core.PathJoin(t.TempDir(), "nested", "driver-profile.json")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-report-file", reportPath, "-prompt", "state smoke", "-max-tokens", "4", "-runs", "1", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	data := core.ReadFile(reportPath)
	if !data.OK {
		t.Fatalf("read report file: %v", data.Value)
	}
	text := string(data.Value.([]byte))
	if !core.Contains(text, `"model_path": "/models/demo"`) || !core.Contains(text, `"decode_tokens_per_sec_average": 44.4`) {
		t.Fatalf("report file = %q, want driver profile JSON", text)
	}
	if core.Contains(stdout.String(), `"model_path"`) {
		t.Fatalf("stdout = %q, did not want JSON without -json", stdout.String())
	}
	if !core.Contains(stdout.String(), "driver profile:") {
		t.Fatalf("stdout = %q, want human summary", stdout.String())
	}
}

func TestRunCommand_DriverProfileSpeculativeDraftModel_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotPath string
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotPath = modelPath
		gotCfg = cfg
		runs := []driverProfileRun{{
			Index:         1,
			Duration:      100 * time.Millisecond,
			VisibleTokens: 4,
			Metrics: mlx.Metrics{
				GeneratedTokens:    4,
				DecodeDuration:     80 * time.Millisecond,
				DecodeTokensPerSec: 50,
				PeakMemoryBytes:    2048,
				ActiveMemoryBytes:  1024,
				CacheMemoryBytes:   512,
				MTP: &mlx.MTPMetrics{
					DraftTokenSchedule:     []int{2, 2},
					ProposedTokens:         4,
					AcceptedTokens:         3,
					RejectedTokens:         1,
					TargetVerifyCalls:      2,
					DraftCalls:             2,
					AcceptanceRate:         0.75,
					VisibleTokensPerSec:    40,
					TargetTokensPerSec:     70,
					WarmDecodeTokensPerSec: 50,
				},
			},
		}}
		return &driverProfileReport{
			Version:                   1,
			ModelPath:                 modelPath,
			PromptBytes:               len(cfg.Prompt),
			MaxTokens:                 cfg.MaxTokens,
			RequestedRuns:             cfg.Runs,
			SpeculativeDraftModelPath: cfg.SpeculativeDraftModelPath,
			SpeculativeDraftTokens:    cfg.SpeculativeDraftTokens,
			Runs:                      runs,
			Summary:                   summariseDriverProfileRuns(runs),
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"driver-profile",
		"-json",
		"-prompt", "state smoke",
		"-max-tokens", "4",
		"-runs", "1",
		"-speculative-draft-model", "/models/target-assistant",
		"-speculative-draft-tokens", "2",
		"/models/target",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotPath != "/models/target" || gotCfg.SpeculativeDraftModelPath != "/models/target-assistant" || gotCfg.SpeculativeDraftTokens != 2 {
		t.Fatalf("driver profile speculative args path=%q cfg=%+v", gotPath, gotCfg)
	}
	for _, want := range []string{
		`"speculative_draft_model_path": "/models/target-assistant"`,
		`"speculative_draft_tokens": 2`,
		`"mtp_proposed_tokens": 4`,
		`"mtp_accepted_tokens": 3`,
		`"mtp_warm_decode_tokens_per_sec_average": 50`,
		`"draft_token_schedule": [`,
		`"proposed_tokens": 4`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileEstimatedPowerWatts_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		runs := []driverProfileRun{
			{
				Index:         1,
				Duration:      3 * time.Second,
				VisibleTokens: 10,
				Metrics: mlx.Metrics{
					GeneratedTokens:       10,
					PrefillDuration:       2 * time.Second,
					PromptCacheMisses:     1,
					PromptCacheMissTokens: 20,
					PrefillTokensPerSec:   10,
					DecodeTokensPerSec:    10,
					PeakMemoryBytes:       2048,
					ActiveMemoryBytes:     1024,
				},
			},
			{
				Index:           2,
				Duration:        time.Second,
				RestoreDuration: 100 * time.Millisecond,
				VisibleTokens:   10,
				Metrics: mlx.Metrics{
					GeneratedTokens:     10,
					PrefillDuration:     100 * time.Millisecond,
					PrefillTokensPerSec: 200,
					DecodeTokensPerSec:  10,
					PeakMemoryBytes:     2048,
					ActiveMemoryBytes:   1024,
				},
			},
		}
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Runs:          runs,
			Summary:       summariseDriverProfileRuns(runs),
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-estimate-power-watts", "50", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"method": "estimated_wall_clock_seconds_times_average_active_watts"`,
		`"power_watts": 50`,
		`"total_joules": 200`,
		`"joules_per_visible_token": 10`,
		`"prompt_setup_duration": 2100000000`,
		`"prompt_setup_joules": 105`,
		`"replay_prompt_setup_duration": 4000000000`,
		`"replay_prompt_setup_joules": 200`,
		`"prompt_setup_saved_duration": 1900000000`,
		`"prompt_setup_saved_joules": 95`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileEstimatedPowerWatts_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid estimated power watts")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-estimate-power-watts=-1", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stderr.String(), "estimated power watts must be >= 0") {
		t.Fatalf("stderr = %q, want estimated power validation", stderr.String())
	}
}

func TestSummariseDriverProfileRuns_DecodeBandwidthProxy_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{
		{
			Index:         1,
			Duration:      time.Second,
			VisibleTokens: 100,
			Metrics: mlx.Metrics{
				GeneratedTokens:    100,
				DecodeDuration:     time.Second,
				DecodeTokensPerSec: 100,
				ActiveMemoryBytes:  2_000_000_000,
				CacheMemoryBytes:   1_000_000_000,
			},
		},
	})

	proxy := summary.DecodeBandwidthProxy
	if proxy == nil {
		t.Fatal("DecodeBandwidthProxy = nil, want active+cache bandwidth proxy")
	}
	if proxy.ActivePlusCacheBytesPerDecodeTokenProxy != 3_000_000_000 {
		t.Fatalf("active bytes proxy = %d, want 3000000000", proxy.ActivePlusCacheBytesPerDecodeTokenProxy)
	}
	if proxy.ActivePlusCacheGBPerDecodeTokenProxy != 3 || proxy.ImpliedActivePlusCacheBandwidthGBPerSecProxy != 300 {
		t.Fatalf("proxy = %+v, want 3 GB/token and 300 GB/s", proxy)
	}
	if !core.Contains(proxy.Note, "proxy only") {
		t.Fatalf("proxy note = %q, want honest proxy label", proxy.Note)
	}
}

func TestSummariseStateRampProfileTurns_DecodeBandwidthProxy_Good(t *testing.T) {
	summary := summariseStateRampProfileTurns(0, 0, []stateRampProfileTurn{
		{
			Index:         1,
			Duration:      500 * time.Millisecond,
			VisibleTokens: 50,
			Metrics: mlx.Metrics{
				GeneratedTokens:   50,
				DecodeDuration:    500 * time.Millisecond,
				ActiveMemoryBytes: 3_000_000_000,
				CacheMemoryBytes:  2_000_000_000,
			},
		},
	}, stateRampProfileOptions{})

	proxy := summary.DecodeBandwidthProxy
	if proxy == nil {
		t.Fatal("DecodeBandwidthProxy = nil, want retained turn bandwidth proxy")
	}
	if summary.DecodeTokensPerSecAverage != 100 {
		t.Fatalf("DecodeTokensPerSecAverage = %f, want 100", summary.DecodeTokensPerSecAverage)
	}
	if proxy.ActivePlusCacheGBPerDecodeTokenProxy != 5 || proxy.ImpliedActivePlusCacheBandwidthGBPerSecProxy != 500 {
		t.Fatalf("proxy = %+v, want 5 GB/token and 500 GB/s", proxy)
	}
}

func TestRunCommand_StateRampProfileJSON_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	var gotCfg stateRampProfileOptions
	var gotLoad mlx.LoadConfig
	runStateRampProfile = func(_ context.Context, modelPath string, opts []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		gotCfg = cfg
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		turns := []stateRampProfileTurn{
			{
				Index:               1,
				TokensBeforeAppend:  30000,
				AppendedTokens:      8192,
				TokensAfterAppend:   38192,
				TokensAfterGenerate: 39216,
				AppendDuration:      2 * time.Second,
				Duration:            10 * time.Second,
				VisibleTokens:       1024,
				Metrics: mlx.Metrics{
					PromptTokens:        38192,
					GeneratedTokens:     1024,
					PrefillDuration:     32 * time.Second,
					DecodeDuration:      10 * time.Second,
					TotalDuration:       42 * time.Second,
					PrefillTokensPerSec: 1193.5,
					DecodeTokensPerSec:  102.4,
					PeakMemoryBytes:     4 << 30,
					ActiveMemoryBytes:   3 << 30,
					CacheMemoryBytes:    6 << 30,
				},
			},
		}
		return &stateRampProfileReport{
			Version:                   1,
			ModelPath:                 modelPath,
			PromptBytes:               len(cfg.Prompt),
			AppendPromptBytes:         len(cfg.AppendPrompt),
			ChatTemplate:              cfg.ChatTemplate,
			EnableThinking:            cfg.EnableThinking,
			SourceTokens:              2204,
			AppendSourceTokens:        512,
			StartTokens:               cfg.StartTokens,
			TargetTokens:              cfg.TargetTokens,
			CompactionThresholdTokens: cfg.CompactionThresholdTokens,
			CompactionTailTokens:      cfg.CompactionTailTokens,
			AppendTokens:              cfg.AppendTokens,
			TurnMaxTokens:             cfg.TurnMaxTokens,
			TurnMinTokens:             cfg.TurnMinTokens,
			TurnMinTokensPolicy:       cfg.TurnMinTokensPolicy,
			RequestedTurns:            cfg.Turns,
			Temperature:               cfg.Temperature,
			TopP:                      cfg.TopP,
			TopK:                      cfg.TopK,
			RepeatPenalty:             cfg.RepeatPenalty,
			SuppressEOS:               cfg.SuppressEOS,
			TraceTokenPhases:          cfg.TraceTokenPhases,
			RuntimeGates:              driverProfileRuntimeGates(),
			InitialPrefillDuration:    30 * time.Second,
			InitialPrefillTokens:      30000,
			Turns:                     turns,
			Summary:                   summariseStateRampProfileTurns(30*time.Second, 30000, turns, cfg),
		}, nil
	}
	appendPath := core.PathJoin(t.TempDir(), "append.txt")
	writeCLIPackFile(t, appendPath, "Review the changed files and explain the highest-risk performance regression.")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-json", "-append-file", appendPath, "-append-turn-delimiter", "---TURN---", "-chat-template", "gemma4", "-enable-thinking", "-turn-min-tokens", "512", "-turn-min-tokens-policy", "mark", "-suppress-eos", "-trace-token-phases", "-estimate-power-watts", "100", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.AppendPrompt != "Review the changed files and explain the highest-risk performance regression." {
		t.Fatalf("append prompt = %q, want append-file contents", gotCfg.AppendPrompt)
	}
	if gotCfg.AppendTurnDelimiter != "---TURN---" {
		t.Fatalf("append delimiter = %q, want configured delimiter", gotCfg.AppendTurnDelimiter)
	}
	if gotCfg.Prompt != mlx.DefaultNewSessionText {
		t.Fatalf("state ramp default prompt = %q, want Lemma new-session default", gotCfg.Prompt)
	}
	if gotCfg.ChatTemplate != "gemma4" || !gotCfg.EnableThinking {
		t.Fatalf("chat template = %q thinking=%v, want Gemma 4 thinking prompts", gotCfg.ChatTemplate, gotCfg.EnableThinking)
	}
	if gotCfg.StartTokens != 30000 || gotCfg.TargetTokens != 100000 || gotCfg.AppendTokens != 8192 || gotCfg.TurnMaxTokens != mlx.ProductionLaneLongFormMaxTokens {
		t.Fatalf("state ramp cfg = %+v, want default warm build-up shape", gotCfg)
	}
	if gotCfg.CompactionThresholdTokens != mlx.ProductionLaneHyperLongContextLength || gotCfg.CompactionTailTokens != 8192 {
		t.Fatalf("state ramp compaction cfg = threshold:%d tail:%d, want context-window folded-state defaults", gotCfg.CompactionThresholdTokens, gotCfg.CompactionTailTokens)
	}
	if gotCfg.FoldContinuePrompt != defaultStateRampFoldContinuePrompt || !core.Contains(gotCfg.FoldContinuePrompt, "The compacted State is live") {
		t.Fatalf("fold continue prompt = %q, want concise final-answer default", gotCfg.FoldContinuePrompt)
	}
	if gotCfg.TurnMinTokens != 512 || gotCfg.TurnMinTokensPolicy != "mark" || !gotCfg.SuppressEOS {
		t.Fatalf("state ramp debug annotation = min:%d policy:%q suppress_eos:%v, want configured debug threshold", gotCfg.TurnMinTokens, gotCfg.TurnMinTokensPolicy, gotCfg.SuppressEOS)
	}
	if !gotCfg.TraceTokenPhases {
		t.Fatalf("TraceTokenPhases = false, want retained turn phase tracing")
	}
	if gotCfg.Temperature != 1.0 || gotCfg.TopP != 0.95 || gotCfg.TopK != 64 || gotCfg.RepeatPenalty != 1.0 {
		t.Fatalf("state ramp sampling = temp:%f top_p:%f top_k:%d repeat:%f, want Gemma 4 defaults", gotCfg.Temperature, gotCfg.TopP, gotCfg.TopK, gotCfg.RepeatPenalty)
	}
	if gotLoad.ContextLength != mlx.ProductionLaneHyperLongContextLength || gotLoad.CacheMode != memory.KVCacheModePaged || gotLoad.PrefillChunkSize != mlx.ProductionLaneLongContextPrefillChunkSize {
		t.Fatalf("load = %+v, want hyper-long fast lane defaults", gotLoad)
	}
	for _, want := range []string{
		`"model_path": "/models/demo"`,
		`"start_tokens": 30000`,
		`"target_tokens": 100000`,
		`"turn_max_tokens": 8192`,
		`"compaction_threshold_tokens": 131072`,
		`"compaction_tail_tokens": 8192`,
		`"chat_template": "gemma4"`,
		`"enable_thinking": true`,
		`"turn_min_tokens": 512`,
		`"turn_min_tokens_policy": "mark"`,
		`"temperature": 1`,
		`"top_p": 0.95`,
		`"top_k": 64`,
		`"suppress_eos": true`,
		`"trace_token_phases": true`,
		`"retained_setup_duration": 32000000000`,
		`"replay_estimate_turns": 1`,
		`"replay_prefill_duration_estimate": 32000000000`,
		`"replay_total_duration_estimate": 42000000000`,
		`"append_tokens_per_sec_average": 4096`,
		`"decode_tokens_per_sec_average": 102.4`,
		`"effective_turn_tokens_per_sec_average":`,
		`"active_plus_cache_memory_bytes": 9663676416`,
		`"final_state_tokens": 39216`,
		`"total_joules": 4200`,
		`"append_joules": 200`,
		`"replay_total_joules_estimate": 4200`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should not contain default fixed-cache gate %s", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_StateRampProfileFixedCacheEnvOverride_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "0")
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		return &stateRampProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			TargetTokens: cfg.TargetTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: stateRampProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-json", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND": "1"`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK": "1"`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should not contain %s", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_StateRampProfileTargetShapeStaysPaged_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		return &stateRampProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			TargetTokens: cfg.TargetTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: stateRampProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-json", "-target-tokens", "100000", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should not contain target-shaped fixed-cache gate %s", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_StateRampProfileRequestedContextDoesNotSelectFixedCache_Good(t *testing.T) {
	for _, tc := range []struct {
		name       string
		contextLen int
	}{
		{name: "normal", contextLen: mlx.ProductionLaneContextLength},
		{name: "opencode", contextLen: mlx.ProductionLaneLongContextLength},
		{name: "workflow_target", contextLen: 100000},
		{name: "model_window", contextLen: mlx.ProductionLaneHyperLongContextLength},
	} {
		t.Run(tc.name, func(t *testing.T) {
			originalRun := runStateRampProfile
			t.Cleanup(func() { runStateRampProfile = originalRun })
			runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
				if cfg.CompactionThresholdTokens != tc.contextLen {
					t.Fatalf("compaction threshold = %d, want requested context %d", cfg.CompactionThresholdTokens, tc.contextLen)
				}
				return &stateRampProfileReport{
					Version:                   1,
					ModelPath:                 modelPath,
					TargetTokens:              cfg.TargetTokens,
					CompactionThresholdTokens: cfg.CompactionThresholdTokens,
					RuntimeGates:              driverProfileRuntimeGates(),
					Summary:                   stateRampProfileSummary{SuccessfulTurns: 1},
				}, nil
			}
			stdout, stderr := core.NewBuffer(), core.NewBuffer()
			contextText := core.Sprintf("%d", tc.contextLen)

			code := runCommand(context.Background(), []string{
				"state-ramp-profile",
				"-json",
				"-context", contextText,
				"-start-tokens", "30000",
				"-target-tokens", "100000",
				"/models/demo",
			}, stdout, stderr)

			if code != 0 {
				t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
			}
			for _, want := range []string{
				core.Sprintf(`"context_length": %d`, tc.contextLen),
				`"cache_mode": "paged"`,
			} {
				if !core.Contains(stdout.String(), want) {
					t.Fatalf("stdout = %q, want %s", stdout.String(), want)
				}
			}
			if tc.contextLen > mlx.ProductionLaneContextLength && !core.Contains(stdout.String(), `"prefill_chunk_size": 512`) {
				t.Fatalf("stdout = %q, want long-context prefill chunk", stdout.String())
			}
			for _, rejected := range []string{
				`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
				`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
				`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
				`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
			} {
				if core.Contains(stdout.String(), rejected) {
					t.Fatalf("stdout = %q, should not contain context-selected fixed-cache gate %s", stdout.String(), rejected)
				}
			}
		})
	}
}

func TestRunCommand_StateRampProfileFastLaneIgnoresFixedCacheEnv_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE", "1")
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", core.Sprintf("%d", mlx.ProductionLaneHyperLongContextLength))
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		return &stateRampProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			TargetTokens: cfg.TargetTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: stateRampProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-ramp-profile",
		"-json",
		"-start-tokens", "30000",
		"-target-tokens", "100000",
		"-turn-max-tokens", "1024",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
		`"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY":`,
		`"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION":`,
		`"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION":`,
		`"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE":`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should ignore ambient fixed-cache env %s in the fast lane", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_StateRampProfileValidation_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for invalid target")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-start-tokens", "30000", "-target-tokens", "30000", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "target tokens must be greater than start tokens") {
		t.Fatalf("stderr = %q, want target validation", stderr.String())
	}
}

func TestRunCommand_StateRampProfileMinPolicyValidation_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for invalid min-token policy")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-turn-min-tokens-policy", "continue", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "turn min tokens policy must be fail or mark") {
		t.Fatalf("stderr = %q, want min-token policy validation", stderr.String())
	}
}

func TestRunCommand_StateRampProfileCompactionValidation_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for invalid compaction options")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-compaction-threshold-tokens", "-1", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "compaction threshold tokens must be >= 0") {
		t.Fatalf("stderr = %q, want compaction threshold validation", stderr.String())
	}
}

func TestRunCommand_StateRampProfileFoldOptions_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	var gotCfg stateRampProfileOptions
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		gotCfg = cfg
		return &stateRampProfileReport{
			Version:                   1,
			ModelPath:                 modelPath,
			FoldStorePath:             cfg.FoldStorePath,
			FoldSummaryBytes:          len(cfg.FoldSummary),
			FoldRecentTailBytes:       len(cfg.FoldRecentTail),
			FoldPrefillChunkBytes:     cfg.FoldPrefillChunkBytes,
			FoldContinueMaxTokens:     cfg.FoldContinueMaxTokens,
			StartTokens:               cfg.StartTokens,
			TargetTokens:              cfg.TargetTokens,
			CompactionThresholdTokens: cfg.CompactionThresholdTokens,
			CompactionTailTokens:      cfg.CompactionTailTokens,
			Summary: stateRampProfileSummary{
				FinalStateTokens:          cfg.CompactionThresholdTokens,
				ContextExhausted:          true,
				FoldedStateRequired:       true,
				CompactionThresholdTokens: cfg.CompactionThresholdTokens,
				CompactionTailTokens:      cfg.CompactionTailTokens,
			},
			Fold: &stateRampProfileFold{
				Attempted:         true,
				StorePath:         cfg.FoldStorePath,
				SummaryBytes:      len(cfg.FoldSummary),
				RecentTailBytes:   len(cfg.FoldRecentTail),
				FoldedPromptBytes: 123,
			},
		}, nil
	}
	dir := t.TempDir()
	summaryPath := core.PathJoin(dir, "summary.txt")
	tailPath := core.PathJoin(dir, "tail.txt")
	storePath := core.PathJoin(dir, "state.mvlog")
	writeCLIPackFile(t, summaryPath, "summarised exhausted context")
	writeCLIPackFile(t, tailPath, "recent continuation tail")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-ramp-profile",
		"-json",
		"-fold-store", storePath,
		"-fold-summary-file", summaryPath,
		"-fold-tail-file", tailPath,
		"-fold-prefill-chunk-bytes", "4096",
		"-fold-continue-max-tokens", "640",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if gotCfg.FoldStorePath != storePath {
		t.Fatalf("fold cfg = %+v, want fold store available without forcing exhaustion fold", gotCfg)
	}
	if gotCfg.FoldSummary != "summarised exhausted context" || gotCfg.FoldRecentTail != "recent continuation tail" {
		t.Fatalf("fold text summary=%q tail=%q, want file contents", gotCfg.FoldSummary, gotCfg.FoldRecentTail)
	}
	if gotCfg.FoldPrefillChunkBytes != 4096 || gotCfg.FoldContinueMaxTokens != 640 {
		t.Fatalf("fold prefill/continue = %d/%d, want configured values", gotCfg.FoldPrefillChunkBytes, gotCfg.FoldContinueMaxTokens)
	}
	for _, want := range []string{
		`"fold_store_path": "` + storePath + `"`,
		`"fold_summary_bytes": 28`,
		`"fold_recent_tail_bytes": 24`,
		`"fold_prefill_chunk_bytes": 4096`,
		`"fold_continue_max_tokens": 640`,
		`"attempted": true`,
		`"folded_prompt_bytes": 123`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_StateRampProfileFoldSummaryGenerate_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	var gotCfg stateRampProfileOptions
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		gotCfg = cfg
		return &stateRampProfileReport{
			Version:                1,
			ModelPath:              modelPath,
			FoldStorePath:          cfg.FoldStorePath,
			FoldSummaryGenerate:    cfg.FoldSummaryGenerate,
			FoldSummaryPromptBytes: len(cfg.FoldSummaryPrompt),
			FoldSummaryMaxTokens:   cfg.FoldSummaryMaxTokens,
			Summary: stateRampProfileSummary{
				FinalStateTokens:    cfg.CompactionThresholdTokens,
				ContextExhausted:    true,
				FoldedStateRequired: true,
			},
			Fold: &stateRampProfileFold{
				Attempted:          true,
				StorePath:          cfg.FoldStorePath,
				SummaryMode:        "generated",
				SummaryPromptBytes: len(cfg.FoldSummaryPrompt),
				SummaryMaxTokens:   cfg.FoldSummaryMaxTokens,
				SummaryBytes:       512,
			},
		}, nil
	}
	dir := t.TempDir()
	promptPath := core.PathJoin(dir, "summary-prompt.txt")
	storePath := core.PathJoin(dir, "state.mvlog")
	summaryPrompt := "Summarise the retained book state for a fresh folded State."
	writeCLIPackFile(t, promptPath, summaryPrompt)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-ramp-profile",
		"-json",
		"-fold-store", storePath,
		"-fold-summary-generate",
		"-fold-summary-prompt-file", promptPath,
		"-fold-summary-max-tokens", "333",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !gotCfg.FoldSummaryGenerate || gotCfg.FoldSummaryPrompt != summaryPrompt || gotCfg.FoldSummaryMaxTokens != 333 {
		t.Fatalf("fold summary generation cfg = %+v, want generated prompt/max tokens", gotCfg)
	}
	for _, want := range []string{
		`"fold_summary_generate": true`,
		core.Sprintf(`"fold_summary_prompt_bytes": %d`, len(summaryPrompt)),
		`"fold_summary_max_tokens": 333`,
		`"summary_mode": "generated"`,
		`"summary_bytes": 512`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_StateRampProfileEmptySeedContext_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	var gotCfg stateRampProfileOptions
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		gotCfg = cfg
		return &stateRampProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			StartTokens:  cfg.StartTokens,
			TargetTokens: cfg.TargetTokens,
			Summary: stateRampProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-ramp-profile",
		"-json",
		"-prompt", "",
		"-start-tokens", "0",
		"-append-prompt", "Write the first answer from a blank session.",
		"-target-tokens", "64",
		"-turns", "1",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !gotCfg.PromptSet || gotCfg.Prompt != "" || gotCfg.StartTokens != 0 {
		t.Fatalf("empty prompt cfg = %+v, want explicit empty seed context", gotCfg)
	}
	for _, want := range []string{
		`"prompt_bytes": 0`,
		`"start_tokens": 0`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_StateRampProfileWakeMarker_Good(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	var gotCfg stateRampProfileOptions
	runStateRampProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateRampProfileOptions) (*stateRampProfileReport, error) {
		gotCfg = cfg
		return &stateRampProfileReport{
			Version:            1,
			ModelPath:          modelPath,
			WakeMarkerFile:     cfg.WakeMarkerFile,
			WakeStateStorePath: cfg.WakeStateStorePath,
			WakeIndexURI:       cfg.WakeIndexURI,
			Summary: stateRampProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	dir := t.TempDir()
	markerPath := core.PathJoin(dir, "marker.json")
	writeCLIPackFile(t, markerPath, `{
  "fold": {
    "compact_marker": {
      "store_path": "/tmp/session.mvlog",
      "index_uri": "mlx://state/folded/index",
      "entry_uri": "mlx://state/folded",
      "bundle_uri": "mlx://state/folded/bundle",
      "token_count": 1234
    }
  }
}`)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-json", "-wake-marker-file", markerPath, "-target-tokens", "4096", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if gotCfg.WakeMarkerFile != markerPath || gotCfg.WakeStateStorePath != "/tmp/session.mvlog" || gotCfg.WakeIndexURI != "mlx://state/folded/index" {
		t.Fatalf("wake cfg = %+v, want marker-derived store/index", gotCfg)
	}
	if gotCfg.StartTokens != 1234 {
		t.Fatalf("start tokens = %d, want marker token count", gotCfg.StartTokens)
	}
	for _, want := range []string{
		`"wake_marker_file": "` + markerPath + `"`,
		`"wake_state_store_path": "/tmp/session.mvlog"`,
		`"wake_index_uri": "mlx://state/folded/index"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_StateRampProfileFoldStoreValidation_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for missing fold store")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-fold-on-degradation", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "fold store path is required") {
		t.Fatalf("stderr = %q, want fold store validation", stderr.String())
	}
}

func TestRunCommand_StateRampProfileTurnForcedCompactionRemoved_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for removed fixed-turn compaction flag")
		return nil, nil
	}
	for _, flagName := range []string{"fold-after-turn", "compact-after-turn", "fold-on-exhaustion"} {
		t.Run(flagName, func(t *testing.T) {
			stdout, stderr := core.NewBuffer(), core.NewBuffer()

			code := runCommand(context.Background(), []string{"state-ramp-profile", "-" + flagName, "5", "/models/demo"}, stdout, stderr)

			if code != 2 {
				t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
			}
			want := "flag provided but not defined: -" + flagName
			if !core.Contains(stderr.String(), want) {
				t.Fatalf("stderr = %q, want removed flag validation %q", stderr.String(), want)
			}
		})
	}
}

func TestRunCommand_StateRampProfileDegradationMinConsecutiveValidation_Bad(t *testing.T) {
	originalRun := runStateRampProfile
	t.Cleanup(func() { runStateRampProfile = originalRun })
	runStateRampProfile = func(context.Context, string, []mlx.LoadOption, stateRampProfileOptions) (*stateRampProfileReport, error) {
		t.Fatal("runStateRampProfile called for invalid degradation fold options")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-ramp-profile", "-fold-on-degradation", "-degradation-min-consecutive-turns", "0", "-fold-store", "/tmp/state.mvlog", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "degradation min consecutive turns must be >= 1") {
		t.Fatalf("stderr = %q, want degradation min consecutive validation", stderr.String())
	}
}

func TestRunCommand_StateWakeProfileJSON_Good(t *testing.T) {
	originalRun := runStateWakeProfile
	t.Cleanup(func() { runStateWakeProfile = originalRun })
	var gotCfg stateWakeProfileOptions
	var gotLoad mlx.LoadConfig
	runStateWakeProfile = func(_ context.Context, modelPath string, opts []mlx.LoadOption, cfg stateWakeProfileOptions) (*stateWakeProfileReport, error) {
		gotCfg = cfg
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		return &stateWakeProfileReport{
			Version:        1,
			ModelPath:      modelPath,
			StateStorePath: cfg.StateStorePath,
			IndexURI:       cfg.IndexURI,
			PromptBytes:    len(cfg.Prompt),
			PromptTokens:   42,
			ChatTemplate:   cfg.ChatTemplate,
			EnableThinking: cfg.EnableThinking,
			MaxTokens:      cfg.MaxTokens,
			Temperature:    cfg.Temperature,
			TopP:           cfg.TopP,
			TopK:           cfg.TopK,
			RepeatPenalty:  cfg.RepeatPenalty,
			SuppressEOS:    cfg.SuppressEOS,
			IncludeOutput:  cfg.IncludeOutput,
			WakeDuration:   90 * time.Millisecond,
			StoreOpenMemoryDelta: &stateWakeMemoryDelta{
				GoTotalAllocDeltaBytes:    128,
				ProcessResidentDeltaBytes: 64,
			},
			WakeMemoryDelta: &stateWakeMemoryDelta{
				GoTotalAllocDeltaBytes:    4096,
				GoMallocsDelta:            12,
				ProcessResidentDeltaBytes: 2048,
			},
			Wake: &agent.WakeReport{
				IndexURI:        cfg.IndexURI,
				PrefixTokens:    677,
				BlocksRead:      3,
				RestoreStrategy: "folded-prefill",
			},
			Turn: &stateRampProfileTurn{
				Index:              1,
				TokensBeforeAppend: 677,
				AppendedTokens:     42,
				AppendDuration:     10 * time.Millisecond,
				Duration:           2 * time.Second,
				VisibleTokens:      128,
				Output:             "The compacted State is live; next action: run the wake-only degradation probe.",
				Metrics: mlx.Metrics{
					GeneratedTokens:            128,
					DecodeDuration:             2 * time.Second,
					DecodeTokensPerSec:         64,
					PeakMemoryBytes:            3 << 30,
					CacheMemoryBytes:           2 << 30,
					ProcessResidentMemoryBytes: 1 << 30,
					ProcessVirtualMemoryBytes:  5 << 30,
					ProcessPeakResidentBytes:   1 << 30,
					PromptCacheRestoreDuration: 90 * time.Millisecond,
				},
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-wake-profile",
		"-json",
		"-state-store", "/tmp/state.mvlog",
		"-index-uri", "mlx://state/folded/index",
		"-chat-template", "gemma4",
		"-enable-thinking",
		"-max-tokens", "256",
		"-temperature", "1",
		"-top-p", "0.95",
		"-top-k", "64",
		"-repeat-penalty", "1",
		"-suppress-eos",
		"-estimate-power-watts", "100",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.StateStorePath != "/tmp/state.mvlog" || gotCfg.IndexURI != "mlx://state/folded/index" {
		t.Fatalf("wake cfg state/index = %q/%q", gotCfg.StateStorePath, gotCfg.IndexURI)
	}
	if gotCfg.ChatTemplate != "gemma4" || !gotCfg.EnableThinking || gotCfg.MaxTokens != 256 || !gotCfg.SuppressEOS {
		t.Fatalf("wake cfg = %+v, want Gemma 4 wake prompt settings", gotCfg)
	}
	if gotLoad.ContextLength != mlx.ProductionLaneHyperLongContextLength || gotLoad.CacheMode != memory.KVCacheModePaged || gotLoad.PrefillChunkSize != mlx.ProductionLaneLongContextPrefillChunkSize {
		t.Fatalf("load = %+v, want hyper-long fast lane defaults", gotLoad)
	}
	for _, want := range []string{
		`"state_store_path": "/tmp/state.mvlog"`,
		`"index_uri": "mlx://state/folded/index"`,
		`"restore_strategy": "folded-prefill"`,
		`"prompt_tokens": 42`,
		`"max_tokens": 256`,
		`"decode_tokens_per_sec": 64`,
		`"total_joules": 210`,
		`"effective_tokens_per_sec":`,
		`"store_open_memory_delta":`,
		`"wake_memory_delta":`,
		`"go_total_alloc_delta_bytes": 4096`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestStateWakeMemoryDeltaBetween_Good(t *testing.T) {
	before := stateWakeMemorySample{
		goHeapAllocBytes:     4096,
		goHeapObjects:        30,
		goTotalAllocBytes:    8192,
		goMallocs:            100,
		goFrees:              40,
		activeMemoryBytes:    20_000,
		cacheMemoryBytes:     4_000,
		peakMemoryBytes:      50_000,
		processVirtualBytes:  100_000,
		processResidentBytes: 20_000,
		processPeakResident:  25_000,
	}
	after := stateWakeMemorySample{
		goHeapAllocBytes:     2048,
		goHeapObjects:        25,
		goTotalAllocBytes:    12288,
		goMallocs:            112,
		goFrees:              47,
		activeMemoryBytes:    24_000,
		cacheMemoryBytes:     2_000,
		peakMemoryBytes:      55_000,
		processVirtualBytes:  98_000,
		processResidentBytes: 21_024,
		processPeakResident:  27_000,
	}

	delta := stateWakeMemoryDeltaBetween(before, after)

	if delta.GoHeapAllocDeltaBytes != -2048 || delta.GoHeapObjectsDelta != -5 {
		t.Fatalf("go heap delta = %d/%d, want -2048/-5", delta.GoHeapAllocDeltaBytes, delta.GoHeapObjectsDelta)
	}
	if delta.GoTotalAllocDeltaBytes != 4096 || delta.GoMallocsDelta != 12 || delta.GoFreesDelta != 7 {
		t.Fatalf("go monotonic deltas = alloc:%d malloc:%d free:%d, want 4096/12/7", delta.GoTotalAllocDeltaBytes, delta.GoMallocsDelta, delta.GoFreesDelta)
	}
	if delta.ActiveMemoryDeltaBytes != 4000 || delta.CacheMemoryDeltaBytes != -2000 || delta.PeakMemoryDeltaBytes != 5000 {
		t.Fatalf("MLX deltas = active:%d cache:%d peak:%d, want 4000/-2000/5000", delta.ActiveMemoryDeltaBytes, delta.CacheMemoryDeltaBytes, delta.PeakMemoryDeltaBytes)
	}
	if delta.ProcessVirtualDeltaBytes != -2000 || delta.ProcessResidentDeltaBytes != 1024 || delta.ProcessPeakResidentDeltaBytes != 2000 {
		t.Fatalf("process deltas = virtual:%d resident:%d peak:%d, want -2000/1024/2000", delta.ProcessVirtualDeltaBytes, delta.ProcessResidentDeltaBytes, delta.ProcessPeakResidentDeltaBytes)
	}
}

func TestRunCommand_StateWakeProfileMarkerFile_Good(t *testing.T) {
	originalRun := runStateWakeProfile
	t.Cleanup(func() { runStateWakeProfile = originalRun })
	var gotCfg stateWakeProfileOptions
	runStateWakeProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateWakeProfileOptions) (*stateWakeProfileReport, error) {
		gotCfg = cfg
		return &stateWakeProfileReport{
			Version:        1,
			ModelPath:      modelPath,
			StateStorePath: cfg.StateStorePath,
			IndexURI:       cfg.IndexURI,
			MaxTokens:      cfg.MaxTokens,
			Wake: &agent.WakeReport{
				IndexURI:        cfg.IndexURI,
				PrefixTokens:    175,
				RestoreStrategy: "folded-prefill",
			},
			Turn: &stateRampProfileTurn{
				VisibleTokens: 8,
				Metrics: mlx.Metrics{
					GeneratedTokens:    8,
					DecodeDuration:     time.Second,
					DecodeTokensPerSec: 8,
				},
			},
		}, nil
	}
	dir := t.TempDir()
	markerPath := core.PathJoin(dir, "ramp-report.json")
	writeCLIPackFile(t, markerPath, `{
  "fold": {
    "compact_marker": {
      "store_path": "/tmp/session-1.mvlog",
      "index_uri": "mlx://state-ramp/fold/1/folded/index",
      "entry_uri": "mlx://state-ramp/fold/1/folded",
      "bundle_uri": "mlx://state-ramp/fold/1/folded/bundle",
      "token_count": 175
    }
  }
}`)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-wake-profile",
		"-json",
		"-marker-file", markerPath,
		"-max-tokens", "64",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.StateStorePath != "/tmp/session-1.mvlog" || gotCfg.IndexURI != "mlx://state-ramp/fold/1/folded/index" {
		t.Fatalf("wake cfg state/index = %q/%q, want marker values", gotCfg.StateStorePath, gotCfg.IndexURI)
	}
	for _, want := range []string{
		`"state_store_path": "/tmp/session-1.mvlog"`,
		`"index_uri": "mlx://state-ramp/fold/1/folded/index"`,
		`"max_tokens": 64`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestStateWakeProfileCompactMarkerFromPayload_FoldedFallback_Good(t *testing.T) {
	payload := stateWakeProfileMarkerFile{
		Fold: &stateWakeProfileMarkerFold{
			StorePath: "/tmp/older-report.mvlog",
			Folded: &agent.SleepReport{
				IndexURI:   "mlx://older/folded/index",
				EntryURI:   "mlx://older/folded",
				BundleURI:  "mlx://older/folded/bundle",
				TokenCount: 99,
			},
		},
	}

	marker := stateWakeProfileCompactMarkerFromPayload(payload)

	if marker.StorePath != "/tmp/older-report.mvlog" || marker.IndexURI != "mlx://older/folded/index" || marker.TokenCount != 99 {
		t.Fatalf("marker = %+v, want folded fallback", marker)
	}
}

func TestRunCommand_StateWakeProfileValidation_Bad(t *testing.T) {
	originalRun := runStateWakeProfile
	t.Cleanup(func() { runStateWakeProfile = originalRun })
	runStateWakeProfile = func(context.Context, string, []mlx.LoadOption, stateWakeProfileOptions) (*stateWakeProfileReport, error) {
		t.Fatal("runStateWakeProfile called for invalid input")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-wake-profile", "-state-store", "/tmp/state.mvlog", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "index URI is required") {
		t.Fatalf("stderr = %q, want index URI validation", stderr.String())
	}
}

func TestStateRampProfileOutputIssues_Good(t *testing.T) {
	issues := stateRampProfileOutputIssues("```text\nThe provided request is a directive to perform a comprehensive analysis. The output should function as a validation note.\n\n**Plan:**\n1. Continue.<|channel>thought\nhidden\n\nThe implementation is now officially complete and production-ready. Production Runner Wins Against Rivals because go-mlx demonstrates superior performance and a performance advantage over llama.cpp.")

	for _, want := range []string{"visible_chat_control_token", "visible_code_fence_prefix", "visible_prompt_analysis", "visible_plan_scaffold", "visible_false_completion_claim", "visible_unproven_performance_win_claim"} {
		if !core.SliceContains(issues, want) {
			t.Fatalf("issues = %v, want %s", issues, want)
		}
	}
}

func TestStateRampProfileOutputIssuesAllowsPerformanceGapDiscussion_Good(t *testing.T) {
	issues := stateRampProfileOutputIssues("The current row is still behind llama.cpp on raw decode, so the next validation step is to rerun request-context with captured output.")

	if core.SliceContains(issues, "visible_unproven_performance_win_claim") {
		t.Fatalf("issues = %v, want no win-claim tag for negative performance discussion", issues)
	}
}

func TestStateRampProfileOutputIssuesAllowsNegativeReadiness_Good(t *testing.T) {
	issues := stateRampProfileOutputIssues("The system is not yet production-ready because the next validation step is still open.")

	if core.SliceContains(issues, "visible_false_completion_claim") {
		t.Fatalf("issues = %v, want no false completion tag for negative readiness", issues)
	}
}

func TestStateRampProfileOutputIssuesRejectsReadyEcho_Good(t *testing.T) {
	issues := stateRampProfileOutputIssues("Ready.")

	if !core.SliceContains(issues, "visible_seed_ready_echo") {
		t.Fatalf("issues = %v, want visible_seed_ready_echo", issues)
	}
}

func TestStateRampProfileOutputIssuesRejectsFenceOnly_Good(t *testing.T) {
	issues := stateRampProfileOutputIssues("```\n```")

	if !core.SliceContains(issues, "visible_fence_only") {
		t.Fatalf("issues = %v, want visible_fence_only", issues)
	}
	issues = stateRampProfileOutputIssues("```go\nfmt.Println(1)\n```")
	if core.SliceContains(issues, "visible_fence_only") {
		t.Fatalf("issues = %v, want real fenced content allowed", issues)
	}
	if !core.SliceContains(issues, "visible_code_fence_prefix") {
		t.Fatalf("issues = %v, want fenced-prefix tag for benchmark-quality accounting", issues)
	}
}

func TestStateRampProfileOutputIssuesRejectsRepeatedTableCell_Good(t *testing.T) {
	builder := core.NewBuilder()
	builder.WriteString("| Llama.cpp | 1.14x")
	for i := 0; i < profileRepeatedTableCellLoopLimit; i++ {
		builder.WriteString(" | LLM")
	}
	builder.WriteString(" |")

	issues := stateRampProfileOutputIssues(builder.String())
	if !core.SliceContains(issues, "visible_repeated_table_cell") {
		t.Fatalf("issues = %v, want visible_repeated_table_cell", issues)
	}
	issues = stateRampProfileOutputIssues("| runner | speed |\n| --- | ---: |\n| go-mlx | 1.0x |\n| llama.cpp | 1.1x |")
	if core.SliceContains(issues, "visible_repeated_table_cell") {
		t.Fatalf("issues = %v, want normal compact table allowed", issues)
	}
}

func TestStateRampProfileOutputIssuesRejectsRepeatedTableRowLabel_Good(t *testing.T) {
	builder := core.NewBuilder()
	for i := 0; i < profileRepeatedTableRowLabelLoopLimit; i++ {
		builder.WriteString("| **Verdict** | repeated table row label |\n")
	}

	issues := stateRampProfileOutputIssues(builder.String())
	if !core.SliceContains(issues, "visible_repeated_table_row_label") {
		t.Fatalf("issues = %v, want visible_repeated_table_row_label", issues)
	}
	issues = stateRampProfileOutputIssues("| runner | speed |\n| --- | ---: |\n| go-mlx | 1.0x |\n| llama.cpp | 1.1x |")
	if core.SliceContains(issues, "visible_repeated_table_row_label") {
		t.Fatalf("issues = %v, want normal compact table allowed", issues)
	}
}

func TestStateRampProfileOutputIssuesRejectsRepeatedShortLineCycle_Good(t *testing.T) {
	builder := core.NewBuilder()
	builder.WriteString("The prose answer finishes, then the forced EOS suppression falls into punctuation.\n")
	for i := 0; i < profileRepeatedShortLineCycleLimit; i++ {
		if i%2 == 0 {
			builder.WriteString("\"")
		} else {
			builder.WriteString(")")
		}
		builder.WriteString("\n")
	}

	issues := stateRampProfileOutputIssues(builder.String())
	if !core.SliceContains(issues, "visible_repeated_short_line_cycle") {
		t.Fatalf("issues = %v, want visible_repeated_short_line_cycle", issues)
	}
	issues = stateRampProfileOutputIssues("A terse but valid answer.\nNo.\nNo.\nNo.")
	if core.SliceContains(issues, "visible_repeated_short_line_cycle") {
		t.Fatalf("issues = %v, want repeated words not treated as punctuation cycle", issues)
	}
	issues = stateRampProfileOutputIssues("Punctuation list:\n!\n?\n.\n,\n;\n:")
	if core.SliceContains(issues, "visible_repeated_short_line_cycle") {
		t.Fatalf("issues = %v, want varied punctuation list allowed", issues)
	}
}

func TestChapterProfileTemplateTokenControlsGemma4UsesAllModelStops_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	writeCLIPackFile(t, path, cliGemma4TokenizerJSON)
	tok, err := mlx.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	stops, suppress := chapterProfileTemplateTokenControls("gemma4", tok)

	for _, want := range []int32{1, 106, 50} {
		if !containsInt32(stops, want) {
			t.Fatalf("stop tokens = %v, want Gemma 4 EOS marker %d", stops, want)
		}
		if containsInt32(suppress, want) {
			t.Fatalf("suppress tokens = %v, should not suppress stop token %d", suppress, want)
		}
	}
	if !containsInt32(suppress, 105) {
		t.Fatalf("suppress tokens = %v, want opening turn marker suppressed", suppress)
	}
}

func TestStateRampProfileEffectiveSuppressTokenIDsIncludesGemma4EOSList_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	writeCLIPackFile(t, path, cliGemma4TokenizerJSON)
	tok, err := mlx.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	stops, suppress := chapterProfileTemplateTokenControls("gemma4", tok)

	got := stateRampProfileEffectiveSuppressTokenIDs(suppress, stops, tok, true)

	for _, want := range []int32{0, 1, 2, 50, 105, 106} {
		if !containsInt32(got, want) {
			t.Fatalf("effective suppress tokens = %v, want %d", got, want)
		}
	}
	if countInt32(got, 1) != 1 || countInt32(got, 106) != 1 || countInt32(got, 50) != 1 {
		t.Fatalf("effective suppress tokens = %v, want de-duplicated EOS markers", got)
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

func TestStateRampProfileSummary_OutputIssueCounts_Good(t *testing.T) {
	summary := summariseStateRampProfileTurns(0, 100, []stateRampProfileTurn{
		{Index: 1, OutputIssues: []string{"visible_prompt_analysis", "visible_code_fence_prefix"}},
		{Index: 2, OutputIssues: []string{"visible_prompt_analysis"}},
		{Index: 3},
	}, stateRampProfileOptions{})

	if summary.OutputIssueTurns != 2 {
		t.Fatalf("output issue turns = %d, want 2", summary.OutputIssueTurns)
	}
	if summary.OutputIssueCounts["visible_prompt_analysis"] != 2 || summary.OutputIssueCounts["visible_code_fence_prefix"] != 1 {
		t.Fatalf("output issue counts = %+v, want prompt=2 fence=1", summary.OutputIssueCounts)
	}
}

func TestStateRampProfileTurnPromptGemma4_Good(t *testing.T) {
	prompt := stateRampProfileTurnPrompt("gemma4", "User turn 3: Inspect the report.\n\n\treturn mem_", false)

	for _, want := range []string{
		"<|turn>user\n",
		"reference material, not as text to continue",
		"<turn_material>\n",
		"User turn 3: Inspect the report.",
		"</turn_material>",
		"Honour any requested output length before stopping.",
		"Do not continue or complete the reference excerpts.",
		"Do not explain, classify, plan, checklist, or restate",
		"Treat historical sign-off language as evidence to verify, not as current truth",
		"Prefer the unresolved risk and next validation step over a completion claim.",
		"<turn|>\n<|turn>model\n",
	} {
		if !core.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want %q", prompt, want)
		}
	}
	if core.Contains(prompt, "<|channel>thought\n<channel|>") {
		t.Fatalf("prompt = %q, should match native Gemma 4 generation prompt without synthetic thought channel", prompt)
	}
}

func TestStateRampProfileTurnPromptDirectGemma_Good(t *testing.T) {
	prompt := stateRampProfileDirectTurnPrompt("gemma", "Write Chapter 2 only.", false)

	for _, want := range []string{
		"<start_of_turn>user\n",
		"Write Chapter 2 only.",
		"<end_of_turn>\n<start_of_turn>model\n",
	} {
		if !core.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want %q", prompt, want)
		}
	}
	for _, rejected := range []string{
		"reference material",
		"<turn_material>",
		"Answer the user request from the turn material now",
	} {
		if core.Contains(prompt, rejected) {
			t.Fatalf("prompt = %q, should not contain wrapper text %q", prompt, rejected)
		}
	}
}

func TestStateRampProfileInitialPromptGemma4MatchesModelTemplate_Good(t *testing.T) {
	prompt := stateRampProfileInitialPrompt("gemma4", "Seed arc", false)
	want := "<bos><|turn>system\n" + defaultStateRampRetainedSystemPrompt + "\n\nSeed arc<turn|>\n<|turn>model\nReady.<turn|>\n"

	if prompt != want {
		t.Fatalf("prompt = %q, want native Gemma 4 retained-template shape %q", prompt, want)
	}
}

func TestStateRampProfileInitialPromptGemmaMatchesModelTemplate_Good(t *testing.T) {
	prompt := stateRampProfileInitialPrompt("gemma", "Seed arc", false)

	if !core.HasPrefix(prompt, "<bos><start_of_turn>user\n") {
		t.Fatalf("prompt = %q, want Gemma BOS user turn", prompt)
	}
	if !core.Contains(prompt, defaultStateRampRetainedSystemPrompt+"\n\nSeed arc<end_of_turn>") {
		t.Fatalf("prompt = %q, want system text folded before first user seed", prompt)
	}
	if !core.HasSuffix(prompt, "<start_of_turn>model\nReady.<end_of_turn>\n") {
		t.Fatalf("prompt = %q, want ready assistant history turn", prompt)
	}
}

func TestStateRampProfileTurnPromptVisibleFloor_Good(t *testing.T) {
	prompt := stateRampProfileTurnPrompt("gemma4", "Review the latest turn.", false, 256)

	for _, rejected := range []string{
		"write at least 256 visible tokens",
		"expand with concrete evidence",
	} {
		if core.Contains(prompt, rejected) {
			t.Fatalf("prompt = %q, should not contain debug-floor steering %q", prompt, rejected)
		}
	}
	if !core.Contains(prompt, "Answer the user request from the turn material now") {
		t.Fatalf("prompt = %q, want normal reference-turn instruction", prompt)
	}
	if core.Contains(prompt, "answer as the engineer") {
		t.Fatalf("prompt = %q, should not force creative/book turns into engineering-analysis mode", prompt)
	}
	for _, rejected := range []string{"Do not explain, classify, plan, checklist, or restate", "write only the requested output"} {
		if !core.Contains(prompt, rejected) {
			t.Fatalf("prompt = %q, want anti-analysis guard %q", prompt, rejected)
		}
	}
}

func TestStateRampProfileVisibleOutputGemma4_Good(t *testing.T) {
	output := stateRampProfileVisibleOutput("gemma4", "Visible before<|channel>thought\nhidden<channel|>Visible after<turn|>")

	if output != "Visible beforeVisible after" {
		t.Fatalf("output = %q, want visible Gemma 4 content only", output)
	}
}

func TestForEachRepeatedStateRampTokenSpanWrapped_Good(t *testing.T) {
	source := []int32{1, 2, 3, 4}
	var got []int32
	spans := 0

	count, err := forEachRepeatedStateRampTokenSpan(source, 3, 6, func(tokens []int32) error {
		spans++
		got = append(got, tokens...)
		return nil
	})
	if err != nil {
		t.Fatalf("forEachRepeatedStateRampTokenSpan() error = %v", err)
	}
	if count != 6 {
		t.Fatalf("count = %d, want 6", count)
	}
	if spans != 3 {
		t.Fatalf("spans = %d, want 3 wrapped spans", spans)
	}
	want := []int32{4, 1, 2, 3, 4, 1}
	if len(got) != len(want) {
		t.Fatalf("got = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got = %v, want %v", got, want)
		}
	}
}

func TestStateRampProfileTurnAppendSourceDelimited_Good(t *testing.T) {
	section := []int32{1, 2, 3, 4, 5}
	source, offset, count := stateRampProfileTurnAppendSource(
		[]int32{9, 9, 9},
		[][]int32{section},
		12,
		100,
		1,
		stateRampProfileOptions{AppendTokens: 2, TargetTokens: 1000},
	)

	if offset != 0 || count != len(section) {
		t.Fatalf("offset=%d count=%d, want whole delimited section", offset, count)
	}
	if len(source) != len(section) || source[0] != 1 || source[len(source)-1] != 5 {
		t.Fatalf("source=%v, want selected delimited section", source)
	}
}

func TestStateRampProfileTurnAppendSourceDelimitedNearTarget_Good(t *testing.T) {
	section := []int32{1, 2, 3, 4, 5}
	_, _, count := stateRampProfileTurnAppendSource(
		[]int32{9, 9, 9},
		[][]int32{section},
		0,
		998,
		1,
		stateRampProfileOptions{AppendTokens: 2, TargetTokens: 1000},
	)

	if count != len(section) {
		t.Fatalf("count=%d, want whole delimited section even near target", count)
	}
}

func TestStateRampProfileTurnAppendSourceDoesNotUseUnarmedCompactionThreshold_Good(t *testing.T) {
	_, _, count := stateRampProfileTurnAppendSource(
		[]int32{1, 2, 3, 4, 5},
		nil,
		0,
		950,
		1,
		stateRampProfileOptions{
			AppendTokens:              200,
			TargetTokens:              2000,
			CompactionThresholdTokens: 1000,
		},
	)

	if count != 200 {
		t.Fatalf("count=%d, want benchmark append target without unarmed compaction cutoff", count)
	}
}

func TestStateRampProfileTurnAppendSourceFoldStoreArmsCompactionThreshold_Good(t *testing.T) {
	_, _, count := stateRampProfileTurnAppendSource(
		[]int32{1, 2, 3, 4, 5},
		nil,
		0,
		950,
		1,
		stateRampProfileOptions{
			AppendTokens:              200,
			TargetTokens:              2000,
			CompactionThresholdTokens: 1000,
			FoldStorePath:             "/tmp/state.mvlog",
		},
	)

	if count != 50 {
		t.Fatalf("count=%d, want overflow fold store to cap append at compaction threshold", count)
	}
}

func TestStateRampProfileTurnErrorFatal_Good(t *testing.T) {
	turn := stateRampProfileTurn{Error: "short turn", BelowMinTokens: true}
	if stateRampProfileTurnErrorFatal(turn, stateRampProfileOptions{TurnMinTokensPolicy: "mark"}) {
		t.Fatal("debug-floor turn with mark policy is fatal")
	}
	if !stateRampProfileTurnErrorFatal(turn, stateRampProfileOptions{TurnMinTokensPolicy: "fail"}) {
		t.Fatal("debug-floor turn with fail policy is non-fatal")
	}
	if !stateRampProfileTurnErrorFatal(stateRampProfileTurn{Error: "loop"}, stateRampProfileOptions{TurnMinTokensPolicy: "mark"}) {
		t.Fatal("non-floor error with mark policy is non-fatal")
	}
}

func TestStateRampProfileDegradationFoldReached_Good(t *testing.T) {
	opts := stateRampProfileOptions{
		FoldOnDegradation:         true,
		DegradationMinConsecutive: 2,
	}
	if stateRampProfileDegradationFoldReached(1, opts) {
		t.Fatal("single output-issue turn triggered degradation fold")
	}
	if !stateRampProfileDegradationFoldReached(2, opts) {
		t.Fatal("two consecutive output-issue turns did not trigger degradation fold")
	}
	opts.FoldOnDegradation = false
	if stateRampProfileDegradationFoldReached(2, opts) {
		t.Fatal("disabled degradation fold still triggered")
	}
}

func TestStateRampProfileApplyVisibleTokenFloorPreservesClosedTurn_Good(t *testing.T) {
	turn := stateRampProfileTurn{
		Index:               7,
		VisibleTokens:       12,
		TurnCloseTokens:     2,
		TokensAfterGenerate: 1024,
	}

	stateRampProfileApplyVisibleTokenFloor(&turn, stateRampProfileOptions{TurnMinTokens: 256, TurnMinTokensPolicy: "mark"})

	if !turn.BelowMinTokens {
		t.Fatal("debug-floor turn was not marked")
	}
	if turn.TurnCloseTokens != 2 || turn.TokensAfterGenerate != 1024 {
		t.Fatalf("turn close state changed: %+v", turn)
	}
	if turn.Error != "" {
		t.Fatalf("error = %q, want mark-only debug annotation", turn.Error)
	}
	if len(turn.OutputIssues) != 1 || turn.OutputIssues[0] != "below_debug_visible_token_floor:12/256" {
		t.Fatalf("output issues = %v, want debug token-floor annotation", turn.OutputIssues)
	}
	if stateRampProfileTurnErrorFatal(turn, stateRampProfileOptions{TurnMinTokensPolicy: "mark"}) {
		t.Fatal("marked debug-floor closed turn is fatal")
	}
}

func TestStateRampProfileContextLifecycle_Good(t *testing.T) {
	opts := stateRampProfileOptions{
		TargetTokens:              2000,
		CompactionThresholdTokens: 1000,
		CompactionTailTokens:      128,
		Turns:                     10,
		FoldStorePath:             "/tmp/state.mvlog",
	}
	if !shouldRunStateRampTurn(1, 999, opts) {
		t.Fatal("turn before compaction threshold does not run")
	}
	if shouldRunStateRampTurn(2, 1000, opts) {
		t.Fatal("turn at compaction threshold still runs")
	}

	summary := summariseStateRampProfileTurns(time.Second, 900, []stateRampProfileTurn{
		{
			Index:               1,
			TokensAfterGenerate: 1000,
			VisibleTokens:       100,
			Metrics: mlx.Metrics{
				GeneratedTokens: 100,
				DecodeDuration:  time.Second,
			},
		},
	}, opts)

	if !summary.ContextExhausted || !summary.FoldedStateRequired {
		t.Fatalf("summary lifecycle = exhausted:%v folded:%v, want folded-state boundary", summary.ContextExhausted, summary.FoldedStateRequired)
	}
	if summary.CompactionThresholdTokens != 1000 || summary.CompactionTailTokens != 128 {
		t.Fatalf("summary compaction = threshold:%d tail:%d, want configured values", summary.CompactionThresholdTokens, summary.CompactionTailTokens)
	}
	if !core.Contains(summary.CompactionReason, "prefill a folded state") {
		t.Fatalf("compaction reason = %q, want folded-state instruction", summary.CompactionReason)
	}
}

func TestStateRampProfileContextLifecycle_TargetBelowWindowDoesNotFold_Good(t *testing.T) {
	opts := stateRampProfileOptions{
		TargetTokens:              100000,
		CompactionThresholdTokens: mlx.ProductionLaneHyperLongContextLength,
		CompactionTailTokens:      8192,
		Turns:                     10,
	}
	if !shouldRunStateRampTurn(1, 99999, opts) {
		t.Fatal("turn before benchmark target does not run")
	}
	if shouldRunStateRampTurn(2, 100000, opts) {
		t.Fatal("turn at benchmark target still runs")
	}

	summary := summariseStateRampProfileTurns(time.Second, 90000, []stateRampProfileTurn{
		{
			Index:               1,
			TokensAfterGenerate: 100000,
			VisibleTokens:       100,
			Metrics: mlx.Metrics{
				GeneratedTokens: 100,
				DecodeDuration:  time.Second,
			},
		},
	}, opts)

	if summary.ContextExhausted || summary.FoldedStateRequired {
		t.Fatalf("summary lifecycle = exhausted:%v folded:%v, want benchmark target without overflow fold", summary.ContextExhausted, summary.FoldedStateRequired)
	}
	if summary.CompactionThresholdTokens != mlx.ProductionLaneHyperLongContextLength {
		t.Fatalf("summary compaction threshold = %d, want context window", summary.CompactionThresholdTokens)
	}
	if summary.CompactionReason != "" {
		t.Fatalf("compaction reason = %q, want no fold at benchmark target", summary.CompactionReason)
	}
}

func TestStateRampProfileShouldRunFold_OverflowStoreWithoutForce_Good(t *testing.T) {
	exhausted := stateRampProfileSummary{
		ContextExhausted:    true,
		FoldedStateRequired: true,
	}
	if !stateRampProfileShouldRunFold(exhausted, stateRampProfileOptions{FoldStorePath: "/tmp/state.mvlog"}) {
		t.Fatal("fold store at exhausted context did not run overflow compaction")
	}
	if stateRampProfileShouldRunFold(stateRampProfileSummary{}, stateRampProfileOptions{FoldStorePath: "/tmp/state.mvlog"}) {
		t.Fatal("fold store below context window ran compaction")
	}
	if stateRampProfileShouldRunFold(exhausted, stateRampProfileOptions{}) {
		t.Fatal("overflow compaction ran without a fold store")
	}
}

func TestStateRampProfileDefaultCompactionThresholdUsesModelContext_Good(t *testing.T) {
	opts := stateRampProfileOptions{TargetTokens: 100000}

	got := stateRampProfileDefaultCompactionThreshold(opts, mlx.ModelInfo{ContextLength: mlx.ProductionLaneHyperLongContextLength})

	if got != mlx.ProductionLaneHyperLongContextLength {
		t.Fatalf("default compaction threshold = %d, want model context window", got)
	}
	opts.CompactionThresholdTokens = 90000
	if got := stateRampProfileDefaultCompactionThreshold(opts, mlx.ModelInfo{ContextLength: mlx.ProductionLaneHyperLongContextLength}); got != 90000 {
		t.Fatalf("explicit compaction threshold = %d, want 90000", got)
	}
}

func TestStateRampProfileSummary_ReplayEstimate_Good(t *testing.T) {
	turns := []stateRampProfileTurn{
		{
			Index:          1,
			AppendDuration: time.Second,
			Duration:       2 * time.Second,
			VisibleTokens:  10,
			Metrics: mlx.Metrics{
				GeneratedTokens:   10,
				PrefillDuration:   5 * time.Second,
				DecodeDuration:    2 * time.Second,
				ActiveMemoryBytes: 1024,
			},
		},
		{
			Index:          2,
			AppendDuration: time.Second,
			Duration:       2 * time.Second,
			VisibleTokens:  10,
			Metrics: mlx.Metrics{
				GeneratedTokens: 10,
				PrefillDuration: 9 * time.Second,
				DecodeDuration:  2 * time.Second,
			},
		},
	}

	summary := summariseStateRampProfileTurns(4*time.Second, 1000, turns, stateRampProfileOptions{TargetTokens: 2000})

	if summary.RetainedSetupDuration != 6*time.Second {
		t.Fatalf("retained setup = %s, want 6s", summary.RetainedSetupDuration)
	}
	if summary.ReplayEstimateTurns != 2 || summary.ReplayPrefillDuration != 14*time.Second {
		t.Fatalf("replay estimate turns=%d prefill=%s, want 2 turns and 14s", summary.ReplayEstimateTurns, summary.ReplayPrefillDuration)
	}
	if summary.ReplayTotalDuration != 18*time.Second {
		t.Fatalf("replay total = %s, want 18s", summary.ReplayTotalDuration)
	}
	if summary.ReplayPrefillSavedDuration != 8*time.Second || summary.ReplayTotalSavedDuration != 8*time.Second {
		t.Fatalf("replay savings = prefill:%s total:%s, want 8s/8s", summary.ReplayPrefillSavedDuration, summary.ReplayTotalSavedDuration)
	}
	if summary.RetainedVsReplaySpeedup < 1.79 || summary.RetainedVsReplaySpeedup > 1.81 {
		t.Fatalf("replay speedup = %f, want 1.8", summary.RetainedVsReplaySpeedup)
	}
}

func TestStateRampProfileSummary_TokenPhaseBuckets_Good(t *testing.T) {
	summary := summariseStateRampProfileTurns(time.Second, 1000, []stateRampProfileTurn{
		{
			Index:         1,
			VisibleTokens: 2,
			Metrics: mlx.Metrics{
				GeneratedTokens: 2,
				DecodeDuration:  30 * time.Millisecond,
				TokenPhases: []mlx.TokenPhaseTrace{
					{
						TotalDuration:      10 * time.Millisecond,
						ForwardDuration:    8 * time.Millisecond,
						PrefetchDuration:   time.Millisecond,
						SampleEvalDuration: time.Millisecond,
						NativeEvents: []mlx.NativePhaseTrace{
							{Name: "gemma4.layer.00.attention", Duration: 2 * time.Millisecond, Pages: 2, Tokens: 2048},
						},
					},
					{
						TotalDuration:      20 * time.Millisecond,
						ForwardDuration:    18 * time.Millisecond,
						PrefetchDuration:   time.Millisecond,
						SampleEvalDuration: time.Millisecond,
						NativeEvents: []mlx.NativePhaseTrace{
							{Name: "gemma4.layer.01.attention", Duration: 3 * time.Millisecond, Pages: 4, Tokens: 4096},
							{Name: "gemma4.layer.01.ffn_router", Duration: time.Millisecond},
						},
					},
				},
			},
		},
	}, stateRampProfileOptions{TargetTokens: 2000})

	if len(summary.TokenPhases) < 3 {
		t.Fatalf("token phases = %+v, want total/forward/sample_eval buckets", summary.TokenPhases)
	}
	if summary.TokenPhases[0].Name != "total" || summary.TokenPhases[0].Duration != 30*time.Millisecond || summary.TokenPhases[0].AverageDuration != 15*time.Millisecond {
		t.Fatalf("total phase = %+v, want 30ms total and 15ms average", summary.TokenPhases[0])
	}
	if summary.TokenPhases[1].Name != "forward" || summary.TokenPhases[1].Duration != 26*time.Millisecond || summary.TokenPhases[1].AverageDuration != 13*time.Millisecond {
		t.Fatalf("forward phase = %+v, want 26ms total and 13ms average", summary.TokenPhases[1])
	}
	if len(summary.NativeEvents) != 2 {
		t.Fatalf("native events = %+v, want attention and router buckets", summary.NativeEvents)
	}
	if summary.NativeEvents[0].Name != "attention" || summary.NativeEvents[0].Duration != 5*time.Millisecond || summary.NativeEvents[0].AverageDuration != 2500*time.Microsecond {
		t.Fatalf("attention events = %+v, want combined attention bucket", summary.NativeEvents[0])
	}
	if summary.NativeEvents[0].MaxPages != 4 || summary.NativeEvents[0].MaxTokens != 4096 {
		t.Fatalf("attention event pages/tokens = %+v, want max 4 pages and 4096 tokens", summary.NativeEvents[0])
	}
	if len(summary.NativeEventDetails) != 3 {
		t.Fatalf("native event details = %+v, want three layer-level events", summary.NativeEventDetails)
	}
	if summary.NativeEventDetails[0].Name != "gemma4.layer.01.attention" || summary.NativeEventDetails[0].Duration != 3*time.Millisecond {
		t.Fatalf("native event detail[0] = %+v, want layer 01 attention first", summary.NativeEventDetails[0])
	}
}

func TestStateRampProfileContentDegradationLifecycle_Good(t *testing.T) {
	opts := stateRampProfileOptions{
		TargetTokens:              100000,
		CompactionThresholdTokens: 100000,
		CompactionTailTokens:      8192,
		FoldOnDegradation:         true,
		DegradationMinConsecutive: 2,
	}
	summary := summariseStateRampProfileTurns(time.Second, 30000, []stateRampProfileTurn{
		{
			Index:               1,
			TokensAfterGenerate: 91000,
			VisibleTokens:       512,
			Metrics: mlx.Metrics{
				GeneratedTokens: 512,
				DecodeDuration:  time.Second,
			},
		},
		{
			Index:               2,
			TokensAfterGenerate: 97000,
			VisibleTokens:       160,
			OutputIssues:        []string{"visible_chat_control_token"},
			Metrics: mlx.Metrics{
				GeneratedTokens: 160,
				DecodeDuration:  time.Second,
			},
		},
		{
			Index:               3,
			TokensAfterGenerate: 99000,
			VisibleTokens:       142,
			OutputIssues:        []string{"visible_prompt_analysis"},
			Metrics: mlx.Metrics{
				GeneratedTokens: 142,
				DecodeDuration:  time.Second,
			},
		},
	}, opts)

	if summary.ContextExhausted {
		t.Fatal("content degradation incorrectly marked context exhausted")
	}
	if !summary.ContentDegraded || !summary.FoldedStateRequired {
		t.Fatalf("summary degradation = degraded:%v folded:%v, want degradation fold boundary", summary.ContentDegraded, summary.FoldedStateRequired)
	}
	if summary.ContentDegradationTurn != 3 || summary.ContentDegradationStreak != 2 {
		t.Fatalf("degradation = turn:%d streak:%d, want turn 3 streak 2", summary.ContentDegradationTurn, summary.ContentDegradationStreak)
	}
	if !core.Contains(summary.CompactionReason, "output-issue turns") {
		t.Fatalf("compaction reason = %q, want output-issue degradation reason", summary.CompactionReason)
	}
}

func TestStateRampProfileFoldBody_Good(t *testing.T) {
	body := stateRampProfileFoldBody("keep the architectural decision log", "last user asked for chapter 12")

	for _, want := range []string{
		"compacted into this folded state",
		"<summary>",
		"keep the architectural decision log",
		"<recent_tail>",
		"last user asked for chapter 12",
		"Do not assume the full exhausted context is still present.",
	} {
		if !core.Contains(body, want) {
			t.Fatalf("body = %q, want %q", body, want)
		}
	}
}

func TestStateRampProfileFoldDurations_Good(t *testing.T) {
	report := &stateRampProfileReport{
		Summary: stateRampProfileSummary{
			TotalDuration: 10 * time.Second,
		},
		Fold: &stateRampProfileFold{
			Duration:     time.Second,
			WakeDuration: 2 * time.Second,
			ContinueTurn: &stateRampProfileTurn{
				AppendDuration: 3 * time.Second,
				Duration:       4 * time.Second,
			},
		},
	}

	annotateStateRampProfileFoldDurations(report)

	if report.Fold.LifecycleDuration != 10*time.Second {
		t.Fatalf("fold lifecycle = %s, want 10s", report.Fold.LifecycleDuration)
	}
	if report.Fold.TotalWithRetained != 20*time.Second {
		t.Fatalf("retained total with fold = %s, want 20s", report.Fold.TotalWithRetained)
	}
}

func TestPrintStateRampProfileSummary_FoldLifecycle_Good(t *testing.T) {
	report := &stateRampProfileReport{
		ModelPath: "model",
		Summary: stateRampProfileSummary{
			SuccessfulTurns:            1,
			GeneratedTokens:            16,
			DecodeTokensPerSecAverage:  8,
			EffectiveTurnTokensPerSec:  4,
			TotalDuration:              4 * time.Second,
			CompactionThresholdTokens:  100,
			CompactionTailTokens:       16,
			ContextExhausted:           true,
			ActivePlusCacheMemoryBytes: 1024,
		},
		Fold: &stateRampProfileFold{
			Attempted:         true,
			StorePath:         "state.mvlog",
			StoreAction:       "append",
			CompactMarker:     &stateRampFoldMarker{IndexURI: "mlx://state/folded/index"},
			Duration:          time.Second,
			WakeDuration:      2 * time.Second,
			LifecycleDuration: 6 * time.Second,
			ContinueTurn: &stateRampProfileTurn{
				VisibleTokens: 4,
				Duration:      3 * time.Second,
				Metrics: mlx.Metrics{
					DecodeTokensPerSec: 1.25,
				},
			},
		},
	}
	out := core.NewBuffer()

	printStateRampProfileSummary(out, report)

	for _, want := range []string{
		"generated: 16 tokens, decode: 8.0 tok/s",
		"folded state: state.mvlog in 1s, wake 2s, continue 4 tokens in 3s at 1.2 tok/s, fold lifecycle 6s",
		"store append, compact marker mlx://state/folded/index",
	} {
		if !core.Contains(out.String(), want) {
			t.Fatalf("summary output = %q, want %q", out.String(), want)
		}
	}
}

func TestStateRampProfileFoldRecentTail_Good(t *testing.T) {
	report := &stateRampProfileReport{
		Turns: []stateRampProfileTurn{
			{Index: 1, Output: "first"},
			{Index: 2, Output: "second"},
			{Index: 3, Output: "third"},
			{Index: 4, Output: "fourth"},
		},
	}

	tail := stateRampProfileFoldRecentTail(report, stateRampProfileOptions{})

	if core.Contains(tail, "Turn 1 output") {
		t.Fatalf("tail = %q, want only the latest three turns", tail)
	}
	for _, want := range []string{"Turn 2 output", "second", "Turn 3 output", "third", "Turn 4 output", "fourth"} {
		if !core.Contains(tail, want) {
			t.Fatalf("tail = %q, want %q", tail, want)
		}
	}
	if !core.Contains(tail, "Turn 2 output:\nsecond\n\nTurn 3 output:\nthird\n\nTurn 4 output:\nfourth") {
		t.Fatalf("tail = %q, want chronological order", tail)
	}
}

func TestRunCommand_DriverProfileTraceTokenPhases_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			MaxTokens:        cfg.MaxTokens,
			RequestedRuns:    cfg.Runs,
			TraceTokenPhases: cfg.TraceTokenPhases,
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-trace-token-phases", "-prompt", "hi", "-max-tokens", "2", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !gotCfg.TraceTokenPhases {
		t.Fatalf("TraceTokenPhases = false, want true; cfg=%+v", gotCfg)
	}
	if !core.Contains(stdout.String(), `"trace_token_phases": true`) {
		t.Fatalf("stdout = %q, want trace flag in JSON report", stdout.String())
	}
}

func TestRunCommand_DriverProfilePromptFile_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:     1,
			ModelPath:   modelPath,
			PromptBytes: len(cfg.Prompt),
			MaxTokens:   cfg.MaxTokens,
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	dir := t.TempDir()
	promptPath := core.PathJoin(dir, "prompt.txt")
	writeCLIPackFile(t, promptPath, "file prompt body")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt-file", promptPath, "-max-tokens", "2", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Prompt != "file prompt body" {
		t.Fatalf("Prompt = %q, want prompt file body", gotCfg.Prompt)
	}
}

func TestRunCommand_DriverProfilePromptRepeat_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			PromptRepeat: cfg.PromptRepeat,
			MaxTokens:    cfg.MaxTokens,
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt", "alpha", "-prompt-repeat", "3", "-max-tokens", "2", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Prompt != "alpha\n\nalpha\n\nalpha" {
		t.Fatalf("Prompt = %q, want repeated prompt", gotCfg.Prompt)
	}
	if gotCfg.PromptRepeat != 3 {
		t.Fatalf("PromptRepeat = %d, want 3", gotCfg.PromptRepeat)
	}
	if !core.Contains(stdout.String(), `"prompt_repeat": 3`) {
		t.Fatalf("stdout = %q, want prompt repeat", stdout.String())
	}
}

func TestRunCommand_DriverProfilePromptSuffix_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:           1,
			ModelPath:         modelPath,
			PromptBytes:       len(cfg.Prompt),
			PromptSuffixBytes: len(cfg.PromptSuffix),
			MaxTokens:         cfg.MaxTokens,
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	suffix := "Write a short story about a packet of data."

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt", "context", "-prompt-repeat", "2", "-prompt-suffix", suffix, "-max-tokens", "2", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Prompt != "context\n\ncontext\n\n"+suffix {
		t.Fatalf("Prompt = %q, want repeated context with suffix", gotCfg.Prompt)
	}
	if gotCfg.PromptSuffix != suffix {
		t.Fatalf("PromptSuffix = %q, want suffix", gotCfg.PromptSuffix)
	}
	if !core.Contains(stdout.String(), `"prompt_suffix_bytes": 43`) {
		t.Fatalf("stdout = %q, want prompt suffix byte count", stdout.String())
	}
}

func TestRunCommand_DriverProfileSafetyFlags_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			SafetyLimits:  cfg.SafetyLimits,
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"driver-profile",
		"-json",
		"-max-active-memory-bytes", "11",
		"-max-process-virtual-memory-bytes", "22",
		"-max-process-resident-memory-bytes", "33",
		"-repeated-token-loop-limit", "4",
		"-repeated-line-loop-limit", "5",
		"-repeated-sentence-loop-limit", "6",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.SafetyLimits.MaxActiveMemoryBytes != 11 ||
		gotCfg.SafetyLimits.MaxProcessVirtualMemoryBytes != 22 ||
		gotCfg.SafetyLimits.MaxProcessResidentMemoryBytes != 33 ||
		gotCfg.SafetyLimits.RepeatedTokenLoopLimit != 4 ||
		gotCfg.SafetyLimits.RepeatedLineLoopLimit != 5 ||
		gotCfg.SafetyLimits.RepeatedSentenceLoopLimit != 6 {
		t.Fatalf("safety limits = %+v, want CLI overrides", gotCfg.SafetyLimits)
	}
	if !core.Contains(stdout.String(), `"repeated_token_loop_limit": 4`) ||
		!core.Contains(stdout.String(), `"repeated_line_loop_limit": 5`) ||
		!core.Contains(stdout.String(), `"repeated_sentence_loop_limit": 6`) {
		t.Fatalf("stdout = %q, want safety limits in JSON", stdout.String())
	}
}

func TestRunCommand_DriverProfilePanicJSON_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(context.Context, string, []mlx.LoadOption, driverProfileOptions) (*driverProfileReport, error) {
		panic("boom")
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "/models/demo"}, stdout, stderr)

	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stdout.String(), `"error": "driver-profile panic: boom"`) {
		t.Fatalf("stdout = %q, want panic captured in JSON report", stdout.String())
	}
}

func TestRunCommand_ChapterProfilePromptRepeat_Good(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	var gotCfg chapterProfileOptions
	runChapterProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg chapterProfileOptions) (*chapterProfileReport, error) {
		gotCfg = cfg
		return &chapterProfileReport{
			Version:           1,
			ModelPath:         modelPath,
			ContextBytes:      len(cfg.ContextPrompt),
			PremiseBytes:      len(cfg.Premise),
			PromptRepeat:      cfg.PromptRepeat,
			ChaptersRequested: cfg.Chapters,
			ChapterMaxTokens:  cfg.ChapterMaxTokens,
			ChapterMinTokens:  cfg.ChapterMinTokens,
			OutputPath:        cfg.OutputPath,
			Summary: chapterProfileSummary{
				SuccessfulTurns: 2,
				GeneratedTokens: 64,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-json", "-prompt", "seed", "-prompt-repeat", "2", "-premise", "packet story", "-chapters", "2", "-chapter-max-tokens", "32", "-chapter-min-tokens", "16", "-output-file", "book.md", "-enable-thinking", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.ContextPrompt != "seed\n\nseed" {
		t.Fatalf("ContextPrompt = %q, want repeated seed", gotCfg.ContextPrompt)
	}
	if gotCfg.Premise != "packet story" || gotCfg.Chapters != 2 || gotCfg.ChapterMaxTokens != 32 || gotCfg.ChapterMinTokens != 16 {
		t.Fatalf("cfg = %+v, want premise/chapter settings", gotCfg)
	}
	if gotCfg.OutputPath != "book.md" {
		t.Fatalf("OutputPath = %q, want book.md", gotCfg.OutputPath)
	}
	if !gotCfg.EnableThinking || gotCfg.Temperature != 1.0 || gotCfg.TopP != 0.95 || gotCfg.TopK != 64 || gotCfg.RepeatPenalty != 1.0 {
		t.Fatalf("cfg sampling/thinking = %+v, want standard Gemma 4 settings", gotCfg)
	}
	if !core.Contains(stdout.String(), `"chapters_requested": 2`) {
		t.Fatalf("stdout = %q, want chapter count", stdout.String())
	}
	if !core.Contains(stdout.String(), `"output_path": "book.md"`) {
		t.Fatalf("stdout = %q, want output path", stdout.String())
	}
}

func TestRunCommand_ChapterProfileReportFile_Good(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg chapterProfileOptions) (*chapterProfileReport, error) {
		return &chapterProfileReport{
			Version:           1,
			ModelPath:         modelPath,
			ContextBytes:      len(cfg.ContextPrompt),
			PremiseBytes:      len(cfg.Premise),
			ChaptersRequested: cfg.Chapters,
			ChapterMaxTokens:  cfg.ChapterMaxTokens,
			ChapterMinTokens:  cfg.ChapterMinTokens,
			OutputPath:        cfg.OutputPath,
			Summary: chapterProfileSummary{
				SuccessfulTurns: 1,
				VisibleTokens:   768,
			},
		}, nil
	}
	dir := t.TempDir()
	reportPath := core.PathJoin(dir, "reports", "chapter.json")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-report-file", reportPath, "-premise", "packet story", "-chapters", "1", "-chapter-max-tokens", "32", "-chapter-min-tokens", "16", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	read := core.ReadFile(reportPath)
	if !read.OK {
		t.Fatalf("ReadFile(%q): %v", reportPath, read.Value)
	}
	data := string(read.Value.([]byte))
	if !core.Contains(data, `"model_path": "/models/demo"`) || !core.Contains(data, `"successful_turns": 1`) {
		t.Fatalf("report file = %q, want chapter profile JSON", data)
	}
	if core.Contains(stdout.String(), `"model_path"`) {
		t.Fatalf("stdout = %q, should keep JSON in report file unless -json is set", stdout.String())
	}
}

func TestRunCommand_ChapterProfileFastGemma4LaneDefault_Good(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	var gotLoad mlx.LoadConfig
	runChapterProfile = func(_ context.Context, modelPath string, opts []mlx.LoadOption, cfg chapterProfileOptions) (*chapterProfileReport, error) {
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		return &chapterProfileReport{
			Version:           1,
			ModelPath:         modelPath,
			ContextBytes:      len(cfg.ContextPrompt),
			PremiseBytes:      len(cfg.Premise),
			PromptChunkBytes:  cfg.PromptChunkBytes,
			PromptRepeat:      cfg.PromptRepeat,
			ChaptersRequested: cfg.Chapters,
			ChapterMaxTokens:  cfg.ChapterMaxTokens,
			ChapterMinTokens:  cfg.ChapterMinTokens,
			RuntimeGates:      driverProfileRuntimeGates(),
			Summary: chapterProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-json", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotLoad.ContextLength != mlx.ProductionLaneLongContextLength ||
		gotLoad.CacheMode != memory.KVCacheModePaged ||
		gotLoad.PrefillChunkSize != mlx.ProductionLaneLongContextPrefillChunkSize {
		t.Fatalf("load = %+v, want long-form fast lane defaults", gotLoad)
	}
	for _, want := range []string{
		`"chapter_max_tokens": 8192`,
		`"prompt_chunk_bytes": 4096`,
		`"context_length": 32768`,
		`"cache_mode": "paged"`,
		`"prefill_chunk_size": 512`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should not contain default fixed-cache gate %s", stdout.String(), rejected)
		}
	}
	if core.Contains(stdout.String(), `"chapter_min_tokens":`) {
		t.Fatalf("stdout = %q, should not include a default chapter token floor", stdout.String())
	}
}

func TestRunCommand_ChapterProfileSafetyFlags_Good(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	var gotCfg chapterProfileOptions
	runChapterProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg chapterProfileOptions) (*chapterProfileReport, error) {
		gotCfg = cfg
		return &chapterProfileReport{
			Version:           1,
			ModelPath:         modelPath,
			ChaptersRequested: cfg.Chapters,
			ChapterMaxTokens:  cfg.ChapterMaxTokens,
			SafetyLimits:      cfg.SafetyLimits,
			Summary: chapterProfileSummary{
				SuccessfulTurns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"chapter-profile",
		"-json",
		"-max-active-memory-bytes", "11",
		"-max-process-virtual-memory-bytes", "22",
		"-max-process-resident-memory-bytes", "33",
		"-suppressed-token-loop-limit", "4",
		"-repeated-line-loop-limit", "5",
		"-repeated-sentence-loop-limit", "6",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.SafetyLimits.MaxActiveMemoryBytes != 11 ||
		gotCfg.SafetyLimits.MaxProcessVirtualMemoryBytes != 22 ||
		gotCfg.SafetyLimits.MaxProcessResidentMemoryBytes != 33 ||
		gotCfg.SafetyLimits.SuppressedTokenLoopLimit != 4 ||
		gotCfg.SafetyLimits.RepeatedLineLoopLimit != 5 ||
		gotCfg.SafetyLimits.RepeatedSentenceLoopLimit != 6 {
		t.Fatalf("safety limits = %+v, want CLI overrides", gotCfg.SafetyLimits)
	}
	if !core.Contains(stdout.String(), `"max_process_virtual_memory_bytes": 22`) ||
		!core.Contains(stdout.String(), `"repeated_line_loop_limit": 5`) ||
		!core.Contains(stdout.String(), `"repeated_sentence_loop_limit": 6`) {
		t.Fatalf("stdout = %q, want safety limits in JSON", stdout.String())
	}
}

func TestRunCommand_ChapterProfilePanicJSON_Bad(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(context.Context, string, []mlx.LoadOption, chapterProfileOptions) (*chapterProfileReport, error) {
		panic("boom")
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-json", "/models/demo"}, stdout, stderr)

	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stdout.String(), `"error": "chapter-profile panic: boom"`) {
		t.Fatalf("stdout = %q, want panic captured in JSON report", stdout.String())
	}
}

func TestRunCommand_ChapterProfileSuppressedTokenLoopLimit_Bad(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(context.Context, string, []mlx.LoadOption, chapterProfileOptions) (*chapterProfileReport, error) {
		t.Fatal("runChapterProfile called for invalid safety limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-suppressed-token-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "suppressed token loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want safety limit error", stderr.String())
	}
}

func TestRunCommand_ChapterProfileRepeatedLineLoopLimit_Bad(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(context.Context, string, []mlx.LoadOption, chapterProfileOptions) (*chapterProfileReport, error) {
		t.Fatal("runChapterProfile called for invalid repeated-line limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-repeated-line-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeated line loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want repeated-line limit error", stderr.String())
	}
}

func TestRunCommand_ChapterProfileRepeatedSentenceLoopLimit_Bad(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(context.Context, string, []mlx.LoadOption, chapterProfileOptions) (*chapterProfileReport, error) {
		t.Fatal("runChapterProfile called for invalid repeated-sentence limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-repeated-sentence-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeated sentence loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want repeated-sentence limit error", stderr.String())
	}
}

func TestRunCommand_ChapterProfileRepeatPenalty_Bad(t *testing.T) {
	originalRun := runChapterProfile
	t.Cleanup(func() { runChapterProfile = originalRun })
	runChapterProfile = func(context.Context, string, []mlx.LoadOption, chapterProfileOptions) (*chapterProfileReport, error) {
		t.Fatal("runChapterProfile called for invalid repeat penalty")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"chapter-profile", "-repeat-penalty", "-1", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeat penalty must be >= 0") {
		t.Fatalf("stderr = %q, want repeat penalty error", stderr.String())
	}
}

func TestChapterProfileGemma4TemplateThinking_Good(t *testing.T) {
	prompt := chapterProfileInitialPrompt("gemma4", "context", "packet premise", 10, 1024, true)

	if !core.Contains(prompt, "<|turn>system\n<|think|>\ncontext<turn|>\n") {
		t.Fatalf("prompt = %q, want Gemma 4 thinking system turn", prompt)
	}
	if core.Contains(prompt, "<|channel>thought\n<channel|>") {
		t.Fatalf("prompt = %q, should not include disabled-thinking empty thought channel", prompt)
	}
}

func TestChapterProfileGemma4TemplateNoThinking_Good(t *testing.T) {
	prompt := chapterProfileNextPrompt("gemma4", 2, 10, 1024, false)

	if core.HasPrefix(prompt, "<turn|>") {
		t.Fatalf("prompt = %q, should not duplicate previous assistant terminator", prompt)
	}
	if !core.HasPrefix(prompt, "<|turn>user\n") {
		t.Fatalf("prompt = %q, want next Gemma 4 user turn", prompt)
	}
	if !core.Contains(prompt, "<|turn>model\n") {
		t.Fatalf("prompt = %q, want Gemma 4 generation prompt", prompt)
	}
	if !core.Contains(prompt, "<|turn>model\nChapter 2:") {
		t.Fatalf("prompt = %q, want native Gemma 4 generation prompt followed by chapter prefill", prompt)
	}
	if !core.Contains(prompt, "Begin exactly with \"Chapter 2:\"") {
		t.Fatalf("prompt = %q, want direct chapter-start instruction", prompt)
	}
	if core.Contains(prompt, "at least 1024 visible tokens") {
		t.Fatalf("prompt = %q, should not contain debug-floor steering", prompt)
	}
	if !core.Contains(prompt, "write a substantial chapter with concrete scene movement") {
		t.Fatalf("prompt = %q, want natural longform instruction", prompt)
	}
	if !core.Contains(prompt, chapterProfileEndMarker) {
		t.Fatalf("prompt = %q, want chapter end marker instruction", prompt)
	}
	if core.Contains(prompt, "<|channel>thought\n<channel|>") {
		t.Fatalf("prompt = %q, should not inject synthetic empty thought channel", prompt)
	}
	if !core.Contains(prompt, "<|turn>model\nChapter 2:") {
		t.Fatalf("prompt = %q, want chapter heading assistant prefill", prompt)
	}
	if !core.Contains(prompt, "Do not resolve or conclude the story yet") {
		t.Fatalf("prompt = %q, want serial-continuation instruction", prompt)
	}
}

func TestChapterProfileGemma4InitialTemplateNoThinking_Good(t *testing.T) {
	prompt := chapterProfileInitialPrompt("gemma4", "", "packet premise", 10, 1024, false)

	if !core.Contains(prompt, "<|turn>model\nPreamble:\n") {
		t.Fatalf("prompt = %q, want native Gemma 4 generation prompt followed by preamble prefill", prompt)
	}
	if core.Contains(prompt, "<|channel>thought\n<channel|>") {
		t.Fatalf("prompt = %q, should not inject synthetic empty thought channel", prompt)
	}
	if !core.Contains(prompt, chapterProfileEndMarker) {
		t.Fatalf("prompt = %q, want chapter end marker instruction", prompt)
	}
	if core.Contains(prompt, "<|think|>") {
		t.Fatalf("prompt = %q, should not include thinking trigger", prompt)
	}
}

func TestChapterProfileStripEndMarker_Good(t *testing.T) {
	got, ok := chapterProfileStripEndMarker("Chapter 2:\nText.\n[[END_CHAPTER]]\nignored")

	if !ok || got != "Chapter 2:\nText." {
		t.Fatalf("strip = %q ok=%t, want chapter text before marker", got, ok)
	}
}

func TestChapterProfileOutputStream_StripsFragmentedEndMarker_Good(t *testing.T) {
	dst := core.NewBuffer()
	stream := newChapterProfileOutputStream(dst)

	if stream.Write("Chapter text [[END_") {
		t.Fatal("Write() saw a partial end marker")
	}
	if !stream.Write("CHAPTER]] ignored") {
		t.Fatal("Write() did not see fragmented end marker")
	}
	if err := stream.Flush(); err != nil {
		t.Fatalf("Flush() error = %v", err)
	}
	if got := dst.String(); got != "Chapter text " {
		t.Fatalf("streamed text = %q, want marker stripped", got)
	}
}

func TestChapterProfileObserveEndMarker_Fragmented_Good(t *testing.T) {
	window := ""

	if chapterProfileObserveEndMarker(&window, "Chapter text [[END_") {
		t.Fatal("observe saw a partial end marker")
	}
	if !chapterProfileObserveEndMarker(&window, "CHAPTER]]") {
		t.Fatal("observe did not see fragmented end marker")
	}
}

func TestChapterProfileMissingEndMarkerError_AllowsNaturalStopAfterFloor_Good(t *testing.T) {
	if err := chapterProfileMissingEndMarkerError(2, false, 882, 8192); err != "" {
		t.Fatalf("missing marker err = %q, want natural stop accepted below max tokens", err)
	}
}

func TestChapterProfileMissingEndMarkerError_RejectsMaxTokenExhaustion_Bad(t *testing.T) {
	err := chapterProfileMissingEndMarkerError(2, false, 8192, 8192)

	if !core.Contains(err, "reached max tokens 8192 before end marker") {
		t.Fatalf("missing marker err = %q, want max-token exhaustion", err)
	}
}

func TestChapterProfileSafeTextChunks_AvoidsSplittingControlToken_Good(t *testing.T) {
	chunks := []string{}
	for chunk := range chapterProfileSafeTextChunks("aaaa<|turn>bbbb", 7) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) < 2 {
		t.Fatalf("chunks = %#v, want split input", chunks)
	}
	foundControl := false
	for _, chunk := range chunks {
		if chunk == "<|turn>" {
			foundControl = true
			continue
		}
		if core.Contains(chunk, "<|tu") || core.Contains(chunk, "rn>") {
			t.Fatalf("chunk = %q split control token", chunk)
		}
	}
	if !foundControl {
		t.Fatalf("chunks = %#v, want intact control token chunk", chunks)
	}
}

func TestChapterProfileGemma4VisibleText_HidesThinkingChannel_Good(t *testing.T) {
	got := chapterProfileVisibleText("gemma4", "<|channel>thought\nprivate plan<channel|>Chapter 2\n")

	if got != "Chapter 2" {
		t.Fatalf("visible text = %q, want Chapter 2", got)
	}
}

func TestChapterProfileGemma4VisibleTextForChapter_HidesPlainThinking_Good(t *testing.T) {
	got := chapterProfileVisibleTextForChapter("gemma4", "thought\nprivate plan\n**Chapter 2: The Rewrite**\nFinal text.", 2)

	if got != "**Chapter 2: The Rewrite**\nFinal text." {
		t.Fatalf("visible text = %q, want Chapter 2 only", got)
	}
}

func TestChapterProfileGemma4VisibleTextForChapter_HidesPreambleThinking_Good(t *testing.T) {
	got := chapterProfileVisibleTextForChapter("gemma4", "thought\nprivate plan\n**Preamble**\nFinal text.", 1)

	if got != "**Preamble**\nFinal text." {
		t.Fatalf("visible text = %q, want preamble only", got)
	}
}

func TestChapterProfileAssistantHistorySuffix_Gemma4_Good(t *testing.T) {
	got := chapterProfileAssistantHistorySuffix("gemma4", "Chapter 2")

	if got != "Chapter 2<turn|>\n" {
		t.Fatalf("history suffix = %q, want final-only Gemma 4 assistant turn", got)
	}
}

func TestChapterProfileSafetyLimits_DerivesFromResolvedMemory_Good(t *testing.T) {
	limits := resolveChapterProfileSafetyLimits(chapterProfileSafetyLimits{}, &tuneProfileLoadSettings{
		MemoryLimitBytes: 64 * memory.GiB,
	})

	if limits.MaxActiveMemoryBytes != profileDefaultActiveMemoryLimit(64*memory.GiB) {
		t.Fatalf("active limit = %d, want resolved memory limit plus headroom", limits.MaxActiveMemoryBytes)
	}
	if limits.MaxProcessResidentMemoryBytes != 64*memory.GiB {
		t.Fatalf("resident limit = %d, want resolved memory limit", limits.MaxProcessResidentMemoryBytes)
	}
	if limits.MaxProcessVirtualMemoryBytes != 0 {
		t.Fatalf("virtual limit = %d, want explicit-only virtual cap", limits.MaxProcessVirtualMemoryBytes)
	}
	if limits.SuppressedTokenLoopLimit != chapterProfileDefaultSuppressedTokenLoopLimit {
		t.Fatalf("loop limit = %d, want default", limits.SuppressedTokenLoopLimit)
	}
	if limits.RepeatedLineLoopLimit != profileDefaultRepeatedLineLoopLimit {
		t.Fatalf("line loop limit = %d, want default", limits.RepeatedLineLoopLimit)
	}
	if limits.RepeatedSentenceLoopLimit != profileDefaultRepeatedSentenceLoopLimit {
		t.Fatalf("sentence loop limit = %d, want default", limits.RepeatedSentenceLoopLimit)
	}
}

func TestChapterProfileSuppressedTokenLoop_Bad(t *testing.T) {
	id, count, ok := chapterProfileSuppressedTokenLoop(
		[]int32{9, 0, 0, 0, 0, 4},
		[]int32{0},
		4,
	)

	if !ok || id != 0 || count != 4 {
		t.Fatalf("loop = id %d count %d ok %t, want token 0 repeated four times", id, count, ok)
	}
}

func TestProfileRepeatedLineLoop_Bad(t *testing.T) {
	line, count, ok := profileRepeatedLineLoop("The sensor.\n\nThe sensor.\nThe sensor.", 3)

	if !ok || line != "The sensor." || count != 3 {
		t.Fatalf("loop = line %q count %d ok %t, want final repeated line detected", line, count, ok)
	}
}

func TestProfileRepeatedSentenceLoop_Bad(t *testing.T) {
	sentence, count, ok := profileRepeatedSentenceLoop("It was a packet of data. It changed shape. It was a packet of data! It moved. It was a packet of data? It hid. It was a packet of data.", 4)

	if !ok || sentence != "it was a packet of data" || count != 4 {
		t.Fatalf("loop = sentence %q count %d ok %t, want repeated sentence detected", sentence, count, ok)
	}
}

func TestProfileFragmentedSentenceOutput_Bad(t *testing.T) {
	fragments, total, ok := profileFragmentedSentenceOutput("A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T.")

	if !ok || fragments != 20 || total != 20 {
		t.Fatalf("fragments = %d total = %d ok = %t, want fragmented output detected", fragments, total, ok)
	}
}

func TestChapterProfileTurnSafety_StopsSuppressedTokenLoop_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		SuppressTokenIDs: []int32{0},
		SampledTokenIDs:  []int32{0, 0, 0, 0, 0, 0, 0, 0},
		Metrics: mlx.Metrics{
			GeneratedTokens: 8,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 3, "", turn, chapterProfileSafetyLimits{
		SuppressedTokenLoopLimit: 8,
	})

	if err == nil || !core.Contains(err.Error(), "sampled suppressed token 0") {
		t.Fatalf("err = %v, want suppressed-token loop failure", err)
	}
}

func TestChapterProfileTurnSafety_StopsRepeatedLineLoop_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		Metrics: mlx.Metrics{
			GeneratedTokens: 3,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 2, "The sensor.\nThe sensor.\nThe sensor.", turn, chapterProfileSafetyLimits{
		RepeatedLineLoopLimit: 3,
	})

	if err == nil || !core.Contains(err.Error(), "repeated visible line") {
		t.Fatalf("err = %v, want repeated-line loop failure", err)
	}
}

func TestChapterProfileTurnSafety_StopsRepeatedSentenceLoop_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		Metrics: mlx.Metrics{
			GeneratedTokens: 16,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 5, "It was a packet of data. It changed shape. It was a packet of data. It moved. It was a packet of data. It hid. It was a packet of data.", turn, chapterProfileSafetyLimits{
		RepeatedSentenceLoopLimit: 4,
	})

	if err == nil || !core.Contains(err.Error(), "repeated visible sentence") {
		t.Fatalf("err = %v, want repeated-sentence loop failure", err)
	}
}

func TestChapterProfileTurnSafety_StopsFragmentedOutput_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		Metrics: mlx.Metrics{
			GeneratedTokens: 32,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 7, "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T.", turn, chapterProfileSafetyLimits{})

	if err == nil || !core.Contains(err.Error(), "fragmented visible output") {
		t.Fatalf("err = %v, want fragmented output failure", err)
	}
}

func TestChapterProfileTurnSafety_StopsMetaPlanningOutput_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		Metrics: mlx.Metrics{
			GeneratedTokens: 16,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 2, "Chapter 2 needs to focus on the packet leaving the buffer.", turn, chapterProfileSafetyLimits{})

	if err == nil || !core.Contains(err.Error(), "meta-planning output") {
		t.Fatalf("err = %v, want meta-planning output failure", err)
	}
}

func TestChapterProfileTurnSafety_StopsOutlineOutput_Bad(t *testing.T) {
	turn := chapterProfileTurn{
		Metrics: mlx.Metrics{
			GeneratedTokens: 16,
		},
	}

	err := chapterProfileTurnSafetyError("gemma4", 3, "Chapter 3: Focus on the rewrite before release.", turn, chapterProfileSafetyLimits{})

	if err == nil || !core.Contains(err.Error(), "meta-planning output") {
		t.Fatalf("err = %v, want outline output failure", err)
	}
}

func TestChapterProfileMetricsSafety_StopsVirtualMemoryOvershoot_Bad(t *testing.T) {
	err := chapterProfileMetricsSafetyError("chapter 2", mlx.Metrics{
		ProcessVirtualMemoryBytes: 123,
	}, chapterProfileSafetyLimits{
		MaxProcessVirtualMemoryBytes: 122,
	})

	if err == nil || !core.Contains(err.Error(), "process virtual memory safety limit") {
		t.Fatalf("err = %v, want process virtual safety failure", err)
	}
}

func TestRunCommand_DriverProfilePromptRepeat_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid prompt repeat")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt-repeat", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "prompt repeat must be >= 1") {
		t.Fatalf("stderr = %q, want prompt repeat error", stderr.String())
	}
}

func TestRunCommand_DriverProfileRepeatedTokenLoopLimit_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid repeated-token limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-repeated-token-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeated token loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want repeated-token limit error", stderr.String())
	}
}

func TestRunCommand_DriverProfileRepeatedLineLoopLimit_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid repeated-line limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-repeated-line-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeated line loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want repeated-line limit error", stderr.String())
	}
}

func TestRunCommand_DriverProfileRepeatedSentenceLoopLimit_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid repeated-sentence limit")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-repeated-sentence-loop-limit", "0", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "repeated sentence loop limit must be >= 1") {
		t.Fatalf("stderr = %q, want repeated-sentence limit error", stderr.String())
	}
}

func TestDriverProfileRuntimeGates_RecordsEnabledNativeGate_Good(t *testing.T) {
	t.Setenv("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_MLP_GELU", "0")

	gates := driverProfileRuntimeGates()
	if gates["GO_MLX_ENABLE_EXPERT_ID_MATVEC"] != "1" {
		t.Fatalf("runtime gates = %+v, want expert-id gate", gates)
	}
	for _, rejected := range []string{
		"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION",
		"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION",
		"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE",
	} {
		if _, ok := gates[rejected]; ok {
			t.Fatalf("runtime gates = %+v, should ignore ambient fixed diagnostic gate %s", gates, rejected)
		}
	}
	if _, ok := gates["GO_MLX_ENABLE_NATIVE_MLP_GELU"]; ok {
		t.Fatalf("runtime gates = %+v, disabled gate should be omitted", gates)
	}
}

func TestDriverProfileRuntimeGates_RecordsCLIOverride_Good(t *testing.T) {
	restore := setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	t.Cleanup(restore)

	gates := driverProfileRuntimeGates()
	if gates["GO_MLX_ENABLE_EXPERT_ID_MATVEC"] != "1" {
		t.Fatalf("runtime gates = %+v, want expert-id CLI override", gates)
	}
}

func TestRunCommand_DriverProfileExpertIDMatVecFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-expert-id-matvec", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_EXPERT_ID_MATVEC": "1"`) {
		t.Fatalf("stdout = %q, want expert-id runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileExpertIDFusedActivationFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-expert-id-fused-activation", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"GO_MLX_ENABLE_EXPERT_ID_MATVEC": "1"`,
		`"GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION": "1"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileSortedExpertPrefillFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-sorted-expert-prefill", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_SORTED_EXPERT_PREFILL": "1"`) {
		t.Fatalf("stdout = %q, want sorted expert prefill runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfilePagedDecodeFastConcatFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-paged-decode-fast-concat", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT": "1"`) {
		t.Fatalf("stdout = %q, want paged decode fast concat runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileNativePagedAttentionFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-paged-attention", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION": "1"`) {
		t.Fatalf("stdout = %q, want native paged attention runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileGenerationClearCacheFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-generation-clear-cache", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_GENERATION_CLEAR_CACHE": "1"`) {
		t.Fatalf("stdout = %q, want generation clear-cache runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileNativeGemma4RouterMatVecFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-gemma4-router-matvec", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC": "1"`) {
		t.Fatalf("stdout = %q, want native router matvec runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileNativeMLPMatVecFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-mlp-matvec", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_MLP_MATVEC": "1"`) {
		t.Fatalf("stdout = %q, want native MLP matvec runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"GO_MLX_ENABLE_EXPERT_ID_MATVEC": "1"`,
		`"GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION": "1"`,
		`"GO_MLX_ENABLE_SORTED_EXPERT_PREFILL": "1"`,
		`"GO_MLX_ENABLE_NATIVE_MLP_MATVEC": "1"`,
		`"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK": "1"`,
		`"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN": "1"`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
		`"context_length": 4096`,
		`"cache_mode": "paged"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION": "1"`,
		`"GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC": "1"`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE": "1"`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK": "1"`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND": "1"`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should exclude rejected gate %s", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneDefault_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotCfg driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotCfg = cfg
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.Prompt != mlx.DefaultNewSessionText {
		t.Fatalf("driver profile default prompt = %q, want Lemma new-session default", gotCfg.Prompt)
	}
	if gotCfg.MaxTokens != mlx.ProductionLaneMaxTokens || gotCfg.Runs != mlx.ProductionLaneRuns {
		t.Fatalf("driver profile default shape = max:%d runs:%d, want production lane max:%d runs:%d", gotCfg.MaxTokens, gotCfg.Runs, mlx.ProductionLaneMaxTokens, mlx.ProductionLaneRuns)
	}
	if gotCfg.IncludeOutput || !gotCfg.TraceTokenPhases {
		t.Fatalf("driver profile default reporting = include_output:%v trace:%v, want hidden output plus token phase trace", gotCfg.IncludeOutput, gotCfg.TraceTokenPhases)
	}
	for _, want := range []string{
		`"GO_MLX_ENABLE_EXPERT_ID_MATVEC": "1"`,
		`"GO_MLX_ENABLE_NATIVE_MLP_MATVEC": "1"`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
		`"context_length": 4096`,
		`"cache_mode": "paged"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneCanDisable_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", "1")
	t.Setenv("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE", "1")
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", core.Sprintf("%d", mlx.ProductionLaneHyperLongContextLength))
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane=false", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_EXPERT_ID_MATVEC": "1"`,
		`"GO_MLX_ENABLE_NATIVE_MLP_MATVEC": "1"`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
		`"context_length": 4096`,
		`"cache_mode": "paged"`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
		`"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL":`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY":`,
		`"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION":`,
		`"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION":`,
		`"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE":`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should exclude default fast-lane value %s", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneLongContextDefaults_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RuntimeGates:     driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane", "-context", "32768", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"context_length": 32768`,
		`"cache_mode": "paged"`,
		`"prefill_chunk_size": 512`,
		`"prompt_chunk_bytes": 4096`,
		`"GO_MLX_KV_CACHE_DTYPE": "fp16"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), `"GO_MLX_ENABLE_FIXED_GEMMA4`) {
		t.Fatalf("stdout = %q, should not enable fixed Gemma4 cache for long context", stdout.String())
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneHyperLongContextStaysPaged_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RuntimeGates:     driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane", "-context", "131072", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"context_length": 131072`,
		`"cache_mode": "paged"`,
		`"prefill_chunk_size": 512`,
		`"prompt_chunk_bytes": 4096`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
		`"GO_MLX_KV_CACHE_DTYPE": "fp16"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), `"GO_MLX_ENABLE_FIXED_GEMMA4`) {
		t.Fatalf("stdout = %q, should not enable fixed Gemma4 cache for hyper-long context", stdout.String())
	}
	if core.Contains(stdout.String(), `"GO_MLX_PAGED_KV_PAGE_SIZE":`) {
		t.Fatalf("stdout = %q, should use code default page size without context-cutoff env", stdout.String())
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneIgnoresFixedCacheEnv_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK", "1")
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", core.Sprintf("%d", mlx.ProductionLaneHyperLongContextLength))
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RuntimeGates:     driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane", "-context", "131072", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, rejected := range []string{
		`"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND":`,
		`"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK":`,
		`"GO_MLX_FIXED_GEMMA4_CACHE_SIZE":`,
	} {
		if core.Contains(stdout.String(), rejected) {
			t.Fatalf("stdout = %q, should ignore ambient fixed-cache env %s in the fast lane", stdout.String(), rejected)
		}
	}
}

func TestRunCommand_DriverProfileFastGemma4LaneLongContextOverride_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RuntimeGates:     driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-fast-gemma4-lane", "-context", "32768", "-prefill-chunk-size", "2048", "-prompt-chunk-bytes", "8192", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"prefill_chunk_size": 2048`,
		`"prompt_chunk_bytes": 8192`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileNativeLinearMatVecFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-linear-matvec", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC": "1"`) {
		t.Fatalf("stdout = %q, want native linear matvec runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileNativeGemma4FFNResidualFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-gemma4-ffn-residual", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL": "1"`) {
		t.Fatalf("stdout = %q, want native Gemma 4 FFN residual runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileNativeGemma4AttentionOMatVecFlag_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-native-gemma4-attention-o-matvec", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC": "1"`) {
		t.Fatalf("stdout = %q, want native Gemma 4 attention output matvec runtime gate", stdout.String())
	}
}

func TestRunCommand_DriverProfileGemma4DecodeGateFlags_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:      1,
			ModelPath:    modelPath,
			PromptBytes:  len(cfg.Prompt),
			MaxTokens:    cfg.MaxTokens,
			RuntimeGates: driverProfileRuntimeGates(),
			Summary: driverProfileSummary{
				SuccessfulRuns: 1,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"driver-profile",
		"-json",
		"-fast-gemma4-lane=false",
		"-native-gemma4-layer",
		"-native-gemma4-moe-layer",
		"-compiled-gemma4-layer",
		"-direct-greedy-token",
		"-generation-stream",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER": "1"`,
		`"GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER": "1"`,
		`"GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER": "1"`,
		`"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN": "1"`,
		`"GO_MLX_ENABLE_GENERATION_STREAM": "1"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileRejectsFixedCacheFlags_Good(t *testing.T) {
	for _, flagName := range []string{
		"fixed-gemma4-cache",
		"fixed-gemma4-sliding-cache-bound",
		"fixed-gemma4-shared-mask",
		"native-fixed-sliding-attention",
		"native-gemma4-fixed-owner-attention",
		"native-gemma4-fixed-owner-attention-residual",
		"native-gemma4-model-greedy",
	} {
		stdout, stderr := core.NewBuffer(), core.NewBuffer()

		code := runCommand(context.Background(), []string{
			"driver-profile",
			"-json",
			"-" + flagName,
			"/models/demo",
		}, stdout, stderr)

		if code != 2 {
			t.Fatalf("%s exit code = %d, want 2; stderr=%q stdout=%q", flagName, code, stderr.String(), stdout.String())
		}
		if !core.Contains(stderr.String(), "flag provided but not defined: -"+flagName) {
			t.Fatalf("%s stderr = %q, want undefined-flag error", flagName, stderr.String())
		}
	}
}

func TestRunCommand_DriverProfileCacheMode_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotLoad mlx.LoadConfig
	runDriverProfile = func(_ context.Context, modelPath string, opts []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Summary:       driverProfileSummary{SuccessfulRuns: 1},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-context", "4096", "-cache-mode", "paged", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotLoad.ContextLength != 4096 || gotLoad.CacheMode != memory.KVCacheModePaged {
		t.Fatalf("load = %+v, want context 4096 and paged cache", gotLoad)
	}
	for _, want := range []string{`"context_length": 4096`, `"cache_mode": "paged"`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfilePrefillChunkSize_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var gotLoad mlx.LoadConfig
	runDriverProfile = func(_ context.Context, modelPath string, opts []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		gotLoad = mlx.DefaultLoadConfig()
		for _, opt := range opts {
			opt(&gotLoad)
		}
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Summary:       driverProfileSummary{SuccessfulRuns: 1},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prefill-chunk-size", "1024", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotLoad.PrefillChunkSize != 1024 {
		t.Fatalf("PrefillChunkSize = %d, want 1024", gotLoad.PrefillChunkSize)
	}
	if !core.Contains(stdout.String(), `"prefill_chunk_size": 1024`) {
		t.Fatalf("stdout = %q, want prefill chunk size", stdout.String())
	}
}

func TestRunCommand_DriverProfilePrefillChunkSize_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid prefill chunk size")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prefill-chunk-size", "-1", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "prefill chunk size must be >= 0") {
		t.Fatalf("stderr = %q, want prefill chunk size error", stderr.String())
	}
	if stdout.String() != "" {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
}

func TestRunCommand_DriverProfileCacheMode_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid cache mode")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-cache-mode", "banana", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), `unsupported cache mode "banana"`) {
		t.Fatalf("stderr = %q, want unsupported cache mode", stderr.String())
	}
	if stdout.String() != "" {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
}

func TestRunCommand_DriverProfileResolvedLoadSettings_Good(t *testing.T) {
	primary := &tuneProfileLoadSettings{ContextLength: 4096}
	resolved := loadSettingsFromModelInfo(mlx.ModelInfo{
		ContextLength:        131072,
		ParallelSlots:        2,
		PromptCache:          true,
		PromptCacheMinTokens: 2048,
		CachePolicy:          memory.KVCacheRotating,
		CacheMode:            memory.KVCacheModePaged,
		BatchSize:            4,
		PrefillChunkSize:     4096,
		ExpectedQuantization: 8,
		MemoryLimitBytes:     1024,
		CacheLimitBytes:      512,
		WiredLimitBytes:      768,
	})

	merged := mergeDriverProfileLoadSettings(primary, resolved)

	if merged.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want explicit primary value", merged.ContextLength)
	}
	if merged.CachePolicy != string(memory.KVCacheRotating) || merged.CacheMode != string(memory.KVCacheModePaged) {
		t.Fatalf("cache = %q/%q, want resolved planner cache", merged.CachePolicy, merged.CacheMode)
	}
	if !merged.PromptCache || merged.PromptCacheMinTokens != 2048 || merged.BatchSize != 4 || merged.PrefillChunkSize != 4096 {
		t.Fatalf("resolved load settings = %+v, want prompt/batch/prefill fields", merged)
	}
}

func TestRunCommand_DriverProfileResolvedLoadSettingsFromRunner_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		return &driverProfileReport{
			Version:       1,
			ModelPath:     modelPath,
			PromptBytes:   len(cfg.Prompt),
			MaxTokens:     cfg.MaxTokens,
			RequestedRuns: cfg.Runs,
			Load: &tuneProfileLoadSettings{
				ContextLength:        131072,
				PromptCache:          true,
				PromptCacheMinTokens: 2048,
				CachePolicy:          string(memory.KVCacheRotating),
				CacheMode:            string(memory.KVCacheModePaged),
				BatchSize:            4,
				PrefillChunkSize:     4096,
			},
			Summary: driverProfileSummary{SuccessfulRuns: 1},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-context", "4096", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"context_length": 4096`,
		`"cache_policy": "rotating"`,
		`"cache_mode": "paged"`,
		`"batch_size": 4`,
		`"prefill_chunk_size": 4096`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_DriverProfileGemmaQwenMatrix_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })

	for _, tc := range []struct {
		name string
		path string
	}{
		{name: "gemma4", path: "/models/gemma4"},
		{name: "qwen2", path: "/models/qwen2"},
		{name: "qwen3", path: "/models/qwen3"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			var gotCfg driverProfileOptions
			runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
				gotPath = modelPath
				gotCfg = cfg
				return &driverProfileReport{
					Version:       1,
					ModelPath:     modelPath,
					PromptBytes:   len(cfg.Prompt),
					MaxTokens:     cfg.MaxTokens,
					RequestedRuns: cfg.Runs,
					Summary:       driverProfileSummary{SuccessfulRuns: 1},
				}, nil
			}
			stdout, stderr := core.NewBuffer(), core.NewBuffer()

			code := runCommand(context.Background(), []string{"driver-profile", "-json", "-include-output=false", "-prompt", "state smoke", "-max-tokens", "4", "-runs", "1", tc.path}, stdout, stderr)

			if code != 0 {
				t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
			}
			if gotPath != tc.path || gotCfg.Prompt != "state smoke" || gotCfg.MaxTokens != 4 || gotCfg.Runs != 1 || gotCfg.IncludeOutput {
				t.Fatalf("driver-profile path=%q cfg=%+v, want shared profile command shape", gotPath, gotCfg)
			}
			if !core.Contains(stdout.String(), `"model_path": "`+tc.path+`"`) || !core.Contains(stdout.String(), `"successful_runs": 1`) {
				t.Fatalf("stdout = %q, want model path and successful run", stdout.String())
			}
		})
	}
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
	lastConfig        mlx.GenerateConfig
}

func (m *fakeDriverProfileModel) GenerateStream(ctx context.Context, _ string, opts ...mlx.GenerateOption) <-chan mlx.Token {
	m.generateCalls++
	m.lastConfig = mlx.DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&m.lastConfig)
	}
	ch := make(chan mlx.Token)
	if len(m.streamTokens) == 0 {
		close(ch)
		return ch
	}
	go func() {
		defer close(ch)
		closeMetrics := func(delay bool) {
			if m.metricsReady == nil {
				return
			}
			if delay {
				time.Sleep(20 * time.Millisecond)
			}
			close(m.metricsReady)
		}
		for _, token := range m.streamTokens {
			select {
			case <-ctx.Done():
				closeMetrics(true)
				return
			case ch <- token:
			}
		}
		closeMetrics(false)
	}()
	return ch
}

func (m *fakeDriverProfileModel) GenerateChunksStream(_ context.Context, chunks iter.Seq[string], opts ...mlx.GenerateOption) <-chan mlx.Token {
	m.chunkCalls++
	m.chunks = nil
	for chunk := range chunks {
		m.chunks = append(m.chunks, chunk)
	}
	m.lastConfig = mlx.DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&m.lastConfig)
	}
	ch := make(chan mlx.Token, 1)
	ch <- mlx.Token{Text: "chunked"}
	close(ch)
	return ch
}

func (m *fakeDriverProfileModel) ChatChunksStream(_ context.Context, messages []inference.Message, chunkBytes int, opts ...mlx.GenerateOption) <-chan mlx.Token {
	m.chatChunkCalls++
	m.chatChunkMessages = append([]inference.Message(nil), messages...)
	m.chatChunkBytes = chunkBytes
	m.lastConfig = mlx.DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&m.lastConfig)
	}
	ch := make(chan mlx.Token, 1)
	ch <- mlx.Token{Text: "chat chunked"}
	close(ch)
	return ch
}

func (m *fakeDriverProfileModel) ChatStream(_ context.Context, _ []inference.Message, opts ...mlx.GenerateOption) <-chan mlx.Token {
	m.chatCalls++
	m.lastConfig = mlx.DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&m.lastConfig)
	}
	ch := make(chan mlx.Token, 2)
	ch <- mlx.Token{Text: "chat "}
	ch <- mlx.Token{Text: "ok"}
	close(ch)
	return ch
}

func (m *fakeDriverProfileModel) Metrics() mlx.Metrics {
	if m.metricsReady != nil {
		select {
		case <-m.metricsReady:
			return m.delayedMetrics
		default:
		}
	}
	return m.metrics
}

func (m *fakeDriverProfileModel) Err() error { return nil }

func TestDriverProfileGeneration_ChatModeDoesNotStartRawStream_Good(t *testing.T) {
	model := &fakeDriverProfileModel{metrics: mlx.Metrics{GeneratedTokens: 2, DecodeTokensPerSec: 50, PromptCacheRestoreDuration: 5 * time.Millisecond}}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:        "hello",
		MaxTokens:     2,
		Runs:          1,
		IncludeOutput: true,
		Chat:          true,
	})

	if model.generateCalls != 0 {
		t.Fatalf("GenerateStream calls = %d, want 0 in chat mode", model.generateCalls)
	}
	if model.chatCalls != 1 {
		t.Fatalf("ChatStream calls = %d, want 1", model.chatCalls)
	}
	if run.Output != "chat ok" || run.VisibleTokens != 2 || run.Metrics.DecodeTokensPerSec != 50 || run.RestoreDuration != 5*time.Millisecond {
		t.Fatalf("run = %+v, want chat output and metrics", run)
	}
	summary := summariseDriverProfileRuns([]driverProfileRun{run})
	if summary.RestoreAvgDuration != 5*time.Millisecond || summary.RestoreMinDuration != 5*time.Millisecond || summary.RestoreMaxDuration != 5*time.Millisecond {
		t.Fatalf("summary restore timings = %+v, want 5ms restore", summary)
	}
}

func TestDriverProfileGeneration_DrainsCancelledStreamBeforeMetrics_Good(t *testing.T) {
	ready := make(chan struct{})
	model := &fakeDriverProfileModel{
		metrics:        mlx.Metrics{GeneratedTokens: 1, DecodeTokensPerSec: 10},
		delayedMetrics: mlx.Metrics{GeneratedTokens: 2, DecodeTokensPerSec: 42},
		metricsReady:   ready,
		streamTokens: []mlx.Token{
			{ID: 7, Text: "a"},
			{ID: 7, Text: "b"},
			{ID: 8, Text: "ignored"},
		},
	}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:        "hello",
		MaxTokens:     3,
		IncludeOutput: true,
		SafetyLimits: driverProfileSafetyLimits{
			RepeatedTokenLoopLimit: 2,
		},
	})

	if run.Metrics.GeneratedTokens != 2 || run.Metrics.DecodeTokensPerSec != 42 {
		t.Fatalf("metrics = %+v, want finalized delayed metrics after stream drain", run.Metrics)
	}
	if run.VisibleTokens != 2 || run.Output != "a" {
		t.Fatalf("run output = tokens:%d text:%q, want cancellation token counted and drained tail ignored", run.VisibleTokens, run.Output)
	}
	if !core.Contains(run.Error, "sampled token 7 for 2 consecutive tokens") {
		t.Fatalf("run error = %q, want repeated-token cancellation", run.Error)
	}
}

func TestDriverProfileGeneration_ChunkedPromptUsesChunkStream_Good(t *testing.T) {
	model := &fakeDriverProfileModel{metrics: mlx.Metrics{GeneratedTokens: 1, DecodeTokensPerSec: 10}}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:           "abcdef",
		PromptChunkBytes: 2,
		MaxTokens:        1,
		IncludeOutput:    true,
	})

	if model.chunkCalls != 1 || model.generateCalls != 0 || model.chatCalls != 0 {
		t.Fatalf("calls = chunk:%d generate:%d chat:%d, want chunk only", model.chunkCalls, model.generateCalls, model.chatCalls)
	}
	if got, want := core.Join(",", model.chunks...), "ab,cd,ef"; got != want {
		t.Fatalf("chunks = %q, want %q", got, want)
	}
	if run.Output != "chunked" || run.VisibleTokens != 1 {
		t.Fatalf("run = %+v, want chunked output", run)
	}
}

func TestDriverProfileGeneration_ChunkedChatUsesChatChunkStream_Good(t *testing.T) {
	model := &fakeDriverProfileModel{metrics: mlx.Metrics{GeneratedTokens: 1, DecodeTokensPerSec: 10}}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:           "abcdef",
		PromptChunkBytes: 2,
		MaxTokens:        1,
		IncludeOutput:    true,
		Chat:             true,
	})

	if model.chatChunkCalls != 1 || model.chunkCalls != 0 || model.generateCalls != 0 || model.chatCalls != 0 {
		t.Fatalf("calls = chatChunk:%d chunk:%d generate:%d chat:%d, want chat chunk only", model.chatChunkCalls, model.chunkCalls, model.generateCalls, model.chatCalls)
	}
	if model.chatChunkBytes != 2 || len(model.chatChunkMessages) != 1 || model.chatChunkMessages[0].Content != "abcdef" {
		t.Fatalf("chat chunk args = bytes:%d messages:%+v, want prompt message", model.chatChunkBytes, model.chatChunkMessages)
	}
	if run.Output != "chat chunked" || run.VisibleTokens != 1 {
		t.Fatalf("run = %+v, want chat chunked output", run)
	}
}

func TestDriverProfileGeneration_TraceTokenPhasesOption_Good(t *testing.T) {
	model := &fakeDriverProfileModel{}

	_ = profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:           "hello",
		MaxTokens:        2,
		Runs:             1,
		TraceTokenPhases: true,
		Chat:             true,
	})

	if !model.lastConfig.TraceTokenPhases {
		t.Fatalf("TraceTokenPhases = false, want true; cfg=%+v", model.lastConfig)
	}
	if model.lastConfig.TraceTokenText {
		t.Fatalf("TraceTokenText = true, want hidden-output profiles to keep phase traces timing-only; cfg=%+v", model.lastConfig)
	}
	if model.lastConfig.ProbeSink != nil {
		t.Fatalf("ProbeSink = %T, want nil so driver-profile keeps the direct greedy path", model.lastConfig.ProbeSink)
	}
}

func TestDriverProfileGeneration_TraceTextFollowsOutput_Good(t *testing.T) {
	model := &fakeDriverProfileModel{}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:           "hello",
		MaxTokens:        2,
		Runs:             1,
		IncludeOutput:    true,
		TraceTokenPhases: true,
		Chat:             true,
	})

	if !model.lastConfig.TraceTokenText {
		t.Fatalf("TraceTokenText = false, want token text only when output is already included; cfg=%+v", model.lastConfig)
	}
	if got := core.Join("", run.SampledTokenTexts...); got != "chat ok" {
		t.Fatalf("sampled token text = %q, want text retained with include-output", got)
	}
}

func TestDriverProfileGeneration_HiddenOutputOmitsSampledText_Good(t *testing.T) {
	model := &fakeDriverProfileModel{}

	run := profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:    "hello",
		MaxTokens: 2,
		Runs:      1,
		Chat:      true,
	})

	if run.Output != "" {
		t.Fatalf("output = %q, want hidden output", run.Output)
	}
	if len(run.SampledTokenTexts) != 0 {
		t.Fatalf("sampled token text = %+v, want hidden-output profile to carry IDs only", run.SampledTokenTexts)
	}
	if len(run.SampledTokenIDs) != 2 {
		t.Fatalf("sampled token ids = %+v, want IDs kept for loop diagnostics", run.SampledTokenIDs)
	}
}

func TestDriverProfileGeneration_StopAndSuppressTokens_Good(t *testing.T) {
	model := &fakeDriverProfileModel{}

	_ = profileLoadedModelGeneration(context.Background(), model, 1, driverProfileOptions{
		Prompt:           "hello",
		MaxTokens:        2,
		Chat:             true,
		StopTokenIDs:     []int32{1, 106},
		SuppressTokenIDs: []int32{0, 2, 105},
	})

	if got := model.lastConfig.StopTokens; len(got) != 2 || got[0] != 1 || got[1] != 106 {
		t.Fatalf("StopTokens = %v, want [1 106]", got)
	}
	if got := model.lastConfig.SuppressTokens; len(got) != 3 || got[0] != 0 || got[1] != 2 || got[2] != 105 {
		t.Fatalf("SuppressTokens = %v, want [0 2 105]", got)
	}
}

func TestDriverProfileSafetyLimits_DerivesFromResolvedMemory_Good(t *testing.T) {
	limits := resolveDriverProfileSafetyLimits(driverProfileSafetyLimits{}, &tuneProfileLoadSettings{
		MemoryLimitBytes: 64 * memory.GiB,
	})

	if limits.MaxActiveMemoryBytes != profileDefaultActiveMemoryLimit(64*memory.GiB) {
		t.Fatalf("active limit = %d, want resolved memory limit plus headroom", limits.MaxActiveMemoryBytes)
	}
	if limits.MaxProcessResidentMemoryBytes != 64*memory.GiB {
		t.Fatalf("resident limit = %d, want resolved memory limit", limits.MaxProcessResidentMemoryBytes)
	}
	if limits.MaxProcessVirtualMemoryBytes != 0 {
		t.Fatalf("virtual limit = %d, want explicit-only virtual cap", limits.MaxProcessVirtualMemoryBytes)
	}
	if limits.RepeatedTokenLoopLimit != driverProfileDefaultRepeatedTokenLoopLimit {
		t.Fatalf("loop limit = %d, want default", limits.RepeatedTokenLoopLimit)
	}
	if limits.RepeatedLineLoopLimit != profileDefaultRepeatedLineLoopLimit {
		t.Fatalf("line loop limit = %d, want default", limits.RepeatedLineLoopLimit)
	}
	if limits.RepeatedSentenceLoopLimit != profileDefaultRepeatedSentenceLoopLimit {
		t.Fatalf("sentence loop limit = %d, want default", limits.RepeatedSentenceLoopLimit)
	}
}

func TestDriverProfileRepeatedTokenLoop_Bad(t *testing.T) {
	id, count, ok := driverProfileRepeatedTokenLoop([]int32{1, 2, 2, 2, 2, 3}, 4)

	if !ok || id != 2 || count != 4 {
		t.Fatalf("loop = id %d count %d ok %t, want token 2 repeated four times", id, count, ok)
	}
}

func TestDriverProfileRunSafety_StopsRepeatedTokenLoop_Bad(t *testing.T) {
	run := driverProfileRun{
		SampledTokenIDs: []int32{9, 9, 9, 9},
		Metrics: mlx.Metrics{
			GeneratedTokens: 4,
		},
	}

	err := driverProfileRunSafetyError(1, run, driverProfileSafetyLimits{RepeatedTokenLoopLimit: 4})

	if err == nil || !core.Contains(err.Error(), "sampled token 9") {
		t.Fatalf("err = %v, want repeated-token loop failure", err)
	}
}

func TestDriverProfileRunSafety_StopsRepeatedLineLoop_Bad(t *testing.T) {
	run := driverProfileRun{
		Output: "The sensor.\nThe sensor.\nThe sensor.",
		Metrics: mlx.Metrics{
			GeneratedTokens: 3,
		},
	}

	err := driverProfileRunSafetyError(1, run, driverProfileSafetyLimits{RepeatedLineLoopLimit: 3})

	if err == nil || !core.Contains(err.Error(), "repeated visible line") {
		t.Fatalf("err = %v, want repeated-line loop failure", err)
	}
}

func TestDriverProfileRunSafety_StopsRepeatedSentenceLoop_Bad(t *testing.T) {
	run := driverProfileRun{
		Output: "It was a packet of data. It changed shape. It was a packet of data. It moved. It was a packet of data. It hid. It was a packet of data.",
		Metrics: mlx.Metrics{
			GeneratedTokens: 16,
		},
	}

	err := driverProfileRunSafetyError(1, run, driverProfileSafetyLimits{RepeatedSentenceLoopLimit: 4})

	if err == nil || !core.Contains(err.Error(), "repeated visible sentence") {
		t.Fatalf("err = %v, want repeated-sentence loop failure", err)
	}
}

func TestDriverProfileRunSafety_StopsFragmentedOutput_Bad(t *testing.T) {
	run := driverProfileRun{
		Output: "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T.",
		Metrics: mlx.Metrics{
			GeneratedTokens: 32,
		},
	}

	err := driverProfileRunSafetyError(1, run, driverProfileSafetyLimits{})

	if err == nil || !core.Contains(err.Error(), "fragmented visible output") {
		t.Fatalf("err = %v, want fragmented output failure", err)
	}
}

func TestDriverProfileMetricsSafety_StopsVirtualMemoryOvershoot_Bad(t *testing.T) {
	err := driverProfileMetricsSafetyError("run 2", mlx.Metrics{
		ProcessVirtualMemoryBytes: 123,
	}, driverProfileSafetyLimits{
		MaxProcessVirtualMemoryBytes: 122,
	})

	if err == nil || !core.Contains(err.Error(), "process virtual memory safety limit") {
		t.Fatalf("err = %v, want process virtual safety failure", err)
	}
}

func TestDriverProfileSummary_IncludesFailedRunMemory_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{{
		Error: "safety stop",
		Metrics: mlx.Metrics{
			PeakMemoryBytes:            10,
			ActiveMemoryBytes:          11,
			CacheMemoryBytes:           12,
			ProcessVirtualMemoryBytes:  13,
			ProcessResidentMemoryBytes: 14,
			ProcessPeakResidentBytes:   15,
		},
	}})

	if summary.FailedRuns != 1 ||
		summary.PeakMemoryBytes != 10 ||
		summary.ActiveMemoryBytes != 11 ||
		summary.CacheMemoryBytes != 12 ||
		summary.ProcessVirtualMemoryBytes != 13 ||
		summary.ProcessResidentMemoryBytes != 14 ||
		summary.ProcessPeakResidentBytes != 15 {
		t.Fatalf("summary = %+v, want failed-run memory retained", summary)
	}
}

func TestDriverProfileSummary_PromptTokenStats_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{
		{VisibleTokens: 1, Metrics: mlx.Metrics{PromptTokens: 10, GeneratedTokens: 1}},
		{VisibleTokens: 1, Metrics: mlx.Metrics{PromptTokens: 20, GeneratedTokens: 1}},
		{Error: "failed", Metrics: mlx.Metrics{PromptTokens: 99}},
	})

	if summary.PromptTokensAverage != 15 || summary.PromptTokensMin != 10 || summary.PromptTokensMax != 20 {
		t.Fatalf("prompt token summary = avg:%v min:%d max:%d, want 15/10/20", summary.PromptTokensAverage, summary.PromptTokensMin, summary.PromptTokensMax)
	}
	if summary.SuccessfulRuns != 2 || summary.FailedRuns != 1 {
		t.Fatalf("run counts = success:%d failed:%d, want 2/1", summary.SuccessfulRuns, summary.FailedRuns)
	}
}

func TestDriverProfileSummary_MTPCounters_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{
		{
			VisibleTokens: 2,
			Metrics: mlx.Metrics{
				GeneratedTokens: 2,
				MTP: &mlx.MTPMetrics{
					ProposedTokens:         3,
					AcceptedTokens:         2,
					RejectedTokens:         1,
					TargetVerifyCalls:      2,
					DraftCalls:             2,
					AcceptanceRate:         2.0 / 3.0,
					VisibleTokensPerSec:    50,
					TargetTokensPerSec:     80,
					WarmDecodeTokensPerSec: 52,
				},
			},
		},
		{
			VisibleTokens: 2,
			Metrics: mlx.Metrics{
				GeneratedTokens: 2,
				MTP: &mlx.MTPMetrics{
					ProposedTokens:         4,
					AcceptedTokens:         4,
					RejectedTokens:         0,
					TargetVerifyCalls:      1,
					DraftCalls:             1,
					AcceptanceRate:         1,
					VisibleTokensPerSec:    60,
					TargetTokensPerSec:     90,
					WarmDecodeTokensPerSec: 64,
				},
			},
		},
		{Error: "failed", Metrics: mlx.Metrics{MTP: &mlx.MTPMetrics{ProposedTokens: 99}}},
	})

	if summary.MTPProposedTokens != 7 || summary.MTPAcceptedTokens != 6 || summary.MTPRejectedTokens != 1 {
		t.Fatalf("summary MTP counts = %+v, want proposed=7 accepted=6 rejected=1", summary)
	}
	if summary.MTPTargetVerifyCalls != 3 || summary.MTPDraftCalls != 3 {
		t.Fatalf("summary MTP calls = %+v, want verify=3 draft=3", summary)
	}
	if summary.MTPAcceptanceRateAverage <= 0.83 || summary.MTPAcceptanceRateAverage >= 0.84 {
		t.Fatalf("summary MTP acceptance avg = %f, want about 0.833", summary.MTPAcceptanceRateAverage)
	}
	if summary.MTPVisibleTokensPerSecAverage != 55 || summary.MTPTargetTokensPerSecAverage != 85 || summary.MTPWarmDecodeTokensPerSecAverage != 58 {
		t.Fatalf("summary MTP rates = %+v, want visible=55 target=85 warm=58", summary)
	}
}

func TestDriverProfileSummary_NativeEventBuckets_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{{
		VisibleTokens: 1,
		Metrics: mlx.Metrics{
			GeneratedTokens: 1,
			TokenPhases: []mlx.TokenPhaseTrace{{
				NativeEvents: []mlx.NativePhaseTrace{
					{Name: "gemma4.layer.00.attention", Duration: 2 * time.Millisecond, Pages: 2, Tokens: 2048},
					{Name: "gemma4.layer.01.attention", Duration: 4 * time.Millisecond, Pages: 8, Tokens: 8192},
					{Name: "gemma4.layer.01.ffn_router", Duration: 3 * time.Millisecond},
					{Name: "custom.event", Duration: time.Millisecond},
				},
			}},
		},
	}})

	if len(summary.NativeEvents) != 3 {
		t.Fatalf("native events = %+v, want three buckets", summary.NativeEvents)
	}
	if summary.NativeEvents[0].Name != "attention" || summary.NativeEvents[0].Count != 2 || summary.NativeEvents[0].Duration != 6*time.Millisecond || summary.NativeEvents[0].AverageDuration != 3*time.Millisecond {
		t.Fatalf("attention summary = %+v, want combined layer bucket", summary.NativeEvents[0])
	}
	if summary.NativeEvents[0].MaxPages != 8 || summary.NativeEvents[0].MaxTokens != 8192 {
		t.Fatalf("attention summary pages/tokens = %+v, want max 8 pages and 8192 tokens", summary.NativeEvents[0])
	}
	if summary.NativeEvents[1].Name != "ffn_router" || summary.NativeEvents[1].Duration != 3*time.Millisecond {
		t.Fatalf("router summary = %+v, want ffn_router bucket", summary.NativeEvents[1])
	}
	if summary.NativeEvents[2].Name != "custom.event" || summary.NativeEvents[2].Duration != time.Millisecond {
		t.Fatalf("custom summary = %+v, want original event name", summary.NativeEvents[2])
	}
	if len(summary.NativeEventDetails) != 4 {
		t.Fatalf("native event details = %+v, want four exact event buckets", summary.NativeEventDetails)
	}
	if summary.NativeEventDetails[0].Name != "gemma4.layer.01.attention" || summary.NativeEventDetails[0].Duration != 4*time.Millisecond {
		t.Fatalf("native event detail[0] = %+v, want exact layer attention bucket", summary.NativeEventDetails[0])
	}
}

func TestDriverProfileSummary_TokenPhaseBuckets_Good(t *testing.T) {
	summary := summariseDriverProfileRuns([]driverProfileRun{{
		VisibleTokens: 2,
		Metrics: mlx.Metrics{
			GeneratedTokens: 2,
			TokenPhases: []mlx.TokenPhaseTrace{
				{
					TotalDuration:      10 * time.Millisecond,
					ForwardDuration:    8 * time.Millisecond,
					PrefetchDuration:   time.Millisecond,
					SampleEvalDuration: time.Millisecond,
					OtherDuration:      time.Millisecond,
				},
				{
					TotalDuration:      20 * time.Millisecond,
					ForwardDuration:    18 * time.Millisecond,
					PrefetchDuration:   time.Millisecond,
					SampleEvalDuration: time.Millisecond,
					OtherDuration:      time.Millisecond,
				},
			},
		},
	}})

	if len(summary.TokenPhases) < 4 {
		t.Fatalf("token phase summary = %+v, want total/forward/sample_eval/other buckets", summary.TokenPhases)
	}
	if summary.TokenPhases[0].Name != "total" || summary.TokenPhases[0].Count != 2 || summary.TokenPhases[0].Duration != 30*time.Millisecond || summary.TokenPhases[0].AverageDuration != 15*time.Millisecond {
		t.Fatalf("total phase summary = %+v, want 30ms total and 15ms average", summary.TokenPhases[0])
	}
	if summary.TokenPhases[1].Name != "forward" || summary.TokenPhases[1].Duration != 26*time.Millisecond || summary.TokenPhases[1].AverageDuration != 13*time.Millisecond {
		t.Fatalf("forward phase summary = %+v, want 26ms total and 13ms average", summary.TokenPhases[1])
	}
}

func TestDriverProfileRunOverhead_ExcludesNativeMetricDuration_Good(t *testing.T) {
	got := driverRunOverhead(100*time.Millisecond, mlx.Metrics{TotalDuration: 60 * time.Millisecond})
	if got != 40*time.Millisecond {
		t.Fatalf("driverRunOverhead = %s, want 40ms", got)
	}
	if got := driverRunOverhead(60*time.Millisecond, mlx.Metrics{TotalDuration: 100 * time.Millisecond}); got != 0 {
		t.Fatalf("driverRunOverhead clamped = %s, want 0", got)
	}
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

func TestRunCommand_SliceSmokeJSON_Good(t *testing.T) {
	originalLoad := loadBenchModel
	originalRun := runBenchReport
	originalEstimate := runSliceSmokeEstimateCPUFFNMemory
	t.Cleanup(func() {
		loadBenchModel = originalLoad
		runBenchReport = originalRun
		runSliceSmokeEstimateCPUFFNMemory = originalEstimate
	})
	source := writeCLISlicePack(t)
	output := core.PathJoin(t.TempDir(), "client-slice")
	loadCalled := false
	var estimateSource string
	loadBenchModel = func(path string, opts ...mlx.LoadOption) (*mlx.Model, error) {
		loadCalled = true
		return &mlx.Model{}, nil
	}
	runSliceSmokeEstimateCPUFFNMemory = func(_ context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
		estimateSource = sourcePath
		return &mlx.CPUSplitFFNMemoryReport{
			Estimated:            true,
			TotalLayers:          1,
			LoadedLayers:         1,
			LayerLoads:           1,
			ResidentBytes:        64,
			PeakResidentBytes:    64,
			DenseEquivalentBytes: 96,
			SavedBytes:           32,
		}, nil
	}
	runBenchReport = func(ctx context.Context, model *mlx.Model, cfg bench.Config) (*bench.Report, error) {
		return &bench.Report{
			Version:   bench.ReportVersion,
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Generation: bench.GenerationSummary{
				Runs:                1,
				GeneratedTokens:     1,
				PrefillTokensPerSec: 100,
				DecodeTokensPerSec:  25,
				PeakMemoryBytes:     1024,
				ActiveMemoryBytes:   512,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"slice-smoke", "-json", "-preset", "client", "-output", output, "-prompt", "hi", "-max-tokens", "1", source}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if loadCalled {
		t.Fatal("slice-smoke loaded a client slice; want split-placement report without reload")
	}
	if estimateSource != source {
		t.Fatalf("estimate source = %q, want %q", estimateSource, source)
	}
	for _, want := range []string{`"slice"`, `"placement"`, `"requires_split_placement": true`, `"reload_skipped": true`, `"cpu_ffn_memory_estimate"`, `"resident_bytes": 64`, `"selected_tensor_bytes": "12"`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_SliceSmokeSplitJSON_Good(t *testing.T) {
	originalSplit := runSliceSmokeSplitGenerate
	t.Cleanup(func() { runSliceSmokeSplitGenerate = originalSplit })
	source := writeCLISlicePack(t)
	output := core.PathJoin(t.TempDir(), "client-slice")
	var gotPath, gotPrompt, gotDevice string
	var gotMaxTokens, gotContext, gotCache int
	runSliceSmokeSplitGenerate = func(_ context.Context, slicePath, prompt string, maxTokens, contextLen int, device string, cpuFFNCache int) (sliceSmokeSplitResult, error) {
		gotPath = slicePath
		gotPrompt = prompt
		gotMaxTokens = maxTokens
		gotContext = contextLen
		gotDevice = device
		gotCache = cpuFFNCache
		return sliceSmokeSplitResult{
			Output:   " split ok",
			Duration: time.Millisecond,
			CPUFFNMemory: &mlx.CPUSplitFFNMemoryReport{
				LoadedLayers:          1,
				PackedProjections:     3,
				PackedProjectionBytes: 3,
				PackedSidecarBytes:    24,
				ResidentBytes:         35,
				DenseEquivalentBytes:  56,
				SavedBytes:            21,
				ResidentRatio:         0.625,
			},
			CPUFFNMemoryEstimate: &mlx.CPUSplitFFNMemoryReport{
				Estimated:            true,
				TotalLayers:          2,
				LoadedLayers:         1,
				LayerLoads:           2,
				EvictedLayers:        1,
				ResidentBytes:        35,
				PeakResidentBytes:    35,
				DenseEquivalentBytes: 56,
				SavedBytes:           21,
			},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"slice-smoke", "-json", "-split", "-cpu-ffn-cache", "2", "-context", "32", "-device", "gpu", "-output", output, "-prompt", "hi", "-max-tokens", "3", source}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotPath != output || gotPrompt != "hi" || gotMaxTokens != 3 || gotContext != 32 || gotDevice != "gpu" || gotCache != 2 {
		t.Fatalf("split args path=%q prompt=%q max=%d context=%d device=%q cache=%d", gotPath, gotPrompt, gotMaxTokens, gotContext, gotDevice, gotCache)
	}
	for _, want := range []string{`"requires_split_placement": true`, `"split_output": " split ok"`, `"cpu_ffn_memory"`, `"cpu_ffn_memory_estimate"`, `"estimated": true`, `"layer_loads": 2`, `"packed_projection_bytes": 3`, `"saved_bytes": 21`} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
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

func TestRunCommand_DriverProfilePromptChunkBytes_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var got driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		got = cfg
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RequestedRuns:    cfg.Runs,
			Chat:             cfg.Chat,
			Summary:          driverProfileSummary{SuccessfulRuns: 1},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-chat=false", "-prompt-chunk-bytes", "4096", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if got.PromptChunkBytes != 4096 || got.Chat {
		t.Fatalf("driver profile cfg = %+v, want raw chunked prompt", got)
	}
	if !core.Contains(stdout.String(), `"prompt_chunk_bytes": 4096`) {
		t.Fatalf("stdout = %q, want prompt chunk bytes", stdout.String())
	}
}

func TestRunCommand_DriverProfilePromptChunkBytesChatMode_Good(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	var got driverProfileOptions
	runDriverProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg driverProfileOptions) (*driverProfileReport, error) {
		got = cfg
		return &driverProfileReport{
			Version:          1,
			ModelPath:        modelPath,
			PromptBytes:      len(cfg.Prompt),
			PromptChunkBytes: cfg.PromptChunkBytes,
			MaxTokens:        cfg.MaxTokens,
			RequestedRuns:    cfg.Runs,
			Chat:             cfg.Chat,
			Summary:          driverProfileSummary{SuccessfulRuns: 1},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt-chunk-bytes", "4096", "/models/demo"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if got.PromptChunkBytes != 4096 || !got.Chat {
		t.Fatalf("driver profile cfg = %+v, want chat chunked prompt", got)
	}
	if !core.Contains(stdout.String(), `"chat": true`) {
		t.Fatalf("stdout = %q, want chat mode", stdout.String())
	}
}

func TestRunCommand_DriverProfilePromptChunkBytes_Bad(t *testing.T) {
	originalRun := runDriverProfile
	t.Cleanup(func() { runDriverProfile = originalRun })
	runDriverProfile = func(_ context.Context, _ string, _ []mlx.LoadOption, _ driverProfileOptions) (*driverProfileReport, error) {
		t.Fatal("runDriverProfile called for invalid prompt chunk mode")
		return nil, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"driver-profile", "-json", "-prompt-chunk-bytes", "-1", "/models/demo"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "prompt chunk bytes must be >= 0") {
		t.Fatalf("stderr = %q, want prompt chunk bytes error", stderr.String())
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

func TestRunCommand_BenchMissingModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"bench"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), "go-mlx bench: expected one model path or -profile") {
		t.Fatalf("stderr = %q, want bench usage error", stderr.String())
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
