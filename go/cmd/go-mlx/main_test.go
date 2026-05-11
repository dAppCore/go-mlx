// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
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

	code := runCommand(context.Background(), []string{"pack", "-json", "-quantization", "4", "-max-context", "65536", dir}, stdout, stderr)
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

func TestRunCommand_BenchMissingModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"bench"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), "go-mlx bench: expected exactly one model path") {
		t.Fatalf("stderr = %q, want bench usage error", stderr.String())
	}
}
