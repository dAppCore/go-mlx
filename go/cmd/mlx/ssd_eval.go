// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"
	"strconv"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

type ssdEvalPlanReport struct {
	Version       int                 `json:"version"`
	Kind          string              `json:"kind"`
	NoPython      bool                `json:"no_python"`
	SamplePath    string              `json:"sample_path,omitempty"`
	OutputPath    string              `json:"output_path,omitempty"`
	LiveCodeBench bool                `json:"livecodebench_v6,omitempty"`
	Samples       int                 `json:"samples"`
	Config        ssdRecipeEvalConfig `json:"config"`
	Notes         []string            `json:"notes,omitempty"`
}

func runSSDEvalCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ssd-eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON eval plan")
	samplesPath := fs.String("samples", "", "LiveCodeBench-style task JSONL path")
	outputPath := fs.String("output", "", "output path for a later benchmark report")
	liveCodeBenchV6 := fs.Bool("livecodebench-v6", true, "filter JSONL to the LiveCodeBench-v6 contest-date window")
	nRepeat := fs.Int("n-repeat", 0, "number of generated candidates per task")
	maxTokens := fs.Int("max-tokens", 0, "maximum generated tokens per candidate")
	temperature := fs.Float64("temperature", -1, "sampling temperature")
	topP := fs.Float64("top-p", -1, "sampling top-p")
	topK := fs.Int("top-k", -1, "sampling top-k")
	minP := fs.Float64("min-p", -1, "sampling min-p")
	samplingParams := fs.String("sampling-params", "", "comma-separated sampling params, e.g. temperature=0.9,top_p=0.8,top_k=20")
	fs.Usage = func() {
		name := cliCommandName("ssd-eval")
		core.WriteString(stderr, core.Sprintf("Usage: %s -samples <livecodebench.jsonl> [flags]\n", name))
		core.WriteString(stderr, "Prepare a native Simple Self-Distillation LiveCodeBench eval plan.\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s ssd-eval: expected no positional arguments", cliName())
		return 2
	}
	if core.Trim(*samplesPath) == "" {
		core.Print(stderr, "%s ssd-eval: samples path is required", cliName())
		return 2
	}
	cfg := mlx.DefaultSSDCodeBenchmarkConfig()
	cfg.OutputPath = core.Trim(*outputPath)
	if *nRepeat > 0 {
		cfg.NRepeat = *nRepeat
	}
	if err := applySSDEvalSamplingParams(&cfg, *samplingParams); err != nil {
		core.Print(stderr, "%s ssd-eval: %v", cliName(), err)
		return 2
	}
	if *maxTokens > 0 {
		cfg.Generate.MaxTokens = *maxTokens
	}
	if *temperature >= 0 {
		cfg.Generate.Temperature = float32(*temperature)
	}
	if *topP >= 0 {
		cfg.Generate.TopP = float32(*topP)
	}
	if *topK >= 0 {
		cfg.Generate.TopK = *topK
	}
	if *minP >= 0 {
		cfg.Generate.MinP = float32(*minP)
	}
	samples, err := loadSSDEvalSamples(*samplesPath, *liveCodeBenchV6)
	if err != nil {
		core.Print(stderr, "%s ssd-eval: %v", cliName(), err)
		return 2
	}
	report := ssdEvalPlanReport{
		Version:       1,
		Kind:          "simple-self-distillation-eval-plan",
		NoPython:      true,
		SamplePath:    core.Trim(*samplesPath),
		OutputPath:    cfg.OutputPath,
		LiveCodeBench: *liveCodeBenchV6,
		Samples:       len(samples),
		Config:        ssdRecipeEvalConfigFromConfig(cfg),
		Notes: []string{
			"RunSSDCodeBenchmark owns the native generate-and-test loop; CLI planning stops before model wiring and language execution.",
			"LiveCodeBench code execution remains caller-supplied through SSDCodeBenchmarkRunner.RunTests.",
		},
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s ssd-eval: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, core.AsString(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	core.WriteString(stdout, "simple self-distillation eval plan\n")
	core.WriteString(stdout, core.Sprintf("  samples: %d\n", report.Samples))
	core.WriteString(stdout, core.Sprintf("  benchmark: %s n_repeat=%d max_tokens=%d temperature=%.3g top_p=%.3g top_k=%d\n",
		report.Config.Benchmark,
		report.Config.NRepeat,
		report.Config.Generate.MaxTokens,
		report.Config.Generate.Temperature,
		report.Config.Generate.TopP,
		report.Config.Generate.TopK,
	))
	return 0
}

func loadSSDEvalSamples(path string, liveCodeBenchV6 bool) ([]mlx.SSDCodeBenchmarkSample, error) {
	if liveCodeBenchV6 {
		return mlx.LoadSSDLiveCodeBenchV6JSONLFile(path)
	}
	return mlx.LoadSSDCodeBenchmarkJSONLFile(path)
}

func applySSDEvalSamplingParams(cfg *mlx.SSDCodeBenchmarkConfig, raw string) error {
	raw = core.Trim(raw)
	if raw == "" {
		return nil
	}
	for _, part := range core.Split(raw, ",") {
		part = core.Trim(part)
		if part == "" {
			continue
		}
		separator := core.Index(part, "=")
		if separator < 0 {
			return core.Errorf("invalid sampling param %q", part)
		}
		key := part[:separator]
		value := part[separator+1:]
		key = core.Replace(core.Trim(key), "-", "_")
		value = core.Trim(value)
		switch key {
		case "temperature", "temp":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return core.Errorf("invalid temperature %q", value)
			}
			cfg.Generate.Temperature = parsed
		case "top_p":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return core.Errorf("invalid top_p %q", value)
			}
			cfg.Generate.TopP = parsed
		case "top_k":
			parsed := core.Atoi(value)
			if !parsed.OK {
				return core.Errorf("invalid top_k %q", value)
			}
			cfg.Generate.TopK = parsed.Value.(int)
		case "min_p":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return core.Errorf("invalid min_p %q", value)
			}
			cfg.Generate.MinP = parsed
		case "max_tokens":
			parsed := core.Atoi(value)
			if !parsed.OK {
				return core.Errorf("invalid max_tokens %q", value)
			}
			cfg.Generate.MaxTokens = parsed.Value.(int)
		default:
			return core.Errorf("unknown sampling param %q", key)
		}
	}
	return nil
}

func parseSSDEvalFloat32(value string) (float32, error) {
	parsed, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return 0, err
	}
	return float32(parsed), nil
}
