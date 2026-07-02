// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"time"

	core "dappco.re/go"
	traininf "dappco.re/go/inference/train"
	"dappco.re/go/mlx/spine"
)

// SSDCodeBenchmarkConfig configures native code-generation
// benchmark runs such as LiveCodeBench-v6.
type SSDCodeBenchmarkConfig struct {
	Benchmark  string               `json:"benchmark,omitempty"`
	NRepeat    int                  `json:"n_repeat,omitempty"`
	Generate   spine.GenerateConfig `json:"generate"`
	Seeds      []uint64             `json:"seeds,omitempty"`
	OutputPath string               `json:"output_path,omitempty"`
}

// SSDCodeBenchmarkRunner supplies generation and native
// code-execution test evaluation for each candidate.
type SSDCodeBenchmarkRunner struct {
	Generate func(context.Context, string, spine.GenerateConfig) (string, error)
	RunTests func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error)
}

// SSDCodeBenchmarkSample is one code benchmark task — an alias onto the
// canonical traininf.SSDCodeBenchmarkSample contract (identical fields,
// no engine dependency).
type SSDCodeBenchmarkSample = traininf.SSDCodeBenchmarkSample

// SSDCodeCandidate records one generated solution.
type SSDCodeCandidate struct {
	Repeat      int                  `json:"repeat"`
	Solution    string               `json:"solution"`
	RawSolution string               `json:"raw_solution,omitempty"`
	HasCode     bool                 `json:"has_code,omitempty"`
	Config      spine.GenerateConfig `json:"config"`
}

// SSDCodeExecution records the code-test outcome for one generated
// solution — an alias onto the canonical traininf.SSDCodeExecution
// contract (identical fields, no engine dependency).
type SSDCodeExecution = traininf.SSDCodeExecution

// SSDCodeBenchmarkCandidateResult joins a candidate with
// its native code-test execution result.
type SSDCodeBenchmarkCandidateResult struct {
	Candidate SSDCodeCandidate `json:"candidate"`
	Execution SSDCodeExecution `json:"execution"`
}

// SSDCodeBenchmarkSampleResult records all candidates for
// one benchmark task.
type SSDCodeBenchmarkSampleResult struct {
	Sample     SSDCodeBenchmarkSample            `json:"sample"`
	Candidates []SSDCodeBenchmarkCandidateResult `json:"candidates"`
}

// SSDCodeBenchmarkMetrics aggregates benchmark pass rates — an alias onto
// the canonical traininf.SSDCodeBenchmarkMetrics contract (identical
// fields, no engine dependency).
type SSDCodeBenchmarkMetrics = traininf.SSDCodeBenchmarkMetrics

// SSDCodeBenchmarkReport is the JSON-serialisable output of
// a native SSD code benchmark run.
type SSDCodeBenchmarkReport struct {
	Version   int                            `json:"version"`
	Benchmark string                         `json:"benchmark,omitempty"`
	Config    SSDCodeBenchmarkConfig         `json:"config"`
	Metrics   SSDCodeBenchmarkMetrics        `json:"metrics"`
	Results   []SSDCodeBenchmarkSampleResult `json:"results"`
	Duration  time.Duration                  `json:"duration,omitempty"`
}

// LoadSSDCodeBenchmarkJSONLFile loads benchmark tasks from a JSONL file
// path — delegates to the shared dappco.re/go/inference/train engine
// (byte-identical parsing, ported verbatim from this package).
func LoadSSDCodeBenchmarkJSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	return traininf.LoadSSDCodeBenchmarkJSONLFile(path)
}

// LoadSSDLiveCodeBenchV6JSONLFile loads the LiveCodeBench-v6 task subset
// from a JSONL file path — delegates to the shared
// dappco.re/go/inference/train engine.
func LoadSSDLiveCodeBenchV6JSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	return traininf.LoadSSDLiveCodeBenchV6JSONLFile(path)
}

// LoadSSDCodeBenchmarkJSONL loads LiveCodeBench-style JSONL task rows into
// native SSD code benchmark samples — delegates to the shared
// dappco.re/go/inference/train engine.
func LoadSSDCodeBenchmarkJSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	return traininf.LoadSSDCodeBenchmarkJSONL(raw)
}

// LoadSSDLiveCodeBenchV6JSONL loads LiveCodeBench-style JSONL and filters
// it to the v6 contest-date window — delegates to the shared
// dappco.re/go/inference/train engine.
func LoadSSDLiveCodeBenchV6JSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	return traininf.LoadSSDLiveCodeBenchV6JSONL(raw)
}

// FilterSSDLiveCodeBenchV6Samples keeps samples from the LiveCodeBench-v6
// contest-date window — delegates to the shared
// dappco.re/go/inference/train engine.
func FilterSSDLiveCodeBenchV6Samples(samples []SSDCodeBenchmarkSample) []SSDCodeBenchmarkSample {
	return traininf.FilterSSDLiveCodeBenchV6Samples(samples)
}

// RunSSDCodeBenchmark samples candidate code solutions and
// delegates native execution of each candidate against the sample tests.
func RunSSDCodeBenchmark(ctx context.Context, runner SSDCodeBenchmarkRunner, samples []SSDCodeBenchmarkSample, cfg SSDCodeBenchmarkConfig) (*SSDCodeBenchmarkReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if runner.Generate == nil {
		return nil, core.NewError("mlx: SSD code benchmark generate function is nil")
	}
	if runner.RunTests == nil {
		return nil, core.NewError("mlx: SSD code benchmark RunTests function is nil")
	}
	cfg = normalizeSSDCodeBenchmarkConfig(cfg)
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD code benchmark samples are empty")
	}

	start := time.Now()
	report := &SSDCodeBenchmarkReport{
		Version:   1,
		Benchmark: cfg.Benchmark,
		Config:    cfg,
		Results:   make([]SSDCodeBenchmarkSampleResult, 0, len(samples)),
	}
	for _, sample := range samples {
		if err := ctx.Err(); err != nil {
			return report, err
		}
		sampleResult := SSDCodeBenchmarkSampleResult{
			Sample:     cloneSSDCodeBenchmarkSample(sample),
			Candidates: make([]SSDCodeBenchmarkCandidateResult, 0, cfg.NRepeat),
		}
		for repeat := 0; repeat < cfg.NRepeat; repeat++ {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			prompt := ssdCodeBenchmarkGeneratePrompt(sample)
			generateCfg := ssdCodeBenchmarkRepeatGenerateConfig(cfg, repeat)
			rawSolution, err := runner.Generate(ctx, prompt, generateCfg)
			if err != nil {
				return report, err
			}
			solution, hasCode := SSDPostProcessCode(rawSolution)
			candidate := SSDCodeCandidate{
				Repeat:      repeat,
				Solution:    solution,
				RawSolution: rawSolution,
				HasCode:     hasCode,
				Config:      generateCfg,
			}
			execution, err := runner.RunTests(ctx, sample, candidate)
			if err != nil {
				return report, err
			}
			sampleResult.Candidates = append(sampleResult.Candidates, SSDCodeBenchmarkCandidateResult{
				Candidate: candidate,
				Execution: execution,
			})
			report.Metrics.Candidates++
			if execution.Passed {
				report.Metrics.Passed++
			}
		}
		report.Results = append(report.Results, sampleResult)
	}
	report.Metrics.Samples = len(samples)
	report.Metrics.Failed = report.Metrics.Candidates - report.Metrics.Passed
	if report.Metrics.Candidates > 0 {
		report.Metrics.PassRate = float64(report.Metrics.Passed) / float64(report.Metrics.Candidates)
	}
	report.Metrics.PassAtK = computeSSDCodeBenchmarkPassAtK(report.Results, cfg.NRepeat)
	report.Metrics.Difficulty = computeSSDCodeBenchmarkDifficultyMetrics(report.Results, cfg.NRepeat)
	report.Duration = nonZeroSSDCodeBenchmarkDuration(time.Since(start))
	if cfg.OutputPath != "" {
		if err := writeSSDCodeBenchmarkReport(cfg.OutputPath, report); err != nil {
			return report, err
		}
	}
	return report, nil
}

// SSDPostProcessCode extracts the final fenced code block from a model
// response and applies the LiveCodeBench code cleanup — delegates to the
// shared dappco.re/go/inference/train engine.
func SSDPostProcessCode(response string) (string, bool) {
	return traininf.SSDPostProcessCode(response)
}

// FormatSSDLiveCodeBenchPrompt returns the native prompt shape used for
// LiveCodeBench-v6-style code-generation tasks — delegates to the shared
// dappco.re/go/inference/train engine.
func FormatSSDLiveCodeBenchPrompt(sample SSDCodeBenchmarkSample) string {
	return traininf.FormatSSDLiveCodeBenchPrompt(sample)
}

func ssdCodeBenchmarkGeneratePrompt(sample SSDCodeBenchmarkSample) string {
	if sample.Meta == nil {
		return sample.Prompt
	}
	if _, ok := sample.Meta["is_stdin"]; !ok {
		return sample.Prompt
	}
	if prompt := FormatSSDLiveCodeBenchPrompt(sample); prompt != "" {
		return prompt
	}
	return sample.Prompt
}

func ssdCodeBenchmarkRepeatGenerateConfig(cfg SSDCodeBenchmarkConfig, repeat int) spine.GenerateConfig {
	generate := cfg.Generate
	if len(cfg.Seeds) > 0 {
		generate.Seed = cfg.Seeds[0] + uint64(repeat)
		generate.SeedSet = true
	}
	return generate
}

func normalizeSSDCodeBenchmarkConfig(cfg SSDCodeBenchmarkConfig) SSDCodeBenchmarkConfig {
	if cfg.NRepeat <= 0 {
		cfg.NRepeat = 1
	}
	if cfg.Generate.MaxTokens <= 0 {
		cfg.Generate.MaxTokens = defaultSSDMaxTokens
	}
	if cfg.Generate.TopK == 0 {
		cfg.Generate.TopK = defaultSSDTopK
	}
	if cfg.Generate.TopP == 0 {
		cfg.Generate.TopP = defaultSSDTopP
	}
	return cfg
}

func computeSSDCodeBenchmarkPassAtK(results []SSDCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := ssdCodeBenchmarkKList(nRepeat)
	if len(kList) == 0 || len(results) == 0 {
		return nil
	}
	sums := make(map[string]float64, len(kList))
	counts := make(map[string]int, len(kList))
	for _, result := range results {
		total := len(result.Candidates)
		if total == 0 {
			continue
		}
		correct := 0
		for _, candidate := range result.Candidates {
			if candidate.Execution.Passed {
				correct++
			}
		}
		for _, k := range kList {
			if total < k {
				continue
			}
			key := core.Sprintf("pass@%d", k)
			sums[key] += estimateSSDCodeBenchmarkPassAtK(total, correct, k)
			counts[key]++
		}
	}
	out := make(map[string]float64, len(sums))
	for key, sum := range sums {
		if counts[key] > 0 {
			out[key] = sum / float64(counts[key])
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func computeSSDCodeBenchmarkDifficultyMetrics(results []SSDCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := ssdCodeBenchmarkKList(nRepeat)
	if len(kList) == 0 {
		return nil
	}
	type bucket struct {
		sum   float64
		count int
	}
	buckets := make(map[string]bucket)
	for _, result := range results {
		if result.Sample.Meta == nil {
			continue
		}
		difficulty := core.Trim(result.Sample.Meta["difficulty"])
		if difficulty == "" {
			continue
		}
		total := len(result.Candidates)
		if total == 0 {
			continue
		}
		correct := 0
		for _, candidate := range result.Candidates {
			if candidate.Execution.Passed {
				correct++
			}
		}
		for _, k := range kList {
			if total < k {
				continue
			}
			key := core.Sprintf("pass@%d_%s", k, difficulty)
			value := buckets[key]
			value.sum += estimateSSDCodeBenchmarkPassAtK(total, correct, k)
			value.count++
			buckets[key] = value
		}
	}
	out := make(map[string]float64, len(buckets))
	for key, bucket := range buckets {
		if bucket.count > 0 {
			out[key] = bucket.sum / float64(bucket.count)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func ssdCodeBenchmarkKList(nRepeat int) []int {
	kList := []int{1}
	if nRepeat >= 10 {
		kList = append(kList, 5)
	}
	if nRepeat >= 20 {
		kList = append(kList, 10)
	}
	if nRepeat >= 32 {
		kList = append(kList, 16)
	}
	if nRepeat >= 40 {
		kList = append(kList, 20)
	}
	if nRepeat >= 64 {
		kList = append(kList, 32)
	}
	return kList
}

func estimateSSDCodeBenchmarkPassAtK(total, correct, k int) float64 {
	if total <= 0 || correct <= 0 || k <= 0 {
		return 0
	}
	if total-correct < k {
		return 1
	}
	fail := 1.0
	for i := 0; i < k; i++ {
		fail *= float64(total-correct-i) / float64(total-i)
	}
	return 1 - fail
}

func cloneSSDCodeBenchmarkSample(sample SSDCodeBenchmarkSample) SSDCodeBenchmarkSample {
	return SSDCodeBenchmarkSample{
		ID:     sample.ID,
		Prompt: sample.Prompt,
		Tests:  core.SliceClone(sample.Tests),
		Meta:   core.MapClone(sample.Meta),
	}
}

func writeSSDCodeBenchmarkReport(path string, report *SSDCodeBenchmarkReport) error {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		return data.Value.(error)
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return result.Value.(error)
		}
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return result.Value.(error)
	}
	return nil
}

func nonZeroSSDCodeBenchmarkDuration(value time.Duration) time.Duration {
	if value <= 0 {
		return time.Nanosecond
	}
	return value
}
