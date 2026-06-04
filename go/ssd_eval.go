// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
)

// SimpleSelfDistillationCodeBenchmarkConfig configures native code-generation
// benchmark runs such as LiveCodeBench-v6.
type SimpleSelfDistillationCodeBenchmarkConfig struct {
	Benchmark  string         `json:"benchmark,omitempty"`
	NRepeat    int            `json:"n_repeat,omitempty"`
	Generate   GenerateConfig `json:"generate"`
	Seeds      []uint64       `json:"seeds,omitempty"`
	OutputPath string         `json:"output_path,omitempty"`
}

// SimpleSelfDistillationCodeBenchmarkRunner supplies generation and native
// code-execution test evaluation for each candidate.
type SimpleSelfDistillationCodeBenchmarkRunner struct {
	Generate func(context.Context, string, GenerateConfig) (string, error)
	RunTests func(context.Context, SimpleSelfDistillationCodeBenchmarkSample, SimpleSelfDistillationCodeCandidate) (SimpleSelfDistillationCodeExecution, error)
}

// SimpleSelfDistillationCodeBenchmarkSample is one code benchmark task.
type SimpleSelfDistillationCodeBenchmarkSample struct {
	ID     string            `json:"id,omitempty"`
	Prompt string            `json:"prompt"`
	Tests  []string          `json:"tests,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

// SimpleSelfDistillationCodeCandidate records one generated solution.
type SimpleSelfDistillationCodeCandidate struct {
	Repeat      int            `json:"repeat"`
	Solution    string         `json:"solution"`
	RawSolution string         `json:"raw_solution,omitempty"`
	HasCode     bool           `json:"has_code,omitempty"`
	Config      GenerateConfig `json:"config"`
}

// SimpleSelfDistillationCodeExecution records the code-test outcome for one
// generated solution.
type SimpleSelfDistillationCodeExecution struct {
	Passed      bool          `json:"passed"`
	PassedTests int           `json:"passed_tests,omitempty"`
	TotalTests  int           `json:"total_tests,omitempty"`
	Duration    time.Duration `json:"duration,omitempty"`
	DurationMS  int64         `json:"duration_ms,omitempty"`
	Stdout      string        `json:"stdout,omitempty"`
	Stderr      string        `json:"stderr,omitempty"`
	Error       string        `json:"error,omitempty"`
}

// SimpleSelfDistillationCodeBenchmarkCandidateResult joins a candidate with
// its native code-test execution result.
type SimpleSelfDistillationCodeBenchmarkCandidateResult struct {
	Candidate SimpleSelfDistillationCodeCandidate `json:"candidate"`
	Execution SimpleSelfDistillationCodeExecution `json:"execution"`
}

// SimpleSelfDistillationCodeBenchmarkSampleResult records all candidates for
// one benchmark task.
type SimpleSelfDistillationCodeBenchmarkSampleResult struct {
	Sample     SimpleSelfDistillationCodeBenchmarkSample            `json:"sample"`
	Candidates []SimpleSelfDistillationCodeBenchmarkCandidateResult `json:"candidates"`
}

// SimpleSelfDistillationCodeBenchmarkMetrics aggregates benchmark pass rates.
type SimpleSelfDistillationCodeBenchmarkMetrics struct {
	Samples    int                `json:"samples,omitempty"`
	Candidates int                `json:"candidates,omitempty"`
	Passed     int                `json:"passed,omitempty"`
	Failed     int                `json:"failed,omitempty"`
	PassRate   float64            `json:"pass_rate,omitempty"`
	PassAtK    map[string]float64 `json:"pass_at_k,omitempty"`
	Difficulty map[string]float64 `json:"difficulty,omitempty"`
}

// SimpleSelfDistillationCodeBenchmarkReport is the JSON-serialisable output of
// a native SSD code benchmark run.
type SimpleSelfDistillationCodeBenchmarkReport struct {
	Version   int                                               `json:"version"`
	Benchmark string                                            `json:"benchmark,omitempty"`
	Config    SimpleSelfDistillationCodeBenchmarkConfig         `json:"config"`
	Metrics   SimpleSelfDistillationCodeBenchmarkMetrics        `json:"metrics"`
	Results   []SimpleSelfDistillationCodeBenchmarkSampleResult `json:"results"`
	Duration  time.Duration                                     `json:"duration,omitempty"`
}

type simpleSelfDistillationCodeBenchmarkJSONLRecord struct {
	ID               string            `json:"id"`
	QuestionID       string            `json:"question_id"`
	TaskID           string            `json:"task_id"`
	Prompt           string            `json:"prompt"`
	Question         string            `json:"question"`
	QuestionContent  string            `json:"question_content"`
	Problem          string            `json:"problem"`
	StarterCode      string            `json:"starter_code"`
	EntryPoint       string            `json:"entry_point"`
	IsStdin          *bool             `json:"is_stdin"`
	ContestDate      string            `json:"contest_date"`
	Test             string            `json:"test"`
	Tests            []string          `json:"tests"`
	PublicTestCases  []string          `json:"public_test_cases"`
	PrivateTestCases []string          `json:"private_test_cases"`
	Metadata         map[string]string `json:"metadata"`
	Difficulty       string            `json:"difficulty"`
	Platform         string            `json:"platform"`
}

// LoadSimpleSelfDistillationCodeBenchmarkJSONLFile loads benchmark tasks from
// a JSONL file path.
func LoadSimpleSelfDistillationCodeBenchmarkJSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	return LoadSimpleSelfDistillationCodeBenchmarkJSONL(core.AsString(read.Value.([]byte)))
}

// LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile loads the LiveCodeBench-v6
// task subset from a JSONL file path.
func LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	return LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(core.AsString(read.Value.([]byte)))
}

// LoadSimpleSelfDistillationCodeBenchmarkJSONL loads LiveCodeBench-style JSONL
// task rows into native SSD code benchmark samples.
func LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	lines := core.Split(raw, "\n")
	samples := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, len(lines))
	for index, line := range lines {
		line = core.Trim(line)
		if line == "" {
			continue
		}
		var record simpleSelfDistillationCodeBenchmarkJSONLRecord
		if result := core.JSONUnmarshalString(line, &record); !result.OK {
			return nil, core.Errorf("mlx: parse SSD code benchmark JSONL record %d: %w", index+1, result.Value.(error))
		}
		sample, ok := record.simpleSelfDistillationCodeBenchmarkSample()
		if !ok {
			continue
		}
		samples = append(samples, sample)
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD code benchmark JSONL produced no samples")
	}
	return samples, nil
}

// LoadSimpleSelfDistillationLiveCodeBenchV6JSONL loads LiveCodeBench-style
// JSONL and filters it to the v6 contest-date window.
func LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(raw string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	samples, err := LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw)
	if err != nil {
		return nil, err
	}
	samples = FilterSimpleSelfDistillationLiveCodeBenchV6Samples(samples)
	if len(samples) == 0 {
		return nil, core.NewError("mlx: LiveCodeBench-v6 JSONL produced no samples")
	}
	return samples, nil
}

// FilterSimpleSelfDistillationLiveCodeBenchV6Samples keeps samples from the
// LiveCodeBench-v6 contest-date window.
func FilterSimpleSelfDistillationLiveCodeBenchV6Samples(samples []SimpleSelfDistillationCodeBenchmarkSample) []SimpleSelfDistillationCodeBenchmarkSample {
	filtered := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, len(samples))
	for _, sample := range samples {
		if simpleSelfDistillationLiveCodeBenchV6ContestDate(sample.Meta["contest_date"]) {
			filtered = append(filtered, cloneSimpleSelfDistillationCodeBenchmarkSample(sample))
		}
	}
	return filtered
}

func simpleSelfDistillationLiveCodeBenchV6ContestDate(date string) bool {
	date = core.Trim(date)
	return date >= "2025-02-01" && date < "2025-06-01"
}

func (r simpleSelfDistillationCodeBenchmarkJSONLRecord) simpleSelfDistillationCodeBenchmarkSample() (SimpleSelfDistillationCodeBenchmarkSample, bool) {
	prompt := firstSimpleSelfDistillationCodeBenchmarkString(r.Prompt, r.QuestionContent, r.Question, r.Problem)
	if prompt == "" {
		return SimpleSelfDistillationCodeBenchmarkSample{}, false
	}
	if starterCode := core.Trim(r.StarterCode); starterCode != "" {
		prompt = core.Concat(prompt, "\n\nstarter code:\n", starterCode)
	}
	tests := appendSimpleSelfDistillationCodeBenchmarkTests(nil, r.Tests...)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, r.Test)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, r.PublicTestCases...)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, r.PrivateTestCases...)
	meta := core.MapClone(r.Metadata)
	if meta == nil {
		meta = make(map[string]string, 2)
	}
	if difficulty := core.Trim(r.Difficulty); difficulty != "" {
		meta["difficulty"] = difficulty
	}
	if platform := core.Trim(r.Platform); platform != "" {
		meta["platform"] = platform
	}
	if entryPoint := core.Trim(r.EntryPoint); entryPoint != "" {
		meta["entry_point"] = entryPoint
	}
	if contestDate := core.Trim(r.ContestDate); contestDate != "" {
		meta["contest_date"] = contestDate
	}
	if r.IsStdin != nil {
		meta["is_stdin"] = core.Sprintf("%t", *r.IsStdin)
	}
	if len(meta) == 0 {
		meta = nil
	}
	return SimpleSelfDistillationCodeBenchmarkSample{
		ID:     firstSimpleSelfDistillationCodeBenchmarkString(r.ID, r.QuestionID, r.TaskID),
		Prompt: prompt,
		Tests:  tests,
		Meta:   meta,
	}, true
}

// RunSimpleSelfDistillationCodeBenchmark samples candidate code solutions and
// delegates native execution of each candidate against the sample tests.
func RunSimpleSelfDistillationCodeBenchmark(ctx context.Context, runner SimpleSelfDistillationCodeBenchmarkRunner, samples []SimpleSelfDistillationCodeBenchmarkSample, cfg SimpleSelfDistillationCodeBenchmarkConfig) (*SimpleSelfDistillationCodeBenchmarkReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if runner.Generate == nil {
		return nil, core.NewError("mlx: SSD code benchmark generate function is nil")
	}
	if runner.RunTests == nil {
		return nil, core.NewError("mlx: SSD code benchmark RunTests function is nil")
	}
	cfg = normalizeSimpleSelfDistillationCodeBenchmarkConfig(cfg)
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD code benchmark samples are empty")
	}

	start := time.Now()
	report := &SimpleSelfDistillationCodeBenchmarkReport{
		Version:   1,
		Benchmark: cfg.Benchmark,
		Config:    cfg,
		Results:   make([]SimpleSelfDistillationCodeBenchmarkSampleResult, 0, len(samples)),
	}
	for _, sample := range samples {
		if err := ctx.Err(); err != nil {
			return report, err
		}
		sampleResult := SimpleSelfDistillationCodeBenchmarkSampleResult{
			Sample:     cloneSimpleSelfDistillationCodeBenchmarkSample(sample),
			Candidates: make([]SimpleSelfDistillationCodeBenchmarkCandidateResult, 0, cfg.NRepeat),
		}
		for repeat := 0; repeat < cfg.NRepeat; repeat++ {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			prompt := simpleSelfDistillationCodeBenchmarkGeneratePrompt(sample)
			generateCfg := simpleSelfDistillationCodeBenchmarkRepeatGenerateConfig(cfg, repeat)
			rawSolution, err := runner.Generate(ctx, prompt, generateCfg)
			if err != nil {
				return report, err
			}
			solution, hasCode := SimpleSelfDistillationPostProcessCode(rawSolution)
			candidate := SimpleSelfDistillationCodeCandidate{
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
			sampleResult.Candidates = append(sampleResult.Candidates, SimpleSelfDistillationCodeBenchmarkCandidateResult{
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
	report.Metrics.PassAtK = computeSimpleSelfDistillationCodeBenchmarkPassAtK(report.Results, cfg.NRepeat)
	report.Metrics.Difficulty = computeSimpleSelfDistillationCodeBenchmarkDifficultyMetrics(report.Results, cfg.NRepeat)
	report.Duration = nonZeroSimpleSelfDistillationCodeBenchmarkDuration(time.Since(start))
	if cfg.OutputPath != "" {
		if err := writeSimpleSelfDistillationCodeBenchmarkReport(cfg.OutputPath, report); err != nil {
			return report, err
		}
	}
	return report, nil
}

// SimpleSelfDistillationPostProcessCode extracts the final fenced code block
// from a model response and applies the LiveCodeBench code cleanup.
func SimpleSelfDistillationPostProcessCode(response string) (string, bool) {
	code, ok := lastSimpleSelfDistillationCodeFence(response)
	if !ok {
		return "", false
	}
	return simpleSelfDistillationPostProcessCode(code), true
}

// FormatSimpleSelfDistillationLiveCodeBenchPrompt returns the native prompt
// shape used for LiveCodeBench-v6-style code-generation tasks.
func FormatSimpleSelfDistillationLiveCodeBenchPrompt(sample SimpleSelfDistillationCodeBenchmarkSample) string {
	prompt := core.Trim(sample.Prompt)
	if prompt == "" {
		return ""
	}
	if sample.Meta != nil && sample.Meta["is_stdin"] == "false" {
		if entryPoint := core.Trim(sample.Meta["entry_point"]); entryPoint != "" {
			return core.Concat(
				"Write a Python solution for the problem. Return only the program inside one python code block.\n\nProblem:\n",
				prompt,
				"\n\nStarter code:\n```python\n",
				entryPoint,
				"\n```",
			)
		}
	}
	return core.Concat(
		"Write a Python program for the problem. Read from stdin, write to stdout, and return only the program inside one python code block.\n\nProblem:\n",
		prompt,
	)
}

func simpleSelfDistillationCodeBenchmarkGeneratePrompt(sample SimpleSelfDistillationCodeBenchmarkSample) string {
	if sample.Meta == nil {
		return sample.Prompt
	}
	if _, ok := sample.Meta["is_stdin"]; !ok {
		return sample.Prompt
	}
	if prompt := FormatSimpleSelfDistillationLiveCodeBenchPrompt(sample); prompt != "" {
		return prompt
	}
	return sample.Prompt
}

func simpleSelfDistillationCodeBenchmarkRepeatGenerateConfig(cfg SimpleSelfDistillationCodeBenchmarkConfig, repeat int) GenerateConfig {
	generate := cfg.Generate
	if len(cfg.Seeds) > 0 {
		generate.Seed = cfg.Seeds[0] + uint64(repeat)
		generate.SeedSet = true
	}
	return generate
}

func normalizeSimpleSelfDistillationCodeBenchmarkConfig(cfg SimpleSelfDistillationCodeBenchmarkConfig) SimpleSelfDistillationCodeBenchmarkConfig {
	if cfg.NRepeat <= 0 {
		cfg.NRepeat = 1
	}
	if cfg.Generate.MaxTokens <= 0 {
		cfg.Generate.MaxTokens = defaultSimpleSelfDistillationMaxTokens
	}
	if cfg.Generate.TopK == 0 {
		cfg.Generate.TopK = defaultSimpleSelfDistillationTopK
	}
	if cfg.Generate.TopP == 0 {
		cfg.Generate.TopP = defaultSimpleSelfDistillationTopP
	}
	return cfg
}

func computeSimpleSelfDistillationCodeBenchmarkPassAtK(results []SimpleSelfDistillationCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := simpleSelfDistillationCodeBenchmarkKList(nRepeat)
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
			sums[key] += estimateSimpleSelfDistillationCodeBenchmarkPassAtK(total, correct, k)
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

func computeSimpleSelfDistillationCodeBenchmarkDifficultyMetrics(results []SimpleSelfDistillationCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := simpleSelfDistillationCodeBenchmarkKList(nRepeat)
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
			value.sum += estimateSimpleSelfDistillationCodeBenchmarkPassAtK(total, correct, k)
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

func simpleSelfDistillationCodeBenchmarkKList(nRepeat int) []int {
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

func estimateSimpleSelfDistillationCodeBenchmarkPassAtK(total, correct, k int) float64 {
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

func cloneSimpleSelfDistillationCodeBenchmarkSample(sample SimpleSelfDistillationCodeBenchmarkSample) SimpleSelfDistillationCodeBenchmarkSample {
	return SimpleSelfDistillationCodeBenchmarkSample{
		ID:     sample.ID,
		Prompt: sample.Prompt,
		Tests:  core.SliceClone(sample.Tests),
		Meta:   core.MapClone(sample.Meta),
	}
}

func firstSimpleSelfDistillationCodeBenchmarkString(values ...string) string {
	for _, value := range values {
		if trimmed := core.Trim(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func appendSimpleSelfDistillationCodeBenchmarkTests(target []string, values ...string) []string {
	for _, value := range values {
		if trimmed := core.Trim(value); trimmed != "" {
			target = append(target, trimmed)
		}
	}
	return target
}

func lastSimpleSelfDistillationCodeFence(response string) (string, bool) {
	var last string
	found := false
	remaining := response
	for {
		start := core.Index(remaining, "```")
		if start < 0 {
			break
		}
		afterStart := remaining[start+3:]
		newline := core.Index(afterStart, "\n")
		if newline < 0 {
			break
		}
		bodyStart := newline + 1
		afterLanguage := afterStart[bodyStart:]
		end := core.Index(afterLanguage, "```")
		if end < 0 {
			break
		}
		last = afterLanguage[:end]
		found = true
		remaining = afterLanguage[end+3:]
	}
	return last, found
}

func simpleSelfDistillationPostProcessCode(code string) string {
	code = firstSimpleSelfDistillationSegment(code, "</code>")
	code = core.Replace(code, "```python", "")
	code = firstSimpleSelfDistillationSegment(code, "```")
	code = core.Replace(code, "<code>", "")
	return code
}

func firstSimpleSelfDistillationSegment(value, delimiter string) string {
	if index := core.Index(value, delimiter); index >= 0 {
		return value[:index]
	}
	return value
}

func writeSimpleSelfDistillationCodeBenchmarkReport(path string, report *SimpleSelfDistillationCodeBenchmarkReport) error {
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

func nonZeroSimpleSelfDistillationCodeBenchmarkDuration(value time.Duration) time.Duration {
	if value <= 0 {
		return time.Nanosecond
	}
	return value
}
