// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the native SSD code-benchmark surface. Each
// carries an Output: comment so it executes under `go test` and doubles as the
// usage doc (AX principle 2). Generation and test-execution are injected, so
// the benchmark loop, the LiveCodeBench JSONL loader, and the code-fence
// extraction run with no model and no Python sandbox.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	"context"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/mlx/spine"
)

// ExampleRunSSDCodeBenchmark samples one candidate per task and delegates
// native test execution to the injected RunTests hook. The "add" solution
// contains a "+" so its tests pass; the "sub" solution does not, so the pass
// rate over the two single-candidate tasks is 0.5. pass@1 equals the pass rate
// when there is one candidate per task.
func ExampleRunSSDCodeBenchmark() {
	report, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			if strings.Contains(prompt, "add") {
				return "```python\ndef add(a, b): return a + b\n```", nil
			}
			return "```python\ndef sub(a, b): return a - b\n```", nil
		},
		RunTests: func(_ context.Context, _ SSDCodeBenchmarkSample, candidate SSDCodeCandidate) (SSDCodeExecution, error) {
			return SSDCodeExecution{Passed: strings.Contains(candidate.Solution, "+")}, nil
		},
	}, []SSDCodeBenchmarkSample{
		{ID: "add", Prompt: "write add"},
		{ID: "sub", Prompt: "write sub"},
	}, SSDCodeBenchmarkConfig{NRepeat: 1})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", report.Metrics.Samples)
	core.Println("candidates:", report.Metrics.Candidates)
	core.Println("passed:", report.Metrics.Passed)
	core.Println("pass_rate:", report.Metrics.PassRate)
	core.Println("pass@1:", report.Metrics.PassAtK["pass@1"])
	// Output:
	// samples: 2
	// candidates: 2
	// passed: 1
	// pass_rate: 0.5
	// pass@1: 0.5
}

// ExampleLoadSSDLiveCodeBenchV6JSONL loads LiveCodeBench-style task rows and
// filters them to the v6 contest-date window (2025-02-01 inclusive to
// 2025-06-01 exclusive). The January and June rows fall outside the window, so
// only the February and May tasks survive.
func ExampleLoadSSDLiveCodeBenchV6JSONL() {
	raw := strings.Join([]string{
		`{"id":"jan","prompt":"old","contest_date":"2025-01-31"}`,
		`{"id":"feb","prompt":"first v6","contest_date":"2025-02-01"}`,
		`{"id":"may","prompt":"last v6","contest_date":"2025-05-31"}`,
		`{"id":"jun","prompt":"new","contest_date":"2025-06-01"}`,
	}, "\n")

	samples, err := LoadSSDLiveCodeBenchV6JSONL(raw)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(samples))
	core.Println("ids:", samples[0].ID, samples[1].ID)
	// Output:
	// samples: 2
	// ids: feb may
}

// ExampleSSDPostProcessCode extracts the final fenced code block from a model
// response and strips the LiveCodeBench markup. An earlier go fence is ignored
// — the last python fence wins — and the <code> tags are removed.
func ExampleSSDPostProcessCode() {
	response := "analysis\n```go\nnot this\n```\nfinal\n```python\n<code>def add(a, b):\n    return a + b</code>\n```\n"
	code, ok := SSDPostProcessCode(response)
	core.Println("ok:", ok)
	core.Println(core.Trim(code))
	// Output:
	// ok: true
	// def add(a, b):
	//     return a + b
}
