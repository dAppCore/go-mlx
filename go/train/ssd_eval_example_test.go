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

// ExampleLoadSSDCodeBenchmarkJSONL parses LiveCodeBench-style JSONL rows into
// native code-benchmark samples. The id/prompt/test fields map straight across;
// blank lines are skipped.
func ExampleLoadSSDCodeBenchmarkJSONL() {
	raw := strings.Join([]string{
		`{"id":"q1","prompt":"Write add.","tests":["assert add(1, 2) == 3"]}`,
		`{"id":"q2","prompt":"Write sub.","test":"assert sub(3, 1) == 2"}`,
	}, "\n")

	samples, err := LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(samples))
	core.Println("first id:", samples[0].ID)
	core.Println("first test:", samples[0].Tests[0])
	// Output:
	// samples: 2
	// first id: q1
	// first test: assert add(1, 2) == 3
}

// ExampleLoadSSDCodeBenchmarkJSONLFile loads benchmark tasks from a JSONL file
// path. The file is written to a temp dir, then read back into native samples.
func ExampleLoadSSDCodeBenchmarkJSONLFile() {
	dirResult := core.MkdirTemp("", "lcb-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "lcb.jsonl")
	if w := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","tests":["assert f(1) == 1"]}`+"\n"), 0o644); !w.OK {
		core.Println("write error:", w.Value)
		return
	}

	samples, err := LoadSSDCodeBenchmarkJSONLFile(path)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(samples))
	core.Println("id:", samples[0].ID)
	// Output:
	// samples: 1
	// id: q
}

// ExampleLoadSSDLiveCodeBenchV6JSONLFile loads the v6 contest-date subset from a
// JSONL file path. The single in-window row survives the filter.
func ExampleLoadSSDLiveCodeBenchV6JSONLFile() {
	dirResult := core.MkdirTemp("", "lcb-v6-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "lcb-v6.jsonl")
	if w := core.WriteFile(path, []byte(`{"id":"mar","prompt":"v6 task","contest_date":"2025-03-15"}`+"\n"), 0o644); !w.OK {
		core.Println("write error:", w.Value)
		return
	}

	samples, err := LoadSSDLiveCodeBenchV6JSONLFile(path)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(samples))
	core.Println("id:", samples[0].ID)
	// Output:
	// samples: 1
	// id: mar
}

// ExampleFilterSSDLiveCodeBenchV6Samples keeps only the samples whose
// contest_date meta falls in the v6 window (Feb–May 2025), dropping the rest.
func ExampleFilterSSDLiveCodeBenchV6Samples() {
	in := []SSDCodeBenchmarkSample{
		{ID: "jan", Meta: map[string]string{"contest_date": "2025-01-15"}},
		{ID: "mar", Meta: map[string]string{"contest_date": "2025-03-15"}},
		{ID: "jun", Meta: map[string]string{"contest_date": "2025-06-15"}},
	}
	kept := FilterSSDLiveCodeBenchV6Samples(in)
	core.Println("kept:", len(kept))
	core.Println("id:", kept[0].ID)
	// Output:
	// kept: 1
	// id: mar
}

// ExampleFormatSSDLiveCodeBenchPrompt frames a stdin-style task with the
// read-from-stdin instruction wrapped around the problem statement.
func ExampleFormatSSDLiveCodeBenchPrompt() {
	prompt := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{
		Prompt: "Add two numbers from stdin.",
		Meta:   map[string]string{"is_stdin": "true"},
	})
	core.Println("has stdin framing:", strings.Contains(prompt, "stdin"))
	core.Println("has problem:", strings.Contains(prompt, "Add two numbers from stdin."))
	// Output:
	// has stdin framing: true
	// has problem: true
}
