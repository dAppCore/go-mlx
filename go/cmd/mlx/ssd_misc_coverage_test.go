// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// ssd with a bad flag fails to parse (exit 2).
func TestRunSSD_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"ssd", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// ssd with malformed prompt JSONL reaches the data-parse error (exit 1).
func TestRunSSD_DataParseError_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "prompts.jsonl")
	if r := core.WriteFile(data, []byte("{not json\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ssd", "--model", dir, "--data", data}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (data parse error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "prompt data parse") {
		t.Fatalf("stderr = %q, want the data-parse notice", stderr.String())
	}
}

// ssd-recipes (human-readable, no -json) prints the recipe summary lines.
func TestRunSSDRecipes_HumanSummary_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ssd-recipes"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "simple self-distillation recipes") || !strings.Contains(out, "data-gen:") || !strings.Contains(out, "eval:") {
		t.Fatalf("stdout = %q, want the recipe summary", out)
	}
}

// ssd-recipes with a bad flag fails to parse (exit 2); with a positional arg it
// is a usage error (exit 2).
func TestRunSSDRecipes_BadInputs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"ssd-recipes", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("bad-flag exit = %d, want 2", code)
	}
	stdout, stderr = core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ssd-recipes", "stray"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("positional exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "expected no positional arguments") {
		t.Fatalf("stderr = %q, want the positional notice", stderr.String())
	}
}

// ssd-eval (human-readable, no -json) prints the eval-plan summary, with the
// explicit sampling flags applied and the non-v6 loader exercised.
func TestRunSSDEval_HumanSummaryExplicitFlags_Good(t *testing.T) {
	dir := t.TempDir()
	samples := core.JoinPath(dir, "tasks.jsonl")
	if r := core.WriteFile(samples, []byte(
		`{"id":"t1","prompt":"Write add.","tests":["assert add(1,2)==3"]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd-eval", "-samples", samples, "-livecodebench-v6=false",
		"-n-repeat", "5", "-max-tokens", "128", "-temperature", "0.7", "-top-p", "0.9", "-top-k", "40", "-min-p", "0.05",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "simple self-distillation eval plan") || !strings.Contains(out, "n_repeat=5") {
		t.Fatalf("stdout = %q, want the eval-plan summary with the explicit flags", out)
	}
}

// ssd-eval with a malformed sampling-params string surfaces the parse error
// (exit 2).
func TestRunSSDEval_BadSamplingParams_Bad(t *testing.T) {
	dir := t.TempDir()
	samples := core.JoinPath(dir, "tasks.jsonl")
	if r := core.WriteFile(samples, []byte(`{"id":"t1","prompt":"x"}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd-eval", "-samples", samples, "-sampling-params", "temperature=not-a-number",
	}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad sampling params); stderr=%q", code, stderr.String())
	}
}

// ssd-eval with a bad flag fails to parse (exit 2); with a positional arg it is
// a usage error (exit 2).
func TestRunSSDEval_BadInputs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"ssd-eval", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("bad-flag exit = %d, want 2", code)
	}
	stdout, stderr = core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"ssd-eval", "-samples", "/x", "stray"}, stdout, stderr); code != 2 {
		t.Fatalf("positional exit = %d, want 2", code)
	}
}

// memory-pretrain-build with a bad flag (exit 2) and with a missing corpus
// (exit 2 validation).
func TestRunMemoryPretrainBuild_BadInputs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"memory-pretrain-build", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("bad-flag exit = %d, want 2", code)
	}
	stdout, stderr = core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"memory-pretrain-build", "-router", "/r.json"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("missing-corpus exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "corpus path is required") {
		t.Fatalf("stderr = %q, want the corpus-required notice", stderr.String())
	}
}

// memory-pretrain-build with an unreadable corpus reaches the corpus open-error
// path (exit 2, after flag validation).
func TestRunMemoryPretrainBuild_CorpusUnreadable_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"memory-pretrain-build", "-corpus", core.JoinPath(t.TempDir(), "absent.jsonl"),
		"-router", core.JoinPath(t.TempDir(), "r.json"), "-ffn-memory", core.JoinPath(t.TempDir(), "f.json"),
	}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (corpus open error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no such file") && !strings.Contains(stderr.String(), "open") {
		t.Fatalf("stderr = %q, want the corpus open error", stderr.String())
	}
}

// memory-pretrain-build with a malformed -tokens list surfaces the int-parse
// error (exit 2).
func TestRunMemoryPretrainBuild_BadTokens_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"memory-pretrain-build", "-corpus", "/c.jsonl", "-tokens", "8,oops,32",
	}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad tokens); stderr=%q", code, stderr.String())
	}
}

// memory-pretrain-build (human-readable, no -json) builds the artifacts and
// prints the human summary lines — the non-JSON output lane the JSON test skips.
func TestRunMemoryPretrainBuild_HumanSummary_Good(t *testing.T) {
	dir := t.TempDir()
	corpus := core.JoinPath(dir, "corpus.jsonl")
	if r := core.WriteFile(corpus, []byte(
		`{"id":"a","text":"Go memory planning","meta":{"source":"docs"}}`+"\n"+
			`{"id":"b","text":"winter proof poem","meta":{"source":"creative"}}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write corpus: %v", r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"memory-pretrain-build",
		"-corpus", corpus,
		"-router", core.JoinPath(dir, "router.json"),
		"-ffn-memory", core.JoinPath(dir, "ffn.json"),
		"-hidden-size", "8", "-layers", "2", "-levels", "1", "-tokens", "2",
		"-branching", "2", "-depth", "1", "-min-cluster-size", "1", "-kmeans-iters", "4",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "memory-pretraining artifacts") && !strings.Contains(stdout.String(), "corpus") {
		t.Fatalf("stdout = %q, want the human summary", stdout.String())
	}
}
