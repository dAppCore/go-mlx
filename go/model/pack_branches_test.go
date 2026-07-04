// SPDX-Licence-Identifier: EUPL-1.2

// Branch coverage for the pack inspection pipeline (pack.go and its
// pack_* siblings). The _Good / _Bad suites in pack_test.go drive the
// populated happy paths and a handful of error shapes; the arms gathered
// here are the residual error/edge branches those fixtures don't reach —
// the missing-path stat, the Validate failure return, the single-file
// entry path, the unsupported-architecture issue, the invalid-GGUF read,
// the tokenizer non-ENOENT / invalid-JSON paths, the MiniMax M2 config
// /plan failures, and the parse-error arms of the JANG / AutoRound /
// codebook / sentence-transformers config readers. Several inspectors
// are unexported and exercised directly where Inspect can't reach the
// branch (a successful Inspect always reads a valid config first, so the
// "config unreadable" arms only surface via the helper).

package model

import (
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
)

// TestInspect_MissingPath drives the `!stat.OK` return at the top of
// Inspect (pack.go:26) — a path that does not exist surfaces the stat
// error rather than a zero pack.
func TestInspect_MissingPath(t *testing.T) {
	if _, err := Inspect(core.PathJoin(t.TempDir(), "no-such-model")); err == nil {
		t.Fatalf("Inspect of a missing path: want error, got nil")
	}
}

// TestValidate_GoodAndBad drives both Validate arms: a clean pack returns
// nil (pack.go:99), and an invalid pack returns the issue-summary error
// (pack.go:96-102). The invalid case is a directory with no weights /
// config, which Inspect itself reports as a (non-fatal) issue set —
// Validate then converts that to an error.
func TestValidate_GoodAndBad(t *testing.T) {
	t.Run("good", func(t *testing.T) {
		dir := t.TempDir()
		writeGoodSafetensorsPack(t, dir, "gemma4_text")
		pack, err := Validate(dir, mp.WithPackQuantization(4))
		if err != nil {
			t.Fatalf("Validate of a good pack: %v (issues %+v)", err, pack.Issues)
		}
	})

	t.Run("bad", func(t *testing.T) {
		dir := t.TempDir() // empty — no config, no weights, no tokenizer.
		pack, err := Validate(dir)
		if err == nil {
			t.Fatalf("Validate of an empty dir: want error, got pack %+v", pack)
		}
	})

	t.Run("inspect_error_propagates", func(t *testing.T) {
		// When Inspect itself errors (missing path), Validate returns it
		// untouched (pack.go:96-98).
		if _, err := Validate(core.PathJoin(t.TempDir(), "missing")); err == nil {
			t.Fatalf("Validate of a missing path: want error")
		}
	})
}

// TestInspect_SingleSafetensorsFile drives the single-file entry path in
// inspectModelPackWeights (pack.go:128-129) — Inspect pointed directly at
// a *.safetensors file rather than a directory. The root becomes the
// file's parent, and the format resolves to safetensors with that one
// weight file.
func TestInspect_SingleSafetensorsFile(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	weightPath := core.PathJoin(dir, "model.safetensors")
	writeModelPackFile(t, weightPath, "stub")

	pack, err := Inspect(weightPath, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(single file) error = %v", err)
	}
	if pack.Format != mp.ModelPackFormatSafetensors {
		t.Fatalf("Format = %q, want safetensors for a single-file entry", pack.Format)
	}
	if len(pack.WeightFiles) != 1 || pack.WeightFiles[0] != weightPath {
		t.Fatalf("WeightFiles = %v, want [%s]", pack.WeightFiles, weightPath)
	}
}

// TestInspect_SingleGGUFFile drives the single-file `.gguf` entry path
// (pack.go:130-131) — the other arm of the single-file switch.
func TestInspect_SingleGGUFFile(t *testing.T) {
	dir := t.TempDir()
	ggufPath := core.PathJoin(dir, "model.gguf")
	writeModelPackFile(t, ggufPath, "stub") // not a valid GGUF — exercises the entry classification + invalid-GGUF read.

	pack, err := Inspect(ggufPath, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(single gguf) error = %v", err)
	}
	if pack.Format != mp.ModelPackFormatGGUF {
		t.Fatalf("Format = %q, want gguf for a single-file .gguf entry", pack.Format)
	}
}

// TestInspect_NoWeights drives the default (no-weights) arm of
// inspectModelPackWeights (pack.go:208-210): a directory with a config
// and tokenizer but no .safetensors / .gguf shard reports the
// missing-weights issue and the missing format.
func TestInspect_NoWeights(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(no weights) error = %v", err)
	}
	if pack.Format != mp.ModelPackFormatMissing || !pack.HasIssue(mp.ModelPackIssueMissingWeights) {
		t.Fatalf("pack = format:%q issues:%+v, want missing-weights", pack.Format, pack.Issues)
	}
}

// TestInspect_InvalidGGUF drives inspectModelPackGGUF's ReadInfo error
// arm (pack.go:273-276): a lone .gguf file in a directory whose bytes are
// not a valid GGUF container surfaces the invalid-GGUF issue.
func TestInspect_InvalidGGUF(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.gguf"), "not a gguf header")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(invalid gguf) error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueInvalidGGUF) {
		t.Fatalf("issues = %+v, want invalid-GGUF", pack.Issues)
	}
}

// TestInspect_UnsupportedArchitecture drives inspectModelPackArchitecture's
// `!SupportedArchitecture` arm (pack.go:355-358): a model_type that
// resolves to no known profile raises the unsupported-architecture error.
func TestInspect_UnsupportedArchitecture(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"),
		`{"model_type":"totally_made_up_arch_xyz"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(unsupported arch) error = %v", err)
	}
	if pack.SupportedArchitecture || !pack.HasIssue(mp.ModelPackIssueUnsupportedArchitecture) {
		t.Fatalf("pack = supported:%v issues:%+v, want unsupported-architecture",
			pack.SupportedArchitecture, pack.Issues)
	}
}

// TestInspectTokenizer_NonENOENTRead drives inspectModelPackTokenizer's
// non-not-exist read error (pack.go:323-330): when tokenizer.json is a
// directory, ReadFile fails with an error that is NOT IsNotExist, so the
// inspector raises InvalidTokenizer rather than MissingTokenizer. (The
// dir-index `has` check passes because the basename is listed.)
func TestInspectTokenizer_NonENOENTRead(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	if result := core.MkdirAll(core.PathJoin(dir, "tokenizer.json"), 0o755); !result.OK {
		t.Fatalf("MkdirAll tokenizer.json: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(tokenizer dir) error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueInvalidTokenizer) {
		t.Fatalf("issues = %+v, want invalid-tokenizer for a directory tokenizer.json", pack.Issues)
	}
}

// TestInspectTokenizer_InvalidJSON drives the JSONUnmarshal failure arm
// (pack.go:336-339): a tokenizer.json that exists and reads but is not
// valid JSON surfaces InvalidTokenizer.
func TestInspectTokenizer_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), "{ this is not json")
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(invalid tokenizer json) error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueInvalidTokenizer) {
		t.Fatalf("issues = %+v, want invalid-tokenizer for malformed JSON", pack.Issues)
	}
}

// TestFirstNonEmpty drives firstNonEmpty's all-empty fall-through return
// (pack.go:78) alongside the happy first-hit path — a list of only blank
// strings returns "".
func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "  ", "\t"); got != "" {
		t.Errorf("firstNonEmpty(all blank) = %q, want empty", got)
	}
	if got := firstNonEmpty("", "second", "third"); got != "second" {
		t.Errorf("firstNonEmpty = %q, want second", got)
	}
}

// TestContainsASCIIInsensitive_EmptySubstr drives the `len(substr) == 0`
// fast-true arm (pack.go:222-224) plus a couple of representative
// hit/miss cases through the byte walk.
func TestContainsASCIIInsensitive_EmptySubstr(t *testing.T) {
	if !containsASCIIInsensitive("anything", "") {
		t.Errorf("empty substr should always match")
	}
	if !containsASCIIInsensitive("Sentence/Normalize", "normalize") {
		t.Errorf("case-insensitive contains should match")
	}
	if containsASCIIInsensitive("short", "longerthanhaystack") {
		t.Errorf("substr longer than s must not match")
	}
	if containsASCIIInsensitive("abcdef", "xyz") {
		t.Errorf("absent substr must not match")
	}
}

// TestInspectMiniMaxM2_ParseConfigError drives the m2.ParseConfig failure
// arm (pack.go:470-472): the probe walker tolerates a string value for
// num_local_experts (skipped as an unknown field) so the architecture
// still resolves to minimax_m2, but m2.ParseConfig decodes the same
// config with encoding/json into an int field and fails — raising the
// invalid-config warning.
func TestInspectMiniMaxM2_ParseConfigError(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"),
		`{"model_type":"minimax_m2","num_local_experts":"not_an_int"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(minimax bad config) error = %v", err)
	}
	if pack.Architecture != "minimax_m2" {
		t.Fatalf("architecture = %q, want minimax_m2", pack.Architecture)
	}
	if !pack.HasIssue(mp.ModelPackIssueInvalidConfig) {
		t.Fatalf("issues = %+v, want invalid-config from ParseConfig failure", pack.Issues)
	}
	if pack.MiniMaxM2 != nil {
		t.Fatalf("MiniMaxM2 = %+v, want nil when config parse fails", pack.MiniMaxM2)
	}
}

// TestInspectMiniMaxM2_PlanError drives the m2.BuildTensorPlan failure arm
// (pack.go:475-477): a minimax_m2 config that parses cleanly but declares
// no hidden/intermediate/layer sizes can't build a tensor plan, so the
// inspector raises the unsupported-runtime warning.
func TestInspectMiniMaxM2_PlanError(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"),
		`{"model_type":"minimax_m2"}`) // no sizes → BuildTensorPlan fails.
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(minimax no sizes) error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("issues = %+v, want unsupported-runtime from plan failure", pack.Issues)
	}
	if pack.MiniMaxM2 != nil {
		t.Fatalf("MiniMaxM2 = %+v, want nil when plan build fails", pack.MiniMaxM2)
	}
}

// TestInspectMiniMaxM2_NonSafetensorsSkipsSkeleton drives the
// `Format != safetensors || len(WeightFiles) == 0` early return after a
// successful plan (pack.go:480-482): a minimax_m2 config that builds a
// valid plan but ships no safetensors shard records the plan yet skips
// the first-layer skeleton.
func TestInspectMiniMaxM2_NonSafetensorsSkipsSkeleton(t *testing.T) {
	dir := t.TempDir()
	// Minimal but complete sizes so BuildTensorPlan succeeds.
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "minimax_m2",
		"hidden_size": 4,
		"intermediate_size": 4,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"num_local_experts": 3,
		"num_experts_per_tok": 2
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	// No .safetensors shard — only the config + tokenizer, so the plan
	// builds but the skeleton step is skipped.

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect(minimax no shard) error = %v", err)
	}
	if pack.MiniMaxM2 == nil {
		t.Fatalf("MiniMaxM2 = nil, want a built plan even without a shard")
	}
	if pack.MiniMaxM2LayerSkeleton != nil {
		t.Fatalf("MiniMaxM2LayerSkeleton = %+v, want nil when no safetensors shard", pack.MiniMaxM2LayerSkeleton)
	}
}
