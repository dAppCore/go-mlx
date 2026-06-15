// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"
)

func TestGRPORewardContainsAnswer_ExtractsReasoningAnswer_Good(t *testing.T) {
	sample := GRPOSample{
		Prompt:          "Solve",
		ReferenceAnswer: "reasoning trace\n\n42",
		ExpectedAnswer:  ExtractGRPOExpectedAnswer(dataset.Sample{Response: "reasoning trace\n\n42"}),
	}
	reward, err := GRPORewardContainsAnswer(2)(GRPORewardContext{
		Sample:  sample,
		Rollout: GRPORollout{Text: "The final answer is 42."},
	})
	if err != nil {
		t.Fatalf("GRPORewardContainsAnswer() error = %v", err)
	}
	if reward.Score != 2 || reward.Name == "" {
		t.Fatalf("reward = %+v, want weighted answer match", reward)
	}
}

func TestGRPORewardExactAnswerAndMetadataErrors_Bad(t *testing.T) {
	reward, err := GRPORewardExactAnswer(0)(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "alpha"},
		Rollout: GRPORollout{Answer: "beta"},
	})
	if err != nil {
		t.Fatalf("GRPORewardExactAnswer() error = %v", err)
	}
	if reward.Score != 0 || reward.Weight != 1 || reward.Detail != "missing" {
		t.Fatalf("reward = %+v, want default weight miss", reward)
	}
	if err := SaveGRPOCheckpointMetadata("", GRPOCheckpointMetadata{}); err == nil {
		t.Fatal("SaveGRPOCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadGRPOCheckpointMetadata(""); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(empty) error = nil")
	}
	dir := t.TempDir()
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir), "{")
	if _, err := LoadGRPOCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(invalid JSON) error = nil")
	}
	if _, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
			return nil, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{ResumePath: dir}); err == nil {
		t.Fatal("RunGRPOReasoningTraining(invalid resume metadata) error = nil")
	}
}

func TestExtractGRPOExpectedAnswer_MetaPrefixMultiLine_Good(t *testing.T) {
	// Explicit meta answer key wins over the response body.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "lots of reasoning here\nfinal answer: 99",
		Meta:     map[string]string{"solution": " 1729 "},
	}); got != "1729" {
		t.Fatalf("meta key answer = %q, want trimmed 1729", got)
	}
	// No meta — last non-empty line carries an "answer:" prefix that must
	// be stripped case-insensitively (drives cleanGRPOAnswerLine +
	// asciiHasPrefixFold against the mixed-case prefix).
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "Add the numbers.\nFinal Answer: 42",
	}); got != "42" {
		t.Fatalf("multi-line answer prefix = %q, want stripped 42", got)
	}
	// CRLF normalisation path + a blank trailing line: the backward walk
	// must skip the empty tail and recover the real answer line.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "step one\r\nstep two\r\nSOLUTION: seven\r\n",
	}); got != "seven" {
		t.Fatalf("CRLF + trailing-blank answer = %q, want seven", got)
	}
	// Falls back to Text when Response is empty; single-line fast path.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Text: "answer: Paris"}); got != "Paris" {
		t.Fatalf("text fallback single line = %q, want Paris", got)
	}
}

func TestExtractGRPOExpectedAnswer_EmptyAndUnprefixed_Bad(t *testing.T) {
	// Nothing to extract from — empty meta, empty response, empty text.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{}); got != "" {
		t.Fatalf("empty sample answer = %q, want empty", got)
	}
	// A single-char line "a" is shorter than every prefix, so
	// asciiHasPrefixFold returns false on the len(s)<len(prefix) gate and
	// the raw line is returned verbatim — not mistaken for "answer:".
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "a"}); got != "a" {
		t.Fatalf("one-char trigger line = %q, want verbatim a", got)
	}
	// A multi-line body whose only non-blank line has no prefix returns
	// that line unchanged via the backward walk.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "\n\njust a sentence"}); got != "just a sentence" {
		t.Fatalf("unprefixed multi-line = %q, want the sentence", got)
	}
}

func TestExtractGRPOExpectedAnswer_AllBlankLines_Ugly(t *testing.T) {
	// Every line is whitespace-only — the backward walk must exhaust all
	// boundaries and return empty rather than emitting a blank "answer".
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "   \n\t\n  "}); got != "" {
		t.Fatalf("all-blank multi-line answer = %q, want empty", got)
	}
}

func TestExtractGRPOExpectedAnswer_WalkSkipsCleanedEmptyTail_Ugly(t *testing.T) {
	// The outer Trim keeps the tail non-blank, but a trailing answer-PREFIX
	// line ("answer:" with nothing after it) cleans to empty inside
	// cleanGRPOAnswerLine. The backward walk must step past that emptied
	// tail (end = start) and return the earlier real line — it is not the
	// same as a whitespace-only tail (which Trim removes upstream).
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "context line\nanswer:"}); got != "context line" {
		t.Fatalf("cleaned-empty tail = %q, want the earlier context line", got)
	}
	// Every line is an answer-prefix that cleans to empty: the walk steps
	// through all boundaries (end = start) and then hits start < 0 on the
	// first segment, returning empty rather than a blank answer.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "answer:\nsolution:"}); got != "" {
		t.Fatalf("all-prefix-empty multi-line = %q, want empty", got)
	}
}

func TestGRPOSampleFromSFT_ReasoningMetaAndSuffixStrip_Good(t *testing.T) {
	// meta["reasoning"] is taken verbatim and short-circuits the
	// suffix-strip path in extractGRPOReasoningWithAnswer.
	got := GRPOSampleFromSFT(dataset.Sample{
		Prompt:   "  Solve 1+1  ",
		Response: "the answer is 2",
		Meta:     map[string]string{"reasoning": "carry the one", "answer": "2"},
	})
	if got.Prompt != "Solve 1+1" {
		t.Fatalf("prompt = %q, want trimmed", got.Prompt)
	}
	if got.ExpectedAnswer != "2" || got.Reasoning != "carry the one" {
		t.Fatalf("sample = %+v, want meta answer + meta reasoning", got)
	}
	if got.Meta["answer"] != "2" {
		t.Fatalf("meta = %+v, want cloned meta carried through", got.Meta)
	}
}

func TestGRPOSampleFromSFT_ThinkingKeyAndComputedSuffix_Good(t *testing.T) {
	// No reasoning key but a thinking key — the second meta branch.
	thinking := GRPOSampleFromSFT(dataset.Sample{
		Prompt: "p", Response: "trace then 7", Meta: map[string]string{"thinking": "ponder"},
	})
	if thinking.Reasoning != "ponder" {
		t.Fatalf("reasoning = %q, want thinking-key value", thinking.Reasoning)
	}
	// No meta at all: the answer is the last line ("42"), and reasoning is
	// the full response with that answer suffix trimmed off (drives
	// extractGRPOReasoningWithAnswer's TrimSuffix branch).
	computed := GRPOSampleFromSFT(dataset.Sample{Prompt: "p", Response: "reasoning trace\n42"})
	if computed.ExpectedAnswer != "42" {
		t.Fatalf("expected answer = %q, want 42", computed.ExpectedAnswer)
	}
	if computed.Reasoning != "reasoning trace" {
		t.Fatalf("reasoning = %q, want answer suffix trimmed", computed.Reasoning)
	}
}

func TestGRPOSampleFromSFT_PromptlessAndAnswerlessReasoning_Bad(t *testing.T) {
	// Prompt falls back to Text when Prompt is empty.
	fromText := GRPOSampleFromSFT(dataset.Sample{Text: "  prompt in text  "})
	if fromText.Prompt != "prompt in text" {
		t.Fatalf("prompt = %q, want text fallback", fromText.Prompt)
	}
	// Empty response → no answer → empty reasoning (the answer=="" early
	// return inside extractGRPOReasoningWithAnswer).
	noAnswer := GRPOSampleFromSFT(dataset.Sample{Prompt: "p", Response: ""})
	if noAnswer.ExpectedAnswer != "" || noAnswer.Reasoning != "" {
		t.Fatalf("sample = %+v, want empty answer and reasoning", noAnswer)
	}
}

func TestGRPORewardContainsAnswer_EmptyAndUnicodeFallback_Bad(t *testing.T) {
	fn := GRPORewardContainsAnswer(1)
	// Empty expected answer → neutral reward with the explanatory detail.
	empty, err := fn(GRPORewardContext{Sample: GRPOSample{ExpectedAnswer: "  "}})
	if err != nil {
		t.Fatalf("empty expected error = %v", err)
	}
	if empty.Score != 0 || empty.Detail != "no expected answer" {
		t.Fatalf("reward = %+v, want neutral no-answer reward", empty)
	}
	// Non-ASCII expected answer forces the unicode core.Join+Lower
	// fallback; a match still scores the weight.
	hit, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "café"},
		Rollout: GRPORollout{Text: "Le CAFÉ est ouvert"},
	})
	if err != nil {
		t.Fatalf("unicode reward error = %v", err)
	}
	if hit.Score != 1 || hit.Detail != "matched" {
		t.Fatalf("reward = %+v, want unicode-fallback match", hit)
	}
	// Non-ASCII expected absent from all fragments → miss via the same
	// fallback path.
	miss, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "café"},
		Rollout: GRPORollout{Text: "no match here"},
	})
	if err != nil {
		t.Fatalf("unicode miss error = %v", err)
	}
	if miss.Score != 0 || miss.Detail != "missing" {
		t.Fatalf("reward = %+v, want unicode-fallback miss", miss)
	}
}

// TestGRPORewardContainsAnswer_WeightDefaultsToOne_Good covers the
// weight==0 normalisation in the reward constructor: a zero weight is
// rewritten to 1 so a matching rollout still scores a unit reward.
func TestGRPORewardContainsAnswer_WeightDefaultsToOne_Good(t *testing.T) {
	reward, err := GRPORewardContainsAnswer(0)(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "42"},
		Rollout: GRPORollout{Text: "the answer is 42"},
	})
	if err != nil {
		t.Fatalf("GRPORewardContainsAnswer(0) error = %v", err)
	}
	if reward.Weight != 1 || reward.Score != 1 || reward.Detail != "matched" {
		t.Fatalf("reward = %+v, want zero weight normalised to 1 with unit score", reward)
	}
}

// TestContainsFoldASCII_DirectBranches_Ugly exercises containsFoldASCII
// directly — the reward closure that normally calls it short-circuits
// the empty-expected case and gates non-ASCII out beforehand, so these
// branches are only reachable by calling the helper itself. It is the
// case-insensitive ASCII substring primitive behind the contains-answer
// reward fast path.
func TestContainsFoldASCII_DirectBranches_Ugly(t *testing.T) {
	// Empty substr → vacuously contained, ASCII-ok.
	if hit, ok := containsFoldASCII("anything", ""); !hit || !ok {
		t.Fatalf("containsFoldASCII(_, \"\") = (%v,%v), want (true,true)", hit, ok)
	}
	// Non-ASCII byte in substr → rejected for the ASCII fast path.
	if hit, ok := containsFoldASCII("le café", "café"); hit || ok {
		t.Fatalf("containsFoldASCII(non-ASCII substr) = (%v,%v), want (false,false)", hit, ok)
	}
	// substr longer than s → no match, ASCII-ok.
	if hit, ok := containsFoldASCII("hi", "hello"); hit || !ok {
		t.Fatalf("containsFoldASCII(too-long substr) = (%v,%v), want (false,true)", hit, ok)
	}
	// Case-insensitive hit where s carries uppercase that must fold to match.
	if hit, ok := containsFoldASCII("The PARIS line", "paris"); !hit || !ok {
		t.Fatalf("containsFoldASCII(uppercase fold) = (%v,%v), want (true,true)", hit, ok)
	}
	// First byte matches at an offset but a later byte mismatches (drives
	// the inner case-fold compare + break): "para" vs "PARIS" — 'p','a','r'
	// fold-match, then 'a' != 'i' breaks; no other start offset matches.
	if hit, ok := containsFoldASCII("a PARANORMAL clue", "paris"); hit || !ok {
		t.Fatalf("containsFoldASCII(inner mismatch) = (%v,%v), want (false,true)", hit, ok)
	}
}

// TestGRPORewardStats_EmptyRollouts_Bad covers the zero-length guard in
// grpoRewardStats: with no rollouts the mean and std are both zero
// rather than a divide-by-zero NaN.
func TestGRPORewardStats_EmptyRollouts_Bad(t *testing.T) {
	mean, std := grpoRewardStats(nil)
	if mean != 0 || std != 0 {
		t.Fatalf("grpoRewardStats(nil) = (%v,%v), want (0,0)", mean, std)
	}
}
