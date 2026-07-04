// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"testing"

	"dappco.re/go/mlx/dataset"
)

// TestGrpoReward_GRPOSampleFromSFT_Good covers the happy reasoning-sample
// extraction paths: the reasoning meta key wins verbatim, the thinking
// meta key is the second branch, and with no meta at all the answer is
// recovered from the final response line and the reasoning is the body
// with that answer suffix trimmed off.
func TestGrpoReward_GRPOSampleFromSFT_Good(t *testing.T) {
	t.Run("reasoning_meta_and_suffix_strip", func(t *testing.T) {
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
	})
	t.Run("thinking_key_and_computed_suffix", func(t *testing.T) {
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
	})
}

// TestGrpoReward_GRPOSampleFromSFT_Bad covers the degenerate inputs: the
// prompt falls back to Text when Prompt is empty, and an empty response
// yields no answer and therefore empty reasoning (the answer=="" early
// return inside extractGRPOReasoningWithAnswer).
func TestGrpoReward_GRPOSampleFromSFT_Bad(t *testing.T) {
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

// TestGrpoReward_GRPOSampleFromSFT_Ugly covers the fully-empty sample:
// no prompt, no text, no response, no meta. Every derived field must be
// empty and the cloned Meta must be nil (the omitempty contract), with no
// panic on the nil maps walked by the inner extractors.
func TestGrpoReward_GRPOSampleFromSFT_Ugly(t *testing.T) {
	got := GRPOSampleFromSFT(dataset.Sample{})
	if got.Prompt != "" || got.ReferenceAnswer != "" || got.ExpectedAnswer != "" || got.Reasoning != "" {
		t.Fatalf("sample = %+v, want all-empty derived fields", got)
	}
	if got.Meta != nil {
		t.Fatalf("meta = %v, want nil for an empty sample (omitempty contract)", got.Meta)
	}
	// Whitespace-only prompt collapses to empty after the trim, and a
	// whitespace-only response leaves the answer empty too.
	blank := GRPOSampleFromSFT(dataset.Sample{Prompt: "   \t  ", Response: "  \n  "})
	if blank.Prompt != "" || blank.ExpectedAnswer != "" {
		t.Fatalf("blank sample = %+v, want trimmed-to-empty prompt and answer", blank)
	}
}

// TestGrpoReward_ExtractGRPOExpectedAnswer_Good covers the answer-recovery
// happy paths: an explicit meta key wins over the response body, a
// case-insensitive "answer:" prefix on the final line is stripped, CRLF
// normalisation plus a blank trailing line still recovers the real answer
// line, and an empty Response falls back to Text on the single-line fast
// path.
func TestGrpoReward_ExtractGRPOExpectedAnswer_Good(t *testing.T) {
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

// TestGrpoReward_ExtractGRPOExpectedAnswer_Bad covers the empty and
// unprefixed cases: nothing to extract returns empty, a single-char line
// is shorter than every prefix so it is returned verbatim (not mistaken
// for "answer:"), and a multi-line body whose only non-blank line has no
// prefix returns that line unchanged via the backward walk.
func TestGrpoReward_ExtractGRPOExpectedAnswer_Bad(t *testing.T) {
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

// TestGrpoReward_ExtractGRPOExpectedAnswer_Ugly covers the pathological
// whitespace and emptied-tail cases: an all-whitespace body exhausts the
// backward walk and returns empty, a trailing answer-PREFIX line that
// cleans to empty inside cleanGRPOAnswerLine forces the walk to step past
// the emptied tail to the earlier real line, and an all-prefix body walks
// every boundary to empty.
func TestGrpoReward_ExtractGRPOExpectedAnswer_Ugly(t *testing.T) {
	// Every line is whitespace-only — the backward walk must exhaust all
	// boundaries and return empty rather than emitting a blank "answer".
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "   \n\t\n  "}); got != "" {
		t.Fatalf("all-blank multi-line answer = %q, want empty", got)
	}
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

// TestGrpoReward_GRPORewardContainsAnswer_Good covers the matching paths:
// a weighted match where the answer is recovered from the reasoning trace
// and appears in the rollout text, and the weight==0 normalisation that
// rewrites a zero weight to 1 so a matching rollout still scores a unit
// reward.
func TestGrpoReward_GRPORewardContainsAnswer_Good(t *testing.T) {
	t.Run("extracts_reasoning_answer_weighted", func(t *testing.T) {
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
	})
	t.Run("weight_defaults_to_one", func(t *testing.T) {
		// weight==0 normalisation: a zero weight is rewritten to 1 so a
		// matching rollout still scores a unit reward.
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
	})
}

// TestGrpoReward_GRPORewardContainsAnswer_Bad covers the non-matching and
// fallback paths: an empty expected answer yields a neutral reward with the
// explanatory detail, and a non-ASCII expected answer forces the unicode
// core.Join+Lower fallback for both a match and a miss.
func TestGrpoReward_GRPORewardContainsAnswer_Bad(t *testing.T) {
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

// TestGrpoReward_GRPORewardContainsAnswer_Ugly drives the reward against
// the awkward fragment shapes: an ASCII expected answer that only appears
// once a fragment boundary is crossed (so it must NOT match across the
// implicit "\n" join on the ASCII fast path), and an expected answer that
// itself contains a newline forcing the cross-fragment join fallback to
// find it.
func TestGrpoReward_GRPORewardContainsAnswer_Ugly(t *testing.T) {
	fn := GRPORewardContainsAnswer(1)
	// "abc" is split across Answer and Text — on the ASCII fast path each
	// fragment is scanned independently, so a value that would only appear
	// across the join must NOT match.
	split, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "abcdef"},
		Rollout: GRPORollout{Answer: "abc", Text: "def"},
	})
	if err != nil {
		t.Fatalf("split-fragment reward error = %v", err)
	}
	if split.Score != 0 || split.Detail != "missing" {
		t.Fatalf("reward = %+v, want no cross-fragment ASCII match", split)
	}
	// An expected answer carrying a newline forces the unicode join+lower
	// fallback, where the fragments ARE joined by "\n" so the value spans
	// the Answer/Text boundary and matches.
	joined, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "abc\ndef"},
		Rollout: GRPORollout{Answer: "abc", Text: "def"},
	})
	if err != nil {
		t.Fatalf("joined-fragment reward error = %v", err)
	}
	if joined.Score != 1 || joined.Detail != "matched" {
		t.Fatalf("reward = %+v, want cross-fragment join match on newline expected", joined)
	}
}

// TestGrpoReward_GRPORewardExactAnswer_Good covers the exact-match happy
// path: a trimmed, case-folded answer that equals the expected answer
// scores the configured weight with the "matched" detail.
func TestGrpoReward_GRPORewardExactAnswer_Good(t *testing.T) {
	reward, err := GRPORewardExactAnswer(3)(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: " Paris "},
		Rollout: GRPORollout{Answer: "paris"},
	})
	if err != nil {
		t.Fatalf("GRPORewardExactAnswer() error = %v", err)
	}
	if reward.Score != 3 || reward.Detail != "matched" || reward.Name != "exact_answer" {
		t.Fatalf("reward = %+v, want weighted exact match after trim+fold", reward)
	}
}

// TestGrpoReward_GRPORewardExactAnswer_Bad covers the non-matching path
// and the related metadata-error guards that an exact-answer training run
// exercises end-to-end: a mismatched answer scores zero with the default
// weight normalisation, empty checkpoint paths are rejected, invalid JSON
// fails to load, and an invalid resume sidecar aborts the run.
func TestGrpoReward_GRPORewardExactAnswer_Bad(t *testing.T) {
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

// TestGrpoReward_GRPORewardExactAnswer_Ugly covers the awkward-but-valid
// inputs: an empty expected answer never matches (even against an empty
// rollout answer), and surrounding whitespace plus mixed case on both
// sides still resolves to a match after the trim+fold normalisation.
func TestGrpoReward_GRPORewardExactAnswer_Ugly(t *testing.T) {
	fn := GRPORewardExactAnswer(1)
	// Empty expected answer must never score, even when the rollout answer
	// is also empty — the expected != "" guard blocks the empty==empty match.
	blank, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "   "},
		Rollout: GRPORollout{Answer: ""},
	})
	if err != nil {
		t.Fatalf("empty-expected exact error = %v", err)
	}
	if blank.Score != 0 || blank.Detail != "missing" {
		t.Fatalf("reward = %+v, want no match for empty expected", blank)
	}
	// Whitespace + mixed case on both sides still folds to a match.
	folded, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "  HeLLo  "},
		Rollout: GRPORollout{Answer: "\thello\n"},
	})
	if err != nil {
		t.Fatalf("folded exact error = %v", err)
	}
	if folded.Score != 1 || folded.Detail != "matched" {
		t.Fatalf("reward = %+v, want trim+case-fold match", folded)
	}
}
