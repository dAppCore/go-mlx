// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"

	// The qwen3 template registers from the model package; the chat-shape
	// rows below render the plain fallback without it, but the survivor
	// assertions only inspect Response/Format so registration is not
	// strictly required here — kept for parity with the sibling tests.
	_ "dappco.re/go/mlx/pkg/metal/model/qwen3/chat"
)

// These cover the message-normalisation skip branches that the primary
// LoadJSONL_Ugly suite leaves untouched: a turn whose role and content
// both collapse to empty *after* NormaliseRole+Trim (distinct from the
// raw-empty short-circuit), the ShareGPT raw-empty short-circuit, and the
// nil-buffer path of claimMessageBuf reached by the documented "pass nil
// when no reuse is available" contract.

// Ugly: an OpenAI messages row carrying a whitespace-only role with
// whitespace-only content. The raw fields are non-empty so the leading
// short-circuit (Role=="" && Content=="") does not fire; NormaliseRole
// lowercases the trimmed role to "" and Trim collapses the content to "",
// so the *post-normalisation* skip drops the turn. The surviving user/
// assistant pair must still normalise to a single sample.
func TestJsonl_LoadJSONL_OpenAIPostNormaliseSkip(t *testing.T) {
	input := `{"messages":[{"role":"   ","content":"  "},{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}`
	ds, err := LoadJSONL(strings.NewReader(input), Config{ChatTemplate: chat.Config{Architecture: "qwen3"}})
	if err != nil {
		t.Fatalf("LoadJSONL(whitespace-role) error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	if len(samples) != 1 {
		t.Fatalf("whitespace-role samples = %d, want 1 (skipped turn must not yield a sample)", len(samples))
	}
	if samples[0].Response != "a" || samples[0].Format != "openai_messages" {
		t.Fatalf("whitespace-role surviving sample = %+v, want Response 'a' format openai_messages", samples[0])
	}
}

// Ugly: a ShareGPT conversations row whose leading entry is wholly empty
// (From=="" && Value==""), exercising the raw short-circuit, followed by a
// whitespace-only entry that survives the short-circuit but collapses to
// empty after NormaliseRole+Trim (the post-normalisation skip), then a
// valid human/gpt pair. Both skip branches fire; one sample survives.
func TestJsonl_LoadJSONL_ShareGPTSkips(t *testing.T) {
	input := `{"conversations":[{"from":"","value":""},{"from":"   ","value":"  "},{"from":"human","value":"hi"},{"from":"gpt","value":"there"}]}`
	ds, err := LoadJSONL(strings.NewReader(input), Config{ChatTemplate: chat.Config{Architecture: "qwen3"}})
	if err != nil {
		t.Fatalf("LoadJSONL(sharegpt-skips) error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	if len(samples) != 1 {
		t.Fatalf("sharegpt-skips samples = %d, want 1 (both empty entries must be skipped)", len(samples))
	}
	if samples[0].Response != "there" || samples[0].Format != "sharegpt" {
		t.Fatalf("sharegpt-skips surviving sample = %+v, want Response 'there' format sharegpt", samples[0])
	}
}

// Ugly: appendMessagesFromOpenAI honours the documented "pass nil when no
// reuse is available" contract — with a nil buffer it allocates a fresh
// slice via claimMessageBuf's nil branch rather than reusing a backing
// array. The LoadJSONL path always passes &msgBuf, so this contract is
// only reachable by an external/no-reuse caller. Assert the returned slice
// end-to-end (normalised roles, trimmed content, empties dropped) rather
// than poking the private helper in isolation.
func TestJsonl_AppendMessagesFromOpenAI_NilBuffer(t *testing.T) {
	records := []messageRecord{
		{Role: "system", Content: "steady"},
		{Role: "", Content: ""},          // raw short-circuit skip
		{Role: "human", Content: " hi "}, // human -> user, content trimmed
		{Role: "gpt", Content: "yo"},     // gpt -> assistant
	}
	out := appendMessagesFromOpenAI(nil, records)
	want := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "yo"},
	}
	if len(out) != len(want) {
		t.Fatalf("appendMessagesFromOpenAI(nil) len = %d, want %d: %+v", len(out), len(want), out)
	}
	// inference.Message carries an Images [][]byte field (not comparable
	// with ==); this path never sets it, so compare the scalar fields.
	for i := range want {
		if out[i].Role != want[i].Role || out[i].Content != want[i].Content {
			t.Fatalf("appendMessagesFromOpenAI(nil)[%d] = %+v, want %+v", i, out[i], want[i])
		}
	}
}
