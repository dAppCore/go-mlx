// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"

	// The qwen3 template registers from the model package (family
	// formatters live beside their families); without it LoadJSONL
	// renders the plain fallback and the prompt assertions fail.
	_ "dappco.re/go/mlx/pkg/metal/model/qwen3/chat"
	"strings"
)

func TestMessagesToSample_Gemma4SPORUsesSharedChatFormatter_Good(t *testing.T) {
	messages := []inference.Message{
		{Role: "system", Content: " be exact "},
		{Role: "user", Content: "Write one line."},
		{Role: "assistant", Content: " one line "},
	}
	cfg := chat.Config{Architecture: "gemma4_text", EnableThinking: true}

	sample, ok, err := MessagesToSample(messages, cfg, "openai_messages")
	if err != nil {
		t.Fatalf("MessagesToSample() error = %v", err)
	}
	if !ok {
		t.Fatal("MessagesToSample() ok = false, want sample")
	}

	wantPrompt := chat.Format(messages[:2], cfg)
	if sample.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want shared chat.Format prompt %q", sample.Prompt, wantPrompt)
	}
	if sample.Response != "one line" {
		t.Fatalf("Response = %q, want trimmed assistant response", sample.Response)
	}
	if sample.Format != "openai_messages" {
		t.Fatalf("format = %q, want openai_messages", sample.Format)
	}
}

// --- merged from the root dataset_stream_test.go (orphan sweep: these
// exercise the dataset package JSONL surface directly) ---
func TestLoadJSONLDataset_RecognizesTrainingFormats_Good(t *testing.T) {
	input := core.Join("\n",
		`{"text":"plain corpus row"}`,
		`{"prompt":"p","response":"r"}`,
		`{"instruction":"summarise","input":"lem notes","output":"short answer"}`,
		`{"messages":[{"role":"system","content":"steady"},{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}]}`,
		`{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"there"}]}`,
		`{"problem":"2+2","thinking":"add the pair","solution":"4"}`,
	)
	ds, err := LoadJSONL(strings.NewReader(input), Config{
		ChatTemplate: chat.Config{Architecture: "qwen3"},
	})
	if err != nil {
		t.Fatalf("LoadJSONL() error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	if len(samples) != 6 {
		t.Fatalf("samples len = %d, want 6", len(samples))
	}
	if samples[0].Text != "plain corpus row" || samples[0].Format != "text" {
		t.Fatalf("text sample = %+v", samples[0])
	}
	if samples[1].Prompt != "p" || samples[1].Response != "r" || samples[1].Format != "prompt_response" {
		t.Fatalf("prompt/response sample = %+v", samples[1])
	}
	if !core.Contains(samples[2].Prompt, "summarise") || !core.Contains(samples[2].Prompt, "lem notes") || samples[2].Response != "short answer" || samples[2].Format != "alpaca" {
		t.Fatalf("alpaca sample = %+v", samples[2])
	}
	if !core.Contains(samples[3].Prompt, "<|im_start|>system\nsteady<|im_end|>") ||
		!core.Contains(samples[3].Prompt, "<|im_start|>assistant\n") ||
		core.Contains(samples[3].Prompt, "pong") ||
		samples[3].Response != "pong" ||
		samples[3].Format != "openai_messages" {
		t.Fatalf("openai messages sample = %+v", samples[3])
	}
	if !core.Contains(samples[4].Prompt, "<|im_start|>user\nhi<|im_end|>") || samples[4].Response != "there" || samples[4].Format != "sharegpt" {
		t.Fatalf("sharegpt sample = %+v", samples[4])
	}
	if samples[5].Prompt != "2+2" || !core.Contains(samples[5].Response, "add the pair") || !core.Contains(samples[5].Response, "4") || samples[5].Format != "reasoning" {
		t.Fatalf("reasoning sample = %+v", samples[5])
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() after Reset error = %v", err)
	}
	if !ok || again.Text != "plain corpus row" {
		t.Fatalf("Next() after Reset = %+v ok=%v", again, ok)
	}
}

func TestLoadJSONLDataset_InvalidJSON_Bad(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader("{not-json}\n"), Config{})
	if err == nil {
		t.Fatal("expected invalid JSONL error")
	}
}

func TestNewJSONLDataset_ClonesSamples_Good(t *testing.T) {
	samples := []Sample{{Text: "a", Meta: map[string]string{"k": "v"}}}
	ds := NewJSONL(samples)
	samples[0].Text = "mutated"
	samples[0].Meta["k"] = "changed"

	got, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || got.Text != "a" || got.Meta["k"] != "v" {
		t.Fatalf("Next() = %+v ok=%v, want cloned original", got, ok)
	}
}

func TestJSONLDataset_NilReceiver_Bad(t *testing.T) {
	var ds *JSONLDataset
	if _, _, err := ds.Next(); err == nil {
		t.Fatal("expected nil Next error")
	}
	if err := ds.Reset(); err == nil {
		t.Fatal("expected nil Reset error")
	}
}

func TestJSONLDataset_SamplesReturnsCopy_Ugly(t *testing.T) {
	ds := NewJSONL([]Sample{{Text: "a", Meta: map[string]string{"format": "text"}}})
	samples := ds.Samples()
	samples[0].Text = "changed"
	samples[0].Meta["format"] = "changed"
	again := ds.Samples()
	if again[0].Text != "a" || again[0].Meta["format"] != "text" {
		t.Fatalf("Samples() aliased storage: %+v", again)
	}
}

// LoadJSONL rejects a nil reader with the hoisted sentinel before
// touching the decoder.
func TestLoadJSONL_NilReader_Bad(t *testing.T) {
	if _, err := LoadJSONL(nil, Config{}); err == nil {
		t.Fatal("LoadJSONL(nil) expected error, got nil")
	}
}

// Empty input (and whitespace-only lines, which the decoder skips) yields
// an empty-but-valid dataset, not an error.
func TestLoadJSONL_EmptyInput_Ugly(t *testing.T) {
	ds, err := LoadJSONL(strings.NewReader("\n  \n\n"), Config{})
	if err != nil {
		t.Fatalf("LoadJSONL(blank) error = %v", err)
	}
	if got := collectDatasetSamples(t, ds); len(got) != 0 {
		t.Fatalf("LoadJSONL(blank) samples = %d, want 0", len(got))
	}
}

// Rows that drop into the prompt-only, response-only, alpaca half-row and
// reasoning half-row branches — these sweep firstNonEmpty,
// formatInstructionPrompt and formatReasoningResponse through their empty
// sub-cases. Skipped (unrecognised) rows must not produce samples.
func TestLoadJSONL_PartialShapes_Ugly(t *testing.T) {
	input := core.Join("\n",
		`{"response":"bare completion"}`,             // response-only -> prompt_response
		`{"completion":"via completion key"}`,        // completion alias
		`{"instruction":"do the thing"}`,             // alpaca, no input/output
		`{"input":"only input text","output":"out"}`, // alpaca, instruction empty
		`{"thinking":"reason only"}`,                 // not a recognised shape (no solution) -> skipped
		`{"solution":"42"}`,                          // reasoning, solution-only, no thinking
		`{"problem":"why","thinking":"because"}`,     // reasoning, problem + thinking, no solution
		`{"unknown":"field"}`,                        // wholly unrecognised -> skipped
	)
	ds, err := LoadJSONL(strings.NewReader(input), Config{})
	if err != nil {
		t.Fatalf("LoadJSONL(partial) error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	// 6 recognised rows; the thinking-only and unknown rows are dropped.
	if len(samples) != 6 {
		t.Fatalf("partial-shape samples = %d, want 6: %+v", len(samples), samples)
	}
	if samples[0].Response != "bare completion" || samples[0].Format != "prompt_response" {
		t.Fatalf("response-only sample = %+v", samples[0])
	}
	if samples[1].Response != "via completion key" {
		t.Fatalf("completion-alias sample = %+v", samples[1])
	}
	// instruction-only: prompt is the instruction, no input appended.
	if samples[2].Prompt != "do the thing" || samples[2].Format != "alpaca" {
		t.Fatalf("instruction-only sample = %+v", samples[2])
	}
	// input-only (empty instruction): prompt is just the input.
	if samples[3].Prompt != "only input text" || samples[3].Response != "out" {
		t.Fatalf("input-only alpaca sample = %+v", samples[3])
	}
	// solution-only reasoning: response is the bare solution (no thinking prefix).
	if samples[4].Response != "42" || samples[4].Format != "reasoning" {
		t.Fatalf("solution-only reasoning sample = %+v", samples[4])
	}
	// problem + thinking, no solution: response is the thinking text alone.
	if samples[5].Prompt != "why" || samples[5].Response != "because" || samples[5].Format != "reasoning" {
		t.Fatalf("problem+thinking reasoning sample = %+v", samples[5])
	}
}

// A messages row containing an empty message object exercises the
// empty-skip short-circuit in appendMessagesFromOpenAI; the surviving
// user+assistant turns still normalise.
func TestLoadJSONL_OpenAIEmptyMessageSkipped_Ugly(t *testing.T) {
	input := `{"messages":[{"role":"","content":""},{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}`
	ds, err := LoadJSONL(strings.NewReader(input), Config{ChatTemplate: chat.Config{Architecture: "qwen3"}})
	if err != nil {
		t.Fatalf("LoadJSONL(empty-msg) error = %v", err)
	}
	samples := collectDatasetSamples(t, ds)
	if len(samples) != 1 {
		t.Fatalf("empty-msg samples = %d, want 1", len(samples))
	}
	if samples[0].Response != "a" || samples[0].Format != "openai_messages" {
		t.Fatalf("empty-msg sample = %+v", samples[0])
	}
}

// MessagesToSample with no messages returns (zero, false, nil) — the
// empty-slice guard.
func TestMessagesToSample_Empty_Bad(t *testing.T) {
	if _, ok, err := MessagesToSample(nil, chat.Config{}, "openai_messages"); ok || err != nil {
		t.Fatalf("MessagesToSample(nil) = ok %v err %v, want false,nil", ok, err)
	}
}

// A conversation that ends without an assistant turn drops into the
// no-assistant text-fallback branch (NoGenerationPrompt) and returns a
// Text sample rather than a Prompt/Response pair.
func TestMessagesToSample_NoAssistantTextFallback_Ugly(t *testing.T) {
	messages := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "ping"},
	}
	cfg := chat.Config{Architecture: "qwen3"}

	sample, ok, err := MessagesToSample(messages, cfg, "openai_messages")
	if err != nil {
		t.Fatalf("MessagesToSample() error = %v", err)
	}
	if !ok {
		t.Fatal("MessagesToSample() ok = false, want true")
	}
	if sample.Response != "" || sample.Prompt != "" {
		t.Fatalf("no-assistant sample should be Text-only, got Prompt=%q Response=%q", sample.Prompt, sample.Response)
	}
	if sample.Text == "" {
		t.Fatal("no-assistant sample Text is empty, want formatted transcript")
	}
	// The fallback suppresses the generation prompt, so the rendered text
	// must not open an empty assistant turn for the model to continue.
	wantText := chat.Format(messages, chat.Config{Architecture: "qwen3", NoGenerationPrompt: true})
	if sample.Text != wantText {
		t.Fatalf("Text = %q, want NoGenerationPrompt render %q", sample.Text, wantText)
	}
	if sample.Format != "openai_messages" {
		t.Fatalf("format = %q, want openai_messages", sample.Format)
	}
}

// Samples() on a nil receiver returns nil rather than panicking — the
// 66.7% gap.
func TestJSONLDataset_SamplesNilReceiver_Bad(t *testing.T) {
	var ds *JSONLDataset
	if got := ds.Samples(); got != nil {
		t.Fatalf("nil JSONLDataset.Samples() = %v, want nil", got)
	}
}

func collectDatasetSamples(t *testing.T, ds Dataset) []Sample {
	t.Helper()
	var samples []Sample
	for {
		sample, ok, err := ds.Next()
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		if !ok {
			return samples
		}
		samples = append(samples, sample)
	}
}
