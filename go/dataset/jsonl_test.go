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
