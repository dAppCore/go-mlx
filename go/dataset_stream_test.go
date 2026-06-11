// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/dataset"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

func TestLoadJSONLDataset_RecognizesTrainingFormats_Good(t *testing.T) {
	input := core.Join("\n",
		`{"text":"plain corpus row"}`,
		`{"prompt":"p","response":"r"}`,
		`{"instruction":"summarise","input":"lem notes","output":"short answer"}`,
		`{"messages":[{"role":"system","content":"steady"},{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}]}`,
		`{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"there"}]}`,
		`{"problem":"2+2","thinking":"add the pair","solution":"4"}`,
	)
	ds, err := dataset.LoadJSONL(strings.NewReader(input), dataset.Config{
		ChatTemplate: chat.Config{Architecture: "qwen3"},
	})
	if err != nil {
		t.Fatalf("dataset.LoadJSONL() error = %v", err)
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

func TestFormatChatMessages_ModelTemplates_Good(t *testing.T) {
	messages := []inference.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hi"}}
	qwen := chat.Format(messages, chat.Config{Architecture: "qwen3"})
	if qwen != "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n" {
		t.Fatalf("qwen template = %q", qwen)
	}
	gemma := chat.Format(messages, chat.Config{Architecture: "gemma4_text"})
	if gemma != "<bos><|turn>system\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n" {
		t.Fatalf("gemma template = %q", gemma)
	}
	gemma3 := chat.Format(messages, chat.Config{Architecture: "gemma3_text"})
	if gemma3 != "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n" {
		t.Fatalf("gemma3 template = %q", gemma3)
	}
	llama := chat.Format([]inference.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "llama"})
	if llama != "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" {
		t.Fatalf("llama template = %q", llama)
	}
	plain := chat.Format([]inference.Message{{Role: "system"}, {Role: "user", Content: "plain"}}, chat.Config{Template: "plain", NoGenerationPrompt: true})
	if plain != "plain\n" {
		t.Fatalf("plain template = %q, want plain line", plain)
	}
}

func TestBuildDatasetBatches_PacksResponseMaskedExamples_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	}}
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildDatasetBatches(tokenizer, ds, dataset.BatchConfig{
		BatchSize:       1,
		MaxSeqLen:       8,
		SequencePacking: true,
	})
	if err != nil {
		t.Fatalf("BuildDatasetBatches() error = %v", err)
	}
	if len(batches) != 1 || len(batches[0].Batch.Tokens) != 1 {
		t.Fatalf("batches = %+v, want one packed sequence", batches)
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{1, 2, 3, 4}) {
		t.Fatalf("packed inputs = %v, want [1 2 3 4]", batches[0].Batch.Tokens[0])
	}
	if !equalIntSlices(batches[0].Targets[0], []int{2, 9, 4, 9}) {
		t.Fatalf("packed targets = %v, want [2 9 4 9]", batches[0].Targets[0])
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1, 1, 1}) {
		t.Fatalf("packed mask = %v, want all trainable", batches[0].Batch.LossMask[0])
	}
}

func TestBuildDatasetBatches_TruncatesToMaxSeqLen_Ugly(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"long prompt":   {1, 2, 3, 4},
			"long response": {5, 6, 7},
		},
		eos: 9,
	}}
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "long prompt", Response: "long response"}})

	batches, err := BuildDatasetBatches(tokenizer, ds, dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 3})
	if err != nil {
		t.Fatalf("BuildDatasetBatches() error = %v", err)
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{5, 6, 7}) {
		t.Fatalf("truncated inputs = %v, want response tail", batches[0].Batch.Tokens[0])
	}
	if !equalIntSlices(batches[0].Targets[0], []int{6, 7, 9}) {
		t.Fatalf("truncated targets = %v, want response tail + EOS", batches[0].Targets[0])
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1, 1}) {
		t.Fatalf("truncated mask = %v, want response mask retained", batches[0].Batch.LossMask[0])
	}
}

func TestLoadJSONLDataset_InvalidJSON_Bad(t *testing.T) {
	_, err := dataset.LoadJSONL(strings.NewReader("{not-json}\n"), dataset.Config{})
	if err == nil {
		t.Fatal("expected invalid JSONL error")
	}
}

func TestNewJSONLDataset_ClonesSamples_Good(t *testing.T) {
	samples := []dataset.Sample{{Text: "a", Meta: map[string]string{"k": "v"}}}
	ds := dataset.NewJSONL(samples)
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
	var ds *dataset.JSONLDataset
	if _, _, err := ds.Next(); err == nil {
		t.Fatal("expected nil Next error")
	}
	if err := ds.Reset(); err == nil {
		t.Fatal("expected nil Reset error")
	}
}

func TestJSONLDataset_SamplesReturnsCopy_Ugly(t *testing.T) {
	ds := dataset.NewJSONL([]dataset.Sample{{Text: "a", Meta: map[string]string{"format": "text"}}})
	samples := ds.Samples()
	samples[0].Text = "changed"
	samples[0].Meta["format"] = "changed"
	again := ds.Samples()
	if again[0].Text != "a" || again[0].Meta["format"] != "text" {
		t.Fatalf("Samples() aliased storage: %+v", again)
	}
}

func TestBuildDatasetBatches_NilTokenizer_Bad(t *testing.T) {
	_, err := BuildDatasetBatches(nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), dataset.BatchConfig{SequencePacking: true})
	if err == nil {
		t.Fatal("expected nil tokenizer error")
	}
}

func collectDatasetSamples(t *testing.T, ds dataset.Dataset) []dataset.Sample {
	t.Helper()
	var samples []dataset.Sample
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
