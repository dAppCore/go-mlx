// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

const scannerMaxBytes = 16 * 1024 * 1024

// Config controls JSONL ingestion and chat sample normalization.
type Config struct {
	ChatTemplate chat.Config
}

// BatchConfig controls tokenizer batching for training/eval streams.
type BatchConfig struct {
	BatchSize       int
	MaxSeqLen       int
	SequencePacking bool
	NoEOS           bool
}

// JSONLDataset is a replayable in-memory dataset loaded from JSONL records.
type JSONLDataset struct {
	samples []Sample
	index   int
}

type jsonRecord struct {
	Text          string           `json:"text"`
	Prompt        string           `json:"prompt"`
	Response      string           `json:"response"`
	Completion    string           `json:"completion"`
	Instruction   string           `json:"instruction"`
	Input         string           `json:"input"`
	Output        string           `json:"output"`
	Problem       string           `json:"problem"`
	Question      string           `json:"question"`
	Thinking      string           `json:"thinking"`
	Reasoning     string           `json:"reasoning"`
	Solution      string           `json:"solution"`
	Answer        string           `json:"answer"`
	Messages      []messageRecord  `json:"messages"`
	Conversations []shareGPTRecord `json:"conversations"`
}

type messageRecord struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type shareGPTRecord struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

// LoadJSONL reads JSONL into a replayable Dataset.
//
//	d, err := dataset.LoadJSONL(reader, dataset.Config{})
func LoadJSONL(reader io.Reader, cfg Config) (*JSONLDataset, error) {
	if reader == nil {
		return nil, core.NewError("dataset: reader is nil")
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), scannerMaxBytes)

	var samples []Sample
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := core.Trim(scanner.Text())
		if line == "" {
			continue
		}
		var record jsonRecord
		if result := core.JSONUnmarshalString(line, &record); !result.OK {
			return nil, core.Errorf("dataset: parse JSONL line %d: %w", lineNo, resultError(result))
		}
		sample, ok, err := record.toSample(cfg)
		if err != nil {
			return nil, core.Errorf("dataset: normalize JSONL line %d: %w", lineNo, err)
		}
		if ok {
			samples = append(samples, sample)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, core.Errorf("dataset: read JSONL: %w", err)
	}
	// samples was built locally — every entry's Meta map was
	// constructed fresh by labelled(). The slice is owned by the
	// dataset, so the defensive CloneSamples pass here is pure
	// duplication. Hand off the freshly built slice directly.
	return &JSONLDataset{samples: samples}, nil
}

// NewJSONL returns a replayable dataset from already-normalized samples.
//
//	d := dataset.NewJSONL(samples)
func NewJSONL(samples []Sample) *JSONLDataset {
	return &JSONLDataset{samples: CloneSamples(samples)}
}

// Next returns the next normalized sample.
func (d *JSONLDataset) Next() (Sample, bool, error) {
	if d == nil {
		return Sample{}, false, core.NewError("dataset: JSONL dataset is nil")
	}
	if d.index >= len(d.samples) {
		return Sample{}, false, nil
	}
	sample := CloneSample(d.samples[d.index])
	d.index++
	return sample, true, nil
}

// Reset rewinds the replayable dataset.
func (d *JSONLDataset) Reset() error {
	if d == nil {
		return core.NewError("dataset: JSONL dataset is nil")
	}
	d.index = 0
	return nil
}

// Samples returns a defensive copy of all normalized samples.
//
//	samples := d.Samples()
func (d *JSONLDataset) Samples() []Sample {
	if d == nil {
		return nil
	}
	return CloneSamples(d.samples)
}

func (r jsonRecord) toSample(cfg Config) (Sample, bool, error) {
	if text := core.Trim(r.Text); text != "" {
		return labelled(Sample{Text: text}, "text"), true, nil
	}
	if len(r.Messages) > 0 {
		return MessagesToSample(messagesFromOpenAI(r.Messages), cfg.ChatTemplate, "openai_messages")
	}
	if len(r.Conversations) > 0 {
		return MessagesToSample(messagesFromShareGPT(r.Conversations), cfg.ChatTemplate, "sharegpt")
	}
	if core.Trim(r.Prompt) != "" || core.Trim(firstNonEmpty(r.Response, r.Completion)) != "" {
		return labelled(Sample{
			Prompt:   core.Trim(r.Prompt),
			Response: core.Trim(firstNonEmpty(r.Response, r.Completion)),
		}, "prompt_response"), true, nil
	}
	if core.Trim(r.Instruction) != "" || core.Trim(r.Output) != "" {
		return labelled(Sample{
			Prompt:   formatInstructionPrompt(r.Instruction, r.Input),
			Response: core.Trim(r.Output),
		}, "alpaca"), true, nil
	}
	if core.Trim(firstNonEmpty(r.Problem, r.Question)) != "" || core.Trim(firstNonEmpty(r.Solution, r.Answer)) != "" {
		return labelled(Sample{
			Prompt:   core.Trim(firstNonEmpty(r.Problem, r.Question)),
			Response: formatReasoningResponse(firstNonEmpty(r.Thinking, r.Reasoning), firstNonEmpty(r.Solution, r.Answer)),
		}, "reasoning"), true, nil
	}
	return Sample{}, false, nil
}

func messagesFromOpenAI(records []messageRecord) []inference.Message {
	out := make([]inference.Message, 0, len(records))
	for _, record := range records {
		role := chat.NormaliseRole(record.Role)
		content := core.Trim(record.Content)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	return out
}

func messagesFromShareGPT(records []shareGPTRecord) []inference.Message {
	out := make([]inference.Message, 0, len(records))
	for _, record := range records {
		role := chat.NormaliseRole(record.From)
		content := core.Trim(record.Value)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	return out
}

// MessagesToSample converts a message list into a normalised Sample,
// using the assistant's last message as the response (if any).
//
//	sample, ok, err := dataset.MessagesToSample(messages, cfg, "sharegpt")
func MessagesToSample(messages []inference.Message, cfg chat.Config, format string) (Sample, bool, error) {
	if len(messages) == 0 {
		return Sample{}, false, nil
	}
	assistantIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if chat.NormaliseRole(messages[i].Role) == "assistant" {
			assistantIdx = i
			break
		}
	}
	if assistantIdx < 0 {
		text := chat.Format(messages, chat.Config{
			Architecture:       cfg.Architecture,
			Template:           cfg.Template,
			NoGenerationPrompt: true,
		})
		return labelled(Sample{Text: text}, format), true, nil
	}
	promptMessages := cloneMessages(messages[:assistantIdx])
	response := core.Trim(messages[assistantIdx].Content)
	prompt := chat.Format(promptMessages, cfg)
	return labelled(Sample{Prompt: prompt, Response: response}, format), true, nil
}

func labelled(sample Sample, format string) Sample {
	sample.Meta = cloneStringMap(sample.Meta)
	if sample.Meta == nil {
		sample.Meta = map[string]string{}
	}
	sample.Meta["format"] = format
	return sample
}

func formatInstructionPrompt(instruction, input string) string {
	instruction = core.Trim(instruction)
	input = core.Trim(input)
	if instruction == "" {
		return input
	}
	if input == "" {
		return instruction
	}
	return instruction + "\n\n" + input
}

func formatReasoningResponse(thinking, solution string) string {
	thinking = core.Trim(thinking)
	solution = core.Trim(solution)
	if thinking == "" {
		return solution
	}
	if solution == "" {
		return thinking
	}
	return thinking + "\n\n" + solution
}

func cloneMessages(messages []inference.Message) []inference.Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]inference.Message, len(messages))
	copy(out, messages)
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
