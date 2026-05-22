// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"bytes"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

const scannerMaxBytes = 16 * 1024 * 1024

// Sentinel errors hoisted from the nil-guard call sites so they
// allocate exactly once at package init instead of one *Err per
// nil-receiver call. These are cold paths but the package contract
// is the same either way, and resultError's "core result failed"
// fallback fires whenever a non-error Value is wrapped in a failed
// Result.
var (
	errReaderNil        = core.NewError("dataset: reader is nil")
	errJSONLDatasetNil  = core.NewError("dataset: JSONL dataset is nil")
	errCoreResultFailed = core.NewError("core result failed")
)

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
		return nil, errReaderNil
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), scannerMaxBytes)

	var samples []Sample
	// Hoist the record buffer out of the loop. The original `var
	// record jsonRecord` inside the loop escaped to the heap on every
	// iteration (json.Unmarshal takes the pointer reflectively). Once
	// hoisted, json.Unmarshal still ignores keys that are absent in
	// the current row, so the previous row's string fields would
	// carry over — zero the struct via assignment to a zero literal
	// before each Unmarshal call. The slice fields (Messages,
	// Conversations) are reset to length 0 in-place so we keep the
	// backing array across rows of the same shape and avoid an
	// allocation per chat-shape row. msgBuf reuses the
	// []inference.Message backing across openai/sharegpt rows —
	// chat.Format consumes its argument synchronously so reuse is
	// safe.
	var record jsonRecord
	var msgBuf []inference.Message
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		// scanner.Bytes() aliases the scanner's internal buffer (no
		// allocation), bytes.TrimSpace returns a sub-slice, and
		// core.JSONUnmarshal eats []byte directly. The prior
		// scanner.Text() path allocated a fresh string per row —
		// shaving 1 alloc/row over 100k-row corpora is load-bearing.
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		messagesBuf := record.Messages[:0]
		conversationsBuf := record.Conversations[:0]
		record = jsonRecord{Messages: messagesBuf, Conversations: conversationsBuf}
		if result := core.JSONUnmarshal(line, &record); !result.OK {
			return nil, core.Errorf("dataset: parse JSONL line %d: %w", lineNo, resultError(result))
		}
		sample, ok, err := record.toSample(cfg, &msgBuf)
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
		return Sample{}, false, errJSONLDatasetNil
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
		return errJSONLDatasetNil
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

// toSample normalises a parsed jsonRecord. msgBuf is an optional
// pointer to a reusable []inference.Message backing array for the
// openai/sharegpt branches — pass nil when no reuse is available.
// The helpers write back through *msgBuf so a grown backing array
// is captured for the next row, saving one alloc per chat-shape row
// over the lifetime of a LoadJSONL call. chat.Format does not retain
// its messages argument, so the caller can safely reuse the buffer.
func (r jsonRecord) toSample(cfg Config, msgBuf *[]inference.Message) (Sample, bool, error) {
	if text := core.Trim(r.Text); text != "" {
		return labelled(Sample{Text: text}, "text"), true, nil
	}
	if len(r.Messages) > 0 {
		return MessagesToSample(appendMessagesFromOpenAI(msgBuf, r.Messages), cfg.ChatTemplate, "openai_messages")
	}
	if len(r.Conversations) > 0 {
		return MessagesToSample(appendMessagesFromShareGPT(msgBuf, r.Conversations), cfg.ChatTemplate, "sharegpt")
	}
	// Trim each candidate once per row — these used to be called 4-6
	// times each because firstNonEmpty pre-trimmed for the check then
	// returned an untrimmed value the caller trimmed again, and the
	// outer guard re-trimmed for the empty check. The prompt-response
	// and reasoning branches additionally recomputed firstNonEmpty
	// inside the labelled Sample literal — split into prompt-present
	// and response-only sub-cases so each call site touches its inputs
	// exactly once. Branch order matches frequency: prompt-response,
	// alpaca, reasoning.
	if prompt := core.Trim(r.Prompt); prompt != "" {
		return labelled(Sample{
			Prompt:   prompt,
			Response: firstNonEmpty(r.Response, r.Completion),
		}, "prompt_response"), true, nil
	}
	if response := firstNonEmpty(r.Response, r.Completion); response != "" {
		return labelled(Sample{
			Response: response,
		}, "prompt_response"), true, nil
	}
	if output := core.Trim(r.Output); core.Trim(r.Instruction) != "" || output != "" {
		return labelled(Sample{
			Prompt:   formatInstructionPrompt(r.Instruction, r.Input),
			Response: output,
		}, "alpaca"), true, nil
	}
	if problem := firstNonEmpty(r.Problem, r.Question); problem != "" {
		return labelled(Sample{
			Prompt:   problem,
			Response: formatReasoningResponse(firstNonEmpty(r.Thinking, r.Reasoning), firstNonEmpty(r.Solution, r.Answer)),
		}, "reasoning"), true, nil
	}
	if solution := firstNonEmpty(r.Solution, r.Answer); solution != "" {
		return labelled(Sample{
			Response: formatReasoningResponse(firstNonEmpty(r.Thinking, r.Reasoning), solution),
		}, "reasoning"), true, nil
	}
	return Sample{}, false, nil
}

// appendMessagesFromOpenAI fills *buf with normalised messages from
// records, writing back through buf so a grown backing array is
// captured for the next call. When buf is nil (no reuse available)
// the slice is allocated fresh; otherwise we reset the existing
// backing in place if cap is sufficient. Pass a reusable buffer
// (typical: one per LoadJSONL call) to avoid the per-row slice alloc
// the original `make([]Message, 0, n)` form triggered.
func appendMessagesFromOpenAI(buf *[]inference.Message, records []messageRecord) []inference.Message {
	out := claimMessageBuf(buf, len(records))
	for _, record := range records {
		// Short-circuit empty rows before the Trim/NormaliseRole
		// work — JSON unmarshal leaves missing fields as "" so
		// this is a hot skip for sparse messages.
		if record.Role == "" && record.Content == "" {
			continue
		}
		role := chat.NormaliseRole(record.Role)
		content := core.Trim(record.Content)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	if buf != nil {
		*buf = out
	}
	return out
}

// appendMessagesFromShareGPT mirrors appendMessagesFromOpenAI for the
// ShareGPT-shape record (from/value rather than role/content).
func appendMessagesFromShareGPT(buf *[]inference.Message, records []shareGPTRecord) []inference.Message {
	out := claimMessageBuf(buf, len(records))
	for _, record := range records {
		if record.From == "" && record.Value == "" {
			continue
		}
		role := chat.NormaliseRole(record.From)
		content := core.Trim(record.Value)
		if role == "" && content == "" {
			continue
		}
		out = append(out, inference.Message{Role: role, Content: content})
	}
	if buf != nil {
		*buf = out
	}
	return out
}

// claimMessageBuf returns an empty slice with at least n capacity,
// reusing *buf's backing array when possible. Hoisted from the two
// append helpers since the prelude is identical.
func claimMessageBuf(buf *[]inference.Message, n int) []inference.Message {
	if buf == nil {
		return make([]inference.Message, 0, n)
	}
	if cap(*buf) < n {
		return make([]inference.Message, 0, n)
	}
	return (*buf)[:0]
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
	// chat.Format only reads from its slice argument (verified: all
	// per-template formatters iterate with `for _, msg := range
	// messages` without retaining), and the resulting Prompt is an
	// immutable string baked into the returned Sample. The defensive
	// cloneMessages copy was protecting nothing — drop it and pass
	// the sub-slice directly.
	response := core.Trim(messages[assistantIdx].Content)
	prompt := chat.Format(messages[:assistantIdx], cfg)
	return labelled(Sample{Prompt: prompt, Response: response}, format), true, nil
}

func labelled(sample Sample, format string) Sample {
	// Fast path — toSample always hands a Sample with nil Meta to
	// labelled, so the clone path returns nil. Pre-size the fresh
	// map to one entry to skip the runtime growth step the
	// untyped map literal would trigger.
	if len(sample.Meta) == 0 {
		sample.Meta = make(map[string]string, 1)
	} else {
		sample.Meta = cloneStringMap(sample.Meta)
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

// firstNonEmpty returns the first value with a non-empty trimmed form,
// already trimmed. Callers were universally trimming the result a
// second time before use; returning the trimmed value eliminates the
// duplicate Trim per row.
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := core.Trim(value); trimmed != "" {
			return trimmed
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
	return errCoreResultFailed
}
