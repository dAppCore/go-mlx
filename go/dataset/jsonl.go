// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"encoding/json"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

// Sentinel errors hoisted from the nil-guard call sites so they
// allocate exactly once at package init instead of one *Err per
// nil-receiver call. These are cold paths but the package contract
// is the same either way.
var (
	errReaderNil       = core.NewError("dataset: reader is nil")
	errJSONLDatasetNil = core.NewError("dataset: JSONL dataset is nil")
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
	// One streaming decoder for the whole file — json.Unmarshal would
	// allocate a fresh decodeState (~5 allocs per call) per row,
	// whereas Decoder reuses its internal scratch buffers across
	// Decode() calls. Decoder handles inter-record whitespace
	// (including empty lines) on its own.
	dec := json.NewDecoder(bufio.NewReaderSize(reader, 64*1024))

	// Pre-size the samples buffer — corpora of any meaningful size
	// run through several growslice rounds otherwise (nil → 1 → 2 →
	// 4 → 8 → ... ). Starting at 64 covers the first ~6 doublings
	// and is small enough to be no waste on tiny inputs. Larger
	// corpora still grow naturally past this initial capacity.
	samples := make([]Sample, 0, 64)
	// Hoist the record buffer out of the loop. The original `var
	// record jsonRecord` inside the loop escaped to the heap on every
	// iteration (json.Decode takes the pointer reflectively). Once
	// hoisted, json.Decode still ignores keys that are absent in
	// the current row, so the previous row's string fields would
	// carry over — zero each string field by hand before each
	// Decode call (per-field assignment skips the struct-literal
	// memclr the compiler emits for `record = jsonRecord{...}`,
	// saving ~2 ns/row in the steady-state loop). The slice fields
	// (Messages, Conversations) are reset to length 0 in-place so we
	// keep the backing array across rows of the same shape and avoid
	// an allocation per chat-shape row. msgBuf reuses the
	// []inference.Message backing across openai/sharegpt rows —
	// chat.Format consumes its argument synchronously so reuse is
	// safe.
	var record jsonRecord
	var msgBuf []inference.Message
	// recordNo numbers non-empty input records — empty/whitespace-only
	// lines do not bump it. Error messages name "record N" for that
	// reason, matching what the original "line N" form meant since the
	// prior scanner loop incremented for every line but skipped empty
	// ones before decoding.
	recordNo := 0
	for dec.More() {
		recordNo++
		// Per-field zero — see hoisted-record comment above. Order
		// matches struct declaration so the compiler can fold
		// consecutive stores into a single SIMD memstore on arm64.
		record.Text = ""
		record.Prompt = ""
		record.Response = ""
		record.Completion = ""
		record.Instruction = ""
		record.Input = ""
		record.Output = ""
		record.Problem = ""
		record.Question = ""
		record.Thinking = ""
		record.Reasoning = ""
		record.Solution = ""
		record.Answer = ""
		record.Messages = record.Messages[:0]
		record.Conversations = record.Conversations[:0]
		if err := dec.Decode(&record); err != nil {
			return nil, core.Errorf("dataset: parse JSONL record %d: %w", recordNo, err)
		}
		sample, ok, err := record.toSample(cfg, &msgBuf)
		if err != nil {
			return nil, core.Errorf("dataset: normalize JSONL record %d: %w", recordNo, err)
		}
		if ok {
			samples = append(samples, sample)
		}
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
		// Copy + tweak the supplied config rather than rebuilding from
		// fields. The literal form duplicates the field list (drift risk
		// when chat.Config gains a field) and forces the compiler to
		// re-emit each field store; the copy is a single 24-byte stack
		// move on arm64 (chat.Config is two strings + bool padded).
		noPromptCfg := cfg
		noPromptCfg.NoGenerationPrompt = true
		text := chat.Format(messages, noPromptCfg)
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

