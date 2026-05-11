// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"bufio"
	"io"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

const datasetScannerMaxBytes = 16 * 1024 * 1024

// DatasetConfig controls JSONL ingestion and chat sample normalization.
type DatasetConfig struct {
	ChatTemplate ChatTemplateConfig
}

// ChatTemplateConfig selects the native chat template used for message
// datasets. Aliased from dappco.re/go/mlx/chat/.
type ChatTemplateConfig = chat.Config

// DatasetBatchConfig controls tokenizer batching for training/eval streams.
type DatasetBatchConfig struct {
	BatchSize       int
	MaxSeqLen       int
	SequencePacking bool
	NoEOS           bool
}

// JSONLDataset is a replayable in-memory dataset loaded from JSONL records.
type JSONLDataset struct {
	samples []SFTSample
	index   int
}

type datasetJSONRecord struct {
	Text          string                  `json:"text"`
	Prompt        string                  `json:"prompt"`
	Response      string                  `json:"response"`
	Completion    string                  `json:"completion"`
	Instruction   string                  `json:"instruction"`
	Input         string                  `json:"input"`
	Output        string                  `json:"output"`
	Problem       string                  `json:"problem"`
	Question      string                  `json:"question"`
	Thinking      string                  `json:"thinking"`
	Reasoning     string                  `json:"reasoning"`
	Solution      string                  `json:"solution"`
	Answer        string                  `json:"answer"`
	Messages      []datasetMessageRecord  `json:"messages"`
	Conversations []datasetShareGPTRecord `json:"conversations"`
}

type datasetMessageRecord struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type datasetShareGPTRecord struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

// LoadJSONLDataset reads JSONL into a replayable SFTDataset.
func LoadJSONLDataset(reader io.Reader, cfg DatasetConfig) (*JSONLDataset, error) {
	if reader == nil {
		return nil, core.NewError("mlx: dataset reader is nil")
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), datasetScannerMaxBytes)

	var samples []SFTSample
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := core.Trim(scanner.Text())
		if line == "" {
			continue
		}
		var record datasetJSONRecord
		if result := core.JSONUnmarshalString(line, &record); !result.OK {
			return nil, core.Errorf("mlx: parse JSONL line %d: %w", lineNo, datasetResultError(result))
		}
		sample, ok, err := record.toSFTSample(cfg)
		if err != nil {
			return nil, core.Errorf("mlx: normalize JSONL line %d: %w", lineNo, err)
		}
		if ok {
			samples = append(samples, sample)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, core.Errorf("mlx: read JSONL dataset: %w", err)
	}
	return &JSONLDataset{samples: cloneSFTSamples(samples)}, nil
}

// NewJSONLDataset returns a replayable dataset from already-normalized samples.
func NewJSONLDataset(samples []SFTSample) *JSONLDataset {
	return &JSONLDataset{samples: cloneSFTSamples(samples)}
}

// Next returns the next normalized sample.
func (d *JSONLDataset) Next() (SFTSample, bool, error) {
	if d == nil {
		return SFTSample{}, false, core.NewError("mlx: JSONL dataset is nil")
	}
	if d.index >= len(d.samples) {
		return SFTSample{}, false, nil
	}
	sample := cloneSFTSample(d.samples[d.index])
	d.index++
	return sample, true, nil
}

// Reset rewinds the replayable dataset.
func (d *JSONLDataset) Reset() error {
	if d == nil {
		return core.NewError("mlx: JSONL dataset is nil")
	}
	d.index = 0
	return nil
}

// Samples returns a defensive copy of all normalized samples.
func (d *JSONLDataset) Samples() []SFTSample {
	if d == nil {
		return nil
	}
	return cloneSFTSamples(d.samples)
}

func (r datasetJSONRecord) toSFTSample(cfg DatasetConfig) (SFTSample, bool, error) {
	if text := core.Trim(r.Text); text != "" {
		return datasetSample(SFTSample{Text: text}, "text"), true, nil
	}
	if len(r.Messages) > 0 {
		return messagesToSFTSample(datasetMessages(r.Messages), cfg.ChatTemplate, "openai_messages")
	}
	if len(r.Conversations) > 0 {
		return messagesToSFTSample(datasetShareGPTMessages(r.Conversations), cfg.ChatTemplate, "sharegpt")
	}
	if core.Trim(r.Prompt) != "" || core.Trim(firstNonEmpty(r.Response, r.Completion)) != "" {
		return datasetSample(SFTSample{
			Prompt:   core.Trim(r.Prompt),
			Response: core.Trim(firstNonEmpty(r.Response, r.Completion)),
		}, "prompt_response"), true, nil
	}
	if core.Trim(r.Instruction) != "" || core.Trim(r.Output) != "" {
		return datasetSample(SFTSample{
			Prompt:   formatInstructionPrompt(r.Instruction, r.Input),
			Response: core.Trim(r.Output),
		}, "alpaca"), true, nil
	}
	if core.Trim(firstNonEmpty(r.Problem, r.Question)) != "" || core.Trim(firstNonEmpty(r.Solution, r.Answer)) != "" {
		return datasetSample(SFTSample{
			Prompt:   core.Trim(firstNonEmpty(r.Problem, r.Question)),
			Response: formatReasoningResponse(firstNonEmpty(r.Thinking, r.Reasoning), firstNonEmpty(r.Solution, r.Answer)),
		}, "reasoning"), true, nil
	}
	return SFTSample{}, false, nil
}

func datasetMessages(records []datasetMessageRecord) []Message {
	out := make([]Message, 0, len(records))
	for _, record := range records {
		role := normalizeDatasetRole(record.Role)
		content := core.Trim(record.Content)
		if role == "" && content == "" {
			continue
		}
		out = append(out, Message{Role: role, Content: content})
	}
	return out
}

func datasetShareGPTMessages(records []datasetShareGPTRecord) []Message {
	out := make([]Message, 0, len(records))
	for _, record := range records {
		role := normalizeDatasetRole(record.From)
		content := core.Trim(record.Value)
		if role == "" && content == "" {
			continue
		}
		out = append(out, Message{Role: role, Content: content})
	}
	return out
}

func messagesToSFTSample(messages []Message, cfg ChatTemplateConfig, format string) (SFTSample, bool, error) {
	if len(messages) == 0 {
		return SFTSample{}, false, nil
	}
	assistantIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if normalizeDatasetRole(messages[i].Role) == "assistant" {
			assistantIdx = i
			break
		}
	}
	if assistantIdx < 0 {
		text := FormatChatMessages(messages, ChatTemplateConfig{
			Architecture:       cfg.Architecture,
			Template:           cfg.Template,
			NoGenerationPrompt: true,
		})
		return datasetSample(SFTSample{Text: text}, format), true, nil
	}
	promptMessages := cloneMessages(messages[:assistantIdx])
	response := core.Trim(messages[assistantIdx].Content)
	prompt := FormatChatMessages(promptMessages, cfg)
	return datasetSample(SFTSample{Prompt: prompt, Response: response}, format), true, nil
}

// FormatChatMessages applies a native model-family chat template.
// Forwards to dappco.re/go/mlx/chat/.
//
//	text := mlx.FormatChatMessages(messages, cfg)
func FormatChatMessages(messages []Message, cfg ChatTemplateConfig) string {
	return chat.Format(messages, cfg)
}

func chatTemplateName(cfg ChatTemplateConfig) string {
	return chat.TemplateName(cfg)
}

func normalizeDatasetRole(role string) string {
	return chat.NormaliseRole(role)
}

// BuildDatasetBatches tokenizes an SFT dataset with optional sequence packing.
func BuildDatasetBatches(tok *Tokenizer, dataset SFTDataset, cfg DatasetBatchConfig) ([]SFTBatch, error) {
	if !cfg.SequencePacking {
		return BuildSFTBatches(tok, dataset, SFTConfig{
			BatchSize: cfg.BatchSize,
			MaxSeqLen: cfg.MaxSeqLen,
			NoEOS:     cfg.NoEOS,
		})
	}
	if tok == nil || tok.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if dataset == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	cfg = normalizeDatasetBatchConfig(cfg)
	builder := newSFTBatchBuilder(cfg.BatchSize)
	packer := newDatasetPacker(cfg.MaxSeqLen, builder)
	for {
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, SFTConfig{MaxSeqLen: cfg.MaxSeqLen, NoEOS: cfg.NoEOS})
		if err != nil {
			return nil, err
		}
		if usable {
			packer.add(example)
		}
	}
	packer.finish()
	return builder.finish(), nil
}

func normalizeDatasetBatchConfig(cfg DatasetBatchConfig) DatasetBatchConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	return cfg
}

type datasetPacker struct {
	maxSeqLen int
	builder   *sftBatchBuilder
	current   sftExample
}

func newDatasetPacker(maxSeqLen int, builder *sftBatchBuilder) *datasetPacker {
	return &datasetPacker{maxSeqLen: maxSeqLen, builder: builder}
}

func (p *datasetPacker) add(example sftExample) {
	if p == nil || p.builder == nil {
		return
	}
	if len(example.inputs) == 0 {
		return
	}
	if p.maxSeqLen > 0 && len(p.current.inputs) > 0 && len(p.current.inputs)+len(example.inputs) > p.maxSeqLen {
		p.flush()
	}
	if p.maxSeqLen > 0 && len(example.inputs) > p.maxSeqLen {
		start := len(example.inputs) - p.maxSeqLen
		example.inputs = append([]int(nil), example.inputs[start:]...)
		example.targets = append([]int(nil), example.targets[start:]...)
		example.mask = append([]float32(nil), example.mask[start:]...)
	}
	p.current.inputs = append(p.current.inputs, example.inputs...)
	p.current.targets = append(p.current.targets, example.targets...)
	p.current.mask = append(p.current.mask, example.mask...)
}

func (p *datasetPacker) finish() {
	if p != nil {
		p.flush()
	}
}

func (p *datasetPacker) flush() {
	if p == nil || p.builder == nil || len(p.current.inputs) == 0 {
		return
	}
	p.builder.add(sftExample{
		inputs:  append([]int(nil), p.current.inputs...),
		targets: append([]int(nil), p.current.targets...),
		mask:    append([]float32(nil), p.current.mask...),
	})
	p.current = sftExample{}
}

func datasetSample(sample SFTSample, format string) SFTSample {
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

func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	return out
}

func cloneSFTSamples(samples []SFTSample) []SFTSample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]SFTSample, len(samples))
	for i, sample := range samples {
		out[i] = cloneSFTSample(sample)
	}
	return out
}

func cloneSFTSample(sample SFTSample) SFTSample {
	sample.Meta = cloneStringMap(sample.Meta)
	return sample
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func datasetResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
