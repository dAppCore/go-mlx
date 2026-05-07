// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

// ThinkingMode controls how model-internal thinking/reasoning channels are exposed.
type ThinkingMode string

const (
	// ThinkingShow leaves model output untouched. This is the compatibility default.
	ThinkingShow ThinkingMode = "show"
	// ThinkingHide removes recognized thinking-channel text from visible output.
	ThinkingHide ThinkingMode = "hide"
	// ThinkingCapture removes recognized thinking-channel text and emits it separately.
	ThinkingCapture ThinkingMode = "capture"
)

// ThinkingChunk is one captured model-internal reasoning block.
type ThinkingChunk struct {
	Text    string `json:"text"`
	Channel string `json:"channel,omitempty"`
	Model   string `json:"model,omitempty"`
}

// ThinkingConfig configures model-aware thinking-channel handling.
type ThinkingConfig struct {
	Mode    ThinkingMode        `json:"mode,omitempty"`
	Capture func(ThinkingChunk) `json:"-"`
}

// ThinkingResult is the filtered visible text plus extracted reasoning text.
type ThinkingResult struct {
	Text      string          `json:"text"`
	Reasoning string          `json:"reasoning,omitempty"`
	Chunks    []ThinkingChunk `json:"chunks,omitempty"`
}

// WithThinkingMode sets whether reasoning text is shown, hidden, or captured.
func WithThinkingMode(mode ThinkingMode) GenerateOption {
	return func(c *GenerateConfig) { c.Thinking.Mode = mode }
}

// WithShowThinking leaves reasoning markers and content in the visible output.
func WithShowThinking() GenerateOption {
	return WithThinkingMode(ThinkingShow)
}

// WithHideThinking removes recognized reasoning markers and content.
func WithHideThinking() GenerateOption {
	return WithThinkingMode(ThinkingHide)
}

// WithCaptureThinking removes reasoning from visible output and calls capture for each block.
func WithCaptureThinking(capture func(ThinkingChunk)) GenerateOption {
	return func(c *GenerateConfig) {
		c.Thinking.Mode = ThinkingCapture
		c.Thinking.Capture = capture
	}
}

// WithThinkingCapture is an alias for WithCaptureThinking.
func WithThinkingCapture(capture func(ThinkingChunk)) GenerateOption {
	return WithCaptureThinking(capture)
}

// FilterThinkingText applies thinking-channel handling to a complete text buffer.
func FilterThinkingText(text string, cfg ThinkingConfig, info ModelInfo) ThinkingResult {
	processor := newThinkingChannelProcessor(cfg, info)
	builder := core.NewBuilder()
	builder.WriteString(processor.Process(text))
	builder.WriteString(processor.Flush())
	return ThinkingResult{
		Text:      builder.String(),
		Reasoning: processor.Reasoning(),
		Chunks:    processor.Chunks(),
	}
}

// FilterThinkingTokens applies thinking-channel handling token by token using decoded token pieces.
func FilterThinkingTokens(tok *Tokenizer, ids []int32, cfg ThinkingConfig, info ModelInfo) (ThinkingResult, error) {
	if tok == nil || tok.tok == nil {
		return ThinkingResult{}, core.NewError("mlx: tokenizer is nil")
	}
	processor := newThinkingChannelProcessor(cfg, info)
	builder := core.NewBuilder()
	for _, id := range ids {
		piece := tok.IDToken(id)
		if piece == "" {
			decoded, err := tok.Decode([]int32{id})
			if err != nil {
				return ThinkingResult{}, err
			}
			piece = decoded
		}
		builder.WriteString(processor.Process(piece))
	}
	builder.WriteString(processor.Flush())
	return ThinkingResult{
		Text:      builder.String(),
		Reasoning: processor.Reasoning(),
		Chunks:    processor.Chunks(),
	}, nil
}

type thinkingMarker struct {
	start   string
	end     string
	channel string
	model   string
}

type thinkingChannelProcessor struct {
	cfg            ThinkingConfig
	mode           ThinkingMode
	markers        []thinkingMarker
	pending        string
	inReasoning    bool
	current        thinkingMarker
	reasoningParts []string
	blockParts     []string
	chunks         []ThinkingChunk
}

func newThinkingChannelProcessor(cfg ThinkingConfig, info ModelInfo) *thinkingChannelProcessor {
	mode := normalizeThinkingMode(cfg.Mode)
	return &thinkingChannelProcessor{
		cfg:     cfg,
		mode:    mode,
		markers: thinkingMarkersForModel(info),
	}
}

func normalizeThinkingMode(mode ThinkingMode) ThinkingMode {
	switch mode {
	case "", ThinkingShow:
		return ThinkingShow
	case ThinkingHide, ThinkingCapture:
		return mode
	default:
		return ThinkingShow
	}
}

func thinkingMarkersForModel(info ModelInfo) []thinkingMarker {
	arch := core.Lower(info.Architecture)
	modelType := core.Lower(info.Adapter.Name)
	markers := []thinkingMarker{
		{start: "<think>", end: "</think>", channel: "thinking", model: "qwen"},
		{start: "<thinking>", end: "</thinking>", channel: "thinking", model: "generic"},
		{start: "<thought>", end: "</thought>", channel: "thinking", model: "generic"},
		{start: "<reasoning>", end: "</reasoning>", channel: "reasoning", model: "generic"},
	}
	if core.Contains(arch, "gemma") || core.Contains(modelType, "gemma") {
		markers = append(markers,
			thinkingMarker{start: "<start_of_turn>thinking\n", end: "<end_of_turn>", channel: "thinking", model: "gemma"},
			thinkingMarker{start: "<start_of_turn>thought\n", end: "<end_of_turn>", channel: "thinking", model: "gemma"},
			thinkingMarker{start: "<start_of_turn>analysis\n", end: "<end_of_turn>", channel: "analysis", model: "gemma"},
			thinkingMarker{start: "<start_of_turn>reasoning\n", end: "<end_of_turn>", channel: "reasoning", model: "gemma"},
		)
	}
	return markers
}

func (p *thinkingChannelProcessor) Process(text string) string {
	if p.mode == ThinkingShow || text == "" {
		return text
	}
	p.pending += text
	return p.drain(false)
}

func (p *thinkingChannelProcessor) Flush() string {
	if p.mode == ThinkingShow {
		return ""
	}
	out := p.drain(true)
	if p.pending == "" {
		if p.inReasoning {
			p.emitReasoningBlock()
			p.inReasoning = false
		}
		return out
	}
	if p.inReasoning {
		p.addReasoning(p.pending)
		p.pending = ""
		p.emitReasoningBlock()
		p.inReasoning = false
		return out
	}
	out += p.pending
	p.pending = ""
	return out
}

func (p *thinkingChannelProcessor) Reasoning() string {
	return core.Join("", p.reasoningParts...)
}

func (p *thinkingChannelProcessor) Chunks() []ThinkingChunk {
	if len(p.chunks) == 0 {
		return nil
	}
	return append([]ThinkingChunk(nil), p.chunks...)
}

func (p *thinkingChannelProcessor) drain(final bool) string {
	out := core.NewBuilder()
	for p.pending != "" {
		if p.inReasoning {
			idx := indexString(p.pending, p.current.end)
			if idx >= 0 {
				p.addReasoning(p.pending[:idx])
				p.pending = p.pending[idx+len(p.current.end):]
				p.emitReasoningBlock()
				p.inReasoning = false
				continue
			}
			keep := 0
			if !final {
				keep = longestSuffixPrefix(p.pending, []string{p.current.end})
			}
			consume := len(p.pending) - keep
			if consume > 0 {
				p.addReasoning(p.pending[:consume])
				p.pending = p.pending[consume:]
			}
			break
		}

		idx, marker, ok := p.findStart(p.pending)
		if ok {
			out.WriteString(p.pending[:idx])
			p.pending = p.pending[idx+len(marker.start):]
			p.current = marker
			p.inReasoning = true
			continue
		}
		keep := 0
		if !final {
			keep = longestSuffixPrefix(p.pending, p.startMarkers())
		}
		consume := len(p.pending) - keep
		if consume > 0 {
			out.WriteString(p.pending[:consume])
			p.pending = p.pending[consume:]
		}
		break
	}
	return out.String()
}

func (p *thinkingChannelProcessor) findStart(text string) (int, thinkingMarker, bool) {
	best := -1
	var marker thinkingMarker
	for _, candidate := range p.markers {
		idx := indexString(text, candidate.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best || idx == best && len(candidate.start) > len(marker.start) {
			best = idx
			marker = candidate
		}
	}
	return best, marker, best >= 0
}

func (p *thinkingChannelProcessor) startMarkers() []string {
	out := make([]string, len(p.markers))
	for i, marker := range p.markers {
		out[i] = marker.start
	}
	return out
}

func (p *thinkingChannelProcessor) addReasoning(text string) {
	if text == "" {
		return
	}
	p.reasoningParts = append(p.reasoningParts, text)
	p.blockParts = append(p.blockParts, text)
}

func (p *thinkingChannelProcessor) emitReasoningBlock() {
	text := core.Join("", p.blockParts...)
	p.blockParts = nil
	if text == "" {
		return
	}
	chunk := ThinkingChunk{
		Text:    text,
		Channel: p.current.channel,
		Model:   p.current.model,
	}
	p.chunks = append(p.chunks, chunk)
	if p.mode == ThinkingCapture && p.cfg.Capture != nil {
		p.cfg.Capture(chunk)
	}
}

func longestSuffixPrefix(text string, markers []string) int {
	best := 0
	for _, marker := range markers {
		max := len(marker) - 1
		if max > len(text) {
			max = len(text)
		}
		for size := max; size > best; size-- {
			if core.HasPrefix(marker, text[len(text)-size:]) {
				best = size
				break
			}
		}
	}
	return best
}
