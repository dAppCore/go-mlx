// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ModelOutputParser is the go-mlx parser surface for model-family reasoning
// channels and tool-call syntax.
type ModelOutputParser interface {
	ParserID() string
	inference.ReasoningParser
	inference.ToolParser
}

// ParserRegistry maps model families and architecture aliases to output parsers.
type ParserRegistry struct {
	parsers  map[string]ModelOutputParser
	fallback ModelOutputParser
}

// NewParserRegistry creates a registry with the generic fallback parser.
func NewParserRegistry() *ParserRegistry {
	generic := newBuiltinOutputParser("generic", genericReasoningMarkers())
	return &ParserRegistry{
		parsers:  map[string]ModelOutputParser{"generic": generic},
		fallback: generic,
	}
}

// DefaultParserRegistry returns the built-in go-mlx parser registry.
func DefaultParserRegistry() *ParserRegistry {
	registry := NewParserRegistry()
	registry.Register(newBuiltinOutputParser("qwen", qwenReasoningMarkers()), "qwen", "qwen2", "qwen3")
	registry.Register(newBuiltinOutputParser("gemma", gemmaReasoningMarkers()), "gemma", "gemma3", "gemma4", "gemma4_text")
	registry.Register(newBuiltinOutputParser("minimax", qwenReasoningMarkers()), "minimax", "minimax_m2", "minimax-m2")
	registry.Register(newBuiltinOutputParser("deepseek-r1", qwenReasoningMarkers()), "deepseek", "deepseek_r1", "deepseek-r1")
	registry.Register(newBuiltinOutputParser("gpt-oss", gptOSSReasoningMarkers()), "gpt-oss", "gpt_oss", "gptoss")
	registry.Register(newBuiltinOutputParser("mistral", genericReasoningMarkers()), "mistral", "mixtral")
	registry.Register(newBuiltinOutputParser("kimi", qwenReasoningMarkers()), "kimi", "kimi_k2", "moonshot")
	registry.Register(newBuiltinOutputParser("glm", qwenReasoningMarkers()), "glm", "glm4", "chatglm")
	registry.Register(newBuiltinOutputParser("hermes", genericReasoningMarkers()), "hermes", "hermes2", "hermes3")
	registry.Register(newBuiltinOutputParser("granite", genericReasoningMarkers()), "granite", "ibm-granite")
	return registry
}

// Register adds aliases for parser. Empty aliases are ignored.
func (registry *ParserRegistry) Register(parser ModelOutputParser, aliases ...string) {
	if registry == nil || parser == nil {
		return
	}
	if registry.parsers == nil {
		registry.parsers = map[string]ModelOutputParser{}
	}
	registry.parsers[normaliseParserKey(parser.ParserID())] = parser
	for _, alias := range aliases {
		key := normaliseParserKey(alias)
		if key == "" {
			continue
		}
		registry.parsers[key] = parser
	}
	if registry.fallback == nil {
		registry.fallback = parser
	}
}

// Lookup returns the parser registered for name.
func (registry *ParserRegistry) Lookup(name string) (ModelOutputParser, bool) {
	if registry == nil {
		return nil, false
	}
	parser, ok := registry.parsers[normaliseParserKey(name)]
	return parser, ok
}

// LookupModel returns the best parser for info, falling back to generic.
func (registry *ParserRegistry) LookupModel(info ModelInfo) ModelOutputParser {
	if registry == nil {
		return DefaultParserRegistry().LookupModel(info)
	}
	if parser, ok := registry.Lookup(modelParserFamily(info)); ok {
		return parser
	}
	if registry.fallback != nil {
		return registry.fallback
	}
	return newBuiltinOutputParser("generic", genericReasoningMarkers())
}

// ParserForModel resolves the default parser for info.
func ParserForModel(info ModelInfo) ModelOutputParser {
	return DefaultParserRegistry().LookupModel(info)
}

// ParserForInferenceModel resolves the default parser for a shared inference
// model identity.
func ParserForInferenceModel(info inference.ModelInfo) ModelOutputParser {
	return ParserForModel(modelInfoFromInference(info))
}

func modelInfoFromInference(info inference.ModelInfo) ModelInfo {
	return ModelInfo{
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	}
}

func normaliseParserKey(value string) string {
	value = core.Lower(core.Trim(value))
	value = replaceAll(value, "-", "_")
	value = replaceAll(value, ".", "_")
	return value
}

func modelParserFamily(info ModelInfo) string {
	arch := normaliseParserKey(info.Architecture)
	adapter := normaliseParserKey(info.Adapter.Name)
	combined := core.Concat(arch, " ", adapter)
	switch {
	case core.Contains(combined, "qwen"):
		return "qwen"
	case core.Contains(combined, "gemma"):
		return "gemma"
	case core.Contains(combined, "minimax"):
		return "minimax"
	case core.Contains(combined, "deepseek"):
		return "deepseek_r1"
	case core.Contains(combined, "gpt_oss") || core.Contains(combined, "gptoss"):
		return "gpt_oss"
	case core.Contains(combined, "mistral") || core.Contains(combined, "mixtral"):
		return "mistral"
	case core.Contains(combined, "kimi") || core.Contains(combined, "moonshot"):
		return "kimi"
	case core.Contains(combined, "glm") || core.Contains(combined, "chatglm"):
		return "glm"
	case core.Contains(combined, "hermes"):
		return "hermes"
	case core.Contains(combined, "granite"):
		return "granite"
	default:
		return "generic"
	}
}

type reasoningMarkerSpec struct {
	start string
	ends  []string
	kind  string
}

type builtinOutputParser struct {
	id      string
	markers []reasoningMarkerSpec
}

func newBuiltinOutputParser(id string, markers []reasoningMarkerSpec) *builtinOutputParser {
	return &builtinOutputParser{id: id, markers: append([]reasoningMarkerSpec(nil), markers...)}
}

func (parser *builtinOutputParser) ParserID() string {
	if parser == nil || parser.id == "" {
		return "generic"
	}
	return parser.id
}

func (parser *builtinOutputParser) ParseReasoning(_ []inference.Token, text string) (inference.ReasoningParseResult, error) {
	if parser == nil {
		parser = newBuiltinOutputParser("generic", genericReasoningMarkers())
	}
	return parseReasoningText(text, parser.markers), nil
}

func (parser *builtinOutputParser) ParseTools(_ []inference.Token, text string) (inference.ToolParseResult, error) {
	return parseToolText(text)
}

func qwenReasoningMarkers() []reasoningMarkerSpec {
	return append([]reasoningMarkerSpec{
		{start: "<think>", ends: []string{"</think>"}, kind: "thinking"},
	}, genericReasoningMarkers()...)
}

func gemmaReasoningMarkers() []reasoningMarkerSpec {
	return append([]reasoningMarkerSpec{
		{start: "<start_of_turn>thinking\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
		{start: "<start_of_turn>thought\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
		{start: "<start_of_turn>analysis\n", ends: []string{"<end_of_turn>"}, kind: "analysis"},
		{start: "<start_of_turn>reasoning\n", ends: []string{"<end_of_turn>"}, kind: "reasoning"},
	}, genericReasoningMarkers()...)
}

func gptOSSReasoningMarkers() []reasoningMarkerSpec {
	return append([]reasoningMarkerSpec{
		{start: "<|channel>analysis\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "analysis"},
		{start: "<|channel>thought\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "thinking"},
		{start: "<|channel>reasoning\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "reasoning"},
		{start: "<|channel>analysis", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "analysis"},
		{start: "<|channel>thought", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "thinking"},
		{start: "<|channel>reasoning", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "reasoning"},
	}, genericReasoningMarkers()...)
}

func genericReasoningMarkers() []reasoningMarkerSpec {
	return []reasoningMarkerSpec{
		{start: "<thinking>", ends: []string{"</thinking>"}, kind: "thinking"},
		{start: "<thought>", ends: []string{"</thought>"}, kind: "thinking"},
		{start: "<reasoning>", ends: []string{"</reasoning>"}, kind: "reasoning"},
		{start: "<analysis>", ends: []string{"</analysis>"}, kind: "analysis"},
	}
}

func parseReasoningText(text string, markers []reasoningMarkerSpec) inference.ReasoningParseResult {
	visible := core.NewBuilder()
	segments := []inference.ReasoningSegment{}
	pending := text
	tokenOffset := 0
	for pending != "" {
		idx, marker, ok := findReasoningStart(pending, markers)
		if !ok {
			visible.WriteString(pending)
			break
		}
		visible.WriteString(pending[:idx])
		tokenOffset += idx
		afterStart := pending[idx+len(marker.start):]
		end, endSize := firstReasoningEnd(afterStart, marker.ends)
		if end < 0 {
			reasoning := trimReasoningText(afterStart)
			if reasoning != "" {
				segments = append(segments, inference.ReasoningSegment{Kind: marker.kind, Text: reasoning, StartToken: tokenOffset})
			}
			break
		}
		reasoning := trimReasoningText(afterStart[:end])
		if reasoning != "" {
			segments = append(segments, inference.ReasoningSegment{Kind: marker.kind, Text: reasoning, StartToken: tokenOffset, EndToken: tokenOffset + end})
		}
		pending = afterStart[end+endSize:]
		tokenOffset += len(marker.start) + end + endSize
	}
	return inference.ReasoningParseResult{VisibleText: visible.String(), Reasoning: segments}
}

func findReasoningStart(text string, markers []reasoningMarkerSpec) (int, reasoningMarkerSpec, bool) {
	best := -1
	var marker reasoningMarkerSpec
	for _, candidate := range markers {
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

func firstReasoningEnd(text string, ends []string) (int, int) {
	best := -1
	bestSize := 0
	for _, end := range ends {
		idx := indexString(text, end)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			bestSize = len(end)
		}
	}
	return best, bestSize
}

func trimReasoningText(text string) string {
	return core.Trim(text)
}

type toolBlockMarker struct {
	start string
	end   string
}

var toolBlockMarkers = []toolBlockMarker{
	{start: "<tool_call>", end: "</tool_call>"},
	{start: "<tool_calls>", end: "</tool_calls>"},
	{start: "<function_call>", end: "</function_call>"},
}

func parseToolText(text string) (inference.ToolParseResult, error) {
	visible := core.NewBuilder()
	calls := []inference.ToolCall{}
	pending := text
	foundTagged := false
	for pending != "" {
		idx, marker, ok := findToolBlockStart(pending)
		if !ok {
			visible.WriteString(pending)
			break
		}
		foundTagged = true
		visible.WriteString(pending[:idx])
		afterStart := pending[idx+len(marker.start):]
		end := indexString(afterStart, marker.end)
		if end < 0 {
			visible.WriteString(pending[idx:])
			break
		}
		parsed, err := parseToolPayload(afterStart[:end])
		if err != nil {
			return inference.ToolParseResult{}, err
		}
		calls = append(calls, parsed...)
		pending = afterStart[end+len(marker.end):]
	}
	if !foundTagged {
		parsed, err := parseToolPayload(text)
		if err == nil && len(parsed) > 0 {
			return inference.ToolParseResult{VisibleText: "", Calls: parsed}, nil
		}
	}
	return inference.ToolParseResult{VisibleText: visible.String(), Calls: calls}, nil
}

func findToolBlockStart(text string) (int, toolBlockMarker, bool) {
	best := -1
	var marker toolBlockMarker
	for _, candidate := range toolBlockMarkers {
		idx := indexString(text, candidate.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			marker = candidate
		}
	}
	return best, marker, best >= 0
}

type parsedToolCall struct {
	ID            string           `json:"id"`
	Type          string           `json:"type"`
	Name          string           `json:"name"`
	Arguments     any              `json:"arguments"`
	ArgumentsJSON string           `json:"arguments_json"`
	Function      *parsedFunction  `json:"function"`
	ToolCalls     []parsedToolCall `json:"tool_calls"`
	Calls         []parsedToolCall `json:"calls"`
}

type parsedFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

func parseToolPayload(payload string) ([]inference.ToolCall, error) {
	payload = core.Trim(payload)
	if payload == "" {
		return nil, nil
	}
	var list []parsedToolCall
	if core.HasPrefix(payload, "[") {
		result := core.JSONUnmarshalString(payload, &list)
		if !result.OK {
			return nil, resultError("mlx.parser.tool", result)
		}
		return convertParsedToolCalls(list), nil
	}
	var envelope parsedToolCall
	result := core.JSONUnmarshalString(payload, &envelope)
	if !result.OK {
		return nil, resultError("mlx.parser.tool", result)
	}
	if len(envelope.ToolCalls) > 0 {
		return convertParsedToolCalls(envelope.ToolCalls), nil
	}
	if len(envelope.Calls) > 0 {
		return convertParsedToolCalls(envelope.Calls), nil
	}
	call := convertParsedToolCall(envelope)
	if call.Name == "" {
		return nil, nil
	}
	return []inference.ToolCall{call}, nil
}

func convertParsedToolCalls(input []parsedToolCall) []inference.ToolCall {
	out := make([]inference.ToolCall, 0, len(input))
	for _, parsed := range input {
		call := convertParsedToolCall(parsed)
		if call.Name != "" {
			out = append(out, call)
		}
	}
	return out
}

func convertParsedToolCall(parsed parsedToolCall) inference.ToolCall {
	name := parsed.Name
	args := parsed.Arguments
	if parsed.Function != nil {
		if parsed.Function.Name != "" {
			name = parsed.Function.Name
		}
		if parsed.Function.Arguments != nil {
			args = parsed.Function.Arguments
		}
	}
	callType := parsed.Type
	if callType == "" {
		callType = "function"
	}
	return inference.ToolCall{
		ID:            parsed.ID,
		Type:          callType,
		Name:          name,
		ArgumentsJSON: normaliseArgumentsJSON(parsed.ArgumentsJSON, args),
	}
}

func normaliseArgumentsJSON(existing string, args any) string {
	if core.Trim(existing) != "" {
		return core.Trim(existing)
	}
	if args == nil {
		return ""
	}
	if raw, ok := args.(string); ok {
		return core.Trim(raw)
	}
	return core.JSONMarshalString(args)
}

func resultError(scope string, result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return core.Wrap(err, scope, "parse JSON")
	}
	return core.E(scope, "parse JSON", nil)
}

func replaceAll(text, old, next string) string {
	if old == "" {
		return text
	}
	out := core.NewBuilder()
	for {
		idx := indexString(text, old)
		if idx < 0 {
			out.WriteString(text)
			return out.String()
		}
		out.WriteString(text[:idx])
		out.WriteString(next)
		text = text[idx+len(old):]
	}
}
