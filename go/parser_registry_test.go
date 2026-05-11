// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

func TestParserRegistry_DefaultLookup_Good_ModelFamilies(t *testing.T) {
	cases := map[string]string{
		"qwen3":       "qwen",
		"gemma4_text": "gemma",
		"minimax_m2":  "minimax",
		"deepseek_r1": "deepseek-r1",
		"gpt_oss":     "gpt-oss",
		"mistral":     "mistral",
		"kimi_k2":     "kimi",
		"glm4":        "glm",
		"hermes3":     "hermes",
		"granite":     "granite",
		"unknown":     "generic",
	}

	for arch, want := range cases {
		parser := ParserForModel(ModelInfo{Architecture: arch})
		if parser == nil {
			t.Fatalf("ParserForModel(%q) returned nil", arch)
		}
		if parser.ParserID() != want {
			t.Fatalf("ParserForModel(%q) = %q, want %q", arch, parser.ParserID(), want)
		}
	}
}

func TestParserRegistry_ReasoningParsers_Good(t *testing.T) {
	cases := []struct {
		name      string
		arch      string
		text      string
		visible   string
		reasoning string
		kind      string
	}{
		{
			name:      "qwen think tags",
			arch:      "qwen3",
			text:      "pre<think>plan</think>answer",
			visible:   "preanswer",
			reasoning: "plan",
			kind:      "thinking",
		},
		{
			name:      "gemma turn markers",
			arch:      "gemma4_text",
			text:      "<start_of_turn>thinking\nplan<end_of_turn>done",
			visible:   "done",
			reasoning: "plan",
			kind:      "thinking",
		},
		{
			name:      "gpt oss channel markers",
			arch:      "gpt_oss",
			text:      "<|channel>analysis\nplan<|channel>final\nanswer",
			visible:   "answer",
			reasoning: "plan",
			kind:      "analysis",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParserForModel(ModelInfo{Architecture: tc.arch}).ParseReasoning(nil, tc.text)
			if err != nil {
				t.Fatalf("ParseReasoning() error = %v", err)
			}
			if got.VisibleText != tc.visible {
				t.Fatalf("VisibleText = %q, want %q", got.VisibleText, tc.visible)
			}
			if len(got.Reasoning) != 1 {
				t.Fatalf("Reasoning len = %d, want 1: %+v", len(got.Reasoning), got.Reasoning)
			}
			if got.Reasoning[0].Text != tc.reasoning || got.Reasoning[0].Kind != tc.kind {
				t.Fatalf("Reasoning[0] = %+v, want %q/%q", got.Reasoning[0], tc.kind, tc.reasoning)
			}
		})
	}
}

func TestParserRegistry_ToolParser_Good_TaggedAndJSONFallback(t *testing.T) {
	parser := ParserForModel(ModelInfo{Architecture: "hermes3"})

	tagged, err := parser.ParseTools(nil, `before <tool_call>{"name":"search","arguments":{"q":"core"}}</tool_call> after`)
	if err != nil {
		t.Fatalf("ParseTools(tagged) error = %v", err)
	}
	if tagged.VisibleText != "before  after" {
		t.Fatalf("tagged visible = %q", tagged.VisibleText)
	}
	if len(tagged.Calls) != 1 || tagged.Calls[0].Name != "search" || tagged.Calls[0].ArgumentsJSON != `{"q":"core"}` {
		t.Fatalf("tagged calls = %+v", tagged.Calls)
	}

	jsonFallback, err := parser.ParseTools(nil, `{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"id":7}}}]}`)
	if err != nil {
		t.Fatalf("ParseTools(json) error = %v", err)
	}
	if jsonFallback.VisibleText != "" {
		t.Fatalf("json visible = %q, want empty", jsonFallback.VisibleText)
	}
	if len(jsonFallback.Calls) != 1 || jsonFallback.Calls[0].ID != "call_1" || jsonFallback.Calls[0].Name != "lookup" || jsonFallback.Calls[0].ArgumentsJSON != `{"id":7}` {
		t.Fatalf("json calls = %+v", jsonFallback.Calls)
	}
}

type customOutputParser struct{}

func (customOutputParser) ParserID() string { return "custom" }

func (customOutputParser) ParseReasoning(_ []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return inference.ReasoningParseResult{VisibleText: "custom:" + text}, nil
}

func (customOutputParser) ParseTools(_ []inference.Token, text string) (inference.ToolParseResult, error) {
	return inference.ToolParseResult{VisibleText: text}, nil
}

func TestParserRegistry_RegisterCustomParser_Good(t *testing.T) {
	registry := NewParserRegistry()
	registry.Register(customOutputParser{}, "custom-family")

	parser, ok := registry.Lookup("custom-family")
	if !ok {
		t.Fatal("Lookup(custom-family) = false")
	}
	got, err := parser.ParseReasoning(nil, "answer")
	if err != nil {
		t.Fatalf("ParseReasoning() error = %v", err)
	}
	if parser.ParserID() != "custom" || got.VisibleText != "custom:answer" {
		t.Fatalf("parser/result = %q %+v", parser.ParserID(), got)
	}
}

func TestParserRegistry_FallbacksAndNilReceivers_Good(t *testing.T) {
	var nilRegistry *ParserRegistry
	if parser, ok := nilRegistry.Lookup("qwen"); ok || parser != nil {
		t.Fatalf("nil Lookup() = %+v/%v, want nil/false", parser, ok)
	}
	parser := nilRegistry.LookupModel(ModelInfo{Architecture: "qwen3"})
	if parser == nil || parser.ParserID() != "qwen" {
		t.Fatalf("nil LookupModel() = %v, want default qwen parser", parser)
	}
	registry := &ParserRegistry{}
	registry.Register(nil, "ignored")
	if parser := registry.LookupModel(ModelInfo{}); parser == nil || parser.ParserID() != "generic" {
		t.Fatalf("empty registry LookupModel() = %v, want generic fallback", parser)
	}
	registry.Register(customOutputParser{}, "", "custom.alias")
	if parser, ok := registry.Lookup("custom-alias"); !ok || parser.ParserID() != "custom" {
		t.Fatalf("Lookup(custom-alias) = %v/%v, want custom parser", parser, ok)
	}

	var nilParser *builtinOutputParser
	if nilParser.ParserID() != "generic" {
		t.Fatalf("nil builtin ParserID() = %q, want generic", nilParser.ParserID())
	}
	reasoning, err := nilParser.ParseReasoning(nil, "<analysis>plan</analysis>answer")
	if err != nil || reasoning.VisibleText != "answer" || len(reasoning.Reasoning) != 1 {
		t.Fatalf("nil builtin ParseReasoning() = %+v/%v, want generic parse", reasoning, err)
	}
}

func TestParserRegistry_ToolParser_BadAndUglyPayloads(t *testing.T) {
	parser := ParserForModel(ModelInfo{Architecture: "qwen3"})
	if _, err := parser.ParseTools(nil, `<tool_call>{bad}</tool_call>`); err == nil {
		t.Fatal("ParseTools(malformed tagged JSON) error = nil")
	}
	unclosed, err := parser.ParseTools(nil, `before <tool_call>{"name":"search"}`)
	if err != nil {
		t.Fatalf("ParseTools(unclosed tag) error = %v", err)
	}
	if unclosed.VisibleText != `before <tool_call>{"name":"search"}` || len(unclosed.Calls) != 0 {
		t.Fatalf("unclosed tool parse = %+v, want visible passthrough", unclosed)
	}
	if calls, err := parseToolPayload(`[{"name":"search","arguments_json":"{\"q\":\"core\"}"},{"name":""}]`); err != nil || len(calls) != 1 || calls[0].ArgumentsJSON != `{"q":"core"}` {
		t.Fatalf("parseToolPayload(array) = %+v/%v, want one call with existing args JSON", calls, err)
	}
	if calls, err := parseToolPayload(`{"calls":[{"name":"lookup","arguments":"{\"id\":7}"}]}`); err != nil || len(calls) != 1 || calls[0].ArgumentsJSON != `{"id":7}` {
		t.Fatalf("parseToolPayload(calls) = %+v/%v, want string arguments normalised", calls, err)
	}
	if calls, err := parseToolPayload(`{"type":"function"}`); err != nil || len(calls) != 0 {
		t.Fatalf("parseToolPayload(no name) = %+v/%v, want no call", calls, err)
	}
	if _, err := parseToolPayload(`{bad}`); err == nil {
		t.Fatal("parseToolPayload(bad JSON) error = nil")
	}
}
