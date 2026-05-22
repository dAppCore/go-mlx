// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
)

// defaultOutputParser is the no-hint fallback parser. Hoisted to package
// scope so the nil-adapter / nil-model path does not allocate a fresh
// parser interface box on every ParseReasoning / ParseTools call.
var defaultOutputParser = parser.ForHint(parser.Hint{})

func (adapter *metaladapter) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return adapter.outputParser().ParseReasoning(tokens, text)
}

func (adapter *metaladapter) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	return adapter.outputParser().ParseTools(tokens, text)
}

func (adapter *metaladapter) outputParser() parser.OutputParser {
	if adapter == nil || adapter.model == nil {
		return defaultOutputParser
	}
	// Bypass rootModel(). rootModel() allocates a fresh *Model + *Tokenizer
	// every call (~3 allocs) and itself calls adapter.model.Info() to seed
	// LoadConfig.ContextLength — work we don't need here. parserHint reads
	// only Architecture + Adapter.Name, both already on metal.ModelInfo
	// (metal.Model.Info() populates info.Adapter via m.Adapter()).
	info := adapter.model.Info()
	return parser.ForHint(parser.Hint{
		Architecture: info.Architecture,
		AdapterName:  info.Adapter.Name,
	})
}
