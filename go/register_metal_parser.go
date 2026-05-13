// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
)

func (adapter *metaladapter) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return adapter.outputParser().ParseReasoning(tokens, text)
}

func (adapter *metaladapter) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	return adapter.outputParser().ParseTools(tokens, text)
}

func (adapter *metaladapter) outputParser() parser.OutputParser {
	if adapter == nil || adapter.model == nil {
		return parser.ForHint(parser.Hint{})
	}
	return parser.ForHint(parserHint(adapter.rootModel().Info()))
}
