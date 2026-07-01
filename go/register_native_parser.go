// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
)

func (m *nativeTextModel) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return m.outputParser().ParseReasoning(tokens, text)
}

func (m *nativeTextModel) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	return m.outputParser().ParseTools(tokens, text)
}

func (m *nativeTextModel) outputParser() parser.OutputParser {
	if m == nil {
		return defaultOutputParser
	}
	return parser.ForHint(parser.HintFromInference(m.Info()))
}
