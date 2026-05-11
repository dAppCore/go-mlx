// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import "dappco.re/go/inference"

func (adapter *metaladapter) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return adapter.outputParser().ParseReasoning(tokens, text)
}

func (adapter *metaladapter) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	return adapter.outputParser().ParseTools(tokens, text)
}

func (adapter *metaladapter) outputParser() ModelOutputParser {
	if adapter == nil || adapter.model == nil {
		return ParserForModel(ModelInfo{})
	}
	return ParserForModel(adapter.rootModel().Info())
}
