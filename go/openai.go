// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"net/http"

	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

// NewOpenAIResolver returns a resolver that lazily loads modelPath through the
// native Metal backend registered by this package.
func NewOpenAIResolver(modelPath string, opts ...inference.LoadOption) *openaicompat.BackendResolver {
	return openaicompat.NewBackendResolver("metal", modelPath, opts...)
}

// NewOpenAIHandler exposes modelPath through the shared OpenAI-compatible chat
// completions handler.
func NewOpenAIHandler(modelPath string, opts ...inference.LoadOption) http.Handler {
	return openaicompat.NewHandler(NewOpenAIResolver(modelPath, opts...))
}
