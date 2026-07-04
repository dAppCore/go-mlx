// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/parser"
)

func ExampleWithThinkingMode() {
	cfg := DefaultGenerateConfig()
	WithThinkingMode(parser.Hide)(&cfg)
	core.Println(cfg.Thinking.Mode)
	// Output: hide
}

func ExampleWithShowThinking() {
	cfg := DefaultGenerateConfig()
	WithShowThinking()(&cfg)
	core.Println(cfg.Thinking.Mode)
	// Output: show
}
