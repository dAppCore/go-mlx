// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

func ExampleWithSuppressTokens() {
	cfg := DefaultGenerateConfig()
	WithSuppressTokens(13, 42)(&cfg)
	core.Println(cfg.SuppressTokens)
	// Output: [13 42]
}

func ExampleWithThinkingBudget() {
	cfg := DefaultGenerateConfig()
	WithThinkingBudget(256)(&cfg)
	core.Println(cfg.ThinkingBudget)
	// Output: 256
}

func ExampleWithProbeCallback() {
	cfg := DefaultGenerateConfig()
	count := 0
	WithProbeCallback(func(probe.Event) { count++ })(&cfg)
	cfg.ProbeSink.EmitProbe(probe.Event{})
	core.Println(count)
	// Output: 1
}
