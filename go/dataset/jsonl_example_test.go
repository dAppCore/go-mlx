// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"fmt"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"

	// The qwen3 template registers from the model package; without it the
	// chat-shape rows render the plain fallback.
	_ "dappco.re/go/mlx/pkg/metal/model/qwen3/chat"
)

// ExampleLoadJSONL ingests a small multi-shape JSONL corpus and reports the
// normalised provenance label the loader stamps on each row. Scalar fields
// only — the templated prompts are intentionally not printed (version-
// sensitive).
func ExampleLoadJSONL() {
	corpus := strings.Join([]string{
		`{"text":"plain corpus row"}`,
		`{"prompt":"capital of France?","response":"Paris"}`,
		`{"instruction":"summarise","input":"notes","output":"short"}`,
	}, "\n")

	ds, err := dataset.LoadJSONL(strings.NewReader(corpus), dataset.Config{})
	if err != nil {
		panic(err)
	}
	for {
		s, ok, err := ds.Next()
		if err != nil {
			panic(err)
		}
		if !ok {
			break
		}
		fmt.Println(s.Format)
	}
	// Output:
	// text
	// prompt_response
	// alpaca
}

// ExampleMessagesToSample converts an OpenAI-shape message list into a
// supervised sample, using the trailing assistant turn as the response.
func ExampleMessagesToSample() {
	messages := []inference.Message{
		{Role: "user", Content: "ping"},
		{Role: "assistant", Content: "  pong  "},
	}

	sample, ok, err := dataset.MessagesToSample(messages, chat.Config{Architecture: "qwen3"}, "openai_messages")
	if err != nil {
		panic(err)
	}
	fmt.Println("ok:", ok)
	fmt.Println("response:", sample.Response)
	fmt.Println("format:", sample.Format)
	// Output:
	// ok: true
	// response: pong
	// format: openai_messages
}
