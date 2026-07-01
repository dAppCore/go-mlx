// SPDX-Licence-Identifier: EUPL-1.2

package gemma4chat

import (
	"testing"

	"dappco.re/go/mlx/chat"
)

func TestModelGemma4ChatRegistersNeutralFormatter_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "gemma4_text"})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 neutral formatter = %q, want %q", got, want)
	}
}
