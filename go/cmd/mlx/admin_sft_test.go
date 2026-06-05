// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
)

func TestAdminSFTDatasetConfig_Gemma4LargeMessagesUseSharedFormatter_Good(t *testing.T) {
	input := `{"messages":[{"role":"user","content":"Write one line."},{"role":"assistant","content":"ok"}]}`
	cfg := adminSFTDatasetConfig(mlx.ModelInfo{Architecture: "gemma4_text", NumHeads: 16})

	ds, err := dataset.LoadJSONL(strings.NewReader(input), cfg)
	if err != nil {
		t.Fatalf("LoadJSONL() error = %v", err)
	}
	sample, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok {
		t.Fatal("Next() ok = false, want sample")
	}

	wantPrompt := chat.Format([]inference.Message{{Role: "user", Content: "Write one line."}}, chat.Config{
		Architecture: "gemma4_text",
		LargeVariant: true,
	})
	if sample.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want shared Gemma4 formatter %q", sample.Prompt, wantPrompt)
	}
	if sample.Response != "ok" {
		t.Fatalf("Response = %q, want assistant message", sample.Response)
	}
}
