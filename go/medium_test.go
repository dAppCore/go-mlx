// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestMediumStagePathHelpers_GoodBad(t *testing.T) {
	if _, cleanup, err := stagePathFromMedium(nil, "models/demo"); err == nil || cleanup != nil {
		t.Fatalf("stagePathFromMedium(nil) cleanup set=%t err=%v, want error without cleanup", cleanup != nil, err)
	}

	medium := coreio.NewMemoryMedium()
	if err := medium.Write("models/demo/config.json", `{"model_type":"demo"}`); err != nil {
		t.Fatalf("write medium config: %v", err)
	}
	if err := medium.Write("models/demo/sub/tokenizer.json", `{}`); err != nil {
		t.Fatalf("write medium tokenizer: %v", err)
	}
	if err := medium.Write("models/demo/model.safetensors", "stub"); err != nil {
		t.Fatalf("write medium weights: %v", err)
	}
	if _, cleanup, err := stagePathFromMedium(medium, "models/missing/model.gguf"); err == nil || cleanup != nil {
		t.Fatalf("stage missing path cleanup set=%t err=%v, want missing path error", cleanup != nil, err)
	}
	staged, cleanup, err := stagePathFromMedium(medium, "models/demo/model.safetensors")
	if err != nil {
		t.Fatalf("stagePathFromMedium(file) error = %v", err)
	}
	if cleanup == nil {
		t.Fatal("stage cleanup = nil, want cleanup")
	}
	t.Cleanup(func() { _ = cleanup() })
	if core.PathBase(staged) != "model.safetensors" {
		t.Fatalf("staged path = %q, want model.safetensors target", staged)
	}
	if stat := core.Stat(staged); !stat.OK {
		t.Fatalf("staged file missing: %v", stat.Value)
	}

	if got := cleanMediumPath(" models/demo/ "); got != "models/demo" {
		t.Fatalf("cleanMediumPath = %q, want models/demo", got)
	}
	if got := mediumModelRoot("models/demo/model.safetensors"); got != "models/demo" {
		t.Fatalf("mediumModelRoot(file) = %q, want models/demo", got)
	}
	if got := mediumRelativePath("models/demo", "models/demo/sub/tokenizer.json"); got != "sub/tokenizer.json" {
		t.Fatalf("mediumRelativePath = %q, want sub/tokenizer.json", got)
	}
	if got := fromSlashPath("a/b"); got == "" {
		t.Fatal("fromSlashPath returned empty path")
	}
}
