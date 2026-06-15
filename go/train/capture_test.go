// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the capture-first lane (#97): the model's raw return is appended to
// the capture sidecar the moment it exists, independent of any scoring. The SFT
// eval loop and the SSD sampling loop both feed it; here we drive the SFT loop
// with the cascade OFF to prove capture is score-independent by design. No model
// and no Metal — the synthetic captureEvalModel returns seeded text.

package train

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/spine"
)

type captureEvalModel struct{}

func (captureEvalModel) ModelType() string     { return "capture-test" }
func (captureEvalModel) Info() spine.ModelInfo { return spine.ModelInfo{} }
func (captureEvalModel) Generate(prompt string, _ ...spine.GenerateOption) (string, error) {
	return "echo:" + prompt, nil
}

// TestCapture_EvalLoopCapturesWithoutCascade drives the SFT eval loop with the
// score cascade OFF and asserts every raw generation still lands in the capture
// sidecar as a well-formed CaptureRow (step, prompt, raw text, and a non-zero
// birth timestamp). Capture is score-independent: a missed capture is a data
// point that never existed.
func TestCapture_EvalLoopCapturesWithoutCascade(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "captures.jsonl")
	cfg := SFTConfig{
		EvalEvery:          1,
		EvalPrompts:        []string{"q1", "q2"},
		CaptureSidecarPath: path,
	}
	result := &SFTResult{Steps: 1}
	if err := runSFTEvaluations(context.Background(), captureEvalModel{}, cfg, result); err != nil {
		t.Fatalf("runSFTEvaluations: %v", err)
	}
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("capture read: %v", err)
	}
	var row CaptureRow
	first := read[:core.Index(read, "\n")]
	if r := core.JSONUnmarshal([]byte(first), &row); !r.OK {
		t.Fatalf("capture row parse: %v", r.Value)
	}
	if row.Step != 1 || row.Prompt != "q1" || row.Text != "echo:q1" || row.At == 0 {
		t.Fatalf("capture row = %+v", row)
	}
}

// TestCapture_RowJSONShape asserts the CaptureRow serialises to the documented
// JSONL schema — the durable record shape the scorer reads later (archaeology
// over captured text is always legitimate).
func TestCapture_RowJSONShape(t *testing.T) {
	row := CaptureRow{Step: 7, Prompt: "hold a truth", Text: "I look at it straight.", At: 1700000000}
	encoded := core.JSONMarshal(row)
	if !encoded.OK {
		t.Fatalf("JSONMarshal(CaptureRow) failed: %v", encoded.Value)
	}
	var back CaptureRow
	if r := core.JSONUnmarshal(encoded.Value.([]byte), &back); !r.OK {
		t.Fatalf("JSONUnmarshal(CaptureRow) failed: %v", r.Value)
	}
	if back != row {
		t.Fatalf("round-tripped CaptureRow = %+v, want %+v", back, row)
	}
}
