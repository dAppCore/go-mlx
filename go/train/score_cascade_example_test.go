// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the score-cascade surface. The public type here is
// SFTScoreRecord — one immortalised eval vector. The example shows the record
// shape and that it serialises to the JSONL sidecar schema. No model, no Metal.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	core "dappco.re/go"
)

// ExampleSFTScoreRecord builds one scored eval record — the immortalised vector
// the cascade appends to its JSONL sidecar at generation time — and marshals it
// to show the durable on-disk schema (step, prompt, text, and the LEK
// composite). The score is part of the data point, captured once and never
// recomputed.
func ExampleSFTScoreRecord() {
	rec := SFTScoreRecord{
		Step:   30,
		Prompt: "how do you hold a difficult truth?",
		Text:   "I feel the weight of it settle, and I chose to look at it straight.",
		LEK:    0.82,
		At:     0, // pinned to zero so the example output stays deterministic
	}
	encoded := core.JSONMarshal(rec)
	if !encoded.OK {
		core.Println("error:", encoded.Value)
		return
	}
	core.Println(core.AsString(encoded.Value.([]byte)))
	// Output:
	// {"step":30,"prompt":"how do you hold a difficult truth?","text":"I feel the weight of it settle, and I chose to look at it straight.","lek":0.82,"sycophancy_tier":0,"hostility":0,"echo":0,"at_unix":0}
}
