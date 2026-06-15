// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for the SFT public surface — these double as usage docs
// (AX principle 2: comments as usage examples) and execute under `go test`
// because each carries an Output: comment. All are deterministic and load no
// model: synthetic tokenizer + t-free pure helpers only.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	core "dappco.re/go"
)

// ExampleSaveSFTCheckpointMetadata writes checkpoint metadata beside an
// adapter package and reads it back — the portable JSON sidecar that lets a
// run resume. The metadata path is derived from the adapter path.
func ExampleSaveSFTCheckpointMetadata() {
	dirResult := core.MkdirTemp("", "sft-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	adapterPath := core.PathJoin(dir, "adapter.safetensors")

	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", SFTConfig{BatchSize: 2}, &SFTResult{Steps: 5}, 1)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("model:", loaded.Model)
	core.Println("step:", loaded.Step)
	core.Println("epoch:", loaded.Epoch)
	// Output:
	// model: gemma4
	// step: 5
	// epoch: 1
}
