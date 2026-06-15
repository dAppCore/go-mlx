// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	core "dappco.re/go"
)

// ExampleNewDistillCheckpointMetadata builds a metadata record from a
// config, a result and a loss, then reports the identity it captured —
// the step, the loss kind and the recorded scalar loss.
func ExampleNewDistillCheckpointMetadata() {
	result := &DistillResult{}
	result.Metrics.Steps = 10

	meta := NewDistillCheckpointMetadata(
		"checkpoints/step-000010",
		DistillConfig{Temperature: 2, Loss: DistillLossKL},
		result,
		DistillLoss{Value: 0.5, KL: 0.5},
		1,
	)
	core.Println(meta.Step, string(meta.LossKind), meta.Loss)
	// Output: 10 kl 0.5
}

// ExampleSaveDistillCheckpointMetadata writes a metadata sidecar beside a
// checkpoint directory and reports that the write succeeded.
func ExampleSaveDistillCheckpointMetadata() {
	made := core.MkdirTemp("", "distill-example-save")
	if !made.OK {
		core.Println(made.Value)
		return
	}
	dir := made.Value.(string)
	defer core.RemoveAll(dir)

	err := SaveDistillCheckpointMetadata(dir, DistillCheckpointMetadata{Step: 7, Loss: 0.25})
	core.Println(err == nil)
	// Output: true
}

// ExampleLoadDistillCheckpointMetadata writes a checkpoint via Save and then
// reads it back, showing the round-trip preserves the recorded step.
func ExampleLoadDistillCheckpointMetadata() {
	made := core.MkdirTemp("", "distill-example-load")
	if !made.OK {
		core.Println(made.Value)
		return
	}
	dir := made.Value.(string)
	defer core.RemoveAll(dir)

	if err := SaveDistillCheckpointMetadata(dir, DistillCheckpointMetadata{Step: 3, Loss: 0.1}); err != nil {
		core.Println(err.Error())
		return
	}
	meta, err := LoadDistillCheckpointMetadata(dir)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(meta.Step)
	// Output: 3
}
