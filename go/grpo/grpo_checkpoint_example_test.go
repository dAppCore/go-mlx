// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	core "dappco.re/go"
)

// ExampleNewGRPOCheckpointMetadata builds the portable checkpoint sidecar
// from a config + update. Group size defaults are normalised and the
// metadata is stamped experimental at the current version.
func ExampleNewGRPOCheckpointMetadata() {
	meta := NewGRPOCheckpointMetadata(
		"runs/step-000010",
		GRPOConfig{KLCoefficient: 0.1},
		nil,
		GRPOUpdate{Step: 10, Epoch: 1},
	)
	core.Println(meta.Version, meta.Experimental)
	core.Println(meta.Step, meta.GroupSize)
	// Output:
	// 1 true
	// 10 4
}

// Example_grpoStepName shows the checkpoint step-directory naming: the
// step index is zero-padded to six digits below 1e5 and rendered at its
// natural width once it overflows that range. These names lay out the
// per-step checkpoint directories on disk.
func Example_grpoStepName() {
	core.Println(grpoStepName(0))
	core.Println(grpoStepName(42))
	core.Println(grpoStepName(12345))
	core.Println(grpoStepName(1234567))
	// Output:
	// step-000000
	// step-000042
	// step-012345
	// step-1234567
}

// ExampleSaveGRPOCheckpointMetadata round-trips checkpoint metadata to a
// temporary directory: Save writes the sidecar JSON and Load reads it
// back, backfilling the version stamp.
func ExampleSaveGRPOCheckpointMetadata() {
	dir := core.PathJoin(core.TempDir(), "grpo_example_ckpt_"+core.Itoa(core.Getpid()))
	defer core.RemoveAll(dir)

	err := SaveGRPOCheckpointMetadata(dir, GRPOCheckpointMetadata{Step: 3, GroupSize: 4})
	core.Println(err == nil)

	loaded, err := LoadGRPOCheckpointMetadata(dir)
	core.Println(err == nil)
	core.Println(loaded.Step, loaded.Version, loaded.Experimental)
	// Output:
	// true
	// true
	// 3 1 true
}

// ExampleLoadGRPOCheckpointMetadata reads back a checkpoint sidecar
// written by Save. A sidecar persisted with version 0 is stamped at the
// current metadata version on load, so the returned metadata always
// carries a concrete version.
func ExampleLoadGRPOCheckpointMetadata() {
	dir := core.PathJoin(core.TempDir(), "grpo_example_load_"+core.Itoa(core.Getpid()))
	defer core.RemoveAll(dir)

	// Persist a checkpoint with an explicit step + group size; Save stamps
	// the version and experimental flag for us.
	saveErr := SaveGRPOCheckpointMetadata(dir, GRPOCheckpointMetadata{Step: 12, GroupSize: 8})
	core.Println(saveErr == nil)

	loaded, err := LoadGRPOCheckpointMetadata(dir)
	core.Println(err == nil)
	core.Println(loaded.Step, loaded.GroupSize)
	core.Println(loaded.Version, loaded.Experimental)
	// Output:
	// true
	// true
	// 12 8
	// 1 true
}
