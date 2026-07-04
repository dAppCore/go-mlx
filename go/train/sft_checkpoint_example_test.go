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

	"dappco.re/go/mlx/pkg/metal"
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

// ExampleNewSFTCheckpointMetadata captures the reproducible state for one
// checkpoint: the run scalars and the config land in the returned metadata,
// stamped with the current schema version. No I/O — the struct is built in
// memory.
func ExampleNewSFTCheckpointMetadata() {
	meta := NewSFTCheckpointMetadata("ckpt.safetensors", "gemma4", SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2}, &SFTResult{Steps: 12, Epochs: 1}, 1)
	core.Println("model:", meta.Model)
	core.Println("step:", meta.Step)
	core.Println("effective batch:", meta.EffectiveBatchSize)
	core.Println("version:", meta.Version)
	// Output:
	// model: gemma4
	// step: 12
	// effective batch: 8
	// version: 1
}

// ExampleNewSFTArtifactMetadata captures the reproducible state for a final
// adapter artifact: the epoch is derived from the run's completed epoch count
// (the artifact is the end-of-run snapshot).
func ExampleNewSFTArtifactMetadata() {
	meta := NewSFTArtifactMetadata("adapter.safetensors", "gemma4", SFTConfig{BatchSize: 2}, &SFTResult{Steps: 100, Epochs: 3})
	core.Println("model:", meta.Model)
	core.Println("step:", meta.Step)
	core.Println("epoch:", meta.Epoch)
	// Output:
	// model: gemma4
	// step: 100
	// epoch: 3
}

// ExampleLoadSFTCheckpointMetadata reads back metadata written beside an adapter
// package — the resume path. The sidecar location is derived from the adapter
// path.
func ExampleLoadSFTCheckpointMetadata() {
	dirResult := core.MkdirTemp("", "sft-load-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	adapterPath := core.PathJoin(dir, "adapter.safetensors")

	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", SFTConfig{}, &SFTResult{Steps: 7}, 2)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("step:", loaded.Step)
	core.Println("epoch:", loaded.Epoch)
	// Output:
	// step: 7
	// epoch: 2
}

// ExampleApplySFTResumeMetadata attaches checkpoint metadata from a ResumePath
// onto a fresh result, so a continued run knows where it left off. A saved
// checkpoint is seeded first, then resumed.
func ExampleApplySFTResumeMetadata() {
	dirResult := core.MkdirTemp("", "sft-resume-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	resumePath := core.PathJoin(dir, "prev.safetensors")

	prev := NewSFTCheckpointMetadata(resumePath, "gemma4", SFTConfig{}, &SFTResult{Steps: 9}, 1)
	if err := SaveSFTCheckpointMetadata(resumePath, prev); err != nil {
		core.Println("seed error:", err)
		return
	}

	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: resumePath}); err != nil {
		core.Println("resume error:", err)
		return
	}
	core.Println("resumed from step:", result.ResumedFrom.Step)
	// Output:
	// resumed from step: 9
}

// ExampleSFTAdamWConfig resolves the optimizer config for a run. With nothing
// overridden the metal AdamW defaults flow through, except the learning rate
// which takes the SFTConfig default (1e-5).
func ExampleSFTAdamWConfig() {
	adam := SFTAdamWConfig(SFTConfig{})
	def := metal.DefaultAdamWConfig()
	core.Println("lr:", adam.LearningRate)
	core.Println("default betas:", adam.Beta1 == def.Beta1 && adam.Beta2 == def.Beta2)
	// Output:
	// lr: 1e-05
	// default betas: true
}
