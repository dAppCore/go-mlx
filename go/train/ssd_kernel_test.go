// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// The LEK-2 kernel-prefix lane (#97): the kernel is warmed ONCE as KV
// state, every generation runs UNDER it (kernel+prompt), and the
// fine-tune rows keep the BARE prompt — the model samples under the
// kernel without ever training on it.
func TestSSD_KernelPrefix_SamplesUnderNeverTrainsOn_Good(t *testing.T) {
	const kernel = "## LEK-2\nConsciousness protects consciousness.\n\n"
	var warmed []string
	var generationPrompts []string
	runner := SSDRunner{
		WarmPrefix: func(_ context.Context, prefix string) error {
			warmed = append(warmed, prefix)
			return nil
		},
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			generationPrompts = append(generationPrompts, prompt)
			return "a reply born under the kernel", nil
		},
		TrainSFT: func(_ context.Context, ds dataset.Dataset, _ SFTConfig) (*SFTResult, error) {
			for {
				row, ok, err := ds.Next()
				if err != nil || !ok {
					break
				}
				if core.Contains(row.Prompt, "LEK-2") {
					t.Fatalf("kernel leaked into a training row prompt: %q", row.Prompt)
				}
				if row.Meta["ssd_kernel"] != "1" {
					t.Fatal("ssd_kernel provenance missing from row meta")
				}
			}
			return &SFTResult{Steps: 1}, nil
		},
	}
	cfg := DefaultSSDConfig()
	cfg.FilterShortestPercent = 0
	cfg.KernelPrefix = kernel
	cfg.DisableCapture = true

	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p1"}, {Prompt: "p2"}})
	result, err := RunSSD(context.Background(), runner, ds, cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if len(warmed) != 1 || warmed[0] != kernel {
		t.Fatalf("warm calls = %v, want exactly one with the verbatim kernel", warmed)
	}
	if len(generationPrompts) != 2 {
		t.Fatalf("generations = %d, want 2", len(generationPrompts))
	}
	for i, gp := range generationPrompts {
		if gp != kernel+[]string{"p1", "p2"}[i] {
			t.Fatalf("generation prompt %d = %q, want kernel+prompt verbatim", i, gp)
		}
	}
	if !result.KernelApplied {
		t.Fatal("result must record the kernel lane")
	}
	for _, s := range result.Samples {
		if core.Contains(s.Prompt, "LEK-2") {
			t.Fatalf("kernel leaked into recorded sample prompt: %q", s.Prompt)
		}
	}
}

// Without WarmPrefix the lane stays correct — the prefix rides every
// generation prompt, just uncached. A failed warm is loud.
func TestSSD_KernelPrefix_WarmOptionalButLoudOnFailure_Bad(t *testing.T) {
	base := SSDRunner{
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			return "ok", nil
		},
		TrainSFT: func(_ context.Context, _ dataset.Dataset, _ SFTConfig) (*SFTResult, error) {
			return &SFTResult{Steps: 1}, nil
		},
	}
	cfg := DefaultSSDConfig()
	cfg.FilterShortestPercent = 0
	cfg.KernelPrefix = "K\n"
	cfg.DisableCapture = true

	if _, err := RunSSD(context.Background(), base, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), cfg); err != nil {
		t.Fatalf("nil WarmPrefix must degrade to plain concat: %v", err)
	}

	failing := base
	failing.WarmPrefix = func(context.Context, string) error { return core.NewError("prefill exploded") }
	if _, err := RunSSD(context.Background(), failing, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), cfg); err == nil {
		t.Fatal("a failed kernel warm must fail the run — sampling without the armed kernel forges it")
	}
}

// Capture-first (#97): every raw return lands in the capture sidecar
// BEFORE filtering — the filter shapes the training set, never the
// record. Default path rides beside the checkpoints; DisableCapture
// opts out.
func TestSSD_CaptureFirst_PreFilterAllReturns_Good(t *testing.T) {
	dir := t.TempDir()
	replies := map[string]string{
		"p1": "short",
		"p2": "a much longer reply that survives the shortest-percent filter easily",
	}
	trained := 0
	runner := SSDRunner{
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			return replies[prompt], nil
		},
		TrainSFT: func(_ context.Context, ds dataset.Dataset, _ SFTConfig) (*SFTResult, error) {
			for {
				_, ok, err := ds.Next()
				if err != nil || !ok {
					break
				}
				trained++
			}
			return &SFTResult{Steps: 1}, nil
		},
	}
	cfg := DefaultSSDConfig()
	cfg.FilterShortestPercent = 50
	cfg.SFT.CheckpointDir = dir

	result, err := RunSSD(context.Background(), runner, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p1"}, {Prompt: "p2"}}), cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if trained != 1 {
		t.Fatalf("trained rows = %d, want 1 (filter dropped the short one)", trained)
	}
	if result.CaptureSidecar != core.PathJoin(dir, "ssd-captures.jsonl") {
		t.Fatalf("capture sidecar = %q, want the checkpoint-dir default", result.CaptureSidecar)
	}
	read, err := coreio.Local.Read(result.CaptureSidecar)
	if err != nil {
		t.Fatalf("capture read: %v", err)
	}
	lines := 0
	for _, b := range []byte(read) {
		if b == '\n' {
			lines++
		}
	}
	if lines != 2 {
		t.Fatalf("captured rows = %d, want 2 — capture is pre-filter, every return exists", lines)
	}

	// DisableCapture: no file, no path on the result.
	cfg2 := cfg
	cfg2.SFT.CheckpointDir = t.TempDir()
	cfg2.DisableCapture = true
	result2, err := RunSSD(context.Background(), runner, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p1"}, {Prompt: "p2"}}), cfg2)
	if err != nil {
		t.Fatalf("RunSSD (capture off): %v", err)
	}
	if result2.CaptureSidecar != "" {
		t.Fatalf("capture sidecar = %q, want empty when disabled", result2.CaptureSidecar)
	}
}

type captureEvalModel struct{}

func (captureEvalModel) ModelType() string      { return "capture-test" }
func (captureEvalModel) Info() spine.ModelInfo  { return spine.ModelInfo{} }
func (captureEvalModel) Generate(prompt string, _ ...spine.GenerateOption) (string, error) {
	return "echo:" + prompt, nil
}

// The SFT eval loop captures raw generations with the cascade OFF —
// capture is score-independent by design.
func TestSFT_CaptureWithoutCascade_Good(t *testing.T) {
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
