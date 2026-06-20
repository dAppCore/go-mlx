// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// ---- serve.go resolveServeDraftBlock (the profile-lookup lane) --------------

// resolveServeDraftBlock, with an active drafter and no explicit flag block,
// reads the newest matching tuned profile and returns its block + a "tuned:"
// note. The machine-hash seam is mocked so the profile (written hash-less) is
// accepted.
func TestResolveServeDraftBlock_TunedProfileLookup_Good(t *testing.T) {
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{Labels: map[string]string{"machine_hash": "h1"}}, nil
	}

	dir := t.TempDir()
	modelPath := "/models/target"
	// Write a tuned profile for this model carrying the draft-block label.
	profile := inference.TuningProfile{}
	profile.Key.Model = inference.ModelIdentity{Path: modelPath}
	profile.Candidate.Labels = map[string]string{tuneDraftBlockLabel: "6"}
	profile.CreatedAtUnix = 1000
	if err := writeTuningProfile(core.PathJoin(dir, "p.json"), profile); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	active := mlx.DraftDetection{Source: mlx.DraftSourceAssistantDir, DraftPath: "/models/target/assistant"}
	block, note := resolveServeDraftBlock(context.Background(), active, modelPath, 0, false, dir)
	if block != 6 {
		t.Fatalf("block = %d, want 6 from the tuned profile", block)
	}
	if note == "" {
		t.Fatalf("note = %q, want a tuned-profile note", note)
	}
}

// resolveServeDraftBlock with an active drafter but no matching profile returns
// (0, "") — the lookup ran but found nothing.
func TestResolveServeDraftBlock_NoMatchingProfile_Good(t *testing.T) {
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("no hash")
	}
	active := mlx.DraftDetection{Source: mlx.DraftSourceAssistantDir, DraftPath: "/d"}
	block, note := resolveServeDraftBlock(context.Background(), active, "/models/nomatch", 0, false, t.TempDir())
	if block != 0 || note != "" {
		t.Fatalf("no-match = (%d, %q), want (0, \"\")", block, note)
	}
}

// ---- admin_reload.go pure helpers -------------------------------------------

// pathWithinDir accepts the root itself + children, and rejects siblings,
// escapes, and absolute outsiders.
func TestPathWithinDir_Coverage_Good(t *testing.T) {
	if !pathWithinDir("/m/models", "/m/models") {
		t.Fatal("root == itself should be contained")
	}
	if !pathWithinDir("/m/models", "/m/models/gemma") {
		t.Fatal("child should be contained")
	}
	if pathWithinDir("/m/models", "/m/models-evil") {
		t.Fatal("sibling must not be contained")
	}
	if pathWithinDir("/m/models", "/etc/passwd") {
		t.Fatal("absolute outsider must not be contained")
	}
}

// readModelManifest parses the shasum format, skipping comments + blanks, and
// errors on a missing manifest.
func TestReadModelManifest_Coverage_Good(t *testing.T) {
	dir := t.TempDir()
	if r := core.WriteFile(core.PathJoin(dir, ".sha256"), []byte(
		"# a comment\n\nabc123  model.safetensors\ndef456  tokenizer.json\n"), 0o644); !r.OK {
		t.Fatalf("write manifest: %v", r.Value)
	}
	m, err := readModelManifest(dir)
	if err != nil {
		t.Fatalf("readModelManifest: %v", err)
	}
	if m["model.safetensors"] != "abc123" || m["tokenizer.json"] != "def456" {
		t.Fatalf("manifest = %#v, want the two entries", m)
	}
	// Missing manifest errors.
	if _, err := readModelManifest(core.JoinPath(t.TempDir(), "no-dir")); err == nil {
		t.Fatal("missing manifest error = nil, want a read error")
	}
}

// bindModelPathToStandardDir rejects empty, not-found, and escaping paths, and
// accepts a model dir under the standard root that carries a sidecar.
func TestBindModelPathToStandardDir_Coverage_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if _, err := bindModelPathToStandardDir(""); err == nil {
		t.Fatal("empty path error = nil, want required error")
	}
	if _, err := bindModelPathToStandardDir(core.JoinPath(t.TempDir(), "absent")); err == nil {
		t.Fatal("not-found path error = nil, want not-found error")
	}
	// A model under the standard dir with a sidecar resolves.
	modelDir := core.PathJoin(standardModelDir(), "good-model")
	if r := core.MkdirAll(modelDir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(modelDir, ".sha256"), []byte("x  f\n"), 0o644); !r.OK {
		t.Fatalf("write sidecar: %v", r.Value)
	}
	got, err := bindModelPathToStandardDir(modelDir)
	if err != nil || got == "" {
		t.Fatalf("bind(valid) = (%q, %v), want the resolved path", got, err)
	}
	// A model dir WITHOUT a sidecar is refused.
	noSidecar := core.PathJoin(standardModelDir(), "no-sidecar")
	if r := core.MkdirAll(noSidecar, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if _, err := bindModelPathToStandardDir(noSidecar); err == nil {
		t.Fatal("no-sidecar bind error = nil, want the no-manifest refusal")
	}
}

// listKnownModels returns nil for a missing root and lists model dirs that
// carry a sidecar.
func TestListKnownModels_Coverage_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	// Missing root → nil.
	if got := listKnownModels(); got != nil {
		t.Fatalf("listKnownModels(missing root) = %v, want nil", got)
	}
	// Create a model dir with a sidecar + a stray dir without one + a file.
	withSidecar := core.PathJoin(standardModelDir(), "has-sidecar")
	if r := core.MkdirAll(withSidecar, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(withSidecar, ".sha256"), []byte("x  f\n"), 0o644); !r.OK {
		t.Fatalf("write sidecar: %v", r.Value)
	}
	if r := core.MkdirAll(core.PathJoin(standardModelDir(), "no-sidecar"), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(standardModelDir(), "stray.txt"), []byte("x"), 0o644); !r.OK {
		t.Fatalf("write stray: %v", r.Value)
	}
	got := listKnownModels()
	if len(got) != 1 || got[0] != "has-sidecar" {
		t.Fatalf("listKnownModels = %v, want only the sidecar-bearing model", got)
	}
}

// ---- ssd_eval.go applySSDEvalSamplingParams (all key branches) --------------

// applySSDEvalSamplingParams applies each supported key and reports the right
// errors for malformed entries + unknown keys.
func TestApplySSDEvalSamplingParams_Coverage_Good(t *testing.T) {
	base := mlx.DefaultSSDCodeBenchmarkConfig()

	cfg := base
	if err := applySSDEvalSamplingParams(&cfg, "temperature=0.7, ,top_p=0.9,top-k=40,min_p=0.05,max_tokens=128"); err != nil {
		t.Fatalf("apply(valid) error = %v", err)
	}
	if cfg.Generate.TopK != 40 || cfg.Generate.MaxTokens != 128 {
		t.Fatalf("cfg = %+v, want the parsed top_k/max_tokens", cfg.Generate)
	}
	// temp alias.
	cfg = base
	if err := applySSDEvalSamplingParams(&cfg, "temp=0.5"); err != nil {
		t.Fatalf("apply(temp alias) error = %v", err)
	}

	// Empty input is a no-op.
	cfg = base
	if err := applySSDEvalSamplingParams(&cfg, "  "); err != nil {
		t.Fatalf("apply(empty) error = %v", err)
	}

	for _, bad := range []string{
		"noseparator",
		"temperature=nan-ish-xyz",
		"top_p=bad",
		"top_k=bad",
		"min_p=bad",
		"max_tokens=bad",
		"unknown_key=1",
	} {
		cfg = base
		if err := applySSDEvalSamplingParams(&cfg, bad); err == nil {
			t.Fatalf("apply(%q) error = nil, want a parse/unknown error", bad)
		}
	}
}
