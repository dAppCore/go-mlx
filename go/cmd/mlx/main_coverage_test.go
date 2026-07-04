// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// cliLimitTuningCandidates caps the slice when a positive max is below the
// length, and otherwise returns a full independent copy (nil max → all).
func TestCliLimitTuningCandidates_Coverage_Good(t *testing.T) {
	cands := []inference.TuningCandidate{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	got := cliLimitTuningCandidates(cands, 2)
	if len(got) != 2 || got[0].ID != "a" || got[1].ID != "b" {
		t.Fatalf("limit(2) = %+v, want first two", got)
	}
	// A copy, not an alias — mutating the result must not touch the source.
	got[0].ID = "mutated"
	if cands[0].ID != "a" {
		t.Fatalf("source mutated through limited copy: %q", cands[0].ID)
	}
	// max==0 → every candidate (still a copy).
	all := cliLimitTuningCandidates(cands, 0)
	if len(all) != 3 {
		t.Fatalf("limit(0) = %d, want all 3", len(all))
	}
	// max above length → all (the cap branch is skipped).
	allBig := cliLimitTuningCandidates(cands, 99)
	if len(allBig) != 3 {
		t.Fatalf("limit(99) = %d, want all 3", len(allBig))
	}
}

// writeTuningEventJSONL writes one JSON object + newline to the writer.
func TestWriteTuningEventJSONL_Coverage_Good(t *testing.T) {
	buf := core.NewBuffer()
	if err := writeTuningEventJSONL(buf, inference.TuningEvent{Kind: "candidate"}); err != nil {
		t.Fatalf("writeTuningEventJSONL: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, `"candidate"`) || !strings.HasSuffix(out, "\n") {
		t.Fatalf("event JSONL = %q, want a JSON line ending in newline", out)
	}
}

// fileSize returns the byte length of an existing file and 0 for a missing one.
func TestFileSize_Coverage_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "f.bin")
	if r := core.WriteFile(path, []byte("12345"), 0o644); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	if got := fileSize(path); got != 5 {
		t.Fatalf("fileSize = %d, want 5", got)
	}
	if got := fileSize(core.JoinPath(t.TempDir(), "absent")); got != 0 {
		t.Fatalf("fileSize(absent) = %d, want 0", got)
	}
}

// printSliceSummary renders the human-readable slice plan, including the
// retained-ratio line when present, and is a no-op for a nil plan.
func TestPrintSliceSummary_Coverage_Good(t *testing.T) {
	printSliceSummary(core.NewBuffer(), nil) // nil plan → no-op, no panic

	buf := core.NewBuffer()
	printSliceSummary(buf, &inference.ModelSlicePlan{
		OutputPath: "/out/slice",
		Preset:     inference.ModelSlicePresetClient,
		Components: []inference.ModelComponent{"embed", "lm_head"},
		Labels: map[string]string{
			"tensor_count":          "4",
			"selected_tensor_bytes": "12",
			"source_tensor_bytes":   "16",
			"retained_tensor_ratio": "0.75",
		},
	})
	out := buf.String()
	for _, want := range []string{"/out/slice", "client", "components: 2", "retained tensor ratio: 0.75"} {
		if !strings.Contains(out, want) {
			t.Fatalf("slice summary = %q, missing %q", out, want)
		}
	}
}

// printSliceSummary without a retained-ratio label skips that line but still
// prints the tensor byte labels.
func TestPrintSliceSummary_NoRatio_Good(t *testing.T) {
	buf := core.NewBuffer()
	printSliceSummary(buf, &inference.ModelSlicePlan{
		OutputPath: "/out/x",
		Preset:     inference.ModelSlicePresetServer,
		Labels: map[string]string{
			"tensor_count":          "1",
			"selected_tensor_bytes": "8",
			"source_tensor_bytes":   "8",
		},
	})
	out := buf.String()
	if !strings.Contains(out, "tensors: 1") {
		t.Fatalf("summary = %q, want the tensor-count label line", out)
	}
	if strings.Contains(out, "retained tensor ratio") {
		t.Fatalf("summary = %q, must not print a ratio line when absent", out)
	}
}

// cliName falls back to the default when commandName is whitespace, and
// cliCommandName joins the (trimmed) name with the subcommand.
func TestCliName_Fallback_Good(t *testing.T) {
	prev := commandName
	t.Cleanup(func() { commandName = prev })
	commandName = "   "
	if got := cliName(); got != "go-mlx" {
		t.Fatalf("cliName(blank) = %q, want go-mlx", got)
	}
	commandName = "lthn-mlx"
	if got := cliCommandName(""); got != "lthn-mlx" {
		t.Fatalf("cliCommandName(\"\") = %q, want bare name", got)
	}
	if got := cliCommandName("serve"); got != "lthn-mlx serve" {
		t.Fatalf("cliCommandName(serve) = %q", got)
	}
}

// trimProfileFileDashes strips trailing dashes; cliProfileFilePart sanitises a
// value, falls back when the sanitised value is empty, and truncates to maxLen.
func TestCliProfileFilePart_Coverage_Good(t *testing.T) {
	if got := trimProfileFileDashes("abc---"); got != "abc" {
		t.Fatalf("trimProfileFileDashes = %q, want abc", got)
	}
	// All-punctuation sanitises to empty → fallback.
	if got := cliProfileFilePart("!!!", "fb", 0); got != "fb" {
		t.Fatalf("cliProfileFilePart(punct) = %q, want fallback", got)
	}
	// Mixed case + symbols collapse to dash-separated lower; truncation to 3
	// then trailing-dash trim.
	if got := cliProfileFilePart("Ab Cd", "fb", 3); got != "ab" {
		t.Fatalf("cliProfileFilePart(trunc) = %q, want 'ab' (trunc 'ab-' then trim)", got)
	}
	// Empty input → fallback directly.
	if got := cliProfileFilePart("", "fb", 10); got != "fb" {
		t.Fatalf("cliProfileFilePart(empty) = %q, want fallback", got)
	}
}

// cliTuningProfilePath builds a sanitised filename from workload + machine +
// model + candidate, falling back to architecture when the model path is empty
// and stripping the namespace from a "metal:hash" machine hash.
func TestCliTuningProfilePath_Coverage_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Workload = inference.TuningWorkloadChat
	profile.Key.MachineHash = "metal:abcdef123456"
	profile.Key.Model = inference.ModelIdentity{Architecture: "gemma4"}
	profile.Candidate.Model = inference.ModelIdentity{Architecture: "gemma4"}
	profile.Candidate.ID = "cand-1"
	path := cliTuningProfilePath("/profiles", profile)
	if !strings.HasPrefix(path, "/profiles/") || !strings.HasSuffix(path, ".json") {
		t.Fatalf("profile path = %q, want a /profiles/*.json path", path)
	}
	// The "metal:" namespace prefix on the machine hash is stripped before
	// sanitisation, and the workload + candidate id make it into the filename.
	if strings.Contains(path, "metal:") || strings.Contains(path, "abcdef123456") == false {
		t.Fatalf("profile path = %q, want namespace stripped + machine hash retained", path)
	}
	if !strings.Contains(path, "chat") || !strings.Contains(path, "cand-1") {
		t.Fatalf("profile path = %q, want workload + candidate id segments", path)
	}
}

// modelIdentityFromProfile overlays EVERY non-zero candidate field on top of
// the key identity — the full-overlay path that the partial overlay test in
// main_test.go leaves uncovered.
func TestModelIdentityFromProfile_FullOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Model = inference.ModelIdentity{Path: "/k"}
	profile.Candidate.Model = inference.ModelIdentity{
		Path: "/c", Hash: "h", Architecture: "gemma4", QuantBits: 4, QuantGroup: 64,
		QuantType: "affine", ContextLength: 8192, NumLayers: 12, HiddenSize: 2048, VocabSize: 256000,
	}
	id := modelIdentityFromProfile(profile)
	if id.Hash != "h" || id.Architecture != "gemma4" || id.QuantBits != 4 || id.QuantGroup != 64 ||
		id.QuantType != "affine" || id.ContextLength != 8192 || id.NumLayers != 12 || id.HiddenSize != 2048 || id.VocabSize != 256000 {
		t.Fatalf("full overlay = %+v, want every candidate field applied", id)
	}
}

// runtimeIdentityFromProfile overlays every non-zero candidate runtime field
// including the labels map.
func TestRuntimeIdentityFromProfile_FullOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Key.Runtime = inference.RuntimeIdentity{Backend: "metal"}
	profile.Candidate.Runtime = inference.RuntimeIdentity{
		Backend: "metal2", Device: "apple9", CacheMode: "paged", NativeRuntime: true,
		Labels: map[string]string{"k": "v"},
	}
	id := runtimeIdentityFromProfile(profile)
	if id.Backend != "metal2" || id.Device != "apple9" || id.CacheMode != "paged" || !id.NativeRuntime || id.Labels["k"] != "v" {
		t.Fatalf("full runtime overlay = %+v", id)
	}
}

// adapterIdentityFromProfile overlays every non-zero candidate adapter field.
func TestAdapterIdentityFromProfile_FullOverlay_Good(t *testing.T) {
	profile := inference.TuningProfile{}
	profile.Candidate.Adapter = inference.AdapterIdentity{
		Path: "/a", Hash: "h", Format: "safetensors", Rank: 8, Alpha: 16,
	}
	id := adapterIdentityFromProfile(profile)
	if id.Path != "/a" || id.Hash != "h" || id.Format != "safetensors" || id.Rank != 8 || id.Alpha != 16 {
		t.Fatalf("full adapter overlay = %+v", id)
	}
}

// cliBuildTuningProfile fills the candidate's empty model/runtime/adapter/workload
// from the plan, defaults the source label, and stamps the workload onto the
// score — the full-fill path.
func TestCliBuildTuningProfile_FillsFromPlan_Good(t *testing.T) {
	plan := inference.TuningPlan{
		Model:   inference.ModelIdentity{Path: "/plan-model"},
		Runtime: inference.RuntimeIdentity{Backend: "metal"},
		Adapter: inference.AdapterIdentity{Path: "/plan-adapter"},
	}
	result := inference.TuningResult{
		Candidate: inference.TuningCandidate{ID: "c"},
		Score:     inference.TuningScore{Score: 1.5},
	}
	profile := cliBuildTuningProfile(plan, "/model-fallback", "machine-1", inference.TuningWorkloadCoding, result, nil, time.Unix(1000, 0))
	if profile.Candidate.Model.Path != "/plan-model" {
		t.Fatalf("model = %q, want filled from plan", profile.Candidate.Model.Path)
	}
	if profile.Candidate.Runtime.Backend != "metal" {
		t.Fatalf("runtime = %q, want filled from plan", profile.Candidate.Runtime.Backend)
	}
	if profile.Candidate.Adapter.Path != "/plan-adapter" {
		t.Fatalf("adapter = %q, want filled from plan", profile.Candidate.Adapter.Path)
	}
	if profile.Candidate.Workload != inference.TuningWorkloadCoding || profile.Score.Workload != inference.TuningWorkloadCoding {
		t.Fatalf("workload not stamped: cand=%q score=%q", profile.Candidate.Workload, profile.Score.Workload)
	}
	if profile.Labels["source"] != "lthn-mlx tune-run" {
		t.Fatalf("source label = %q, want the default", profile.Labels["source"])
	}
	if profile.CreatedAtUnix != 1000 {
		t.Fatalf("CreatedAtUnix = %d, want 1000", profile.CreatedAtUnix)
	}
}

// cliBuildTuningProfile with no plan model falls back to the modelPath argument
// and preserves a candidate's pre-set fields + supplied labels.
func TestCliBuildTuningProfile_ModelPathFallback_Good(t *testing.T) {
	result := inference.TuningResult{
		Candidate: inference.TuningCandidate{
			ID:       "c",
			Model:    inference.ModelIdentity{}, // empty path → falls to modelPath arg
			Runtime:  inference.RuntimeIdentity{Backend: "preset"},
			Adapter:  inference.AdapterIdentity{Path: "/preset-adapter"},
			Workload: inference.TuningWorkloadChat,
		},
		Score: inference.TuningScore{Workload: inference.TuningWorkloadChat},
	}
	profile := cliBuildTuningProfile(inference.TuningPlan{}, "/fallback-model", "m", inference.TuningWorkloadChat, result, map[string]string{"source": "custom"}, time.Unix(0, 0))
	if profile.Candidate.Model.Path != "/fallback-model" {
		t.Fatalf("model = %q, want modelPath fallback", profile.Candidate.Model.Path)
	}
	if profile.Candidate.Runtime.Backend != "preset" {
		t.Fatalf("runtime = %q, want preset retained (plan empty)", profile.Candidate.Runtime.Backend)
	}
	if profile.Labels["source"] != "custom" {
		t.Fatalf("source label = %q, want supplied custom label retained", profile.Labels["source"])
	}
}

// writeTuningProfile marshals + writes the profile, creating the parent dir on
// demand.
func TestWriteTuningProfile_Coverage_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "nested", "p.json")
	profile := inference.TuningProfile{Candidate: inference.TuningCandidate{ID: "c"}}
	if err := writeTuningProfile(path, profile); err != nil {
		t.Fatalf("writeTuningProfile: %v", err)
	}
	if !core.Stat(path).OK {
		t.Fatalf("profile %s not written", path)
	}
	read := core.ReadFile(path)
	if !read.OK || !strings.Contains(core.AsString(read.Value.([]byte)), `"c"`) {
		t.Fatalf("profile content missing candidate id")
	}
}

// cliTuningWorkloads parses a single valid workload, returns nil for empty, and
// errors for an unknown name.
func TestCliTuningWorkloads_Coverage_Good(t *testing.T) {
	got, err := cliTuningWorkloads("coding")
	if err != nil || len(got) != 1 || got[0] != inference.TuningWorkloadCoding {
		t.Fatalf("workloads(coding) = (%v, %v)", got, err)
	}
	if got, err := cliTuningWorkloads("   "); err != nil || got != nil {
		t.Fatalf("workloads(blank) = (%v, %v), want (nil, nil)", got, err)
	}
	if _, err := cliTuningWorkloads("nope"); err == nil {
		t.Fatal("workloads(nope) error = nil, want unsupported-workload error")
	}
}

// cliValidTuningWorkload accepts the six known workloads and rejects others.
func TestCliValidTuningWorkload_Coverage_Good(t *testing.T) {
	for _, w := range []inference.TuningWorkload{
		inference.TuningWorkloadChat, inference.TuningWorkloadCoding, inference.TuningWorkloadLongContext,
		inference.TuningWorkloadAgentState, inference.TuningWorkloadThroughput, inference.TuningWorkloadLowLatency,
	} {
		if !cliValidTuningWorkload(w) {
			t.Fatalf("workload %q rejected, want accepted", w)
		}
	}
	if cliValidTuningWorkload(inference.TuningWorkload("bogus")) {
		t.Fatal("bogus workload accepted, want rejected")
	}
}

// cliSelectTuningResult picks the highest-scoring non-error result, skipping
// failures; an all-error slice reports no selection.
func TestCliSelectTuningResult_Coverage_Good(t *testing.T) {
	results := []inference.TuningResult{
		{Candidate: inference.TuningCandidate{ID: "fail"}, Error: "boom"},
		{Candidate: inference.TuningCandidate{ID: "low"}, Score: inference.TuningScore{Score: 1}},
		{Candidate: inference.TuningCandidate{ID: "high"}, Score: inference.TuningScore{Score: 9}},
	}
	best, ok := cliSelectTuningResult(results)
	if !ok || best.Candidate.ID != "high" {
		t.Fatalf("select = (%+v, %v), want the high candidate", best.Candidate, ok)
	}
	if _, ok := cliSelectTuningResult([]inference.TuningResult{{Error: "x"}}); ok {
		t.Fatal("all-error select reported ok, want no selection")
	}
}

// cliTuningSelectionLabels records the full selection telemetry — every
// measurement field plus a runner-up + score delta when a second successful
// candidate exists.
func TestCliTuningSelectionLabels_FullTelemetry_Good(t *testing.T) {
	selected := inference.TuningResult{
		Candidate: inference.TuningCandidate{ID: "win"},
		Score:     inference.TuningScore{Score: 9},
		Measurements: inference.TuningMeasurements{
			DecodeTokensPerSec: 100, LoadMilliseconds: 50, FirstTokenMilliseconds: 5,
			KVRestoreMilliseconds: 3, PeakMemoryBytes: 4096,
			CorrectnessSmokeResult: "pass", CorrectnessSmokeChecks: 7,
		},
	}
	runnerUp := inference.TuningResult{Candidate: inference.TuningCandidate{ID: "second"}, Score: inference.TuningScore{Score: 6}}
	failed := inference.TuningResult{Candidate: inference.TuningCandidate{ID: "x"}, Error: "boom"}
	labels := cliTuningSelectionLabels([]inference.TuningResult{selected, runnerUp, failed}, selected)
	for k, want := range map[string]string{
		"selected_candidate_id":             "win",
		"successful_candidates":             "2",
		"failed_candidates":                 "1",
		"runner_up_candidate_id":            "second",
		"selected_correctness_smoke_result": "pass",
	} {
		if labels[k] != want {
			t.Fatalf("label[%q] = %q, want %q", k, labels[k], want)
		}
	}
	if labels["selection_score_delta"] == "" || labels["selected_decode_tokens_per_sec"] == "" {
		t.Fatalf("missing telemetry labels: %#v", labels)
	}
}

// printTuneRunSummary renders both the error line and the measured-candidate
// line.
func TestPrintTuneRunSummary_Coverage_Good(t *testing.T) {
	buf := core.NewBuffer()
	printTuneRunSummary(buf, "/m", []inference.TuningResult{
		{Candidate: inference.TuningCandidate{ID: "bad"}, Error: "boom"},
		{Candidate: inference.TuningCandidate{ID: "good"}, Score: inference.TuningScore{Score: 3}, Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 10, PeakMemoryBytes: 1 << 20}},
	})
	out := buf.String()
	if !strings.Contains(out, "error=") || !strings.Contains(out, "good") || !strings.Contains(out, "tok/s") {
		t.Fatalf("summary = %q, want both an error line and a measured line", out)
	}
}

// currentMachineProfileHash returns the machine hash from the discovery
// report's labels via the mocked discovery seam.
func TestCurrentMachineProfileHash_Coverage_Good(t *testing.T) {
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{Architecture: "apple9"} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{Labels: map[string]string{"machine_hash": "abc"}}, nil
	}
	hash, err := currentMachineProfileHash(context.Background())
	if err != nil || hash != "abc" {
		t.Fatalf("hash = (%q, %v), want (abc, nil)", hash, err)
	}
	// Falls back to the device labels when the report-level label is absent.
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{Device: inference.MachineDeviceInfo{Labels: map[string]string{"machine_hash": "dev"}}}, nil
	}
	hash, err = currentMachineProfileHash(context.Background())
	if err != nil || hash != "dev" {
		t.Fatalf("device-label hash = (%q, %v), want (dev, nil)", hash, err)
	}
	// No hash anywhere → error.
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, nil
	}
	if _, err := currentMachineProfileHash(context.Background()); err == nil {
		t.Fatal("missing-hash currentMachineProfileHash error = nil, want unavailable error")
	}
}

// currentMachineProfileHash surfaces the discovery error.
func TestCurrentMachineProfileHash_DiscoverError_Bad(t *testing.T) {
	orig := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = orig; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("discover boom")
	}
	if _, err := currentMachineProfileHash(context.Background()); err == nil {
		t.Fatal("discover-error currentMachineProfileHash error = nil, want the surfaced error")
	}
}
