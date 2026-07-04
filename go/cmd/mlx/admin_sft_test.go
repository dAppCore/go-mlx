// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
)

func TestAdminSFTDatasetConfig_Gemma4LargeMessagesUseSharedFormatter_Good(t *testing.T) {
	input := `{"messages":[{"role":"user","content":"Write one line."},{"role":"assistant","content":"ok"}]}`
	cfg := adminSFTDatasetConfig(mlx.ModelInfo{Architecture: "gemma4_text", NumHeads: 16})

	ds, err := dataset.LoadJSONL(strings.NewReader(input), cfg)
	if err != nil {
		t.Fatalf("LoadJSONL() error = %v", err)
	}
	sample, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok {
		t.Fatal("Next() ok = false, want sample")
	}

	wantPrompt := chat.Format([]inference.Message{{Role: "user", Content: "Write one line."}}, chat.Config{
		Architecture:   "gemma4_text",
		EnableThinking: true,
		LargeVariant:   true,
	})
	if sample.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want shared Gemma4 formatter %q", sample.Prompt, wantPrompt)
	}
	if sample.Response != "ok" {
		t.Fatalf("Response = %q, want assistant message", sample.Response)
	}
}

// ---- pure helpers ----------------------------------------------------

// TestAdminSFT_PickInt_Good — a positive value wins over the fallback.
func TestAdminSFT_PickInt_Good(t *testing.T) {
	if got := pickInt(7, 3); got != 7 {
		t.Fatalf("pickInt(7, 3) = %d, want 7", got)
	}
}

// TestAdminSFT_PickInt_Bad — zero (the "unset" wire value) falls back.
func TestAdminSFT_PickInt_Bad(t *testing.T) {
	if got := pickInt(0, 16); got != 16 {
		t.Fatalf("pickInt(0, 16) = %d, want fallback 16", got)
	}
}

// TestAdminSFT_PickInt_Ugly — a negative value is treated as unset and
// falls back rather than poisoning the SFTConfig with a nonsense knob.
func TestAdminSFT_PickInt_Ugly(t *testing.T) {
	if got := pickInt(-5, 8); got != 8 {
		t.Fatalf("pickInt(-5, 8) = %d, want fallback 8", got)
	}
}

// TestAdminSFT_PickFloat_Good — a positive learning rate wins.
func TestAdminSFT_PickFloat_Good(t *testing.T) {
	if got := pickFloat(2e-4, 1e-4); got != 2e-4 {
		t.Fatalf("pickFloat(2e-4, 1e-4) = %g, want 2e-4", got)
	}
}

// TestAdminSFT_PickFloat_Bad — zero falls back to the recipe default.
func TestAdminSFT_PickFloat_Bad(t *testing.T) {
	if got := pickFloat(0, 1e-4); got != 1e-4 {
		t.Fatalf("pickFloat(0, 1e-4) = %g, want fallback 1e-4", got)
	}
}

// TestAdminSFT_PickFloat_Ugly — a negative LR is unset, not a request to
// train downhill backwards; the fallback is used.
func TestAdminSFT_PickFloat_Ugly(t *testing.T) {
	if got := pickFloat(-0.5, 3e-4); got != 3e-4 {
		t.Fatalf("pickFloat(-0.5, 3e-4) = %g, want fallback 3e-4", got)
	}
}

// TestAdminSFT_DeriveAdapterName_Good — the model basename anchors the
// derived dir name so `ls` is readable, with a unix-seconds suffix.
func TestAdminSFT_DeriveAdapterName_Good(t *testing.T) {
	name := deriveAdapterName("/models/lemer-lite")
	if !strings.HasPrefix(name, "lemer-lite-") {
		t.Fatalf("deriveAdapterName = %q, want lemer-lite- prefix", name)
	}
	// suffix is a digit run (unix seconds) — at least one digit present.
	suffix := strings.TrimPrefix(name, "lemer-lite-")
	if suffix == "" || strings.ContainsAny(suffix, "/ ") {
		t.Fatalf("deriveAdapterName suffix = %q, want unix-seconds run", suffix)
	}
}

// TestAdminSFT_DeriveAdapterName_Bad — a trailing-slash path still yields
// the cleaned basename, not an empty segment.
func TestAdminSFT_DeriveAdapterName_Bad(t *testing.T) {
	name := deriveAdapterName("/models/gemma4/")
	if !strings.HasPrefix(name, "gemma4-") {
		t.Fatalf("deriveAdapterName(trailing slash) = %q, want gemma4- prefix", name)
	}
}

// TestAdminSFT_DeriveAdapterName_Ugly — a path that cleans to "." (an
// empty model path or a bare ".") falls back to the literal "adapter-"
// stem so the dir is never created at a degenerate name.
func TestAdminSFT_DeriveAdapterName_Ugly(t *testing.T) {
	for _, in := range []string{"", "."} {
		name := deriveAdapterName(in)
		if !strings.HasPrefix(name, "adapter-") {
			t.Fatalf("deriveAdapterName(%q) = %q, want adapter- fallback", in, name)
		}
	}
}

// TestAdminSFT_NewJobID_Good — the id carries the sft- prefix and is
// non-empty; the base36 nanosecond tail makes same-second collisions
// implausible.
func TestAdminSFT_NewJobID_Good(t *testing.T) {
	id := newJobID()
	if !strings.HasPrefix(id, "sft-") || len(id) <= len("sft-") {
		t.Fatalf("newJobID = %q, want non-empty sft- id", id)
	}
}

// TestAdminSFT_CloneSFTJob_Good — the clone is a deep copy: mutating the
// clone's loss ring must not write through to the source.
func TestAdminSFT_CloneSFTJob_Good(t *testing.T) {
	src := &adminSFTJob{
		JobID: "sft-x",
		State: adminSFTStateRunning,
		Loss:  []adminSFTLossSample{{Step: 1, Loss: 0.5}},
	}
	clone := cloneSFTJob(src)
	if clone == src {
		t.Fatal("cloneSFTJob returned the same pointer")
	}
	clone.Loss[0].Loss = 9.9
	if src.Loss[0].Loss != 0.5 {
		t.Fatalf("source loss mutated through clone: got %g, want 0.5", src.Loss[0].Loss)
	}
}

// TestAdminSFT_CloneSFTJob_Bad — cloning nil yields nil, never a panic.
func TestAdminSFT_CloneSFTJob_Bad(t *testing.T) {
	if got := cloneSFTJob(nil); got != nil {
		t.Fatalf("cloneSFTJob(nil) = %#v, want nil", got)
	}
}

// TestAdminSFT_CloneSFTJob_Ugly — the live cancel func must be dropped on
// the snapshot so a caller holding the copy can't cancel the real job.
func TestAdminSFT_CloneSFTJob_Ugly(t *testing.T) {
	called := false
	src := &adminSFTJob{JobID: "sft-y", cancel: func() { called = true }}
	clone := cloneSFTJob(src)
	if clone.cancel != nil {
		t.Fatal("clone retained the cancel func; snapshot must not be able to cancel the job")
	}
	if called {
		t.Fatal("cancel invoked during clone")
	}
}

// TestAdminSFT_DirSizeBytes_Good — sums regular-file bytes under a dir.
func TestAdminSFT_DirSizeBytes_Good(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.bin"), []byte("hello"), 0o600); err != nil {
		t.Fatalf("write a.bin: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.bin"), []byte("world!"), 0o600); err != nil {
		t.Fatalf("write b.bin: %v", err)
	}
	if got := dirSizeBytes(dir); got != int64(len("hello")+len("world!")) {
		t.Fatalf("dirSizeBytes = %d, want %d", got, len("hello")+len("world!"))
	}
}

// TestAdminSFT_DirSizeBytes_Bad — a non-existent dir collapses to 0 (the
// walk error is swallowed; the size column is best-effort).
func TestAdminSFT_DirSizeBytes_Bad(t *testing.T) {
	if got := dirSizeBytes(filepath.Join(t.TempDir(), "does-not-exist")); got != 0 {
		t.Fatalf("dirSizeBytes(missing) = %d, want 0", got)
	}
}

// ---- registry --------------------------------------------------------

// TestAdminSFT_RegistrySnapshot_Good — snapshot("") returns a copy of the
// active job; the registry lock is released before the caller sees it.
func TestAdminSFT_RegistrySnapshot_Good(t *testing.T) {
	r := newAdminSFTRegistry()
	r.active = &adminSFTJob{JobID: "sft-active", State: adminSFTStateRunning}
	snap := r.snapshot("")
	if snap == nil || snap.JobID != "sft-active" {
		t.Fatalf("snapshot(\"\") = %#v, want active job", snap)
	}
	if snap == r.active {
		t.Fatal("snapshot returned the live pointer, not a copy")
	}
}

// TestAdminSFT_RegistrySnapshot_Bad — an unknown job_id resolves to nil
// rather than silently returning the active job.
func TestAdminSFT_RegistrySnapshot_Bad(t *testing.T) {
	r := newAdminSFTRegistry()
	r.active = &adminSFTJob{JobID: "sft-active"}
	if got := r.snapshot("sft-nope"); got != nil {
		t.Fatalf("snapshot(unknown) = %#v, want nil", got)
	}
}

// TestAdminSFT_RegistrySnapshot_Ugly — a finished job that has rolled
// from active to last is still resolvable by its id (Status-after-end).
func TestAdminSFT_RegistrySnapshot_Ugly(t *testing.T) {
	r := newAdminSFTRegistry()
	r.last = &adminSFTJob{JobID: "sft-done", State: adminSFTStateDone}
	snap := r.snapshot("sft-done")
	if snap == nil || snap.State != adminSFTStateDone {
		t.Fatalf("snapshot(last job) = %#v, want done job from last slot", snap)
	}
}

// TestAdminSFT_StateFlippers_Good — markRunning/markDone/markStopped/
// failJob each move the active job to the expected terminal state and
// stamp the timestamps.
func TestAdminSFT_StateFlippers_Good(t *testing.T) {
	t.Run("running", func(t *testing.T) {
		r := newAdminSFTRegistry()
		job := &adminSFTJob{JobID: "j", State: adminSFTStatePending}
		r.active = job
		r.markRunning(job)
		if job.State != adminSFTStateRunning {
			t.Fatalf("state = %q, want running", job.State)
		}
	})
	t.Run("done", func(t *testing.T) {
		r := newAdminSFTRegistry()
		job := &adminSFTJob{JobID: "j", State: adminSFTStateRunning}
		r.active = job
		r.markDone(job)
		if job.State != adminSFTStateDone || job.EndedUnix == 0 {
			t.Fatalf("done flip = %#v, want done + EndedUnix", job)
		}
	})
	t.Run("stopped", func(t *testing.T) {
		r := newAdminSFTRegistry()
		job := &adminSFTJob{JobID: "j", State: adminSFTStateRunning}
		r.active = job
		r.markStopped(job)
		if job.State != adminSFTStateStopped || job.EndedUnix == 0 {
			t.Fatalf("stopped flip = %#v, want stopped + EndedUnix", job)
		}
	})
	t.Run("failed", func(t *testing.T) {
		r := newAdminSFTRegistry()
		job := &adminSFTJob{JobID: "j", State: adminSFTStateRunning}
		r.active = job
		r.failJob(job, "load model: boom")
		if job.State != adminSFTStateFailed || job.Error != "load model: boom" {
			t.Fatalf("fail flip = %#v, want failed + reason", job)
		}
	})
}

// TestAdminSFT_StateFlippers_Bad — a flipper aimed at a job that is no
// longer the active job is a no-op (late event after the slot rolled).
func TestAdminSFT_StateFlippers_Bad(t *testing.T) {
	r := newAdminSFTRegistry()
	r.active = &adminSFTJob{JobID: "current", State: adminSFTStateRunning}
	stale := &adminSFTJob{JobID: "stale", State: adminSFTStateRunning}
	r.markDone(stale)
	if r.active.State != adminSFTStateRunning {
		t.Fatalf("active job perturbed by stale flip: %q", r.active.State)
	}
	if stale.State != adminSFTStateRunning {
		t.Fatalf("stale job mutated: %q, want untouched", stale.State)
	}
}

// ---- HTTP handlers ---------------------------------------------------

// TestAdminSFT_StartHandler_Good — a well-formed POST with an existing
// dataset claims the single-flight slot and returns 202 + a pending job
// snapshot. The background runner fails on the bogus model path, but the
// handler's own contract (validate → claim → 202) is what we assert here;
// adapterRoot is redirected to a temp HOME so no real dir is touched.
func TestAdminSFT_StartHandler_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dsPath := filepath.Join(t.TempDir(), "train.jsonl")
	if err := os.WriteFile(dsPath, []byte(`{"messages":[]}`+"\n"), 0o600); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	reg := newAdminSFTRegistry()
	body := `{"model_path":"/no/such/model","dataset_path":"` + dsPath + `"}`
	req := httptest.NewRequest(http.MethodPost, adminPathSFTStart, strings.NewReader(body))
	rr := httptest.NewRecorder()
	adminSFTStartHandler(reg).ServeHTTP(rr, req)

	if rr.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202; body=%q", rr.Code, rr.Body.String())
	}
	var job adminSFTJob
	if err := json.Unmarshal(rr.Body.Bytes(), &job); err != nil {
		t.Fatalf("decode job: %v", err)
	}
	if job.JobID == "" || job.ModelPath != "/no/such/model" {
		t.Fatalf("job snapshot = %#v, want job_id + model_path", job)
	}
	// The handler launched runSFTJob in a goroutine. Cancel it and wait
	// for the registry to roll the slot (the bogus model path makes
	// LoadModel fail fast at the file probe — no GPU work). Waiting keeps
	// the goroutine from outliving the test and touching torn-down state.
	reg.mu.Lock()
	if reg.active != nil && reg.active.cancel != nil {
		reg.active.cancel()
	}
	reg.mu.Unlock()
	deadline := time.Now().Add(5 * time.Second)
	for {
		reg.mu.RLock()
		done := reg.active == nil
		reg.mu.RUnlock()
		if done {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("background SFT job did not finish within 5s")
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// TestAdminSFT_StartHandler_Bad — the validation matrix: wrong method,
// malformed JSON, missing model_path, missing dataset_path, and a
// dataset path that doesn't exist on disk all reject before any slot is
// claimed.
func TestAdminSFT_StartHandler_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cases := []struct {
		name   string
		method string
		body   string
		want   int
	}{
		{"wrong method", http.MethodGet, "", http.StatusMethodNotAllowed},
		{"malformed json", http.MethodPost, "{not json", http.StatusBadRequest},
		{"missing model", http.MethodPost, `{"dataset_path":"/tmp/x"}`, http.StatusBadRequest},
		{"missing dataset", http.MethodPost, `{"model_path":"/m"}`, http.StatusBadRequest},
		{"dataset not found", http.MethodPost, `{"model_path":"/m","dataset_path":"/no/such/ds.jsonl"}`, http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := newAdminSFTRegistry()
			req := httptest.NewRequest(tc.method, adminPathSFTStart, strings.NewReader(tc.body))
			rr := httptest.NewRecorder()
			adminSFTStartHandler(reg).ServeHTTP(rr, req)
			if rr.Code != tc.want {
				t.Fatalf("status = %d, want %d; body=%q", rr.Code, tc.want, rr.Body.String())
			}
			if reg.active != nil {
				t.Fatalf("slot claimed despite rejection (%s)", tc.name)
			}
		})
	}
}

// TestAdminSFT_StartHandler_Ugly — when a job already holds the slot, a
// second Start is rejected with 409 Conflict and the in-flight job is
// left untouched (single-flight guarantee).
func TestAdminSFT_StartHandler_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dsPath := filepath.Join(t.TempDir(), "train.jsonl")
	if err := os.WriteFile(dsPath, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	reg := newAdminSFTRegistry()
	reg.active = &adminSFTJob{JobID: "incumbent", State: adminSFTStateRunning}
	body := `{"model_path":"/m","dataset_path":"` + dsPath + `"}`
	req := httptest.NewRequest(http.MethodPost, adminPathSFTStart, strings.NewReader(body))
	rr := httptest.NewRecorder()
	adminSFTStartHandler(reg).ServeHTTP(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409", rr.Code)
	}
	if reg.active.JobID != "incumbent" {
		t.Fatalf("incumbent job replaced: %q", reg.active.JobID)
	}
}

// TestAdminSFT_StatusHandler_Good — GET with the active job's id returns
// 200 + the snapshot.
func TestAdminSFT_StatusHandler_Good(t *testing.T) {
	reg := newAdminSFTRegistry()
	reg.active = &adminSFTJob{JobID: "sft-live", State: adminSFTStateRunning, Step: 4}
	req := httptest.NewRequest(http.MethodGet, adminPathSFTStatus+"?job=sft-live", nil)
	rr := httptest.NewRecorder()
	adminSFTStatusHandler(reg).ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	var job adminSFTJob
	if err := json.Unmarshal(rr.Body.Bytes(), &job); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if job.JobID != "sft-live" || job.Step != 4 {
		t.Fatalf("job = %#v, want live snapshot", job)
	}
}

// TestAdminSFT_StatusHandler_Bad — GET when no job has ever run returns
// 404 (the empty registry resolves to no snapshot).
func TestAdminSFT_StatusHandler_Bad(t *testing.T) {
	reg := newAdminSFTRegistry()
	req := httptest.NewRequest(http.MethodGet, adminPathSFTStatus, nil)
	rr := httptest.NewRecorder()
	adminSFTStatusHandler(reg).ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rr.Code)
	}
}

// TestAdminSFT_StatusHandler_Ugly — a non-GET method is rejected 405.
func TestAdminSFT_StatusHandler_Ugly(t *testing.T) {
	reg := newAdminSFTRegistry()
	req := httptest.NewRequest(http.MethodPost, adminPathSFTStatus, nil)
	rr := httptest.NewRecorder()
	adminSFTStatusHandler(reg).ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}

// TestAdminSFT_StopHandler_Good — POST cancels the active job's context
// and returns 200 + the snapshot. The cancel func is invoked.
func TestAdminSFT_StopHandler_Good(t *testing.T) {
	reg := newAdminSFTRegistry()
	cancelled := false
	reg.active = &adminSFTJob{JobID: "sft-run", State: adminSFTStateRunning, cancel: func() { cancelled = true }}
	req := httptest.NewRequest(http.MethodPost, adminPathSFTStop+"?job=sft-run", nil)
	rr := httptest.NewRecorder()
	adminSFTStopHandler(reg).ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	if !cancelled {
		t.Fatal("cancel func not invoked")
	}
}

// TestAdminSFT_StopHandler_Bad — Stop for an id that isn't the active job
// returns 404 and does not cancel anything.
func TestAdminSFT_StopHandler_Bad(t *testing.T) {
	reg := newAdminSFTRegistry()
	cancelled := false
	reg.active = &adminSFTJob{JobID: "sft-run", cancel: func() { cancelled = true }}
	req := httptest.NewRequest(http.MethodPost, adminPathSFTStop+"?job=other", nil)
	rr := httptest.NewRecorder()
	adminSFTStopHandler(reg).ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rr.Code)
	}
	if cancelled {
		t.Fatal("cancelled a non-matching job")
	}
}

// TestAdminSFT_StopHandler_Ugly — a non-POST method is rejected 405.
func TestAdminSFT_StopHandler_Ugly(t *testing.T) {
	reg := newAdminSFTRegistry()
	req := httptest.NewRequest(http.MethodGet, adminPathSFTStop, nil)
	rr := httptest.NewRecorder()
	adminSFTStopHandler(reg).ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}

// TestAdminSFT_AdaptersHandler_Good — GET lists adapter dirs under
// adapterRoot (redirected via HOME) with name + size + mtime.
func TestAdminSFT_AdaptersHandler_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	root := filepath.Join(home, "Lethean", "data", "adapters")
	adapterDir := filepath.Join(root, "lemer-lite-123")
	if err := os.MkdirAll(adapterDir, 0o755); err != nil {
		t.Fatalf("mkdir adapter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(adapterDir, "adapter.safetensors"), []byte("weights"), 0o600); err != nil {
		t.Fatalf("write adapter file: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, adminPathSFTAdapters, nil)
	rr := httptest.NewRecorder()
	adminSFTAdaptersHandler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	var out struct {
		Dir      string `json:"dir"`
		Adapters []struct {
			Name      string `json:"name"`
			SizeBytes int64  `json:"size_bytes"`
		} `json:"adapters"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Adapters) != 1 || out.Adapters[0].Name != "lemer-lite-123" {
		t.Fatalf("adapters = %#v, want one named entry", out.Adapters)
	}
	if out.Adapters[0].SizeBytes != int64(len("weights")) {
		t.Fatalf("size = %d, want %d", out.Adapters[0].SizeBytes, len("weights"))
	}
}

// TestAdminSFT_AdaptersHandler_Bad — when the adapters dir has never been
// created (no SFT has ever run), the handler returns 200 with an empty
// list, not a 500.
func TestAdminSFT_AdaptersHandler_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // root won't exist
	req := httptest.NewRequest(http.MethodGet, adminPathSFTAdapters, nil)
	rr := httptest.NewRecorder()
	adminSFTAdaptersHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 empty-state", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), `"adapters":[]`) {
		t.Fatalf("body = %q, want empty adapters array", rr.Body.String())
	}
}

// TestAdminSFT_AdaptersHandler_Ugly — a non-GET method is rejected 405.
func TestAdminSFT_AdaptersHandler_Ugly(t *testing.T) {
	req := httptest.NewRequest(http.MethodDelete, adminPathSFTAdapters, nil)
	rr := httptest.NewRecorder()
	adminSFTAdaptersHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}
