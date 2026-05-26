// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"io/fs"
	"net/http"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
)

// Admin HTTP API — the surface a higher-level orchestrator (lthn-desktop
// GUI, or Lemma's tool-calling) composes to express the "Lemma, try the
// new Qwen model" UX without operator gymnastics.
//
// Endpoints under /v1/admin/*:
//
//	GET  /v1/admin/machine            current machine identity (hash, hostname, runtime info)
//	GET  /v1/admin/profiles           list saved tuning profiles in standard dir
//	POST /v1/admin/auto-tune          kick off async auto-tune job, returns job id
//	GET  /v1/admin/auto-tune?job=ID   poll the job's status / result
//	POST /v1/admin/models/download    501 — needs shared HF downloader extraction
//	POST /v1/admin/serve/reload       501 — needs hot-swap runtime support
//
// v1 has NO auth. Endpoints are reachable by anyone who can connect to
// the serve --addr. For local serve (default 127.0.0.1:11434) this is
// the same machine only. For tunnel-exposed serve (Owlet's setup) this
// is the tunneled user. Future: Authorization: Bearer token check
// gated on a serve flag — defer until the tunnel setup proves the need.

const (
	adminPathMachine     = "/v1/admin/machine"
	adminPathProfiles    = "/v1/admin/profiles"
	adminPathAutoTune    = "/v1/admin/auto-tune"
	adminPathDownload    = "/v1/admin/models/download"
	adminPathReload      = "/v1/admin/serve/reload"
)

// adminMachineInfo is the response shape for GET /v1/admin/machine.
type adminMachineInfo struct {
	Hash      string `json:"hash"`
	Hostname  string `json:"hostname,omitempty"`
	Runtime   string `json:"runtime"`
	GoVersion string `json:"go_version,omitempty"`
	OS        string `json:"os,omitempty"`
	Arch      string `json:"arch,omitempty"`
	Time      int64  `json:"time"`
}

// adminProfilesList is the response shape for GET /v1/admin/profiles.
type adminProfilesList struct {
	Object   string                  `json:"object"`
	Dir      string                  `json:"dir"`
	Profiles []adminProfilesEntry    `json:"profiles"`
}

// adminProfilesEntry is one row in adminProfilesList — the metadata
// callers care about (workload, model, machine, score, created_at)
// without the full nested TuningProfile payload by default.
type adminProfilesEntry struct {
	Path           string  `json:"path"`
	Workload       string  `json:"workload,omitempty"`
	ModelPath      string  `json:"model_path,omitempty"`
	MachineHash    string  `json:"machine_hash,omitempty"`
	Score          float64 `json:"score,omitempty"`
	CandidateID    string  `json:"candidate_id,omitempty"`
	CreatedAtUnix  int64   `json:"created_at_unix,omitempty"`
	IsCurrentMatch bool    `json:"is_current_machine,omitempty"`
}

// adminAutoTuneRequest is the body shape for POST /v1/admin/auto-tune.
// All fields optional; defaults match the auto-tune CLI subcommand.
type adminAutoTuneRequest struct {
	Model         string `json:"model"`           // required
	Workload      string `json:"workload"`        // default "chat"
	MaxCandidates int    `json:"max_candidates"`  // 0 = planner decides
	MaxTokens     int    `json:"max_tokens"`      // 0 = bench default
	Runs          int    `json:"runs"`            // 0 = bench default
}

// adminAutoTuneJob is one entry in the in-process job registry, also
// the response shape for GET /v1/admin/auto-tune?job=ID. Status moves
// pending → running → done | failed. ProfilePath is set on done.
type adminAutoTuneJob struct {
	ID           string    `json:"id"`
	Status       string    `json:"status"`
	Model        string    `json:"model"`
	Workload     string    `json:"workload"`
	MachineHash  string    `json:"machine_hash,omitempty"`
	StartedAt    time.Time `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
	ProfilePath  string    `json:"profile_path,omitempty"`
	Error        string    `json:"error,omitempty"`
}

// Auto-tune resource caps — defend against adversarial request
// inputs that would monopolise the GPU or grow the registry without
// bound. Tuning is single-machine, GPU-bound; one job at a time is
// the only sane policy.
const (
	maxAutoTuneCandidates = 32
	maxAutoTuneTokens     = 4096
	maxAutoTuneRuns       = 10
	maxJobAge             = time.Hour
)

// adminJobRegistry holds in-process auto-tune job state. Jobs are
// lost on restart — that's acceptable for v1 because auto-tune is
// idempotent (re-trigger if needed). Future: persist via go-store.
//
// activeSlots is a counting semaphore (size 1) that caps concurrent
// kickoffs. ctx is the server-shutdown context; tuning goroutines
// derive their ctx from this so SIGTERM cancels in-flight runs
// instead of letting them burn GPU after the process is asked to
// quit. stderr is the audit-emit destination for kickoff lines.
type adminJobRegistry struct {
	mu          sync.Mutex
	jobs        map[string]*adminAutoTuneJob
	activeSlots chan struct{}
	ctx         context.Context
	stderr      io.Writer
}

func newAdminJobRegistry(ctx context.Context, stderr io.Writer) *adminJobRegistry {
	return &adminJobRegistry{
		jobs:        make(map[string]*adminAutoTuneJob),
		activeSlots: make(chan struct{}, 1),
		ctx:         ctx,
		stderr:      stderr,
	}
}

// tryAcquireSlot non-blocking acquires the auto-tune semaphore.
// Returns false when another job is already in flight; caller must
// refuse the new request with 429.
func (r *adminJobRegistry) tryAcquireSlot() bool {
	select {
	case r.activeSlots <- struct{}{}:
		return true
	default:
		return false
	}
}

// releaseSlot frees the auto-tune semaphore. Called from the
// goroutine that finishes a tuning job (success or failure).
func (r *adminJobRegistry) releaseSlot() {
	<-r.activeSlots
}

// prune evicts completed jobs older than maxJobAge. Caller MUST
// hold r.mu. In-flight jobs (status "pending" or "running") are
// never evicted — the semaphore caps the in-flight count at 1.
func (r *adminJobRegistry) prune() {
	cutoff := time.Now().Add(-maxJobAge)
	for id, job := range r.jobs {
		if job.Status == "pending" || job.Status == "running" {
			continue
		}
		if job.StartedAt.Before(cutoff) {
			delete(r.jobs, id)
		}
	}
}

// clampAutoTuneRequest caps user-supplied numeric ceilings to defend
// against resource-exhaustion via gigantic candidate/token/run counts.
func clampAutoTuneRequest(req adminAutoTuneRequest) adminAutoTuneRequest {
	if req.MaxCandidates > maxAutoTuneCandidates {
		req.MaxCandidates = maxAutoTuneCandidates
	}
	if req.MaxTokens > maxAutoTuneTokens {
		req.MaxTokens = maxAutoTuneTokens
	}
	if req.Runs > maxAutoTuneRuns {
		req.Runs = maxAutoTuneRuns
	}
	return req
}

// newAdminMux mounts the /v1/admin/* handlers. Returns a Handler that
// only knows the admin paths — compose with the openai mux via a
// root mux for end-to-end serve. ctx is the server-shutdown context
// (cancellation propagates into tuning goroutines); stderr is where
// admin-level audit lines emit.
func newAdminMux(ctx context.Context, stderr io.Writer) *http.ServeMux {
	mux := http.NewServeMux()
	registry := newAdminJobRegistry(ctx, stderr)

	mux.HandleFunc(adminPathMachine, adminMachineHandler)
	mux.HandleFunc(adminPathProfiles, adminProfilesHandler)
	mux.HandleFunc(adminPathAutoTune, func(w http.ResponseWriter, r *http.Request) {
		adminAutoTuneHandler(w, r, registry)
	})
	mux.HandleFunc(adminPathDownload, adminNotImplementedHandler("models/download", "needs shared HF downloader extraction from lthn/desktop pkg"))
	mux.HandleFunc(adminPathReload, adminNotImplementedHandler("serve/reload", "needs hot-swap runtime support in lthn-mlx; replace-plan computes the plan but applying it requires resolver-level work"))
	return mux
}

// adminMachineHandler answers GET /v1/admin/machine with the current
// machine identity. Used by orchestrators to decide which profiles
// belong to this machine + report on the runtime.
func adminMachineHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	hash, err := currentMachineProfileHash(r.Context())
	if err != nil {
		http.Error(w, "machine hash unavailable: "+err.Error(), http.StatusInternalServerError)
		return
	}
	info := adminMachineInfo{
		Hash:      hash,
		Hostname:  core.Env("HOSTNAME"),
		Runtime:   "go-mlx",
		GoVersion: core.Env("GO"),
		OS:        core.Env("OS"),
		Arch:      core.Env("ARCH"),
		Time:      time.Now().Unix(),
	}
	writeJSON(w, http.StatusOK, info)
}

// adminProfilesHandler answers GET /v1/admin/profiles with the list
// of profile files in ~/Lethean/data/profiles/. Profiles for the
// current machine are flagged via IsCurrentMatch — orchestrators
// reading the list can highlight matches without a second call.
func adminProfilesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	dir := standardProfileDir()
	currentHash, _ := currentMachineProfileHash(r.Context())

	resp := adminProfilesList{Object: "list", Dir: dir, Profiles: []adminProfilesEntry{}}

	entries := core.ReadDir(core.DirFS(dir), ".")
	if !entries.OK {
		writeJSON(w, http.StatusOK, resp) // empty list is a valid answer
		return
	}
	dirEntries, ok := entries.Value.([]fs.DirEntry)
	if !ok {
		writeJSON(w, http.StatusOK, resp)
		return
	}
	// Resolve dir once for the loop — used to bound resolved entry
	// paths so a symlink can't trick us into parsing a JSON file
	// outside the profiles tree.
	dirPrefix := dir
	if r := core.PathEvalSymlinks(dir); r.OK {
		dirPrefix = r.Value.(string)
	}
	if !core.HasSuffix(dirPrefix, "/") {
		dirPrefix += "/"
	}
	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		// Skip non-regular files (symlinks, devices, sockets, fifos)
		// — an attacker with FS write access to dir could otherwise
		// drop a symlink pointing at any JSON-shaped file on disk and
		// leak its ModelPath + MachineHash via this response.
		if !entry.Type().IsRegular() {
			continue
		}
		name := entry.Name()
		if !core.HasSuffix(name, ".json") {
			continue
		}
		path := core.PathJoin(dir, name)
		// Belt-and-braces: verify the resolved path stays under dir.
		// Defends against cwd-relative resolution differences or
		// dir-level path-aliasing the IsRegular check can't catch.
		if r := core.PathEvalSymlinks(path); !r.OK || !core.HasPrefix(r.Value.(string), dirPrefix) {
			continue
		}
		report, err := readTuneProfileReport(path)
		if err != nil || report.Profile == nil {
			continue
		}
		row := adminProfilesEntry{
			Path:           path,
			Workload:       string(report.Profile.Key.Workload),
			ModelPath:      report.ModelPath,
			MachineHash:    report.Profile.Key.MachineHash,
			Score:          report.Profile.Score.Score,
			CandidateID:    report.Profile.Candidate.ID,
			CreatedAtUnix:  report.Profile.CreatedAtUnix,
			IsCurrentMatch: currentHash != "" && report.Profile.Key.MachineHash == currentHash,
		}
		resp.Profiles = append(resp.Profiles, row)
	}
	writeJSON(w, http.StatusOK, resp)
}

// adminAutoTuneHandler dispatches POST (kickoff) and GET (poll).
// POST returns the new job's record; GET requires ?job=ID.
func adminAutoTuneHandler(w http.ResponseWriter, r *http.Request, registry *adminJobRegistry) {
	switch r.Method {
	case http.MethodGet:
		jobID := core.Trim(r.URL.Query().Get("job"))
		if jobID == "" {
			http.Error(w, "missing job id; use POST to kick off or GET ?job=<id> to poll", http.StatusBadRequest)
			return
		}
		registry.mu.Lock()
		job, ok := registry.jobs[jobID]
		registry.mu.Unlock()
		if !ok {
			http.Error(w, "job not found", http.StatusNotFound)
			return
		}
		writeJSON(w, http.StatusOK, job)
	case http.MethodPost:
		var req adminAutoTuneRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		if core.Trim(req.Model) == "" {
			http.Error(w, "model required", http.StatusBadRequest)
			return
		}
		workloads, err := cliTuningWorkloads(req.Workload)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(workloads) == 0 {
			workloads = []inference.TuningWorkload{inference.TuningWorkloadChat}
		}
		req = clampAutoTuneRequest(req)
		machineHash, err := currentMachineProfileHash(r.Context())
		if err != nil {
			http.Error(w, "machine hash: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if !registry.tryAcquireSlot() {
			http.Error(w, "auto-tune busy — another job in flight", http.StatusTooManyRequests)
			return
		}
		jobID := nowJobID()
		job := &adminAutoTuneJob{
			ID:          jobID,
			Status:      "pending",
			Model:       req.Model,
			Workload:    string(workloads[0]),
			MachineHash: machineHash,
			StartedAt:   time.Now().UTC(),
		}
		registry.mu.Lock()
		registry.prune()
		registry.jobs[jobID] = job
		registry.mu.Unlock()

		core.Print(registry.stderr, "%s admin: auto-tune kickoff job=%s model=%s machine=%s remote=%s",
			cliName(), jobID, req.Model, machineHash, r.RemoteAddr)

		go func() {
			defer registry.releaseSlot()
			runAutoTuneJob(job, req, workloads[0], registry)
		}()
		writeJSON(w, http.StatusAccepted, job)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// runAutoTuneJob is the background worker — mirrors runAutoTuneCommand
// from auto_tune.go but updates job state instead of printing. Runs in
// its own goroutine; callers poll via GET /v1/admin/auto-tune?job=ID.
func runAutoTuneJob(job *adminAutoTuneJob, req adminAutoTuneRequest, workload inference.TuningWorkload, registry *adminJobRegistry) {
	defer func() {
		registry.mu.Lock()
		finishedAt := time.Now().UTC()
		job.FinishedAt = &finishedAt
		registry.mu.Unlock()
	}()
	registry.mu.Lock()
	job.Status = "running"
	registry.mu.Unlock()

	ctx := registry.ctx
	defaultBench := bench.DefaultConfig()
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultBench.MaxTokens
	}
	runs := req.Runs
	if runs <= 0 {
		runs = defaultBench.Runs
	}

	plan, err := runPlanLocalTuning(ctx, inference.TuningPlanRequest{
		Model:     inference.ModelIdentity{Path: req.Model},
		Workloads: []inference.TuningWorkload{workload},
		Budget: inference.TuningBudget{
			MaxCandidates:     req.MaxCandidates,
			SmokeTokens:       maxTokens,
			Runs:              runs,
			AllowStateBench:   true,
			AllowModelReloads: true,
		},
	})
	if err != nil {
		registry.mu.Lock()
		job.Status = "failed"
		job.Error = "plan: " + err.Error()
		registry.mu.Unlock()
		return
	}
	candidates := cliLimitTuningCandidates(plan.Candidates, req.MaxCandidates)
	if len(candidates) == 0 {
		registry.mu.Lock()
		job.Status = "failed"
		job.Error = "no tuning candidates"
		registry.mu.Unlock()
		return
	}

	benchCfg := bench.DefaultConfig()
	benchCfg.Model = core.PathBase(req.Model)
	benchCfg.ModelPath = req.Model
	benchCfg.MaxTokens = maxTokens
	benchCfg.Runs = runs

	results, err := runLocalTuning(ctx, mlx.LocalTuningRunConfig{
		ModelPath:  req.Model,
		Workload:   workload,
		Candidates: candidates,
		Bench:      benchCfg,
	})
	if err != nil {
		registry.mu.Lock()
		job.Status = "failed"
		job.Error = err.Error()
		registry.mu.Unlock()
		return
	}
	selected, ok := cliSelectTuningResult(results)
	if !ok {
		registry.mu.Lock()
		job.Status = "failed"
		job.Error = "no successful tuning result"
		registry.mu.Unlock()
		return
	}
	selectionLabels := cliTuningSelectionLabels(results, selected)
	profile := cliBuildTuningProfile(plan, req.Model, job.MachineHash, workload, selected, selectionLabels, time.Now())
	profilePath := cliTuningProfilePath(standardProfileDir(), profile)
	if err := writeTuningProfile(profilePath, profile); err != nil {
		registry.mu.Lock()
		job.Status = "failed"
		job.Error = "write profile: " + err.Error()
		registry.mu.Unlock()
		return
	}
	registry.mu.Lock()
	job.Status = "done"
	job.ProfilePath = profilePath
	registry.mu.Unlock()
}

// adminNotImplementedHandler is the placeholder for /v1/admin/models/
// download + /v1/admin/serve/reload until their underlying mechanisms
// land. Returns 501 with a clear message naming what's blocking.
func adminNotImplementedHandler(name, blocker string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotImplemented, map[string]string{
			"endpoint": name,
			"status":   "not implemented",
			"blocker":  blocker,
		})
	}
}

// nowJobID returns a UTC nanosecond-based id. Sufficient for v1 in-
// process job tracking; collisions extremely improbable. Future:
// google/uuid if registry persists across restarts.
func nowJobID() string {
	return core.Sprintf("autotune-%d", time.Now().UTC().UnixNano())
}

// writeJSON is a small helper around core.JSONMarshal + http.ResponseWriter.
func writeJSON(w http.ResponseWriter, status int, v any) {
	encoded := core.JSONMarshal(v)
	w.Header().Set("content-type", "application/json")
	if !encoded.OK {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"marshal failed"}`))
		return
	}
	w.WriteHeader(status)
	_, _ = w.Write(encoded.Value.([]byte))
}

// readJSONBody decodes the request body into target via core.JSONUnmarshal.
// Body is capped at 64KB — legitimate admin payloads serialise to <1KB; the
// cap prevents memory-exhaustion DoS via adversarial multi-GB POST.
func readJSONBody(r *http.Request, target any) error {
	defer r.Body.Close()
	body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, 64*1024))
	if err != nil {
		return err
	}
	res := core.JSONUnmarshal(body, target)
	if !res.OK {
		return res.Value.(error)
	}
	return nil
}
