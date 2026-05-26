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
//	GET  /v1/admin/serve/status       snapshot of model + profile + applied config
//	POST /v1/admin/models/download    HF download into ~/Lethean/data/models/, allowlist-gated
//	GET  /v1/admin/models/download?job=ID  poll a download job
//	POST /v1/admin/serve/reload       hot-swap loaded model, confirmation + sha-manifest gated
//
// Bearer auth (admin_auth.go) gates /v1/admin/* on the lthn-mlx_-
// prefixed 256-bit token at ~/Lethean/data/admin.token (mode 0600).
// Reveal the token with `lthn-mlx serve --print-admin-token`; rotate
// with `--rotate-admin-token`. Middleware mounts at the rootMux layer
// in serve.go so inference paths (/v1/chat/completions, /v1/messages,
// etc.) pass through unauthenticated under the localhost / tunnel-
// trust model. Audit emit on every 401 surfaces brute-force attempts.

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

// adminJobRegistry holds in-process auto-tune job state. Persisted
// to persistPath as JSON after every mutation so jobs survive serve
// restarts — caller polling a job that started before the restart
// can still GET /v1/admin/auto-tune?job=ID and see a definitive
// status (jobs that were in flight at restart are restored as
// failed with a "serve restart" message).
//
// activeSlots is a counting semaphore (size 1) that caps concurrent
// kickoffs. ctx is the server-shutdown context; tuning goroutines
// derive their ctx from this so SIGTERM cancels in-flight runs
// instead of letting them burn GPU after the process is asked to
// quit. stderr is the audit-emit destination for kickoff lines.
// persistPath is the JSON snapshot path; empty disables persistence
// (test paths use this).
type adminJobRegistry struct {
	mu          sync.Mutex
	jobs        map[string]*adminAutoTuneJob
	activeSlots chan struct{}
	ctx         context.Context
	stderr      io.Writer
	persistPath string
}

// standardAdminJobsPath returns ~/Lethean/data/admin-jobs.json —
// the canonical persistence location, sibling of admin.token.
func standardAdminJobsPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "admin-jobs.json")
}

func newAdminJobRegistry(ctx context.Context, stderr io.Writer, persistPath string) *adminJobRegistry {
	r := &adminJobRegistry{
		jobs:        make(map[string]*adminAutoTuneJob),
		activeSlots: make(chan struct{}, 1),
		ctx:         ctx,
		stderr:      stderr,
		persistPath: persistPath,
	}
	if persistPath != "" {
		if loaded, ok := loadPersistedAdminJobs(persistPath); ok && len(loaded) > 0 {
			r.jobs = loaded
			// Any pending/running on disk are stale — we just booted, no
			// goroutine is actually running them. Mark them failed with a
			// definitive reason so polling consumers see a resolved state.
			now := time.Now().UTC()
			rewrote := false
			for _, job := range r.jobs {
				if job.Status == "pending" || job.Status == "running" {
					job.Status = "failed"
					job.Error = "serve restart while job in flight; re-trigger to retry"
					if job.FinishedAt == nil {
						finishedAt := now
						job.FinishedAt = &finishedAt
					}
					rewrote = true
				}
			}
			if rewrote {
				_ = r.persistLocked()
			}
			core.Print(stderr, "%s admin: restored %d auto-tune job(s) from %s", cliName(), len(r.jobs), persistPath)
		}
	}
	return r
}

// persistLocked atomically writes r.jobs to r.persistPath as JSON.
// Caller MUST hold r.mu. Best-effort: errors are logged to stderr;
// the in-memory state stays the source of truth for the running
// process — disk is a restart-survival snapshot, not a transaction
// log.
func (r *adminJobRegistry) persistLocked() error {
	if r.persistPath == "" {
		return nil
	}
	encoded := core.JSONMarshal(r.jobs)
	if !encoded.OK {
		err, _ := encoded.Value.(error)
		core.Print(r.stderr, "%s admin: persist failed (marshal): %v", cliName(), err)
		return err
	}
	if dir := core.PathDir(r.persistPath); dir != "" {
		if res := core.MkdirAll(dir, 0o755); !res.OK {
			err, _ := res.Value.(error)
			core.Print(r.stderr, "%s admin: persist failed (mkdir): %v", cliName(), err)
			return err
		}
	}
	tmpPath := r.persistPath + ".tmp"
	if res := core.WriteFile(tmpPath, encoded.Value.([]byte), 0o600); !res.OK {
		err, _ := res.Value.(error)
		core.Print(r.stderr, "%s admin: persist failed (write): %v", cliName(), err)
		return err
	}
	if res := core.Rename(tmpPath, r.persistPath); !res.OK {
		err, _ := res.Value.(error)
		core.Print(r.stderr, "%s admin: persist failed (rename): %v", cliName(), err)
		return err
	}
	return nil
}

// loadPersistedAdminJobs reads the JSON snapshot at path. Returns
// (nil, false) for any read / parse failure — caller treats as
// "no prior state" rather than fatal.
func loadPersistedAdminJobs(path string) (map[string]*adminAutoTuneJob, bool) {
	res := core.ReadFile(path)
	if !res.OK {
		return nil, false
	}
	data, ok := res.Value.([]byte)
	if !ok || len(data) == 0 {
		return nil, false
	}
	var jobs map[string]*adminAutoTuneJob
	if r := core.JSONUnmarshal(data, &jobs); !r.OK {
		return nil, false
	}
	return jobs, true
}

// commitLocked runs fn under r.mu + persists the post-fn state to
// disk. Centralises the lock + mutate + persist pattern so every
// status change automatically survives restart. Callers describe
// the mutation in fn; the helper handles lock + persist + unlock.
//
//	registry.commitLocked(func() {
//	    job.Status = "failed"
//	    job.Error = "plan: " + err.Error()
//	})
func (r *adminJobRegistry) commitLocked(fn func()) {
	r.mu.Lock()
	defer r.mu.Unlock()
	fn()
	_ = r.persistLocked()
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

// adminMuxConfig bundles the dependencies newAdminMux needs. Pulled
// out of a positional parameter list so future surfaces (per-orchestrator
// tokens, audit-sink registration, future endpoints) can attach without
// breaking call sites.
type adminMuxConfig struct {
	Stderr        io.Writer
	ServeStatus   adminServeStatus
	JobsPath      string
	Resolver      *hotSwapResolver
	HFTreeAPI     hfTreeAPI
}

// newAdminMux mounts the /v1/admin/* handlers. Returns a Handler that
// only knows the admin paths — compose with the openai mux via a
// root mux for end-to-end serve. ctx is the server-shutdown context
// (cancellation propagates into tuning + download goroutines);
// cfg.Stderr is where admin-level audit lines emit; cfg.ServeStatus is
// the boot-time snapshot of what serve was configured with — captured
// once so the /v1/admin/serve/status endpoint reports the effective
// config without recomputation; cfg.Resolver is the hot-swap resolver
// reload mutates; cfg.HFTreeAPI is the HF tree-API seam (production
// path = newHFTreeClient, tests substitute).
func newAdminMux(ctx context.Context, cfg adminMuxConfig) *http.ServeMux {
	mux := http.NewServeMux()
	registry := newAdminJobRegistry(ctx, cfg.Stderr, cfg.JobsPath)
	downloads := newAdminDownloadRegistry(ctx, cfg.Stderr)
	hf := cfg.HFTreeAPI
	if hf == nil {
		hf = newHFTreeClient()
	}

	mux.HandleFunc(adminPathMachine, adminMachineHandler)
	mux.HandleFunc(adminPathProfiles, adminProfilesHandler)
	mux.HandleFunc(adminPathAutoTune, func(w http.ResponseWriter, r *http.Request) {
		adminAutoTuneHandler(w, r, registry)
	})
	mux.HandleFunc(adminPathServeStatus, adminServeStatusHandler(cfg.ServeStatus))
	mux.HandleFunc(adminPathDownload, adminDownloadHandler(downloads, hf))
	if cfg.Resolver != nil {
		mux.HandleFunc(adminPathReload, adminReloadHandler(cfg.Resolver, cfg.Stderr))
	} else {
		mux.HandleFunc(adminPathReload, adminNotImplementedHandler("serve/reload", "no resolver wired — caller built admin mux without hotSwapResolver"))
	}
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
		registry.commitLocked(func() {
			registry.prune()
			registry.jobs[jobID] = job
		})

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
		registry.commitLocked(func() {
			finishedAt := time.Now().UTC()
			job.FinishedAt = &finishedAt
		})
	}()
	registry.commitLocked(func() {
		job.Status = "running"
	})

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
		registry.commitLocked(func() {
			job.Status = "failed"
			job.Error = "plan: " + err.Error()
		})
		return
	}
	candidates := cliLimitTuningCandidates(plan.Candidates, req.MaxCandidates)
	if len(candidates) == 0 {
		registry.commitLocked(func() {
			job.Status = "failed"
			job.Error = "no tuning candidates"
		})
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
		registry.commitLocked(func() {
			job.Status = "failed"
			job.Error = err.Error()
		})
		return
	}
	selected, ok := cliSelectTuningResult(results)
	if !ok {
		registry.commitLocked(func() {
			job.Status = "failed"
			job.Error = "no successful tuning result"
		})
		return
	}
	selectionLabels := cliTuningSelectionLabels(results, selected)
	profile := cliBuildTuningProfile(plan, req.Model, job.MachineHash, workload, selected, selectionLabels, time.Now())
	profilePath := cliTuningProfilePath(standardProfileDir(), profile)
	if err := writeTuningProfile(profilePath, profile); err != nil {
		registry.commitLocked(func() {
			job.Status = "failed"
			job.Error = "write profile: " + err.Error()
		})
		return
	}
	registry.commitLocked(func() {
		job.Status = "done"
		job.ProfilePath = profilePath
	})
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
