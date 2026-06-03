// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"slices"
	"sync"
	"syscall"
	"time"

	core "dappco.re/go"
)

// /v1/admin/models/download — fetch a model from HuggingFace into
// the operator-allowlisted set under ~/Lethean/data/models/.
//
// CRITICAL-class endpoint (Cerberus DREAD §4.F-6). The threat
// surface is arbitrary URL → arbitrary filesystem write → arbitrary
// code execution if a tokeniser/config is parsed eagerly. Gated
// with eight checks before bytes flow:
//
//  1. URL allowlist (huggingface.co only). Request shape is
//     {repo, revision}, never {url} — server composes the URL.
//  2. Repo allowlist gate (~/Lethean/data/allowed-models.json).
//     Default empty → refuse all. Operator curates.
//  3. Destination is server-controlled. Path = standardModelDir() /
//     canonicalised_repo / revision. Request CANNOT supply path.
//  4. Disk-space check (Statfs) — refuse if free < 2× advertised.
//  5. Single-slot semaphore (F-1 pattern). One download in flight.
//  6. Integrity verification — sha256 from HF lfs metadata. Bytes
//     write to a .quarantine dir; promoted to final on verify.
//  7. Audit emit on kickoff + complete + fail.
//  8. NO coupling to serve/reload. Download lands files only; the
//     operator must POST /v1/admin/serve/reload separately.

// adminDownloadRequest is the body shape for POST /v1/admin/models/download.
// Per §4.F-6.1 callers supply repo + revision, NEVER a URL. The server
// composes URLs from the HF allowlist.
type adminDownloadRequest struct {
	// Repo is the HuggingFace "<org>/<name>" identifier. Must be in
	// the operator's allowlist (~/Lethean/data/allowed-models.json).
	Repo string `json:"repo"`

	// Revision is the HF tree ref — branch name ("main") or commit
	// sha. Bare "main" accepts a moving target; the audit row
	// stamps "moving=true" so the operator knows the integrity is
	// HF-tree-API-current, not pinned.
	Revision string `json:"revision"`

	// Files is an optional whitelist of files to fetch. Empty =
	// fetch all files the HF tree API lists. Mostly used for partial
	// downloads of multi-file repos (e.g. GGUF-only when the repo
	// also carries safetensors).
	Files []string `json:"files,omitempty"`
}

// adminDownloadJob mirrors adminAutoTuneJob — status transitions
// pending → running → done | failed. Single in-flight job tracked in
// memory; not persisted across restarts (downloads are restartable
// from scratch, unlike auto-tune which loses computed candidates).
type adminDownloadJob struct {
	ID         string     `json:"id"`
	Status     string     `json:"status"`
	Repo       string     `json:"repo"`
	Revision   string     `json:"revision"`
	StartedAt  time.Time  `json:"started_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	DestPath   string     `json:"dest_path,omitempty"`
	BytesTotal int64      `json:"bytes_total,omitempty"`
	BytesDone  int64      `json:"bytes_done,omitempty"`
	FileCount  int        `json:"file_count,omitempty"`
	Error      string     `json:"error,omitempty"`
}

// maxDownloadJobsRetained bounds the in-memory job map (F-6 N-3). Each
// download leaves one job record behind; without eviction the map grows
// for the process lifetime. Only one job runs at a time, so a small cap
// keeps enough recent history for polling while staying bounded. The
// in-flight job is never evicted.
const maxDownloadJobsRetained = 32

// adminDownloadRegistry — single in-flight job, semaphore-gated.
// Pattern mirrors adminJobRegistry (F-1) but simpler (one slot, no
// persistence — restarted downloads start over).
type adminDownloadRegistry struct {
	mu          sync.Mutex
	jobs        map[string]*adminDownloadJob
	activeSlots chan struct{}
	ctx         context.Context
	stderr      io.Writer
}

// evictOldDownloadJobsLocked prunes finished jobs (done/failed) oldest-
// first until the map is back under maxDownloadJobsRetained. Caller must
// hold r.mu. Running/pending jobs are never evicted regardless of age.
func (r *adminDownloadRegistry) evictOldDownloadJobsLocked() {
	for len(r.jobs) > maxDownloadJobsRetained {
		var oldestID string
		var oldest time.Time
		for id, j := range r.jobs {
			if j.Status != "done" && j.Status != "failed" {
				continue
			}
			if oldestID == "" || j.StartedAt.Before(oldest) {
				oldestID = id
				oldest = j.StartedAt
			}
		}
		if oldestID == "" {
			// Nothing evictable (all remaining jobs are in flight) —
			// stop rather than spin.
			return
		}
		delete(r.jobs, oldestID)
	}
}

func newAdminDownloadRegistry(ctx context.Context, stderr io.Writer) *adminDownloadRegistry {
	return &adminDownloadRegistry{
		jobs:        make(map[string]*adminDownloadJob),
		activeSlots: make(chan struct{}, 1),
		ctx:         ctx,
		stderr:      stderr,
	}
}

func (r *adminDownloadRegistry) tryAcquire() bool {
	select {
	case r.activeSlots <- struct{}{}:
		return true
	default:
		return false
	}
}

func (r *adminDownloadRegistry) release() {
	<-r.activeSlots
}

// allowedModelsPath is where the operator-curated repo allowlist
// lives. Sibling of admin.token. Default-absent → empty allowlist →
// refuse all downloads (fail-closed). Operator creates the file
// with the repos they want to permit:
//
//	{"repos": ["meta-llama/Llama-3.1-8B", "google/gemma-2-9b"]}
func allowedModelsPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "allowed-models.json")
}

type allowedModelsFile struct {
	Repos []string `json:"repos"`
}

// loadAllowedModels reads the allowlist file. Returns empty slice
// + no error when the file doesn't exist (fail-closed default).
// Parse failure surfaces as error so the operator notices the typo.
func loadAllowedModels(path string) ([]string, error) {
	res := core.ReadFile(path)
	if !res.OK {
		return []string{}, nil
	}
	body, _ := res.Value.([]byte)
	if len(body) == 0 {
		return []string{}, nil
	}
	var f allowedModelsFile
	if r := core.JSONUnmarshal(body, &f); !r.OK {
		return nil, core.E("admin.allowedModels", "parse", r.Value.(error))
	}
	return f.Repos, nil
}

// isRepoAllowed checks repo membership against the loaded list.
// O(N) is fine — allowlists are operator-curated, expect tens not
// thousands.
func isRepoAllowed(allowed []string, repo string) bool {
	return slices.Contains(allowed, repo)
}

// canonicaliseRepoName turns "<org>/<name>" into the on-disk dir
// basename. HF allows / in repo ids; we collapse to __ so the dir
// tree stays one level deep. Inverse mapping isn't needed —
// downloads are addressed by name post-write, the original repo
// only lives in the audit log.
//
//	canonicaliseRepoName("meta-llama/Llama-3.1-8B")
//	// → "meta-llama__Llama-3.1-8B"
func canonicaliseRepoName(repo string) string {
	return core.Replace(repo, "/", "__")
}

// validateRevision restricts revision to alphanumeric + `-._`. HF
// accepts branch names + commit shas; both fit this charset. Defends
// against shell-injection-shaped chars in dir names.
func validateRevision(rev string) error {
	if rev == "" {
		return core.NewError("revision required")
	}
	if len(rev) > 64 {
		return core.NewError("revision too long")
	}
	for _, c := range rev {
		ok := (c >= '0' && c <= '9') ||
			(c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			c == '-' || c == '.' || c == '_'
		if !ok {
			return core.NewError("revision contains disallowed character")
		}
	}
	return nil
}

// diskFreeBytes returns the free space at path. Best-effort —
// returns 0 on any Statfs failure (caller treats 0 as "unknown,
// proceed with caution"; the disk-space pre-check warns but does
// not refuse when free is unknown). syscall.Statfs is Unix-only;
// non-Unix builds skip the check via _other.go (not present today
// because lthn-mlx is darwin-only — the lthn-cuda + lthn-amd
// siblings will add their own platform variants).
func diskFreeBytes(path string) uint64 {
	var s syscall.Statfs_t
	if err := syscall.Statfs(path, &s); err != nil {
		return 0
	}
	return uint64(s.Bavail) * uint64(s.Bsize)
}

// adminDownloadHandler — POST kicks off a job; GET polls. Single
// in-flight slot per §4.F-6.5. Audit emits on kickoff / complete /
// fail per §4.F-6.7.
func adminDownloadHandler(registry *adminDownloadRegistry, hf hfTreeAPI) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
			var req adminDownloadRequest
			if err := readJSONBody(r, &req); err != nil {
				http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
				return
			}
			repo := core.Trim(req.Repo)
			revision := core.Trim(req.Revision)
			if revision == "" {
				revision = "main"
			}
			if repo == "" {
				http.Error(w, "repo required", http.StatusBadRequest)
				return
			}
			if err := validateRevision(revision); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}

			// Allowlist gate per §4.F-6.2. Fail-closed default —
			// missing file = empty allowlist = refuse.
			allowed, err := loadAllowedModels(allowedModelsPath())
			if err != nil {
				http.Error(w, "allowlist parse: "+err.Error(), http.StatusInternalServerError)
				return
			}
			if !isRepoAllowed(allowed, repo) {
				core.Print(registry.stderr, "%s admin: model_download deny repo=%s remote=%s reason=not_in_allowlist",
					cliName(), repo, r.RemoteAddr)
				http.Error(w, "repo not in allowlist (~/Lethean/data/allowed-models.json)", http.StatusForbidden)
				return
			}

			// Single-slot semaphore per §4.F-6.5.
			if !registry.tryAcquire() {
				http.Error(w, "download busy — another job in flight", http.StatusTooManyRequests)
				return
			}

			jobID := nowDownloadJobID()
			destRoot := core.PathJoin(standardModelDir(), canonicaliseRepoName(repo), revision)
			job := &adminDownloadJob{
				ID:        jobID,
				Status:    "pending",
				Repo:      repo,
				Revision:  revision,
				StartedAt: time.Now().UTC(),
				DestPath:  destRoot,
			}
			registry.mu.Lock()
			registry.jobs[jobID] = job
			registry.evictOldDownloadJobsLocked()
			registry.mu.Unlock()

			core.Print(registry.stderr, "%s admin: model_download kickoff job=%s repo=%s revision=%s remote=%s",
				cliName(), jobID, repo, revision, r.RemoteAddr)

			go func() {
				defer registry.release()
				runDownloadJob(job, req, hf, registry)
			}()
			writeJSON(w, http.StatusAccepted, job)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

// runDownloadJob is the background worker. Resolves the HF tree,
// disk-space checks, fetches each file to .quarantine, verifies
// sha256, atomically promotes, writes the .sha256 sidecar. On any
// failure the quarantine dir is left for forensic inspection — the
// operator decides whether to retry (and we don't half-delete state).
func runDownloadJob(job *adminDownloadJob, req adminDownloadRequest, hf hfTreeAPI, registry *adminDownloadRegistry) {
	defer func() {
		registry.mu.Lock()
		finishedAt := time.Now().UTC()
		job.FinishedAt = &finishedAt
		registry.mu.Unlock()
	}()

	registry.mu.Lock()
	job.Status = "running"
	registry.mu.Unlock()

	entries, err := hf.ResolveTree(registry.ctx, req.Repo, req.Revision)
	if err != nil {
		setDownloadFailed(job, registry, "resolve tree: "+err.Error())
		return
	}

	// Filter to req.Files if non-empty. Empty = all files.
	wanted := entries
	if len(req.Files) > 0 {
		want := map[string]struct{}{}
		for _, f := range req.Files {
			want[f] = struct{}{}
		}
		wanted = wanted[:0]
		for _, e := range entries {
			if _, ok := want[e.Path]; ok {
				wanted = append(wanted, e)
			}
		}
	}
	if len(wanted) == 0 {
		setDownloadFailed(job, registry, "no files matched (check repo + revision + files filter)")
		return
	}

	var totalBytes int64
	for _, e := range wanted {
		totalBytes += e.Size
	}
	registry.mu.Lock()
	job.BytesTotal = totalBytes
	job.FileCount = len(wanted)
	registry.mu.Unlock()

	// Disk-space check per §4.F-6.4. Refuse if free < 2× total
	// (write into quarantine + final = 2× peak during promote).
	free := diskFreeBytes(core.PathDir(job.DestPath))
	if free > 0 && totalBytes > 0 && free < uint64(totalBytes*2) {
		setDownloadFailed(job, registry, core.Sprintf("disk-space: free=%d need=%d (2× model size)", free, totalBytes*2))
		return
	}

	// Prepare the dest tree: final dir + quarantine sibling. The
	// quarantine path is .<dest>.quarantine — atomic promote via
	// dir rename at the end.
	finalDir := job.DestPath
	quarantineDir := finalDir + ".quarantine"
	if r := core.MkdirAll(quarantineDir, 0o755); !r.OK {
		setDownloadFailed(job, registry, "mkdir quarantine: "+r.Value.(error).Error())
		return
	}

	digests := make(map[string]string, len(wanted))
	var done int64
	for _, e := range wanted {
		if registry.ctx.Err() != nil {
			setDownloadFailed(job, registry, "cancelled")
			return
		}
		if e.Digest == "" {
			// Tokeniser / config files lack lfs.sha256 — accept
			// in soft-mode (audit-trail per spec §4.F-6.6 note).
			core.Print(registry.stderr, "%s admin: model_download warn job=%s file=%s digest_missing (HF non-LFS file)",
				cliName(), job.ID, e.Path)
		}
		destFile := core.PathJoin(quarantineDir, e.Path)
		if r := core.MkdirAll(core.PathDir(destFile), 0o755); !r.OK {
			setDownloadFailed(job, registry, "mkdir file dir: "+r.Value.(error).Error())
			return
		}
		written, sha, err := fetchAndVerify(registry.ctx, e.URL, destFile, e.Digest, e.Size)
		if err != nil {
			setDownloadFailed(job, registry, "fetch "+e.Path+": "+err.Error())
			return
		}
		digests[e.Path] = sha
		done += written
		registry.mu.Lock()
		job.BytesDone = done
		registry.mu.Unlock()
	}

	// Write the .sha256 sidecar in the quarantine dir BEFORE
	// promoting — the F-7 reload handler refuses any model dir
	// without this file, so writing it last (post-rename) would
	// leave a window where reload sees the dir but no sidecar.
	if err := writeModelManifest(quarantineDir, digests); err != nil {
		setDownloadFailed(job, registry, "write manifest: "+err.Error())
		return
	}

	// Remove any old final dir + promote quarantine. We use
	// rename(2) for atomic-ish promote — the dir is renamed in one
	// syscall; readers see either old or new, never partial.
	// core.RemoveAll is idempotent — silent on not-exist — so the
	// pre-check is a Stat round-trip we can skip.
	if r := core.RemoveAll(finalDir); !r.OK {
		setDownloadFailed(job, registry, "remove old: "+r.Value.(error).Error())
		return
	}
	if r := core.Rename(quarantineDir, finalDir); !r.OK {
		setDownloadFailed(job, registry, "promote: "+r.Value.(error).Error())
		return
	}

	registry.mu.Lock()
	job.Status = "done"
	registry.mu.Unlock()
	core.Print(registry.stderr, "%s admin: model_download done job=%s repo=%s revision=%s files=%d bytes=%d",
		cliName(), job.ID, job.Repo, job.Revision, job.FileCount, job.BytesDone)
}

// setDownloadFailed centralises the failure path so audit-emit +
// state update stay consistent across the dozen-or-so error sites.
func setDownloadFailed(job *adminDownloadJob, registry *adminDownloadRegistry, reason string) {
	registry.mu.Lock()
	job.Status = "failed"
	job.Error = reason
	registry.mu.Unlock()
	core.Print(registry.stderr, "%s admin: model_download fail job=%s repo=%s revision=%s reason=%s",
		cliName(), job.ID, job.Repo, job.Revision, reason)
}

func nowDownloadJobID() string {
	return core.Sprintf("download-%d", time.Now().UTC().UnixNano())
}
