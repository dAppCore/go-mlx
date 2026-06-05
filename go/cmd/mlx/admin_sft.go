// SPDX-License-Identifier: EUPL-1.2

// Admin endpoints for native LoRA supervised fine-tuning.
//
// Surface (all behind the same Bearer auth as the rest of /v1/admin/*):
//
//	POST /v1/admin/sft/start          start a job, returns job_id + initial status
//	GET  /v1/admin/sft/status?job=ID  poll job state + metrics + recent loss
//	POST /v1/admin/sft/stop?job=ID    cancel a running job (preserves checkpoints)
//	GET  /v1/admin/sft/adapters       list completed adapter directories on disk
//
// Single-flight by design: only one SFT job at a time. SFT is GPU-bound
// and would starve concurrent inference; the registry rejects a second
// Start until the first completes (success, failure, or cancel).
//
// Per the binary-is-model rule: the model load for SFT is independent of
// the resolver-held serve model. mlx.LoadModel is called per-job so the
// gradient ops don't perturb the KV-cache state the serving model relies
// on. Memory cost is ~2× model footprint during a run; a future pass can
// share the underlying weights once go-mlx exposes a read-only Model view.

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
)

const (
	adminPathSFTStart    = "/v1/admin/sft/start"
	adminPathSFTStatus   = "/v1/admin/sft/status"
	adminPathSFTStop     = "/v1/admin/sft/stop"
	adminPathSFTAdapters = "/v1/admin/sft/adapters"

	// sftLossRingSize caps the per-job loss-sample ring buffer. The UI
	// curve renders the last N samples; older samples roll off so a
	// long run doesn't unbounded-grow the job record.
	sftLossRingSize = 512

	// sftDefaultEpochs / sftDefaultBatchSize / sftDefaultLR are the
	// shipped LoRA recipe defaults — match the design literal in the
	// distillation-window for users who Run without tweaking knobs.
	sftDefaultEpochs    = 3
	sftDefaultBatchSize = 8
	sftDefaultLR        = 1e-4
	sftDefaultLoRARank  = 16
	sftDefaultLoRAAlpha = 32
)

// adminSFTRequest is the POST /v1/admin/sft/start body shape. ModelPath
// + DatasetPath are required; the rest defaults to the shipped recipe.
type adminSFTRequest struct {
	ModelPath     string  `json:"model_path"`
	DatasetPath   string  `json:"dataset_path"`
	AdapterName   string  `json:"adapter_name,omitempty"` // becomes the on-disk dir name; empty → derived from model+timestamp
	BatchSize     int     `json:"batch_size,omitempty"`
	Epochs        int     `json:"epochs,omitempty"`
	LearningRate  float64 `json:"learning_rate,omitempty"`
	LoRARank      int     `json:"lora_rank,omitempty"`
	LoRAAlpha     int     `json:"lora_alpha,omitempty"`
	LoRADropout   float64 `json:"lora_dropout,omitempty"`
	MaxSeqLen     int     `json:"max_seq_len,omitempty"`
	ContextLength int     `json:"context_length,omitempty"`
}

// adminSFTLossSample is one (step, loss, epoch) datapoint. The job's
// probe sink converts each probe.KindTraining event into this shape and
// pushes it into the ring buffer so the UI loss curve has live data.
type adminSFTLossSample struct {
	Step  int     `json:"step"`
	Epoch int     `json:"epoch"`
	Loss  float64 `json:"loss"`
	TS    int64   `json:"ts_unix"`
}

// adminSFTJobState names the lifecycle of one SFT job.
type adminSFTJobState string

const (
	adminSFTStatePending adminSFTJobState = "pending"
	adminSFTStateRunning adminSFTJobState = "running"
	adminSFTStateDone    adminSFTJobState = "done"
	adminSFTStateFailed  adminSFTJobState = "failed"
	adminSFTStateStopped adminSFTJobState = "stopped"
)

// adminSFTJob is the live record for one SFT run. Mutated only behind
// adminSFTRegistry.mu; the JSON snapshot returned to callers is a copy
// so the registry's lock isn't held while the response serialises.
type adminSFTJob struct {
	JobID       string               `json:"job_id"`
	State       adminSFTJobState     `json:"state"`
	ModelPath   string               `json:"model_path"`
	DatasetPath string               `json:"dataset_path"`
	AdapterDir  string               `json:"adapter_dir"`
	StartedUnix int64                `json:"started_unix"`
	UpdatedUnix int64                `json:"updated_unix"`
	EndedUnix   int64                `json:"ended_unix,omitempty"`
	Step        int                  `json:"step"`
	Epoch       int                  `json:"epoch"`
	LastLoss    float64              `json:"last_loss"`
	Samples     int                  `json:"samples"`
	Error       string               `json:"error,omitempty"`
	Loss        []adminSFTLossSample `json:"loss,omitempty"`

	cancel context.CancelFunc `json:"-"`
}

// adminSFTRegistry is the single-flight job manager. One job at a time;
// new Start requests fail with 409 Conflict when the slot is busy.
type adminSFTRegistry struct {
	mu     sync.RWMutex
	active *adminSFTJob
	last   *adminSFTJob // last completed/failed/stopped — survives so Status by job_id still works after the run ends
}

func newAdminSFTRegistry() *adminSFTRegistry {
	return &adminSFTRegistry{}
}

// snapshot returns a deep copy of the named job (or the active job
// when jobID is empty). Returns nil when no match. Callers JSON-encode
// the snapshot — registry lock is released before encoding.
func (r *adminSFTRegistry) snapshot(jobID string) *adminSFTJob {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, j := range []*adminSFTJob{r.active, r.last} {
		if j == nil {
			continue
		}
		if jobID == "" || j.JobID == jobID {
			return cloneSFTJob(j)
		}
	}
	return nil
}

// adapterRoot is the on-disk dir new adapters land in. Each job writes
// into <root>/<adapter_name>/. Resolves to ~/Lethean/data/adapters by
// default — listing this dir surfaces all completed adapters to the UI.
func adapterRoot() string {
	homeR := core.UserHomeDir()
	if !homeR.OK {
		return "/tmp/lethean-adapters"
	}
	home, _ := homeR.Value.(string)
	return filepath.Join(home, "Lethean", "data", "adapters")
}

// deriveAdapterName builds the default dir-name when the caller didn't
// supply one. <model-basename>-<unix-seconds> — collision-resistant
// without a UUID, readable in `ls` output.
func deriveAdapterName(modelPath string) string {
	base := filepath.Base(filepath.Clean(modelPath))
	if base == "" || base == "." {
		base = "adapter"
	}
	return base + "-" + strconv.FormatInt(time.Now().Unix(), 10)
}

// newJobID is the short id stamped on each new job. Unix-seconds is
// sufficient given single-flight — collisions would need two starts in
// the same second, which the registry's busy-check already prevents.
func newJobID() string {
	return "sft-" + strconv.FormatInt(time.Now().UnixNano(), 36)
}

// cloneSFTJob deep-copies the loss slice so the caller can hold the
// returned snapshot indefinitely without racing the registry's writer.
func cloneSFTJob(src *adminSFTJob) *adminSFTJob {
	if src == nil {
		return nil
	}
	out := *src
	out.cancel = nil
	if len(src.Loss) > 0 {
		out.Loss = make([]adminSFTLossSample, len(src.Loss))
		copy(out.Loss, src.Loss)
	}
	return &out
}

// adminSFTStartHandler validates the body, claims the single-flight
// slot, and kicks the job in a goroutine. Returns 409 when busy, 400
// when the body is malformed or required paths missing.
func adminSFTStartHandler(registry *adminSFTRegistry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<14))
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		var req adminSFTRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "decode body: "+err.Error(), http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.ModelPath) == "" {
			http.Error(w, "model_path required", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.DatasetPath) == "" {
			http.Error(w, "dataset_path required", http.StatusBadRequest)
			return
		}
		if _, err := os.Stat(req.DatasetPath); err != nil {
			http.Error(w, "dataset_path not found: "+err.Error(), http.StatusBadRequest)
			return
		}

		registry.mu.Lock()
		if registry.active != nil {
			registry.mu.Unlock()
			http.Error(w, "another SFT job is already running", http.StatusConflict)
			return
		}
		adapterName := strings.TrimSpace(req.AdapterName)
		if adapterName == "" {
			adapterName = deriveAdapterName(req.ModelPath)
		}
		adapterDir := filepath.Join(adapterRoot(), adapterName)
		if err := os.MkdirAll(adapterDir, 0o755); err != nil {
			registry.mu.Unlock()
			http.Error(w, "create adapter dir: "+err.Error(), http.StatusInternalServerError)
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		job := &adminSFTJob{
			JobID:       newJobID(),
			State:       adminSFTStatePending,
			ModelPath:   req.ModelPath,
			DatasetPath: req.DatasetPath,
			AdapterDir:  adapterDir,
			StartedUnix: time.Now().Unix(),
			UpdatedUnix: time.Now().Unix(),
			cancel:      cancel,
		}
		registry.active = job
		registry.mu.Unlock()

		go runSFTJob(ctx, registry, job, req)

		writeJSON(w, http.StatusAccepted, cloneSFTJob(job))
	}
}

// adminSFTStatusHandler returns the snapshot for the job_id query param
// (or the active job when omitted). 404 when no match.
func adminSFTStatusHandler(registry *adminSFTRegistry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		jobID := strings.TrimSpace(r.URL.Query().Get("job"))
		snap := registry.snapshot(jobID)
		if snap == nil {
			http.Error(w, "no SFT job", http.StatusNotFound)
			return
		}
		writeJSON(w, http.StatusOK, snap)
	}
}

// adminSFTStopHandler cancels the active job's context. The runner
// goroutine observes the cancellation and flips State to "stopped";
// checkpoints written before the cancel survive on disk.
func adminSFTStopHandler(registry *adminSFTRegistry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		jobID := strings.TrimSpace(r.URL.Query().Get("job"))
		registry.mu.Lock()
		if registry.active == nil || (jobID != "" && registry.active.JobID != jobID) {
			registry.mu.Unlock()
			http.Error(w, "no active SFT job for that id", http.StatusNotFound)
			return
		}
		if registry.active.cancel != nil {
			registry.active.cancel()
		}
		snap := cloneSFTJob(registry.active)
		registry.mu.Unlock()
		writeJSON(w, http.StatusOK, snap)
	}
}

// adminSFTAdaptersHandler lists adapter directories under
// ~/Lethean/data/adapters/. Each entry carries the dir name + size +
// last-modified so the UI can show a Recent Adapters list ordered by
// freshness.
func adminSFTAdaptersHandler() http.HandlerFunc {
	type adapterEntry struct {
		Name       string `json:"name"`
		Path       string `json:"path"`
		SizeBytes  int64  `json:"size_bytes"`
		ModifiedAt int64  `json:"modified_unix"`
	}
	type adaptersList struct {
		Dir      string         `json:"dir"`
		Adapters []adapterEntry `json:"adapters"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		root := adapterRoot()
		out := adaptersList{Dir: root, Adapters: []adapterEntry{}}
		entries, err := os.ReadDir(root)
		if err != nil {
			// Dir doesn't exist yet (no SFT has ever run) — return
			// the empty list rather than 500. The UI renders an
			// empty-state hint.
			writeJSON(w, http.StatusOK, out)
			return
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			out.Adapters = append(out.Adapters, adapterEntry{
				Name:       e.Name(),
				Path:       filepath.Join(root, e.Name()),
				SizeBytes:  dirSizeBytes(filepath.Join(root, e.Name())),
				ModifiedAt: info.ModTime().Unix(),
			})
		}
		writeJSON(w, http.StatusOK, out)
	}
}

// dirSizeBytes sums up the regular-file bytes under dir. Best-effort —
// any errors collapse to the bytes summed so far. Used only for the
// adapter list's "size" column; doesn't need to be exact.
func dirSizeBytes(dir string) int64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return nil
		}
		total += info.Size()
		return nil
	})
	return total
}

func adminSFTDatasetConfig(info mlx.ModelInfo) dataset.Config {
	return mlx.DatasetConfigForModel(info)
}

// runSFTJob is the goroutine body. Loads the model, opens the dataset,
// builds SFTConfig with a probe sink that updates the job record, calls
// TrainSFT, persists the final state. Owned by the registry — when this
// returns, `active` becomes `last` so subsequent Status by job_id still
// resolves.
func runSFTJob(ctx context.Context, registry *adminSFTRegistry, job *adminSFTJob, req adminSFTRequest) {
	defer func() {
		registry.mu.Lock()
		registry.last = registry.active
		registry.active = nil
		registry.mu.Unlock()
	}()

	loadOpts := []mlx.LoadOption{}
	if req.ContextLength > 0 {
		loadOpts = append(loadOpts, mlx.WithContextLength(req.ContextLength))
	}
	model, err := mlx.LoadModel(req.ModelPath, loadOpts...)
	if err != nil {
		registry.failJob(job, "load model: "+err.Error())
		return
	}
	defer func() { _ = model.Close() }()

	f, err := os.Open(req.DatasetPath)
	if err != nil {
		registry.failJob(job, "open dataset: "+err.Error())
		return
	}
	defer f.Close()
	ds, err := dataset.LoadJSONL(f, adminSFTDatasetConfig(model.Info()))
	if err != nil {
		registry.failJob(job, "parse dataset: "+err.Error())
		return
	}

	// Mark running once the heavy load+parse work succeeded — the job
	// state only flips off "pending" when we're actually about to call
	// TrainSFT. Probe sink updates the same struct as more samples land.
	registry.markRunning(job)

	cfg := mlx.SFTConfig{
		LoRA: mlx.LoRAConfig{
			Rank:  pickInt(req.LoRARank, sftDefaultLoRARank),
			Alpha: float32(pickInt(req.LoRAAlpha, sftDefaultLoRAAlpha)),
		},
		BatchSize:     pickInt(req.BatchSize, sftDefaultBatchSize),
		Epochs:        pickInt(req.Epochs, sftDefaultEpochs),
		LearningRate:  pickFloat(req.LearningRate, sftDefaultLR),
		MaxSeqLen:     req.MaxSeqLen,
		CheckpointDir: job.AdapterDir,
		SavePath:      filepath.Join(job.AdapterDir, "adapter.safetensors"),
		ProbeSink:     newSFTProbeSink(registry, job),
	}
	// LoRADropout request field is parked — upstream LoRAConfig
	// doesn't expose a dropout knob in the current implementation.
	// Kept on the wire so the UI can render it as informational; if
	// upstream adds it later this is a single-line plumb.
	_ = req.LoRADropout

	if _, runErr := model.TrainSFT(ctx, ds, cfg); runErr != nil {
		// Cancelled-mid-run lands as either "context canceled" or
		// "context deadline exceeded" — surface as stopped, not
		// failed, so the UI can show a calmer "you stopped this"
		// rather than a red-alert error frame.
		if ctx.Err() != nil {
			registry.markStopped(job)
			return
		}
		registry.failJob(job, runErr.Error())
		return
	}
	registry.markDone(job)
}

// newSFTProbeSink returns a probe.Sink that funnels Training events
// into the job's metrics + loss ring. Event copy is cheap (the Training
// payload is small), happens under the registry write lock to keep the
// snapshot reader-safe.
func newSFTProbeSink(registry *adminSFTRegistry, job *adminSFTJob) probe.Sink {
	return probe.SinkFunc(func(e probe.Event) {
		if e.Kind != probe.KindTraining || e.Training == nil {
			return
		}
		registry.mu.Lock()
		defer registry.mu.Unlock()
		if registry.active == nil || registry.active.JobID != job.JobID {
			return // job ended; ignore late events
		}
		j := registry.active
		j.Step = e.Training.Step
		j.Epoch = e.Training.Epoch
		j.LastLoss = e.Training.Loss
		j.Samples++
		j.UpdatedUnix = time.Now().Unix()
		sample := adminSFTLossSample{
			Step:  e.Training.Step,
			Epoch: e.Training.Epoch,
			Loss:  e.Training.Loss,
			TS:    time.Now().Unix(),
		}
		if len(j.Loss) >= sftLossRingSize {
			j.Loss = append(j.Loss[1:], sample)
		} else {
			j.Loss = append(j.Loss, sample)
		}
	})
}

// markRunning / markDone / markStopped / failJob are the registry's
// terminal-state flippers. Centralised so the UpdatedUnix +
// EndedUnix stamps stay consistent across exit paths.
func (r *adminSFTRegistry) markRunning(job *adminSFTJob) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil && r.active.JobID == job.JobID {
		r.active.State = adminSFTStateRunning
		r.active.UpdatedUnix = time.Now().Unix()
	}
}

func (r *adminSFTRegistry) markDone(job *adminSFTJob) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil && r.active.JobID == job.JobID {
		r.active.State = adminSFTStateDone
		r.active.EndedUnix = time.Now().Unix()
		r.active.UpdatedUnix = r.active.EndedUnix
	}
}

func (r *adminSFTRegistry) markStopped(job *adminSFTJob) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil && r.active.JobID == job.JobID {
		r.active.State = adminSFTStateStopped
		r.active.EndedUnix = time.Now().Unix()
		r.active.UpdatedUnix = r.active.EndedUnix
	}
}

func (r *adminSFTRegistry) failJob(job *adminSFTJob, reason string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil && r.active.JobID == job.JobID {
		r.active.State = adminSFTStateFailed
		r.active.Error = reason
		r.active.EndedUnix = time.Now().Unix()
		r.active.UpdatedUnix = r.active.EndedUnix
	}
}

// pickInt / pickFloat are small null-coalesce helpers — keep the
// SFTConfig builder readable.
func pickInt(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

func pickFloat(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}
