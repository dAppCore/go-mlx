// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
)

// Admin HTTP API — the surface a higher-level orchestrator (lthn-desktop
// GUI, or Lemma's tool-calling) composes to express the "Lemma, try the
// new Qwen model" UX without operator gymnastics.
//
// Endpoints under /v1/admin/*:
//
//	GET  /v1/admin/machine            current machine identity (hash, hostname, runtime info)
//	GET  /v1/admin/serve/status       snapshot of model + applied config
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
	adminPathMachine  = "/v1/admin/machine"
	adminPathDownload = "/v1/admin/models/download"
	adminPathReload   = "/v1/admin/serve/reload"
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

// adminMuxConfig bundles the dependencies newAdminMux needs. Pulled
// out of a positional parameter list so future surfaces (per-orchestrator
// tokens, audit-sink registration, future endpoints) can attach without
// breaking call sites.
type adminMuxConfig struct {
	Stderr      io.Writer
	ServeStatus adminServeStatus
	Resolver    *hotSwapResolver
	HFTreeAPI   hfTreeAPI
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
	downloads := newAdminDownloadRegistry(ctx, cfg.Stderr)
	sft := newAdminSFTRegistry()
	hf := cfg.HFTreeAPI
	if hf == nil {
		hf = newHFTreeClient()
	}

	mux.HandleFunc(adminPathMachine, adminMachineHandler)
	mux.HandleFunc(adminPathServeStatus, adminServeStatusHandler(cfg.ServeStatus))
	mux.HandleFunc(adminPathDownload, adminDownloadHandler(downloads, hf))
	if cfg.Resolver != nil {
		mux.HandleFunc(adminPathReload, adminReloadHandler(cfg.Resolver, cfg.Stderr))
	} else {
		mux.HandleFunc(adminPathReload, adminNotImplementedHandler("serve/reload", "no resolver wired — caller built admin mux without hotSwapResolver"))
	}
	// SFT — native LoRA supervised fine-tuning. Single-flight; the
	// registry rejects concurrent Start calls (returns 409). Loads
	// its own model copy independent of cfg.Resolver so a running job
	// doesn't perturb the serve model's KV state. See admin_sft.go.
	mux.HandleFunc(adminPathSFTStart, adminSFTStartHandler(sft))
	mux.HandleFunc(adminPathSFTStatus, adminSFTStatusHandler(sft))
	mux.HandleFunc(adminPathSFTStop, adminSFTStopHandler(sft))
	mux.HandleFunc(adminPathSFTAdapters, adminSFTAdaptersHandler())
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
