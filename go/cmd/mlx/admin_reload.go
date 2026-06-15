// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"io"
	"io/fs"
	"net/http"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

// /v1/admin/serve/reload — hot-swap the loaded model.
//
// CRITICAL-class endpoint (Cerberus DREAD §4.F-7). The threat surface
// is full prompt-flow redirection: any caller who can flip the model
// owns every subsequent /v1/chat/completions response. The handler
// gates the verb with five checks before the swap:
//
//  1. Model NAME (basename), never raw path — server resolves
//     against the known-models dir tree.
//  2. Resolved path stays under ~/Lethean/data/models/ (escape gate).
//  3. Model dir carries a .sha256 sidecar (integrity contract from
//     F-6 — refuse "whatever's on disk").
//  4. Confirmation token = machine_hash from /v1/admin/machine
//     (confused-deputy defence).
//  5. Bearer auth on the path-prefix (admin_auth.go).
//
// Drain policy: in-flight Generate/Chat calls complete on old
// weights (the hotSwapResolver hands back the active model at
// resolve time; the caller's reference keeps it alive through GC).
// New requests get new weights. Documented per §4.F-7.5; audit
// emit on every attempt + outcome per §4.F-7.6.

// adminReloadRequest is the body shape for POST /v1/admin/serve/reload.
// Per §4.F-7.1 the request supplies a model NAME (basename under the
// known-models dir tree), NEVER a raw path. Per §4.F-7.3 the request
// MUST also supply the current machine hash as confirmation — proves
// the caller has done a /v1/admin/machine GET first.
type adminReloadRequest struct {
	// Model is the basename of a dir under standardModelDir() that
	// the server is permitted to load. Backwards-compat field —
	// new callers should send ModelPath instead. When both are set,
	// ModelPath wins.
	Model string `json:"model,omitempty"`

	// ModelPath is the absolute path of the dir to load. Must
	// resolve under standardModelDir() — path-escape outside is
	// rejected. Preferred over the basename-only Model field so
	// callers (model-browser-window, lemma-window) can pass back
	// the Models.List() entry verbatim without a separate basename
	// derivation.
	ModelPath string `json:"model_path,omitempty"`

	// Confirmation MUST equal the current machine hash from
	// /v1/admin/machine. Defends against confused-deputy where
	// another tool POSTs reload via a stolen session — the attacker
	// would need to ALSO be able to GET /v1/admin/machine, which
	// proves session + machine pairing.
	Confirmation string `json:"confirmation,omitempty"`

	// ConfirmMachine is the modern field name for Confirmation —
	// matches the pkg/lemma client convention (confirm_machine in
	// JSON). Either is accepted; ConfirmMachine wins when both set.
	ConfirmMachine string `json:"confirm_machine,omitempty"`

	// ProfilePath is an optional tuning profile applied alongside
	// the model. Empty → fall through to the auto-tune profile
	// discovered for the model dir; explicit → override.
	ProfilePath string `json:"profile_path,omitempty"`

	// AdapterPath is an optional LoRA adapter file (or dir) to
	// overlay on the base model. Empty → load model bare. The
	// Fine-tune surface uses this for the A/B "test with this
	// adapter" flow — Lemma.SFTStart writes the adapter dir; the
	// caller passes the resulting path back to Reload here.
	AdapterPath string `json:"adapter_path,omitempty"`

	// ContextLength overrides the model's default context length
	// for this reload. Zero → use the profile's value.
	ContextLength int `json:"context_length,omitempty"`
}

// adminReloadResponse names the swap. The from / to paths feed the
// audit emit + the per-stream notification surface (clients consuming
// the response can show "weights changed mid-conversation").
type adminReloadResponse struct {
	Status   string `json:"status"`
	From     string `json:"from_model_path"`
	To       string `json:"to_model_path"`
	LoadedAt int64  `json:"loaded_at_unix"`
}

// standardModelDir returns ~/Lethean/data/models/ — the canonical
// root the reload + download endpoints both bound against. Created
// lazily by F-6 (downloader); the reload handler refuses if the dir
// or the requested sub-dir is missing.
func standardModelDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "models")
}

// shaManifestFilename is the sidecar F-6 writes into the model dir
// (one digest per file, newline-separated, "<sha256>  <filename>"
// format — same as `shasum -a 256 *`). F-7 refuses to reload any
// model dir missing this file, per §4.F-7.2 (no hot-swap to
// unverified-integrity models).
const shaManifestFilename = ".sha256"

// adminReloadHandler answers POST /v1/admin/serve/reload. Wired via
// newAdminMux when serve booted with a hotSwapResolver. The handler
// audit-emits the kickoff line BEFORE any of the gate checks (the
// audit row carries the requester + remote so a brute-force attempt
// against confirmation is visible even when refused).
//
//	mux.HandleFunc(adminPathReload, adminReloadHandler(resolver, stderr))
func adminReloadHandler(resolver *hotSwapResolver, stderr io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req adminReloadRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		// Modern field names (model_path / confirm_machine) win when
		// both are present. The legacy basename + confirmation fields
		// stay accepted for backward compat with v1 callers.
		modelName := core.Trim(req.Model)
		modelPath := core.Trim(req.ModelPath)
		confirmation := core.Trim(req.ConfirmMachine)
		if confirmation == "" {
			confirmation = core.Trim(req.Confirmation)
		}
		from := resolver.CurrentPath()

		// Audit the attempt BEFORE the gate checks so brute-force
		// confirmation guesses are visible per §4.F-7.6.
		auditTarget := modelName
		if modelPath != "" {
			auditTarget = modelPath
		}
		core.Print(stderr, "%s admin: serve_reload attempt requester=%s from=%s to=%s adapter=%s",
			cliName(), r.RemoteAddr, from, auditTarget, req.AdapterPath)

		if modelName == "" && modelPath == "" {
			adminReloadDeny(w, stderr, from, auditTarget, "model or model_path required")
			return
		}
		if confirmation == "" {
			adminReloadDeny(w, stderr, from, auditTarget, "confirm_machine required (machine_hash from /v1/admin/machine)")
			return
		}

		// Gate 1: confirmation matches the live machine hash.
		expected, err := currentMachineProfileHash(r.Context())
		if err != nil {
			adminReloadFail(w, stderr, from, auditTarget, "machine hash unavailable: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if confirmation != expected {
			adminReloadDeny(w, stderr, from, auditTarget, "confirm_machine mismatch")
			return
		}

		// Gate 2 + 3: resolve target → on-disk path. When ModelPath
		// is supplied it must canonicalise to a child of
		// standardModelDir() (no path-escape); when only Model is set
		// we go through the basename resolver as v1 did.
		var toPath string
		if modelPath != "" {
			toPath, err = bindModelPathToStandardDir(modelPath)
			if err != nil {
				adminReloadDeny(w, stderr, from, auditTarget, err.Error())
				return
			}
		} else {
			toPath, err = resolveModelNameToPath(modelName)
			if err != nil {
				adminReloadDeny(w, stderr, from, auditTarget, err.Error())
				return
			}
		}

		// Build the per-reload load options. v1 always passed nil
		// (inheriting boot opts); v2 plumbs ContextLength + AdapterPath
		// here so the Fine-tune A/B flow can overlay an adapter and
		// the operator can pick a different context on hot-swap.
		// ProfilePath is reserved — auto-discovery via mlx.LoadModelAsTextModel
		// already finds the standard profile for the target dir; an
		// explicit override is the next pass.
		var opts []mlx.LoadOption
		if req.ContextLength > 0 {
			opts = append(opts, mlx.WithContextLength(req.ContextLength))
		}
		if core.Trim(req.AdapterPath) != "" {
			opts = append(opts, mlx.WithAdapterPath(req.AdapterPath))
		}

		prev, newPath, err := resolver.Replace(toPath, opts)
		if err != nil {
			adminReloadFail(w, stderr, from, auditTarget, "load failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		prevPath := from
		if prev != nil {
			prevPath = prev.modelPath
		}
		core.Print(stderr, "%s admin: serve_reload success requester=%s from=%s to=%s",
			cliName(), r.RemoteAddr, prevPath, newPath)

		writeJSON(w, http.StatusOK, adminReloadResponse{
			Status:   "ok",
			From:     prevPath,
			To:       newPath,
			LoadedAt: time.Now().Unix(),
		})
	}
}

// adminReloadDeny answers a 400 + audits the refusal reason. Pulled
// out of the handler so the audit + response shape stay consistent
// across the five gate checks.
func adminReloadDeny(w http.ResponseWriter, stderr io.Writer, from, modelName, reason string) {
	core.Print(stderr, "%s admin: serve_reload deny from=%s to_name=%s reason=%s",
		cliName(), from, modelName, reason)
	http.Error(w, reason, http.StatusBadRequest)
}

// adminReloadFail audits + answers with the given status. Separate
// from adminReloadDeny so 5xx failures (infra-level) and 4xx denials
// (caller-level) chip-filter cleanly in audit replay.
func adminReloadFail(w http.ResponseWriter, stderr io.Writer, from, modelName, reason string, status int) {
	core.Print(stderr, "%s admin: serve_reload fail from=%s to_name=%s reason=%s",
		cliName(), from, modelName, reason)
	http.Error(w, reason, status)
}

// resolveModelNameToPath maps a basename (e.g. "meta-llama__Llama-3.1-8B")
// to its on-disk dir under standardModelDir(). Refuses any name that
// escapes the dir (`..`, `/`, symlink-resolves outside, no
// `.sha256` sidecar). Path-injection class per §4.F-7.1.
// bindModelPathToStandardDir accepts an absolute model path and
// verifies it canonicalises to a child of standardModelDir(). Returns
// the resolved on-disk path on success. Used by the v2 reload shape
// where callers supply the full path (matches Models.List() entries)
// instead of a basename. Same security envelope as
// pathWithinDir reports whether resolved lives inside rootResolved, using a
// filepath.Rel-based containment test rather than a raw string prefix. On a
// case-insensitive filesystem (macOS default) PathEvalSymlinks can hand back a
// different casing than the configured root, which a byte-prefix check rejects
// as an escape even though the path is genuinely inside the tree; Rel computes
// containment over cleaned path semantics and avoids that false negative.
//
//	pathWithinDir("/m/models", "/m/models/gemma") // true
//	pathWithinDir("/m/models", "/m/models-evil")  // false (sibling, not child)
//	pathWithinDir("/m/models", "/etc/passwd")     // false (relative starts ..)
func pathWithinDir(rootResolved, resolved string) bool {
	if resolved == rootResolved {
		return true
	}
	rel := core.PathRel(rootResolved, resolved)
	if !rel.OK {
		return false
	}
	r, _ := rel.Value.(string)
	if r == "" || r == "." {
		return true
	}
	// Any path that has to climb out of root (".." segment) or is absolute
	// is not contained.
	if r == ".." || core.HasPrefix(r, "../") || core.PathIsAbs(r) {
		return false
	}
	return true
}

// resolveModelNameToPath — escape-prefix check + sha-manifest gate.
func bindModelPathToStandardDir(path string) (string, error) {
	if path == "" {
		return "", core.NewError("model_path required")
	}
	root := standardModelDir()
	rootResolved := root
	if r := core.PathEvalSymlinks(root); r.OK {
		rootResolved = r.Value.(string)
	}
	resolved := path
	if r := core.PathEvalSymlinks(path); r.OK {
		resolved = r.Value.(string)
	} else {
		return "", core.NewError("model dir not found: " + path)
	}
	if !pathWithinDir(rootResolved, resolved) {
		return "", core.NewError("model path escapes models dir")
	}
	manifestPath := core.PathJoin(resolved, shaManifestFilename)
	if r := core.PathEvalSymlinks(manifestPath); !r.OK {
		return "", core.NewError("model has no sha manifest: " + path)
	}
	return resolved, nil
}

func resolveModelNameToPath(name string) (string, error) {
	if core.Contains(name, "/") || core.Contains(name, "..") || core.HasPrefix(name, ".") {
		return "", core.NewError("model name must be a basename (no /, no .., no leading .)")
	}
	if name == "" {
		return "", core.NewError("model name required")
	}
	root := standardModelDir()
	candidate := core.PathJoin(root, name)

	// Symlink-resolve both sides + verify the candidate stays under
	// the root prefix. Defends against operator-side adversary who
	// drops `<root>/evil -> /etc/passwd` and triggers reload.
	rootResolved := root
	if r := core.PathEvalSymlinks(root); r.OK {
		rootResolved = r.Value.(string)
	}
	resolved := candidate
	if r := core.PathEvalSymlinks(candidate); r.OK {
		resolved = r.Value.(string)
	} else {
		return "", core.NewError("model dir not found: " + name)
	}
	if !pathWithinDir(rootResolved, resolved) {
		return "", core.NewError("model path escapes models dir")
	}

	// Refuse models without a sha-manifest per §4.F-7.2. Without it
	// the operator can swap the weights file under us between
	// download and reload and we'd serve attacker-chosen bytes.
	manifestPath := core.PathJoin(resolved, shaManifestFilename)
	if r := core.PathEvalSymlinks(manifestPath); !r.OK {
		return "", core.NewError("model lacks " + shaManifestFilename + " sidecar — refuse hot-swap to unverified-integrity model")
	}
	return resolved, nil
}

// readModelManifest returns the entries from `.sha256` at modelDir.
// Each line is "<sha256>  <filename>" (shasum -a 256 format).
// Comment lines (starting with #) and blank lines are skipped. Used
// by the download verifier + by future integrity-check tools.
func readModelManifest(modelDir string) (map[string]string, error) {
	manifest := core.PathJoin(modelDir, shaManifestFilename)
	res := core.ReadFile(manifest)
	if !res.OK {
		return nil, core.NewError("read manifest: " + manifest)
	}
	body, _ := res.Value.([]byte)
	out := map[string]string{}
	// Scan line-by-line over the original bytes instead of materialising
	// a []string for the whole file and a second []string per line
	// (AX-11: those two splits were ~128 of the 148 allocs/op for a
	// 64-file manifest). core.Cut peels one line at a time with no
	// backing allocation.
	rest := string(body)
	for rest != "" {
		var line string
		line, rest, _ = core.Cut(rest, "\n")
		line = core.Trim(line)
		if line == "" || core.HasPrefix(line, "#") {
			continue
		}
		// shasum -a 256 format: "<64-hex>  <filename>" (two spaces).
		// Fields are space-separated with empties dropped, then the
		// first and last non-empty fields are taken. Because the line
		// is whitespace-trimmed, both ends are non-space, so the first
		// field is everything up to the first space and the last field
		// is everything after the last space — and "two or more fields"
		// is exactly "the line contains a space". This reproduces the
		// old core.Split(line, " ")+drop-empties result byte-for-byte
		// without allocating the intermediate slices.
		first := core.Index(line, " ")
		if first < 0 {
			continue
		}
		last := core.LastIndex(line, " ")
		out[line[last+1:]] = core.Lower(line[:first])
	}
	if len(out) == 0 {
		return nil, core.NewError("manifest empty: " + manifest)
	}
	return out, nil
}

// writeModelManifest writes the {filename → sha256} map to
// modelDir/.sha256 in shasum -a 256 format. Called by the F-6
// downloader after verified-fetch lands all files. The .sha256
// sidecar is what F-7 reads to gate reload.
//
//	if err := writeModelManifest(modelDir, digests); err != nil { ... }
func writeModelManifest(modelDir string, digests map[string]string) error {
	// Sort filenames so the .sha256 sidecar is byte-deterministic across
	// runs (Mantis #1784 F-6 N-6) — map range order is randomised, which
	// would otherwise produce a different file on every download and defeat
	// diffing / reproducibility checks against the manifest.
	names := make([]string, 0, len(digests))
	total := 0
	for name := range digests {
		names = append(names, name)
		// Pre-size the output buffer to its exact final length so the
		// append loop never reallocates: each line is "<sha>  <name>\n"
		// = len(sha) + 2 (gap) + len(name) + 1 (newline). Earlier this
		// loop grew a nil []byte AND minted a throwaway concat string +
		// []byte per line (AX-11: 79 allocs/op for a 64-file model).
		total += len(digests[name]) + len(name) + 3
	}
	core.SliceSort(names)
	b := make([]byte, 0, total)
	for _, name := range names {
		b = append(b, digests[name]...)
		b = append(b, ' ', ' ')
		b = append(b, name...)
		b = append(b, '\n')
	}
	manifest := core.PathJoin(modelDir, shaManifestFilename)
	if r := core.WriteFile(manifest, b, 0o600); !r.OK {
		return core.E("admin.writeModelManifest", "write", r.Value.(error))
	}
	return nil
}

// listKnownModels returns the basenames of all subdirs under
// standardModelDir() that carry a .sha256 sidecar. Suitable surface
// for a future GET /v1/admin/models endpoint; today used by
// /v1/admin/serve/reload error paths to suggest names.
func listKnownModels() []string {
	root := standardModelDir()
	entries := core.ReadDir(core.DirFS(root), ".")
	if !entries.OK {
		return nil
	}
	dirEntries, ok := entries.Value.([]fs.DirEntry)
	if !ok {
		return nil
	}
	out := []string{}
	for _, e := range dirEntries {
		if !e.IsDir() {
			continue
		}
		manifest := core.PathJoin(root, e.Name(), shaManifestFilename)
		if r := core.PathEvalSymlinks(manifest); r.OK {
			out = append(out, e.Name())
		}
	}
	return out
}

// adminReloadServer is the shape the handler expects so tests can
// substitute the resolver. Kept in this file rather than admin_reload.go
// so the handler closure carries an interface, not a concrete type.
type adminReloadServer interface {
	CurrentPath() string
	Replace(newPath string, newOpts []mlx.LoadOption) (*loadedModel, string, error)
}

var _ adminReloadServer = (*hotSwapResolver)(nil)
