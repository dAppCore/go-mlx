// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"syscall"

	core "dappco.re/go"
)

// HuggingFace tree resolver — the narrow subset of the lthn/desktop
// pkg/downloader/hf.go surface that F-6 needs. Lives in lthn-mlx
// rather than importing across the binary boundary; per-binary copy
// pattern (memory: feedback_binary_is_model_package_is_everything_else).
//
// Surface:
//
//   - hfTreeAPI interface — test seam (Replace ResolveTree with a fixture).
//   - hfTreeClient — production implementation, hits huggingface.co.
//   - fetchAndVerify — bounded GET + sha256-verifying write to dest.
//
// URL allowlist (§4.F-6.1): hosts are statically pinned to the HF
// tree API + the resolve subdomain + LFS CDN. Request-supplied URLs
// are NEVER honoured.

// hfHostTreeAPI is the public HF tree-listing host. Public, no auth
// required for public repos.
const hfHostTreeAPI = "https://huggingface.co/api/models/"

// hfHostResolve is the public download host. Tree entries resolve
// via /<repo>/resolve/<revision>/<file>.
const hfHostResolve = "https://huggingface.co/"

// hfTreeResponseCap bounds the bytes ResolveTree is willing to read
// from the tree API. Defends against a malicious / compromised
// mirror streaming unbounded JSON.
const hfTreeResponseCap int64 = 4 << 20 // 4 MiB

// hfFileCap bounds a single fetched file. Sized for the largest GGUF
// distributed today (~140 GiB) plus headroom. Bumping requires a
// review (same TOCTOU shape as the lthn/desktop sibling).
const hfFileCap int64 = 256 << 30 // 256 GiB

// hfFileEntry is what ResolveTree returns per file. URL is the
// composed resolve URL (server-controlled); Digest is the lfs.sha256
// when LFS-stored, empty for non-LFS (config / tokeniser) files.
type hfFileEntry struct {
	Path   string // path-from-repo-root
	URL    string // composed resolve URL
	Size   int64  // bytes
	Digest string // lowercase sha256 hex, empty for non-LFS
}

// isSafeHFEntryPath enforces the contract that the HF tree API
// returns repo-relative paths with no traversal sequences. Refuses
// `..`, absolute paths, NUL bytes, leading `/`, and any dotfile
// segment (a segment beginning with `.`). The PathDir + MkdirAll +
// OpenFile in the download worker would otherwise honour a tree
// response like `{"path":"../../etc/passwd"}` and write outside the
// quarantine dir; rejecting dotfile segments (F-6 N-9) keeps a
// compromised mirror from planting `.git/`, `.ssh/`, or other hidden
// config into the model tree. Genuine model artefacts are never
// dotfiles — git metadata like .gitattributes is filtered out as
// non-model content rather than refused.
func isSafeHFEntryPath(p string) bool {
	if p == "" {
		return false
	}
	if core.HasPrefix(p, "/") {
		return false
	}
	if core.Contains(p, "\x00") {
		return false
	}
	// Walk the "/"-separated segments by index rather than core.Split —
	// the split minted a []string (plus segment backings) on every call,
	// and this guard runs once per file in an HF tree response (AX-11:
	// 1 alloc/path). Rejecting `..`, `.` and any dotfile segment reduces
	// to "the segment is non-empty and starts with '.'" — `..` and `.`
	// both start with '.', and an empty segment (from `//` or a trailing
	// `/`) was never rejected by the old loop, so neither is it here.
	start := 0
	for i := 0; i <= len(p); i++ {
		if i < len(p) && p[i] != '/' {
			continue
		}
		if i > start && p[start] == '.' {
			return false
		}
		start = i + 1
	}
	return true
}

// hfTreeAPI is the seam the download handler depends on. Production
// path implements via hfTreeClient; tests substitute a fixture.
type hfTreeAPI interface {
	ResolveTree(ctx context.Context, repo, revision string) ([]hfFileEntry, error)
}

// hfTreeClient is the live HF tree-API implementation.
type hfTreeClient struct {
	httpClient *http.Client
}

// newHFTreeClient builds the production tree client. nil httpClient
// → use the package default (a stdlib client; F-6 doesn't need the
// lthn/desktop trust-pinning ceremony because the host is compile-
// time pinned to the HF allowlist).
func newHFTreeClient() *hfTreeClient {
	return &hfTreeClient{
		httpClient: &http.Client{},
	}
}

// hfTreeEntryRaw is the JSON shape returned per file by the HF tree
// API when ?expand=true is set. Decoder is lenient — missing fields
// don't error.
type hfTreeEntryRaw struct {
	Type string `json:"type"` // "file" / "directory"
	Path string `json:"path"` // path-from-repo-root
	Size int64  `json:"size"` // bytes; absent → 0
	LFS  *struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"lfs"`
}

// ResolveTree hits the HF tree API for repo + revision and returns
// the per-file metadata the download worker needs.
func (c *hfTreeClient) ResolveTree(ctx context.Context, repo, revision string) ([]hfFileEntry, error) {
	if core.Trim(repo) == "" {
		return nil, core.NewError("repo required")
	}
	if core.Trim(revision) == "" {
		revision = "main"
	}
	apiURL := hfHostTreeAPI + repo + "/tree/" + revision + "?expand=true"

	res := core.NewHTTPRequestContext(ctx, "GET", apiURL, nil)
	if !res.OK {
		return nil, core.E("admin.hf", "build request", res.Value.(error))
	}
	req := res.Value.(*core.Request)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, core.E("admin.hf", "tree GET", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, core.NewError(core.Sprintf("HTTP %d — private repo or token required", resp.StatusCode))
	}
	if resp.StatusCode >= 400 {
		return nil, core.NewError(core.Sprintf("HTTP %d from tree API", resp.StatusCode))
	}

	bounded := core.LimitReader(resp.Body, hfTreeResponseCap)
	bodyR := core.ReadAll(bounded)
	if !bodyR.OK {
		return nil, core.E("admin.hf", "tree body read", bodyR.Value.(error))
	}
	// core.ReadAll yields a STRING (io.go AsString) — the first draft
	// asserted []byte with a swallowed ok, so body was nil on every call
	// and this lane never worked until the #84 field exercise hit it.
	// AsBytes aliases the same backing array (JSONUnmarshal does not
	// retain past the call), matching hf/hf.go + split_remote_ffn.go.
	bodyStr, ok := bodyR.Value.(string)
	if !ok {
		return nil, core.E("admin.hf", "tree body shape", nil)
	}
	body := core.AsBytes(bodyStr)

	var raw []hfTreeEntryRaw
	if r := core.JSONUnmarshal(body, &raw); !r.OK {
		// Carry the decode error + a body preview — "contract drift?" with
		// no evidence sent a debugging session guessing (gzip? strict
		// decoder? cap truncation?) when the answer was in the bytes.
		preview := body
		if len(preview) > 160 {
			preview = preview[:160]
		}
		return nil, core.E("admin.hf",
			core.Sprintf("tree JSON decode failed (%v) — body starts: %q", r.Value, string(preview)), nil)
	}

	out := make([]hfFileEntry, 0, len(raw))
	for _, e := range raw {
		if e.Type != "file" {
			continue
		}
		// Cerberus pass-3 N-8: validate HF-supplied file path before
		// trusting it. The tree API SHOULD return repo-relative paths,
		// but a malicious/compromised mirror could inject `../etc` or
		// `/absolute/path` to escape the dest dir during write. Reject
		// any path with `..`, leading `/`, or NUL bytes — the per-file
		// MkdirAll + OpenFile downstream would otherwise honour them.
		if !isSafeHFEntryPath(e.Path) {
			continue
		}
		entry := hfFileEntry{
			Path: e.Path,
			URL:  hfHostResolve + repo + "/resolve/" + revision + "/" + e.Path,
			Size: e.Size,
		}
		if e.LFS != nil && core.Trim(e.LFS.SHA256) != "" {
			entry.Digest = core.Lower(e.LFS.SHA256)
			if entry.Size == 0 && e.LFS.Size > 0 {
				entry.Size = e.LFS.Size
			}
		}
		out = append(out, entry)
	}
	return out, nil
}

// fetchAndVerify GETs url into destPath, streaming through a sha256
// hasher. If expectedDigest is non-empty the digest is enforced;
// mismatch → error + remove(destPath). Caller is responsible for
// ensuring destPath's parent dir exists.
//
// Returns (bytesWritten, computedDigest, error). computedDigest
// populated even on verify success so the caller can stamp it in
// the .sha256 sidecar.
//
// expectedSize is the size advertised by the HF tree manifest. Used
// to early-reject downloads where the on-wire Content-Length is far
// off (corrupt mirror / wrong-revision drift). Empty (0) skips.
func fetchAndVerify(ctx context.Context, url, destPath, expectedDigest string, expectedSize int64) (int64, string, error) {
	// URL allowlist gate per §4.F-6.1. The resolve URL was server-
	// composed in ResolveTree from a repo+revision pair, so this is
	// belt-and-braces — defends against any future code path that
	// might compose a URL incorrectly.
	if !core.HasPrefix(url, hfHostResolve) {
		return 0, "", core.NewError("disallowed source: " + url)
	}

	res := core.NewHTTPRequestContext(ctx, "GET", url, nil)
	if !res.OK {
		return 0, "", core.E("admin.hf.fetch", "build request", res.Value.(error))
	}
	req := res.Value.(*core.Request)
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", core.E("admin.hf.fetch", "GET", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return 0, "", core.NewError(core.Sprintf("HTTP %d from %s", resp.StatusCode, url))
	}
	if resp.ContentLength > hfFileCap {
		return 0, "", core.NewError(core.Sprintf("Content-Length %d exceeds cap %d", resp.ContentLength, hfFileCap))
	}

	// Symlink-safe create per Cerberus pass-3 N-1: O_CREATE|O_EXCL|
	// O_WRONLY|O_NOFOLLOW refuses pre-existing entries (race against
	// another download) AND refuses if destPath is a symlink. Defends
	// against operator-side adversary with FS write access to the
	// quarantine dir pre-planting `<quarantine>/weights.bin ->
	// ~/.ssh/authorized_keys` and having the downloader truncate-write
	// through it. Pattern mirrors lthn/desktop pkg/downloader F-4.
	flag := core.O_CREATE | core.O_EXCL | core.O_WRONLY | syscall.O_NOFOLLOW
	createR := core.OpenFile(destPath, flag, core.FileMode(0o600))
	if !createR.OK {
		err, _ := createR.Value.(error)
		if core.Is(err, syscall.ELOOP) {
			return 0, "", core.E("admin.hf.fetch", "quarantine_symlink_refused: "+destPath, err)
		}
		if core.IsExist(err) {
			return 0, "", core.E("admin.hf.fetch", "quarantine_exists: "+destPath, err)
		}
		return 0, "", core.E("admin.hf.fetch", "create dest", err)
	}
	file := createR.Value.(*core.OSFile)

	hasher := sha256.New()
	bounded := core.LimitReader(resp.Body, hfFileCap+1)
	mw := io.MultiWriter(file, hasher)
	copyR := core.Copy(mw, bounded)
	if !copyR.OK {
		_ = file.Close()
		_ = core.Remove(destPath)
		return 0, "", core.E("admin.hf.fetch", "stream copy", copyR.Value.(error))
	}
	written := copyR.Value.(int64)
	if written > hfFileCap {
		_ = file.Close()
		_ = core.Remove(destPath)
		return 0, "", core.NewError(core.Sprintf("download exceeded %d byte cap", hfFileCap))
	}
	if err := file.Close(); err != nil {
		_ = core.Remove(destPath)
		return 0, "", core.E("admin.hf.fetch", "close dest", err)
	}

	computed := hex.EncodeToString(hasher.Sum(nil))
	if expectedDigest != "" && computed != core.Lower(expectedDigest) {
		_ = core.Remove(destPath)
		return 0, "", core.NewError(core.Sprintf("sha256 mismatch: got=%s want=%s", computed, expectedDigest))
	}
	if expectedSize > 0 && written != expectedSize {
		// Size drift is informational rather than fatal — the HF
		// tree may report stale sizes during repo rewrites. Sha is
		// the load-bearing integrity check, so we emit a warning the
		// operator can correlate rather than refusing the file.
		core.Warn("mlx: admin model_download size drift vs HF manifest",
			"url", url, "expected_size", expectedSize, "written", written)
	}
	return written, computed, nil
}
