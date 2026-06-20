// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"net/http"
	"strings"
	"testing"

	core "dappco.re/go"
)

// errRoundTripper makes the HTTP client's Do return a transport error, so the
// tree GET fails at the network layer.
type errRoundTripper struct{}

func (errRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, core.NewError("dial tcp: connection refused")
}

// ResolveTree requires a non-empty repo.
func TestResolveTree_EmptyRepo_Bad(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: "[]"}}}
	if _, err := c.ResolveTree(context.Background(), "  ", "main"); err == nil {
		t.Fatal("ResolveTree(empty repo) error = nil, want repo-required")
	}
}

// ResolveTree defaults an empty revision to "main" and resolves successfully.
func TestResolveTree_DefaultsRevision_Good(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: `[{"type":"file","path":"config.json","size":12}]`}}}
	entries, err := c.ResolveTree(context.Background(), "org/repo", "")
	if err != nil {
		t.Fatalf("ResolveTree(empty revision) error = %v", err)
	}
	if len(entries) != 1 || entries[0].Path != "config.json" {
		t.Fatalf("entries = %+v, want one config.json entry resolved on main", entries)
	}
	// The composed resolve URL pins the default revision.
	if !strings.Contains(entries[0].URL, "/resolve/main/") {
		t.Fatalf("entry URL = %q, want the default 'main' revision composed in", entries[0].URL)
	}
}

// ResolveTree surfaces a transport-level GET error.
func TestResolveTree_GETError_Bad(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: errRoundTripper{}}}
	if _, err := c.ResolveTree(context.Background(), "org/repo", "main"); err == nil {
		t.Fatal("ResolveTree(transport error) error = nil, want the surfaced GET error")
	}
}

// ResolveTree maps a 5xx (or any >=400 non-auth) status to a tree-API error.
func TestResolveTree_ServerError_Bad(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 500, body: "boom"}}}
	_, err := c.ResolveTree(context.Background(), "org/repo", "main")
	if err == nil || !strings.Contains(err.Error(), "tree API") {
		t.Fatalf("ResolveTree(500) error = %v, want the tree-API status error", err)
	}
}

// ResolveTree carries a body preview when the JSON decode fails (the
// "contract drift?" diagnostic that needs evidence).
func TestResolveTree_MalformedJSON_Bad(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: "<html>not json</html>"}}}
	_, err := c.ResolveTree(context.Background(), "org/repo", "main")
	if err == nil {
		t.Fatal("ResolveTree(malformed JSON) error = nil, want a decode error")
	}
	if !strings.Contains(err.Error(), "body starts") {
		t.Fatalf("error = %v, want the body-preview diagnostic", err)
	}
}

// ResolveTree fills LFS digest + size when the entry advertises lfs.sha256 and
// has a zero top-level size (the LFS-size backfill branch).
func TestResolveTree_LFSBackfill_Good(t *testing.T) {
	body := `[{"type":"file","path":"model.safetensors","size":0,"lfs":{"sha256":"ABC123","size":4096}}]`
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: body}}}
	entries, err := c.ResolveTree(context.Background(), "org/repo", "main")
	if err != nil {
		t.Fatalf("ResolveTree error = %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(entries))
	}
	if entries[0].Digest != "abc123" {
		t.Fatalf("digest = %q, want lowercased lfs sha256", entries[0].Digest)
	}
	if entries[0].Size != 4096 {
		t.Fatalf("size = %d, want the lfs size backfilled", entries[0].Size)
	}
}

// isSafeHFEntryPath rejects traversal/absolute/NUL/dotfile segments and accepts
// a plain repo-relative path.
func TestIsSafeHFEntryPath_Coverage_Good(t *testing.T) {
	for _, bad := range []string{"", "/abs/path", "a/../b", "../escape", ".gitattributes", "dir/.hidden", "with\x00nul"} {
		if isSafeHFEntryPath(bad) {
			t.Fatalf("isSafeHFEntryPath(%q) = true, want false (unsafe)", bad)
		}
	}
	for _, ok := range []string{"model.safetensors", "subdir/weights.bin"} {
		if !isSafeHFEntryPath(ok) {
			t.Fatalf("isSafeHFEntryPath(%q) = false, want true (safe)", ok)
		}
	}
}
