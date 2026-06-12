// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

// fixtureRoundTripper serves a canned HF tree response for any request —
// the download tests fake the hfTreeAPI interface, which left the REAL
// ResolveTree HTTP/decode path uncovered (a nil-body bug lived there from
// day one). This exercises the real implementation up to the wire.
type fixtureRoundTripper struct {
	status int
	body   string
}

func (f fixtureRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: f.status,
		Body:       io.NopCloser(strings.NewReader(f.body)),
		Header:     http.Header{},
	}, nil
}

// A trimmed real response from the HF tree API (?expand=true): rich unknown
// fields, a dotfile (deliberately dropped by isSafeHFEntryPath), an LFS
// entry, a config alongside it, a directory to skip, and a traversal path.
const hfTreeFixture = `[
  {"type":"file","oid":"52373fe2","size":1570,"path":".gitattributes","lastCommit":{"id":"903ae66f","title":"Add files","date":"2025-03-12T08:57:19.000Z"},"securityFileStatus":{"status":"safe"}},
  {"type":"file","path":"model.safetensors","size":4,"lfs":{"sha256":"abc123","size":806000000}},
  {"type":"file","path":"config.json","size":910},
  {"type":"directory","path":"assets"},
  {"type":"file","path":"../escape.bin","size":9}
]`

func TestResolveTree_RealDecodePath_Good(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: hfTreeFixture}}}

	entries, err := c.ResolveTree(context.Background(), "mlx-community/gemma-3-1b-it-4bit", "main")
	if err != nil {
		t.Fatalf("ResolveTree() error = %v", err)
	}
	// model.safetensors + config.json: directories skipped, the traversal
	// path dropped, and dotfiles rejected by design (isSafeHFEntryPath).
	if len(entries) != 2 {
		t.Fatalf("entries = %d, want 2: %+v", len(entries), entries)
	}
	if entries[0].Path != "model.safetensors" {
		t.Fatalf("entry[0] = %+v, want model.safetensors", entries[0])
	}
	if entries[1].Path != "config.json" || entries[1].Size != 910 {
		t.Fatalf("entry[1] = %+v, want config.json/910", entries[1])
	}
	for _, e := range entries {
		if strings.Contains(e.Path, "..") || strings.HasPrefix(e.Path, ".") {
			t.Fatalf("unsafe path survived: %s", e.Path)
		}
	}
}

func TestResolveTree_EmptyBody_Bad(t *testing.T) {
	c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: 200, body: ""}}}
	if _, err := c.ResolveTree(context.Background(), "org/repo", "main"); err == nil {
		t.Fatal("empty body decoded, want a loud decode error")
	}
}

func TestResolveTree_AuthStatuses_Bad(t *testing.T) {
	for _, status := range []int{401, 403} {
		c := &hfTreeClient{httpClient: &http.Client{Transport: fixtureRoundTripper{status: status, body: "denied"}}}
		_, err := c.ResolveTree(context.Background(), "org/gated", "main")
		if err == nil || !strings.Contains(err.Error(), "private repo or token") {
			t.Fatalf("status %d: err = %v, want the gated-repo hint", status, err)
		}
	}
}
