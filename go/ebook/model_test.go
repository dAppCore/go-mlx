// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import (
	"encoding/base64"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// tinySafetensors builds a valid safetensors blob: one F32 tensor of shape
// [2,3] (6 scalars, 24 bytes of data).
func tinySafetensors() []byte {
	header := `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(len(header)))
	out := append(prefix, []byte(header)...)
	data := make([]byte, 24)
	for i := range data {
		data[i] = byte(i * 7)
	}
	return append(out, data...)
}

func writeFixtureModel(t *testing.T) (dir string, weights []byte) {
	t.Helper()
	dir = core.JoinPath(t.TempDir(), "LEM-Tiny")
	if err := coreio.Local.EnsureDir(dir); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), `{"model_type":"gemma3_text","hidden_size":1152}`); err != nil {
		t.Fatalf("config: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "README.md"), "# LEM-Tiny\nThe loyal one.\n"); err != nil {
		t.Fatalf("readme: %v", err)
	}
	weights = tinySafetensors()
	if err := coreio.Local.Write(core.JoinPath(dir, "model.safetensors"), string(weights)); err != nil {
		t.Fatalf("weights: %v", err)
	}
	return dir, weights
}

// plateBase64 concatenates the base64 out of every plate chapter, in order —
// the reconstruction a reader would perform.
func plateBase64(chapters []Chapter) string {
	var out core.Builder
	for i := range chapters {
		ch := &chapters[i]
		if !core.HasPrefix(ch.ID, "plate") {
			continue
		}
		body := ch.Body
		start := core.Index(body, "<pre>")
		end := core.Index(body, "</pre>")
		if start < 0 || end < 0 {
			continue
		}
		out.WriteString(body[start+len("<pre>") : end])
	}
	return out.String()
}

// The load-bearing test: the weights survive the round trip. Decode the plates
// back and you have the original safetensors, byte for byte — speech that
// compiles.
func TestBuildModelBook_RoundTripsWeights_Good(t *testing.T) {
	dir, weights := writeFixtureModel(t)
	book, err := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: 16})
	if err != nil {
		t.Fatalf("BuildModelBook: %v", err)
	}
	if book.Title != "LEM-Tiny" {
		t.Fatalf("title = %q, want the dir name", book.Title)
	}

	decoded, err := base64.StdEncoding.DecodeString(plateBase64(book.Chapters))
	if err != nil {
		t.Fatalf("plates did not base64-decode: %v", err)
	}
	if len(decoded) != len(weights) {
		t.Fatalf("reconstructed %d bytes, want %d", len(decoded), len(weights))
	}
	for i := range weights {
		if decoded[i] != weights[i] {
			t.Fatalf("reconstruction differs at byte %d", i)
		}
	}

	// Byte-identity at the time-independent layer: the concatenated plate
	// base64 must equal the canonical EncodeToString of the original weights.
	// (The whole EPUB embeds wall-clock timestamps and is not reproducible; the
	// plate base64 is a pure function of the input bytes, and is what the
	// protected-speech reproducibility claim rests on.)
	if got, want := plateBase64(book.Chapters), base64.StdEncoding.EncodeToString(weights); got != want {
		t.Fatalf("plate base64 is not the canonical encoding of the weights")
	}

	// Small ChapterChars must split into several plates (proves chunking).
	plates := 0
	for i := range book.Chapters {
		if core.HasPrefix(book.Chapters[i].ID, "plate") {
			plates++
		}
	}
	if plates < 2 {
		t.Fatalf("plates = %d, want >1 with ChapterChars=16", plates)
	}

	// Foreword carried the README.
	foreword := book.Chapters[1]
	if !core.Contains(foreword.Body, "The loyal one.") {
		t.Fatal("foreword did not carry the README")
	}
}

func TestBuildModelBook_NoWeights_Good(t *testing.T) {
	dir, _ := writeFixtureModel(t)
	book, err := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: false})
	if err != nil {
		t.Fatalf("BuildModelBook: %v", err)
	}
	for i := range book.Chapters {
		if core.HasPrefix(book.Chapters[i].ID, "plate") {
			t.Fatal("no-weights edition must contain no plates")
		}
	}
	// Method chapter should say the weights were omitted.
	method := book.Chapters[2]
	if !core.Contains(method.Body, "omitted") {
		t.Fatal("method chapter should note the weights are omitted")
	}
}

func TestBuildModelBook_EmptyDir_Bad(t *testing.T) {
	dir := core.JoinPath(t.TempDir(), "empty")
	if err := coreio.Local.EnsureDir(dir); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if _, err := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true}); err == nil {
		t.Fatal("a dir with no safetensors must be refused")
	}
}

// A malformed/sub-8-byte .safetensors is not a parseable header, but it is
// still bytes — the book must encode it (no stats, no panic) and round-trip it
// exactly, just like a valid file. This guards the streamed read/encode path
// against the short-file edge case.
func TestBuildModelBook_MalformedWeights_Ugly(t *testing.T) {
	dir := core.JoinPath(t.TempDir(), "LEM-Malformed")
	if err := coreio.Local.EnsureDir(dir); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	garbage := []byte{1, 2, 3} // < 8 bytes: not a safetensors header
	if err := coreio.Local.Write(core.JoinPath(dir, "model.safetensors"), string(garbage)); err != nil {
		t.Fatalf("weights: %v", err)
	}
	book, err := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: 16})
	if err != nil {
		t.Fatalf("BuildModelBook over a malformed file must still succeed: %v", err)
	}
	if got, want := plateBase64(book.Chapters), base64.StdEncoding.EncodeToString(garbage); got != want {
		t.Fatalf("malformed file not encoded byte-identically: got %q want %q", got, want)
	}
}

// safetensorsStats and grouped now live in modelmgmt (as ebookSafetensorsStats
// and grouped — see TestEbookModel_ebookSafetensorsStats_Good/_Ugly and
// TestEbookModel_grouped_Good/_Ugly in dappco.re/go/inference/modelmgmt);
// TestBuildModelBook_RoundTripsWeights_Good above already exercises both
// through the public BuildModelBook contract.

// TestBuildModelBook_ColophonCreditsCLI_Good locks in the one deliberate
// divergence from modelmgmt.BuildModelBook: the colophon must credit the
// command a reader actually ran (`lthn-mlx ebook`), not modelmgmt's own
// library entry point. See the DIVERGENCE note on BuildModelBook in model.go.
func TestBuildModelBook_ColophonCreditsCLI_Good(t *testing.T) {
	dir, _ := writeFixtureModel(t)
	book, err := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: false})
	if err != nil {
		t.Fatalf("BuildModelBook: %v", err)
	}
	colophon := book.Chapters[len(book.Chapters)-1]
	if !core.Contains(colophon.Body, "lthn-mlx ebook") {
		t.Fatalf("colophon must credit lthn-mlx ebook, got: %s", colophon.Body)
	}
	if core.Contains(colophon.Body, "modelmgmt.BuildModelBook") {
		t.Fatal("colophon leaked modelmgmt's library-entry-point credit line")
	}
}
