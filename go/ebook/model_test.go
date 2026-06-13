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

func TestSafetensorsStats_Good(t *testing.T) {
	tensors, elements, ok := safetensorsStats(tinySafetensors())
	if !ok || tensors != 1 || elements != 6 {
		t.Fatalf("stats = (%d tensors, %d elements, ok=%v), want (1, 6, true)", tensors, elements, ok)
	}
	if _, _, ok := safetensorsStats([]byte{1, 2, 3}); ok {
		t.Fatal("garbage must not parse as a safetensors header")
	}
}

func TestGrouped_Good(t *testing.T) {
	cases := map[int64]string{0: "0", 42: "42", 1000: "1,000", 999888777: "999,888,777", -1234: "-1,234"}
	for n, want := range cases {
		if got := grouped(n); got != want {
			t.Fatalf("grouped(%d) = %q, want %q", n, got, want)
		}
	}
}
