// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import (
	"archive/zip"
	"bytes"
	"io"
	"testing"
)

func readZipEntry(t *testing.T, zr *zip.Reader, name string) string {
	t.Helper()
	for _, f := range zr.File {
		if f.Name == name {
			rc, err := f.Open()
			if err != nil {
				t.Fatalf("open %s: %v", name, err)
			}
			defer rc.Close()
			data, err := io.ReadAll(rc)
			if err != nil {
				t.Fatalf("read %s: %v", name, err)
			}
			return string(data)
		}
	}
	t.Fatalf("entry %s not found in epub", name)
	return ""
}

// A valid EPUB3: mimetype first and stored, container points at the OPF, the
// OPF carries the metadata + every chapter, nav lists only nav chapters, and
// each chapter is its own well-formed file.
func TestWriteEPUB_ValidContainer_Good(t *testing.T) {
	b := &Book{
		Title:  "LEM & friends <test>",
		Author: "Lethean",
		Chapters: []Chapter{
			{ID: "ch000", Title: "Title", Body: "<h1>Hi</h1>", InNav: true},
			{ID: "plate0001", Title: "Plate 1", Body: "<pre>QUJD</pre>", InNav: false},
			{ID: "ch999", Title: "Colophon", Body: "<p>end</p>", InNav: true},
		},
	}
	var buf bytes.Buffer
	if err := b.WriteEPUB(&buf); err != nil {
		t.Fatalf("WriteEPUB: %v", err)
	}
	zr, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("epub is not a valid zip: %v", err)
	}

	// mimetype MUST be the first entry, stored, exact content.
	if zr.File[0].Name != "mimetype" {
		t.Fatalf("first entry = %q, want mimetype", zr.File[0].Name)
	}
	if zr.File[0].Method != zip.Store {
		t.Fatal("mimetype must be stored uncompressed")
	}
	const wantMimetype = "application/epub+zip" // EPUB3 spec-mandated content type
	if got := readZipEntry(t, zr, "mimetype"); got != wantMimetype {
		t.Fatalf("mimetype content = %q, want %q", got, wantMimetype)
	}

	readZipEntry(t, zr, "META-INF/container.xml")
	opf := readZipEntry(t, zr, "OEBPS/content.opf")
	for _, want := range []string{"<dc:title>LEM &amp; friends &lt;test&gt;</dc:title>", "<dc:rights>EUPL-1.2</dc:rights>", `idref="plate0001"`, "dcterms:modified"} {
		if !bytes.Contains([]byte(opf), []byte(want)) {
			t.Fatalf("OPF missing %q", want)
		}
	}
	nav := readZipEntry(t, zr, "OEBPS/nav.xhtml")
	if bytes.Contains([]byte(nav), []byte("plate0001")) {
		t.Fatal("non-nav plate leaked into the table of contents")
	}
	if !bytes.Contains([]byte(nav), []byte("ch999.xhtml")) {
		t.Fatal("nav chapter missing from the table of contents")
	}
	readZipEntry(t, zr, "OEBPS/ch000.xhtml")
	readZipEntry(t, zr, "OEBPS/plate0001.xhtml")
}

func TestWriteEPUB_EmptyRejected_Bad(t *testing.T) {
	var buf bytes.Buffer
	if err := (&Book{Title: "x"}).WriteEPUB(&buf); err == nil {
		t.Fatal("a book with no chapters must be refused")
	}
}

// xmlEscape itself now lives in modelmgmt (TestEbook_xmlEscape_Good/_Ugly in
// dappco.re/go/inference/modelmgmt) — WriteEPUB above already exercises it
// end to end via the escaped title/rights assertions.
