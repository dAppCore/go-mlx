// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"bytes"
	"io"
	"testing"
)

// bytesRWS wraps a bytes.Buffer to satisfy io.ReadWriteSeeker.
// bytes.Buffer only provides Read and Write; this adds Seek support.
type bytesRWS struct {
	data []byte
	pos  int
	end  int
}

func newBytesRWS(initial []byte) *bytesRWS {
	cp := make([]byte, len(initial))
	copy(cp, initial)
	return &bytesRWS{data: cp, pos: 0, end: len(cp)}
}

func newBytesRWSSize(size int) *bytesRWS {
	return &bytesRWS{data: make([]byte, size), pos: 0, end: 0}
}

func (b *bytesRWS) Read(p []byte) (int, error) {
	if b.pos >= b.end {
		return 0, io.EOF
	}
	n := copy(p, b.data[b.pos:b.end])
	b.pos += n
	return n, nil
}

func (b *bytesRWS) Write(p []byte) (int, error) {
	// Grow if needed
	needed := b.pos + len(p)
	if needed > len(b.data) {
		grown := make([]byte, needed)
		copy(grown, b.data)
		b.data = grown
	}
	n := copy(b.data[b.pos:], p)
	b.pos += n
	if b.pos > b.end {
		b.end = b.pos
	}
	return n, nil
}

func (b *bytesRWS) Seek(offset int64, whence int) (int64, error) {
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = int64(b.pos) + offset
	case io.SeekEnd:
		newPos = int64(b.end) + offset
	default:
		return 0, io.ErrUnexpectedEOF
	}
	if newPos < 0 {
		return 0, io.ErrUnexpectedEOF
	}
	b.pos = int(newPos)
	return newPos, nil
}

func (b *bytesRWS) Bytes() []byte {
	return b.data[:b.end]
}

func TestBytesRWS_BytesUsesHighWaterMark_Good(t *testing.T) {
	buf := newBytesRWSSize(4)
	if _, err := buf.Write([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := buf.Seek(1, io.SeekStart); err != nil {
		t.Fatalf("Seek: %v", err)
	}
	if got := buf.Bytes(); !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatalf("Bytes() = %v, want full high-water contents", got)
	}
}

// --- Good: Round-trip through custom I/O ---

func TestIOCustom_RoundTrip_Good(t *testing.T) {
	// Create some tensors to save.
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{10, 20, 30}, 3)
	Materialize(a, b)

	tensors := map[string]*Array{
		"weight": a,
		"bias":   b,
	}

	// Save to in-memory buffer.
	buf := newBytesRWSSize(8192)
	err := SaveSafetensorsToWriter(buf, 8192, "test-memory", tensors, nil)
	if err != nil {
		t.Fatalf("SaveSafetensorsToWriter: %v", err)
	}

	written := buf.Bytes()
	if len(written) == 0 {
		t.Fatal("nothing written to buffer")
	}

	// Load back from the same bytes.
	reader := newBytesRWS(written)
	loaded, err := LoadAllSafetensorsFromReader(reader, int64(len(written)), "test-memory")
	if err != nil {
		t.Fatalf("LoadAllSafetensorsFromReader: %v", err)
	}

	if len(loaded) != 2 {
		t.Fatalf("loaded %d tensors, want 2", len(loaded))
	}

	// Verify weight tensor.
	w, ok := loaded["weight"]
	if !ok {
		t.Fatal("missing 'weight' tensor")
	}
	Materialize(w)
	if w.Size() != 4 {
		t.Errorf("weight size = %d, want 4", w.Size())
	}
	wShape := w.Shape()
	if wShape[0] != 2 || wShape[1] != 2 {
		t.Errorf("weight shape = %v, want [2 2]", wShape)
	}
	floatSliceApprox(t, w.Floats(), []float32{1, 2, 3, 4})

	// Verify bias tensor.
	bi, ok := loaded["bias"]
	if !ok {
		t.Fatal("missing 'bias' tensor")
	}
	Materialize(bi)
	floatSliceApprox(t, bi.Floats(), []float32{10, 20, 30})
}

// --- Good: Round-trip with metadata ---

func TestIOCustom_WithMetadata_Good(t *testing.T) {
	a := FromValues([]float32{42}, 1)
	Materialize(a)

	tensors := map[string]*Array{"val": a}
	meta := map[string]string{"format": "pt", "version": "1"}

	buf := newBytesRWSSize(4096)
	err := SaveSafetensorsToWriter(buf, 4096, "meta-test", tensors, meta)
	if err != nil {
		t.Fatalf("save with metadata: %v", err)
	}

	written := buf.Bytes()
	reader := newBytesRWS(written)
	loaded := make(map[string]*Array)
	for name, arr := range LoadSafetensorsFromReader(reader, int64(len(written)), "meta-test") {
		loaded[name] = arr
	}

	if len(loaded) != 1 {
		t.Fatalf("loaded %d tensors, want 1", len(loaded))
	}
	v, ok := loaded["val"]
	if !ok {
		t.Fatal("missing 'val' tensor")
	}
	Materialize(v)
	floatSliceApprox(t, v.Floats(), []float32{42})
}

// --- Bad: Empty reader produces zero tensors ---

func TestIOCustom_EmptyReader_Bad(t *testing.T) {
	empty := newBytesRWS([]byte{})
	loaded, err := LoadAllSafetensorsFromReader(empty, 0, "empty")
	if err == nil {
		t.Error("expected error loading from empty reader")
	}
	if loaded != nil && len(loaded) > 0 {
		t.Error("expected no tensors from empty reader")
	}
}

// --- Bad: Corrupt data produces error ---

func TestIOCustom_CorruptData_Bad(t *testing.T) {
	garbage := bytes.Repeat([]byte{0xFF}, 256)
	reader := newBytesRWS(garbage)
	loaded, err := LoadAllSafetensorsFromReader(reader, int64(len(garbage)), "corrupt")
	if err == nil {
		t.Error("expected error loading corrupt safetensors data")
	}
	if loaded != nil && len(loaded) > 0 {
		t.Error("expected no tensors from corrupt data")
	}
}

// --- Ugly: Iterator break mid-stream ---

func TestIOCustom_IteratorBreak_Ugly(t *testing.T) {
	// Create multiple tensors.
	a := FromValues([]float32{1, 2}, 2)
	b := FromValues([]float32{3, 4}, 2)
	c := FromValues([]float32{5, 6}, 2)
	Materialize(a, b, c)

	tensors := map[string]*Array{"a": a, "b": b, "c": c}
	buf := newBytesRWSSize(8192)
	err := SaveSafetensorsToWriter(buf, 8192, "break-test", tensors, nil)
	if err != nil {
		t.Fatalf("save: %v", err)
	}

	written := buf.Bytes()
	reader := newBytesRWS(written)

	// Break after first tensor -- should not panic or leak.
	count := 0
	for range LoadSafetensorsFromReader(reader, int64(len(written)), "break-test") {
		count++
		break
	}
	if count != 1 {
		t.Errorf("expected exactly 1 iteration before break, got %d", count)
	}
}
