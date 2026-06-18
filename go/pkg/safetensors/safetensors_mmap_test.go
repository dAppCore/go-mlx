// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"bytes"
	"testing"
	"unsafe"

	coreio "dappco.re/go/io"
)

// TestLoadMmapZeroCopy round-trips a synthetic checkpoint through Encode → file → LoadMmap and
// proves the key property: each Tensor.Data is a VIEW into the page-aligned mmap, not a heap
// copy. That view-into-an-aligned-base is exactly what the no-copy GPU buffer path needs. No
// model load — AX-11 synthetic.
func TestLoadMmapZeroCopy(t *testing.T) {
	want := map[string]Tensor{
		"a.weight": {Dtype: "F32", Shape: []int{2, 2}, Data: []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}},
		"b.scales": {Dtype: "BF16", Shape: []int{4}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	blob, err := Encode(want)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	path := t.TempDir() + "/m.safetensors"
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	m, err := LoadMmap(path)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}
	defer m.Close()

	if len(m.Data) != len(blob) {
		t.Fatalf("mapped %d bytes, want %d", len(m.Data), len(blob))
	}
	base := uintptr(unsafe.Pointer(&m.Data[0]))
	end := base + uintptr(len(m.Data))
	for name, w := range want {
		got, ok := m.Tensors[name]
		if !ok {
			t.Fatalf("missing tensor %s", name)
		}
		if got.Dtype != w.Dtype || !bytes.Equal(got.Data, w.Data) {
			t.Fatalf("tensor %s content mismatch", name)
		}
		// The whole point: Data must be a VIEW into the mmap, not a heap copy.
		ptr := uintptr(unsafe.Pointer(&got.Data[0]))
		if ptr < base || ptr >= end {
			t.Fatalf("tensor %s Data is a copy, not a view into the mmap — zero-copy broken", name)
		}
	}
	t.Logf("LoadMmap: %d tensors view the %d-byte page-aligned mmap (zero-copy)", len(m.Tensors), len(m.Data))
}
