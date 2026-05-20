// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

func TestWriteSubset_Good(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "attention.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{
		"model.embed_tokens.weight":                  {1, 2, 3, 4},
		"model.layers.0.self_attn.q_proj.weight":     {5, 6, 7, 8},
		"model.layers.0.mlp.down_proj.weight":        {9, 10, 11, 12},
		"model.layers.0.self_attn.q_proj.weight.idx": {13, 14, 15, 16},
	})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}

	err = WriteSubset(context.Background(), target, []TensorRef{
		index.Tensors["model.embed_tokens.weight"],
		index.Tensors["model.layers.0.self_attn.q_proj.weight"],
	})
	if err != nil {
		t.Fatalf("WriteSubset: %v", err)
	}

	got, err := ReadIndex(target)
	if err != nil {
		t.Fatalf("ReadIndex(target): %v", err)
	}
	if len(got.Names) != 2 {
		t.Fatalf("names = %v, want two tensors", got.Names)
	}
	if _, ok := got.Tensors["model.layers.0.mlp.down_proj.weight"]; ok {
		t.Fatalf("target contains excluded MLP tensor: %v", got.Names)
	}
	assertRawTensorEqual(t, index.Tensors["model.embed_tokens.weight"], got.Tensors["model.embed_tokens.weight"])
	assertRawTensorEqual(t, index.Tensors["model.layers.0.self_attn.q_proj.weight"], got.Tensors["model.layers.0.self_attn.q_proj.weight"])
}

func TestWriteSubset_BadEmpty(t *testing.T) {
	err := WriteSubset(context.Background(), core.PathJoin(t.TempDir(), "empty.safetensors"), nil)

	if err == nil {
		t.Fatal("WriteSubset(nil) error = nil")
	}
}

func TestWriteSubset_UglyContextCancelled(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "cancelled.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = WriteSubset(ctx, target, []TensorRef{index.Tensors["x"]})

	if err == nil {
		t.Fatal("WriteSubset(cancelled) error = nil")
	}
}

func assertRawTensorEqual(t *testing.T, want, got TensorRef) {
	t.Helper()
	wantRaw, err := ReadRefRaw(want)
	if err != nil {
		t.Fatalf("ReadRefRaw(want): %v", err)
	}
	gotRaw, err := ReadRefRaw(got)
	if err != nil {
		t.Fatalf("ReadRefRaw(got): %v", err)
	}
	if string(wantRaw) != string(gotRaw) {
		t.Fatalf("raw tensor mismatch: want %v got %v", wantRaw, gotRaw)
	}
}

func writeRawSafetensors(t *testing.T, path string, tensors map[string][]byte) {
	t.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		raw := tensors[name]
		header[name] = HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(len(raw))},
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}
