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

// TestSubsetHeaderEncoded_ParityWithJSONMarshal anchors the hand-rolled
// JSON encoder against the reflection-driven core.JSONMarshal form. The
// W10-R refactor of subsetHeader → subsetHeaderEncoded swapped a
// map[string]HeaderEntry + JSONMarshal pipeline for a single byte
// append. This test fixes that "bit-exact" claim — any structural drift
// (key order, integer width, dtype canonicalisation, string escapes)
// would break model-extract round-trips and pack-time golden files.
func TestSubsetHeaderEncoded_ParityWithJSONMarshal(t *testing.T) {
	cases := []struct {
		name string
		refs []TensorRef
	}{
		{
			name: "single_2d_f32",
			refs: []TensorRef{
				{Name: "weight", DType: "F32", Shape: []uint64{2048, 2048}, ByteLen: 2048 * 2048 * 4},
			},
		},
		{
			name: "multi_dim_mix",
			refs: []TensorRef{
				{Name: "model.layers.0.self_attn.q_proj.weight", DType: "F16", Shape: []uint64{4, 28, 2048, 64}, ByteLen: 4 * 28 * 2048 * 64 * 2},
				{Name: "model.layers.0.self_attn.k_proj.weight", DType: "BF16", Shape: []uint64{4, 28, 2048, 64}, ByteLen: 4 * 28 * 2048 * 64 * 2},
				{Name: "alpha", DType: "U8", Shape: []uint64{16}, ByteLen: 16},
			},
		},
		{
			name: "lowercase_dtype_canonicalised",
			refs: []TensorRef{
				{Name: "x", DType: "f32", Shape: []uint64{4}, ByteLen: 16},
			},
		},
		{
			name: "single_one_dim",
			refs: []TensorRef{
				{Name: "bias", DType: "F32", Shape: []uint64{128}, ByteLen: 512},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, got, err := subsetHeaderEncoded(tc.refs)
			if err != nil {
				t.Fatalf("subsetHeaderEncoded: %v", err)
			}
			// Reference: build the same map[string]HeaderEntry the old
			// subsetHeader produced, then JSONMarshal it.
			byName := map[string]TensorRef{}
			names := make([]string, 0, len(tc.refs))
			for _, ref := range tc.refs {
				byName[ref.Name] = ref
				names = append(names, ref.Name)
			}
			core.SliceSort(names)
			header := make(map[string]HeaderEntry, len(names))
			var offset int64
			for _, name := range names {
				ref := byName[name]
				shape := make([]int64, len(ref.Shape))
				for i, d := range ref.Shape {
					shape[i] = int64(d)
				}
				header[name] = HeaderEntry{
					DType:       core.Upper(ref.DType),
					Shape:       shape,
					DataOffsets: []int64{offset, offset + ref.ByteLen},
				}
				offset += ref.ByteLen
			}
			encoded := core.JSONMarshal(header)
			if !encoded.OK {
				t.Fatalf("JSONMarshal reference: %v", encoded.Value)
			}
			want := encoded.Value.([]byte)
			if string(got) != string(want) {
				t.Fatalf("encoder drift:\n got=%s\nwant=%s", got, want)
			}
		})
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
