// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

func TestMetadata_Metadata_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "gemma4"},
			{Key: "general.name", ValueType: ValueTypeString, Value: "metadata-good"},
			{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)},
			{Key: "general.alignment", ValueType: ggufValueTypeUint64, Value: uint64(32)},
			{Key: "general.use_mlock", ValueType: ggufValueTypeBool, Value: true},
			{Key: "gemma4-assistant.tokens", ValueType: ggufValueTypeArray, Value: ggufArraySpec{ElementType: ValueTypeString, Values: []any{"<bos>", "<eos>", "<pad>"}}},
		},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}}},
	)

	meta, err := Metadata(ggufPath)
	if err != nil {
		t.Fatalf("Metadata() error = %v", err)
	}
	if arch, _ := meta["general.architecture"].(string); arch != "gemma4" {
		t.Fatalf("general.architecture = %q, want gemma4", arch)
	}
	if name, _ := meta["general.name"].(string); name != "metadata-good" {
		t.Fatalf("general.name = %q, want metadata-good", name)
	}
	if got := metadataInt(meta["general.file_type"]); got != 7 {
		t.Fatalf("general.file_type = %d, want 7", got)
	}
	if got := metadataInt(meta["general.alignment"]); got != 32 {
		t.Fatalf("general.alignment = %d, want 32", got)
	}
	if locked, ok := meta["general.use_mlock"].(bool); !ok || !locked {
		t.Fatalf("general.use_mlock = %#v, want true", meta["general.use_mlock"])
	}
	// String-element arrays surface their length only (#92 drafter config
	// reads counts, not token strings), so the array lands as the count.
	if got := metadataArrayLen(meta["gemma4-assistant.tokens"]); got != 3 {
		t.Fatalf("metadataArrayLen(tokens) = %d, want 3", got)
	}
}

func TestMetadata_Metadata_Bad(t *testing.T) {
	t.Run("invalid_magic", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "magic.gguf")
		if result := core.WriteFile(path, []byte("NOPExxxxxxxxxxxxxxxxxxxxxxxx"), 0o644); !result.OK {
			t.Fatalf("write bad-magic file: %v", result.Value)
		}
		if _, err := Metadata(path); err == nil {
			t.Fatal("expected Metadata() to fail for invalid magic")
		}
	})

	t.Run("missing_file", func(t *testing.T) {
		missing := core.PathJoin(t.TempDir(), "does-not-exist.gguf")
		if _, err := Metadata(missing); err == nil {
			t.Fatal("expected Metadata() to fail for a missing file")
		}
	})

	t.Run("unsupported_version", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "v1.gguf")
		created := core.Create(path)
		if !created.OK {
			t.Fatalf("create gguf: %v", created.Value)
		}
		file := created.Value.(*core.OSFile)
		// GGUF magic + version 1 (< 2 is rejected) + zero counts.
		if _, err := file.Write([]byte("GGUF")); err != nil {
			t.Fatalf("write magic: %v", err)
		}
		for _, v := range []any{uint32(1), uint64(0), uint64(0)} {
			if err := binary.Write(file, binary.LittleEndian, v); err != nil {
				t.Fatalf("write header field: %v", err)
			}
		}
		file.Close()
		if _, err := Metadata(path); err == nil {
			t.Fatal("expected Metadata() to reject unsupported gguf version 1")
		}
	})
}

func TestMetadata_Metadata_Ugly(t *testing.T) {
	t.Run("truncated_header", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "truncated.gguf")
		// Valid magic but only 4 of the required 24 header bytes present.
		if result := core.WriteFile(path, []byte("GGUF"), 0o644); !result.OK {
			t.Fatalf("write truncated file: %v", result.Value)
		}
		if _, err := Metadata(path); err == nil {
			t.Fatal("expected Metadata() to fail for a truncated header")
		}
	})

	t.Run("empty_metadata", func(t *testing.T) {
		// A structurally valid GGUF with zero metadata entries and zero
		// tensors must parse to a non-nil, empty map (boundary case).
		path := core.PathJoin(t.TempDir(), "empty.gguf")
		writeTestGGUF(t, path, nil, nil)
		meta, err := Metadata(path)
		if err != nil {
			t.Fatalf("Metadata() error = %v", err)
		}
		if meta == nil {
			t.Fatal("Metadata() returned nil map for empty gguf, want non-nil empty map")
		}
		if len(meta) != 0 {
			t.Fatalf("Metadata() = %d entries, want 0", len(meta))
		}
	})

	t.Run("single_tensor_no_metadata", func(t *testing.T) {
		// Edge: tensors present, metadata absent — Metadata() returns the
		// (empty) kv map and discards the tensor directory without error.
		path := core.PathJoin(t.TempDir(), "tensor-only.gguf")
		writeTestGGUF(t, path, nil,
			[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: TensorTypeQ8_0, Dims: []uint64{64, 64}}},
		)
		meta, err := Metadata(path)
		if err != nil {
			t.Fatalf("Metadata() error = %v", err)
		}
		if len(meta) != 0 {
			t.Fatalf("Metadata() = %d entries, want 0", len(meta))
		}
	})
}
