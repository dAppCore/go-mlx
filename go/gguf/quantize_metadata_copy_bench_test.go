// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark + byte-identity guard for the model-pack metadata-copy path:
// copyModelPackMetadata → copyLocalFile. QuantizeModelPack mirrors every
// non-weight metadata file (config.json, tokenizer.json, *.model, *.txt)
// from the source pack into the GGUF output directory before it writes the
// quantized tensors.
//
// AX-11 / trap #5 (whole-file read → stream in chunks): copyLocalFile used
// to slurp each file whole via core.ReadFile (one heap []byte the size of
// the file) then core.WriteFile it back. config.json is tiny, but
// tokenizer.json is multiple MB on real checkpoints (the BPE merge table
// dominates), so that slurp was a large transient buffer per conversion and
// the dominant B/op of the copy path at -alloc_space. The repo's tiny
// 58-byte tokenizer fixture (modelPackTokenizerJSON) hides it entirely —
// this bench uses a REALISTIC multi-MB metadata file so the buffer shows.
//
// Run:
//   go test -bench='BenchmarkCopyModelPackMetadata' -benchmem -benchtime=20x \
//     -run='^$' ./go/gguf
//   go test -bench='BenchmarkCopyModelPackMetadata_LargeTokenizer' -benchmem \
//     -benchtime=50x -run='^$' -memprofile=/tmp/cmm.mem ./go/gguf
//   # B/op before→after is the headline (-alloc_space). The slurp's bytes are
//   # allocated inside os.ReadFile's make([]byte,size) — a callee — so the
//   # pprof FLAT for copyLocalFile reads ~0; the drop shows as CUM on the
//   # core.ReadFile(...) line and in the top -alloc_space view:
//   go tool pprof -alloc_space -list 'copyLocalFile' /tmp/cmm.mem
//   go tool pprof -alloc_space -top /tmp/cmm.mem | head

package gguf

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	core "dappco.re/go"
)

// writeMetadataCopyFixture lays out a source pack directory holding the
// metadata files copyModelPackMetadata mirrors, sized to exercise BOTH sides
// of copyLocalFile's size gate:
//   - config.json      — small, hits the direct-read branch
//   - tokenizer.json   — tokenizerBytes large, hits the streamed branch
//   - tokenizer.model  — a *.model entry (also copied)
//
// A model.safetensors weight file is written too so the *.safetensors copy
// SKIP path (isModelWeightMetadataCopySkip) is exercised — it must NOT be
// mirrored, and a regression that copied it would inflate both timing and
// the byte-identity set.
func writeMetadataCopyFixture(t testing.TB, tokenizerBytes int) string {
	t.Helper()
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"vocab_size": 151936,
		"hidden_size": 2048
	}`)
	// Content need not be valid JSON — copyLocalFile copies bytes, it does
	// not parse. A deterministic blob the size of a real tokenizer.json is
	// enough to make the whole-file slurp dominate B/op.
	big := bytes.Repeat([]byte("lethean-tokenizer-merge-table-row\n"), tokenizerBytes/34+1)
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), big, 0o644); !result.OK {
		t.Fatalf("write tokenizer.json: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.model"), "tokenizer-model-sentinel")
	// A weight file that the copy filter must skip.
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("not-copied"), 0o644); !result.OK {
		t.Fatalf("write model.safetensors: %v", result.Value)
	}
	return dir
}

var cmmSinkErr error

// benchCopyModelPackMetadata drives copyModelPackMetadata once per iteration
// into a fresh output directory (the precomputed paths keep their PathJoin
// allocs outside the measured region).
func benchCopyModelPackMetadata(b *testing.B, tokenizerBytes int) {
	source := writeMetadataCopyFixture(b, tokenizerBytes)
	base := b.TempDir()
	outs := make([]string, b.N)
	for i := range outs {
		outs[i] = core.PathJoin(base, "out"+itoaCMM(i))
		if result := core.MkdirAll(outs[i], 0o755); !result.OK {
			b.Fatalf("mkdir out: %v", result.Value)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmmSinkErr = copyModelPackMetadata(source, outs[i])
		if cmmSinkErr != nil {
			b.Fatalf("copyModelPackMetadata: %v", cmmSinkErr)
		}
	}
}

// itoaCMM avoids pulling strconv into the bench's import set for a single use.
func itoaCMM(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[pos:])
}

// LargeTokenizer — 4 MiB tokenizer.json, the realistic case. This is where
// the slurp's per-file buffer dominated B/op; streaming through core.Copy's
// fixed staging buffer (or the kernel copy fast-path) flattens it.
func BenchmarkCopyModelPackMetadata_LargeTokenizer(b *testing.B) {
	benchCopyModelPackMetadata(b, 4<<20)
}

// SmallOnly — every metadata file under the gate threshold, so the direct
// read branch carries the whole copy. Guards that the gate does not regress
// the tiny-file path the slurp already handled cheaply.
func BenchmarkCopyModelPackMetadata_SmallOnly(b *testing.B) {
	benchCopyModelPackMetadata(b, 16<<10)
}

// TestCopyModelPackMetadata_ByteIdentical proves the copied bytes are
// identical to the source across BOTH gate branches (a small config.json on
// the direct side, a >128 KiB tokenizer.json on the streamed side) and that
// the weight file is NOT mirrored. It passes on the pre-stream (slurp) code
// too — the slurp was byte-identical — so it is a genuine regression guard
// for the streaming rewrite, not a tautology against it.
func TestCopyModelPackMetadata_ByteIdentical(t *testing.T) {
	// 1 MiB tokenizer crosses the 128 KiB gate → streamed branch; config.json
	// stays on the direct branch.
	source := writeMetadataCopyFixture(t, 1<<20)
	output := core.PathJoin(t.TempDir(), "out")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir out: %v", result.Value)
	}

	if err := copyModelPackMetadata(source, output); err != nil {
		t.Fatalf("copyModelPackMetadata: %v", err)
	}

	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer.model"} {
		src := readFileOrFatal(t, core.PathJoin(source, name))
		dst := readFileOrFatal(t, core.PathJoin(output, name))
		if !bytes.Equal(src, dst) {
			t.Fatalf("%s: copied bytes differ (src %d B sha %s, dst %d B sha %s)",
				name, len(src), shortSHA(src), len(dst), shortSHA(dst))
		}
	}

	// The weight file must be skipped by the copy filter.
	if stat := core.Stat(core.PathJoin(output, "model.safetensors")); stat.OK {
		t.Fatal("model.safetensors was copied into the output dir; the weight-skip filter regressed")
	}
}

func shortSHA(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:12]
}
