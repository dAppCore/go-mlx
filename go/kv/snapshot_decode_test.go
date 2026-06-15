// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

func TestKVSnapshot_UnmarshalTruncated_Bad(t *testing.T) {
	// A valid serialised buffer fed in truncated prefixes must fail closed
	// at the reader's bounds guard rather than panic. Walking several cut
	// points exercises the read/u32 truncation branches across the header
	// and the tensor body without hand-building a byte layout.
	full, err := testSnapshot().MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	if len(full) < 8 {
		t.Fatalf("serialised snapshot len = %d, want a non-trivial buffer", len(full))
	}
	// Magic-length prefix, mid-header, and one-byte-short all truncate.
	for _, cut := range []int{len(kvSnapshotMagic), len(kvSnapshotMagic) + 2, len(full) / 2, len(full) - 1} {
		var loaded Snapshot
		if err := loaded.UnmarshalBinary(full[:cut]); err == nil {
			t.Fatalf("UnmarshalBinary(truncated to %d/%d) error = nil, want truncation error", cut, len(full))
		}
	}
	// Sanity: the untruncated buffer still round-trips.
	var ok Snapshot
	if err := ok.UnmarshalBinary(full); err != nil {
		t.Fatalf("UnmarshalBinary(full) error = %v, want clean decode", err)
	}
}

func TestLoadKVSnapshot_Bad(t *testing.T) {
	_, err := Load(core.PathJoin(t.TempDir(), "missing.kvbin"))

	if err == nil {
		t.Fatal("Load() error = nil, want missing file error")
	}
}

func TestLoadKVSnapshot_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.kvbin")
	if result := core.WriteFile(path, []byte("not-a-kv-snapshot"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	_, err := Load(path)

	if err == nil {
		t.Fatal("Load() error = nil, want corrupt file error")
	}
}

// TestKVSnapshot_ParseTokensCorrupt_Bad drives parseKVSnapshotTokens down its
// magic (snapshot.go:647), version (651), and token-count overflow (663)
// guards, plus the post-header reader.err arm (678) reached when the header
// truncates mid-field so tokenCount reads 0 and the token block is skipped.
// parseKVSnapshotTokens wraps via core.E with a nil cause, so assert on the
// message rather than errors.Is against the exported sentinels.
func TestKVSnapshot_ParseTokensCorrupt_Bad(t *testing.T) {
	if _, err := parseKVSnapshotTokens([]byte("xx")); err == nil || !core.Contains(err.Error(), "magic") {
		t.Fatalf("parseKVSnapshotTokens(short) error = %v, want magic error", err)
	}

	badVersion := append([]byte(kvSnapshotMagic), 0, 0, 0, 0) // version 0
	if _, err := parseKVSnapshotTokens(badVersion); err == nil || !core.Contains(err.Error(), "version") {
		t.Fatalf("parseKVSnapshotTokens(version 0) error = %v, want version error", err)
	}

	// Header that claims more tokens than the trailing bytes can supply must
	// trip the overflow guard before the token block read.
	overflow := snapshotErrTokenHeader(SnapshotVersion, 1_000_000)
	if _, err := parseKVSnapshotTokens(overflow); err == nil || !core.Contains(err.Error(), "token count") {
		t.Fatalf("parseKVSnapshotTokens(overflow) error = %v, want token-count error", err)
	}

	// Valid magic + version but the architecture-length u32 truncates mid-read:
	// reader.err is set, tokenCount falls through as 0, and the block is
	// skipped, landing on the trailing reader.err guard (snapshot.go:678).
	truncHeader := append([]byte(kvSnapshotMagic), 6, 0, 0, 0) // version 6, then nothing
	truncHeader = append(truncHeader, 0, 0)                    // 2 of 4 archLen bytes
	if _, err := parseKVSnapshotTokens(truncHeader); err == nil || !core.Contains(err.Error(), "State tokens") {
		t.Fatalf("parseKVSnapshotTokens(truncated header) error = %v, want parse-State-tokens error", err)
	}
}

// TestKVSnapshot_ParseTokensInto_Bad drives parseKVSnapshotTokensInto down its
// bare-sentinel guards: magic (snapshot.go:689 → errInvalidSnapshotMagic),
// version (693 → errUnsupportedSnapshotVersion), and token-count overflow
// (705 → errStateTokenBlockTokenCount). The Good arm appends a real token
// block onto a non-empty dst, exercising the slice-extension path.
func TestKVSnapshot_ParseTokensInto_Bad(t *testing.T) {
	dst := []int32{99}

	out, err := parseKVSnapshotTokensInto(dst, []byte("xx"))
	if err == nil || !equalInt32s(out, dst) {
		t.Fatalf("parseKVSnapshotTokensInto(short) = %v/%v, want unchanged dst + magic error", out, err)
	}

	badVersion := append([]byte(kvSnapshotMagic), 0, 0, 0, 0)
	if _, err := parseKVSnapshotTokensInto(dst, badVersion); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(version 0) error = nil, want version error")
	}

	overflow := snapshotErrTokenHeader(SnapshotVersion, 1_000_000)
	if _, err := parseKVSnapshotTokensInto(dst, overflow); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(overflow) error = nil, want token-count error")
	}

	// Good: two real tokens appended to the existing dst.
	withTokens := snapshotErrTokenHeader(SnapshotVersion, 2)
	withTokens = appendKVI32sRaw(withTokens, []int32{5, 6})
	got, err := parseKVSnapshotTokensInto(dst, withTokens)
	if err != nil || !equalInt32s(got, []int32{99, 5, 6}) {
		t.Fatalf("parseKVSnapshotTokensInto(valid) = %v/%v, want [99 5 6]", got, err)
	}
}

// TestKVSnapshot_ParseTokens_Good covers the clean parseKVSnapshotTokens path
// (zero-token header returns an empty slice; a populated header decodes the
// block) so the function's success arms are exercised alongside the Bad cases.
func TestKVSnapshot_ParseTokens_Good(t *testing.T) {
	empty, err := parseKVSnapshotTokens(snapshotErrTokenHeader(SnapshotVersion, 0))
	if err != nil || len(empty) != 0 {
		t.Fatalf("parseKVSnapshotTokens(zero) = %v/%v, want empty slice", empty, err)
	}

	buf := snapshotErrTokenHeader(SnapshotVersion, 3)
	buf = appendKVI32sRaw(buf, []int32{7, 8, 9})
	tokens, err := parseKVSnapshotTokens(buf)
	if err != nil || !equalInt32s(tokens, []int32{7, 8, 9}) {
		t.Fatalf("parseKVSnapshotTokens(three) = %v/%v, want [7 8 9]", tokens, err)
	}
}

// TestKVSnapshot_UnsupportedEncoding_Bad clones the hand-built valid buffer but
// stamps encoding tag 3 on the head key tensor, driving the encodedTensor
// reader's default arm (snapshot.go:1323 → errUnsupportedTensorEncoding).
func TestKVSnapshot_UnsupportedEncoding_Bad(t *testing.T) {
	data := snapshotBadEncodingBytes(3)

	var loaded Snapshot
	err := loaded.UnmarshalBinary(data)
	if err == nil || !core.Contains(err.Error(), "unsupported KV tensor encoding") {
		t.Fatalf("UnmarshalBinary(encoding tag 3) error = %v, want unsupported-encoding error", err)
	}
}

// TestKVSnapshot_ReaderCase0Truncated_Bad stamps a float32 (encoding 0) head
// tensor with a size larger than the trailing bytes, driving the case-0
// chunk==nil arm in the encodedTensor reader (snapshot.go:1282) via the
// underlying read() truncation guard.
func TestKVSnapshot_ReaderCase0Truncated_Bad(t *testing.T) {
	data := snapshotBadEncodingBytes(0)
	// snapshotBadEncodingBytes(0) writes encoding 0 with size 1 and one f32
	// (4 bytes). Rewrite the size to claim 9999 elements without supplying the
	// bytes — the batched read(size*4) overruns and returns nil.
	patchKVU32(data, snapshotKeyTensorSizeOffset(), 9999)

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err == nil || !core.Contains(err.Error(), "truncated") {
		t.Fatalf("UnmarshalBinary(case-0 oversized) error = %v, want truncation error", err)
	}
}

// TestKVSnapshot_LoadWithOptionsParseError_Bad writes a corrupt-but-present
// file so LoadWithOptions reaches parseKVSnapshotWithOptions and returns its
// parse error (the read succeeds; the parse fails). Complements the existing
// missing-file Bad case.
func TestKVSnapshot_LoadWithOptionsParseError_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "badmagic.kvbin")
	if result := core.WriteFile(path, []byte("XXXXXXXX____"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}

	if _, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true}); err == nil || !core.Contains(err.Error(), "magic") {
		t.Fatalf("LoadWithOptions(bad magic) error = %v, want magic parse error", err)
	}
}

// TestKVSnapshot_ParseLegacyV2_Good hand-builds a version-2 buffer whose heads
// carry plain float32 Key/Value blocks (no per-tensor encoding header). This is
// the only way to drive the version<3 head read arm in parseKVSnapshotWithOptions
// (snapshot.go:611-614, the reader.f32s() fallback) — the writer always emits
// the current version, so a round-trip can't reach it.
func TestKVSnapshot_ParseLegacyV2_Good(t *testing.T) {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, 2) // version 2 (<3 → f32s head path, ≥2 → token offset/generated/logits)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 2) // SeqLen
	data = appendKVU32(data, 2) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVU32(data, 2) // TokenOffset (v>=2)
	data = appendKVI32s(data, []int32{1, 2})
	data = appendKVI32s(data, []int32{2}) // generated (v>=2)
	data = appendKVU32(data, 1)           // layer count
	data = appendKVI32(data, 0)           // Layer
	data = appendKVI32(data, 0)           // CacheIndex
	data = appendKVU32(data, 1)           // head count
	data = appendKVF32s(data, []float32{1, 0, 0, 1})
	data = appendKVF32s(data, []float32{0, 1, 1, 0})
	data = appendKVI32s(data, []int32{1, 1, 3}) // logit shape (v>=2)
	data = appendKVF32s(data, []float32{0.1, 0.2, 0.7})

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(v2 legacy) error = %v", err)
	}
	if loaded.Version != 2 || len(loaded.Layers) != 1 {
		t.Fatalf("loaded v2 = version %d / %d layers, want version 2 / 1 layer", loaded.Version, len(loaded.Layers))
	}
	head := loaded.Layers[0].Heads[0]
	if len(head.Key) != 4 || head.Key[0] != 1 || len(head.Value) != 4 || head.Value[1] != 1 {
		t.Fatalf("loaded v2 head = %+v, want float32 key/value from the legacy read path", head)
	}
}

// snapshotErrTokenHeader builds the State-block header parseKVSnapshotTokens and
// parseKVSnapshotTokensInto consume: magic, version, length-prefixed
// architecture, five u32 dimension fields, the v>=2 token-offset field, and the
// token count. Callers append the token bytes (or omit them to trip the
// overflow guard).
func snapshotErrTokenHeader(version, tokenCount uint32) []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, version)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	for range 5 {
		data = appendKVU32(data, 0) // NumLayers/NumHeads/SeqLen/HeadDim/NumQueryHeads
	}
	if version >= 2 {
		data = appendKVU32(data, 0) // TokenOffset
	}
	data = appendKVU32(data, tokenCount)
	return data
}

// snapshotBadEncodingBytes builds a complete valid single-head v6 buffer (the
// kvSnapshotTurboQuantNoPayloadBytes layout, minus the turboquant cache mode)
// but writes encoding tag `encodingTag` on the key tensor, with one float32
// element of payload. With tag 3 it drives the reader's default arm; with tag 0
// it is a valid float32 tensor whose size field can be patched to overrun.
func snapshotBadEncodingBytes(encodingTag uint32) []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, SnapshotVersion)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 1) // SeqLen
	data = appendKVU32(data, 1) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVU32(data, 1) // TokenOffset (v>=2)
	data = appendKVI32s(data, []int32{1})
	data = appendKVU32(data, 0) // generated count (v>=2)
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0) // Layer
	data = appendKVI32(data, 0) // CacheIndex
	data = appendKVU32(data, 1) // head count
	data = appendKVBytes(data, core.AsBytes(""))
	data = appendKVU32(data, 0)    // TurboQuant payload count (v>=5)
	data = appendKVU32(data, 0)    // MaxSize (v>=6)
	data = appendKVI32s(data, nil) // KeyShape (v>=4)
	data = appendKVU32(data, 0)    // key tensor encoding (RawKVOnly path)
	data = appendKVU32(data, 0)    // key tensor size
	data = appendKVI32s(data, nil) // ValueShape (v>=4)
	data = appendKVU32(data, 0)    // value tensor encoding
	data = appendKVU32(data, 0)    // value tensor size
	// Head 0 (v>=3): key tensor with the chosen encoding tag, then a clean
	// value tensor. snapshotKeyTensorSizeOffset() points at the size u32 below.
	data = appendKVU32(data, encodingTag) // key tensor encoding
	data = appendKVU32(data, 1)           // key tensor size (1 element)
	data = appendKVF32Raw(data, []float32{1})
	data = appendKVU32(data, 0) // value tensor encoding (float32)
	data = appendKVU32(data, 0) // value tensor size
	data = appendKVU32(data, 0) // logit shape (v>=2)
	data = appendKVF32s(data, nil)
	return data
}

// snapshotKeyTensorSizeOffset returns the byte offset of the head-0 key
// tensor's size u32 within a snapshotBadEncodingBytes buffer, so a test can
// rewrite the size to overrun the trailing bytes. It is the position of the
// "key tensor size" field written after the head-0 encoding tag.
func snapshotKeyTensorSizeOffset() int {
	// Recompute by re-walking the prefix the builder writes up to (and
	// including) the head-0 encoding tag. Mirrors snapshotBadEncodingBytes.
	var prefix []byte
	prefix = append(prefix, kvSnapshotMagic...)
	prefix = appendKVU32(prefix, SnapshotVersion)
	prefix = appendKVBytes(prefix, core.AsBytes("gemma4_text"))
	for range 5 {
		prefix = appendKVU32(prefix, 1)
	}
	prefix = appendKVU32(prefix, 1)           // TokenOffset
	prefix = appendKVI32s(prefix, []int32{1}) // tokens
	prefix = appendKVU32(prefix, 0)           // generated count
	prefix = appendKVU32(prefix, 1)           // layer count
	prefix = appendKVI32(prefix, 0)           // Layer
	prefix = appendKVI32(prefix, 0)           // CacheIndex
	prefix = appendKVU32(prefix, 1)           // head count
	prefix = appendKVBytes(prefix, core.AsBytes(""))
	prefix = appendKVU32(prefix, 0)    // TurboQuant payload count
	prefix = appendKVU32(prefix, 0)    // MaxSize
	prefix = appendKVI32s(prefix, nil) // KeyShape
	prefix = appendKVU32(prefix, 0)    // layer key encoding
	prefix = appendKVU32(prefix, 0)    // layer key size
	prefix = appendKVI32s(prefix, nil) // ValueShape
	prefix = appendKVU32(prefix, 0)    // layer value encoding
	prefix = appendKVU32(prefix, 0)    // layer value size
	prefix = appendKVU32(prefix, 0)    // head-0 key encoding tag
	return len(prefix)                 // next u32 written is the key size
}

// patchKVU32 overwrites the little-endian u32 at offset within buf.
func patchKVU32(buf []byte, offset int, value uint32) {
	binary.LittleEndian.PutUint32(buf[offset:offset+4], value)
}

// equalInt32s reports whether two int32 slices hold the same values.
func equalInt32s(left, right []int32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

// TestSnapshot_ReaderEncodedF32s_Good covers the kvSnapshotReader.encodedF32s
// wrapper (snapshot.go), which forwards encodedTensor(LoadOptions{}).Values.
// A hand-built encoding-0 (float32) tensor block is decoded back to its values.
func TestSnapshot_ReaderEncodedF32s_Good(t *testing.T) {
	want := []float32{1.5, -2.25, 3.75}
	buf := appendKVEncodedF32s(nil, want, KVSnapshotEncodingFloat32)

	reader := &kvSnapshotReader{data: buf}
	got := reader.encodedF32s()
	if reader.err != nil {
		t.Fatalf("encodedF32s reader.err = %v", reader.err)
	}
	if len(got) != len(want) {
		t.Fatalf("encodedF32s len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("encodedF32s = %v, want %v", got, want)
		}
	}
}

// TestSnapshot_DtypeString_GoodBadUgly covers every arm of the
// kvSnapshotReader.dtypeString length-prefixed tag reader: the canonical
// short-form (F32/F16/BF16) and long-form (float32/float16/bfloat16) tags each
// return their literal, an unknown tag is returned verbatim, and a truncated
// length prefix yields the empty string (the read-nil guard).
func TestSnapshot_DtypeString_GoodBadUgly(t *testing.T) {
	// dtypeTag builds a length-prefixed dtype buffer the reader consumes.
	dtypeTag := func(tag string) []byte {
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(len(tag)))
		return append(buf, tag...)
	}

	for _, tag := range []string{"F32", "F16", "BF16", "float32", "float16", "bfloat16"} {
		reader := &kvSnapshotReader{data: dtypeTag(tag)}
		if got := reader.dtypeString(); got != tag {
			t.Fatalf("dtypeString(%q) = %q, want the canonical literal", tag, got)
		}
	}

	// Unknown tag of a recognised length is returned verbatim (validator
	// rejects it downstream).
	if got := (&kvSnapshotReader{data: dtypeTag("abc")}).dtypeString(); got != "abc" {
		t.Fatalf("dtypeString(unknown 3-byte) = %q, want \"abc\"", got)
	}
	// Unknown tag of an unrecognised length also falls through to verbatim.
	if got := (&kvSnapshotReader{data: dtypeTag("int8")}).dtypeString(); got != "int8" {
		t.Fatalf("dtypeString(unknown 4-byte) = %q, want \"int8\"", got)
	}

	// Ugly: a length prefix claiming more bytes than remain → read returns
	// nil → dtypeString returns "".
	truncated := make([]byte, 4)
	binary.LittleEndian.PutUint32(truncated, 99)
	if got := (&kvSnapshotReader{data: truncated}).dtypeString(); got != "" {
		t.Fatalf("dtypeString(truncated) = %q, want empty string", got)
	}
}
