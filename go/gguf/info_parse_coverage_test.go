// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// TestReadGGUFValue_AllScalarTruncations_Bad sweeps every fixed-width value
// type and confirms that a header promising the value, followed by a stream
// too short to carry it, surfaces a short-read error rather than a partial
// decode. The existing suite only exercised float64; this table mops up the
// uint8/int8/uint16/int16/uint32/int32/float32/bool/uint64/int64 arms in one
// pass (the advisor's "loop over all types" lever).
func TestReadGGUFValue_AllScalarTruncations_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)
	cases := []struct {
		name      string
		valueType uint32
	}{
		{"uint8", ggufValueTypeUint8},
		{"int8", ggufValueTypeInt8},
		{"uint16", ggufValueTypeUint16},
		{"int16", ggufValueTypeInt16},
		{"uint32", ValueTypeUint32},
		{"int32", ggufValueTypeInt32},
		{"float32", ggufValueTypeFloat32},
		{"bool", ggufValueTypeBool},
		{"uint64", ggufValueTypeUint64},
		{"int64", ggufValueTypeInt64},
		{"float64", ggufValueTypeFloat64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Empty stream: even a 1-byte scalar's read fails.
			if _, err := readGGUFValue(bytes.NewReader(nil), tc.valueType, scratch, &arena); err == nil {
				t.Fatalf("readGGUFValue(%s, empty) error = nil, want short-read", tc.name)
			}
		})
	}
}

// TestReadGGUFValue_ArrayArms_Bad drives readGGUFValue's array-element arms
// that the happy-path array tests do not reach:
//   - truncated element-type prefix
//   - truncated length prefix
//   - length exceeding maxGGUFCollectionEntries
//   - a recursive element decode failing mid-array (numeric element runs out)
//   - a string-array element skip failing mid-array
func TestReadGGUFValue_ArrayArms_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)

	// Element-type read fails: fewer than 4 bytes available.
	if _, err := readGGUFValue(bytes.NewReader([]byte{0, 0}), ggufValueTypeArray, scratch, &arena); err == nil {
		t.Fatal("array element-type truncation: error = nil")
	}

	// Length read fails: element type present (4 bytes) but length prefix short.
	var lenTrunc [6]byte
	binary.LittleEndian.PutUint32(lenTrunc[:4], ValueTypeUint32)
	if _, err := readGGUFValue(bytes.NewReader(lenTrunc[:]), ggufValueTypeArray, scratch, &arena); err == nil {
		t.Fatal("array length truncation: error = nil")
	}

	// Length exceeds the collection-entry cap.
	var tooMany [12]byte
	binary.LittleEndian.PutUint32(tooMany[:4], ValueTypeUint32)
	binary.LittleEndian.PutUint64(tooMany[4:12], maxGGUFCollectionEntries+1)
	if _, err := readGGUFValue(bytes.NewReader(tooMany[:]), ggufValueTypeArray, scratch, &arena); err == nil {
		t.Fatal("array length over cap: error = nil")
	}

	// Numeric element decode fails partway: declare 2 uint32 elements but
	// supply only 4 payload bytes (one element).
	var numTrunc [16]byte
	binary.LittleEndian.PutUint32(numTrunc[:4], ValueTypeUint32)
	binary.LittleEndian.PutUint64(numTrunc[4:12], 2)
	binary.LittleEndian.PutUint32(numTrunc[12:16], 42) // first element only
	if _, err := readGGUFValue(bytes.NewReader(numTrunc[:]), ggufValueTypeArray, scratch, &arena); err == nil {
		t.Fatal("array numeric element truncation: error = nil")
	}

	// String-array element skip fails partway: declare 2 string elements but
	// the second element's length prefix is missing.
	var strBuf bytes.Buffer
	binary.Write(&strBuf, binary.LittleEndian, uint32(ValueTypeString)) // element type
	binary.Write(&strBuf, binary.LittleEndian, uint64(2))               // 2 elements
	binary.Write(&strBuf, binary.LittleEndian, uint64(3))               // first len = 3
	strBuf.WriteString("abc")                                           // first body
	// second element omitted entirely -> skipGGUFString hits EOF.
	if _, err := readGGUFValue(bytes.NewReader(strBuf.Bytes()), ggufValueTypeArray, scratch, &arena); err == nil {
		t.Fatal("string-array element skip truncation: error = nil")
	}
}

// TestSkipGGUFString_Arms_Bad covers skipGGUFString's guard + truncation arms
// directly: an over-long declared length (16 MiB cap) and a body shorter than
// the declared length. The happy path is already exercised via the string-
// array fast path in ReadInfo.
func TestSkipGGUFString_Arms_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)

	// Over the 16 MiB cap.
	var tooLong [8]byte
	binary.LittleEndian.PutUint64(tooLong[:], 17<<20)
	if err := skipGGUFString(bytes.NewReader(tooLong[:]), scratch); err == nil {
		t.Fatal("skipGGUFString(over cap) error = nil")
	}

	// Length prefix itself truncated.
	if err := skipGGUFString(bytes.NewReader([]byte{1, 2, 3}), scratch); err == nil {
		t.Fatal("skipGGUFString(short prefix) error = nil")
	}

	// Declares 100 bytes, supplies 10 -> body short read (exercises the
	// chunked discard loop hitting EOF).
	var bodyTrunc [18]byte
	binary.LittleEndian.PutUint64(bodyTrunc[:8], 100)
	if err := skipGGUFString(bytes.NewReader(bodyTrunc[:]), scratch); err == nil {
		t.Fatal("skipGGUFString(short body) error = nil")
	}
}

// TestReadStringIntoArena_OverflowArms_Ugly drives readStringIntoArena's two
// overflow fallbacks. Both need a string longer than the arena's remaining
// capacity:
//   - when the string still fits the scratch buffer, it is read through
//     scratch and returned as a fresh copy (the intern-miss arm);
//   - when the string exceeds scratch too, a fresh make([]byte) is used.
func TestReadStringIntoArena_OverflowArms_Ugly(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize) // 64

	// Arena with zero spare capacity forces the overflow branch immediately.
	emptyArena := make([]byte, 0, 0)

	// 40-byte string: <= 64-byte scratch -> scratch-copy fallback. Use a
	// value that is not in the intern table so it takes the string() copy
	// arm (not the interned-singleton arm).
	short := string(bytes.Repeat([]byte("x"), 40))
	payload := stringPayload(short)
	a := emptyArena
	got, err := readStringIntoArena(bytes.NewReader(payload), scratch, &a)
	if err != nil {
		t.Fatalf("readStringIntoArena(scratch overflow) error = %v", err)
	}
	if got != short {
		t.Fatalf("readStringIntoArena(scratch overflow) = %q, want %q", got, short)
	}

	// 200-byte string: > 64-byte scratch -> make([]byte, length) fallback.
	long := string(bytes.Repeat([]byte("y"), 200))
	payload = stringPayload(long)
	a = emptyArena
	got, err = readStringIntoArena(bytes.NewReader(payload), scratch, &a)
	if err != nil {
		t.Fatalf("readStringIntoArena(make fallback) error = %v", err)
	}
	if got != long {
		t.Fatalf("readStringIntoArena(make fallback) = %q, want %q", got, long)
	}

	// make-fallback path with a truncated body -> short read surfaces.
	var hdr [8]byte
	binary.LittleEndian.PutUint64(hdr[:], 200)
	a = emptyArena
	if _, err := readStringIntoArena(bytes.NewReader(hdr[:]), scratch, &a); err == nil {
		t.Fatal("readStringIntoArena(make fallback, truncated body) error = nil")
	}

	// scratch-fallback path with a truncated body -> short read surfaces.
	binary.LittleEndian.PutUint64(hdr[:], 40)
	a = emptyArena
	if _, err := readStringIntoArena(bytes.NewReader(hdr[:]), scratch, &a); err == nil {
		t.Fatal("readStringIntoArena(scratch fallback, truncated body) error = nil")
	}

	// Over-cap length prefix -> errGGUFStringTooLong before any arena work.
	binary.LittleEndian.PutUint64(hdr[:], 17<<20)
	a = emptyArena
	if _, err := readStringIntoArena(bytes.NewReader(hdr[:]), scratch, &a); err == nil {
		t.Fatal("readStringIntoArena(over cap) error = nil")
	}

	// Truncated length prefix -> read fails.
	a = emptyArena
	if _, err := readStringIntoArena(bytes.NewReader([]byte{1, 2}), scratch, &a); err == nil {
		t.Fatal("readStringIntoArena(short prefix) error = nil")
	}

	// Zero length -> empty string, no arena consumed.
	binary.LittleEndian.PutUint64(hdr[:], 0)
	a = emptyArena
	if got, err := readStringIntoArena(bytes.NewReader(hdr[:]), scratch, &a); err != nil || got != "" {
		t.Fatalf("readStringIntoArena(zero len) = %q, %v; want \"\", nil", got, err)
	}
}

// TestReadStringIntoArena_ArenaHit_Good drives the in-arena (non-overflow)
// path with both an interned key (rolled back, singleton returned) and a
// non-interned value (parked in the arena, view returned).
func TestReadStringIntoArena_ArenaHit_Good(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)

	// Interned key: lands in arena, intern probe hits, cursor rolled back.
	a := arena
	payload := stringPayload("general.architecture")
	got, err := readStringIntoArena(bytes.NewReader(payload), scratch, &a)
	if err != nil {
		t.Fatalf("readStringIntoArena(interned) error = %v", err)
	}
	if got != "general.architecture" {
		t.Fatalf("readStringIntoArena(interned) = %q", got)
	}
	if len(a) != 0 {
		t.Fatalf("interned hit should roll back arena cursor, len = %d", len(a))
	}

	// Non-interned value: stays in the arena, advancing the cursor.
	a = arena
	payload = stringPayload("a-unique-tensor-name")
	got, err = readStringIntoArena(bytes.NewReader(payload), scratch, &a)
	if err != nil {
		t.Fatalf("readStringIntoArena(non-interned) error = %v", err)
	}
	if got != "a-unique-tensor-name" {
		t.Fatalf("readStringIntoArena(non-interned) = %q", got)
	}
	if len(a) != len("a-unique-tensor-name") {
		t.Fatalf("non-interned should park in arena, cursor = %d", len(a))
	}
}

// TestParseGGUF_HeaderGuards_Bad drives parseGGUF's header validation arms
// that writeTestGGUF (always well-formed) cannot reach: a too-small file, a
// bad magic, an unsupported version, and tensor/metadata counts exceeding
// maxGGUFCollectionEntries.
func TestParseGGUF_HeaderGuards_Bad(t *testing.T) {
	write := func(t *testing.T, name string, data []byte) string {
		t.Helper()
		path := core.PathJoin(t.TempDir(), name)
		if result := core.WriteFile(path, data, 0o644); !result.OK {
			t.Fatalf("write %s: %v", name, result.Value)
		}
		return path
	}

	// File shorter than the 24-byte header.
	if _, _, err := parseGGUF(write(t, "tiny.gguf", []byte{1, 2, 3})); err == nil {
		t.Fatal("parseGGUF(tiny) error = nil, want short header")
	}

	// Correct length, wrong magic.
	badMagic := make([]byte, 24)
	copy(badMagic[:4], "XXXX")
	binary.LittleEndian.PutUint32(badMagic[4:8], 3)
	if _, _, err := parseGGUF(write(t, "magic.gguf", badMagic)); err == nil {
		t.Fatal("parseGGUF(bad magic) error = nil")
	}

	// Valid magic, version 1 (< 2 unsupported).
	oldVer := make([]byte, 24)
	copy(oldVer[:4], "GGUF")
	binary.LittleEndian.PutUint32(oldVer[4:8], 1)
	if _, _, err := parseGGUF(write(t, "ver.gguf", oldVer)); err == nil {
		t.Fatal("parseGGUF(version 1) error = nil")
	}

	// Tensor count over the limit.
	bigTensors := make([]byte, 24)
	copy(bigTensors[:4], "GGUF")
	binary.LittleEndian.PutUint32(bigTensors[4:8], 3)
	binary.LittleEndian.PutUint64(bigTensors[8:16], maxGGUFCollectionEntries+1)
	if _, _, err := parseGGUF(write(t, "tcount.gguf", bigTensors)); err == nil {
		t.Fatal("parseGGUF(tensor count over limit) error = nil")
	}

	// Metadata count over the limit.
	bigMeta := make([]byte, 24)
	copy(bigMeta[:4], "GGUF")
	binary.LittleEndian.PutUint32(bigMeta[4:8], 3)
	binary.LittleEndian.PutUint64(bigMeta[8:16], 0)
	binary.LittleEndian.PutUint64(bigMeta[16:24], maxGGUFCollectionEntries+1)
	if _, _, err := parseGGUF(write(t, "mcount.gguf", bigMeta)); err == nil {
		t.Fatal("parseGGUF(metadata count over limit) error = nil")
	}

	// Missing file (open failure).
	if _, _, err := parseGGUF(core.PathJoin(t.TempDir(), "does-not-exist.gguf")); err == nil {
		t.Fatal("parseGGUF(missing file) error = nil")
	}
}

// TestParseGGUF_TruncatedBody_Bad drives the per-entry short-read arms inside
// parseGGUF's metadata and tensor loops: a header promising one metadata
// entry / one tensor whose body bytes are then truncated must surface the
// wrapped read error.
func TestParseGGUF_TruncatedBody_Bad(t *testing.T) {
	write := func(t *testing.T, data []byte) string {
		t.Helper()
		path := core.PathJoin(t.TempDir(), "trunc.gguf")
		if result := core.WriteFile(path, data, 0o644); !result.OK {
			t.Fatalf("write trunc gguf: %v", result.Value)
		}
		return path
	}
	header := func(tensors, metadata uint64) []byte {
		h := make([]byte, 24)
		copy(h[:4], "GGUF")
		binary.LittleEndian.PutUint32(h[4:8], 3)
		binary.LittleEndian.PutUint64(h[8:16], tensors)
		binary.LittleEndian.PutUint64(h[16:24], metadata)
		return h
	}

	// Promises 1 metadata entry but no bytes follow -> key read fails.
	if _, _, err := parseGGUF(write(t, header(0, 1))); err == nil {
		t.Fatal("parseGGUF(missing metadata key) error = nil")
	}

	// Metadata key present, but the value-type prefix is missing.
	var keyOnly bytes.Buffer
	keyOnly.Write(header(0, 1))
	binary.Write(&keyOnly, binary.LittleEndian, uint64(3))
	keyOnly.WriteString("abc")
	// no value type follows
	if _, _, err := parseGGUF(write(t, keyOnly.Bytes())); err == nil {
		t.Fatal("parseGGUF(missing metadata value type) error = nil")
	}

	// Metadata key + type present, but the value body is missing.
	var noValue bytes.Buffer
	noValue.Write(header(0, 1))
	binary.Write(&noValue, binary.LittleEndian, uint64(3))
	noValue.WriteString("abc")
	binary.Write(&noValue, binary.LittleEndian, uint32(ValueTypeUint32))
	// no 4-byte uint32 value follows
	if _, _, err := parseGGUF(write(t, noValue.Bytes())); err == nil {
		t.Fatal("parseGGUF(missing metadata value body) error = nil")
	}

	// Promises 1 tensor but no bytes follow -> tensor name read fails.
	if _, _, err := parseGGUF(write(t, header(1, 0))); err == nil {
		t.Fatal("parseGGUF(missing tensor name) error = nil")
	}

	// Tensor name present, ndim prefix missing.
	var nameOnly bytes.Buffer
	nameOnly.Write(header(1, 0))
	binary.Write(&nameOnly, binary.LittleEndian, uint64(4))
	nameOnly.WriteString("blk0")
	if _, _, err := parseGGUF(write(t, nameOnly.Bytes())); err == nil {
		t.Fatal("parseGGUF(missing tensor ndim) error = nil")
	}

	// Tensor name + ndim present, dimension value missing.
	var noDim bytes.Buffer
	noDim.Write(header(1, 0))
	binary.Write(&noDim, binary.LittleEndian, uint64(4))
	noDim.WriteString("blk0")
	binary.Write(&noDim, binary.LittleEndian, uint32(2))  // ndim = 2
	binary.Write(&noDim, binary.LittleEndian, uint64(32)) // only 1 dim follows
	if _, _, err := parseGGUF(write(t, noDim.Bytes())); err == nil {
		t.Fatal("parseGGUF(missing tensor dimension) error = nil")
	}

	// Tensor name + ndim + dims present, type/offset block missing.
	var noType bytes.Buffer
	noType.Write(header(1, 0))
	binary.Write(&noType, binary.LittleEndian, uint64(4))
	noType.WriteString("blk0")
	binary.Write(&noType, binary.LittleEndian, uint32(1)) // ndim = 1
	binary.Write(&noType, binary.LittleEndian, uint64(32))
	// no 12-byte type+offset block follows
	if _, _, err := parseGGUF(write(t, noType.Bytes())); err == nil {
		t.Fatal("parseGGUF(missing tensor type/offset) error = nil")
	}
}

// TestReadStringIntoArena_InArenaShortRead_Bad covers the in-arena (non-
// overflow) body short-read arm: the arena has capacity for the declared
// length, but the stream's body is truncated, so the read into the arena
// slice fails.
func TestReadStringIntoArena_InArenaShortRead_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256) // ample capacity -> in-arena path
	var hdr [8]byte
	binary.LittleEndian.PutUint64(hdr[:], 20) // declares 20 bytes, none follow
	a := arena
	if _, err := readStringIntoArena(bytes.NewReader(hdr[:]), scratch, &a); err == nil {
		t.Fatal("readStringIntoArena(in-arena short body) error = nil")
	}
}

// TestReadStringIntoArena_InternedOverflow_Ugly targets the
// interned-hit-on-scratch-overflow arm: a zero-capacity arena forces the
// overflow branch, the interned key fits the scratch buffer, and the intern
// probe hits — so the singleton is returned without touching the arena.
func TestReadStringIntoArena_InternedOverflow_Ugly(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize) // 64 -> holds the 20-byte key
	a := make([]byte, 0, 0)                      // zero spare -> overflow branch
	payload := stringPayload("general.architecture")
	got, err := readStringIntoArena(bytes.NewReader(payload), scratch, &a)
	if err != nil {
		t.Fatalf("readStringIntoArena(interned overflow) error = %v", err)
	}
	if got != "general.architecture" {
		t.Fatalf("readStringIntoArena(interned overflow) = %q", got)
	}
}

// TestReadGGUFString_MakeShortRead_Bad covers readGGUFString's make-path body
// short-read: a length larger than the scratch buffer forces the
// make([]byte, length) arm, and a truncated body makes that read fail.
// Reached through readGGUFValue with a nil arena (which delegates to
// readGGUFString).
func TestReadGGUFString_MakeShortRead_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize) // 64
	var hdr [8]byte
	binary.LittleEndian.PutUint64(hdr[:], 200) // > 64 -> make path; no body
	if _, err := readGGUFValue(bytes.NewReader(hdr[:]), ValueTypeString, scratch, nil); err == nil {
		t.Fatal("readGGUFString(make-path short body) error = nil")
	}
}

// TestParseGGUF_HighRankTensor_Good drives parseGGUF's shape-arena overflow
// fallback: with a single tensor the shape arena reserves only 4 uint64s, so
// a rank-5 tensor exceeds the remaining capacity and takes the per-tensor
// make([]uint64, ndim) path instead of the arena slab.
func TestParseGGUF_HighRankTensor_Good(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 tensor
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // 0 metadata
	// tensor: name, ndim=5, five dims, type, offset
	binary.Write(&buf, binary.LittleEndian, uint64(len("blk.0.w")))
	buf.WriteString("blk.0.w")
	binary.Write(&buf, binary.LittleEndian, uint32(5)) // ndim = 5 > arena budget of 4
	for _, d := range []uint64{2, 3, 4, 5, 6} {
		binary.Write(&buf, binary.LittleEndian, d)
	}
	binary.Write(&buf, binary.LittleEndian, uint32(ggufTensorTypeF32))
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	path := core.PathJoin(t.TempDir(), "rank5.gguf")
	if result := core.WriteFile(path, buf.Bytes(), 0o644); !result.OK {
		t.Fatalf("write rank5 gguf: %v", result.Value)
	}
	_, tensors, err := parseGGUF(path)
	if err != nil {
		t.Fatalf("parseGGUF(rank-5) error = %v", err)
	}
	if len(tensors) != 1 || len(tensors[0].Shape) != 5 {
		t.Fatalf("parseGGUF(rank-5) shape = %v, want 5 dims", tensors[0].Shape)
	}
	want := []uint64{2, 3, 4, 5, 6}
	for i, w := range want {
		if tensors[0].Shape[i] != w {
			t.Fatalf("rank-5 dim[%d] = %d, want %d", i, tensors[0].Shape[i], w)
		}
	}
}

// stringPayload builds the GGUF [uint64 length][bytes] encoding of value, the
// wire form readStringIntoArena / skipGGUFString consume.
func stringPayload(value string) []byte {
	out := make([]byte, 8+len(value))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(value)))
	copy(out[8:], value)
	return out
}
