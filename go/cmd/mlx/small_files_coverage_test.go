// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// ---- wav.go -----------------------------------------------------------------

// buildWAV assembles a minimal RIFF/WAVE byte stream: fmt chunk + data chunk.
func buildWAV(format, channels uint16, sampleRate uint32, bits uint16, body []byte) []byte {
	le := binary.LittleEndian
	fmtChunk := make([]byte, 16)
	le.PutUint16(fmtChunk[0:2], format)
	le.PutUint16(fmtChunk[2:4], channels)
	le.PutUint32(fmtChunk[4:8], sampleRate)
	byteRate := sampleRate * uint32(channels) * uint32(bits/8)
	le.PutUint32(fmtChunk[8:12], byteRate)
	le.PutUint16(fmtChunk[12:14], channels*bits/8)
	le.PutUint16(fmtChunk[14:16], bits)

	out := []byte("RIFF")
	riffLen := 4 + (8 + len(fmtChunk)) + (8 + len(body))
	out = le.AppendUint32(out, uint32(riffLen))
	out = append(out, []byte("WAVE")...)
	out = append(out, []byte("fmt ")...)
	out = le.AppendUint32(out, uint32(len(fmtChunk)))
	out = append(out, fmtChunk...)
	out = append(out, []byte("data")...)
	out = le.AppendUint32(out, uint32(len(body)))
	out = append(out, body...)
	return out
}

// readWAVMono rejects the structural-edge inputs the existing wav_test does not
// cover: a too-short file, a truncated chunk, a malformed fmt chunk, a data
// chunk before fmt, a file with no data chunk, and an unsupported encoding.
func TestReadWAVMono_StructuralEdges_Bad(t *testing.T) {
	le := binary.LittleEndian
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := core.JoinPath(dir, name)
		if r := core.WriteFile(p, b, 0o644); !r.OK {
			t.Fatalf("write %s: %v", name, r.Value)
		}
		return p
	}

	// Too short (< 44 bytes) but starts with a partial RIFF.
	if _, err := readWAVMono(write("short.wav", []byte("RIFF12345678")), 16000); err == nil {
		t.Fatal("short file: error = nil, want too-short error")
	}

	// Valid RIFF/WAVE header (44 bytes of zeros after the magic) so the
	// length guard passes but the body is junk → chunk loop hits no data
	// chunk (and never asserts fmt) → "no data chunk".
	noData := make([]byte, 44)
	copy(noData[0:4], "RIFF")
	copy(noData[8:12], "WAVE")
	if _, err := readWAVMono(write("nodata.wav", noData), 16000); err == nil {
		t.Fatal("no data chunk: error = nil, want a no-data-chunk error")
	}

	// A data chunk that arrives before any fmt chunk.
	dataBeforeFmt := []byte("RIFF")
	dataBeforeFmt = le.AppendUint32(dataBeforeFmt, 0)
	dataBeforeFmt = append(dataBeforeFmt, []byte("WAVE")...)
	dataBeforeFmt = append(dataBeforeFmt, []byte("data")...)
	dataBeforeFmt = le.AppendUint32(dataBeforeFmt, 4)
	dataBeforeFmt = append(dataBeforeFmt, 0, 0, 0, 0)
	if _, err := readWAVMono(write("databeforefmt.wav", dataBeforeFmt), 16000); err == nil {
		t.Fatal("data before fmt: error = nil, want the ordering error")
	}

	// A fmt chunk that claims a length longer than the file has (truncated).
	truncated := []byte("RIFF")
	truncated = le.AppendUint32(truncated, 0)
	truncated = append(truncated, []byte("WAVE")...)
	truncated = append(truncated, []byte("fmt ")...)
	truncated = le.AppendUint32(truncated, 999) // claims 999 bytes that aren't there
	if _, err := readWAVMono(write("trunc.wav", truncated), 16000); err == nil {
		t.Fatal("truncated chunk: error = nil, want the truncated-chunk error")
	}

	// A fmt chunk shorter than 16 bytes (malformed).
	malformedFmt := []byte("RIFF")
	malformedFmt = le.AppendUint32(malformedFmt, 0)
	malformedFmt = append(malformedFmt, []byte("WAVE")...)
	malformedFmt = append(malformedFmt, []byte("fmt ")...)
	malformedFmt = le.AppendUint32(malformedFmt, 8) // < 16
	malformedFmt = append(malformedFmt, make([]byte, 8)...)
	if _, err := readWAVMono(write("malformedfmt.wav", malformedFmt), 16000); err == nil {
		t.Fatal("malformed fmt: error = nil, want the malformed-fmt error")
	}

	// An unsupported encoding (8-bit PCM) propagates the decode error.
	if _, err := readWAVMono(write("enc.wav", buildWAV(1, 1, 16000, 8, []byte{0})), 16000); err == nil {
		t.Fatal("8-bit: error = nil, want unsupported-encoding error")
	}
}

// decodeWAVSamples rejects a zero-channel declaration directly.
func TestDecodeWAVSamples_ZeroChannels_Bad(t *testing.T) {
	if _, err := decodeWAVSamples([]byte{0, 0}, 1, 0, 16); err == nil {
		t.Fatal("zero channels: error = nil, want a zero-channel error")
	}
}

// ---- admin_auth.go ----------------------------------------------------------

// loadAdminToken treats an empty/whitespace token file as "no token".
func TestLoadAdminToken_EmptyFile_Bad(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "admin.token")
	if r := core.WriteFile(path, []byte("   \n"), 0o600); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	tok, exists, err := loadAdminToken(path)
	if err != nil || exists || tok != "" {
		t.Fatalf("loadAdminToken(empty) = (%q, %v, %v), want (\"\", false, nil)", tok, exists, err)
	}
}

// writeAdminToken fails when the parent path is a FILE (mkdir can't create the
// dir) — the fail-closed write-error branch.
func TestWriteAdminToken_ParentIsFile_Bad(t *testing.T) {
	blocker := core.JoinPath(t.TempDir(), "blocker")
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("write blocker: %v", r.Value)
	}
	// The token path is under the blocker FILE, so MkdirAll(parent) fails.
	if err := writeAdminToken(core.JoinPath(blocker, "sub", "admin.token"), "lthn-mlx_x"); err == nil {
		t.Fatal("writeAdminToken under a file error = nil, want the mkdir/write failure")
	}
}

// ensureAdminToken returns the existing token without regenerating when one is
// already present.
func TestEnsureAdminToken_LoadsExisting_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "admin.token")
	if r := core.WriteFile(path, []byte("lthn-mlx_existing\n"), 0o600); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	tok, generated, err := ensureAdminToken(path)
	if err != nil || generated || tok != "lthn-mlx_existing" {
		t.Fatalf("ensureAdminToken(existing) = (%q, %v, %v), want the existing token not regenerated", tok, generated, err)
	}
}

// ---- admin.go (machine handler) ---------------------------------------------

// adminMachineHandler rejects a non-GET method (405).
func TestAdminMachineHandler_MethodGuard_Bad(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/machine", nil)
	w := httptest.NewRecorder()
	adminMachineHandler(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", w.Code)
	}
}

// adminMachineHandler returns 500 when the machine hash is unavailable.
func TestAdminMachineHandler_HashUnavailable_Bad(t *testing.T) {
	orig := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = orig; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("no device")
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	w := httptest.NewRecorder()
	adminMachineHandler(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", w.Code)
	}
}

// adminMachineHandler returns the machine identity JSON on success.
func TestAdminMachineHandler_Success_Good(t *testing.T) {
	orig := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = orig; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{Architecture: "apple9"} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{Labels: map[string]string{"machine_hash": "h-123"}}, nil
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	w := httptest.NewRecorder()
	adminMachineHandler(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%q", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "h-123") || !strings.Contains(w.Body.String(), "go-mlx") {
		t.Fatalf("body = %q, want the machine identity", w.Body.String())
	}
}

// ---- fuse.go / diffuse.go bad-flag branches ---------------------------------

// fuse with a bad flag fails to parse (exit 2).
func TestRunFuse_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"fuse", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// fuse with valid flags but a nonexistent base reaches the FuseModelDir error
// (exit 1).
func TestRunFuse_FuseError_Bad(t *testing.T) {
	dir := t.TempDir()
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"fuse", "-base", core.JoinPath(dir, "absent-base"), "-adapter", core.JoinPath(dir, "absent-adapter"), "-out", core.JoinPath(dir, "out"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (fuse error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "fuse:") {
		t.Fatalf("stderr = %q, want the fuse error", stderr.String())
	}
}

// diffuse with a bad flag fails to parse (exit 2).
func TestRunDiffuse_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"diffuse", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// ebook with a bad flag fails to parse (exit 2).
func TestRunEbook_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	if code := runCommand(context.Background(), []string{"ebook", "-not-a-flag"}, stdout, stderr); code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}
