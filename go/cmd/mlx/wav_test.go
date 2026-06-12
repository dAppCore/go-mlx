// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

// writeTestWAV synthesises a minimal RIFF/WAVE file.
func writeTestWAV(t *testing.T, path string, format, channels uint16, rate uint32, samples []float32) {
	t.Helper()
	bits := uint16(16)
	perSample := 2
	if format == 3 {
		bits, perSample = 32, 4
	}
	le := binary.LittleEndian
	dataLen := len(samples) * perSample
	buf := make([]byte, 0, 44+dataLen)
	u32 := func(v uint32) []byte { b := make([]byte, 4); le.PutUint32(b, v); return b }
	u16 := func(v uint16) []byte { b := make([]byte, 2); le.PutUint16(b, v); return b }

	buf = append(buf, "RIFF"...)
	buf = append(buf, u32(uint32(36+dataLen))...)
	buf = append(buf, "WAVE"...)
	buf = append(buf, "fmt "...)
	buf = append(buf, u32(16)...)
	buf = append(buf, u16(format)...)
	buf = append(buf, u16(channels)...)
	buf = append(buf, u32(rate)...)
	buf = append(buf, u32(rate*uint32(channels)*uint32(perSample))...)
	buf = append(buf, u16(channels*uint16(perSample))...)
	buf = append(buf, u16(bits)...)
	buf = append(buf, "data"...)
	buf = append(buf, u32(uint32(dataLen))...)
	for _, s := range samples {
		if format == 3 {
			buf = append(buf, u32(math.Float32bits(s))...)
		} else {
			buf = append(buf, u16(uint16(int16(s*32767)))...)
		}
	}
	if r := core.WriteFile(path, buf, 0o600); !r.OK {
		t.Fatalf("write test wav: %v", r)
	}
}

func TestReadWAVMono_PCM16_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "tone.wav")
	want := []float32{0, 0.25, -0.25, 0.5, -0.5, 1, -1, 0}
	writeTestWAV(t, path, 1, 1, 16000, want)

	got, err := readWAVMono(path, 16000)
	if err != nil {
		t.Fatalf("readWAVMono: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("samples = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3 {
			t.Fatalf("sample %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestReadWAVMono_Float32Stereo_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "stereo.wav")
	// Interleaved L/R pairs; mono downmix averages each frame.
	writeTestWAV(t, path, 3, 2, 16000, []float32{0.5, 0.1, -0.4, -0.2})

	got, err := readWAVMono(path, 16000)
	if err != nil {
		t.Fatalf("readWAVMono: %v", err)
	}
	want := []float32{0.3, -0.3}
	if len(got) != len(want) {
		t.Fatalf("frames = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
			t.Fatalf("frame %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestReadWAVMono_Bad(t *testing.T) {
	dir := t.TempDir()
	rateMismatch := core.PathJoin(dir, "rate.wav")
	writeTestWAV(t, rateMismatch, 1, 1, 44100, []float32{0, 0.5})
	if _, err := readWAVMono(rateMismatch, 16000); err == nil {
		t.Fatal("44.1 kHz accepted for a 16 kHz model")
	}

	notWav := core.PathJoin(dir, "not.wav")
	if r := core.WriteFile(notWav, []byte("definitely not a riff file, just text padding"), 0o600); !r.OK {
		t.Fatal("write stub")
	}
	if _, err := readWAVMono(notWav, 16000); err == nil {
		t.Fatal("non-WAV accepted")
	}

	if _, err := readWAVMono(core.PathJoin(dir, "missing.wav"), 16000); err == nil {
		t.Fatal("missing file accepted")
	}
}
