// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

// readWAVMono reads a RIFF/WAVE file into mono float32 samples in [-1, 1].
// Accepts PCM16 (format 1) and IEEE float32 (format 3); stereo downmixes by
// averaging. The sample rate must match wantRate — resampling is out of
// scope (the honest error names the fix).
func readWAVMono(path string, wantRate int32) ([]float32, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("mlx.audio", core.Sprintf("read %s", path), nil)
	}
	data, ok := read.Value.([]byte)
	if !ok || len(data) < 44 {
		return nil, core.NewError("mlx: not a WAV file (too short)")
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, core.NewError("mlx: not a RIFF/WAVE file")
	}

	var (
		format      uint16
		channels    uint16
		sampleRate  uint32
		bitsPerSamp uint16
		samples     []float32
		haveFmt     bool
	)
	le := binary.LittleEndian
	offset := 12
	for offset+8 <= len(data) {
		chunkID := string(data[offset : offset+4])
		chunkLen := int(le.Uint32(data[offset+4 : offset+8]))
		body := offset + 8
		if body+chunkLen > len(data) {
			return nil, core.NewError("mlx: truncated WAV chunk")
		}
		switch chunkID {
		case "fmt ":
			if chunkLen < 16 {
				return nil, core.NewError("mlx: malformed WAV fmt chunk")
			}
			format = le.Uint16(data[body : body+2])
			channels = le.Uint16(data[body+2 : body+4])
			sampleRate = le.Uint32(data[body+4 : body+8])
			bitsPerSamp = le.Uint16(data[body+14 : body+16])
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, core.NewError("mlx: WAV data chunk before fmt chunk")
			}
			decoded, err := decodeWAVSamples(data[body:body+chunkLen], format, channels, bitsPerSamp)
			if err != nil {
				return nil, err
			}
			samples = decoded
		}
		// Chunks are word-aligned: odd lengths carry one pad byte.
		offset = body + chunkLen + (chunkLen & 1)
	}
	if samples == nil {
		return nil, core.NewError("mlx: WAV file has no data chunk")
	}
	if int32(sampleRate) != wantRate {
		return nil, core.E("mlx.audio", core.Sprintf(
			"WAV sample rate %d Hz, model wants %d Hz — resample first (e.g. ffmpeg -i in.wav -ar %d -ac 1 out.wav)",
			sampleRate, wantRate, wantRate), nil)
	}
	return samples, nil
}

func decodeWAVSamples(body []byte, format, channels, bits uint16) ([]float32, error) {
	if channels == 0 {
		return nil, core.NewError("mlx: WAV declares zero channels")
	}
	var perSample int
	switch {
	case format == 1 && bits == 16:
		perSample = 2
	case format == 3 && bits == 32:
		perSample = 4
	default:
		return nil, core.E("mlx.audio", core.Sprintf("unsupported WAV encoding: format %d, %d-bit (want PCM16 or float32)", format, bits), nil)
	}
	frame := perSample * int(channels)
	frames := len(body) / frame
	out := make([]float32, frames)
	le := binary.LittleEndian
	for i := 0; i < frames; i++ {
		sum := float32(0)
		for c := 0; c < int(channels); c++ {
			at := i*frame + c*perSample
			switch perSample {
			case 2:
				sum += float32(int16(le.Uint16(body[at:at+2]))) / 32768.0
			case 4:
				sum += math.Float32frombits(le.Uint32(body[at : at+4]))
			}
		}
		out[i] = sum / float32(channels)
	}
	return out, nil
}
