// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// This file is the payload-level guard for the K-quant encoders
// (quantizeQ2_K / quantizeQ3_K / quantizeQ6_K / quantizeQ8_K). The
// encoders were rewritten to the canonical ggml block layout
// (lib/gguflib/gguflib.c + upstream ggml-common.h); these tests assert
// the produced bytes at the decoder's struct-field offsets and that a
// canonical decode round-trips within the K-quant error tolerance — so a
// future regression is caught at the bytes, not merely the block size.
//
// The reference decoders below mirror upstream dequantize_row_q{2,3,6,8}_K
// exactly (verbatim index/shift arithmetic), and are the payload arbiter
// because the go/gguf package is pure-Go and never links the C decoder.

// --- canonical reference decoders (mirror upstream dequantize_row_*) ---

func refF16(b []byte) float32 { return safetensors.Float16ToFloat32(binary.LittleEndian.Uint16(b)) }

// dequantQ6KRef mirrors dequantize_row_q6_K verbatim: ql[128] qh[64]
// scales[16 i8] d(f16), decoded in 128-element groups (ql+=64, qh+=32,
// sc+=8 per group).
func dequantQ6KRef(t *testing.T, data []byte, n int) []float32 {
	t.Helper()
	const blk = 210
	out := make([]float32, n)
	for sb := 0; sb*blk < len(data); sb++ {
		block := data[sb*blk : sb*blk+blk]
		d := refF16(block[208:210])
		ql := block[0:128]
		qh := block[128:192]
		sc := block[192:208]
		y := out[sb*256:]
		qlOff, qhOff, scOff := 0, 0, 0
		for nn := 0; nn < 256; nn += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int32(int8((ql[qlOff+l+0]&0xF)|(((qh[qhOff+l]>>0)&3)<<4))) - 32
				q2 := int32(int8((ql[qlOff+l+32]&0xF)|(((qh[qhOff+l]>>2)&3)<<4))) - 32
				q3 := int32(int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4))) - 32
				q4 := int32(int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4))) - 32
				y[nn+l+0] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
				y[nn+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
				y[nn+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
				y[nn+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
	return out
}

// dequantQ2KRef mirrors dequantize_row_q2_K: scales[16] qs[64] d(f16) dmin(f16).
func dequantQ2KRef(t *testing.T, data []byte, n int) []float32 {
	t.Helper()
	const blk = 84
	out := make([]float32, 0, n)
	for sb := 0; sb*blk < len(data); sb++ {
		block := data[sb*blk : sb*blk+blk]
		scales := block[0:16]
		qs := block[16:80]
		d := refF16(block[80:82])
		dmin := refF16(block[82:84])
		is := 0
		for nn := 0; nn < 256; nn += 128 {
			q := qs[nn/4:]
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl, ml := d*float32(sc&0xF), dmin*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out = append(out, dl*float32((q[l]>>shift)&3)-ml)
				}
				sc = scales[is]
				is++
				dl, ml = d*float32(sc&0xF), dmin*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out = append(out, dl*float32((q[l+16]>>shift)&3)-ml)
				}
				shift += 2
			}
		}
	}
	return out
}

// dequantQ3KRef mirrors dequantize_row_q3_K: hmask[32] qs[64] scales[12] d(f16).
func dequantQ3KRef(t *testing.T, data []byte, n int) []float32 {
	t.Helper()
	const blk = 110
	out := make([]float32, 0, n)
	for sb := 0; sb*blk < len(data); sb++ {
		block := data[sb*blk : sb*blk+blk]
		hmask := block[0:32]
		qs := block[32:96]
		d := refF16(block[108:110])
		scales := unpackQ3KScalesRef(block[96:108])
		m := byte(1)
		is := 0
		for nn := 0; nn < 256; nn += 128 {
			q := qs[nn/4:]
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := d * float32(scales[is])
				is++
				for l := 0; l < 16; l++ {
					high := int32(0)
					if hmask[l]&m == 0 {
						high = -4
					}
					out = append(out, dl*float32(int32((q[l]>>shift)&3)+high))
				}
				dl = d * float32(scales[is])
				is++
				for l := 0; l < 16; l++ {
					high := int32(0)
					if hmask[16+l]&m == 0 {
						high = -4
					}
					out = append(out, dl*float32(int32((q[l+16]>>shift)&3)+high))
				}
				shift += 2
				m <<= 1
			}
		}
	}
	return out
}

// unpackQ3KScalesRef mirrors the kmask aux logic in dequantize_row_q3_K.
func unpackQ3KScalesRef(sc []byte) [16]int8 {
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(sc[0:4])
	aux[1] = binary.LittleEndian.Uint32(sc[4:8])
	aux[2] = binary.LittleEndian.Uint32(sc[8:12])
	const k1, k2 = uint32(0x03030303), uint32(0x0f0f0f0f)
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
	aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
	aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
	aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
	var out [16]int8
	for i := 0; i < 4; i++ {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], aux[i])
		for k := 0; k < 4; k++ {
			out[i*4+k] = int8(b[k]) - 32
		}
	}
	return out
}

// dequantQ8KRef mirrors dequantize_row_q8_K: d(f32) qs[256 i8] bsums[16 i16].
func dequantQ8KRef(t *testing.T, data []byte, n int) []float32 {
	t.Helper()
	const blk = 292
	out := make([]float32, 0, n)
	for sb := 0; sb*blk < len(data); sb++ {
		block := data[sb*blk : sb*blk+blk]
		d := math.Float32frombits(binary.LittleEndian.Uint32(block[0:4]))
		qs := block[4:260]
		for i := 0; i < 256; i++ {
			out = append(out, d*float32(int8(qs[i])))
		}
	}
	return out
}

// --- payload-level tests ---

// noisyKQuantInput builds an input whose blocks have real per-sub-block
// dynamic range (not a monotone ramp), so the affine/symmetric fits are
// exercised and round-trip error is meaningful.
func noisyKQuantInput(nElems int) []float32 {
	v := make([]float32, nElems)
	for i := range v {
		// Deterministic pseudo-random in roughly [-4, 4].
		x := math.Sin(float64(i)*0.37) + 0.5*math.Cos(float64(i)*1.11)
		v[i] = float32(x * 3.0)
	}
	return v
}

func TestQuantizeQ6K_PayloadRoundTrip_Good(t *testing.T) {
	in := noisyKQuantInput(256 * 3)
	enc := quantizeQ6_K(in)
	if len(enc) != 3*210 {
		t.Fatalf("Q6_K emitted %d bytes, want %d", len(enc), 3*210)
	}
	got := dequantQ6KRef(t, enc, len(in))
	assertRoundTrip(t, "Q6_K", in, got, 0.08)
}

func TestQuantizeQ2K_PayloadRoundTrip_Good(t *testing.T) {
	in := noisyKQuantInput(256 * 3)
	enc := quantizeQ2_K(in)
	if len(enc) != 3*84 {
		t.Fatalf("Q2_K emitted %d bytes, want %d", len(enc), 3*84)
	}
	got := dequantQ2KRef(t, enc, len(in))
	// 2-bit affine is coarse, so the tolerance is loose — but relative
	// RMSE (not correlation) is the regression pin: it catches a
	// systematic scale-packing drift that a magnitude-blind Pearson
	// correlation would sail through.
	logRelRMSE(t, "Q2_K", in, got) // measured ~0.25 for this input
	assertRoundTrip(t, "Q2_K", in, got, 0.32)
}

func TestQuantizeQ3K_PayloadRoundTrip_Good(t *testing.T) {
	in := noisyKQuantInput(256 * 3)
	enc := quantizeQ3_K(in)
	if len(enc) != 3*110 {
		t.Fatalf("Q3_K emitted %d bytes, want %d", len(enc), 3*110)
	}
	got := dequantQ3KRef(t, enc, len(in))
	logRelRMSE(t, "Q3_K", in, got) // measured ~0.16 for this input
	assertRoundTrip(t, "Q3_K", in, got, 0.2)
}

func TestQuantizeQ8K_PayloadRoundTrip_Good(t *testing.T) {
	in := noisyKQuantInput(256 * 3)
	enc := quantizeQ8_K(in)
	if len(enc) != 3*292 {
		t.Fatalf("Q8_K emitted %d bytes, want %d", len(enc), 3*292)
	}
	got := dequantQ8KRef(t, enc, len(in))
	assertRoundTrip(t, "Q8_K", in, got, 0.02)
	// bsums field must equal the sum of the qs sub-block it summarises.
	assertQ8KBsums(t, enc)
}

// assertRoundTrip checks the dequantised values track the input within
// maxRelRMSE (RMSE normalised by the input's RMS magnitude).
func assertRoundTrip(t *testing.T, name string, in, got []float32, maxRelRMSE float64) {
	t.Helper()
	if len(in) != len(got) {
		t.Fatalf("%s: decoded %d values, want %d", name, len(got), len(in))
	}
	var se, ss float64
	for i := range in {
		d := float64(got[i] - in[i])
		se += d * d
		ss += float64(in[i]) * float64(in[i])
	}
	rmse := math.Sqrt(se / float64(len(in)))
	rms := math.Sqrt(ss / float64(len(in)))
	rel := rmse / (rms + 1e-9)
	if rel > maxRelRMSE {
		t.Fatalf("%s: relative RMSE %.4f exceeds tolerance %.4f (rmse=%.4f rms=%.4f)", name, rel, maxRelRMSE, rmse, rms)
	}
}

// logRelRMSE prints the measured relative RMSE so the chosen tolerances
// are visible (and re-tunable) in test output.
func logRelRMSE(t *testing.T, name string, in, got []float32) {
	t.Helper()
	var se, ss float64
	for i := range in {
		d := float64(got[i] - in[i])
		se += d * d
		ss += float64(in[i]) * float64(in[i])
	}
	rmse := math.Sqrt(se / float64(len(in)))
	rms := math.Sqrt(ss / float64(len(in)))
	t.Logf("%s relative RMSE = %.4f (rmse=%.4f rms=%.4f)", name, rmse/(rms+1e-9), rmse, rms)
}

// assertQ8KBsums verifies each int16 bsums entry equals the sum of the 16
// qs it covers — a structural check on that field's offset and content.
func assertQ8KBsums(t *testing.T, enc []byte) {
	t.Helper()
	const blk = 292
	for sb := 0; sb*blk < len(enc); sb++ {
		block := enc[sb*blk : sb*blk+blk]
		qs := block[4:260]
		for g := 0; g < 16; g++ {
			var want int16
			for j := 0; j < 16; j++ {
				want += int16(int8(qs[g*16+j]))
			}
			got := int16(binary.LittleEndian.Uint16(block[260+g*2 : 262+g*2]))
			if got != want {
				t.Fatalf("Q8_K block %d bsums[%d] = %d, want %d", sb, g, got, want)
			}
		}
	}
}

// TestQuantizeQ3KScalePack_RoundTrips asserts packQ3KScales is the exact
// inverse of the decoder's kmask unpack across the full 6-bit range.
func TestQuantizeQ3KScalePack_RoundTrips(t *testing.T) {
	for trial := 0; trial < 64; trial++ {
		var raw [qkSubBlocks]uint8
		for i := range raw {
			raw[i] = uint8((trial*7 + i*13) % 64) // 0..63
		}
		var packed [12]byte
		packQ3KScales(raw, &packed)
		got := unpackQ3KScalesRef(packed[:])
		for i := range raw {
			want := int8(raw[i]) - 32
			if got[i] != want {
				t.Fatalf("trial %d scale[%d]: packed %d -> unpacked %d, want %d", trial, i, raw[i], got[i], want)
			}
		}
	}
}

// TestQuantizeModelPack_AllKQuants_Good is the end-to-end guard: every
// K-quant now survives QuantizeModelPack -> ReadInfo with no block-size
// error (the bug the rewrite fixed) and reports the right ggml type.
func TestQuantizeModelPack_AllKQuants_Good(t *testing.T) {
	cases := []struct {
		format   QuantizeFormat
		typeName string
		bits     int
	}{
		{QuantizeQ2_K, "q2_k", 2},
		{QuantizeQ3_K, "q3_k", 3},
		{QuantizeQ6_K, "q6_k", 6},
		{QuantizeQ8_K, "q8_k", 8},
	}
	for _, tc := range cases {
		t.Run(string(tc.format), func(t *testing.T) {
			rows := 2
			source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
				{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{256, rows}, Data: ascendingFloat32s(256 * rows)},
				{Name: "model.norm.weight", Shape: []int{256}, Data: ascendingFloat32s(256)},
			})
			output := core.PathJoin(t.TempDir(), "out-"+string(tc.format))
			result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
				SourcePack: sourcePackFromDir(source),
				OutputPath: output,
				Format:     tc.format,
			})
			if err != nil {
				t.Fatalf("QuantizeModelPack(%s) error = %v", tc.format, err)
			}
			if result.Format != tc.format || result.QuantizedTensors != 2 {
				t.Fatalf("%s: result = %+v", tc.format, result)
			}
			info, err := ReadInfo(output)
			if err != nil {
				t.Fatalf("ReadInfo(%s) error = %v", tc.format, err)
			}
			if !info.Valid() {
				t.Fatalf("%s: validation issues = %+v", tc.format, info.ValidationIssues)
			}
			weight := findTensorByName(info.Tensors, "model.layers.0.self_attn.q_proj.weight")
			if weight == nil {
				t.Fatalf("%s: quantised weight missing", tc.format)
			}
			if weight.TypeName != tc.typeName || weight.Bits != tc.bits || weight.BlockSize != 256 {
				t.Fatalf("%s: weight = type:%q bits:%d block:%d, want %q/%d/256",
					tc.format, weight.TypeName, weight.Bits, weight.BlockSize, tc.typeName, tc.bits)
			}
		})
	}
}
