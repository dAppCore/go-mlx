// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
)

// The quant siblings of the decode bookends: in a 4-bit gemma4 checkpoint the embedding
// table is itself quantised (mlx quantises nn.Embedding, and gemma ties the LM head to it),
// so the input embedding must dequantise a gathered row and the LM head is a quantised
// projection. bf16 []byte throughout (the seam's lingua franca).

// EmbedTokensQuant is the gemma4 input embedding for a 4-bit affine-quantised table: it
// gathers each token's row and dequantises it on the HOST (value = scale·code + bias per
// group, 4-bit codes unpacked from the packed bytes), then applies `scale` (= √hidden,
// metal's EmbeddingScale). Only the gathered rows are dequantised — not the whole table — so a
// 4-bit embedding stays 4-bit in memory. packed is the [vocab × dModel] affine-packed weight
// (dModel·bits/8 bytes per row), scales/biases the per-group bf16 (dModel/groupSize per row).
// Pure host (no device): byte-for-byte equal to metal.Dequantize on the gathered rows (gated).
func EmbedTokensQuant(packed, scales, biases []byte, tokenIDs []int32, vocab, dModel, groupSize, bits int, scale float32) ([][]byte, error) {
	if bits <= 0 || bits > 8 {
		return nil, core.NewError("native.EmbedTokensQuant: bits must be in 1..8")
	}
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, core.NewError("native.EmbedTokensQuant: groupSize must be > 0 and divide dModel")
	}
	groups := dModel / groupSize
	rowPacked := dModel * bits / 8 // packed bytes per row (dModel/2 for 4-bit)
	rowSB := groups * bf16Size     // scales (or biases) bytes per row
	if len(packed) != vocab*rowPacked {
		return nil, core.NewError("native.EmbedTokensQuant: packed size != vocab·dModel·bits/8")
	}
	if len(scales) != vocab*rowSB || len(biases) != vocab*rowSB {
		return nil, core.NewError("native.EmbedTokensQuant: scales/biases size != vocab·(dModel/groupSize) bf16")
	}
	out := make([][]byte, len(tokenIDs))
	for i, tok := range tokenIDs {
		if tok < 0 || int(tok) >= vocab {
			return nil, core.NewError("native.EmbedTokensQuant: token id out of range")
		}
		pRow := packed[int(tok)*rowPacked : (int(tok)+1)*rowPacked]
		sRow := scales[int(tok)*rowSB : (int(tok)+1)*rowSB]
		bRow := biases[int(tok)*rowSB : (int(tok)+1)*rowSB]
		emb := make([]byte, dModel*bf16Size)
		for c := 0; c < dModel; c++ {
			// affine codes are bit-packed LSB-first contiguous (the 4-bit nibble-low-first layout
			// generalised to any width) — extractAffineCode spans byte boundaries for 5/6-bit.
			code := extractAffineCode(pRow, c*bits, bits)
			g := c / groupSize
			s := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
			b := bf16ToF32(bRow[g*bf16Size], bRow[g*bf16Size+1])
			h := f32ToBF16((s*float32(code) + b) * scale)
			emb[c*bf16Size] = byte(h)
			emb[c*bf16Size+1] = byte(h >> 8)
		}
		out[i] = emb
	}
	return out, nil
}

// extractAffineCode reads the bits-wide affine code at bit offset bitOff from a packed row,
// LSB-first contiguous — MLX's affine packing (the 4-bit nibble-low-first layout generalised),
// spanning byte boundaries for non-byte-aligned widths (5/6-bit). For 4-bit it reduces to the
// nibble read, for 8-bit to the byte read.
func extractAffineCode(p []byte, bitOff, bits int) uint32 {
	var v uint32
	for got := 0; got < bits; {
		bi := (bitOff + got) / 8
		off := (bitOff + got) % 8
		take := 8 - off
		if take > bits-got {
			take = bits - got
		}
		chunk := (uint32(p[bi]) >> uint(off)) & ((1 << uint(take)) - 1)
		v |= chunk << uint(got)
		got += take
	}
	return v
}

// LMHeadQuant is the gemma4 output head when the LM projection is 4-bit quantised (the tied
// embedding of a 4-bit checkpoint): final RMSNorm, the quantised output projection (QMVBF16
// over the packed embedding), then the optional final-logit soft-cap (monotonic, preserves the
// argmax). hidden/finalNormW are dModel bf16; packed/scales/biases are the [vocab × dModel]
// affine-quant embedding; returns vocab bf16 logits. Norm + projection run on-device (QMVBF16
// is byte-parity-gated vs metal.QuantizedMatmul), the soft-cap is a host pass. softCap <= 0
// skips the cap.
func LMHeadQuant(hidden, finalNormW, packed, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadQuant: hidden must be dModel bf16 bytes")
	}
	if len(finalNormW) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadQuant: finalNormW must be dModel bf16 bytes")
	}
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, core.NewError("native.LMHeadQuant: groupSize must be > 0 and divide dModel")
	}
	wantPacked := vocab * dModel * bits / 8
	wantSB := vocab * (dModel / groupSize) * bf16Size
	if len(packed) != wantPacked || len(scales) != wantSB || len(biases) != wantSB {
		return nil, core.NewError("native.LMHeadQuant: packed/scales/biases size mismatch vs vocab·dModel")
	}
	logits := make([]byte, vocab*bf16Size)
	if dModel == 0 || vocab == 0 {
		return logits, nil
	}
	var encErr error
	withAutoreleasePool(func() {
		hiddenBuf := sharedBytes(hidden)
		finalNormBuf := residentBytes(finalNormW)
		packedBuf, scalesBuf, biasesBuf := residentBytes(packed), residentBytes(scales), residentBytes(biases)
		normed := scratchBF16(dModel)
		logitsBuf := scratchBF16(vocab)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRMSNormBF16(enc, hiddenBuf, finalNormBuf, normed, 0, dModel, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encQMVBF16(enc, packedBuf, scalesBuf, biasesBuf, normed, logitsBuf, 0, 0, 0, 0, vocab, dModel, groupSize, bits); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(logits, unsafe.Slice((*byte)(logitsBuf.Contents()), len(logits)))
	})
	if encErr != nil {
		return nil, encErr
	}
	if softCap > 0 {
		for i := 0; i < vocab; i++ {
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
			h := f32ToBF16(softCap * float32(math.Tanh(float64(v/softCap))))
			logits[i*bf16Size] = byte(h)
			logits[i*bf16Size+1] = byte(h >> 8)
		}
	}
	return logits, nil
}
