// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"

	core "dappco.re/go"
)

const TurboQuantKVReferenceCodebookUniform = "uniform-fwht"

type TurboQuantKVMSEReferenceVector struct {
	Codec         TurboQuantKVCodec `json:"codec"`
	HeadDim       int32             `json:"head_dim"`
	Norm          float32           `json:"norm"`
	CentroidCodes []byte            `json:"centroid_codes"`
}

func EncodeTurboQuantKVMSEReference(values []float32, codec TurboQuantKVCodec) (TurboQuantKVMSEReferenceVector, error) {
	headDim := int32(len(values))
	if codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE reference requires TurboQuantmse codec")
	}
	if err := codec.Validate("reference", headDim); err != nil {
		return TurboQuantKVMSEReferenceVector{}, err
	}
	if codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE reference codebook is unsupported")
	}
	if codec.NormalBits > 8 {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE reference stores one byte per centroid code")
	}
	if !turboQuantKVReferenceHeadDimSupported(len(values)) {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE reference requires a non-empty power-of-two head dimension")
	}
	encoded := TurboQuantKVMSEReferenceVector{
		Codec:         codec,
		HeadDim:       headDim,
		CentroidCodes: make([]byte, len(values)),
	}
	norm := turboQuantKVReferenceNorm(values)
	encoded.Norm = float32(norm)
	if norm == 0 {
		return encoded, nil
	}
	normalised := make([]float64, len(values))
	for idx, value := range values {
		normalised[idx] = float64(value) / norm
	}
	rotated := make([]float64, len(values))
	turboQuantKVReferenceRotate(rotated, normalised, codec.RotationSeed, false)
	for idx, value := range rotated {
		encoded.CentroidCodes[idx] = turboQuantKVReferenceQuantizeUniform(value, codec.NormalBits)
	}
	return encoded, nil
}

func (encoded TurboQuantKVMSEReferenceVector) DecodeMSE() ([]float32, error) {
	if encoded.HeadDim <= 0 || len(encoded.CentroidCodes) != int(encoded.HeadDim) {
		return nil, core.NewError("mlx: TurboQuant MSE reference vector shape is invalid")
	}
	if encoded.Codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return nil, core.NewError("mlx: TurboQuant MSE reference decode requires TurboQuantmse codec")
	}
	if encoded.Codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return nil, core.NewError("mlx: TurboQuant MSE reference codebook is unsupported")
	}
	if encoded.Codec.NormalBits > 8 {
		return nil, core.NewError("mlx: TurboQuant MSE reference stores one byte per centroid code")
	}
	if !turboQuantKVReferenceHeadDimSupported(int(encoded.HeadDim)) {
		return nil, core.NewError("mlx: TurboQuant MSE reference requires a power-of-two head dimension")
	}
	decoded := make([]float32, encoded.HeadDim)
	if encoded.Norm == 0 {
		return decoded, nil
	}
	rotated := make([]float64, encoded.HeadDim)
	for idx, code := range encoded.CentroidCodes {
		rotated[idx] = turboQuantKVReferenceDequantizeUniform(code, encoded.Codec.NormalBits)
	}
	normalised := make([]float64, encoded.HeadDim)
	turboQuantKVReferenceRotate(normalised, rotated, encoded.Codec.RotationSeed, true)
	for idx, value := range normalised {
		decoded[idx] = float32(value * float64(encoded.Norm))
	}
	return decoded, nil
}

func turboQuantKVReferenceHeadDimSupported(dim int) bool {
	return dim > 0 && dim&(dim-1) == 0
}

func turboQuantKVReferenceNorm(values []float32) float64 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func turboQuantKVReferenceRotate(dst, src []float64, seed uint64, inverse bool) {
	if inverse {
		copy(dst, src)
		turboQuantKVReferenceFWHT(dst)
		turboQuantKVReferenceSignFlip(dst, seed)
		return
	}
	for idx, value := range src {
		if turboQuantKVReferenceSign(seed, idx) < 0 {
			dst[idx] = -value
			continue
		}
		dst[idx] = value
	}
	turboQuantKVReferenceFWHT(dst)
}

func turboQuantKVReferenceFWHT(values []float64) {
	n := len(values)
	for step := 1; step < n; step <<= 1 {
		for start := 0; start < n; start += step << 1 {
			for idx := 0; idx < step; idx++ {
				left := values[start+idx]
				right := values[start+idx+step]
				values[start+idx] = left + right
				values[start+idx+step] = left - right
			}
		}
	}
	scale := 1 / math.Sqrt(float64(n))
	for idx := range values {
		values[idx] *= scale
	}
}

func turboQuantKVReferenceSignFlip(values []float64, seed uint64) {
	for idx := range values {
		if turboQuantKVReferenceSign(seed, idx) < 0 {
			values[idx] = -values[idx]
		}
	}
}

func turboQuantKVReferenceSign(seed uint64, idx int) int {
	mixed := seed + uint64(idx)*0x9e3779b97f4a7c15
	mixed ^= mixed >> 30
	mixed *= 0xbf58476d1ce4e5b9
	mixed ^= mixed >> 27
	mixed *= 0x94d049bb133111eb
	mixed ^= mixed >> 31
	if mixed&1 == 0 {
		return 1
	}
	return -1
}

func turboQuantKVReferenceQuantizeUniform(value float64, bits int) byte {
	levels := (1 << bits) - 1
	if value < -1 {
		value = -1
	}
	if value > 1 {
		value = 1
	}
	quantized := math.Round((value + 1) * float64(levels) / 2)
	if quantized < 0 {
		return 0
	}
	if quantized > float64(levels) {
		return byte(levels)
	}
	return byte(quantized)
}

func turboQuantKVReferenceDequantizeUniform(code byte, bits int) float64 {
	levels := (1 << bits) - 1
	if levels <= 0 {
		return 0
	}
	if int(code) > levels {
		code = byte(levels)
	}
	return (float64(code)*2)/float64(levels) - 1
}
