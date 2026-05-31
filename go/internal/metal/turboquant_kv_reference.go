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

type TurboQuantKVProdReferenceVector struct {
	Codec        TurboQuantKVCodec              `json:"codec"`
	Base         TurboQuantKVMSEReferenceVector `json:"base"`
	ResidualNorm float32                        `json:"residual_norm"`
	QJLSigns     []byte                         `json:"qjl_signs"`
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

func EncodeTurboQuantKVProdReference(values []float32, codec TurboQuantKVCodec) (TurboQuantKVProdReferenceVector, error) {
	headDim := int32(len(values))
	if codec.Algorithm != TurboQuantKVAlgorithmProd {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod reference requires TurboQuantprod codec")
	}
	if err := codec.Validate("reference", headDim); err != nil {
		return TurboQuantKVProdReferenceVector{}, err
	}
	if codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod reference codebook is unsupported")
	}
	if codec.NormalBits > 8 {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod reference stores one byte per centroid code")
	}
	mseCodec := codec
	mseCodec.Algorithm = TurboQuantKVAlgorithmMSE
	mseCodec.QJLSeed = 0
	base, err := EncodeTurboQuantKVMSEReference(values, mseCodec)
	if err != nil {
		return TurboQuantKVProdReferenceVector{}, err
	}
	encoded := TurboQuantKVProdReferenceVector{
		Codec:    codec,
		Base:     base,
		QJLSigns: make([]byte, len(values)),
	}
	if base.Norm == 0 {
		return encoded, nil
	}
	decoded, err := base.DecodeMSE()
	if err != nil {
		return TurboQuantKVProdReferenceVector{}, err
	}
	residual := make([]float64, len(values))
	var residualNormSq float64
	for idx := range values {
		delta := (float64(values[idx]) - float64(decoded[idx])) / float64(base.Norm)
		residual[idx] = delta
		residualNormSq += delta * delta
	}
	residualNorm := math.Sqrt(residualNormSq)
	encoded.ResidualNorm = float32(residualNorm)
	if residualNorm == 0 {
		return encoded, nil
	}
	rotatedResidual := make([]float64, len(values))
	turboQuantKVReferenceRotate(rotatedResidual, residual, codec.QJLSeed, false)
	for idx, value := range rotatedResidual {
		if value < 0 {
			encoded.QJLSigns[idx] = 1
		}
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

func (encoded TurboQuantKVProdReferenceVector) EstimateInnerProduct(query []float32) (float32, error) {
	if len(query) != int(encoded.Base.HeadDim) {
		return 0, core.NewError("mlx: TurboQuantprod reference query shape is invalid")
	}
	if len(encoded.QJLSigns) != len(query) {
		return 0, core.NewError("mlx: TurboQuantprod reference QJL signs are invalid")
	}
	base, err := encoded.Base.DecodeMSE()
	if err != nil {
		return 0, err
	}
	estimate := turboQuantKVReferenceDot(query, base)
	if encoded.Base.Norm == 0 || encoded.ResidualNorm == 0 {
		return estimate, nil
	}
	query64 := make([]float64, len(query))
	for idx, value := range query {
		query64[idx] = float64(value)
	}
	rotatedQuery := make([]float64, len(query))
	turboQuantKVReferenceRotate(rotatedQuery, query64, encoded.Codec.QJLSeed, false)
	scale := float64(encoded.Base.Norm) * float64(encoded.ResidualNorm) / math.Sqrt(float64(len(query)))
	for idx, value := range rotatedQuery {
		sign := 1.0
		if encoded.QJLSigns[idx] != 0 {
			sign = -1
		}
		estimate += float32(scale * sign * value)
	}
	return estimate, nil
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

func turboQuantKVReferenceDot(a, b []float32) float32 {
	var sum float32
	for idx := range a {
		sum += a[idx] * b[idx]
	}
	return sum
}
