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

type TurboQuantKVReferencePage struct {
	Layout TurboQuantKVPageLayout            `json:"layout"`
	Keys   []TurboQuantKVProdReferenceVector `json:"keys"`
	Values []TurboQuantKVMSEReferenceVector  `json:"values"`
}

type turboQuantKVReferenceEncodeScratch struct {
	normalised []float64
	rotated    []float64
	residual   []float64
}

type turboQuantKVReferenceDecodeScratch struct {
	normalised []float64
	rotated    []float64
}

type turboQuantKVReferenceEstimateScratch struct {
	baseNormalised []float64
	rotatedQuery   []float64
}

func (scratch *turboQuantKVReferenceEncodeScratch) ensureMSE(dim int) {
	if cap(scratch.normalised) < dim {
		scratch.normalised = make([]float64, dim)
	} else {
		scratch.normalised = scratch.normalised[:dim]
	}
	if cap(scratch.rotated) < dim {
		scratch.rotated = make([]float64, dim)
	} else {
		scratch.rotated = scratch.rotated[:dim]
	}
}

func (scratch *turboQuantKVReferenceDecodeScratch) ensure(dim int) {
	if cap(scratch.normalised) < dim {
		scratch.normalised = make([]float64, dim)
	} else {
		scratch.normalised = scratch.normalised[:dim]
	}
	if cap(scratch.rotated) < dim {
		scratch.rotated = make([]float64, dim)
	} else {
		scratch.rotated = scratch.rotated[:dim]
	}
}

func (scratch *turboQuantKVReferenceEstimateScratch) ensure(dim int) {
	if cap(scratch.baseNormalised) < dim {
		scratch.baseNormalised = make([]float64, dim)
	} else {
		scratch.baseNormalised = scratch.baseNormalised[:dim]
	}
	if cap(scratch.rotatedQuery) < dim {
		scratch.rotatedQuery = make([]float64, dim)
	} else {
		scratch.rotatedQuery = scratch.rotatedQuery[:dim]
	}
}

func (scratch *turboQuantKVReferenceEncodeScratch) ensureProd(dim int) {
	scratch.ensureMSE(dim)
	if cap(scratch.residual) < dim {
		scratch.residual = make([]float64, dim)
	} else {
		scratch.residual = scratch.residual[:dim]
	}
}

func EncodeTurboQuantKVMSEReference(values []float32, codec TurboQuantKVCodec) (TurboQuantKVMSEReferenceVector, error) {
	var scratch turboQuantKVReferenceEncodeScratch
	return encodeTurboQuantKVMSEReference(values, codec, &scratch)
}

func encodeTurboQuantKVMSEReference(values []float32, codec TurboQuantKVCodec, scratch *turboQuantKVReferenceEncodeScratch) (TurboQuantKVMSEReferenceVector, error) {
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
	if scratch == nil {
		scratch = &turboQuantKVReferenceEncodeScratch{}
	}
	scratch.ensureMSE(len(values))
	normalised := scratch.normalised
	for idx, value := range values {
		normalised[idx] = float64(value) / norm
	}
	rotated := scratch.rotated
	turboQuantKVReferenceRotate(rotated, normalised, codec.RotationSeed, false)
	for idx, value := range rotated {
		encoded.CentroidCodes[idx] = turboQuantKVReferenceQuantizeUniform(value, codec.bitsForChannel(int32(idx)))
	}
	return encoded, nil
}

func EncodeTurboQuantKVProdReference(values []float32, codec TurboQuantKVCodec) (TurboQuantKVProdReferenceVector, error) {
	var scratch turboQuantKVReferenceEncodeScratch
	return encodeTurboQuantKVProdReference(values, codec, &scratch)
}

func encodeTurboQuantKVProdReference(values []float32, codec TurboQuantKVCodec, scratch *turboQuantKVReferenceEncodeScratch) (TurboQuantKVProdReferenceVector, error) {
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
	mseCodec.ResidualNormPolicy = ""
	if scratch == nil {
		scratch = &turboQuantKVReferenceEncodeScratch{}
	}
	base, err := encodeTurboQuantKVMSEReference(values, mseCodec, scratch)
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
	scratch.ensureProd(len(values))
	residual := scratch.residual
	rotatedBase := scratch.rotated
	for idx, code := range base.CentroidCodes {
		rotatedBase[idx] = turboQuantKVReferenceDequantizeUniform(code, base.Codec.bitsForChannel(int32(idx)))
	}
	normalised := scratch.normalised
	turboQuantKVReferenceRotate(normalised, rotatedBase, base.Codec.RotationSeed, true)
	var residualNormSq float64
	baseNorm := float64(base.Norm)
	for idx := range values {
		decoded := float32(normalised[idx] * baseNorm)
		delta := (float64(values[idx]) - float64(decoded)) / baseNorm
		residual[idx] = delta
		residualNormSq += delta * delta
	}
	residualNorm := math.Sqrt(residualNormSq)
	encoded.ResidualNorm = float32(residualNorm)
	if residualNorm == 0 {
		return encoded, nil
	}
	turboQuantKVReferenceRotate(residual, residual, codec.QJLSeed, false)
	for idx, value := range residual {
		if value < 0 {
			encoded.QJLSigns[idx] = 1
		}
	}
	return encoded, nil
}

func EncodeTurboQuantKVReferencePage(keys, values []float32, layout TurboQuantKVPageLayout) (TurboQuantKVReferencePage, error) {
	if err := layout.Validate(); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	pageElements := int(layout.PageElementCount())
	if len(keys) != pageElements || len(values) != pageElements {
		return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference page payload shape is invalid")
	}
	headDim := int(layout.Shape.HeadDim)
	pageVectors := int(layout.PageVectorCount())
	page := TurboQuantKVReferencePage{
		Layout: layout,
		Keys:   make([]TurboQuantKVProdReferenceVector, pageVectors),
		Values: make([]TurboQuantKVMSEReferenceVector, pageVectors),
	}
	var scratch turboQuantKVReferenceEncodeScratch
	for idx := 0; idx < pageVectors; idx++ {
		start := idx * headDim
		end := start + headDim
		key, err := encodeTurboQuantKVProdReference(keys[start:end], layout.Key, &scratch)
		if err != nil {
			return TurboQuantKVReferencePage{}, core.E("mlx: TurboQuant reference page", "encode key", err)
		}
		value, err := encodeTurboQuantKVMSEReference(values[start:end], layout.Value, &scratch)
		if err != nil {
			return TurboQuantKVReferencePage{}, core.E("mlx: TurboQuant reference page", "encode value", err)
		}
		page.Keys[idx] = key
		page.Values[idx] = value
	}
	return page, nil
}

func (encoded TurboQuantKVMSEReferenceVector) DecodeMSE() ([]float32, error) {
	if err := encoded.validateDecodeMSEReference(); err != nil {
		return nil, err
	}
	decoded := make([]float32, encoded.HeadDim)
	var scratch turboQuantKVReferenceDecodeScratch
	encoded.decodeValidMSEInto(decoded, &scratch)
	return decoded, nil
}

func (encoded TurboQuantKVMSEReferenceVector) decodeMSEInto(dst []float32, scratch *turboQuantKVReferenceDecodeScratch) error {
	if len(dst) != int(encoded.HeadDim) {
		return core.NewError("mlx: TurboQuant MSE reference decode destination shape is invalid")
	}
	if err := encoded.validateDecodeMSEReference(); err != nil {
		return err
	}
	encoded.decodeValidMSEInto(dst, scratch)
	return nil
}

func (encoded TurboQuantKVMSEReferenceVector) decodeValidMSEInto(dst []float32, scratch *turboQuantKVReferenceDecodeScratch) {
	if encoded.Norm == 0 {
		clear(dst)
		return
	}
	if scratch == nil {
		scratch = &turboQuantKVReferenceDecodeScratch{}
	}
	scratch.ensure(len(dst))
	rotated := scratch.rotated
	for idx, code := range encoded.CentroidCodes {
		rotated[idx] = turboQuantKVReferenceDequantizeUniform(code, encoded.Codec.bitsForChannel(int32(idx)))
	}
	normalised := scratch.normalised
	turboQuantKVReferenceRotate(normalised, rotated, encoded.Codec.RotationSeed, true)
	for idx, value := range normalised {
		dst[idx] = float32(value * float64(encoded.Norm))
	}
}

func (encoded TurboQuantKVMSEReferenceVector) PackedCentroidBytes() ([]byte, error) {
	if err := encoded.validatePackedMSEReference(); err != nil {
		return nil, err
	}
	return turboQuantKVReferencePackCodecCentroids(encoded.CentroidCodes, encoded.Codec, encoded.HeadDim), nil
}

func DecodeTurboQuantKVMSEReferenceFromPacked(codec TurboQuantKVCodec, headDim int32, norm float32, packedCentroids []byte) (TurboQuantKVMSEReferenceVector, error) {
	if codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE packed centroid decode requires TurboQuantmse codec")
	}
	if err := codec.Validate("packed centroid reference", headDim); err != nil {
		return TurboQuantKVMSEReferenceVector{}, err
	}
	if codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE packed centroid codebook is unsupported")
	}
	if codec.NormalBits > 8 {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE packed centroid bit width exceeds byte storage")
	}
	if !turboQuantKVReferenceHeadDimSupported(int(headDim)) {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE packed centroid requires a power-of-two head dimension")
	}
	wantBytes := int(turboQuantKVPackedBytes(codec.centroidBitsPerVector(headDim)))
	if len(packedCentroids) != wantBytes {
		return TurboQuantKVMSEReferenceVector{}, core.NewError("mlx: TurboQuant MSE packed centroid byte length is invalid")
	}
	return TurboQuantKVMSEReferenceVector{
		Codec:         codec,
		HeadDim:       headDim,
		Norm:          norm,
		CentroidCodes: turboQuantKVReferenceUnpackCodecCentroids(packedCentroids, int(headDim), codec),
	}, nil
}

func (encoded TurboQuantKVProdReferenceVector) EstimateInnerProduct(query []float32) (float32, error) {
	var scratch turboQuantKVReferenceEstimateScratch
	return encoded.estimateInnerProduct(query, &scratch)
}

func (encoded TurboQuantKVProdReferenceVector) estimateInnerProduct(query []float32, scratch *turboQuantKVReferenceEstimateScratch) (float32, error) {
	if len(query) != int(encoded.Base.HeadDim) {
		return 0, core.NewError("mlx: TurboQuantprod reference query shape is invalid")
	}
	if len(encoded.QJLSigns) != len(query) {
		return 0, core.NewError("mlx: TurboQuantprod reference QJL signs are invalid")
	}
	if err := encoded.Base.validateDecodeMSEReference(); err != nil {
		return 0, err
	}
	if scratch == nil {
		scratch = &turboQuantKVReferenceEstimateScratch{}
	}
	scratch.ensure(len(query))
	baseNormalised := scratch.baseNormalised
	for idx, code := range encoded.Base.CentroidCodes {
		baseNormalised[idx] = turboQuantKVReferenceDequantizeUniform(code, encoded.Base.Codec.bitsForChannel(int32(idx)))
	}
	turboQuantKVReferenceRotate(baseNormalised, baseNormalised, encoded.Base.Codec.RotationSeed, true)
	var estimate float32
	baseNorm := float64(encoded.Base.Norm)
	for idx, value := range baseNormalised {
		estimate += query[idx] * float32(value*baseNorm)
	}
	if encoded.Base.Norm == 0 || encoded.ResidualNorm == 0 {
		return estimate, nil
	}
	rotatedQuery := scratch.rotatedQuery
	for idx, value := range query {
		rotatedQuery[idx] = float64(value)
	}
	turboQuantKVReferenceRotate(rotatedQuery, rotatedQuery, encoded.Codec.QJLSeed, false)
	scale := baseNorm * float64(encoded.ResidualNorm) / math.Sqrt(float64(len(query)))
	for idx, value := range rotatedQuery {
		sign := 1.0
		if encoded.QJLSigns[idx] != 0 {
			sign = -1
		}
		estimate += float32(scale * sign * value)
	}
	return estimate, nil
}

func (encoded TurboQuantKVProdReferenceVector) PackedQJLSignBytes() ([]byte, error) {
	if err := encoded.validatePackedProdReference(); err != nil {
		return nil, err
	}
	return turboQuantKVReferencePackBits(encoded.QJLSigns, 1), nil
}

func DecodeTurboQuantKVProdReferenceFromPacked(codec TurboQuantKVCodec, base TurboQuantKVMSEReferenceVector, residualNorm float32, packedQJLSigns []byte) (TurboQuantKVProdReferenceVector, error) {
	if codec.Algorithm != TurboQuantKVAlgorithmProd {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod packed QJL decode requires TurboQuantprod codec")
	}
	if err := codec.Validate("packed QJL reference", base.HeadDim); err != nil {
		return TurboQuantKVProdReferenceVector{}, err
	}
	if base.Codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod packed QJL base requires TurboQuantmse codec")
	}
	if err := base.validatePackedMSEReference(); err != nil {
		return TurboQuantKVProdReferenceVector{}, err
	}
	wantBytes := int(turboQuantKVPackedBytes(uint64(base.HeadDim)))
	if len(packedQJLSigns) != wantBytes {
		return TurboQuantKVProdReferenceVector{}, core.NewError("mlx: TurboQuantprod packed QJL sign byte length is invalid")
	}
	return TurboQuantKVProdReferenceVector{
		Codec:        codec,
		Base:         base,
		ResidualNorm: residualNorm,
		QJLSigns:     turboQuantKVReferenceUnpackBits(packedQJLSigns, int(base.HeadDim), 1),
	}, nil
}

func (page TurboQuantKVReferencePage) DecodeBase() ([]float32, []float32, error) {
	if err := page.validateReferencePage(); err != nil {
		return nil, nil, err
	}
	pageElements := int(page.Layout.PageElementCount())
	headDim := int(page.Layout.Shape.HeadDim)
	keys := make([]float32, pageElements)
	values := make([]float32, pageElements)
	var scratch turboQuantKVReferenceDecodeScratch
	for idx := range page.Keys {
		start := idx * headDim
		end := start + headDim
		if err := page.Keys[idx].Base.decodeMSEInto(keys[start:end], &scratch); err != nil {
			return nil, nil, core.E("mlx: TurboQuant reference page", "decode key", err)
		}
		if err := page.Values[idx].decodeMSEInto(values[start:end], &scratch); err != nil {
			return nil, nil, core.E("mlx: TurboQuant reference page", "decode value", err)
		}
	}
	return keys, values, nil
}

func (page TurboQuantKVReferencePage) EstimateKeyInnerProducts(query []float32) ([]float32, error) {
	if err := page.validateReferencePage(); err != nil {
		return nil, err
	}
	if len(query) != int(page.Layout.Shape.HeadDim) {
		return nil, core.NewError("mlx: TurboQuant reference page query shape is invalid")
	}
	estimates := make([]float32, len(page.Keys))
	var scratch turboQuantKVReferenceEstimateScratch
	for idx := range page.Keys {
		estimate, err := page.Keys[idx].estimateInnerProduct(query, &scratch)
		if err != nil {
			return nil, core.E("mlx: TurboQuant reference page", "estimate key", err)
		}
		estimates[idx] = estimate
	}
	return estimates, nil
}

func (page TurboQuantKVReferencePage) validateReferencePage() error {
	if err := page.Layout.Validate(); err != nil {
		return err
	}
	pageVectors := int(page.Layout.PageVectorCount())
	if len(page.Keys) != pageVectors || len(page.Values) != pageVectors {
		return core.NewError("mlx: TurboQuant reference page vector count is invalid")
	}
	return nil
}

func (encoded TurboQuantKVMSEReferenceVector) validateDecodeMSEReference() error {
	if encoded.HeadDim <= 0 || len(encoded.CentroidCodes) != int(encoded.HeadDim) {
		return core.NewError("mlx: TurboQuant MSE reference vector shape is invalid")
	}
	if encoded.Codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return core.NewError("mlx: TurboQuant MSE reference decode requires TurboQuantmse codec")
	}
	if encoded.Codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return core.NewError("mlx: TurboQuant MSE reference codebook is unsupported")
	}
	if encoded.Codec.NormalBits > 8 {
		return core.NewError("mlx: TurboQuant MSE reference stores one byte per centroid code")
	}
	if !turboQuantKVReferenceHeadDimSupported(int(encoded.HeadDim)) {
		return core.NewError("mlx: TurboQuant MSE reference requires a power-of-two head dimension")
	}
	return nil
}

func (encoded TurboQuantKVMSEReferenceVector) validatePackedMSEReference() error {
	if encoded.HeadDim <= 0 || len(encoded.CentroidCodes) != int(encoded.HeadDim) {
		return core.NewError("mlx: TurboQuant MSE packed centroid shape is invalid")
	}
	if encoded.Codec.Algorithm != TurboQuantKVAlgorithmMSE {
		return core.NewError("mlx: TurboQuant MSE packed centroid requires TurboQuantmse codec")
	}
	if encoded.Codec.CodebookID != TurboQuantKVReferenceCodebookUniform {
		return core.NewError("mlx: TurboQuant MSE packed centroid codebook is unsupported")
	}
	if encoded.Codec.NormalBits <= 0 || encoded.Codec.NormalBits > 8 {
		return core.NewError("mlx: TurboQuant MSE packed centroid bit width is invalid")
	}
	if !turboQuantKVReferenceHeadDimSupported(int(encoded.HeadDim)) {
		return core.NewError("mlx: TurboQuant MSE packed centroid requires a power-of-two head dimension")
	}
	return nil
}

func (encoded TurboQuantKVProdReferenceVector) validatePackedProdReference() error {
	if encoded.Codec.Algorithm != TurboQuantKVAlgorithmProd {
		return core.NewError("mlx: TurboQuantprod packed QJL requires TurboQuantprod codec")
	}
	if err := encoded.Codec.Validate("packed QJL reference", encoded.Base.HeadDim); err != nil {
		return err
	}
	if err := encoded.Base.validatePackedMSEReference(); err != nil {
		return err
	}
	if len(encoded.QJLSigns) != int(encoded.Base.HeadDim) {
		return core.NewError("mlx: TurboQuantprod packed QJL sign shape is invalid")
	}
	return nil
}

func turboQuantKVReferencePackBits(values []byte, bits int) []byte {
	if bits <= 0 {
		return nil
	}
	return turboQuantKVReferenceAppendPackedBits(nil, values, bits)
}

func turboQuantKVReferenceAppendPackedBits(dst []byte, values []byte, bits int) []byte {
	if bits <= 0 {
		return dst
	}
	bytes := int(turboQuantKVPackedBytes(uint64(len(values)) * uint64(bits)))
	dst, packed := turboQuantKVReferenceAppendZeroedBytes(dst, bytes)
	var mask uint16
	if bits >= 8 {
		mask = 0xff
	} else {
		mask = uint16((1 << uint(bits)) - 1)
	}
	bitOffset := 0
	for _, raw := range values {
		value := uint16(raw) & mask
		for bit := 0; bit < bits; bit++ {
			if value&(1<<uint(bit)) != 0 {
				packed[bitOffset/8] |= 1 << uint(bitOffset%8)
			}
			bitOffset++
		}
	}
	return dst
}

func turboQuantKVReferencePackCodecCentroids(values []byte, codec TurboQuantKVCodec, headDim int32) []byte {
	if len(values) == 0 || headDim <= 0 {
		return nil
	}
	return turboQuantKVReferenceAppendPackedCodecCentroids(nil, values, codec, headDim)
}

func turboQuantKVReferenceAppendPackedCodecCentroids(dst []byte, values []byte, codec TurboQuantKVCodec, headDim int32) []byte {
	if len(values) == 0 || headDim <= 0 {
		return dst
	}
	bytes := int(turboQuantKVPackedBytes(codec.centroidBitsPerVector(headDim)))
	dst, packed := turboQuantKVReferenceAppendZeroedBytes(dst, bytes)
	bitOffset := 0
	for idx, raw := range values {
		bits := codec.bitsForChannel(int32(idx))
		var mask uint16
		if bits >= 8 {
			mask = 0xff
		} else {
			mask = uint16((1 << uint(bits)) - 1)
		}
		value := uint16(raw) & mask
		for bit := 0; bit < bits; bit++ {
			if value&(1<<uint(bit)) != 0 {
				packed[bitOffset/8] |= 1 << uint(bitOffset%8)
			}
			bitOffset++
		}
	}
	return dst
}

func turboQuantKVReferenceAppendZeroedBytes(dst []byte, n int) ([]byte, []byte) {
	if n <= 0 {
		return dst, nil
	}
	start := len(dst)
	if cap(dst)-start >= n {
		dst = dst[:start+n]
		clear(dst[start:])
		return dst, dst[start:]
	}
	dst = append(dst, make([]byte, n)...)
	return dst, dst[start:]
}

func turboQuantKVReferenceUnpackBits(packed []byte, count, bits int) []byte {
	if bits <= 0 || count <= 0 {
		return nil
	}
	values := make([]byte, count)
	bitOffset := 0
	for idx := range values {
		var value byte
		for bit := 0; bit < bits; bit++ {
			if packed[bitOffset/8]&(1<<uint(bitOffset%8)) != 0 {
				value |= 1 << uint(bit)
			}
			bitOffset++
		}
		values[idx] = value
	}
	return values
}

func turboQuantKVReferenceUnpackCodecCentroids(packed []byte, count int, codec TurboQuantKVCodec) []byte {
	if count <= 0 {
		return nil
	}
	values := make([]byte, count)
	bitOffset := 0
	for idx := range values {
		bits := codec.bitsForChannel(int32(idx))
		var value byte
		for bit := 0; bit < bits; bit++ {
			if packed[bitOffset/8]&(1<<uint(bitOffset%8)) != 0 {
				value |= 1 << uint(bit)
			}
			bitOffset++
		}
		values[idx] = value
	}
	return values
}

func turboQuantKVReferenceDecodePackedMSEInto(dst []float32, packed []byte, codec TurboQuantKVCodec, norm float32, rotated, normalised []float64) {
	if norm == 0 {
		clear(dst)
		return
	}
	bitOffset := 0
	for idx := range dst {
		bits := codec.bitsForChannel(int32(idx))
		var code byte
		for bit := 0; bit < bits; bit++ {
			if packed[bitOffset/8]&(1<<uint(bitOffset%8)) != 0 {
				code |= 1 << uint(bit)
			}
			bitOffset++
		}
		rotated[idx] = turboQuantKVReferenceDequantizeUniform(code, bits)
	}
	turboQuantKVReferenceRotate(normalised, rotated, codec.RotationSeed, true)
	for idx, value := range normalised {
		dst[idx] = float32(value * float64(norm))
	}
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
