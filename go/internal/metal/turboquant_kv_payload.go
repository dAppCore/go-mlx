// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	TurboQuantKVReferencePayloadAlignment    uint64 = 64
	TurboQuantKVReferencePayloadEndianLittle        = "little"

	TurboQuantKVReferencePayloadKeyCentroids      = "key_centroids"
	TurboQuantKVReferencePayloadKeyQJLSigns       = "key_qjl_signs"
	TurboQuantKVReferencePayloadKeyNorms          = "key_norms_bf16"
	TurboQuantKVReferencePayloadKeyResidualNorms  = "key_residual_norms_bf16"
	TurboQuantKVReferencePayloadValueCentroids    = "value_centroids"
	TurboQuantKVReferencePayloadValueNorms        = "value_norms_bf16"
	TurboQuantKVReferencePayloadOutlierMaskHeader = "outlier_masks"
)

type TurboQuantKVReferencePagePayloadSection struct {
	Name      string `json:"name"`
	Offset    uint64 `json:"offset"`
	Bytes     uint64 `json:"bytes"`
	Alignment uint64 `json:"alignment"`
}

type TurboQuantKVReferencePagePayload struct {
	Layout    TurboQuantKVPageLayout                    `json:"layout"`
	Endian    string                                    `json:"endian"`
	Alignment uint64                                    `json:"alignment"`
	Sections  []TurboQuantKVReferencePagePayloadSection `json:"sections"`
	Data      []byte                                    `json:"data"`
}

func (page TurboQuantKVReferencePage) PackedPayload() (TurboQuantKVReferencePagePayload, error) {
	if err := page.validateReferencePage(); err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	keyCentroids, keyQJLSigns, keyNorms, keyResidualNorms, err := turboQuantKVReferencePackedKeySections(page.Keys)
	if err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	valueCentroids, valueNorms, err := turboQuantKVReferencePackedValueSections(page.Values)
	if err != nil {
		return TurboQuantKVReferencePagePayload{}, err
	}
	outlierMasks := turboQuantKVReferencePackedOutlierMasks(page.Layout)
	sectionCount := 6
	if len(outlierMasks) > 0 {
		sectionCount++
	}
	dataCapacity := 0
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, keyCentroids)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, keyQJLSigns)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, keyNorms)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, keyResidualNorms)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, valueCentroids)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, valueNorms)
	if len(outlierMasks) > 0 {
		dataCapacity = turboQuantKVReferencePayloadCapacityAfter(dataCapacity, outlierMasks)
	}
	payload := TurboQuantKVReferencePagePayload{
		Layout:    page.Layout,
		Endian:    TurboQuantKVReferencePayloadEndianLittle,
		Alignment: TurboQuantKVReferencePayloadAlignment,
		Sections:  make([]TurboQuantKVReferencePagePayloadSection, 0, sectionCount),
		Data:      make([]byte, 0, dataCapacity),
	}
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadKeyCentroids, keyCentroids)
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadKeyQJLSigns, keyQJLSigns)
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadKeyNorms, keyNorms)
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadKeyResidualNorms, keyResidualNorms)
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadValueCentroids, valueCentroids)
	turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadValueNorms, valueNorms)
	if len(outlierMasks) > 0 {
		turboQuantKVReferenceAppendPayloadSection(&payload, TurboQuantKVReferencePayloadOutlierMaskHeader, outlierMasks)
	}
	return payload, nil
}

func DecodeTurboQuantKVReferencePagePayload(payload TurboQuantKVReferencePagePayload) (TurboQuantKVReferencePage, error) {
	if payload.Endian != TurboQuantKVReferencePayloadEndianLittle {
		return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference payload endian marker is invalid")
	}
	if payload.Alignment != TurboQuantKVReferencePayloadAlignment {
		return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference payload alignment is invalid")
	}
	if err := payload.Layout.Validate(); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := payload.validateSections(); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	layout := payload.Layout
	pageVectors := int(layout.PageVectorCount())
	headDim := int(layout.Shape.HeadDim)
	keyCentroids, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyCentroids)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	keyQJLSigns, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyQJLSigns)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	keyNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyNorms)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	keyResidualNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyResidualNorms)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	valueCentroids, err := payload.requiredSection(TurboQuantKVReferencePayloadValueCentroids)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	valueNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadValueNorms)
	if err != nil {
		return TurboQuantKVReferencePage{}, err
	}

	keyCentroidBytes := int(turboQuantKVPackedBytes(layout.Key.centroidBitsPerVector(layout.Shape.HeadDim)))
	keyQJLBytes := int(turboQuantKVPackedBytes(uint64(headDim)))
	valueCentroidBytes := int(turboQuantKVPackedBytes(layout.Value.centroidBitsPerVector(layout.Shape.HeadDim)))
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyCentroids, len(keyCentroids), pageVectors*keyCentroidBytes); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyQJLSigns, len(keyQJLSigns), pageVectors*keyQJLBytes); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyNorms, len(keyNorms), pageVectors*2); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyResidualNorms, len(keyResidualNorms), pageVectors*2); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadValueCentroids, len(valueCentroids), pageVectors*valueCentroidBytes); err != nil {
		return TurboQuantKVReferencePage{}, err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadValueNorms, len(valueNorms), pageVectors*2); err != nil {
		return TurboQuantKVReferencePage{}, err
	}

	keyMSECodec := layout.Key
	keyMSECodec.Algorithm = TurboQuantKVAlgorithmMSE
	keyMSECodec.QJLSeed = 0
	keyMSECodec.ResidualNormPolicy = ""
	page := TurboQuantKVReferencePage{
		Layout: layout,
		Keys:   make([]TurboQuantKVProdReferenceVector, pageVectors),
		Values: make([]TurboQuantKVMSEReferenceVector, pageVectors),
	}
	for idx := 0; idx < pageVectors; idx++ {
		keyBase, err := DecodeTurboQuantKVMSEReferenceFromPacked(
			keyMSECodec,
			layout.Shape.HeadDim,
			turboQuantKVReferenceReadBF16Norm(keyNorms[idx*2:]),
			keyCentroids[idx*keyCentroidBytes:(idx+1)*keyCentroidBytes],
		)
		if err != nil {
			return TurboQuantKVReferencePage{}, core.E("mlx: TurboQuant reference payload", "decode key centroid", err)
		}
		key, err := DecodeTurboQuantKVProdReferenceFromPacked(
			layout.Key,
			keyBase,
			turboQuantKVReferenceReadBF16Norm(keyResidualNorms[idx*2:]),
			keyQJLSigns[idx*keyQJLBytes:(idx+1)*keyQJLBytes],
		)
		if err != nil {
			return TurboQuantKVReferencePage{}, core.E("mlx: TurboQuant reference payload", "decode key QJL", err)
		}
		value, err := DecodeTurboQuantKVMSEReferenceFromPacked(
			layout.Value,
			layout.Shape.HeadDim,
			turboQuantKVReferenceReadBF16Norm(valueNorms[idx*2:]),
			valueCentroids[idx*valueCentroidBytes:(idx+1)*valueCentroidBytes],
		)
		if err != nil {
			return TurboQuantKVReferencePage{}, core.E("mlx: TurboQuant reference payload", "decode value centroid", err)
		}
		page.Keys[idx] = key
		page.Values[idx] = value
	}
	return page, nil
}

// DecodeBaseArrays restores the packed reference payload into MLX arrays shaped
// [batch, heads, page_tokens, head_dim].
func (payload TurboQuantKVReferencePagePayload) DecodeBaseArrays() (*Array, *Array, error) {
	decodedKeys, decodedValues, err := payload.DecodeBaseFloatData()
	if err != nil {
		return nil, nil, err
	}
	shape := payload.Layout.Shape
	keyArray := FromValues(decodedKeys,
		int(shape.Batch),
		int(shape.Heads),
		int(payload.Layout.PageTokens),
		int(shape.HeadDim),
	)
	valueArray := FromValues(decodedValues,
		int(shape.Batch),
		int(shape.Heads),
		int(payload.Layout.PageTokens),
		int(shape.HeadDim),
	)
	return keyArray, valueArray, nil
}

func (payload TurboQuantKVReferencePagePayload) DecodeBaseFloatData() ([]float32, []float32, error) {
	if err := payload.Layout.Validate(); err != nil {
		return nil, nil, err
	}
	pageElements := int(payload.Layout.PageElementCount())
	headDim := int(payload.Layout.Shape.HeadDim)
	keys := make([]float32, pageElements)
	values := make([]float32, pageElements)
	rotated := make([]float64, headDim)
	normalised := make([]float64, headDim)
	if err := payload.decodeBaseFloatDataInto(keys, values, payload.Layout.PageTokens, 0, rotated, normalised); err != nil {
		return nil, nil, err
	}
	return keys, values, nil
}

func (payload TurboQuantKVReferencePagePayload) decodeBaseFloatDataInto(keys, values []float32, totalSeqLen, tokenStart int, rotated, normalised []float64) error {
	if payload.Endian != TurboQuantKVReferencePayloadEndianLittle {
		return core.NewError("mlx: TurboQuant reference payload endian marker is invalid")
	}
	if payload.Alignment != TurboQuantKVReferencePayloadAlignment {
		return core.NewError("mlx: TurboQuant reference payload alignment is invalid")
	}
	if err := payload.Layout.Validate(); err != nil {
		return err
	}
	if err := payload.validateSections(); err != nil {
		return err
	}
	layout := payload.Layout
	pageVectors := int(layout.PageVectorCount())
	headDim := int(layout.Shape.HeadDim)
	pageTokens := layout.PageTokens
	if totalSeqLen <= 0 || tokenStart < 0 || pageTokens <= 0 || tokenStart+pageTokens > totalSeqLen {
		return core.NewError("mlx: TurboQuant reference payload destination sequence range is invalid")
	}
	wantElements := int(layout.Shape.Batch) * int(layout.Shape.Heads) * totalSeqLen * headDim
	if len(keys) < wantElements || len(values) < wantElements {
		return core.NewError("mlx: TurboQuant reference payload destination shape is invalid")
	}
	if len(rotated) < headDim || len(normalised) < headDim {
		return core.NewError("mlx: TurboQuant reference payload decode scratch is invalid")
	}
	rotated = rotated[:headDim]
	normalised = normalised[:headDim]
	keyCentroids, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyCentroids)
	if err != nil {
		return err
	}
	keyQJLSigns, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyQJLSigns)
	if err != nil {
		return err
	}
	keyNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyNorms)
	if err != nil {
		return err
	}
	keyResidualNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadKeyResidualNorms)
	if err != nil {
		return err
	}
	valueCentroids, err := payload.requiredSection(TurboQuantKVReferencePayloadValueCentroids)
	if err != nil {
		return err
	}
	valueNorms, err := payload.requiredSection(TurboQuantKVReferencePayloadValueNorms)
	if err != nil {
		return err
	}

	keyCentroidBytes := int(turboQuantKVPackedBytes(layout.Key.centroidBitsPerVector(layout.Shape.HeadDim)))
	keyQJLBytes := int(turboQuantKVPackedBytes(uint64(headDim)))
	valueCentroidBytes := int(turboQuantKVPackedBytes(layout.Value.centroidBitsPerVector(layout.Shape.HeadDim)))
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyCentroids, len(keyCentroids), pageVectors*keyCentroidBytes); err != nil {
		return err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyQJLSigns, len(keyQJLSigns), pageVectors*keyQJLBytes); err != nil {
		return err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyNorms, len(keyNorms), pageVectors*2); err != nil {
		return err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadKeyResidualNorms, len(keyResidualNorms), pageVectors*2); err != nil {
		return err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadValueCentroids, len(valueCentroids), pageVectors*valueCentroidBytes); err != nil {
		return err
	}
	if err := turboQuantKVReferenceCheckPayloadLength(TurboQuantKVReferencePayloadValueNorms, len(valueNorms), pageVectors*2); err != nil {
		return err
	}

	keyMSECodec := layout.Key
	keyMSECodec.Algorithm = TurboQuantKVAlgorithmMSE
	keyMSECodec.QJLSeed = 0
	keyMSECodec.ResidualNormPolicy = ""
	for idx := 0; idx < pageVectors; idx++ {
		token := idx % pageTokens
		vector := idx / pageTokens
		start := (vector*totalSeqLen + tokenStart + token) * headDim
		end := start + headDim
		turboQuantKVReferenceDecodePackedMSEInto(
			keys[start:end],
			keyCentroids[idx*keyCentroidBytes:(idx+1)*keyCentroidBytes],
			keyMSECodec,
			turboQuantKVReferenceReadBF16Norm(keyNorms[idx*2:]),
			rotated,
			normalised,
		)
		turboQuantKVReferenceDecodePackedMSEInto(
			values[start:end],
			valueCentroids[idx*valueCentroidBytes:(idx+1)*valueCentroidBytes],
			layout.Value,
			turboQuantKVReferenceReadBF16Norm(valueNorms[idx*2:]),
			rotated,
			normalised,
		)
	}
	return nil
}

func (payload TurboQuantKVReferencePagePayload) UnpaddedByteCount() uint64 {
	var total uint64
	for _, section := range payload.Sections {
		total += section.Bytes
	}
	return total
}

func (payload TurboQuantKVReferencePagePayload) SectionBytes(name string) ([]byte, bool) {
	for _, section := range payload.Sections {
		if section.Name != name {
			continue
		}
		end := section.Offset + section.Bytes
		if section.Offset > uint64(len(payload.Data)) || end > uint64(len(payload.Data)) {
			return nil, false
		}
		return payload.Data[section.Offset:end], true
	}
	return nil, false
}

func (payload TurboQuantKVReferencePagePayload) requiredSection(name string) ([]byte, error) {
	data, ok := payload.SectionBytes(name)
	if !ok {
		return nil, core.NewError("mlx: TurboQuant reference payload missing " + name)
	}
	return data, nil
}

func (payload TurboQuantKVReferencePagePayload) validateSections() error {
	for _, section := range payload.Sections {
		if section.Alignment != TurboQuantKVReferencePayloadAlignment || section.Offset%TurboQuantKVReferencePayloadAlignment != 0 {
			return core.NewError("mlx: TurboQuant reference payload section alignment is invalid")
		}
		end := section.Offset + section.Bytes
		if section.Offset > uint64(len(payload.Data)) || end > uint64(len(payload.Data)) {
			return core.NewError("mlx: TurboQuant reference payload section range is invalid")
		}
	}
	return nil
}

func turboQuantKVReferencePackedKeySections(keys []TurboQuantKVProdReferenceVector) ([]byte, []byte, []byte, []byte, error) {
	centroidBytes := 0
	signBytes := 0
	for _, key := range keys {
		if err := key.validatePackedProdReference(); err != nil {
			return nil, nil, nil, nil, core.E("mlx: TurboQuant reference payload", "pack key", err)
		}
		centroidBytes += int(turboQuantKVPackedBytes(key.Base.Codec.centroidBitsPerVector(key.Base.HeadDim)))
		signBytes += int(turboQuantKVPackedBytes(uint64(key.Base.HeadDim)))
	}
	centroids := make([]byte, 0, centroidBytes)
	signs := make([]byte, 0, signBytes)
	norms := make([]byte, 0, len(keys)*2)
	residualNorms := make([]byte, 0, len(keys)*2)
	for _, key := range keys {
		centroids = turboQuantKVReferenceAppendPackedCodecCentroids(centroids, key.Base.CentroidCodes, key.Base.Codec, key.Base.HeadDim)
		signs = turboQuantKVReferenceAppendPackedBits(signs, key.QJLSigns, 1)
		norms = turboQuantKVReferenceAppendBF16Norm(norms, key.Base.Norm)
		residualNorms = turboQuantKVReferenceAppendBF16Norm(residualNorms, key.ResidualNorm)
	}
	return centroids, signs, norms, residualNorms, nil
}

func turboQuantKVReferencePackedValueSections(values []TurboQuantKVMSEReferenceVector) ([]byte, []byte, error) {
	centroidBytes := 0
	for _, value := range values {
		if err := value.validatePackedMSEReference(); err != nil {
			return nil, nil, core.E("mlx: TurboQuant reference payload", "pack value centroid", err)
		}
		centroidBytes += int(turboQuantKVPackedBytes(value.Codec.centroidBitsPerVector(value.HeadDim)))
	}
	centroids := make([]byte, 0, centroidBytes)
	norms := make([]byte, 0, len(values)*2)
	for _, value := range values {
		centroids = turboQuantKVReferenceAppendPackedCodecCentroids(centroids, value.CentroidCodes, value.Codec, value.HeadDim)
		norms = turboQuantKVReferenceAppendBF16Norm(norms, value.Norm)
	}
	return centroids, norms, nil
}

func turboQuantKVReferenceAppendPayloadSection(payload *TurboQuantKVReferencePagePayload, name string, data []byte) {
	offset := turboQuantKVReferenceAlignOffset(uint64(len(payload.Data)), payload.Alignment)
	if pad := int(offset) - len(payload.Data); pad > 0 {
		oldLen := len(payload.Data)
		if cap(payload.Data)-oldLen >= pad {
			payload.Data = payload.Data[:oldLen+pad]
			clear(payload.Data[oldLen:])
		} else {
			payload.Data = append(payload.Data, make([]byte, pad)...)
		}
	}
	payload.Sections = append(payload.Sections, TurboQuantKVReferencePagePayloadSection{
		Name:      name,
		Offset:    offset,
		Bytes:     uint64(len(data)),
		Alignment: payload.Alignment,
	})
	payload.Data = append(payload.Data, data...)
}

func turboQuantKVReferencePayloadCapacityAfter(offset int, data []byte) int {
	aligned := int(turboQuantKVReferenceAlignOffset(uint64(offset), TurboQuantKVReferencePayloadAlignment))
	return aligned + len(data)
}

func turboQuantKVReferenceAlignOffset(offset, alignment uint64) uint64 {
	if alignment == 0 || offset%alignment == 0 {
		return offset
	}
	return offset + alignment - offset%alignment
}

func turboQuantKVReferencePackedOutlierMasks(layout TurboQuantKVPageLayout) []byte {
	if len(layout.Key.OutlierMask) == 0 && len(layout.Value.OutlierMask) == 0 {
		return nil
	}
	out := make([]byte, 0, len(layout.Key.OutlierMask)+len(layout.Value.OutlierMask))
	out = append(out, layout.Key.OutlierMask...)
	out = append(out, layout.Value.OutlierMask...)
	return out
}

func turboQuantKVReferenceAppendBF16Norm(dst []byte, value float32) []byte {
	return binary.LittleEndian.AppendUint16(dst, uint16(math.Float32bits(value)>>16))
}

func turboQuantKVReferenceReadBF16Norm(raw []byte) float32 {
	if len(raw) < 2 {
		return 0
	}
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[:2])) << 16)
}

func turboQuantKVReferenceCheckPayloadLength(name string, got, want int) error {
	if got != want {
		label := core.Replace(name, "_", " ")
		return core.NewError(core.Sprintf("mlx: TurboQuant reference payload %s bytes = %d, want %d", label, got, want))
	}
	return nil
}
