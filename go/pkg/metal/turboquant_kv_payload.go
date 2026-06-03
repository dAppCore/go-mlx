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
	keyCentroidBytes := 0
	keyQJLBytes := 0
	for _, key := range page.Keys {
		if err := key.validatePackedProdReference(); err != nil {
			return TurboQuantKVReferencePagePayload{}, core.E("mlx: TurboQuant reference payload", "pack key", err)
		}
		keyCentroidBytes += int(turboQuantKVPackedBytes(key.Base.Codec.centroidBitsPerVector(key.Base.HeadDim)))
		keyQJLBytes += int(turboQuantKVPackedBytes(uint64(key.Base.HeadDim)))
	}
	valueCentroidBytes := 0
	for _, value := range page.Values {
		if err := value.validatePackedMSEReference(); err != nil {
			return TurboQuantKVReferencePagePayload{}, core.E("mlx: TurboQuant reference payload", "pack value centroid", err)
		}
		valueCentroidBytes += int(turboQuantKVPackedBytes(value.Codec.centroidBitsPerVector(value.HeadDim)))
	}
	outlierMasks := turboQuantKVReferencePackedOutlierMasks(page.Layout)
	sectionCount := 6
	if len(outlierMasks) > 0 {
		sectionCount++
	}
	dataCapacity := 0
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, keyCentroidBytes)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, keyQJLBytes)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, len(page.Keys)*2)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, len(page.Keys)*2)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, valueCentroidBytes)
	dataCapacity = turboQuantKVReferencePayloadCapacityAfterBytes(dataCapacity, len(page.Values)*2)
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
	keyCentroids := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadKeyCentroids, keyCentroidBytes)
	keyQJLSigns := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadKeyQJLSigns, keyQJLBytes)
	keyNorms := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadKeyNorms, len(page.Keys)*2)
	keyResidualNorms := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadKeyResidualNorms, len(page.Keys)*2)
	valueCentroids := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadValueCentroids, valueCentroidBytes)
	valueNorms := turboQuantKVReferenceAppendPayloadSectionBytes(&payload, TurboQuantKVReferencePayloadValueNorms, len(page.Values)*2)
	for _, key := range page.Keys {
		keyCentroids = turboQuantKVReferenceAppendPackedCodecCentroids(keyCentroids, key.Base.CentroidCodes, key.Base.Codec, key.Base.HeadDim)
		keyQJLSigns = turboQuantKVReferenceAppendPackedBits(keyQJLSigns, key.QJLSigns, 1)
		keyNorms = turboQuantKVReferenceAppendBF16Norm(keyNorms, key.Base.Norm)
		keyResidualNorms = turboQuantKVReferenceAppendBF16Norm(keyResidualNorms, key.ResidualNorm)
	}
	for _, value := range page.Values {
		valueCentroids = turboQuantKVReferenceAppendPackedCodecCentroids(valueCentroids, value.CentroidCodes, value.Codec, value.HeadDim)
		valueNorms = turboQuantKVReferenceAppendBF16Norm(valueNorms, value.Norm)
	}
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
	keyCentroidCodes := make([]byte, pageVectors*headDim)
	keyQJLSignCodes := make([]byte, pageVectors*headDim)
	valueCentroidCodes := make([]byte, pageVectors*headDim)
	for idx := range pageVectors {
		codeStart := idx * headDim
		codeEnd := codeStart + headDim
		keyCodes := keyCentroidCodes[codeStart:codeEnd]
		keySigns := keyQJLSignCodes[codeStart:codeEnd]
		valueCodes := valueCentroidCodes[codeStart:codeEnd]
		if !turboQuantKVReferenceUnpackCodecCentroidsInto(
			keyCodes,
			keyCentroids[idx*keyCentroidBytes:(idx+1)*keyCentroidBytes],
			keyMSECodec,
		) {
			return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference payload key centroid bits are invalid")
		}
		if !turboQuantKVReferenceUnpackBitsInto(keySigns, keyQJLSigns[idx*keyQJLBytes:(idx+1)*keyQJLBytes], 1) {
			return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference payload key QJL bits are invalid")
		}
		if !turboQuantKVReferenceUnpackCodecCentroidsInto(
			valueCodes,
			valueCentroids[idx*valueCentroidBytes:(idx+1)*valueCentroidBytes],
			layout.Value,
		) {
			return TurboQuantKVReferencePage{}, core.NewError("mlx: TurboQuant reference payload value centroid bits are invalid")
		}
		page.Keys[idx] = TurboQuantKVProdReferenceVector{
			Codec: layout.Key,
			Base: TurboQuantKVMSEReferenceVector{
				Codec:         keyMSECodec,
				HeadDim:       layout.Shape.HeadDim,
				Norm:          turboQuantKVReferenceReadBF16Norm(keyNorms[idx*2:]),
				CentroidCodes: keyCodes,
			},
			ResidualNorm: turboQuantKVReferenceReadBF16Norm(keyResidualNorms[idx*2:]),
			QJLSigns:     keySigns,
		}
		page.Values[idx] = TurboQuantKVMSEReferenceVector{
			Codec:         layout.Value,
			HeadDim:       layout.Shape.HeadDim,
			Norm:          turboQuantKVReferenceReadBF16Norm(valueNorms[idx*2:]),
			CentroidCodes: valueCodes,
		}
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
	arrayShape := [4]int{int(shape.Batch), int(shape.Heads), int(payload.Layout.PageTokens), int(shape.HeadDim)}
	keyArray, keyErr := fromPinnedFloat32Values(decodedKeys, arrayShape[:])
	valueArray, valueErr := fromPinnedFloat32Values(decodedValues, arrayShape[:])
	if keyErr != nil || valueErr != nil {
		Free(keyArray, valueArray)
		if keyErr != nil {
			return nil, nil, keyErr
		}
		return nil, nil, valueErr
	}
	return keyArray, valueArray, nil
}

func (payload TurboQuantKVReferencePagePayload) DecodeBaseFloatData() ([]float32, []float32, error) {
	if err := payload.Layout.Validate(); err != nil {
		return nil, nil, err
	}
	pageElements := int(payload.Layout.PageElementCount())
	keys := make([]float32, pageElements)
	values := make([]float32, pageElements)
	if err := payload.DecodeBaseFloatDataInto(keys, values); err != nil {
		return nil, nil, err
	}
	return keys, values, nil
}

// DecodeBaseFloatDataInto restores the page into caller-owned K/V buffers.
func (payload TurboQuantKVReferencePagePayload) DecodeBaseFloatDataInto(keys, values []float32) error {
	if err := payload.Layout.Validate(); err != nil {
		return err
	}
	pageElements := int(payload.Layout.PageElementCount())
	if len(keys) != pageElements || len(values) != pageElements {
		return core.NewError("mlx: TurboQuant reference payload destination shape is invalid")
	}
	headDim := int(payload.Layout.Shape.HeadDim)
	scratch := borrowTurboQuantKVReferenceDecodeScratch(headDim)
	defer releaseTurboQuantKVReferenceDecodeScratch(scratch)
	if err := payload.decodeBaseFloatDataInto(keys, values, payload.Layout.PageTokens, 0, scratch.rotated, scratch.normalised); err != nil {
		return err
	}
	return nil
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
	for idx := range pageVectors {
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

func turboQuantKVReferenceAppendPayloadSection(payload *TurboQuantKVReferencePagePayload, name string, data []byte) {
	section := turboQuantKVReferenceAppendPayloadSectionBytes(payload, name, len(data))
	copy(section, data)
}

func turboQuantKVReferenceAppendPayloadSectionBytes(payload *TurboQuantKVReferencePagePayload, name string, byteCount int) []byte {
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
		Bytes:     uint64(byteCount),
		Alignment: payload.Alignment,
	})
	oldLen := len(payload.Data)
	if cap(payload.Data)-oldLen >= byteCount {
		payload.Data = payload.Data[:oldLen+byteCount]
		clear(payload.Data[oldLen:])
	} else {
		payload.Data = append(payload.Data, make([]byte, byteCount)...)
	}
	return payload.Data[oldLen : oldLen : oldLen+byteCount]
}

func turboQuantKVReferencePayloadCapacityAfter(offset int, data []byte) int {
	return turboQuantKVReferencePayloadCapacityAfterBytes(offset, len(data))
}

func turboQuantKVReferencePayloadCapacityAfterBytes(offset, byteCount int) int {
	aligned := int(turboQuantKVReferenceAlignOffset(uint64(offset), TurboQuantKVReferencePayloadAlignment))
	return aligned + byteCount
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
