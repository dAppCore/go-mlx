// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestTurboQuantKVReferencePage_PackedPayloadSectionsAligned_Good(t *testing.T) {
	layout := validTurboQuantKVReferencePageLayout()
	keys := turboQuantKVReferencePageValues(layout, 37)
	values := turboQuantKVReferencePageValues(layout, 53)
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v, want nil", err)
	}

	payload, err := page.PackedPayload()
	if err != nil {
		t.Fatalf("PackedPayload() error = %v, want nil", err)
	}
	estimate, err := layout.EstimatePayloadBytes()
	if err != nil {
		t.Fatalf("EstimatePayloadBytes() error = %v, want nil", err)
	}
	if payload.Alignment != TurboQuantKVReferencePayloadAlignment || payload.Endian != TurboQuantKVReferencePayloadEndianLittle {
		t.Fatalf("payload identity = alignment:%d endian:%q, want cache-line little-endian payload", payload.Alignment, payload.Endian)
	}
	if got := payload.UnpaddedByteCount(); got != estimate.TotalBytes {
		t.Fatalf("payload unpadded bytes = %d, want estimate total %d", got, estimate.TotalBytes)
	}
	wantBytes := map[string]uint64{
		TurboQuantKVReferencePayloadKeyCentroids:      estimate.KeyCentroidBytes,
		TurboQuantKVReferencePayloadKeyQJLSigns:       estimate.KeyQJLSignBytes,
		TurboQuantKVReferencePayloadKeyNorms:          estimate.KeyNormBytes,
		TurboQuantKVReferencePayloadKeyResidualNorms:  estimate.KeyResidualNormBytes,
		TurboQuantKVReferencePayloadValueCentroids:    estimate.ValueCentroidBytes,
		TurboQuantKVReferencePayloadValueNorms:        estimate.ValueNormBytes,
		TurboQuantKVReferencePayloadOutlierMaskHeader: estimate.OutlierMaskBytes,
	}
	for _, section := range payload.Sections {
		if section.Offset%TurboQuantKVReferencePayloadAlignment != 0 {
			t.Fatalf("section %s offset = %d, want %d-byte alignment", section.Name, section.Offset, TurboQuantKVReferencePayloadAlignment)
		}
		if section.Alignment != TurboQuantKVReferencePayloadAlignment {
			t.Fatalf("section %s alignment = %d, want %d", section.Name, section.Alignment, TurboQuantKVReferencePayloadAlignment)
		}
		if wantBytes[section.Name] != section.Bytes {
			t.Fatalf("section %s bytes = %d, want %d", section.Name, section.Bytes, wantBytes[section.Name])
		}
	}
}

func TestTurboQuantKVReferencePage_PackedPayloadRoundTrip_Good(t *testing.T) {
	layout := validTurboQuantKVReferencePageLayout()
	keys := turboQuantKVReferencePageValues(layout, 37)
	values := turboQuantKVReferencePageValues(layout, 53)
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v, want nil", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		t.Fatalf("PackedPayload() error = %v, want nil", err)
	}

	restored, err := DecodeTurboQuantKVReferencePagePayload(payload)
	if err != nil {
		t.Fatalf("DecodeTurboQuantKVReferencePagePayload() error = %v, want nil", err)
	}
	decodedKeys, decodedValues, err := restored.DecodeBase()
	if err != nil {
		t.Fatalf("DecodeBase(restored) error = %v, want nil", err)
	}
	if got := cosineSimilarity(keys, decodedKeys); got < 0.99 {
		t.Fatalf("restored key cosine = %.6f, want >= 0.99", got)
	}
	if got := cosineSimilarity(values, decodedValues); got < 0.99 {
		t.Fatalf("restored value cosine = %.6f, want >= 0.99", got)
	}
}

func TestTurboQuantKVReferencePage_RejectsShortPayloadSection_Bad(t *testing.T) {
	layout := validTurboQuantKVReferencePageLayout()
	keys := turboQuantKVReferencePageValues(layout, 37)
	values := turboQuantKVReferencePageValues(layout, 53)
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v, want nil", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		t.Fatalf("PackedPayload() error = %v, want nil", err)
	}
	for idx := range payload.Sections {
		if payload.Sections[idx].Name == TurboQuantKVReferencePayloadKeyCentroids {
			payload.Sections[idx].Bytes--
			break
		}
	}

	_, err = DecodeTurboQuantKVReferencePagePayload(payload)
	if err == nil || !core.Contains(err.Error(), "key centroid") {
		t.Fatalf("DecodeTurboQuantKVReferencePagePayload(short) error = %v, want key centroid diagnostic", err)
	}
}
