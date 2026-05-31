// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkTurboQuantKVMSEReference_Encode_D128(b *testing.B) {
	values := turboQuantKVReferenceBenchVector(128)
	codec := turboQuantKVReferenceBenchMSECodec()
	b.ReportAllocs()
	for b.Loop() {
		encoded, err := EncodeTurboQuantKVMSEReference(values, codec)
		if err != nil {
			b.Fatalf("EncodeTurboQuantKVMSEReference() error = %v", err)
		}
		if encoded.Norm == 0 {
			b.Fatal("encoded norm = 0, want non-zero vector")
		}
	}
}

func BenchmarkTurboQuantKVMSEReference_Decode_D128(b *testing.B) {
	values := turboQuantKVReferenceBenchVector(128)
	encoded, err := EncodeTurboQuantKVMSEReference(values, turboQuantKVReferenceBenchMSECodec())
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVMSEReference() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		decoded, err := encoded.DecodeMSE()
		if err != nil {
			b.Fatalf("DecodeMSE() error = %v", err)
		}
		if len(decoded) != len(values) {
			b.Fatalf("decoded len = %d, want %d", len(decoded), len(values))
		}
	}
}

func BenchmarkTurboQuantKVProdReference_Estimate_D128(b *testing.B) {
	key := turboQuantKVReferenceBenchVector(128)
	query := turboQuantKVReferenceBenchQuery(128)
	encoded, err := EncodeTurboQuantKVProdReference(key, turboQuantKVReferenceBenchProdCodec())
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVProdReference() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		estimated, err := encoded.EstimateInnerProduct(query)
		if err != nil {
			b.Fatalf("EstimateInnerProduct() error = %v", err)
		}
		if estimated == 0 {
			b.Fatal("estimated inner product = 0, want non-zero diagnostic value")
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_Encode_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	b.ReportAllocs()
	for b.Loop() {
		page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
		if err != nil {
			b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
		}
		if len(page.Keys) != int(layout.PageVectorCount()) {
			b.Fatalf("encoded key vectors = %d, want %d", len(page.Keys), layout.PageVectorCount())
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_DecodeBase_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		decodedKeys, decodedValues, err := page.DecodeBase()
		if err != nil {
			b.Fatalf("DecodeBase() error = %v", err)
		}
		if len(decodedKeys) != len(keys) || len(decodedValues) != len(values) {
			b.Fatalf("decoded lengths = %d/%d, want %d/%d", len(decodedKeys), len(decodedValues), len(keys), len(values))
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_EstimateKeys_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	query := turboQuantKVReferenceBenchQuery(int(layout.Shape.HeadDim))
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		estimates, err := page.EstimateKeyInnerProducts(query)
		if err != nil {
			b.Fatalf("EstimateKeyInnerProducts() error = %v", err)
		}
		if len(estimates) != int(layout.PageVectorCount()) {
			b.Fatalf("estimates = %d, want %d", len(estimates), layout.PageVectorCount())
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_PackedPayload_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		payload, err := page.PackedPayload()
		if err != nil {
			b.Fatalf("PackedPayload() error = %v", err)
		}
		if payload.UnpaddedByteCount() == 0 {
			b.Fatal("payload bytes = 0, want packed page payload")
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_DecodePayload_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		b.Fatalf("PackedPayload() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		restored, err := DecodeTurboQuantKVReferencePagePayload(payload)
		if err != nil {
			b.Fatalf("DecodeTurboQuantKVReferencePagePayload() error = %v", err)
		}
		if len(restored.Keys) != int(layout.PageVectorCount()) {
			b.Fatalf("restored keys = %d, want %d", len(restored.Keys), layout.PageVectorCount())
		}
	}
}

func BenchmarkTurboQuantKVReferencePage_DecodePayloadArrays_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		b.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		b.Fatalf("PackedPayload() error = %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		keyArray, valueArray, err := payload.DecodeBaseArrays()
		if err != nil {
			b.Fatalf("DecodeBaseArrays() error = %v", err)
		}
		if keyArray.Dim(3) != int(layout.Shape.HeadDim) || valueArray.Dim(3) != int(layout.Shape.HeadDim) {
			b.Fatalf("restored array head dim = %d/%d, want %d", keyArray.Dim(3), valueArray.Dim(3), layout.Shape.HeadDim)
		}
		Free(keyArray, valueArray)
	}
}

func turboQuantKVReferenceBenchMSECodec() TurboQuantKVCodec {
	return TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		NormPolicy:   TurboQuantKVNormPolicyExplicitVectorBF16V1,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
}

func turboQuantKVReferenceBenchProdCodec() TurboQuantKVCodec {
	return TurboQuantKVCodec{
		Algorithm:          TurboQuantKVAlgorithmProd,
		NormalBits:         4,
		NormPolicy:         TurboQuantKVNormPolicyExplicitVectorBF16V1,
		ResidualNormPolicy: TurboQuantKVResidualNormPolicyExplicitVectorBF16V1,
		RotationSeed:       0x6b,
		QJLSeed:            0x7c,
		CodebookID:         TurboQuantKVReferenceCodebookUniform,
	}
}

func turboQuantKVReferenceBenchPageLayout() TurboQuantKVPageLayout {
	layout := validTurboQuantKVReferencePageLayout()
	layout.Shape = TurboQuantKVShape{Batch: 1, Heads: 2, SeqLen: 4, HeadDim: 128}
	layout.PageTokens = 4
	layout.PageSize = 4
	return layout
}

func turboQuantKVReferenceBenchVector(dim int) []float32 {
	values := make([]float32, dim)
	for idx := range values {
		values[idx] = float32(((idx*37)%101)-50) / 64
	}
	return values
}

func turboQuantKVReferenceBenchQuery(dim int) []float32 {
	values := make([]float32, dim)
	for idx := range values {
		values[idx] = float32(((idx*53)%89)-44) / 57
	}
	return values
}
