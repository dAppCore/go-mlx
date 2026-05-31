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

func turboQuantKVReferenceBenchMSECodec() TurboQuantKVCodec {
	return TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
}

func turboQuantKVReferenceBenchProdCodec() TurboQuantKVCodec {
	return TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmProd,
		NormalBits:   4,
		RotationSeed: 0x6b,
		QJLSeed:      0x7c,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
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
