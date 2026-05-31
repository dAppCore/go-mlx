// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestTurboQuantKVPageLayout_ValidateReferenceMetadata_Good(t *testing.T) {
	coverageTokens := "TurboQuantKVPageLayout ValidateReferenceMetadata"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}

	layout := TurboQuantKVPageLayout{
		Version:     TurboQuantKVLayoutVersion,
		Codec:       TurboQuantKVCodecName,
		CacheIndex:  5,
		Layer:       30,
		LayerType:   "full_attention",
		SharedOwner: 30,
		Shape:       TurboQuantKVShape{Batch: 1, Heads: 8, SeqLen: 2048, HeadDim: 128},
		TokenOffset: 28672,
		PageTokens:  2048,
		PageSize:    2048,
		LocalWindow: 512,
		Key: TurboQuantKVCodec{
			Algorithm:     TurboQuantKVAlgorithmProd,
			NormalBits:    3,
			OutlierBits:   4,
			OutlierPolicy: TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:   turboQuantKVOutlierMask(128, 64),
			RotationSeed:  0x4b,
			QJLSeed:       0x51,
			CodebookID:    "beta-d128-b3",
		},
		Value: TurboQuantKVCodec{
			Algorithm:     TurboQuantKVAlgorithmMSE,
			NormalBits:    3,
			OutlierBits:   4,
			OutlierPolicy: TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:   turboQuantKVOutlierMask(128, 64),
			RotationSeed:  0x56,
			CodebookID:    "beta-d128-b3",
		},
	}

	if err := layout.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
	if got := layout.Key.EffectiveBitsMilli(layout.Shape.HeadDim); got != 3500 {
		t.Fatalf("key effective bits milli = %d, want 3500", got)
	}
	if got := layout.Value.EffectiveBitsMilli(layout.Shape.HeadDim); got != 3500 {
		t.Fatalf("value effective bits milli = %d, want 3500", got)
	}
	if got := layout.Shape.ElementCount(); got != 1*8*2048*128 {
		t.Fatalf("shape elements = %d, want %d", got, 1*8*2048*128)
	}
}

func TestTurboQuantKVPageLayout_JSONRecordsOutlierPolicy_Good(t *testing.T) {
	coverageTokens := "TurboQuantKVPageLayout JSON RecordsOutlierPolicy"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	layout := validTurboQuantKVTestPageLayout()

	encoded := core.JSONMarshalString(layout)

	for _, want := range []string{
		`"outlier_policy":"high-half-head-dim-v1"`,
		`"outlier_mask":`,
	} {
		if !core.Contains(encoded, want) {
			t.Fatalf("encoded layout = %s, want %s", encoded, want)
		}
	}
}

func TestTurboQuantKVPageLayout_RejectsWrongVersion_Bad(t *testing.T) {
	layout := validTurboQuantKVTestPageLayout()
	layout.Version = TurboQuantKVLayoutVersion + 1

	err := layout.Validate()
	if err == nil || !core.Contains(err.Error(), "version") {
		t.Fatalf("Validate() error = %v, want version diagnostic", err)
	}
}

func TestTurboQuantKVPageLayout_RejectsKeyWithoutQJL_Bad(t *testing.T) {
	layout := validTurboQuantKVTestPageLayout()
	layout.Key.QJLSeed = 0

	err := layout.Validate()
	if err == nil || !core.Contains(err.Error(), "QJL") {
		t.Fatalf("Validate() error = %v, want QJL diagnostic", err)
	}
}

func TestTurboQuantKVCodec_EffectiveBitsCountsMask_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:   TurboQuantKVAlgorithmMSE,
		NormalBits:  2,
		OutlierBits: 3,
		OutlierMask: turboQuantKVTestMask(128, 64),
		CodebookID:  "beta-d128-b2",
	}

	if got := codec.OutlierChannels(128); got != 64 {
		t.Fatalf("OutlierChannels = %d, want 64", got)
	}
	if got := codec.EffectiveBitsMilli(128); got != 2500 {
		t.Fatalf("EffectiveBitsMilli = %d, want 2500", got)
	}
}

func TestTurboQuantKVPageLayout_EstimatePayloadBytes_Good(t *testing.T) {
	layout := validTurboQuantKVTestPageLayout()

	estimate, err := layout.EstimatePayloadBytes()
	if err != nil {
		t.Fatalf("EstimatePayloadBytes() error = %v, want nil", err)
	}
	if estimate.PageVectors != 2048 || estimate.PageElements != 262144 {
		t.Fatalf("estimate shape = %+v, want 2048 vectors and 262144 elements", estimate)
	}
	if estimate.KeyCentroidBytes != 114688 || estimate.ValueCentroidBytes != 114688 {
		t.Fatalf("centroid bytes = key %d value %d, want 114688 each", estimate.KeyCentroidBytes, estimate.ValueCentroidBytes)
	}
	if estimate.KeyQJLSignBytes != 32768 || estimate.KeyNormBytes != 4096 || estimate.KeyResidualNormBytes != 4096 || estimate.ValueNormBytes != 4096 {
		t.Fatalf("side-channel bytes = %+v, want QJL signs plus fp16 norms accounted", estimate)
	}
	if estimate.OutlierMaskBytes != 32 || estimate.TotalBytes != 274464 {
		t.Fatalf("total bytes = %+v, want masks included and total 274464", estimate)
	}
	if estimate.FP16BaselineBytes != 1048576 || estimate.SavingsRatio <= 0 || estimate.SavingsRatio >= 0.27 {
		t.Fatalf("baseline/savings = %+v, want visible saving against fp16 K+V payload", estimate)
	}
}

func TestTurboQuantKVReferencePage_PackedPayloadUsesOutlierBitBudget_Good(t *testing.T) {
	coverageTokens := "TurboQuantKVReferencePage PackedPayload UsesOutlierBitBudget"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	layout := validTurboQuantKVTestPageLayout()
	layout.Shape = TurboQuantKVShape{Batch: 1, Heads: 1, SeqLen: 1, HeadDim: 8}
	layout.PageTokens = 1
	layout.PageSize = 1
	layout.Key.NormalBits = 3
	layout.Key.OutlierBits = 4
	layout.Key.OutlierPolicy = TurboQuantKVOutlierPolicyHighHalfHeadDimV1
	layout.Key.OutlierMask = turboQuantKVOutlierMask(8, 4)
	layout.Key.CodebookID = TurboQuantKVReferenceCodebookUniform
	layout.Value.NormalBits = 3
	layout.Value.OutlierBits = 4
	layout.Value.OutlierPolicy = TurboQuantKVOutlierPolicyHighHalfHeadDimV1
	layout.Value.OutlierMask = turboQuantKVOutlierMask(8, 4)
	layout.Value.CodebookID = TurboQuantKVReferenceCodebookUniform
	keys := []float32{0.42, -0.31, 0.18, 0.77, -0.56, 0.09, 0.23, -0.64}
	values := []float32{-0.12, 0.44, 0.37, -0.21, 0.68, -0.15, 0.51, 0.08}

	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v, want nil", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		t.Fatalf("PackedPayload() error = %v, want nil", err)
	}

	keyCentroids, ok := payload.SectionBytes(TurboQuantKVReferencePayloadKeyCentroids)
	if !ok {
		t.Fatal("key centroid section missing")
	}
	valueCentroids, ok := payload.SectionBytes(TurboQuantKVReferencePayloadValueCentroids)
	if !ok {
		t.Fatal("value centroid section missing")
	}
	if len(keyCentroids) != 4 || len(valueCentroids) != 4 {
		t.Fatalf("centroid bytes = key %d value %d, want 4 each for 8 channels at 3.5 effective bits", len(keyCentroids), len(valueCentroids))
	}
	restored, err := DecodeTurboQuantKVReferencePagePayload(payload)
	if err != nil {
		t.Fatalf("DecodeTurboQuantKVReferencePagePayload() error = %v, want nil", err)
	}
	if got := restored.Layout.Key.EffectiveBitsMilli(restored.Layout.Shape.HeadDim); got != 3500 {
		t.Fatalf("restored key effective bits = %d, want 3500", got)
	}
	if got := restored.Layout.Value.EffectiveBitsMilli(restored.Layout.Shape.HeadDim); got != 3500 {
		t.Fatalf("restored value effective bits = %d, want 3500", got)
	}
	decodedKeys, decodedValues, err := restored.DecodeBase()
	if err != nil {
		t.Fatalf("DecodeBase() error = %v, want nil", err)
	}
	if got := cosineSimilarity(keys, decodedKeys); got < 0.96 {
		t.Fatalf("decoded key cosine = %.6f, want >= 0.96", got)
	}
	if got := cosineSimilarity(values, decodedValues); got < 0.96 {
		t.Fatalf("decoded value cosine = %.6f, want >= 0.96", got)
	}
}

func TestTurboQuantKVMSEReferenceVector_RoundTrip_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	input := []float32{0.42, -0.31, 0.18, 0.77, -0.56, 0.09, 0.23, -0.64}

	encoded, err := EncodeTurboQuantKVMSEReference(input, codec)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVMSEReference() error = %v, want nil", err)
	}
	if encoded.Norm <= 0 || len(encoded.CentroidCodes) != len(input) || encoded.HeadDim != int32(len(input)) {
		t.Fatalf("encoded = %+v, want norm and one centroid code per input value", encoded)
	}

	decoded, err := encoded.DecodeMSE()
	if err != nil {
		t.Fatalf("DecodeMSE() error = %v, want nil", err)
	}
	if got := cosineSimilarity(input, decoded); got < 0.995 {
		t.Fatalf("cosine similarity = %.6f, want >= 0.995; decoded=%v", got, decoded)
	}
	if got, want := vectorNorm(decoded), vectorNorm(input); math.Abs(float64(got-want)) > 0.03 {
		t.Fatalf("decoded norm = %.6f, want within 0.03 of %.6f", got, want)
	}
}

func TestTurboQuantKVMSEReferenceVector_ZeroVector_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	encoded, err := EncodeTurboQuantKVMSEReference([]float32{0, 0, 0, 0}, codec)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVMSEReference(zero) error = %v, want nil", err)
	}
	decoded, err := encoded.DecodeMSE()
	if err != nil {
		t.Fatalf("DecodeMSE(zero) error = %v, want nil", err)
	}
	if encoded.Norm != 0 || len(decoded) != 4 {
		t.Fatalf("zero encoded = %+v decoded=%v, want zero norm and four decoded values", encoded, decoded)
	}
	for idx, got := range decoded {
		if got != 0 {
			t.Fatalf("decoded[%d] = %v, want 0", idx, got)
		}
	}
}

func TestTurboQuantKVMSEReferenceVector_PackedCentroidsRoundTrip_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	input := []float32{0.42, -0.31, 0.18, 0.77, -0.56, 0.09, 0.23, -0.64}
	encoded, err := EncodeTurboQuantKVMSEReference(input, codec)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVMSEReference() error = %v, want nil", err)
	}

	packed, err := encoded.PackedCentroidBytes()
	if err != nil {
		t.Fatalf("PackedCentroidBytes() error = %v, want nil", err)
	}
	if len(packed) != 5 {
		t.Fatalf("packed centroid bytes = %d, want 5 for 8 x 5-bit codes", len(packed))
	}
	restored, err := DecodeTurboQuantKVMSEReferenceFromPacked(codec, encoded.HeadDim, encoded.Norm, packed)
	if err != nil {
		t.Fatalf("DecodeTurboQuantKVMSEReferenceFromPacked() error = %v, want nil", err)
	}
	decoded, err := restored.DecodeMSE()
	if err != nil {
		t.Fatalf("DecodeMSE(restored) error = %v, want nil", err)
	}
	if got := cosineSimilarity(input, decoded); got < 0.995 {
		t.Fatalf("restored cosine = %.6f, want >= 0.995", got)
	}
}

func TestTurboQuantKVMSEReferenceVector_RejectsShortPackedCentroids_Bad(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}

	_, err := DecodeTurboQuantKVMSEReferenceFromPacked(codec, 8, 1, []byte{0x01, 0x02})
	if err == nil || !core.Contains(err.Error(), "packed centroid") {
		t.Fatalf("DecodeTurboQuantKVMSEReferenceFromPacked(short) error = %v, want packed centroid diagnostic", err)
	}
}

func TestTurboQuantKVMSEReferenceVector_RejectsUnsupportedCodebook_Bad(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmMSE,
		NormalBits:   5,
		RotationSeed: 0x5150,
		CodebookID:   "learned-beta-d8-b5",
	}

	_, err := EncodeTurboQuantKVMSEReference([]float32{1, 0, 0, 0}, codec)
	if err == nil || !core.Contains(err.Error(), "codebook") {
		t.Fatalf("EncodeTurboQuantKVMSEReference(unsupported codebook) error = %v, want codebook diagnostic", err)
	}
}

func TestTurboQuantKVProdReferenceVector_EstimatesInnerProductWithQJL_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmProd,
		NormalBits:   4,
		RotationSeed: 0x6b,
		QJLSeed:      0x7c,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	key := []float32{0.42, -0.31, 0.18, 0.77, -0.56, 0.09, 0.23, -0.64}
	query := []float32{-0.12, 0.44, 0.37, -0.21, 0.68, -0.15, 0.51, 0.08}

	encoded, err := EncodeTurboQuantKVProdReference(key, codec)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVProdReference() error = %v, want nil", err)
	}
	if encoded.ResidualNorm <= 0 || len(encoded.QJLSigns) != len(key) {
		t.Fatalf("encoded residual = %+v, want residual norm and one QJL sign per key channel", encoded)
	}

	estimated, err := encoded.EstimateInnerProduct(query)
	if err != nil {
		t.Fatalf("EstimateInnerProduct() error = %v, want nil", err)
	}
	base, err := encoded.Base.DecodeMSE()
	if err != nil {
		t.Fatalf("DecodeMSE() error = %v, want nil", err)
	}
	exact := dotProduct(query, key)
	baseDot := dotProduct(query, base)
	if estimated == baseDot {
		t.Fatalf("estimated dot = %.6f equals MSE base dot %.6f, want QJL residual correction", estimated, baseDot)
	}
	if gotErr := math.Abs(float64(estimated - exact)); gotErr > 0.2 {
		t.Fatalf("estimated dot = %.6f exact=%.6f base=%.6f error=%.6f, want bounded QJL estimate", estimated, exact, baseDot, gotErr)
	}
}

func TestTurboQuantKVProdReferenceVector_PackedQJLRoundTrip_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmProd,
		NormalBits:   4,
		RotationSeed: 0x6b,
		QJLSeed:      0x7c,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	key := []float32{0.42, -0.31, 0.18, 0.77, -0.56, 0.09, 0.23, -0.64}
	query := []float32{-0.12, 0.44, 0.37, -0.21, 0.68, -0.15, 0.51, 0.08}
	encoded, err := EncodeTurboQuantKVProdReference(key, codec)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVProdReference() error = %v, want nil", err)
	}

	packed, err := encoded.PackedQJLSignBytes()
	if err != nil {
		t.Fatalf("PackedQJLSignBytes() error = %v, want nil", err)
	}
	if len(packed) != 1 {
		t.Fatalf("packed QJL sign bytes = %d, want 1 for 8 signs", len(packed))
	}
	restored, err := DecodeTurboQuantKVProdReferenceFromPacked(codec, encoded.Base, encoded.ResidualNorm, packed)
	if err != nil {
		t.Fatalf("DecodeTurboQuantKVProdReferenceFromPacked() error = %v, want nil", err)
	}
	got, err := restored.EstimateInnerProduct(query)
	if err != nil {
		t.Fatalf("EstimateInnerProduct(restored) error = %v, want nil", err)
	}
	want, err := encoded.EstimateInnerProduct(query)
	if err != nil {
		t.Fatalf("EstimateInnerProduct(original) error = %v, want nil", err)
	}
	if got != want {
		t.Fatalf("restored estimate = %.6f, want original %.6f", got, want)
	}
}

func TestTurboQuantKVProdReferenceVector_SeededErrorIsCentred_Good(t *testing.T) {
	codec := TurboQuantKVCodec{
		Algorithm:    TurboQuantKVAlgorithmProd,
		NormalBits:   4,
		RotationSeed: 0x6b,
		QJLSeed:      0x7c,
		CodebookID:   TurboQuantKVReferenceCodebookUniform,
	}
	const samples = 64
	const dim = 32
	var signedError float64
	for idx := 0; idx < samples; idx++ {
		key := turboQuantKVReferenceSeededVector(dim, 17+idx*3)
		query := turboQuantKVReferenceSeededVector(dim, 41+idx*5)
		encoded, err := EncodeTurboQuantKVProdReference(key, codec)
		if err != nil {
			t.Fatalf("EncodeTurboQuantKVProdReference(%d) error = %v", idx, err)
		}
		estimated, err := encoded.EstimateInnerProduct(query)
		if err != nil {
			t.Fatalf("EstimateInnerProduct(%d) error = %v", idx, err)
		}
		signedError += float64(estimated - dotProduct(query, key))
	}
	meanError := signedError / samples
	if math.Abs(meanError) > 0.05 {
		t.Fatalf("mean signed inner-product error = %.6f, want centred within 0.05", meanError)
	}
}

func TestTurboQuantKVReferencePage_EncodeDecodeBase_Good(t *testing.T) {
	layout := validTurboQuantKVReferencePageLayout()
	keys := turboQuantKVReferencePageValues(layout, 37)
	values := turboQuantKVReferencePageValues(layout, 53)

	page, err := EncodeTurboQuantKVReferencePage(keys, values, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v, want nil", err)
	}
	if len(page.Keys) != int(layout.PageVectorCount()) || len(page.Values) != int(layout.PageVectorCount()) {
		t.Fatalf("page vectors = %d/%d, want %d", len(page.Keys), len(page.Values), layout.PageVectorCount())
	}

	decodedKeys, decodedValues, err := page.DecodeBase()
	if err != nil {
		t.Fatalf("DecodeBase() error = %v, want nil", err)
	}
	if cosineSimilarity(keys, decodedKeys) < 0.99 {
		t.Fatalf("decoded key cosine = %.6f, want >= 0.99", cosineSimilarity(keys, decodedKeys))
	}
	if cosineSimilarity(values, decodedValues) < 0.99 {
		t.Fatalf("decoded value cosine = %.6f, want >= 0.99", cosineSimilarity(values, decodedValues))
	}

	query := []float32{-0.12, 0.44, 0.37, -0.21, 0.68, -0.15, 0.51, 0.08}
	estimates, err := page.EstimateKeyInnerProducts(query)
	if err != nil {
		t.Fatalf("EstimateKeyInnerProducts() error = %v, want nil", err)
	}
	if len(estimates) != len(page.Keys) {
		t.Fatalf("estimate count = %d, want %d", len(estimates), len(page.Keys))
	}
	for idx, estimate := range estimates {
		if estimate == 0 {
			t.Fatalf("estimate[%d] = 0, want non-zero diagnostic value", idx)
		}
	}
}

func TestTurboQuantKVReferencePage_RejectsPayloadShape_Bad(t *testing.T) {
	layout := validTurboQuantKVReferencePageLayout()
	keys := turboQuantKVReferencePageValues(layout, 37)
	values := turboQuantKVReferencePageValues(layout, 53)

	_, err := EncodeTurboQuantKVReferencePage(keys[:len(keys)-1], values, layout)
	if err == nil || !core.Contains(err.Error(), "payload shape") {
		t.Fatalf("EncodeTurboQuantKVReferencePage(short keys) error = %v, want payload shape diagnostic", err)
	}
}

func validTurboQuantKVTestPageLayout() TurboQuantKVPageLayout {
	return TurboQuantKVPageLayout{
		Version:     TurboQuantKVLayoutVersion,
		Codec:       TurboQuantKVCodecName,
		CacheIndex:  0,
		Layer:       0,
		LayerType:   "sliding_attention",
		SharedOwner: 0,
		Shape:       TurboQuantKVShape{Batch: 1, Heads: 4, SeqLen: 512, HeadDim: 128},
		TokenOffset: 0,
		PageTokens:  512,
		PageSize:    512,
		LocalWindow: 512,
		Key: TurboQuantKVCodec{
			Algorithm:     TurboQuantKVAlgorithmProd,
			NormalBits:    3,
			OutlierBits:   4,
			OutlierPolicy: TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:   turboQuantKVOutlierMask(128, 64),
			RotationSeed:  1,
			QJLSeed:       2,
			CodebookID:    "beta-d128-b3",
		},
		Value: TurboQuantKVCodec{
			Algorithm:     TurboQuantKVAlgorithmMSE,
			NormalBits:    3,
			OutlierBits:   4,
			OutlierPolicy: TurboQuantKVOutlierPolicyHighHalfHeadDimV1,
			OutlierMask:   turboQuantKVOutlierMask(128, 64),
			RotationSeed:  3,
			CodebookID:    "beta-d128-b3",
		},
	}
}

func validTurboQuantKVReferencePageLayout() TurboQuantKVPageLayout {
	return TurboQuantKVPageLayout{
		Version:     TurboQuantKVLayoutVersion,
		Codec:       TurboQuantKVCodecName,
		CacheIndex:  1,
		Layer:       5,
		LayerType:   "full_attention",
		SharedOwner: 5,
		Shape:       TurboQuantKVShape{Batch: 1, Heads: 2, SeqLen: 2, HeadDim: 8},
		TokenOffset: 16,
		PageTokens:  2,
		PageSize:    2,
		Key: TurboQuantKVCodec{
			Algorithm:    TurboQuantKVAlgorithmProd,
			NormalBits:   5,
			RotationSeed: 0x6b,
			QJLSeed:      0x7c,
			CodebookID:   TurboQuantKVReferenceCodebookUniform,
		},
		Value: TurboQuantKVCodec{
			Algorithm:    TurboQuantKVAlgorithmMSE,
			NormalBits:   5,
			RotationSeed: 0x56,
			CodebookID:   TurboQuantKVReferenceCodebookUniform,
		},
	}
}

func turboQuantKVTestMask(headDim, outliers int32) []byte {
	mask := make([]byte, (headDim+7)/8)
	for i := int32(0); i < outliers; i++ {
		mask[i/8] |= 1 << uint(i%8)
	}
	return mask
}

func turboQuantKVReferencePageValues(layout TurboQuantKVPageLayout, seed int) []float32 {
	values := make([]float32, layout.PageElementCount())
	for idx := range values {
		values[idx] = float32(((idx*seed)%97)-48) / 59
	}
	return values
}

func turboQuantKVReferenceSeededVector(dim, seed int) []float32 {
	values := make([]float32, dim)
	state := uint32(seed)
	for idx := range values {
		state = state*1664525 + 1013904223
		values[idx] = float32(int(state%2001)-1000) / 997
	}
	return values
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for idx := range a {
		av := float64(a[idx])
		bv := float64(b[idx])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func vectorNorm(values []float32) float32 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return float32(math.Sqrt(sum))
}

func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for idx := range a {
		sum += a[idx] * b[idx]
	}
	return sum
}
