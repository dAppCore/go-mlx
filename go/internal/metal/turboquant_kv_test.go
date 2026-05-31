// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
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
			Algorithm:    TurboQuantKVAlgorithmProd,
			NormalBits:   3,
			OutlierBits:  4,
			OutlierMask:  turboQuantKVTestMask(128, 64),
			RotationSeed: 0x4b,
			QJLSeed:      0x51,
			CodebookID:   "beta-d128-b3",
		},
		Value: TurboQuantKVCodec{
			Algorithm:    TurboQuantKVAlgorithmMSE,
			NormalBits:   3,
			OutlierBits:  4,
			OutlierMask:  turboQuantKVTestMask(128, 64),
			RotationSeed: 0x56,
			CodebookID:   "beta-d128-b3",
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
			Algorithm:    TurboQuantKVAlgorithmProd,
			NormalBits:   3,
			OutlierBits:  4,
			OutlierMask:  turboQuantKVTestMask(128, 64),
			RotationSeed: 1,
			QJLSeed:      2,
			CodebookID:   "beta-d128-b3",
		},
		Value: TurboQuantKVCodec{
			Algorithm:    TurboQuantKVAlgorithmMSE,
			NormalBits:   3,
			OutlierBits:  4,
			OutlierMask:  turboQuantKVTestMask(128, 64),
			RotationSeed: 3,
			CodebookID:   "beta-d128-b3",
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
