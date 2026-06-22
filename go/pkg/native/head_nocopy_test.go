// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestNewHeadEncoderNilShardBuffersFallsBack(t *testing.T) {
	h, err := newHeadEncoder(nil, nil, nil, nil, nil, 64, 128, 64, 4, 1e-5, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder nil shard buffers: %v", err)
	}
	if h != nil {
		t.Fatalf("newHeadEncoder nil shard buffers = %+v, want nil fallback", h)
	}
}

func TestHeadEncoderRejectsHiddenShapeMismatch(t *testing.T) {
	h := &headEncoder{dModel: 2, vocab: 2}
	if _, err := h.encode(toBF16Bytes([]float32{1}), false); err == nil {
		t.Fatal("expected headEncoder.encode to reject hidden shape mismatch")
	}
}
