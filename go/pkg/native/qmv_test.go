// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestQMVZeroSizedProjection(t *testing.T) {
	requireNativeRuntime(t)

	got, err := QMV(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("QMV zero-sized projection: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("QMV zero-sized projection length = %d, want 0", len(got))
	}

	gotBF16, err := QMVBF16(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("QMVBF16 zero-sized projection: %v", err)
	}
	if len(gotBF16) != 0 {
		t.Fatalf("QMVBF16 zero-sized projection length = %d, want 0", len(gotBF16))
	}
}

func TestQMVRejectsInputShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := QMV([]float32{1}, nil, nil, nil, 0, 2, 64, 4); err == nil {
		t.Fatal("expected QMV to reject len(x) != inDim")
	}
	if _, err := QMVBF16([]byte{0}, nil, nil, nil, 0, 1, 64, 4); err == nil {
		t.Fatal("expected QMVBF16 to reject len(x) != inDim*2")
	}
}
