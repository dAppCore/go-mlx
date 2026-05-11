// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func testJANGTQInfo() *JANGQuantizationInfo {
	return &JANGQuantizationInfo{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
}

func TestJANGPackedTensorDescriptor_MXTQRoutedExpert_Good(t *testing.T) {
	desc, err := NewJANGPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.17.w1.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewJANGPackedTensorDescriptor() error = %v", err)
	}

	if desc.Type != "jangtq" || desc.Format != "mxtq" || desc.Profile != "JANGTQ" {
		t.Fatalf("profile = type:%q format:%q profile:%q", desc.Type, desc.Format, desc.Profile)
	}
	if desc.Role != JANGTensorRoleRoutedExpert || desc.Bits != 2 || desc.GroupSize != 4 {
		t.Fatalf("descriptor = %+v, want routed expert 2-bit group 4", desc)
	}
	if desc.Elements != 8 || desc.Groups != 2 || desc.PackedBytes != 2 || desc.ScaleCount != 2 || desc.BiasCount != 2 {
		t.Fatalf("descriptor sizes = %+v, want 8 elements, 2 groups, 2 packed bytes", desc)
	}
	if desc.BitOrder != JANGBitOrderLSB0 || desc.Encoding != JANGEncodingAffine {
		t.Fatalf("layout = bit_order:%q encoding:%q", desc.BitOrder, desc.Encoding)
	}
}

func TestJANGPackedTensorDescriptor_AttentionUsesWideBits_Good(t *testing.T) {
	desc, err := NewJANGPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewJANGPackedTensorDescriptor() error = %v", err)
	}

	if desc.Role != JANGTensorRoleAttention || desc.Bits != 8 || desc.PackedBytes != 8 {
		t.Fatalf("descriptor = %+v, want attention 8-bit un-nibbled bytes", desc)
	}
}

func TestJANGPackedTensorDescriptor_BadUnsupportedBits(t *testing.T) {
	info := testJANGTQInfo()
	info.RoutedExpertBits = 5

	_, err := NewJANGPackedTensorDescriptor("model.layers.0.mlp.experts.0.down_proj.weight", []uint64{4, 4}, info)
	if err == nil || !core.Contains(err.Error(), "unsupported") || !core.Contains(err.Error(), "5-bit") {
		t.Fatalf("error = %v, want explicit unsupported 5-bit error", err)
	}
}

func TestJANGPackedTensorDequantize_Good(t *testing.T) {
	desc, err := NewJANGPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewJANGPackedTensorDescriptor() error = %v", err)
	}
	packed, err := PackJANGQuantizedValues(desc, []uint8{0, 1, 2, 3, 0, 1, 2, 3})
	if err != nil {
		t.Fatalf("PackJANGQuantizedValues() error = %v", err)
	}

	out, err := DequantizeJANGPackedTensor(desc, packed, []float32{0.5, 1}, []float32{-1, 10})
	if err != nil {
		t.Fatalf("DequantizeJANGPackedTensor() error = %v", err)
	}

	want := []float32{-1, -0.5, 0, 0.5, 10, 11, 12, 13}
	if len(out) != len(want) {
		t.Fatalf("out length = %d, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v (all=%v)", i, out[i], want[i], out)
		}
	}
}

func TestJANGPackedTensorValidate_BadPackedLength(t *testing.T) {
	desc, err := NewJANGPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewJANGPackedTensorDescriptor() error = %v", err)
	}

	err = ValidateJANGPackedTensor(desc, []byte{0}, []float32{1, 1}, []float32{0, 0})
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want packed length validation", err)
	}
}

func TestJANGPackedQuantizationProfile_Good(t *testing.T) {
	profile := BuildJANGPackedQuantizationProfile(testJANGTQInfo())
	if profile == nil {
		t.Fatal("profile = nil")
	}
	if profile.Type != "jangtq" || profile.Format != "mxtq" || !profile.Mixed {
		t.Fatalf("profile = %+v, want JANGTQ/MXTQ mixed profile", profile)
	}
	if profile.MinBits != 2 || profile.MaxBits != 8 || profile.RoleBits[string(JANGTensorRoleRoutedExpert)] != 2 || profile.RoleBits[string(JANGTensorRoleAttention)] != 8 {
		t.Fatalf("role bits = %+v, min/max=%d/%d", profile.RoleBits, profile.MinBits, profile.MaxBits)
	}
}
