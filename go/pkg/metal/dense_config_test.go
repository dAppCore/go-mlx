// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package metal

import "testing"

// TestParseDenseConfig_NestedRope_Good proves rope_theta + partial_rotary_factor
// nested under text_config.rope_parameters (the Qwen3.5/3.6 shape) fill the flat
// fields the engine reads.
func TestParseDenseConfig_NestedRope_Good(t *testing.T) {
	const cfg = `{"model_type":"qwen3_5","text_config":{"hidden_size":256,"num_hidden_layers":4,` +
		`"num_attention_heads":4,"head_dim":64,` +
		`"rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10]},` +
		`"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}}`
	c, err := ParseDenseConfig([]byte(cfg))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.RopeTheta != 1e7 {
		t.Errorf("ropeTheta %g, want 1e7 (nested rope_parameters not lifted)", c.RopeTheta)
	}
	if c.PartialRotaryFactor != 0.25 {
		t.Errorf("partialRotary %g, want 0.25", c.PartialRotaryFactor)
	}
	if c.RotaryDims() != 16 {
		t.Errorf("RotaryDims() = %d, want 16 (64·0.25)", c.RotaryDims())
	}
	if len(c.LayerTypes) != 4 {
		t.Errorf("layerTypes %d, want 4 (from text_config)", len(c.LayerTypes))
	}
}

// TestParseDenseConfig_FlatRopeWins_Good proves a flat rope_theta stays
// authoritative even when rope_parameters is also present (no surprise override
// for a family that declares both).
func TestParseDenseConfig_FlatRopeWins_Good(t *testing.T) {
	const cfg = `{"model_type":"qwen3","hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,` +
		`"head_dim":64,"rope_theta":1000000,"rope_parameters":{"rope_theta":99}}`
	c, err := ParseDenseConfig([]byte(cfg))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.RopeTheta != 1e6 {
		t.Errorf("ropeTheta %g, want 1e6 (flat field must win)", c.RopeTheta)
	}
}

// TestParseDenseConfig_NoRopeUnchanged_Good is the regression for the flat-rope
// dense families (Llama/Mistral/Qwen2/3): no rope_parameters → the transformers
// default theta and full rotary, exactly as before.
func TestParseDenseConfig_NoRopeUnchanged_Good(t *testing.T) {
	const cfg = `{"model_type":"llama","hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,"head_dim":64}`
	c, err := ParseDenseConfig([]byte(cfg))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.RopeTheta != 10000 {
		t.Errorf("ropeTheta %g, want 10000 default", c.RopeTheta)
	}
	if c.PartialRotaryFactor != 0 {
		t.Errorf("partialRotary %g, want 0", c.PartialRotaryFactor)
	}
	if c.RotaryDims() != 64 {
		t.Errorf("RotaryDims() = %d, want 64 (full rotary)", c.RotaryDims())
	}
}

// TestDenseConfig_RotaryDims_Good pins the partial-rotary dim: the leading
// fraction when 0 < factor < 1, the full head dim otherwise (0 or ≥1).
func TestDenseConfig_RotaryDims_Good(t *testing.T) {
	for _, tc := range []struct {
		f          float32
		head, want int
	}{
		{0, 128, 128},   // unset → full
		{1, 128, 128},   // 1 → full
		{0.25, 128, 32}, // Qwen3.5/3.6
		{0.5, 64, 32},
		{2, 128, 128}, // out-of-range → full
	} {
		c := &DenseConfig{PartialRotaryFactor: tc.f}
		c.HeadDim = int32(tc.head)
		if got := c.RotaryDims(); got != tc.want {
			t.Errorf("RotaryDims(factor=%g, head=%d) = %d, want %d", tc.f, tc.head, got, tc.want)
		}
	}
}
