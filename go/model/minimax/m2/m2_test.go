// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/safetensors"
)

const miniMaxM2FixtureConfig = `{
	"architectures": ["MiniMaxM2ForCausalLM"],
	"model_type": "minimax_m2",
	"vocab_size": 200064,
	"hidden_size": 3072,
	"intermediate_size": 1536,
	"num_hidden_layers": 62,
	"num_attention_heads": 48,
	"num_key_value_heads": 8,
	"head_dim": 128,
	"max_position_embeddings": 196608,
	"num_local_experts": 256,
	"num_experts_per_tok": 8,
	"scoring_func": "sigmoid",
	"use_routing_bias": true,
	"use_mtp": true,
	"num_mtp_modules": 3,
	"mtp_transformer_layers": 1,
	"use_qk_norm": true,
	"rotary_dim": 64,
	"rope_theta": 5000000
}`

func TestM2_ParseConfig_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}

	if cfg.ModelType != "minimax_m2" || cfg.HiddenSize != 3072 || cfg.IntermediateSize != 1536 || cfg.NumHiddenLayers != 62 {
		t.Fatalf("shape config = %+v", cfg)
	}
	if cfg.NumLocalExperts != 256 || cfg.NumExpertsPerToken != 8 || cfg.ScoringFunc != "sigmoid" || !cfg.UseRoutingBias {
		t.Fatalf("MoE config = %+v", cfg)
	}
	if !cfg.UseMTP || cfg.NumMTPModules != 3 || cfg.MTPTransformerLayers != 1 || !cfg.UseQKNorm {
		t.Fatalf("extra config = %+v", cfg)
	}
}

func TestM2_BuildTensorPlan_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	// BuildTensorPlan carries the config through and synthesises the MXTQ
	// packed profile (2-bit routed experts) from the JANGTQ info.
	if plan.Config.HiddenSize != 3072 || plan.Config.NumLocalExperts != 256 {
		t.Fatalf("plan config = %+v, want fixture hidden/expert sizes carried through", plan.Config)
	}
	if plan.JANG == nil || plan.Quantization == nil {
		t.Fatalf("plan = %+v, want JANG info and packed quantization profile", plan)
	}
	if plan.Quantization.Format != "mxtq" || plan.Quantization.RoleBits[string(jang.TensorRoleRoutedExpert)] != 2 {
		t.Fatalf("plan quantization = %+v, want MXTQ routed expert profile", plan.Quantization)
	}
}

func TestM2_TensorPlan_LayerTensorSpecs_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	specs, err := plan.LayerTensorSpecs(0, 17)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}

	router := findMiniMaxM2Spec(specs, TensorRoleRouterGate)
	if router.Name != "model.layers.0.block_sparse_moe.gate.weight" || router.Packed != nil {
		t.Fatalf("router spec = %+v, want dense router gate", router)
	}
	attention := findMiniMaxM2Spec(specs, TensorRoleAttentionQ)
	if attention.Packed == nil || attention.Packed.Bits != 8 || attention.Packed.Role != jang.TensorRoleAttention {
		t.Fatalf("attention spec = %+v, want 8-bit packed attention descriptor", attention)
	}
	if len(attention.Shape) != 2 || attention.Shape[0] != 6144 || attention.Shape[1] != 3072 {
		t.Fatalf("attention shape = %+v, want q_size x hidden_size", attention.Shape)
	}
	key := findMiniMaxM2Spec(specs, TensorRoleAttentionK)
	if len(key.Shape) != 2 || key.Shape[0] != 1024 || key.Shape[1] != 3072 {
		t.Fatalf("key shape = %+v, want kv_size x hidden_size", key.Shape)
	}
	expert := findMiniMaxM2Spec(specs, TensorRoleExpertGate)
	if expert.Name != "model.layers.0.block_sparse_moe.experts.17.gate_proj.weight" {
		t.Fatalf("expert name = %q", expert.Name)
	}
	if expert.Packed == nil || expert.Packed.Bits != 2 || expert.Packed.Role != jang.TensorRoleRoutedExpert {
		t.Fatalf("expert spec = %+v, want 2-bit routed expert descriptor", expert)
	}
	if len(expert.Aliases) == 0 || expert.Aliases[0] != "model.layers.0.mlp.experts.17.gate_proj.weight" {
		t.Fatalf("expert aliases = %+v, want mlp checkpoint alias", expert.Aliases)
	}
}

func TestM2_TensorPlan_ValidateTensorNames_Bad(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	err = plan.ValidateTensorNames(map[string]bool{
		"model.layers.0.block_sparse_moe.gate.weight":                true,
		"model.layers.0.block_sparse_moe.e_score_correction_bias":    true,
		"model.layers.0.self_attn.q_proj.weight":                     true,
		"model.layers.0.self_attn.k_proj.weight":                     true,
		"model.layers.0.self_attn.v_proj.weight":                     true,
		"model.layers.0.self_attn.o_proj.weight":                     true,
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": true,
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight": true,
	})
	if err == nil || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want missing expert up_proj", err)
	}
}

func TestM2_TensorPlan_LayerTensorSpecs_Ugly(t *testing.T) {
	// Degenerate-but-valid plan: UseRoutingBias is false, so the optional
	// router-bias spec is omitted and exactly 8 specs come back (4 attention +
	// router gate + 3 expert projections). The down_proj projection transposes
	// the gate/up shape — the branch the bias-on Good path doesn't pin.
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    4,
		NumExpertsPerToken: 2,
	}, nil)
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	if len(specs) != 8 {
		t.Fatalf("specs = %d, want 8 with routing bias disabled", len(specs))
	}
	if findMiniMaxM2Spec(specs, TensorRoleRouterBias).Name != "" {
		t.Fatalf("router bias spec present, want omitted when UseRoutingBias is false")
	}
	down := findMiniMaxM2Spec(specs, TensorRoleExpertDown)
	// down_proj is [hidden, intermediate] — transposed from gate/up.
	if len(down.Shape) != 2 || down.Shape[0] != 4 || down.Shape[1] != 8 {
		t.Fatalf("down_proj shape = %+v, want [hidden intermediate] = [4 8]", down.Shape)
	}
}

func TestM2_TensorPlan_ValidateTensorNames_Good(t *testing.T) {
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    4,
		NumExpertsPerToken: 2,
		UseRoutingBias:     true,
	}, nil)
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	// A checkpoint carrying every required first-layer/first-expert tensor
	// passes the pre-flight with a nil error.
	if err := plan.ValidateTensorNames(map[string]bool{
		"model.layers.0.self_attn.q_proj.weight":                     true,
		"model.layers.0.self_attn.k_proj.weight":                     true,
		"model.layers.0.self_attn.v_proj.weight":                     true,
		"model.layers.0.self_attn.o_proj.weight":                     true,
		"model.layers.0.block_sparse_moe.gate.weight":                true,
		"model.layers.0.block_sparse_moe.e_score_correction_bias":    true,
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": true,
		"model.layers.0.block_sparse_moe.experts.0.up_proj.weight":   true,
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight": true,
	}); err != nil {
		t.Fatalf("ValidateTensorNames(complete set) = %v, want nil", err)
	}
	// Alias acceptance: the qkv_proj fused alias satisfies the q/k/v slots.
	if err := plan.ValidateTensorNames(map[string]bool{
		"model.layers.0.self_attn.qkv_proj.weight":                true,
		"model.layers.0.self_attn.o_proj.weight":                  true,
		"model.layers.0.block_sparse_moe.gate.weight":             true,
		"model.layers.0.block_sparse_moe.e_score_correction_bias": true,
		"model.layers.0.mlp.experts.0.gate_proj.weight":           true,
		"model.layers.0.mlp.experts.0.up_proj.weight":             true,
		"model.layers.0.mlp.experts.0.down_proj.weight":           true,
	}); err != nil {
		t.Fatalf("ValidateTensorNames(alias set) = %v, want nil via qkv/mlp aliases", err)
	}
}

func findMiniMaxM2Spec(specs []TensorSpec, role TensorRole) TensorSpec {
	for _, spec := range specs {
		if spec.Role == role {
			return spec
		}
	}
	return TensorSpec{}
}

func findMiniMaxM2ResolvedTensor(tensors []ResolvedTensor, role TensorRole) ResolvedTensor {
	for _, tensor := range tensors {
		if tensor.Role == role {
			return tensor
		}
	}
	return ResolvedTensor{}
}

func roughlyEqual32(a, b, epsilon float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}

func miniMaxM2Float32SlicesRoughlyEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !roughlyEqual32(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

func miniMaxM2SkeletonRawTensors(t *testing.T, plan TensorPlan, badAttentionShape bool) []miniMaxM2RawSafetensor {
	t.Helper()
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	var tensors []miniMaxM2RawSafetensor
	for _, role := range []TensorRole{
		TensorRoleAttentionQ,
		TensorRoleAttentionK,
		TensorRoleAttentionV,
		TensorRoleAttentionO,
	} {
		spec := findMiniMaxM2Spec(specs, role)
		if spec.Packed == nil {
			t.Fatalf("attention spec %s has no packed descriptor", role)
		}
		packedBytes := spec.Packed.PackedBytes
		if badAttentionShape && role == TensorRoleAttentionQ {
			packedBytes--
		}
		tensors = append(tensors, miniMaxM2RawSafetensor{
			Name:  spec.Name,
			DType: "U8",
			Shape: []int{packedBytes},
			Raw:   make([]byte, packedBytes),
		})
	}
	tensors = append(tensors,
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
			1, 1, 0, 0,
		}, 3, 4),
	)
	if plan.Config.UseRoutingBias {
		tensors = append(tensors, miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.25, -0.25}, 3))
	}
	return tensors
}

func miniMaxM2SmallJANGTQPlan(t testing.TB) TensorPlan {
	t.Helper()
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 1,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	return plan
}

func miniMaxM2LazyExpertFixtureTensors(t testing.TB, expertID int, values []uint8) []miniMaxM2RawSafetensor {
	t.Helper()
	prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", expertID)
	gate := miniMaxM2PackedRawTensor(t, prefix+".gate_proj.weight", values)
	up := miniMaxM2PackedRawTensor(t, prefix+".up_proj.weight", values)
	down := miniMaxM2PackedRawTensor(t, prefix+".down_proj.weight", values)
	return []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			0, 0,
			-1, 0,
			3, 0,
		}, 3, 2),
		gate,
		miniMaxM2F32RawTensor(gate.Name+".scales", []float32{0.5}),
		miniMaxM2F32RawTensor(gate.Name+".biases", []float32{1}),
		up,
		miniMaxM2F32RawTensor(up.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(up.Name+".biases", []float32{0}),
		down,
		miniMaxM2F32RawTensor(down.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(down.Name+".biases", []float32{0}),
	}
}

// miniMaxM2MultiExpertPlan builds a JANGTQ plan with numExperts routed
// experts so expert-load benchmarks can drive the per-expert path at a
// realistic fan-out (MiniMax M2 routes top-k experts per token). The
// quantisation profile matches miniMaxM2SmallJANGTQPlan so the same 1-byte
// packed projection fixtures resolve.
func miniMaxM2MultiExpertPlan(t testing.TB, numExperts int) TensorPlan {
	t.Helper()
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    numExperts,
		NumExpertsPerToken: numExperts,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	return plan
}

// miniMaxM2MultiExpertFixtureTensors writes packed gate/up/down projections
// plus affine sidecars for every expert id in ids, sharing one dense router
// gate. It is the multi-expert analogue of miniMaxM2LazyExpertFixtureTensors
// for benchmarks that load several routed experts in one pass.
func miniMaxM2MultiExpertFixtureTensors(t testing.TB, numExperts int, ids []int, values []uint8) []miniMaxM2RawSafetensor {
	t.Helper()
	gateRows := make([]float32, 0, numExperts*2)
	for e := 0; e < numExperts; e++ {
		gateRows = append(gateRows, float32(e), 0)
	}
	tensors := []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", gateRows, numExperts, 2),
	}
	for _, id := range ids {
		prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", id)
		gate := miniMaxM2PackedRawTensor(t, prefix+".gate_proj.weight", values)
		up := miniMaxM2PackedRawTensor(t, prefix+".up_proj.weight", values)
		down := miniMaxM2PackedRawTensor(t, prefix+".down_proj.weight", values)
		tensors = append(tensors,
			gate,
			miniMaxM2F32RawTensor(gate.Name+".scales", []float32{0.5}),
			miniMaxM2F32RawTensor(gate.Name+".biases", []float32{1}),
			up,
			miniMaxM2F32RawTensor(up.Name+".scales", []float32{1}),
			miniMaxM2F32RawTensor(up.Name+".biases", []float32{0}),
			down,
			miniMaxM2F32RawTensor(down.Name+".scales", []float32{1}),
			miniMaxM2F32RawTensor(down.Name+".biases", []float32{0}),
		)
	}
	return tensors
}

type miniMaxM2RawSafetensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

func miniMaxM2PackedRawTensor(t testing.TB, name string, values []uint8) miniMaxM2RawSafetensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:        name,
		Shape:       []uint64{2, 2},
		Elements:    4,
		Bits:        2,
		GroupSize:   4,
		PackedBytes: 1,
		ScaleCount:  1,
		BiasCount:   1,
	}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues() error = %v", err)
	}
	return miniMaxM2RawSafetensor{Name: name, DType: "U8", Shape: []int{len(packed)}, Raw: packed}
}

func writeMiniMaxM2PackedSafetensors(t *testing.T, path string, tensors []miniMaxM2RawSafetensor) {
	t.Helper()
	withSidecars := make([]miniMaxM2RawSafetensor, 0, len(tensors)*3)
	for _, tensor := range tensors {
		withSidecars = append(withSidecars, tensor)
		withSidecars = append(withSidecars,
			miniMaxM2F32RawTensor(tensor.Name+".scales", []float32{1}),
			miniMaxM2F32RawTensor(tensor.Name+".biases", []float32{0}),
		)
	}
	writeMiniMaxM2RawSafetensors(t, path, withSidecars)
}

func miniMaxM2F32RawTensor(name string, values []float32, shape ...int) miniMaxM2RawSafetensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	if len(shape) == 0 {
		shape = []int{len(values)}
	}
	return miniMaxM2RawSafetensor{Name: name, DType: "F32", Shape: append([]int(nil), shape...), Raw: raw}
}

func writeMiniMaxM2RawSafetensors(t testing.TB, path string, tensors []miniMaxM2RawSafetensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.Raw...)
		header[tensor.Name] = entry{
			DType:       tensor.DType,
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func miniMaxM2PackedExpertFixture(t testing.TB, gateValues, upValues, downValues []uint8) PackedExpertWeights {
	t.Helper()
	return PackedExpertWeights{
		GateProj: miniMaxM2PackedProjectionFixture(t, "gate_proj", gateValues),
		UpProj:   miniMaxM2PackedProjectionFixture(t, "up_proj", upValues),
		DownProj: miniMaxM2PackedProjectionFixture(t, "down_proj", downValues),
	}
}

func miniMaxM2PackedProjectionFixture(t testing.TB, projection string, values []uint8) JANGPackedProjectionTensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0." + projection + ".weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{2, 2},
		Elements:      4,
		Bits:          2,
		GroupSize:     4,
		Groups:        1,
		PackedBytes:   1,
		ValuesPerByte: 4,
		ScaleCount:    1,
		BiasCount:     1,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues(%s) error = %v", projection, err)
	}
	return JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1},
		Biases:     []float32{0},
	}
}

func miniMaxM2PackedDispatchReference(t *testing.T, hidden [][]float32, decisions []RouterDecision, experts map[int]PackedExpertWeights) [][]float32 {
	t.Helper()
	out := make([][]float32, len(hidden))
	for _, decision := range decisions {
		for i, expertID := range decision.ExpertIDs {
			expertOut := miniMaxM2PackedExpertReference(t, hidden[decision.TokenIndex], experts[expertID])
			if out[decision.TokenIndex] == nil {
				out[decision.TokenIndex] = make([]float32, len(expertOut))
			}
			for j, value := range expertOut {
				out[decision.TokenIndex][j] += decision.Weights[i] * value
			}
		}
	}
	return out
}

func miniMaxM2PackedExpertReference(t *testing.T, hidden []float32, expert PackedExpertWeights) []float32 {
	t.Helper()
	gate := miniMaxM2PackedProjectionReference(t, hidden, expert.GateProj)
	up := miniMaxM2PackedProjectionReference(t, hidden, expert.UpProj)
	if len(gate) != len(up) {
		t.Fatalf("gate len = %d, up len = %d", len(gate), len(up))
	}
	activated := make([]float32, len(gate))
	for i := range gate {
		activated[i] = float32(float64(gate[i])/(1+math.Exp(float64(-gate[i])))) * up[i]
	}
	return miniMaxM2PackedProjectionReference(t, activated, expert.DownProj)
}

func miniMaxM2PackedProjectionReference(t *testing.T, input []float32, projection JANGPackedProjectionTensor) []float32 {
	t.Helper()
	weight, err := jang.DequantizePackedTensor(projection.Descriptor, projection.Packed, projection.Scales, projection.Biases)
	if err != nil {
		t.Fatalf("jang.DequantizePackedTensor() error = %v", err)
	}
	outDim := int(projection.Descriptor.Shape[0])
	inDim := int(projection.Descriptor.Shape[1])
	return denseProjectionReference(input, 1, weight, outDim, inDim, projection.Bias)
}

// --- ResolvedTensor.EstimatedBytes ---

func TestM2_ResolvedTensor_EstimatedBytes_Good(t *testing.T) {
	// Dense path: f32 (4 B) over a 2x3 shape = 24 B. Exercises dTypeBytes.
	tensor := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}}
	if got := tensor.EstimatedBytes(); got != 24 {
		t.Fatalf("EstimatedBytes() = %d, want 24 (f32 4B x 6 elems)", got)
	}
	// Packed tensors report their packed byte count and ignore dtype/shape.
	packed := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}, PackedBytes: 7}
	if got := packed.EstimatedBytes(); got != 7 {
		t.Fatalf("EstimatedBytes() = %d, want 7 (packed byte count wins)", got)
	}
}

func TestM2_ResolvedTensor_EstimatedBytes_Bad(t *testing.T) {
	// Unknown dtype has zero byte width, so a dense estimate is unknowable -> 0.
	tensor := ResolvedTensor{DType: "wat", Shape: []uint64{2, 3}}
	if got := tensor.EstimatedBytes(); got != 0 {
		t.Fatalf("EstimatedBytes() = %d, want 0 for unknown dtype", got)
	}
}

func TestM2_ResolvedTensor_EstimatedBytes_Ugly(t *testing.T) {
	// Degenerate metadata: empty shape and a zero dimension both collapse to 0
	// even though the dtype is known.
	empty := ResolvedTensor{DType: "f32"}
	if got := empty.EstimatedBytes(); got != 0 {
		t.Fatalf("EstimatedBytes() = %d, want 0 for empty shape", got)
	}
	zeroDim := ResolvedTensor{DType: "f32", Shape: []uint64{2, 0, 3}}
	if got := zeroDim.EstimatedBytes(); got != 0 {
		t.Fatalf("EstimatedBytes() = %d, want 0 when a dimension is zero", got)
	}
}

// --- LayerForwardSkeleton.EstimatedBytes ---

func TestM2_LayerForwardSkeleton_EstimatedBytes_Good(t *testing.T) {
	bias := ResolvedTensor{DType: "f32", Shape: []uint64{3}} // 12 B
	skeleton := LayerForwardSkeleton{
		Layer:      0,
		RouterGate: ResolvedTensor{DType: "f32", Shape: []uint64{3, 4}}, // 48 B
		Attention: []ResolvedTensor{
			{PackedBytes: 16}, // packed, 16 B
			{PackedBytes: 8},  // packed, 8 B
		},
		RouterBias: &bias,
	}
	// 48 (gate) + 16 + 8 (attention) + 12 (bias) = 84.
	if got := skeleton.EstimatedBytes(); got != 84 {
		t.Fatalf("EstimatedBytes() = %d, want 84 (gate+attention+bias)", got)
	}
}

func TestM2_LayerForwardSkeleton_EstimatedBytes_Bad(t *testing.T) {
	// A skeleton whose every tensor has unknown/absent metadata sums to 0
	// rather than over-reporting.
	skeleton := LayerForwardSkeleton{
		RouterGate: ResolvedTensor{DType: "mystery", Shape: []uint64{3, 4}},
		Attention:  []ResolvedTensor{{DType: "mystery", Shape: []uint64{2}}},
	}
	if got := skeleton.EstimatedBytes(); got != 0 {
		t.Fatalf("EstimatedBytes() = %d, want 0 for unknown-dtype tensors", got)
	}
}

func TestM2_LayerForwardSkeleton_EstimatedBytes_Ugly(t *testing.T) {
	// Degenerate skeleton: no attention tensors and a nil router bias. Only the
	// router gate contributes, and the nil-bias branch must not panic.
	skeleton := LayerForwardSkeleton{
		RouterGate: ResolvedTensor{DType: "f32", Shape: []uint64{1, 2}}, // 8 B
	}
	if got := skeleton.EstimatedBytes(); got != 8 {
		t.Fatalf("EstimatedBytes() = %d, want 8 (router gate only, nil bias)", got)
	}
}

// --- ParseConfig (Bad/Ugly; Good already covered above) ---

func TestM2_ParseConfig_Bad(t *testing.T) {
	// Malformed JSON must surface the unmarshal error, not a zero Config.
	_, err := ParseConfig([]byte("{not valid json"))
	if err == nil {
		t.Fatal("ParseConfig() error = nil, want JSON parse error")
	}
}

func TestM2_ParseConfig_Ugly(t *testing.T) {
	// Empty-but-valid JSON: no scoring_func provided, so ParseConfig fills the
	// sigmoid default and leaves the model type empty.
	cfg, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	if cfg.ScoringFunc != "sigmoid" {
		t.Fatalf("ScoringFunc = %q, want defaulted sigmoid", cfg.ScoringFunc)
	}
	if cfg.ModelType != "" {
		t.Fatalf("ModelType = %q, want empty for config without architecture", cfg.ModelType)
	}
}

// --- BuildTensorPlan (Bad/Ugly; Good already covered above) ---

func TestM2_BuildTensorPlan_Bad(t *testing.T) {
	// Wrong architecture and no architectures list -> rejected up front.
	_, err := BuildTensorPlan(Config{ModelType: "llama"}, testJANGTQInfo())
	if err == nil || !core.Contains(err.Error(), "minimax_m2") {
		t.Fatalf("error = %v, want minimax_m2 architecture diagnostic", err)
	}
}

func TestM2_BuildTensorPlan_Ugly(t *testing.T) {
	base := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   4,
		NumHiddenLayers:    1,
		NumLocalExperts:    8,
		NumExpertsPerToken: 2,
	}

	// Missing shape sizes.
	noShape := base
	noShape.HiddenSize = 0
	if _, err := BuildTensorPlan(noShape, testJANGTQInfo()); err == nil || !core.Contains(err.Error(), "hidden") {
		t.Fatalf("error = %v, want hidden/intermediate/layer size diagnostic", err)
	}

	// Missing MoE expert counts.
	noExperts := base
	noExperts.NumLocalExperts = 0
	if _, err := BuildTensorPlan(noExperts, testJANGTQInfo()); err == nil || !core.Contains(err.Error(), "expert counts") {
		t.Fatalf("error = %v, want MoE expert count diagnostic", err)
	}

	// top-k greater than the expert pool.
	tooManyTopK := base
	tooManyTopK.NumExpertsPerToken = 9
	if _, err := BuildTensorPlan(tooManyTopK, testJANGTQInfo()); err == nil || !core.Contains(err.Error(), "top-k") {
		t.Fatalf("error = %v, want top-k exceeds expert count diagnostic", err)
	}

	// nil JANG info is tolerated: BuildTensorPlan synthesises a default profile.
	okNilInfo := base
	plan, err := BuildTensorPlan(okNilInfo, nil)
	if err != nil {
		t.Fatalf("BuildTensorPlan(nil info) error = %v", err)
	}
	if plan.JANG == nil || plan.Quantization == nil {
		t.Fatalf("plan = %+v, want synthesised default JANG profile", plan)
	}
}

// --- RouteTokens (Bad/Ugly; Good already covered above) ---

func miniMaxM2Index(names ...string) safetensors.Index {
	tensors := make(map[string]safetensors.TensorRef, len(names))
	for _, name := range names {
		tensors[name] = safetensors.TensorRef{Name: name, Path: "synthetic", DType: "U8"}
	}
	return safetensors.Index{Path: "synthetic", Tensors: tensors, Names: names}
}

func TestMiniMaxM2_DTypeBytes_AllWidths(t *testing.T) {
	cases := map[string]int{
		"u8": 1, "INT8": 1,
		"f16": 2, "BF16": 2, "u16": 2,
		"f32": 4, "I32": 4,
		"f64": 8, "UINT64": 8,
		"weird": 0, "": 0, // unknown dtype → 0 (the EstimatedBytes "give up" signal)
	}
	for dtype, want := range cases {
		if got := dTypeBytes(dtype); got != want {
			t.Fatalf("dTypeBytes(%q) = %d, want %d", dtype, got, want)
		}
	}
}

func TestMiniMaxM2_PackedAndFloatDType(t *testing.T) {
	if !packedDType("uint8") || !packedDType("U8") {
		t.Fatal("packedDType should accept the U8 / UINT8 packed dtypes (case-insensitive)")
	}
	if packedDType("f16") {
		t.Fatal("packedDType should reject a float dtype")
	}
	if !floatDType("bf16") || !floatDType("F32") {
		t.Fatal("floatDType should accept the float dtypes (case-insensitive)")
	}
	if floatDType("u8") {
		t.Fatal("floatDType should reject a packed integer dtype")
	}
}

func miniMaxM2StringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestM2_TensorPlan_LayerTensorSpecs_Bad(t *testing.T) {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 4, IntermediateSize: 8,
		NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 4, NumExpertsPerToken: 2,
	}, nil)
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	// Layer out of range (only layer 0 exists). Also lifts ValidateTensorNames,
	// which surfaces this same error from its LayerTensorSpecs(0,0) call when
	// the plan has zero layers — here we hit it directly.
	if _, err := plan.LayerTensorSpecs(5, 0); err == nil || !core.Contains(err.Error(), "layer 5 out of range") {
		t.Fatalf("LayerTensorSpecs(5,0) err = %v, want layer-range diagnostic", err)
	}
	if _, err := plan.LayerTensorSpecs(-1, 0); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("LayerTensorSpecs(-1,0) err = %v, want layer-range diagnostic", err)
	}
	// Expert out of range (only experts 0..3 exist).
	if _, err := plan.LayerTensorSpecs(0, 99); err == nil || !core.Contains(err.Error(), "expert 99 out of range") {
		t.Fatalf("LayerTensorSpecs(0,99) err = %v, want expert-range diagnostic", err)
	}
}

func TestM2_TensorPlan_ValidateTensorNames_Ugly(t *testing.T) {
	// A plan with zero layers makes the internal LayerTensorSpecs(0,0) fail;
	// ValidateTensorNames must propagate that range error, not mask it.
	plan := TensorPlan{Config: Config{ModelType: "minimax_m2", NumHiddenLayers: 0, NumLocalExperts: 4}}
	if err := plan.ValidateTensorNames(map[string]bool{}); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("ValidateTensorNames(zero layers) err = %v, want propagated range error", err)
	}
}

func TestMiniMaxM2_SameUint64Slice(t *testing.T) {
	if !sameUint64Slice([]uint64{1, 2, 3}, []uint64{1, 2, 3}) {
		t.Fatal("sameUint64Slice should report equal slices as equal")
	}
	if sameUint64Slice([]uint64{1, 2}, []uint64{1, 2, 3}) {
		t.Fatal("sameUint64Slice should reject a length mismatch")
	}
	if sameUint64Slice([]uint64{1, 2, 3}, []uint64{1, 9, 3}) {
		t.Fatal("sameUint64Slice should reject a value mismatch")
	}
}
