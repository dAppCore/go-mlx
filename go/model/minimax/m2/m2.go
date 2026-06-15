// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
)

// Config captures the config fields needed before the native sparse
// kernels exist: routing shape, attention shape, MTP flags, and tensor mapping.
type Config struct {
	ModelType            string   `json:"model_type,omitempty"`
	Architectures        []string `json:"architectures,omitempty"`
	VocabSize            int      `json:"vocab_size,omitempty"`
	HiddenSize           int      `json:"hidden_size,omitempty"`
	IntermediateSize     int      `json:"intermediate_size,omitempty"`
	NumHiddenLayers      int      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads    int      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads     int      `json:"num_key_value_heads,omitempty"`
	HeadDim              int      `json:"head_dim,omitempty"`
	ContextLength        int      `json:"max_position_embeddings,omitempty"`
	NumLocalExperts      int      `json:"num_local_experts,omitempty"`
	NumExpertsPerToken   int      `json:"num_experts_per_tok,omitempty"`
	ScoringFunc          string   `json:"scoring_func,omitempty"`
	UseRoutingBias       bool     `json:"use_routing_bias,omitempty"`
	UseMTP               bool     `json:"use_mtp,omitempty"`
	NumMTPModules        int      `json:"num_mtp_modules,omitempty"`
	MTPTransformerLayers int      `json:"mtp_transformer_layers,omitempty"`
	UseQKNorm            bool     `json:"use_qk_norm,omitempty"`
	RotaryDim            int      `json:"rotary_dim,omitempty"`
	RopeTheta            float64  `json:"rope_theta,omitempty"`
}

// TensorRole identifies one expected MiniMax M2 tensor slot.
type TensorRole string

const (
	TensorRoleAttentionQ TensorRole = "attention.q_proj"
	TensorRoleAttentionK TensorRole = "attention.k_proj"
	TensorRoleAttentionV TensorRole = "attention.v_proj"
	TensorRoleAttentionO TensorRole = "attention.o_proj"
	TensorRoleRouterGate TensorRole = "router.gate"
	TensorRoleRouterBias TensorRole = "router.e_score_correction_bias"
	TensorRoleExpertGate TensorRole = "expert.gate_proj"
	TensorRoleExpertUp   TensorRole = "expert.up_proj"
	TensorRoleExpertDown TensorRole = "expert.down_proj"
)

// TensorSpec is one canonical tensor expectation plus compatible
// checkpoint aliases observed in MiniMax M2 loaders.
type TensorSpec struct {
	Name    string                       `json:"name"`
	Aliases []string                     `json:"aliases,omitempty"`
	Role    TensorRole                   `json:"role"`
	Layer   int                          `json:"layer,omitempty"`
	Expert  int                          `json:"expert,omitempty"`
	Shape   []uint64                     `json:"shape,omitempty"`
	DType   string                       `json:"dtype,omitempty"`
	Packed  *jang.PackedTensorDescriptor `json:"packed,omitempty"`
}

// TensorPlan keeps the model-wide mapping knobs and JANG layout.
type TensorPlan struct {
	Config       Config              `json:"config"`
	Quantization *jang.PackedProfile `json:"quantization,omitempty"`
	JANG         *jang.Info          `json:"jang,omitempty"`
}

// RouterDecision is a deterministic top-k route for one token.
type RouterDecision struct {
	TokenIndex int       `json:"token_index"`
	ExpertIDs  []int     `json:"expert_ids"`
	Weights    []float32 `json:"weights"`
}

// ExpertFunc is a fake expert used by fixture dispatch tests and
// future backend parity checks.
//
// Contract: an implementation MUST treat its input row as READ-ONLY and
// MUST NOT mutate it. DispatchExperts makes one defensive copy per token
// and shares that single copy across every expert routed to that token
// (the per-token arena, not a per-expert clone), so a mutating expert
// would pollute the row seen by the next expert for the same token. The
// return slice is freshly owned by the expert and may be retained.
type ExpertFunc func([]float32) []float32

// JANGPackedProjectionTensor is a host-side packed projection payload. It keeps
// the descriptor separate from raw bytes so native backends can validate shape
// and quantisation metadata before dispatch.
type JANGPackedProjectionTensor struct {
	Descriptor jang.PackedTensorDescriptor `json:"descriptor"`
	Packed     []byte                      `json:"-"`
	Scales     []float32                   `json:"-"`
	Biases     []float32                   `json:"-"`
	Bias       []float32                   `json:"bias,omitempty"`
}

// PackedExpertWeights holds one routed expert's SwiGLU projections in
// packed JANG/JANGTQ form.
type PackedExpertWeights struct {
	GateProj JANGPackedProjectionTensor `json:"gate_proj"`
	UpProj   JANGPackedProjectionTensor `json:"up_proj"`
	DownProj JANGPackedProjectionTensor `json:"down_proj"`
}

// RouterWeights holds the dense router projection for one MiniMax M2
// MoE layer. Weight is laid out as [num_experts, hidden_size].
type RouterWeights struct {
	Name       string    `json:"name,omitempty"`
	Weight     []float32 `json:"-"`
	Bias       []float32 `json:"-"`
	NumExperts int       `json:"num_experts,omitempty"`
	HiddenSize int       `json:"hidden_size,omitempty"`
}

// PackedLayerForwardOptions configures the native packed MoE layer
// skeleton used during MiniMax M2 bring-up.
type PackedLayerForwardOptions struct {
	Plan         TensorPlan  `json:"plan"`
	WeightFiles  []string    `json:"weight_files,omitempty"`
	Layer        int         `json:"layer,omitempty"`
	Hidden       [][]float32 `json:"hidden,omitempty"`
	RouterScores [][]float32 `json:"router_scores,omitempty"`
	RouterBias   []float32   `json:"router_bias,omitempty"`
	TokenIDs     []int32     `json:"token_ids,omitempty"`
	ProbeSink    probe.Sink  `json:"-"`
}

// PackedLayerForwardResult reports a routed packed expert layer pass.
type PackedLayerForwardResult struct {
	Output            [][]float32      `json:"output"`
	Decisions         []RouterDecision `json:"decisions,omitempty"`
	SelectedExpertIDs []int            `json:"selected_expert_ids,omitempty"`
	LoadedPackedBytes uint64           `json:"loaded_packed_bytes,omitempty"`
	ProbeEvents       []probe.Event    `json:"probe_events,omitempty"`
}

// LazyExpertLoad is the result of routing hidden states and loading
// only the routed packed experts from safetensors.
type LazyExpertLoad struct {
	Layer             int                         `json:"layer"`
	Router            RouterWeights               `json:"router"`
	Scores            [][]float32                 `json:"scores,omitempty"`
	Decisions         []RouterDecision            `json:"decisions,omitempty"`
	SelectedExpertIDs []int                       `json:"selected_expert_ids,omitempty"`
	Experts           map[int]PackedExpertWeights `json:"experts,omitempty"`
	LoadedPackedBytes uint64                      `json:"loaded_packed_bytes,omitempty"`
	ProbeEvents       []probe.Event               `json:"probe_events,omitempty"`
}

// DenseProjectionTensor is a dequantized host-side projection. It is
// a reference/runtime bridge until native fused kernels consume packed payloads
// directly.
type DenseProjectionTensor struct {
	Descriptor jang.PackedTensorDescriptor `json:"descriptor"`
	Weight     []float32                   `json:"-"`
	Bias       []float32                   `json:"bias,omitempty"`
}

// DenseExpertWeights holds dequantized routed expert projections.
type DenseExpertWeights struct {
	GateProj DenseProjectionTensor `json:"gate_proj"`
	UpProj   DenseProjectionTensor `json:"up_proj"`
	DownProj DenseProjectionTensor `json:"down_proj"`
}

// ResolvedTensor is a safetensors-backed tensor slot resolved for a
// layer skeleton. Shape is the on-disk physical shape; LogicalShape is the
// model-space matrix shape the forward path expects after dequantisation.
type ResolvedTensor struct {
	Name         string     `json:"name"`
	Role         TensorRole `json:"role"`
	Layer        int        `json:"layer,omitempty"`
	DType        string     `json:"dtype,omitempty"`
	Shape        []uint64   `json:"shape,omitempty"`
	LogicalShape []uint64   `json:"logical_shape,omitempty"`
	PackedBytes  int        `json:"packed_bytes,omitempty"`
}

// LayerForwardSkeleton resolves the first pieces a native MiniMax M2
// forward pass needs before full execution: attention projections and the MoE
// router gate/bias. It reads safetensors headers only.
type LayerForwardSkeleton struct {
	Layer      int              `json:"layer"`
	Attention  []ResolvedTensor `json:"attention,omitempty"`
	RouterGate ResolvedTensor   `json:"router_gate"`
	RouterBias *ResolvedTensor  `json:"router_bias,omitempty"`
}

// EstimatedBytes returns the on-disk bytes represented by this resolved tensor
// metadata. Packed tensors report their packed byte count; dense tensors use
// dtype width times shape elements.
func (tensor ResolvedTensor) EstimatedBytes() uint64 {
	if tensor.PackedBytes > 0 {
		return uint64(tensor.PackedBytes)
	}
	bytesPerElement := dTypeBytes(tensor.DType)
	if bytesPerElement == 0 || len(tensor.Shape) == 0 {
		return 0
	}
	elements := uint64(1)
	for _, dim := range tensor.Shape {
		if dim == 0 {
			return 0
		}
		elements *= dim
	}
	return elements * uint64(bytesPerElement)
}

// EstimatedBytes returns the first-layer attention/router bytes proven by the
// skeleton. It is deliberately metadata-only and does not read tensor payloads.
func (skeleton LayerForwardSkeleton) EstimatedBytes() uint64 {
	total := skeleton.RouterGate.EstimatedBytes()
	// Index iteration: ResolvedTensor is 112 B, above the value-copy
	// threshold. Range-by-value would copy each Attention entry per step.
	for i := range skeleton.Attention {
		total += skeleton.Attention[i].EstimatedBytes()
	}
	if skeleton.RouterBias != nil {
		total += skeleton.RouterBias.EstimatedBytes()
	}
	return total
}

// ParseConfig reads the subset of config.json needed for the native
// loader plan and fake routing path.
func ParseConfig(data []byte) (Config, error) {
	var cfg Config
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return Config{}, result.Value.(error)
	}
	cfg.ModelType = profile.NormalizeArchitecture(firstNonEmpty(cfg.ModelType, firstArchitecture(cfg.Architectures)))
	if cfg.ScoringFunc == "" {
		cfg.ScoringFunc = "sigmoid"
	}
	return cfg, nil
}

// BuildTensorPlan creates a model-wide tensor mapping plan.
func BuildTensorPlan(cfg Config, info *jang.Info) (TensorPlan, error) {
	if profile.NormalizeArchitecture(cfg.ModelType) != "minimax_m2" && firstArchitecture(cfg.Architectures) == "" {
		return TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires minimax_m2 architecture")
	}
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires hidden/intermediate/layer sizes")
	}
	if cfg.NumLocalExperts <= 0 || cfg.NumExpertsPerToken <= 0 {
		return TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires MoE expert counts")
	}
	if cfg.NumExpertsPerToken > cfg.NumLocalExperts {
		return TensorPlan{}, core.NewError("mlx: MiniMax M2 top-k experts cannot exceed local expert count")
	}
	if info == nil {
		info = &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 64, BitsDefault: 2, AttentionBits: 8, RoutedExpertBits: 2}
	}
	info = cloneJANGQuantizationInfo(info)
	info.Packed = jang.BuildPackedProfile(info)
	return TensorPlan{
		Config:       cfg,
		Quantization: jang.ClonePackedProfile(info.Packed),
		JANG:         info,
	}, nil
}

// LayerTensorSpecs returns the expected tensors for one layer and one routed
// expert. Full native loading can iterate experts without materialising all
// 62*256 expert specs up front.
func (plan TensorPlan) LayerTensorSpecs(layer, expert int) ([]TensorSpec, error) {
	if layer < 0 || layer >= plan.Config.NumHiddenLayers {
		return nil, core.NewError(core.Concat("mlx: MiniMax M2 layer ", core.Itoa(layer), " out of range"))
	}
	if expert < 0 || expert >= plan.Config.NumLocalExperts {
		return nil, core.NewError(core.Concat("mlx: MiniMax M2 expert ", core.Itoa(expert), " out of range"))
	}
	layerPrefix := core.Concat("model.layers.", core.Itoa(layer), ".")
	// Pre-size to 9 (8 always + 1 optional routing bias). The previous
	// 8-element literal followed by append-when-UseRoutingBias forced
	// a grow + copy of 8×TensorSpec (8×120 B = 960 B copied per call).
	specs := make([]TensorSpec, 0, 9)
	specs = append(specs,
		plan.attentionSpec(layer, "q_proj", TensorRoleAttentionQ),
		plan.attentionSpec(layer, "k_proj", TensorRoleAttentionK),
		plan.attentionSpec(layer, "v_proj", TensorRoleAttentionV),
		plan.attentionSpec(layer, "o_proj", TensorRoleAttentionO),
		TensorSpec{
			Name:  core.Concat(layerPrefix, "block_sparse_moe.gate.weight"),
			Role:  TensorRoleRouterGate,
			Layer: layer,
			Shape: []uint64{uint64(plan.Config.NumLocalExperts), uint64(plan.Config.HiddenSize)},
			DType: "f32",
		},
		plan.expertSpec(layer, expert, "gate_proj", TensorRoleExpertGate),
		plan.expertSpec(layer, expert, "up_proj", TensorRoleExpertUp),
		plan.expertSpec(layer, expert, "down_proj", TensorRoleExpertDown),
	)
	if plan.Config.UseRoutingBias {
		specs = append(specs, TensorSpec{
			Name:  core.Concat(layerPrefix, "block_sparse_moe.e_score_correction_bias"),
			Role:  TensorRoleRouterBias,
			Layer: layer,
			Shape: []uint64{uint64(plan.Config.NumLocalExperts)},
			DType: "f32",
		})
	}
	return specs, nil
}

// ValidateTensorNames reports whether the required first-layer/first-expert
// tensors are present, accepting canonical names and aliases.
func (plan TensorPlan) ValidateTensorNames(names map[string]bool) error {
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		return err
	}
	// Index iteration: TensorSpec is 120 B (well above the value-copy
	// threshold), so range-by-value would copy 120 B per spec.
	var missing []string
	for i := range specs {
		spec := &specs[i]
		if specMatchesName(spec, names) {
			continue
		}
		missing = append(missing, spec.Name)
	}
	if len(missing) > 0 {
		return core.NewError("mlx: MiniMax M2 tensor plan missing required tensors: " + core.Join(", ", missing...))
	}
	return nil
}

func (plan TensorPlan) attentionSpec(layer int, projection string, role TensorRole) TensorSpec {
	name := core.Concat("model.layers.", core.Itoa(layer), ".self_attn.", projection, ".weight")
	qSize := firstPositive(plan.Config.NumAttentionHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	kvSize := firstPositive(plan.Config.NumKeyValueHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	// One shape literal per call. The default was previously allocated up
	// front then overwritten for every attention role (Q/K/V/O), wasting one
	// []uint64 alloc on the dominant path; the default branch keeps the
	// {hidden, hidden} fallback byte-identical for any other role.
	var shape []uint64
	switch role {
	case TensorRoleAttentionQ:
		shape = []uint64{uint64(qSize), uint64(plan.Config.HiddenSize)}
	case TensorRoleAttentionK, TensorRoleAttentionV:
		shape = []uint64{uint64(kvSize), uint64(plan.Config.HiddenSize)}
	case TensorRoleAttentionO:
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(qSize)}
	default:
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.HiddenSize)}
	}
	spec := TensorSpec{
		Name:    name,
		Aliases: attentionAliases(layer, projection, role),
		Role:    role,
		Layer:   layer,
		Shape:   shape,
	}
	if packed, err := jang.NewPackedTensorDescriptor(name, shape, plan.JANG); err == nil {
		spec.Packed = &packed
	}
	return spec
}

func attentionAliases(layer int, projection string, role TensorRole) []string {
	switch role {
	case TensorRoleAttentionQ, TensorRoleAttentionK, TensorRoleAttentionV:
		return []string{core.Concat("model.layers.", core.Itoa(layer), ".self_attn.qkv_proj.weight")}
	default:
		return nil
	}
}

func (plan TensorPlan) expertSpec(layer, expert int, projection string, role TensorRole) TensorSpec {
	layerStr := core.Itoa(layer)
	expertStr := core.Itoa(expert)
	name := core.Concat("model.layers.", layerStr, ".block_sparse_moe.experts.", expertStr, ".", projection, ".weight")
	// One shape literal per call: down_proj transposes the gate/up shape. The
	// previous form allocated the gate/up literal then overwrote it for
	// down_proj, wasting one []uint64 alloc on that branch.
	var shape []uint64
	if projection == "down_proj" {
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.IntermediateSize)}
	} else {
		shape = []uint64{uint64(plan.Config.IntermediateSize), uint64(plan.Config.HiddenSize)}
	}
	spec := TensorSpec{
		Name:    name,
		Aliases: []string{core.Concat("model.layers.", layerStr, ".mlp.experts.", expertStr, ".", projection, ".weight")},
		Role:    role,
		Layer:   layer,
		Expert:  expert,
		Shape:   shape,
	}
	if packed, err := jang.NewPackedTensorDescriptor(name, shape, plan.JANG); err == nil {
		spec.Packed = &packed
	}
	return spec
}

func firstArchitecture(values []string) string {
	for _, value := range values {
		if profile.ArchitectureID(value) == "minimax_m2" {
			return "minimax_m2"
		}
	}
	return ""
}

func cloneJANGQuantizationInfo(info *jang.Info) *jang.Info {
	if info == nil {
		return nil
	}
	cloned := *info
	cloned.Packed = jang.ClonePackedProfile(info.Packed)
	return &cloned
}

func specMatchesName(spec *TensorSpec, names map[string]bool) bool {
	if names[spec.Name] {
		return true
	}
	for _, alias := range spec.Aliases {
		if names[alias] {
			return true
		}
	}
	return false
}

// findTensorSpec returns the spec for the requested role, or the zero
// value. Index iteration + pointer return avoids copying the 120 B
// TensorSpec value-by-value on each step of the scan.
func findTensorSpec(specs []TensorSpec, role TensorRole) TensorSpec {
	for i := range specs {
		if specs[i].Role == role {
			return specs[i]
		}
	}
	return TensorSpec{}
}

func decisionExpertIDs(decisions []RouterDecision) []int {
	// Index iteration: RouterDecision is 56 B, range-by-value would
	// copy each decision per step.
	total := 0
	for d := range decisions {
		total += len(decisions[d].ExpertIDs)
	}
	ids := make([]int, 0, total)
	for d := range decisions {
		ids = append(ids, decisions[d].ExpertIDs...)
	}
	return ids
}

func decisionExpertIDsSorted(decisions []RouterDecision) []int {
	return uniqueExpertIDs(decisionExpertIDs(decisions))
}

func packedExpertLoadedBytes(experts map[int]PackedExpertWeights) uint64 {
	total := uint64(0)
	for _, expert := range experts {
		total += uint64(len(expert.GateProj.Packed))
		total += uint64(len(expert.UpProj.Packed))
		total += uint64(len(expert.DownProj.Packed))
	}
	return total
}

func uniqueExpertIDs(ids []int) []int {
	seen := make(map[int]bool, len(ids))
	out := make([]int, 0, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	sort.Ints(out)
	return out
}

func packedDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "U8", "UINT8":
		return true
	default:
		return false
	}
}

func floatDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "F16", "BF16", "F32", "F64":
		return true
	default:
		return false
	}
}

func dTypeBytes(dtype string) int {
	switch core.Upper(dtype) {
	case "U8", "I8", "UINT8", "INT8":
		return 1
	case "F16", "BF16", "I16", "U16", "INT16", "UINT16":
		return 2
	case "F32", "I32", "U32", "INT32", "UINT32":
		return 4
	case "F64", "I64", "U64", "INT64", "UINT64":
		return 8
	default:
		return 0
	}
}

func sameUint64Slice(a, b []uint64) bool {
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
