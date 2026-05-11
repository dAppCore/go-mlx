// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"math"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/profile"
)

// MiniMaxM2Config captures the config fields needed before the native sparse
// kernels exist: routing shape, attention shape, MTP flags, and tensor mapping.
type MiniMaxM2Config struct {
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

// MiniMaxM2TensorRole identifies one expected MiniMax M2 tensor slot.
type MiniMaxM2TensorRole string

const (
	MiniMaxM2TensorRoleAttentionQ MiniMaxM2TensorRole = "attention.q_proj"
	MiniMaxM2TensorRoleAttentionK MiniMaxM2TensorRole = "attention.k_proj"
	MiniMaxM2TensorRoleAttentionV MiniMaxM2TensorRole = "attention.v_proj"
	MiniMaxM2TensorRoleAttentionO MiniMaxM2TensorRole = "attention.o_proj"
	MiniMaxM2TensorRoleRouterGate MiniMaxM2TensorRole = "router.gate"
	MiniMaxM2TensorRoleRouterBias MiniMaxM2TensorRole = "router.e_score_correction_bias"
	MiniMaxM2TensorRoleExpertGate MiniMaxM2TensorRole = "expert.gate_proj"
	MiniMaxM2TensorRoleExpertUp   MiniMaxM2TensorRole = "expert.up_proj"
	MiniMaxM2TensorRoleExpertDown MiniMaxM2TensorRole = "expert.down_proj"
)

// MiniMaxM2TensorSpec is one canonical tensor expectation plus compatible
// checkpoint aliases observed in MiniMax M2 loaders.
type MiniMaxM2TensorSpec struct {
	Name    string                      `json:"name"`
	Aliases []string                    `json:"aliases,omitempty"`
	Role    MiniMaxM2TensorRole         `json:"role"`
	Layer   int                         `json:"layer,omitempty"`
	Expert  int                         `json:"expert,omitempty"`
	Shape   []uint64                    `json:"shape,omitempty"`
	DType   string                      `json:"dtype,omitempty"`
	Packed  *jang.PackedTensorDescriptor `json:"packed,omitempty"`
}

// MiniMaxM2TensorPlan keeps the model-wide mapping knobs and JANG layout.
type MiniMaxM2TensorPlan struct {
	Config       MiniMaxM2Config                `json:"config"`
	Quantization *jang.PackedProfile `json:"quantization,omitempty"`
	JANG         *jang.Info          `json:"jang,omitempty"`
}

// MiniMaxM2RouterDecision is a deterministic top-k route for one token.
type MiniMaxM2RouterDecision struct {
	TokenIndex int       `json:"token_index"`
	ExpertIDs  []int     `json:"expert_ids"`
	Weights    []float32 `json:"weights"`
}

// MiniMaxM2ExpertFunc is a fake expert used by fixture dispatch tests and
// future backend parity checks.
type MiniMaxM2ExpertFunc func([]float32) []float32

// JANGPackedProjectionTensor is a host-side packed projection payload. It keeps
// the descriptor separate from raw bytes so native backends can validate shape
// and quantisation metadata before dispatch.
type JANGPackedProjectionTensor struct {
	Descriptor jang.PackedTensorDescriptor `json:"descriptor"`
	Packed     []byte                     `json:"-"`
	Scales     []float32                  `json:"-"`
	Biases     []float32                  `json:"-"`
	Bias       []float32                  `json:"bias,omitempty"`
}

// MiniMaxM2PackedExpertWeights holds one routed expert's SwiGLU projections in
// packed JANG/JANGTQ form.
type MiniMaxM2PackedExpertWeights struct {
	GateProj JANGPackedProjectionTensor `json:"gate_proj"`
	UpProj   JANGPackedProjectionTensor `json:"up_proj"`
	DownProj JANGPackedProjectionTensor `json:"down_proj"`
}

// MiniMaxM2RouterWeights holds the dense router projection for one MiniMax M2
// MoE layer. Weight is laid out as [num_experts, hidden_size].
type MiniMaxM2RouterWeights struct {
	Name       string    `json:"name,omitempty"`
	Weight     []float32 `json:"-"`
	Bias       []float32 `json:"-"`
	NumExperts int       `json:"num_experts,omitempty"`
	HiddenSize int       `json:"hidden_size,omitempty"`
}

// MiniMaxM2PackedLayerForwardOptions configures the native packed MoE layer
// skeleton used during MiniMax M2 bring-up.
type MiniMaxM2PackedLayerForwardOptions struct {
	Plan         MiniMaxM2TensorPlan `json:"plan"`
	WeightFiles  []string            `json:"weight_files,omitempty"`
	Layer        int                 `json:"layer,omitempty"`
	Hidden       [][]float32         `json:"hidden,omitempty"`
	RouterScores [][]float32         `json:"router_scores,omitempty"`
	RouterBias   []float32           `json:"router_bias,omitempty"`
	TokenIDs     []int32             `json:"token_ids,omitempty"`
	ProbeSink    ProbeSink           `json:"-"`
}

// MiniMaxM2PackedLayerForwardResult reports a routed packed expert layer pass.
type MiniMaxM2PackedLayerForwardResult struct {
	Output            [][]float32               `json:"output"`
	Decisions         []MiniMaxM2RouterDecision `json:"decisions,omitempty"`
	SelectedExpertIDs []int                     `json:"selected_expert_ids,omitempty"`
	LoadedPackedBytes uint64                    `json:"loaded_packed_bytes,omitempty"`
	ProbeEvents       []ProbeEvent              `json:"probe_events,omitempty"`
}

// MiniMaxM2LazyExpertLoad is the result of routing hidden states and loading
// only the routed packed experts from safetensors.
type MiniMaxM2LazyExpertLoad struct {
	Layer             int                                  `json:"layer"`
	Router            MiniMaxM2RouterWeights               `json:"router,omitempty"`
	Scores            [][]float32                          `json:"scores,omitempty"`
	Decisions         []MiniMaxM2RouterDecision            `json:"decisions,omitempty"`
	SelectedExpertIDs []int                                `json:"selected_expert_ids,omitempty"`
	Experts           map[int]MiniMaxM2PackedExpertWeights `json:"experts,omitempty"`
	LoadedPackedBytes uint64                               `json:"loaded_packed_bytes,omitempty"`
	ProbeEvents       []ProbeEvent                         `json:"probe_events,omitempty"`
}

// MiniMaxM2DenseProjectionTensor is a dequantized host-side projection. It is
// a reference/runtime bridge until native fused kernels consume packed payloads
// directly.
type MiniMaxM2DenseProjectionTensor struct {
	Descriptor jang.PackedTensorDescriptor `json:"descriptor"`
	Weight     []float32                  `json:"-"`
	Bias       []float32                  `json:"bias,omitempty"`
}

// MiniMaxM2DenseExpertWeights holds dequantized routed expert projections.
type MiniMaxM2DenseExpertWeights struct {
	GateProj MiniMaxM2DenseProjectionTensor `json:"gate_proj"`
	UpProj   MiniMaxM2DenseProjectionTensor `json:"up_proj"`
	DownProj MiniMaxM2DenseProjectionTensor `json:"down_proj"`
}

// MiniMaxM2ResolvedTensor is a safetensors-backed tensor slot resolved for a
// layer skeleton. Shape is the on-disk physical shape; LogicalShape is the
// model-space matrix shape the forward path expects after dequantisation.
type MiniMaxM2ResolvedTensor struct {
	Name         string              `json:"name"`
	Role         MiniMaxM2TensorRole `json:"role"`
	Layer        int                 `json:"layer,omitempty"`
	DType        string              `json:"dtype,omitempty"`
	Shape        []uint64            `json:"shape,omitempty"`
	LogicalShape []uint64            `json:"logical_shape,omitempty"`
	PackedBytes  int                 `json:"packed_bytes,omitempty"`
}

// MiniMaxM2LayerForwardSkeleton resolves the first pieces a native MiniMax M2
// forward pass needs before full execution: attention projections and the MoE
// router gate/bias. It reads safetensors headers only.
type MiniMaxM2LayerForwardSkeleton struct {
	Layer      int                       `json:"layer"`
	Attention  []MiniMaxM2ResolvedTensor `json:"attention,omitempty"`
	RouterGate MiniMaxM2ResolvedTensor   `json:"router_gate"`
	RouterBias *MiniMaxM2ResolvedTensor  `json:"router_bias,omitempty"`
}

// EstimatedBytes returns the on-disk bytes represented by this resolved tensor
// metadata. Packed tensors report their packed byte count; dense tensors use
// dtype width times shape elements.
func (tensor MiniMaxM2ResolvedTensor) EstimatedBytes() uint64 {
	if tensor.PackedBytes > 0 {
		return uint64(tensor.PackedBytes)
	}
	bytesPerElement := miniMaxM2DTypeBytes(tensor.DType)
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
func (skeleton MiniMaxM2LayerForwardSkeleton) EstimatedBytes() uint64 {
	total := skeleton.RouterGate.EstimatedBytes()
	for _, tensor := range skeleton.Attention {
		total += tensor.EstimatedBytes()
	}
	if skeleton.RouterBias != nil {
		total += skeleton.RouterBias.EstimatedBytes()
	}
	return total
}

// ParseMiniMaxM2Config reads the subset of config.json needed for the native
// loader plan and fake routing path.
func ParseMiniMaxM2Config(data []byte) (MiniMaxM2Config, error) {
	var cfg MiniMaxM2Config
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return MiniMaxM2Config{}, result.Value.(error)
	}
	cfg.ModelType = normalizeKnownArchitecture(firstNonEmpty(cfg.ModelType, firstMiniMaxM2Architecture(cfg.Architectures)))
	if cfg.ScoringFunc == "" {
		cfg.ScoringFunc = "sigmoid"
	}
	return cfg, nil
}

// BuildMiniMaxM2TensorPlan creates a model-wide tensor mapping plan.
func BuildMiniMaxM2TensorPlan(cfg MiniMaxM2Config, info *jang.Info) (MiniMaxM2TensorPlan, error) {
	if normalizeKnownArchitecture(cfg.ModelType) != "minimax_m2" && firstMiniMaxM2Architecture(cfg.Architectures) == "" {
		return MiniMaxM2TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires minimax_m2 architecture")
	}
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return MiniMaxM2TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires hidden/intermediate/layer sizes")
	}
	if cfg.NumLocalExperts <= 0 || cfg.NumExpertsPerToken <= 0 {
		return MiniMaxM2TensorPlan{}, core.NewError("mlx: MiniMax M2 tensor plan requires MoE expert counts")
	}
	if cfg.NumExpertsPerToken > cfg.NumLocalExperts {
		return MiniMaxM2TensorPlan{}, core.NewError("mlx: MiniMax M2 top-k experts cannot exceed local expert count")
	}
	if info == nil {
		info = &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 64, BitsDefault: 2, AttentionBits: 8, RoutedExpertBits: 2}
	}
	info = cloneJANGQuantizationInfo(info)
	info.Packed = jang.BuildPackedProfile(info)
	return MiniMaxM2TensorPlan{
		Config:       cfg,
		Quantization: jang.ClonePackedProfile(info.Packed),
		JANG:         info,
	}, nil
}

// LayerTensorSpecs returns the expected tensors for one layer and one routed
// expert. Full native loading can iterate experts without materialising all
// 62*256 expert specs up front.
func (plan MiniMaxM2TensorPlan) LayerTensorSpecs(layer, expert int) ([]MiniMaxM2TensorSpec, error) {
	if layer < 0 || layer >= plan.Config.NumHiddenLayers {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 layer %d out of range", layer))
	}
	if expert < 0 || expert >= plan.Config.NumLocalExperts {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 expert %d out of range", expert))
	}
	specs := []MiniMaxM2TensorSpec{
		plan.attentionSpec(layer, "q_proj", MiniMaxM2TensorRoleAttentionQ),
		plan.attentionSpec(layer, "k_proj", MiniMaxM2TensorRoleAttentionK),
		plan.attentionSpec(layer, "v_proj", MiniMaxM2TensorRoleAttentionV),
		plan.attentionSpec(layer, "o_proj", MiniMaxM2TensorRoleAttentionO),
		{
			Name:  core.Sprintf("model.layers.%d.block_sparse_moe.gate.weight", layer),
			Role:  MiniMaxM2TensorRoleRouterGate,
			Layer: layer,
			Shape: []uint64{uint64(plan.Config.NumLocalExperts), uint64(plan.Config.HiddenSize)},
			DType: "f32",
		},
		plan.expertSpec(layer, expert, "gate_proj", MiniMaxM2TensorRoleExpertGate),
		plan.expertSpec(layer, expert, "up_proj", MiniMaxM2TensorRoleExpertUp),
		plan.expertSpec(layer, expert, "down_proj", MiniMaxM2TensorRoleExpertDown),
	}
	if plan.Config.UseRoutingBias {
		specs = append(specs, MiniMaxM2TensorSpec{
			Name:  core.Sprintf("model.layers.%d.block_sparse_moe.e_score_correction_bias", layer),
			Role:  MiniMaxM2TensorRoleRouterBias,
			Layer: layer,
			Shape: []uint64{uint64(plan.Config.NumLocalExperts)},
			DType: "f32",
		})
	}
	return specs, nil
}

// ValidateTensorNames reports whether the required first-layer/first-expert
// tensors are present, accepting canonical names and aliases.
func (plan MiniMaxM2TensorPlan) ValidateTensorNames(names map[string]bool) error {
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		return err
	}
	missing := []string{}
	for _, spec := range specs {
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

// RouteMiniMaxM2Tokens computes deterministic top-k router decisions for a
// batch of router scores. Scores are sigmoid-normalised by default and top-k
// weights are renormalised, matching the MiniMax M2 sparse routing contract.
func RouteMiniMaxM2Tokens(cfg MiniMaxM2Config, scores [][]float32, bias []float32) ([]MiniMaxM2RouterDecision, error) {
	if cfg.NumLocalExperts <= 0 {
		return nil, core.NewError("mlx: MiniMax M2 routing requires local expert count")
	}
	topK := cfg.NumExpertsPerToken
	if topK <= 0 {
		topK = 1
	}
	if topK > cfg.NumLocalExperts {
		return nil, core.NewError("mlx: MiniMax M2 routing top-k exceeds expert count")
	}
	if len(bias) > 0 && len(bias) != cfg.NumLocalExperts {
		return nil, core.NewError("mlx: MiniMax M2 routing bias length does not match expert count")
	}
	decisions := make([]MiniMaxM2RouterDecision, 0, len(scores))
	for tokenIndex, row := range scores {
		if len(row) != cfg.NumLocalExperts {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 routing row %d has %d scores, expected %d", tokenIndex, len(row), cfg.NumLocalExperts))
		}
		scored := make([]miniMaxM2ExpertScore, 0, len(row))
		for expertID, raw := range row {
			value := raw
			if len(bias) > 0 {
				value += bias[expertID]
			}
			scored = append(scored, miniMaxM2ExpertScore{ID: expertID, Score: miniMaxM2Score(value, cfg.ScoringFunc)})
		}
		sort.SliceStable(scored, func(i, j int) bool {
			if scored[i].Score == scored[j].Score {
				return scored[i].ID < scored[j].ID
			}
			return scored[i].Score > scored[j].Score
		})
		decision := MiniMaxM2RouterDecision{TokenIndex: tokenIndex}
		total := float32(0)
		for i := 0; i < topK; i++ {
			decision.ExpertIDs = append(decision.ExpertIDs, scored[i].ID)
			decision.Weights = append(decision.Weights, scored[i].Score)
			total += scored[i].Score
		}
		if total > 0 {
			for i := range decision.Weights {
				decision.Weights[i] /= total
			}
		}
		decisions = append(decisions, decision)
	}
	return decisions, nil
}

// DispatchMiniMaxM2Experts applies fake expert functions and weighted routing.
func DispatchMiniMaxM2Experts(hidden [][]float32, decisions []MiniMaxM2RouterDecision, experts map[int]MiniMaxM2ExpertFunc) ([][]float32, error) {
	out := make([][]float32, len(hidden))
	for _, decision := range decisions {
		if decision.TokenIndex < 0 || decision.TokenIndex >= len(hidden) {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 dispatch token index %d out of range", decision.TokenIndex))
		}
		if len(decision.ExpertIDs) != len(decision.Weights) {
			return nil, core.NewError("mlx: MiniMax M2 dispatch expert/weight length mismatch")
		}
		for i, expertID := range decision.ExpertIDs {
			expert := experts[expertID]
			if expert == nil {
				return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 dispatch missing expert %d", expertID))
			}
			result := expert(append([]float32(nil), hidden[decision.TokenIndex]...))
			if out[decision.TokenIndex] == nil {
				out[decision.TokenIndex] = make([]float32, len(result))
			}
			if len(result) != len(out[decision.TokenIndex]) {
				return nil, core.NewError("mlx: MiniMax M2 dispatch expert output shape mismatch")
			}
			for j, value := range result {
				out[decision.TokenIndex][j] += decision.Weights[i] * value
			}
		}
	}
	return out, nil
}

// LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors reads only the routed
// experts referenced by decisions from safetensors shards.
func LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, decisions []MiniMaxM2RouterDecision) (map[int]MiniMaxM2PackedExpertWeights, error) {
	return LoadMiniMaxM2PackedExpertsFromSafetensors(plan, weightFiles, layer, miniMaxM2DecisionExpertIDs(decisions))
}

// LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors loads the router, computes
// top-k decisions for hidden states, and then reads only the selected routed
// expert payloads from safetensors.
func LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, hidden [][]float32, tokenIDs []int32, sink ProbeSink) (MiniMaxM2LazyExpertLoad, error) {
	router, err := LoadMiniMaxM2RouterFromSafetensors(plan, weightFiles, layer)
	if err != nil {
		return MiniMaxM2LazyExpertLoad{}, err
	}
	scores, err := ProjectMiniMaxM2RouterScores(hidden, router)
	if err != nil {
		return MiniMaxM2LazyExpertLoad{}, err
	}
	decisions, err := RouteMiniMaxM2Tokens(plan.Config, scores, router.Bias)
	if err != nil {
		return MiniMaxM2LazyExpertLoad{}, err
	}
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, weightFiles, layer, decisions)
	if err != nil {
		return MiniMaxM2LazyExpertLoad{}, err
	}
	events := MiniMaxM2RouterProbeEvents(layer, tokenIDs, decisions)
	for _, event := range events {
		if sink != nil {
			sink.EmitProbe(event)
		}
	}
	return MiniMaxM2LazyExpertLoad{
		Layer:             layer,
		Router:            router,
		Scores:            scores,
		Decisions:         decisions,
		SelectedExpertIDs: miniMaxM2DecisionExpertIDsSorted(decisions),
		Experts:           experts,
		LoadedPackedBytes: miniMaxM2PackedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// LoadMiniMaxM2PackedExpertsFromSafetensors resolves selected MiniMax M2 routed
// expert projections from safetensors metadata and reads only their packed
// bytes plus quantisation sidecars.
func LoadMiniMaxM2PackedExpertsFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, expertIDs []int) (map[int]MiniMaxM2PackedExpertWeights, error) {
	if len(weightFiles) == 0 {
		return nil, core.NewError("mlx: MiniMax M2 packed expert loading requires safetensors weight files")
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return nil, core.E("minimax_m2.packed_experts", "index safetensors", err)
	}
	out := make(map[int]MiniMaxM2PackedExpertWeights, len(expertIDs))
	for _, expertID := range miniMaxM2UniqueExpertIDs(expertIDs) {
		specs, err := plan.LayerTensorSpecs(layer, expertID)
		if err != nil {
			return nil, err
		}
		gate, err := loadMiniMaxM2PackedProjection(index, findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleExpertGate))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := loadMiniMaxM2PackedProjection(index, findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleExpertUp))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := loadMiniMaxM2PackedProjection(index, findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleExpertDown))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = MiniMaxM2PackedExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizedExperts expands all loaded packed expert projections with the
// reference JANG dequantizer. Native fused kernels can bypass this host path.
func (load MiniMaxM2LazyExpertLoad) DequantizedExperts() (map[int]MiniMaxM2DenseExpertWeights, error) {
	out := make(map[int]MiniMaxM2DenseExpertWeights, len(load.Experts))
	for expertID, expert := range load.Experts {
		gate, err := DequantizeJANGPackedProjection(expert.GateProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := DequantizeJANGPackedProjection(expert.UpProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := DequantizeJANGPackedProjection(expert.DownProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = MiniMaxM2DenseExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizeJANGPackedProjection expands one packed projection payload using
// its descriptor and affine sidecars.
func DequantizeJANGPackedProjection(tensor JANGPackedProjectionTensor) (MiniMaxM2DenseProjectionTensor, error) {
	weight, err := jang.DequantizePackedTensor(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases)
	if err != nil {
		return MiniMaxM2DenseProjectionTensor{}, err
	}
	return MiniMaxM2DenseProjectionTensor{
		Descriptor: tensor.Descriptor,
		Weight:     weight,
		Bias:       append([]float32(nil), tensor.Bias...),
	}, nil
}

// LoadMiniMaxM2RouterFromSafetensors resolves and reads the dense MiniMax M2
// router gate for one layer from safetensors shards.
func LoadMiniMaxM2RouterFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int) (MiniMaxM2RouterWeights, error) {
	if len(weightFiles) == 0 {
		return MiniMaxM2RouterWeights{}, core.NewError("mlx: MiniMax M2 router loading requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return MiniMaxM2RouterWeights{}, err
	}
	routerSpec := findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleRouterGate)
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return MiniMaxM2RouterWeights{}, core.E("minimax_m2.router", "index safetensors", err)
	}
	ref, name, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2RouterGateCandidates(routerSpec))
	if !ok {
		return MiniMaxM2RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing gate tensor: " + routerSpec.Name)
	}
	weight, err := safetensors.ReadRefValues(ref)
	if err != nil {
		return MiniMaxM2RouterWeights{}, core.E("minimax_m2.router", "read gate", err)
	}
	if len(ref.Shape) != 2 || int(ref.Shape[0]) != plan.Config.NumLocalExperts || int(ref.Shape[1]) != plan.Config.HiddenSize {
		return MiniMaxM2RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router gate shape %+v, expected [%d %d]", ref.Shape, plan.Config.NumLocalExperts, plan.Config.HiddenSize))
	}
	router := MiniMaxM2RouterWeights{
		Name:       name,
		Weight:     weight,
		NumExperts: int(ref.Shape[0]),
		HiddenSize: int(ref.Shape[1]),
	}
	biasSpec := findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleRouterBias)
	if biasRef, _, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2RouterBiasCandidates(biasSpec, layer)); ok {
		router.Bias, err = safetensors.ReadRefValues(biasRef)
		if err != nil {
			return MiniMaxM2RouterWeights{}, core.E("minimax_m2.router", "read correction bias", err)
		}
		if len(router.Bias) != router.NumExperts {
			return MiniMaxM2RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router bias length %d, expected %d", len(router.Bias), router.NumExperts))
		}
	} else if plan.Config.UseRoutingBias {
		return MiniMaxM2RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing correction bias")
	}
	return router, nil
}

// ProjectMiniMaxM2RouterScores computes hidden @ router.weight.T.
func ProjectMiniMaxM2RouterScores(hidden [][]float32, router MiniMaxM2RouterWeights) ([][]float32, error) {
	if router.NumExperts <= 0 || router.HiddenSize <= 0 {
		return nil, core.NewError("mlx: MiniMax M2 router requires expert and hidden sizes")
	}
	if len(router.Weight) != router.NumExperts*router.HiddenSize {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 router weight length %d, expected %d", len(router.Weight), router.NumExperts*router.HiddenSize))
	}
	out := make([][]float32, len(hidden))
	for tokenIndex, row := range hidden {
		if len(row) != router.HiddenSize {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 router hidden row %d has %d values, expected %d", tokenIndex, len(row), router.HiddenSize))
		}
		scores := make([]float32, router.NumExperts)
		for expertID := 0; expertID < router.NumExperts; expertID++ {
			base := expertID * router.HiddenSize
			sum := float32(0)
			for hiddenIndex, value := range row {
				sum += value * router.Weight[base+hiddenIndex]
			}
			scores[expertID] = sum
		}
		out[tokenIndex] = scores
	}
	return out, nil
}

// BuildMiniMaxM2LayerForwardSkeletonFromSafetensors resolves and validates the
// attention/router tensor contract for one MiniMax M2 layer using safetensors
// metadata only. It does not read payloads or run kernels.
func BuildMiniMaxM2LayerForwardSkeletonFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int) (MiniMaxM2LayerForwardSkeleton, error) {
	if len(weightFiles) == 0 {
		return MiniMaxM2LayerForwardSkeleton{}, core.NewError("mlx: MiniMax M2 layer skeleton requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return MiniMaxM2LayerForwardSkeleton{}, err
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return MiniMaxM2LayerForwardSkeleton{}, core.E("minimax_m2.layer_skeleton", "index safetensors", err)
	}
	skeleton := MiniMaxM2LayerForwardSkeleton{Layer: layer}
	for _, role := range []MiniMaxM2TensorRole{
		MiniMaxM2TensorRoleAttentionQ,
		MiniMaxM2TensorRoleAttentionK,
		MiniMaxM2TensorRoleAttentionV,
		MiniMaxM2TensorRoleAttentionO,
	} {
		resolved, err := resolveMiniMaxM2SkeletonTensor(index, findMiniMaxM2TensorSpec(specs, role), miniMaxM2PackedWeightCandidates)
		if err != nil {
			return MiniMaxM2LayerForwardSkeleton{}, err
		}
		skeleton.Attention = append(skeleton.Attention, resolved)
	}
	routerGate, err := resolveMiniMaxM2SkeletonTensor(index, findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleRouterGate), miniMaxM2RouterGateCandidates)
	if err != nil {
		return MiniMaxM2LayerForwardSkeleton{}, err
	}
	skeleton.RouterGate = routerGate
	if plan.Config.UseRoutingBias {
		biasSpec := findMiniMaxM2TensorSpec(specs, MiniMaxM2TensorRoleRouterBias)
		routerBias, err := resolveMiniMaxM2SkeletonTensor(index, biasSpec, func(spec MiniMaxM2TensorSpec) []string {
			return miniMaxM2RouterBiasCandidates(spec, layer)
		})
		if err != nil {
			return MiniMaxM2LayerForwardSkeleton{}, err
		}
		skeleton.RouterBias = &routerBias
	}
	return skeleton, nil
}

// MiniMaxM2RouterProbeEvents converts router decisions into typed probe events.
func MiniMaxM2RouterProbeEvents(layer int, tokenIDs []int32, decisions []MiniMaxM2RouterDecision) []ProbeEvent {
	events := make([]ProbeEvent, 0, len(decisions))
	for _, decision := range decisions {
		tokenID := int32(0)
		if decision.TokenIndex >= 0 && decision.TokenIndex < len(tokenIDs) {
			tokenID = tokenIDs[decision.TokenIndex]
		}
		events = append(events, ProbeEvent{
			Kind: ProbeEventRouterDecision,
			Step: decision.TokenIndex,
			RouterDecision: &ProbeRouterDecision{
				Layer:     layer,
				TokenID:   tokenID,
				ExpertIDs: append([]int(nil), decision.ExpertIDs...),
				Weights:   append([]float32(nil), decision.Weights...),
			},
			Meta: map[string]string{"architecture": "minimax_m2"},
		})
	}
	return events
}

func loadMiniMaxM2PackedProjection(index safetensors.Index, spec MiniMaxM2TensorSpec) (JANGPackedProjectionTensor, error) {
	if spec.Packed == nil {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing descriptor: " + spec.Name)
	}
	weightRef, weightName, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2PackedWeightCandidates(spec))
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing weight tensor: " + spec.Name)
	}
	if !miniMaxM2PackedDType(weightRef.DType) {
		return JANGPackedProjectionTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed projection %s dtype %s is not U8", weightName, weightRef.DType))
	}
	packed, err := safetensors.ReadRefRaw(weightRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	scaleRef, _, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2SidecarCandidates(spec, weightName, "scales"))
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing scales for " + spec.Name)
	}
	scales, err := safetensors.ReadRefValues(scaleRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read scales", err)
	}
	biasRef, _, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2SidecarCandidates(spec, weightName, "biases"))
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing biases for " + spec.Name)
	}
	biases, err := safetensors.ReadRefValues(biasRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read biases", err)
	}
	tensor := JANGPackedProjectionTensor{
		Descriptor: *spec.Packed,
		Packed:     packed,
		Scales:     scales,
		Biases:     biases,
	}
	if projBiasRef, _, ok := findMiniMaxM2SafetensorRef(index, miniMaxM2ProjectionBiasCandidates(spec, weightName)); ok {
		tensor.Bias, err = safetensors.ReadRefValues(projBiasRef)
		if err != nil {
			return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read projection bias", err)
		}
	}
	if err := jang.ValidatePackedTensor(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases); err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	return tensor, nil
}

func resolveMiniMaxM2SkeletonTensor(index safetensors.Index, spec MiniMaxM2TensorSpec, candidates func(MiniMaxM2TensorSpec) []string) (MiniMaxM2ResolvedTensor, error) {
	if spec.Name == "" {
		return MiniMaxM2ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton received empty tensor spec")
	}
	ref, name, ok := findMiniMaxM2SafetensorRef(index, candidates(spec))
	if !ok {
		return MiniMaxM2ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton missing tensor: " + spec.Name)
	}
	resolved := MiniMaxM2ResolvedTensor{
		Name:         name,
		Role:         spec.Role,
		Layer:        spec.Layer,
		DType:        ref.DType,
		Shape:        append([]uint64(nil), ref.Shape...),
		LogicalShape: append([]uint64(nil), spec.Shape...),
	}
	if spec.Packed != nil {
		if !miniMaxM2PackedDType(ref.DType) {
			return MiniMaxM2ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not packed U8", name, ref.DType))
		}
		resolved.PackedBytes = spec.Packed.PackedBytes
		if int(ref.ByteLen) != spec.Packed.PackedBytes || ref.Elements != spec.Packed.PackedBytes {
			return MiniMaxM2ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s packed bytes %d/%d, expected %d", name, ref.ByteLen, ref.Elements, spec.Packed.PackedBytes))
		}
		return resolved, nil
	}
	if !miniMaxM2FloatDType(ref.DType) {
		return MiniMaxM2ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not floating point", name, ref.DType))
	}
	if !sameUint64Slice(ref.Shape, spec.Shape) {
		return MiniMaxM2ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s shape %+v, expected %+v", name, ref.Shape, spec.Shape))
	}
	return resolved, nil
}

type miniMaxM2ExpertScore struct {
	ID    int
	Score float32
}

func (plan MiniMaxM2TensorPlan) attentionSpec(layer int, projection string, role MiniMaxM2TensorRole) MiniMaxM2TensorSpec {
	name := core.Sprintf("model.layers.%d.self_attn.%s.weight", layer, projection)
	qSize := firstPositive(plan.Config.NumAttentionHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	kvSize := firstPositive(plan.Config.NumKeyValueHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	shape := []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.HiddenSize)}
	switch role {
	case MiniMaxM2TensorRoleAttentionQ:
		shape = []uint64{uint64(qSize), uint64(plan.Config.HiddenSize)}
	case MiniMaxM2TensorRoleAttentionK, MiniMaxM2TensorRoleAttentionV:
		shape = []uint64{uint64(kvSize), uint64(plan.Config.HiddenSize)}
	case MiniMaxM2TensorRoleAttentionO:
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(qSize)}
	}
	spec := MiniMaxM2TensorSpec{
		Name:    name,
		Aliases: miniMaxM2AttentionAliases(layer, projection, role),
		Role:    role,
		Layer:   layer,
		Shape:   shape,
	}
	if packed, err := jang.NewPackedTensorDescriptor(name, shape, plan.JANG); err == nil {
		spec.Packed = &packed
	}
	return spec
}

func miniMaxM2AttentionAliases(layer int, projection string, role MiniMaxM2TensorRole) []string {
	switch role {
	case MiniMaxM2TensorRoleAttentionQ, MiniMaxM2TensorRoleAttentionK, MiniMaxM2TensorRoleAttentionV:
		return []string{core.Sprintf("model.layers.%d.self_attn.qkv_proj.weight", layer)}
	default:
		return nil
	}
}

func (plan MiniMaxM2TensorPlan) expertSpec(layer, expert int, projection string, role MiniMaxM2TensorRole) MiniMaxM2TensorSpec {
	name := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d.%s.weight", layer, expert, projection)
	shape := []uint64{uint64(plan.Config.IntermediateSize), uint64(plan.Config.HiddenSize)}
	if projection == "down_proj" {
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.IntermediateSize)}
	}
	spec := MiniMaxM2TensorSpec{
		Name:    name,
		Aliases: []string{core.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", layer, expert, projection)},
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

func firstMiniMaxM2Architecture(values []string) string {
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

func specMatchesName(spec MiniMaxM2TensorSpec, names map[string]bool) bool {
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

func findMiniMaxM2TensorSpec(specs []MiniMaxM2TensorSpec, role MiniMaxM2TensorRole) MiniMaxM2TensorSpec {
	for _, spec := range specs {
		if spec.Role == role {
			return spec
		}
	}
	return MiniMaxM2TensorSpec{}
}

func miniMaxM2DecisionExpertIDs(decisions []MiniMaxM2RouterDecision) []int {
	var ids []int
	for _, decision := range decisions {
		ids = append(ids, decision.ExpertIDs...)
	}
	return ids
}

func miniMaxM2DecisionExpertIDsSorted(decisions []MiniMaxM2RouterDecision) []int {
	return miniMaxM2UniqueExpertIDs(miniMaxM2DecisionExpertIDs(decisions))
}

func miniMaxM2PackedExpertLoadedBytes(experts map[int]MiniMaxM2PackedExpertWeights) uint64 {
	total := uint64(0)
	for _, expert := range experts {
		total += uint64(len(expert.GateProj.Packed))
		total += uint64(len(expert.UpProj.Packed))
		total += uint64(len(expert.DownProj.Packed))
	}
	return total
}

func miniMaxM2UniqueExpertIDs(ids []int) []int {
	seen := map[int]bool{}
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

func miniMaxM2PackedWeightCandidates(spec MiniMaxM2TensorSpec) []string {
	bases := append([]string{spec.Name}, spec.Aliases...)
	out := make([]string, 0, len(bases)*4)
	for _, base := range bases {
		out = append(out, base, base+".packed", base+".qweight", trimMiniMaxM2WeightSuffix(base)+".qweight")
	}
	return out
}

func miniMaxM2RouterGateCandidates(spec MiniMaxM2TensorSpec) []string {
	out := append([]string{spec.Name}, spec.Aliases...)
	if spec.Name != "" {
		out = append(out, trimMiniMaxM2WeightSuffix(spec.Name)+".gate")
	}
	return out
}

func miniMaxM2RouterBiasCandidates(spec MiniMaxM2TensorSpec, layer int) []string {
	names := []string{
		spec.Name,
		core.Sprintf("model.layers.%d.block_sparse_moe.e_score_correction_bias", layer),
		core.Sprintf("model.layers.%d.mlp.e_score_correction_bias", layer),
		core.Sprintf("model.layers.%d.block_sparse_moe.gate.e_score_correction_bias", layer),
	}
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name != "" {
			out = append(out, name)
		}
	}
	return out
}

func miniMaxM2SidecarCandidates(spec MiniMaxM2TensorSpec, weightName, sidecar string) []string {
	names := []string{weightName}
	if trimmed := trimMiniMaxM2PackedSuffix(weightName); trimmed != weightName {
		names = append(names, trimmed)
	}
	names = append(names, spec.Name)
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names)*3)
	for _, name := range names {
		out = append(out, name+"."+sidecar, trimMiniMaxM2WeightSuffix(name)+"."+sidecar, name+"_"+sidecar)
	}
	return out
}

func miniMaxM2ProjectionBiasCandidates(spec MiniMaxM2TensorSpec, weightName string) []string {
	names := []string{weightName, spec.Name}
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names)*3)
	for _, name := range names {
		out = append(out, trimMiniMaxM2WeightSuffix(name)+".bias", name+".proj_bias", trimMiniMaxM2WeightSuffix(name)+".proj_bias")
	}
	return out
}

func findMiniMaxM2SafetensorRef(index safetensors.Index, candidates []string) (safetensors.TensorRef, string, bool) {
	for _, name := range candidates {
		ref, ok := index.Tensors[name]
		if ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

func trimMiniMaxM2WeightSuffix(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")]
	}
	return name
}

func trimMiniMaxM2PackedSuffix(name string) string {
	for _, suffix := range []string{".packed", ".qweight"} {
		if core.HasSuffix(name, suffix) {
			return name[:len(name)-len(suffix)]
		}
	}
	return name
}

func miniMaxM2PackedDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "U8", "UINT8":
		return true
	default:
		return false
	}
}

func miniMaxM2FloatDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "F16", "BF16", "F32", "F64":
		return true
	default:
		return false
	}
}

func miniMaxM2DTypeBytes(dtype string) int {
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

func miniMaxM2Score(value float32, scoringFunc string) float32 {
	switch core.Lower(scoringFunc) {
	case "", "sigmoid":
		return float32(1 / (1 + math.Exp(float64(-value))))
	default:
		return value
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
