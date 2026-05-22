// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
	mlxjang "dappco.re/go/mlx/quant/jang"
	"dappco.re/go/mlx/safetensors"
	"math"
	"sort"
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
	Router            RouterWeights               `json:"router,omitempty"`
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
	for _, tensor := range skeleton.Attention {
		total += tensor.EstimatedBytes()
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
	cfg.ModelType = normalizeKnownArchitecture(firstNonEmpty(cfg.ModelType, firstArchitecture(cfg.Architectures)))
	if cfg.ScoringFunc == "" {
		cfg.ScoringFunc = "sigmoid"
	}
	return cfg, nil
}

// BuildTensorPlan creates a model-wide tensor mapping plan.
func BuildTensorPlan(cfg Config, info *jang.Info) (TensorPlan, error) {
	if normalizeKnownArchitecture(cfg.ModelType) != "minimax_m2" && firstArchitecture(cfg.Architectures) == "" {
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
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 layer %d out of range", layer))
	}
	if expert < 0 || expert >= plan.Config.NumLocalExperts {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 expert %d out of range", expert))
	}
	specs := []TensorSpec{
		plan.attentionSpec(layer, "q_proj", TensorRoleAttentionQ),
		plan.attentionSpec(layer, "k_proj", TensorRoleAttentionK),
		plan.attentionSpec(layer, "v_proj", TensorRoleAttentionV),
		plan.attentionSpec(layer, "o_proj", TensorRoleAttentionO),
		{
			Name:  core.Sprintf("model.layers.%d.block_sparse_moe.gate.weight", layer),
			Role:  TensorRoleRouterGate,
			Layer: layer,
			Shape: []uint64{uint64(plan.Config.NumLocalExperts), uint64(plan.Config.HiddenSize)},
			DType: "f32",
		},
		plan.expertSpec(layer, expert, "gate_proj", TensorRoleExpertGate),
		plan.expertSpec(layer, expert, "up_proj", TensorRoleExpertUp),
		plan.expertSpec(layer, expert, "down_proj", TensorRoleExpertDown),
	}
	if plan.Config.UseRoutingBias {
		specs = append(specs, TensorSpec{
			Name:  core.Sprintf("model.layers.%d.block_sparse_moe.e_score_correction_bias", layer),
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

// RouteTokens computes deterministic top-k router decisions for a
// batch of router scores. Scores are sigmoid-normalised by default and top-k
// weights are renormalised, matching the MiniMax M2 sparse routing contract.
func RouteTokens(cfg Config, scores [][]float32, bias []float32) ([]RouterDecision, error) {
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
	decisions := make([]RouterDecision, 0, len(scores))
	for tokenIndex, row := range scores {
		if len(row) != cfg.NumLocalExperts {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 routing row %d has %d scores, expected %d", tokenIndex, len(row), cfg.NumLocalExperts))
		}
		scored := make([]expertScore, 0, len(row))
		for expertID, raw := range row {
			value := raw
			if len(bias) > 0 {
				value += bias[expertID]
			}
			scored = append(scored, expertScore{ID: expertID, Score: score(value, cfg.ScoringFunc)})
		}
		sort.SliceStable(scored, func(i, j int) bool {
			if scored[i].Score == scored[j].Score {
				return scored[i].ID < scored[j].ID
			}
			return scored[i].Score > scored[j].Score
		})
		decision := RouterDecision{
			TokenIndex: tokenIndex,
			ExpertIDs:  make([]int, topK),
			Weights:    make([]float32, topK),
		}
		total := float32(0)
		for i := 0; i < topK; i++ {
			decision.ExpertIDs[i] = scored[i].ID
			decision.Weights[i] = scored[i].Score
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

// DispatchExperts applies fake expert functions and weighted routing.
func DispatchExperts(hidden [][]float32, decisions []RouterDecision, experts map[int]ExpertFunc) ([][]float32, error) {
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
			result := expert(core.SliceClone(hidden[decision.TokenIndex]))
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

// LoadPackedExpertsForDecisions reads only the routed
// experts referenced by decisions from safetensors shards.
func LoadPackedExpertsForDecisions(plan TensorPlan, weightFiles []string, layer int, decisions []RouterDecision) (map[int]PackedExpertWeights, error) {
	return LoadPackedExperts(plan, weightFiles, layer, decisionExpertIDs(decisions))
}

// LoadLazyExpertsForHidden loads the router, computes
// top-k decisions for hidden states, and then reads only the selected routed
// expert payloads from safetensors.
func LoadLazyExpertsForHidden(plan TensorPlan, weightFiles []string, layer int, hidden [][]float32, tokenIDs []int32, sink probe.Sink) (LazyExpertLoad, error) {
	router, err := LoadRouter(plan, weightFiles, layer)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	scores, err := ProjectRouterScores(hidden, router)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	decisions, err := RouteTokens(plan.Config, scores, router.Bias)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	experts, err := LoadPackedExpertsForDecisions(plan, weightFiles, layer, decisions)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	events := RouterProbeEvents(layer, tokenIDs, decisions)
	for _, event := range events {
		if sink != nil {
			sink.EmitProbe(event)
		}
	}
	return LazyExpertLoad{
		Layer:             layer,
		Router:            router,
		Scores:            scores,
		Decisions:         decisions,
		SelectedExpertIDs: decisionExpertIDsSorted(decisions),
		Experts:           experts,
		LoadedPackedBytes: packedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// LoadPackedExperts resolves selected MiniMax M2 routed
// expert projections from safetensors metadata and reads only their packed
// bytes plus quantisation sidecars.
func LoadPackedExperts(plan TensorPlan, weightFiles []string, layer int, expertIDs []int) (map[int]PackedExpertWeights, error) {
	if len(weightFiles) == 0 {
		return nil, core.NewError("mlx: MiniMax M2 packed expert loading requires safetensors weight files")
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return nil, core.E("minimax_m2.packed_experts", "index safetensors", err)
	}
	out := make(map[int]PackedExpertWeights, len(expertIDs))
	for _, expertID := range uniqueExpertIDs(expertIDs) {
		specs, err := plan.LayerTensorSpecs(layer, expertID)
		if err != nil {
			return nil, err
		}
		gate, err := loadPackedProjection(index, findTensorSpec(specs, TensorRoleExpertGate))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := loadPackedProjection(index, findTensorSpec(specs, TensorRoleExpertUp))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := loadPackedProjection(index, findTensorSpec(specs, TensorRoleExpertDown))
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = PackedExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizedExperts expands all loaded packed expert projections with the
// reference JANG dequantizer. Native fused kernels can bypass this host path.
func (load LazyExpertLoad) DequantizedExperts() (map[int]DenseExpertWeights, error) {
	out := make(map[int]DenseExpertWeights, len(load.Experts))
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
		out[expertID] = DenseExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizeJANGPackedProjection expands one packed projection payload using
// its descriptor and affine sidecars.
func DequantizeJANGPackedProjection(tensor JANGPackedProjectionTensor) (DenseProjectionTensor, error) {
	weight, err := jang.DequantizePackedTensor(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases)
	if err != nil {
		return DenseProjectionTensor{}, err
	}
	return DenseProjectionTensor{
		Descriptor: tensor.Descriptor,
		Weight:     weight,
		Bias:       append([]float32(nil), tensor.Bias...),
	}, nil
}

// LoadRouter resolves and reads the dense MiniMax M2
// router gate for one layer from safetensors shards.
func LoadRouter(plan TensorPlan, weightFiles []string, layer int) (RouterWeights, error) {
	if len(weightFiles) == 0 {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router loading requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return RouterWeights{}, err
	}
	routerSpec := findTensorSpec(specs, TensorRoleRouterGate)
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return RouterWeights{}, core.E("minimax_m2.router", "index safetensors", err)
	}
	ref, name, ok := findSafetensorRef(index, routerGateCandidates(routerSpec))
	if !ok {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing gate tensor: " + routerSpec.Name)
	}
	weight, err := safetensors.ReadRefValues(ref)
	if err != nil {
		return RouterWeights{}, core.E("minimax_m2.router", "read gate", err)
	}
	if len(ref.Shape) != 2 || int(ref.Shape[0]) != plan.Config.NumLocalExperts || int(ref.Shape[1]) != plan.Config.HiddenSize {
		return RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router gate shape %+v, expected [%d %d]", ref.Shape, plan.Config.NumLocalExperts, plan.Config.HiddenSize))
	}
	router := RouterWeights{
		Name:       name,
		Weight:     weight,
		NumExperts: int(ref.Shape[0]),
		HiddenSize: int(ref.Shape[1]),
	}
	biasSpec := findTensorSpec(specs, TensorRoleRouterBias)
	if biasRef, _, ok := findSafetensorRef(index, routerBiasCandidates(biasSpec, layer)); ok {
		router.Bias, err = safetensors.ReadRefValues(biasRef)
		if err != nil {
			return RouterWeights{}, core.E("minimax_m2.router", "read correction bias", err)
		}
		if len(router.Bias) != router.NumExperts {
			return RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router bias length %d, expected %d", len(router.Bias), router.NumExperts))
		}
	} else if plan.Config.UseRoutingBias {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing correction bias")
	}
	return router, nil
}

// ProjectRouterScores computes hidden @ router.weight.T.
func ProjectRouterScores(hidden [][]float32, router RouterWeights) ([][]float32, error) {
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

// BuildLayerForwardSkeleton resolves and validates the
// attention/router tensor contract for one MiniMax M2 layer using safetensors
// metadata only. It does not read payloads or run kernels.
func BuildLayerForwardSkeleton(plan TensorPlan, weightFiles []string, layer int) (LayerForwardSkeleton, error) {
	if len(weightFiles) == 0 {
		return LayerForwardSkeleton{}, core.NewError("mlx: MiniMax M2 layer skeleton requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return LayerForwardSkeleton{}, err
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return LayerForwardSkeleton{}, core.E("minimax_m2.layer_skeleton", "index safetensors", err)
	}
	skeleton := LayerForwardSkeleton{Layer: layer}
	for _, role := range []TensorRole{
		TensorRoleAttentionQ,
		TensorRoleAttentionK,
		TensorRoleAttentionV,
		TensorRoleAttentionO,
	} {
		resolved, err := resolveSkeletonTensor(index, findTensorSpec(specs, role), packedWeightCandidates)
		if err != nil {
			return LayerForwardSkeleton{}, err
		}
		skeleton.Attention = append(skeleton.Attention, resolved)
	}
	routerGate, err := resolveSkeletonTensor(index, findTensorSpec(specs, TensorRoleRouterGate), routerGateCandidates)
	if err != nil {
		return LayerForwardSkeleton{}, err
	}
	skeleton.RouterGate = routerGate
	if plan.Config.UseRoutingBias {
		biasSpec := findTensorSpec(specs, TensorRoleRouterBias)
		routerBias, err := resolveSkeletonTensor(index, biasSpec, func(spec TensorSpec) []string {
			return routerBiasCandidates(spec, layer)
		})
		if err != nil {
			return LayerForwardSkeleton{}, err
		}
		skeleton.RouterBias = &routerBias
	}
	return skeleton, nil
}

// RouterProbeEvents converts router decisions into typed probe events.
func RouterProbeEvents(layer int, tokenIDs []int32, decisions []RouterDecision) []probe.Event {
	events := make([]probe.Event, 0, len(decisions))
	for _, decision := range decisions {
		tokenID := int32(0)
		if decision.TokenIndex >= 0 && decision.TokenIndex < len(tokenIDs) {
			tokenID = tokenIDs[decision.TokenIndex]
		}
		events = append(events, probe.Event{
			Kind: probe.KindRouterDecision,
			Step: decision.TokenIndex,
			RouterDecision: &probe.RouterDecision{
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

func loadPackedProjection(index safetensors.Index, spec TensorSpec) (JANGPackedProjectionTensor, error) {
	if spec.Packed == nil {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing descriptor: " + spec.Name)
	}
	weightRef, weightName, ok := findSafetensorRef(index, packedWeightCandidates(spec))
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing weight tensor: " + spec.Name)
	}
	if !packedDType(weightRef.DType) {
		return JANGPackedProjectionTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed projection %s dtype %s is not U8", weightName, weightRef.DType))
	}
	packed, err := safetensors.ReadRefRaw(weightRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	scaleRef, _, ok := findSafetensorRef(index, sidecarCandidates(spec, weightName, "scales"))
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing scales for " + spec.Name)
	}
	scales, err := safetensors.ReadRefValues(scaleRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read scales", err)
	}
	biasRef, _, ok := findSafetensorRef(index, sidecarCandidates(spec, weightName, "biases"))
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
	if projBiasRef, _, ok := findSafetensorRef(index, projectionBiasCandidates(spec, weightName)); ok {
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

func resolveSkeletonTensor(index safetensors.Index, spec TensorSpec, candidates func(TensorSpec) []string) (ResolvedTensor, error) {
	if spec.Name == "" {
		return ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton received empty tensor spec")
	}
	ref, name, ok := findSafetensorRef(index, candidates(spec))
	if !ok {
		return ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton missing tensor: " + spec.Name)
	}
	resolved := ResolvedTensor{
		Name:         name,
		Role:         spec.Role,
		Layer:        spec.Layer,
		DType:        ref.DType,
		Shape:        append([]uint64(nil), ref.Shape...),
		LogicalShape: append([]uint64(nil), spec.Shape...),
	}
	if spec.Packed != nil {
		if !packedDType(ref.DType) {
			return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not packed U8", name, ref.DType))
		}
		resolved.PackedBytes = spec.Packed.PackedBytes
		if int(ref.ByteLen) != spec.Packed.PackedBytes || ref.Elements != spec.Packed.PackedBytes {
			return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s packed bytes %d/%d, expected %d", name, ref.ByteLen, ref.Elements, spec.Packed.PackedBytes))
		}
		return resolved, nil
	}
	if !floatDType(ref.DType) {
		return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not floating point", name, ref.DType))
	}
	if !sameUint64Slice(ref.Shape, spec.Shape) {
		return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s shape %+v, expected %+v", name, ref.Shape, spec.Shape))
	}
	return resolved, nil
}

type expertScore struct {
	ID    int
	Score float32
}

func (plan TensorPlan) attentionSpec(layer int, projection string, role TensorRole) TensorSpec {
	name := core.Sprintf("model.layers.%d.self_attn.%s.weight", layer, projection)
	qSize := firstPositive(plan.Config.NumAttentionHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	kvSize := firstPositive(plan.Config.NumKeyValueHeads*plan.Config.HeadDim, plan.Config.HiddenSize)
	shape := []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.HiddenSize)}
	switch role {
	case TensorRoleAttentionQ:
		shape = []uint64{uint64(qSize), uint64(plan.Config.HiddenSize)}
	case TensorRoleAttentionK, TensorRoleAttentionV:
		shape = []uint64{uint64(kvSize), uint64(plan.Config.HiddenSize)}
	case TensorRoleAttentionO:
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(qSize)}
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
		return []string{core.Sprintf("model.layers.%d.self_attn.qkv_proj.weight", layer)}
	default:
		return nil
	}
}

func (plan TensorPlan) expertSpec(layer, expert int, projection string, role TensorRole) TensorSpec {
	name := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d.%s.weight", layer, expert, projection)
	shape := []uint64{uint64(plan.Config.IntermediateSize), uint64(plan.Config.HiddenSize)}
	if projection == "down_proj" {
		shape = []uint64{uint64(plan.Config.HiddenSize), uint64(plan.Config.IntermediateSize)}
	}
	spec := TensorSpec{
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

func specMatchesName(spec TensorSpec, names map[string]bool) bool {
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

func findTensorSpec(specs []TensorSpec, role TensorRole) TensorSpec {
	for _, spec := range specs {
		if spec.Role == role {
			return spec
		}
	}
	return TensorSpec{}
}

func decisionExpertIDs(decisions []RouterDecision) []int {
	var ids []int
	for _, decision := range decisions {
		ids = append(ids, decision.ExpertIDs...)
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

func packedWeightCandidates(spec TensorSpec) []string {
	bases := append([]string{spec.Name}, spec.Aliases...)
	out := make([]string, 0, len(bases)*4)
	for _, base := range bases {
		out = append(out, base, base+".packed", base+".qweight", trimWeightSuffix(base)+".qweight")
	}
	return out
}

func routerGateCandidates(spec TensorSpec) []string {
	out := append([]string{spec.Name}, spec.Aliases...)
	if spec.Name != "" {
		out = append(out, trimWeightSuffix(spec.Name)+".gate")
	}
	return out
}

func routerBiasCandidates(spec TensorSpec, layer int) []string {
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

func sidecarCandidates(spec TensorSpec, weightName, sidecar string) []string {
	names := []string{weightName}
	if trimmed := trimPackedSuffix(weightName); trimmed != weightName {
		names = append(names, trimmed)
	}
	names = append(names, spec.Name)
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names)*3)
	for _, name := range names {
		out = append(out, name+"."+sidecar, trimWeightSuffix(name)+"."+sidecar, name+"_"+sidecar)
	}
	return out
}

func projectionBiasCandidates(spec TensorSpec, weightName string) []string {
	names := []string{weightName, spec.Name}
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names)*3)
	for _, name := range names {
		out = append(out, trimWeightSuffix(name)+".bias", name+".proj_bias", trimWeightSuffix(name)+".proj_bias")
	}
	return out
}

func findSafetensorRef(index safetensors.Index, candidates []string) (safetensors.TensorRef, string, bool) {
	for _, name := range candidates {
		ref, ok := index.Tensors[name]
		if ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

func trimWeightSuffix(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")]
	}
	return name
}

func trimPackedSuffix(name string) string {
	for _, suffix := range []string{".packed", ".qweight"} {
		if core.HasSuffix(name, suffix) {
			return name[:len(name)-len(suffix)]
		}
	}
	return name
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

func score(value float32, scoringFunc string) float32 {
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

// DispatchPackedExpertsMetal applies router-selected MiniMax M2
// packed experts using fused JANG/JANGTQ projection kernels for gate, up, and
// down projections. It is intentionally host-shaped for bring-up fixtures and
// model-loader validation; full model execution keeps tensors on device.
func DispatchPackedExpertsMetal(hidden [][]float32, decisions []RouterDecision, experts map[int]PackedExpertWeights) ([][]float32, error) {
	out := make([][]float32, len(hidden))
	for _, decision := range decisions {
		if decision.TokenIndex < 0 || decision.TokenIndex >= len(hidden) {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed dispatch token index %d out of range", decision.TokenIndex))
		}
		if len(decision.ExpertIDs) != len(decision.Weights) {
			return nil, core.NewError("mlx: MiniMax M2 packed dispatch expert/weight length mismatch")
		}
		for i, expertID := range decision.ExpertIDs {
			expert, ok := experts[expertID]
			if !ok {
				return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed dispatch missing expert %d", expertID))
			}
			result, err := runPackedExpertMetal(hidden[decision.TokenIndex], expert)
			if err != nil {
				return nil, core.E("minimax_m2.packed_dispatch", core.Sprintf("expert %d", expertID), err)
			}
			if out[decision.TokenIndex] == nil {
				out[decision.TokenIndex] = make([]float32, len(result))
			}
			if len(result) != len(out[decision.TokenIndex]) {
				return nil, core.NewError("mlx: MiniMax M2 packed dispatch expert output shape mismatch")
			}
			for j, value := range result {
				out[decision.TokenIndex][j] += decision.Weights[i] * value
			}
		}
	}
	return out, nil
}

// DispatchPackedExpertsFromSafetensorsMetal loads the router-selected
// packed experts from safetensors shards and executes the fused Metal dispatch.
func DispatchPackedExpertsFromSafetensorsMetal(plan TensorPlan, weightFiles []string, layer int, hidden [][]float32, decisions []RouterDecision) ([][]float32, error) {
	experts, err := LoadPackedExpertsForDecisions(plan, weightFiles, layer, decisions)
	if err != nil {
		return nil, err
	}
	return DispatchPackedExpertsMetal(hidden, decisions, experts)
}

// ForwardLazyExpertLoadMetal executes an already-routed lazy expert
// load with the native packed projection kernels.
func ForwardLazyExpertLoadMetal(hidden [][]float32, load LazyExpertLoad) (PackedLayerForwardResult, error) {
	output, err := DispatchPackedExpertsMetal(hidden, load.Decisions, load.Experts)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	return PackedLayerForwardResult{
		Output:            output,
		Decisions:         append([]RouterDecision(nil), load.Decisions...),
		SelectedExpertIDs: append([]int(nil), load.SelectedExpertIDs...),
		LoadedPackedBytes: load.LoadedPackedBytes,
		ProbeEvents:       append([]probe.Event(nil), load.ProbeEvents...),
	}, nil
}

// ForwardPackedLayerMetal routes hidden states through a MiniMax M2
// packed MoE layer skeleton, lazily resolving selected experts from safetensors
// and emitting router probe events.
func ForwardPackedLayerMetal(opts PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	if len(opts.Hidden) != len(opts.RouterScores) {
		return PackedLayerForwardResult{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed layer hidden rows %d, router rows %d", len(opts.Hidden), len(opts.RouterScores)))
	}
	decisions, err := RouteTokens(opts.Plan.Config, opts.RouterScores, opts.RouterBias)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	experts, err := LoadPackedExpertsForDecisions(opts.Plan, opts.WeightFiles, opts.Layer, decisions)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	output, err := DispatchPackedExpertsMetal(opts.Hidden, decisions, experts)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	events := RouterProbeEvents(opts.Layer, opts.TokenIDs, decisions)
	for _, event := range events {
		if opts.ProbeSink != nil {
			opts.ProbeSink.EmitProbe(event)
		}
	}
	return PackedLayerForwardResult{
		Output:            output,
		Decisions:         decisions,
		SelectedExpertIDs: decisionExpertIDsSorted(decisions),
		LoadedPackedBytes: packedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// ForwardPackedLayerFromSafetensorsMetal reads the dense router gate,
// computes router scores, then runs the packed layer skeleton with lazy expert
// resolution.
func ForwardPackedLayerFromSafetensorsMetal(opts PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	if len(opts.RouterBias) == 0 {
		load, err := LoadLazyExpertsForHidden(opts.Plan, opts.WeightFiles, opts.Layer, opts.Hidden, opts.TokenIDs, opts.ProbeSink)
		if err != nil {
			return PackedLayerForwardResult{}, err
		}
		return ForwardLazyExpertLoadMetal(opts.Hidden, load)
	}
	router, err := LoadRouter(opts.Plan, opts.WeightFiles, opts.Layer)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	scores, err := ProjectRouterScores(opts.Hidden, router)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	opts.RouterScores = scores
	if len(opts.RouterBias) == 0 {
		opts.RouterBias = router.Bias
	}
	return ForwardPackedLayerMetal(opts)
}

func runPackedExpertMetal(hidden []float32, expert PackedExpertWeights) ([]float32, error) {
	inputShape := []int32{1, int32(len(hidden))}
	gate, err := projectPackedTensorMetal(expert.GateProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "gate_proj", err)
	}
	up, err := projectPackedTensorMetal(expert.UpProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "up_proj", err)
	}
	if len(gate.Values) != len(up.Values) {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed expert gate/up size mismatch %d != %d", len(gate.Values), len(up.Values)))
	}
	activated := make([]float32, len(gate.Values))
	for i := range activated {
		activated[i] = swiGLU(gate.Values[i], up.Values[i])
	}
	downShape := []int32{1, int32(len(activated))}
	down, err := projectPackedTensorMetal(expert.DownProj, activated, downShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "down_proj", err)
	}
	return down.Values, nil
}

func projectPackedTensorMetal(tensor JANGPackedProjectionTensor, input []float32, inputShape []int32) (mlxjang.PackedProjectionResult, error) {
	return mlxjang.ProjectPackedTensorFused(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases, input, inputShape, tensor.Bias)
}

func swiGLU(gate, up float32) float32 {
	return float32(float64(gate)/(1+math.Exp(float64(-gate)))) * up
}
