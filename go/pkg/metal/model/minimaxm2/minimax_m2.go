// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package minimaxm2

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"sort"

	"dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/inference/safetensors"
)

const maxMiniMaxM2SafetensorHeaderBytes = 256 << 20

type miniMaxM2LoadConfig struct {
	ModelType             string   `json:"model_type,omitempty"`
	Architectures         []string `json:"architectures,omitempty"`
	HiddenSize            int      `json:"hidden_size,omitempty"`
	IntermediateSize      int      `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int      `json:"num_key_value_heads,omitempty"`
	HeadDim               int      `json:"head_dim,omitempty"`
	VocabSize             int      `json:"vocab_size,omitempty"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings,omitempty"`
	SlidingWindow         int      `json:"sliding_window,omitempty"`
	NumLocalExperts       int      `json:"num_local_experts,omitempty"`
	NumExpertsPerToken    int      `json:"num_experts_per_tok,omitempty"`
	UseRoutingBias        bool     `json:"use_routing_bias,omitempty"`
}

type miniMaxM2JANGLoadConfig struct {
	WeightFormat string `json:"weight_format,omitempty"`
	Profile      string `json:"profile,omitempty"`
	Quantization struct {
		GroupSize   int    `json:"group_size,omitempty"`
		BitsDefault int    `json:"bits_default,omitempty"`
		Method      string `json:"method,omitempty"`
	} `json:"quantization"`
	MXTQBits struct {
		Attention    int `json:"attention,omitempty"`
		RoutedExpert int `json:"routed_expert,omitempty"`
	} `json:"mxtq_bits"`
}

type miniMaxM2NativeLoadPlan struct {
	Config        miniMaxM2LoadConfig
	JANG          miniMaxM2JANGLoadConfig
	Summary       string
	TensorShards  int
	LayerSkeleton miniMaxM2NativeLayerSkeleton
	TensorRefs    map[string]miniMaxM2SafetensorTensorRef
}

type miniMaxM2StagedModel struct {
	path      string
	plan      miniMaxM2NativeLoadPlan
	tokenizer *metal.Tokenizer
}

type miniMaxM2NativeResolvedTensor struct {
	Name         string
	Role         string
	DType        string
	Shape        []uint64
	LogicalShape []uint64
	PackedBytes  int64
}

type miniMaxM2NativeLayerSkeleton struct {
	Layer      int
	Attention  []miniMaxM2NativeResolvedTensor
	RouterGate miniMaxM2NativeResolvedTensor
	RouterBias *miniMaxM2NativeResolvedTensor
}

type miniMaxM2NativeTensorSpec struct {
	Name        string
	Candidates  []string
	Role        string
	Shape       []uint64
	Packed      bool
	PackedBytes int64
}

type miniMaxM2NativePackedTensorPayloadRef struct {
	Name         string
	Role         string
	Path         string
	DType        string
	Shape        []uint64
	LogicalShape []uint64
	DataStart    int64
	ByteLen      int64
	PackedBytes  int64
}

type miniMaxM2NativeExpertPayloadRefs struct {
	ExpertID    int
	GateProj    miniMaxM2NativePackedTensorPayloadRef
	UpProj      miniMaxM2NativePackedTensorPayloadRef
	DownProj    miniMaxM2NativePackedTensorPayloadRef
	PackedBytes int64
}

type miniMaxM2NativePackedProjectionPayload struct {
	Ref       miniMaxM2NativePackedTensorPayloadRef
	Packed    []byte
	Scales    []float32
	Biases    []float32
	Bias      []float32
	GroupSize int
	Bits      int
}

type miniMaxM2NativeExpertPayload struct {
	ExpertID    int
	GateProj    miniMaxM2NativePackedProjectionPayload
	UpProj      miniMaxM2NativePackedProjectionPayload
	DownProj    miniMaxM2NativePackedProjectionPayload
	PackedBytes int64
}

type miniMaxM2NativeRouterWeights struct {
	Layer      int
	Weight     []float32
	Bias       []float32
	NumExperts int
	HiddenSize int
}

type miniMaxM2NativeRouterDecision struct {
	TokenIndex int
	ExpertIDs  []int
	Weights    []float32
	Scores     []float32
}

type miniMaxM2NativeSparseLayerResult struct {
	Output            [][]float32
	Scores            [][]float32
	Decisions         []miniMaxM2NativeRouterDecision
	SelectedExpertIDs []int
	LoadedPackedBytes int64
}

type miniMaxM2SafetensorTensorRef struct {
	Name      string
	Path      string
	DType     string
	Shape     []uint64
	Elements  int64
	DataStart int64
	ByteLen   int64
}

// validateMiniMaxM2NativeLoad checks the cheap, deterministic parts of a
// MiniMax M2/JANGTQ pack before the native sparse kernels exist. It reads only
// config and safetensors headers, so it is safe to run on very large packs.
func validateMiniMaxM2NativeLoad(modelPath string, configData []byte) (string, error) {
	plan, err := prepareMiniMaxM2NativeLoad(modelPath, configData)
	if err != nil {
		return "", err
	}
	return plan.Summary, nil
}

func init() {
	metal.RegisterModelLoader("minimax_m2", func(modelPath string, configData []byte) (metal.InternalModel, error) {
		model, err := loadMiniMaxM2StagedModel(modelPath, configData)
		if err != nil {
			return nil, core.E("model.loadModel", "validate minimax_m2 native load", err)
		}
		return model, nil
	})
}

func loadMiniMaxM2StagedModel(modelPath string, configData []byte) (*miniMaxM2StagedModel, error) {
	plan, err := prepareMiniMaxM2NativeLoad(modelPath, configData)
	if err != nil {
		return nil, err
	}
	root := metal.ResolveModelRoot(modelPath)
	tokenizer, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("minimax_m2.load", "load tokenizer", err)
	}
	return &miniMaxM2StagedModel{path: root, plan: plan, tokenizer: tokenizer}, nil
}

func prepareMiniMaxM2NativeLoad(modelPath string, configData []byte) (miniMaxM2NativeLoadPlan, error) {
	root := metal.ResolveModelRoot(modelPath)
	cfg, err := parseMiniMaxM2LoadConfig(configData)
	if err != nil {
		return miniMaxM2NativeLoadPlan{}, err
	}
	if err := cfg.validate(); err != nil {
		return miniMaxM2NativeLoadPlan{}, err
	}
	tensors, shards, err := readMiniMaxM2SafetensorRefs(modelPath, root)
	if err != nil {
		return miniMaxM2NativeLoadPlan{}, err
	}
	names := miniMaxM2SafetensorNameSet(tensors)
	missing := cfg.missingRequiredTensorNames(names)
	if len(missing) > 0 {
		return miniMaxM2NativeLoadPlan{}, core.NewError("minimax_m2 tensor validation failed: missing required tensors: " + core.Join(", ", missing...))
	}
	jang := readMiniMaxM2JANGLoadConfig(root)
	skeleton, err := buildMiniMaxM2NativeLayerSkeleton(cfg, jang, tensors, 0)
	if err != nil {
		return miniMaxM2NativeLoadPlan{}, err
	}
	format := firstNonEmptyUpper(jang.WeightFormat, "MXTQ")
	profile := firstNonEmptyUpper(jang.Profile, "JANGTQ")
	return miniMaxM2NativeLoadPlan{
		Config:        cfg,
		JANG:          jang,
		Summary:       core.Sprintf("minimax_m2 %s/%s tensor plan validated from %d safetensors shard(s); layer 0 attention/router skeleton validated", profile, format, shards),
		TensorShards:  shards,
		LayerSkeleton: skeleton,
		TensorRefs:    tensors,
	}, nil
}

func (m *miniMaxM2StagedModel) Forward(_ *metal.Array, _ []metal.Cache) *metal.Array { return nil }

func (m *miniMaxM2StagedModel) ForwardMasked(_ *metal.Array, _ *metal.Array, _ []metal.Cache) *metal.Array {
	return nil
}

func (m *miniMaxM2StagedModel) NewCache() []metal.Cache { return nil }

func (m *miniMaxM2StagedModel) NumLayers() int { return m.plan.Config.NumHiddenLayers }

func (m *miniMaxM2StagedModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }

func (m *miniMaxM2StagedModel) ModelType() string { return "minimax_m2" }

func (m *miniMaxM2StagedModel) ApplyLoRA(_ metal.LoRAConfig) *metal.LoRAAdapter { return nil }

func (m *miniMaxM2StagedModel) DecodeUnavailableError(operation string) error {
	return core.NewError(operation + ": minimax_m2 staged loader has no native decode kernels yet")
}

func (m *miniMaxM2StagedModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = m.plan.Config.VocabSize
	info.HiddenSize = m.plan.Config.HiddenSize
	info.ContextLength = m.plan.Config.MaxPositionEmbeddings
	if info.ContextLength == 0 {
		info.ContextLength = m.plan.Config.SlidingWindow
	}
	info.QuantBits = m.plan.JANG.MXTQBits.RoutedExpert
	if info.QuantBits == 0 {
		info.QuantBits = m.plan.JANG.Quantization.BitsDefault
	}
	info.QuantGroup = m.plan.JANG.Quantization.GroupSize
}

func parseMiniMaxM2LoadConfig(data []byte) (miniMaxM2LoadConfig, error) {
	var cfg miniMaxM2LoadConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return miniMaxM2LoadConfig{}, result.Value.(error)
	}
	cfg.ModelType = metal.NormalizeProbeModelType(firstNonEmptyString(cfg.ModelType, firstMiniMaxM2ArchitectureName(cfg.Architectures)))
	return cfg, nil
}

func (cfg miniMaxM2LoadConfig) validate() error {
	if cfg.ModelType != "minimax_m2" {
		return core.NewError("minimax_m2 validation requires MiniMax M2 config")
	}
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return core.NewError("minimax_m2 validation requires hidden, intermediate, and layer sizes")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.HeadDim <= 0 {
		return core.NewError("minimax_m2 validation requires attention head metadata")
	}
	if cfg.NumLocalExperts <= 0 || cfg.NumExpertsPerToken <= 0 {
		return core.NewError("minimax_m2 validation requires local expert counts")
	}
	if cfg.NumExpertsPerToken > cfg.NumLocalExperts {
		return core.NewError("minimax_m2 validation top-k experts cannot exceed local expert count")
	}
	return nil
}

func (cfg miniMaxM2LoadConfig) missingRequiredTensorNames(names map[string]bool) []string {
	required := [][]string{
		miniMaxM2WeightCandidates("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.qkv_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.self_attn.o_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.block_sparse_moe.gate.weight"),
		miniMaxM2WeightCandidates("model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.block_sparse_moe.experts.0.up_proj.weight", "model.layers.0.mlp.experts.0.up_proj.weight"),
		miniMaxM2WeightCandidates("model.layers.0.block_sparse_moe.experts.0.down_proj.weight", "model.layers.0.mlp.experts.0.down_proj.weight"),
	}
	if cfg.UseRoutingBias {
		required = append(required, miniMaxM2WeightCandidates("model.layers.0.block_sparse_moe.e_score_correction_bias"))
	}
	missing := []string{}
	for _, candidates := range required {
		if hasMiniMaxM2TensorName(names, candidates) {
			continue
		}
		missing = append(missing, candidates[0])
	}
	sort.Strings(missing)
	return missing
}

func miniMaxM2WeightCandidates(names ...string) []string {
	candidates := []string{}
	for _, name := range names {
		candidates = append(candidates, metal.WeightCandidates(name)...)
	}
	return candidates
}

func hasMiniMaxM2TensorName(names map[string]bool, candidates []string) bool {
	for _, candidate := range candidates {
		if names[candidate] {
			return true
		}
	}
	return false
}

func readMiniMaxM2SafetensorNames(modelPath, root string) (map[string]bool, int, error) {
	tensors, shards, err := readMiniMaxM2SafetensorRefs(modelPath, root)
	if err != nil {
		return nil, 0, err
	}
	return miniMaxM2SafetensorNameSet(tensors), shards, nil
}

func readMiniMaxM2SafetensorRefs(modelPath, root string) (map[string]miniMaxM2SafetensorTensorRef, int, error) {
	paths := []string{}
	if core.HasSuffix(core.Lower(modelPath), ".safetensors") {
		paths = []string{modelPath}
	} else {
		paths = core.PathGlob(core.JoinPath(root, "*.safetensors"))
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, 0, core.NewError("minimax_m2 tensor validation found no safetensors weight shards")
	}
	tensors := map[string]miniMaxM2SafetensorTensorRef{}
	for _, path := range paths {
		shardTensors, err := readMiniMaxM2SafetensorHeaderRefs(path)
		if err != nil {
			return nil, 0, err
		}
		for name, tensor := range shardTensors {
			if _, exists := tensors[name]; exists {
				return nil, 0, core.NewError("minimax_m2 tensor validation found duplicate tensor: " + name)
			}
			tensors[name] = tensor
		}
	}
	return tensors, len(paths), nil
}

func miniMaxM2SafetensorNameSet(tensors map[string]miniMaxM2SafetensorTensorRef) map[string]bool {
	names := make(map[string]bool, len(tensors))
	for name := range tensors {
		names[name] = true
	}
	return names
}

func readMiniMaxM2SafetensorHeaderNames(path string) (map[string]bool, error) {
	tensors, err := readMiniMaxM2SafetensorHeaderRefs(path)
	if err != nil {
		return nil, err
	}
	return miniMaxM2SafetensorNameSet(tensors), nil
}

func readMiniMaxM2SafetensorHeaderRefs(path string) (map[string]miniMaxM2SafetensorTensorRef, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, core.E("minimax_m2.safetensors", "open "+core.PathBase(path), err)
	}
	defer file.Close()

	var headerLenBuf [8]byte
	if _, err := io.ReadFull(file, headerLenBuf[:]); err != nil {
		return nil, core.E("minimax_m2.safetensors", "read header length "+core.PathBase(path), err)
	}
	headerLen := binary.LittleEndian.Uint64(headerLenBuf[:])
	if headerLen == 0 || headerLen > maxMiniMaxM2SafetensorHeaderBytes {
		return nil, core.NewError(core.Sprintf("minimax_m2 safetensors header length %d is invalid in %s", headerLen, core.PathBase(path)))
	}
	headerBytes := make([]byte, int(headerLen))
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, core.E("minimax_m2.safetensors", "read header "+core.PathBase(path), err)
	}

	// Delegate header parsing to the shared safetensors walker (W8-I).
	// It hand-rolls the JSON parse, interns canonical dtype strings,
	// and carves all Shape slices out of one slab so per-tensor cost
	// lands at ~1 alloc once the arena is in scope — replacing the
	// reflection-driven map[string]headerEntry decode that previously
	// dominated this path's allocations.
	index, err := safetensors.ParseHeaderRefs(path, headerBytes, int64(8+headerLen))
	if err != nil {
		return nil, core.E("minimax_m2.safetensors", "parse header "+core.PathBase(path), err)
	}
	tensors := make(map[string]miniMaxM2SafetensorTensorRef, len(index.Tensors))
	for name, ref := range index.Tensors {
		tensors[name] = miniMaxM2SafetensorRefFromIndex(ref)
	}
	return tensors, nil
}

// miniMaxM2SafetensorRefFromIndex projects a safetensors.TensorRef into
// the minimax-local view, which carries Elements as int64 (used in
// packed-byte equality checks against int64 sidecar sizes) and is
// otherwise identical in shape. The Shape slice is reused as-is — it
// references the safetensors header's shape slab, which is GC-rooted
// for the lifetime of the returned ref.
func miniMaxM2SafetensorRefFromIndex(ref safetensors.TensorRef) miniMaxM2SafetensorTensorRef {
	return miniMaxM2SafetensorTensorRef{
		Name:      ref.Name,
		Path:      ref.Path,
		DType:     ref.DType,
		Shape:     ref.Shape,
		Elements:  int64(ref.Elements),
		DataStart: ref.DataStart,
		ByteLen:   ref.ByteLen,
	}
}

func buildMiniMaxM2NativeLayerSkeleton(cfg miniMaxM2LoadConfig, jang miniMaxM2JANGLoadConfig, tensors map[string]miniMaxM2SafetensorTensorRef, layer int) (miniMaxM2NativeLayerSkeleton, error) {
	if layer < 0 || layer >= cfg.NumHiddenLayers {
		return miniMaxM2NativeLayerSkeleton{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton layer %d out of range", layer))
	}
	skeleton := miniMaxM2NativeLayerSkeleton{Layer: layer}
	for _, spec := range miniMaxM2NativeAttentionSpecs(cfg, jang, layer) {
		resolved, err := resolveMiniMaxM2NativeSkeletonTensor(tensors, spec)
		if err != nil {
			return miniMaxM2NativeLayerSkeleton{}, err
		}
		skeleton.Attention = append(skeleton.Attention, resolved)
	}
	routerGate, err := resolveMiniMaxM2NativeSkeletonTensor(tensors, miniMaxM2NativeRouterGateSpec(cfg, layer))
	if err != nil {
		return miniMaxM2NativeLayerSkeleton{}, err
	}
	skeleton.RouterGate = routerGate
	if cfg.UseRoutingBias {
		routerBias, err := resolveMiniMaxM2NativeSkeletonTensor(tensors, miniMaxM2NativeRouterBiasSpec(cfg, layer))
		if err != nil {
			return miniMaxM2NativeLayerSkeleton{}, err
		}
		skeleton.RouterBias = &routerBias
	}
	return skeleton, nil
}

func (plan miniMaxM2NativeLoadPlan) ResolveExpertPayloadRefs(layer int, expertIDs []int) (map[int]miniMaxM2NativeExpertPayloadRefs, error) {
	if len(plan.TensorRefs) == 0 {
		return nil, core.NewError("minimax_m2 expert payload refs require safetensors metadata")
	}
	out := make(map[int]miniMaxM2NativeExpertPayloadRefs, len(expertIDs))
	for _, expertID := range miniMaxM2NativeUniqueExpertIDs(expertIDs) {
		if expertID < 0 || expertID >= plan.Config.NumLocalExperts {
			return nil, core.NewError(core.Sprintf("minimax_m2 expert %d out of range", expertID))
		}
		specs := miniMaxM2NativeExpertSpecs(plan.Config, plan.JANG, layer, expertID)
		gate, err := resolveMiniMaxM2NativePackedPayloadRef(plan.TensorRefs, specs[0])
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload_refs", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := resolveMiniMaxM2NativePackedPayloadRef(plan.TensorRefs, specs[1])
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload_refs", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := resolveMiniMaxM2NativePackedPayloadRef(plan.TensorRefs, specs[2])
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload_refs", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = miniMaxM2NativeExpertPayloadRefs{
			ExpertID:    expertID,
			GateProj:    gate,
			UpProj:      up,
			DownProj:    down,
			PackedBytes: gate.PackedBytes + up.PackedBytes + down.PackedBytes,
		}
	}
	return out, nil
}

func (plan miniMaxM2NativeLoadPlan) ReadExpertPayloads(layer int, expertIDs []int) (map[int]miniMaxM2NativeExpertPayload, error) {
	refs, err := plan.ResolveExpertPayloadRefs(layer, expertIDs)
	if err != nil {
		return nil, err
	}
	out := make(map[int]miniMaxM2NativeExpertPayload, len(refs))
	for expertID, expertRefs := range refs {
		gate, err := plan.readPackedProjectionPayload(expertRefs.GateProj)
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := plan.readPackedProjectionPayload(expertRefs.UpProj)
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := plan.readPackedProjectionPayload(expertRefs.DownProj)
		if err != nil {
			return nil, core.E("minimax_m2.expert_payload", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = miniMaxM2NativeExpertPayload{
			ExpertID:    expertID,
			GateProj:    gate,
			UpProj:      up,
			DownProj:    down,
			PackedBytes: expertRefs.PackedBytes,
		}
	}
	return out, nil
}

func (plan miniMaxM2NativeLoadPlan) ForwardSparseLayer(layer int, hidden [][]float32) (miniMaxM2NativeSparseLayerResult, error) {
	router, err := plan.LoadRouter(layer)
	if err != nil {
		return miniMaxM2NativeSparseLayerResult{}, err
	}
	scores, err := router.Project(hidden)
	if err != nil {
		return miniMaxM2NativeSparseLayerResult{}, err
	}
	decisions, selectedExpertIDs, err := routeMiniMaxM2NativeTokens(plan.Config, scores)
	if err != nil {
		return miniMaxM2NativeSparseLayerResult{}, err
	}
	payloads, err := plan.ReadExpertPayloads(layer, selectedExpertIDs)
	if err != nil {
		return miniMaxM2NativeSparseLayerResult{}, err
	}
	output, err := dispatchMiniMaxM2NativeExperts(hidden, decisions, payloads)
	if err != nil {
		return miniMaxM2NativeSparseLayerResult{}, err
	}
	loaded := int64(0)
	for _, expertID := range selectedExpertIDs {
		loaded += payloads[expertID].PackedBytes
	}
	return miniMaxM2NativeSparseLayerResult{
		Output:            output,
		Scores:            scores,
		Decisions:         decisions,
		SelectedExpertIDs: selectedExpertIDs,
		LoadedPackedBytes: loaded,
	}, nil
}

func (plan miniMaxM2NativeLoadPlan) LoadRouter(layer int) (miniMaxM2NativeRouterWeights, error) {
	if layer < 0 || layer >= plan.Config.NumHiddenLayers {
		return miniMaxM2NativeRouterWeights{}, core.NewError(core.Sprintf("minimax_m2 router layer %d out of range", layer))
	}
	gateSpec := miniMaxM2NativeRouterGateSpec(plan.Config, layer)
	gateRef, ok := findMiniMaxM2NativeTensorRef(plan.TensorRefs, gateSpec.Candidates)
	if !ok {
		return miniMaxM2NativeRouterWeights{}, core.NewError("minimax_m2 router missing tensor: " + gateSpec.Name)
	}
	if !sameMiniMaxM2Uint64Slice(gateRef.Shape, gateSpec.Shape) {
		return miniMaxM2NativeRouterWeights{}, core.NewError(core.Sprintf("minimax_m2 router %s shape %+v, expected %+v", gateRef.Name, gateRef.Shape, gateSpec.Shape))
	}
	weights, err := readMiniMaxM2SafetensorFloat32(gateRef)
	if err != nil {
		return miniMaxM2NativeRouterWeights{}, core.E("minimax_m2.router", "read gate", err)
	}
	expectedWeights := plan.Config.NumLocalExperts * plan.Config.HiddenSize
	if len(weights) != expectedWeights {
		return miniMaxM2NativeRouterWeights{}, core.NewError(core.Sprintf("minimax_m2 router weight count %d, expected %d", len(weights), expectedWeights))
	}
	router := miniMaxM2NativeRouterWeights{
		Layer:      layer,
		Weight:     weights,
		NumExperts: plan.Config.NumLocalExperts,
		HiddenSize: plan.Config.HiddenSize,
	}
	if plan.Config.UseRoutingBias {
		biasSpec := miniMaxM2NativeRouterBiasSpec(plan.Config, layer)
		biasRef, ok := findMiniMaxM2NativeTensorRef(plan.TensorRefs, biasSpec.Candidates)
		if !ok {
			return miniMaxM2NativeRouterWeights{}, core.NewError("minimax_m2 router missing tensor: " + biasSpec.Name)
		}
		if !sameMiniMaxM2Uint64Slice(biasRef.Shape, biasSpec.Shape) {
			return miniMaxM2NativeRouterWeights{}, core.NewError(core.Sprintf("minimax_m2 router bias %s shape %+v, expected %+v", biasRef.Name, biasRef.Shape, biasSpec.Shape))
		}
		bias, err := readMiniMaxM2SafetensorFloat32(biasRef)
		if err != nil {
			return miniMaxM2NativeRouterWeights{}, core.E("minimax_m2.router", "read correction bias", err)
		}
		if len(bias) != plan.Config.NumLocalExperts {
			return miniMaxM2NativeRouterWeights{}, core.NewError(core.Sprintf("minimax_m2 router bias count %d, expected %d", len(bias), plan.Config.NumLocalExperts))
		}
		router.Bias = bias
	}
	return router, nil
}

func (router miniMaxM2NativeRouterWeights) Project(hidden [][]float32) ([][]float32, error) {
	if router.NumExperts <= 0 || router.HiddenSize <= 0 {
		return nil, core.NewError("minimax_m2 router metadata is invalid")
	}
	if len(router.Weight) != router.NumExperts*router.HiddenSize {
		return nil, core.NewError("minimax_m2 router weight shape is invalid")
	}
	if len(router.Bias) > 0 && len(router.Bias) != router.NumExperts {
		return nil, core.NewError("minimax_m2 router bias shape is invalid")
	}
	out := make([][]float32, len(hidden))
	for token, vector := range hidden {
		if len(vector) != router.HiddenSize {
			return nil, core.NewError(core.Sprintf("minimax_m2 router token %d hidden width %d, expected %d", token, len(vector), router.HiddenSize))
		}
		tokenScores := make([]float32, router.NumExperts)
		for expert := 0; expert < router.NumExperts; expert++ {
			offset := expert * router.HiddenSize
			score := float32(0)
			for i, value := range vector {
				score += value * router.Weight[offset+i]
			}
			if len(router.Bias) > 0 {
				score += router.Bias[expert]
			}
			tokenScores[expert] = score
		}
		out[token] = tokenScores
	}
	return out, nil
}

func routeMiniMaxM2NativeTokens(cfg miniMaxM2LoadConfig, scores [][]float32) ([]miniMaxM2NativeRouterDecision, []int, error) {
	if cfg.NumExpertsPerToken <= 0 || cfg.NumExpertsPerToken > cfg.NumLocalExperts {
		return nil, nil, core.NewError("minimax_m2 router top-k metadata is invalid")
	}
	decisions := make([]miniMaxM2NativeRouterDecision, len(scores))
	selected := []int{}
	for token, tokenScores := range scores {
		if len(tokenScores) != cfg.NumLocalExperts {
			return nil, nil, core.NewError(core.Sprintf("minimax_m2 router token %d score count %d, expected %d", token, len(tokenScores), cfg.NumLocalExperts))
		}
		ranked := make([]int, cfg.NumLocalExperts)
		for i := range ranked {
			ranked[i] = i
		}
		sort.SliceStable(ranked, func(i, j int) bool {
			left := ranked[i]
			right := ranked[j]
			if tokenScores[left] == tokenScores[right] {
				return left < right
			}
			return tokenScores[left] > tokenScores[right]
		})
		ids := append([]int(nil), ranked[:cfg.NumExpertsPerToken]...)
		weights := miniMaxM2NativeSoftmaxWeights(tokenScores, ids)
		decisionScores := make([]float32, len(ids))
		for i, id := range ids {
			decisionScores[i] = tokenScores[id]
		}
		decisions[token] = miniMaxM2NativeRouterDecision{
			TokenIndex: token,
			ExpertIDs:  ids,
			Weights:    weights,
			Scores:     decisionScores,
		}
		selected = append(selected, ids...)
	}
	return decisions, miniMaxM2NativeUniqueExpertIDs(selected), nil
}

func dispatchMiniMaxM2NativeExperts(hidden [][]float32, decisions []miniMaxM2NativeRouterDecision, payloads map[int]miniMaxM2NativeExpertPayload) ([][]float32, error) {
	if len(hidden) != len(decisions) {
		return nil, core.NewError(core.Sprintf("minimax_m2 sparse dispatch token count %d, decisions %d", len(hidden), len(decisions)))
	}
	output := make([][]float32, len(hidden))
	for token, vector := range hidden {
		if decisions[token].TokenIndex != token {
			return nil, core.NewError(core.Sprintf("minimax_m2 sparse dispatch decision token %d at position %d", decisions[token].TokenIndex, token))
		}
		tokenOutput := make([]float32, len(vector))
		for i, expertID := range decisions[token].ExpertIDs {
			payload, ok := payloads[expertID]
			if !ok {
				return nil, core.NewError(core.Sprintf("minimax_m2 sparse dispatch missing expert %d payload", expertID))
			}
			expertOutput, err := forwardMiniMaxM2NativeExpertPayload(vector, payload)
			if err != nil {
				return nil, core.E("minimax_m2.sparse_dispatch", core.Sprintf("expert %d token %d", expertID, token), err)
			}
			if len(expertOutput) != len(tokenOutput) {
				return nil, core.NewError(core.Sprintf("minimax_m2 sparse dispatch expert %d output width %d, expected %d", expertID, len(expertOutput), len(tokenOutput)))
			}
			weight := float32(1)
			if i < len(decisions[token].Weights) {
				weight = decisions[token].Weights[i]
			}
			for j, value := range expertOutput {
				tokenOutput[j] += value * weight
			}
		}
		output[token] = tokenOutput
	}
	return output, nil
}

func (plan miniMaxM2NativeLoadPlan) readPackedProjectionPayload(ref miniMaxM2NativePackedTensorPayloadRef) (miniMaxM2NativePackedProjectionPayload, error) {
	packed, err := readMiniMaxM2SafetensorRaw(ref.Path, ref.DataStart, ref.ByteLen)
	if err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, err
	}
	scaleRef, err := plan.resolvePayloadSidecarRef(ref.Name, "scales")
	if err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, err
	}
	scales, err := readMiniMaxM2SafetensorFloat32(scaleRef)
	if err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, core.E("minimax_m2.expert_payload", "read scales", err)
	}
	biasRef, err := plan.resolvePayloadSidecarRef(ref.Name, "biases")
	if err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, err
	}
	biases, err := readMiniMaxM2SafetensorFloat32(biasRef)
	if err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, core.E("minimax_m2.expert_payload", "read biases", err)
	}
	groupSize := firstPositiveInt(plan.JANG.Quantization.GroupSize, 64)
	bits := miniMaxM2NativeRoutedExpertBits(plan.JANG)
	if err := validateMiniMaxM2NativePackedPayload(ref, packed, scales, biases, groupSize); err != nil {
		return miniMaxM2NativePackedProjectionPayload{}, err
	}
	return miniMaxM2NativePackedProjectionPayload{
		Ref:       ref,
		Packed:    packed,
		Scales:    scales,
		Biases:    biases,
		GroupSize: groupSize,
		Bits:      bits,
	}, nil
}

func (plan miniMaxM2NativeLoadPlan) resolvePayloadSidecarRef(weightName, sidecar string) (miniMaxM2SafetensorTensorRef, error) {
	candidates := []string{
		weightName + "." + sidecar,
		trimMiniMaxM2NativePackedSuffix(weightName) + "." + sidecar,
		trimMiniMaxM2NativeWeightSuffix(trimMiniMaxM2NativePackedSuffix(weightName)) + "." + sidecar,
		weightName + "_" + sidecar,
	}
	for _, candidate := range candidates {
		if ref, ok := plan.TensorRefs[candidate]; ok {
			return ref, nil
		}
	}
	return miniMaxM2SafetensorTensorRef{}, core.NewError("minimax_m2 payload sidecar missing " + sidecar + " for " + weightName)
}

func forwardMiniMaxM2NativeExpertPayload(hidden []float32, payload miniMaxM2NativeExpertPayload) ([]float32, error) {
	input := metal.FromValues(hidden, 1, len(hidden))
	defer metal.Free(input)
	gate, err := runMiniMaxM2NativeProjection(input, payload.GateProj)
	if err != nil {
		return nil, core.E("minimax_m2.native_expert", "gate_proj", err)
	}
	defer metal.Free(gate)
	up, err := runMiniMaxM2NativeProjection(input, payload.UpProj)
	if err != nil {
		return nil, core.E("minimax_m2.native_expert", "up_proj", err)
	}
	defer metal.Free(up)
	gateActivated := metal.SiLU(gate)
	defer metal.Free(gateActivated)
	activated := metal.Mul(gateActivated, up)
	defer metal.Free(activated)
	down, err := runMiniMaxM2NativeProjection(activated, payload.DownProj)
	if err != nil {
		return nil, core.E("minimax_m2.native_expert", "down_proj", err)
	}
	defer metal.Free(down)
	metal.Materialize(down)
	return down.Floats(), nil
}

func runMiniMaxM2NativeProjection(input *metal.Array, payload miniMaxM2NativePackedProjectionPayload) (*metal.Array, error) {
	shape, err := miniMaxM2NativeInt32Shape(payload.Ref.LogicalShape)
	if err != nil {
		return nil, err
	}
	packed := metal.FromValues(payload.Packed, len(payload.Packed))
	scales := metal.FromValues(payload.Scales, len(payload.Scales))
	biases := metal.FromValues(payload.Biases, len(payload.Biases))
	defer metal.Free(packed, scales, biases)
	return metal.JANGPackedLinearFused(input, packed, scales, biases, nil, shape, payload.GroupSize, payload.Bits)
}

func miniMaxM2NativeAttentionSpecs(cfg miniMaxM2LoadConfig, jang miniMaxM2JANGLoadConfig, layer int) []miniMaxM2NativeTensorSpec {
	qSize := firstPositiveInt(cfg.NumAttentionHeads*cfg.HeadDim, cfg.HiddenSize)
	kvSize := firstPositiveInt(cfg.NumKeyValueHeads*cfg.HeadDim, cfg.HiddenSize)
	return []miniMaxM2NativeTensorSpec{
		miniMaxM2NativePackedTensorSpec(core.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer), []string{core.Sprintf("model.layers.%d.self_attn.qkv_proj.weight", layer)}, "attention.q_proj", []uint64{uint64(qSize), uint64(cfg.HiddenSize)}, miniMaxM2NativeAttentionBits(jang)),
		miniMaxM2NativePackedTensorSpec(core.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer), []string{core.Sprintf("model.layers.%d.self_attn.qkv_proj.weight", layer)}, "attention.k_proj", []uint64{uint64(kvSize), uint64(cfg.HiddenSize)}, miniMaxM2NativeAttentionBits(jang)),
		miniMaxM2NativePackedTensorSpec(core.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer), []string{core.Sprintf("model.layers.%d.self_attn.qkv_proj.weight", layer)}, "attention.v_proj", []uint64{uint64(kvSize), uint64(cfg.HiddenSize)}, miniMaxM2NativeAttentionBits(jang)),
		miniMaxM2NativePackedTensorSpec(core.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer), nil, "attention.o_proj", []uint64{uint64(cfg.HiddenSize), uint64(qSize)}, miniMaxM2NativeAttentionBits(jang)),
	}
}

func miniMaxM2NativeExpertSpecs(cfg miniMaxM2LoadConfig, jang miniMaxM2JANGLoadConfig, layer, expert int) []miniMaxM2NativeTensorSpec {
	gateName := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d.gate_proj.weight", layer, expert)
	upName := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d.up_proj.weight", layer, expert)
	downName := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d.down_proj.weight", layer, expert)
	return []miniMaxM2NativeTensorSpec{
		miniMaxM2NativePackedTensorSpec(gateName, []string{core.Sprintf("model.layers.%d.mlp.experts.%d.gate_proj.weight", layer, expert)}, "expert.gate_proj", []uint64{uint64(cfg.IntermediateSize), uint64(cfg.HiddenSize)}, miniMaxM2NativeRoutedExpertBits(jang)),
		miniMaxM2NativePackedTensorSpec(upName, []string{core.Sprintf("model.layers.%d.mlp.experts.%d.up_proj.weight", layer, expert)}, "expert.up_proj", []uint64{uint64(cfg.IntermediateSize), uint64(cfg.HiddenSize)}, miniMaxM2NativeRoutedExpertBits(jang)),
		miniMaxM2NativePackedTensorSpec(downName, []string{core.Sprintf("model.layers.%d.mlp.experts.%d.down_proj.weight", layer, expert)}, "expert.down_proj", []uint64{uint64(cfg.HiddenSize), uint64(cfg.IntermediateSize)}, miniMaxM2NativeRoutedExpertBits(jang)),
	}
}

func miniMaxM2NativePackedTensorSpec(name string, aliases []string, role string, logicalShape []uint64, bits int) miniMaxM2NativeTensorSpec {
	candidates := miniMaxM2WeightCandidates(name)
	for _, alias := range aliases {
		candidates = append(candidates, miniMaxM2WeightCandidates(alias)...)
	}
	for _, base := range append([]string{name}, aliases...) {
		if base == "" {
			continue
		}
		candidates = append(candidates, base+".packed", base+".qweight")
	}
	return miniMaxM2NativeTensorSpec{
		Name:        name,
		Candidates:  candidates,
		Role:        role,
		Shape:       logicalShape,
		Packed:      true,
		PackedBytes: miniMaxM2NativePackedBytes(logicalShape, bits),
	}
}

func miniMaxM2NativeRouterGateSpec(cfg miniMaxM2LoadConfig, layer int) miniMaxM2NativeTensorSpec {
	name := core.Sprintf("model.layers.%d.block_sparse_moe.gate.weight", layer)
	return miniMaxM2NativeTensorSpec{
		Name:       name,
		Candidates: append(miniMaxM2WeightCandidates(name), core.Sprintf("model.layers.%d.mlp.gate.weight", layer)),
		Role:       "router.gate",
		Shape:      []uint64{uint64(cfg.NumLocalExperts), uint64(cfg.HiddenSize)},
	}
}

func miniMaxM2NativeRouterBiasSpec(cfg miniMaxM2LoadConfig, layer int) miniMaxM2NativeTensorSpec {
	name := core.Sprintf("model.layers.%d.block_sparse_moe.e_score_correction_bias", layer)
	return miniMaxM2NativeTensorSpec{
		Name: name,
		Candidates: []string{
			name,
			core.Sprintf("model.layers.%d.mlp.e_score_correction_bias", layer),
			core.Sprintf("model.layers.%d.block_sparse_moe.gate.e_score_correction_bias", layer),
		},
		Role:  "router.e_score_correction_bias",
		Shape: []uint64{uint64(cfg.NumLocalExperts)},
	}
}

func resolveMiniMaxM2NativeSkeletonTensor(tensors map[string]miniMaxM2SafetensorTensorRef, spec miniMaxM2NativeTensorSpec) (miniMaxM2NativeResolvedTensor, error) {
	ref, ok := findMiniMaxM2NativeTensorRef(tensors, spec.Candidates)
	if !ok {
		return miniMaxM2NativeResolvedTensor{}, core.NewError("minimax_m2 layer skeleton missing tensor: " + spec.Name)
	}
	resolved := miniMaxM2NativeResolvedTensor{
		Name:         ref.Name,
		Role:         spec.Role,
		DType:        ref.DType,
		Shape:        append([]uint64(nil), ref.Shape...),
		LogicalShape: append([]uint64(nil), spec.Shape...),
	}
	if spec.Packed {
		if !miniMaxM2NativePackedDType(ref.DType) {
			return miniMaxM2NativeResolvedTensor{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton %s dtype %s is not packed U8", ref.Name, ref.DType))
		}
		resolved.PackedBytes = spec.PackedBytes
		if ref.Elements != spec.PackedBytes || ref.ByteLen != spec.PackedBytes {
			return miniMaxM2NativeResolvedTensor{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton %s packed bytes %d/%d, expected %d", ref.Name, ref.ByteLen, ref.Elements, spec.PackedBytes))
		}
		return resolved, nil
	}
	if !miniMaxM2NativeFloatDType(ref.DType) {
		return miniMaxM2NativeResolvedTensor{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton %s dtype %s is not floating point", ref.Name, ref.DType))
	}
	if !sameMiniMaxM2Uint64Slice(ref.Shape, spec.Shape) {
		return miniMaxM2NativeResolvedTensor{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton %s shape %+v, expected %+v", ref.Name, ref.Shape, spec.Shape))
	}
	expectedBytes := int64(miniMaxM2NativeDTypeBytes(ref.DType)) * ref.Elements
	if expectedBytes > 0 && ref.ByteLen != expectedBytes {
		return miniMaxM2NativeResolvedTensor{}, core.NewError(core.Sprintf("minimax_m2 layer skeleton %s byte length %d, expected %d", ref.Name, ref.ByteLen, expectedBytes))
	}
	return resolved, nil
}

func resolveMiniMaxM2NativePackedPayloadRef(tensors map[string]miniMaxM2SafetensorTensorRef, spec miniMaxM2NativeTensorSpec) (miniMaxM2NativePackedTensorPayloadRef, error) {
	if !spec.Packed {
		return miniMaxM2NativePackedTensorPayloadRef{}, core.NewError("minimax_m2 payload ref requires packed tensor spec: " + spec.Name)
	}
	ref, ok := findMiniMaxM2NativeTensorRef(tensors, spec.Candidates)
	if !ok {
		return miniMaxM2NativePackedTensorPayloadRef{}, core.NewError("minimax_m2 payload ref missing tensor: " + spec.Name)
	}
	if !miniMaxM2NativePackedDType(ref.DType) {
		return miniMaxM2NativePackedTensorPayloadRef{}, core.NewError(core.Sprintf("minimax_m2 payload ref %s dtype %s is not packed U8", ref.Name, ref.DType))
	}
	if ref.Elements != spec.PackedBytes || ref.ByteLen != spec.PackedBytes {
		return miniMaxM2NativePackedTensorPayloadRef{}, core.NewError(core.Sprintf("minimax_m2 payload ref %s packed bytes %d/%d, expected %d", ref.Name, ref.ByteLen, ref.Elements, spec.PackedBytes))
	}
	return miniMaxM2NativePackedTensorPayloadRef{
		Name:         ref.Name,
		Role:         spec.Role,
		Path:         ref.Path,
		DType:        ref.DType,
		Shape:        append([]uint64(nil), ref.Shape...),
		LogicalShape: append([]uint64(nil), spec.Shape...),
		DataStart:    ref.DataStart,
		ByteLen:      ref.ByteLen,
		PackedBytes:  spec.PackedBytes,
	}, nil
}

func readMiniMaxM2SafetensorRaw(path string, offset, byteLen int64) ([]byte, error) {
	if byteLen < 0 || byteLen > int64(^uint(0)>>1) {
		return nil, core.NewError("minimax_m2 safetensors payload byte length is invalid")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, core.E("minimax_m2.safetensors", "open payload "+core.PathBase(path), err)
	}
	defer file.Close()
	out := make([]byte, int(byteLen))
	n, err := file.ReadAt(out, offset)
	if err != nil && !(err == io.EOF && n == len(out)) {
		return nil, err
	}
	if n != len(out) {
		return nil, core.NewError("minimax_m2 safetensors payload is truncated")
	}
	return out, nil
}

func readMiniMaxM2SafetensorFloat32(ref miniMaxM2SafetensorTensorRef) ([]float32, error) {
	if !miniMaxM2NativeFloatDType(ref.DType) {
		return nil, core.NewError("minimax_m2 tensor is not floating point: " + ref.Name)
	}
	raw, err := readMiniMaxM2SafetensorRaw(ref.Path, ref.DataStart, ref.ByteLen)
	if err != nil {
		return nil, err
	}
	switch core.Upper(ref.DType) {
	case "F16":
		if int64(len(raw)) != ref.Elements*2 {
			return nil, core.NewError("minimax_m2 float16 tensor byte length is invalid: " + ref.Name)
		}
		out := make([]float32, int(ref.Elements))
		for i := range out {
			out[i] = miniMaxM2NativeFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16":
		if int64(len(raw)) != ref.Elements*2 {
			return nil, core.NewError("minimax_m2 bfloat16 tensor byte length is invalid: " + ref.Name)
		}
		out := make([]float32, int(ref.Elements))
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return out, nil
	case "F32":
		if int64(len(raw)) != ref.Elements*4 {
			return nil, core.NewError("minimax_m2 float32 tensor byte length is invalid: " + ref.Name)
		}
		out := make([]float32, int(ref.Elements))
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "F64":
		if int64(len(raw)) != ref.Elements*8 {
			return nil, core.NewError("minimax_m2 float64 tensor byte length is invalid: " + ref.Name)
		}
		out := make([]float32, int(ref.Elements))
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	default:
		return nil, core.NewError("minimax_m2 tensor dtype is not supported: " + ref.Name)
	}
}

func validateMiniMaxM2NativePackedPayload(ref miniMaxM2NativePackedTensorPayloadRef, packed []byte, scales, biases []float32, groupSize int) error {
	if int64(len(packed)) != ref.PackedBytes {
		return core.NewError(core.Sprintf("minimax_m2 payload %s packed length %d, expected %d", ref.Name, len(packed), ref.PackedBytes))
	}
	elements := uint64(1)
	for _, dim := range ref.LogicalShape {
		elements *= dim
	}
	expectedGroups := int((elements + uint64(groupSize) - 1) / uint64(groupSize))
	if len(scales) != expectedGroups {
		return core.NewError(core.Sprintf("minimax_m2 payload %s scale count %d, expected %d", ref.Name, len(scales), expectedGroups))
	}
	if len(biases) != expectedGroups {
		return core.NewError(core.Sprintf("minimax_m2 payload %s bias count %d, expected %d", ref.Name, len(biases), expectedGroups))
	}
	return nil
}

func miniMaxM2NativeInt32Shape(shape []uint64) ([]int32, error) {
	if len(shape) == 0 {
		return nil, core.NewError("minimax_m2 native projection shape is required")
	}
	out := make([]int32, len(shape))
	for i, dim := range shape {
		if dim == 0 || dim > uint64(^uint32(0)>>1) {
			return nil, core.NewError("minimax_m2 native projection shape is invalid")
		}
		out[i] = int32(dim)
	}
	return out, nil
}

func findMiniMaxM2NativeTensorRef(tensors map[string]miniMaxM2SafetensorTensorRef, candidates []string) (miniMaxM2SafetensorTensorRef, bool) {
	for _, candidate := range candidates {
		if ref, ok := tensors[candidate]; ok {
			return ref, true
		}
	}
	return miniMaxM2SafetensorTensorRef{}, false
}

func miniMaxM2NativePackedBytes(shape []uint64, bits int) int64 {
	if bits <= 0 {
		bits = 8
	}
	elements := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0
		}
		elements *= dim
	}
	return int64((elements*uint64(bits) + 7) / 8)
}

func miniMaxM2NativeAttentionBits(jang miniMaxM2JANGLoadConfig) int {
	if jang.MXTQBits.Attention > 0 {
		return jang.MXTQBits.Attention
	}
	return 8
}

func miniMaxM2NativeRoutedExpertBits(jang miniMaxM2JANGLoadConfig) int {
	if jang.MXTQBits.RoutedExpert > 0 {
		return jang.MXTQBits.RoutedExpert
	}
	if jang.Quantization.BitsDefault > 0 {
		return jang.Quantization.BitsDefault
	}
	return 2
}

func miniMaxM2NativePackedDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "U8", "UINT8":
		return true
	default:
		return false
	}
}

func miniMaxM2NativeFloatDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "F16", "BF16", "F32", "F64":
		return true
	default:
		return false
	}
}

func miniMaxM2NativeDTypeBytes(dtype string) int64 {
	switch core.Upper(dtype) {
	case "F16", "BF16":
		return 2
	case "F32":
		return 4
	case "F64":
		return 8
	default:
		return 0
	}
}

func sameMiniMaxM2Uint64Slice(a, b []uint64) bool {
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

func miniMaxM2NativeUniqueExpertIDs(ids []int) []int {
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

func miniMaxM2NativeSoftmaxWeights(scores []float32, ids []int) []float32 {
	if len(ids) == 0 {
		return nil
	}
	maxScore := scores[ids[0]]
	for _, id := range ids[1:] {
		if scores[id] > maxScore {
			maxScore = scores[id]
		}
	}
	weights := make([]float32, len(ids))
	sum := float64(0)
	for i, id := range ids {
		value := math.Exp(float64(scores[id] - maxScore))
		weights[i] = float32(value)
		sum += value
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		uniform := float32(1.0 / float64(len(ids)))
		for i := range weights {
			weights[i] = uniform
		}
		return weights
	}
	for i := range weights {
		weights[i] = float32(float64(weights[i]) / sum)
	}
	return weights
}

func miniMaxM2NativeFloat16ToFloat32(value uint16) float32 {
	sign := uint32(value>>15) & 0x1
	exp := int((value >> 10) & 0x1f)
	frac := uint32(value & 0x03ff)
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for (frac & 0x0400) == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03ff
	} else if exp == 31 {
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}

func trimMiniMaxM2NativeWeightSuffix(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")]
	}
	return name
}

func trimMiniMaxM2NativePackedSuffix(name string) string {
	for _, suffix := range []string{".packed", ".qweight"} {
		if core.HasSuffix(name, suffix) {
			return name[:len(name)-len(suffix)]
		}
	}
	return name
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func readMiniMaxM2JANGLoadConfig(root string) miniMaxM2JANGLoadConfig {
	var cfg miniMaxM2JANGLoadConfig
	read := core.ReadFile(core.JoinPath(root, "jang_config.json"))
	if !read.OK {
		return cfg
	}
	_ = core.JSONUnmarshal(read.Value.([]byte), &cfg)
	return cfg
}

func firstMiniMaxM2ArchitectureName(values []string) string {
	for _, value := range values {
		if core.Contains(value, "MiniMaxM2") {
			return "minimax_m2"
		}
	}
	return ""
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmptyUpper(values ...string) string {
	for _, value := range values {
		if value != "" {
			return core.Upper(value)
		}
	}
	return ""
}
