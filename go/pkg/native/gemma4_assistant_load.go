// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/pkg/tokenizer"
)

const nativeGemma4AssistantLogitsFloor = -3.4028234663852886e38
const nativeGemma4AssistantDefaultDraftTokens = 4

// Gemma4AssistantConfig is the assistant-only MTP drafter config used by the
// native loader. It mirrors pkg/metal's assistant config without materialising
// cgo-backed arrays.
type Gemma4AssistantConfig struct {
	ModelType                string
	BackboneHiddenSize       int
	NumCentroids             int
	CentroidIntermediateTopK int
	UseOrderedEmbeddings     bool
	TextConfig               g4.Config
}

// Gemma4AssistantModel is the native, CGO-free assistant-only checkpoint
// handle. The decode integration uses the mmap-backed tensors directly in a
// later slice; this loader owns the mmap and validates the attached-drafter
// tensor layout up front.
type Gemma4AssistantModel struct {
	Config                   Gemma4AssistantConfig
	Arch                     model.Arch
	Tensors                  map[string]safetensors.Tensor
	BackboneHiddenSize       int
	NumCentroids             int
	CentroidIntermediateTopK int
	UseOrderedEmbeddings     bool
	Tok                      *tokenizer.Tokenizer

	mapping *safetensors.DirMapping
}

// Gemma4AssistantPair is a native target-architecture plus assistant drafter
// compatibility record. Runtime decode attachment is layered on top of this:
// this type proves the two checkpoint configs can share target K/V streams.
type Gemma4AssistantPair struct {
	TargetArch model.Arch
	Assistant  *Gemma4AssistantModel
}

// Gemma4AssistantDraftStepResult is one native assistant proposal from a target
// token, previous target hidden state, and target K/V streams.
type Gemma4AssistantDraftStepResult struct {
	Logits []byte
	Token  int32
	Hidden []byte
}

// Gemma4AssistantDraftBlockResult is a chained native assistant proposal block.
type Gemma4AssistantDraftBlockResult struct {
	Tokens []int32
	Hidden []byte
}

// Gemma4AssistantVerifyResult reports target-side verification of a proposed
// assistant draft block against a native target session. Logits and Hidden are
// caller-owned CPU byte copies.
type Gemma4AssistantVerifyResult struct {
	DraftedTokens    []int32
	TargetTokens     []int32
	AcceptedTokens   []int32
	RejectedTokens   []int32
	AcceptedCount    int
	RejectedCount    int
	ReplacementToken int32
	AllAccepted      bool
	Logits           []byte
	Hidden           []byte
}

// Gemma4AssistantGenerateResult records one native greedy assistant generation
// run over an ArchSession target.
type Gemma4AssistantGenerateResult struct {
	Tokens             []int32
	PromptTokens       int
	TargetTokens       int
	DraftTokens        int
	AcceptedTokens     int
	RejectedTokens     int
	TargetVerifyCalls  int
	TargetCalls        int
	DraftCalls         int
	DraftTokenSchedule []int
}

// Gemma4AssistantTokenSink receives each verified token as the native assistant
// generation loop emits it. Returning false stops generation without error.
type Gemma4AssistantTokenSink func(int32) bool

// Gemma4AssistantTargetKV is a native byte-view of a target K/V stream that the
// assistant can attend to by target layer type.
type Gemma4AssistantTargetKV struct {
	Key     []byte
	Value   []byte
	Offset  int
	Length  int
	KVHeads int
	HeadDim int
}

func (kv Gemma4AssistantTargetKV) HasState() bool {
	return len(kv.Key) > 0 && len(kv.Value) > 0 && kv.Length > 0
}

// Gemma4AssistantKVEntry binds a Gemma 4 layer type to a target K/V byte stream.
type Gemma4AssistantKVEntry struct {
	LayerType string
	KV        Gemma4AssistantTargetKV
}

// Gemma4AssistantTargetKVByType is the native equivalent of pkg/metal's tiny
// layer-type lookup for assistant draft steps. The key set is normally just
// "sliding_attention" and "full_attention", so a slice scan is enough.
type Gemma4AssistantTargetKVByType struct {
	entries []Gemma4AssistantKVEntry
}

func (m *Gemma4AssistantTargetKVByType) set(layerType string, targetKV Gemma4AssistantTargetKV) {
	for i := range m.entries {
		if m.entries[i].LayerType == layerType {
			m.entries[i].KV = targetKV
			return
		}
	}
	if m.entries == nil {
		m.entries = make([]Gemma4AssistantKVEntry, 0, 2)
	}
	m.entries = append(m.entries, Gemma4AssistantKVEntry{LayerType: layerType, KV: targetKV})
}

func (m Gemma4AssistantTargetKVByType) Get(layerType string) (Gemma4AssistantTargetKV, bool) {
	for i := range m.entries {
		if m.entries[i].LayerType == layerType {
			return m.entries[i].KV, true
		}
	}
	return Gemma4AssistantTargetKV{}, false
}

// LoadGemma4AssistantDir loads a Gemma 4 assistant-only drafter checkpoint
// without pkg/metal. The returned tensors are mmap-backed; call Close when the
// assistant runtime no longer needs them.
func LoadGemma4AssistantDir(dir string) (*Gemma4AssistantModel, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Load", "read config.json", err)
	}
	cfg, err := parseNativeGemma4AssistantConfig([]byte(cfgStr))
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Load", "parse config", err)
	}
	arch, err := cfg.TextConfig.Arch()
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Load", "derive arch", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Load", "load tokenizer", err)
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Load", "load weights", err)
	}
	m := &Gemma4AssistantModel{
		Config:                   cfg,
		Arch:                     arch,
		Tensors:                  dm.Tensors,
		BackboneHiddenSize:       cfg.BackboneHiddenSize,
		NumCentroids:             cfg.NumCentroids,
		CentroidIntermediateTopK: cfg.CentroidIntermediateTopK,
		UseOrderedEmbeddings:     cfg.UseOrderedEmbeddings,
		Tok:                      tok,
		mapping:                  dm,
	}
	if err := validateNativeGemma4AssistantModel(m); err != nil {
		_ = m.Close()
		return nil, core.E("native.gemma4.assistant.Load", "validate tensors", err)
	}
	return m, nil
}

// LoadGemma4AssistantPairDirs loads assistant metadata/tensors and validates
// them against the target checkpoint config without loading the target weights.
func LoadGemma4AssistantPairDirs(targetDir, assistantDir string) (*Gemma4AssistantPair, error) {
	if core.Trim(targetDir) == "" {
		return nil, core.NewError("native.gemma4.assistant pair target path is required")
	}
	if core.Trim(assistantDir) == "" {
		return nil, core.NewError("native.gemma4.assistant pair assistant path is required")
	}
	targetArch, err := loadNativeGemma4TargetArch(targetDir)
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Pair", "load target config", err)
	}
	assistant, err := LoadGemma4AssistantDir(assistantDir)
	if err != nil {
		return nil, core.E("native.gemma4.assistant.Pair", "load assistant", err)
	}
	pair := &Gemma4AssistantPair{TargetArch: targetArch, Assistant: assistant}
	if err := validateNativeGemma4AssistantPair(pair); err != nil {
		_ = pair.Close()
		return nil, core.E("native.gemma4.assistant.Pair", "validate attachment", err)
	}
	return pair, nil
}

func loadNativeGemma4TargetArch(dir string) (model.Arch, error) {
	mt, cfg, err := model.ProbeDirArch(dir)
	if err != nil {
		return model.Arch{}, err
	}
	textMT, nestedTextMT := model.ProbeModelTypes(cfg)
	if textMT != "" {
		mt = textMT
	}
	spec, ok := model.LookupArch(mt)
	if !ok && nestedTextMT != "" {
		spec, ok = model.LookupArch(nestedTextMT)
	}
	if !ok {
		return model.Arch{}, core.NewError("native.gemma4.assistant target has no registered architecture: " + mt)
	}
	ac, err := spec.Parse(cfg)
	if err != nil {
		return model.Arch{}, err
	}
	arch, err := ac.Arch()
	if err != nil {
		return model.Arch{}, err
	}
	if arch.Hidden <= 0 || len(arch.Layer) == 0 {
		return model.Arch{}, core.NewError("native.gemma4.assistant target arch is incomplete")
	}
	return arch, nil
}

func parseNativeGemma4AssistantConfig(data []byte) (Gemma4AssistantConfig, error) {
	var raw struct {
		ModelType                string    `json:"model_type"`
		BackboneHiddenSize       int       `json:"backbone_hidden_size"`
		NumCentroids             int       `json:"num_centroids"`
		CentroidIntermediateTopK int       `json:"centroid_intermediate_top_k"`
		UseOrderedEmbeddings     bool      `json:"use_ordered_embeddings"`
		TextConfig               g4.Config `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return Gemma4AssistantConfig{}, core.NewError("gemma4.assistant config parse failed: " + r.Error())
	}
	textConfig := raw.TextConfig
	if textConfig.HiddenSize <= 0 && textConfig.NumHiddenLayers <= 0 {
		var flatText g4.Config
		if r := core.JSONUnmarshal(data, &flatText); !r.OK {
			return Gemma4AssistantConfig{}, core.NewError("gemma4.assistant config parse failed: " + r.Error())
		}
		if flatText.HiddenSize > 0 || flatText.NumHiddenLayers > 0 {
			textConfig = flatText
		}
	}
	cfg := Gemma4AssistantConfig{
		ModelType:                raw.ModelType,
		BackboneHiddenSize:       raw.BackboneHiddenSize,
		NumCentroids:             raw.NumCentroids,
		CentroidIntermediateTopK: raw.CentroidIntermediateTopK,
		UseOrderedEmbeddings:     raw.UseOrderedEmbeddings,
		TextConfig:               textConfig,
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_assistant"
	}
	if err := validateNativeGemma4AssistantConfig(cfg); err != nil {
		return Gemma4AssistantConfig{}, err
	}
	return cfg, nil
}

func validateNativeGemma4AssistantConfig(cfg Gemma4AssistantConfig) error {
	if cfg.ModelType != "gemma4_assistant" && cfg.ModelType != "gemma4_unified_assistant" {
		return core.NewError("gemma4.assistant config has unsupported model_type: " + cfg.ModelType)
	}
	if cfg.BackboneHiddenSize <= 0 {
		return core.NewError("gemma4.assistant config has invalid backbone_hidden_size")
	}
	if cfg.TextConfig.HiddenSize <= 0 {
		return core.NewError("gemma4.assistant config has invalid hidden_size")
	}
	if cfg.TextConfig.NumHiddenLayers <= 0 {
		return core.NewError("gemma4.assistant config has invalid num_hidden_layers")
	}
	if cfg.TextConfig.NumAttentionHeads <= 0 {
		return core.NewError("gemma4.assistant config has invalid num_attention_heads")
	}
	if cfg.TextConfig.HeadDim <= 0 {
		return core.NewError("gemma4.assistant config has invalid head_dim")
	}
	if cfg.UseOrderedEmbeddings && cfg.NumCentroids <= 0 {
		return core.NewError("gemma4.assistant ordered embeddings require num_centroids")
	}
	return nil
}

func validateNativeGemma4AssistantPair(pair *Gemma4AssistantPair) error {
	if pair == nil || pair.TargetArch.Hidden <= 0 {
		return core.NewError("gemma4.assistant pair target is nil")
	}
	assistant := pair.Assistant
	if assistant == nil {
		return core.NewError("gemma4.assistant pair assistant is nil")
	}
	target := pair.TargetArch
	if assistant.BackboneHiddenSize != target.Hidden {
		return core.NewError(core.Sprintf("gemma4.assistant backbone_hidden_size = %d, want target hidden_size %d", assistant.BackboneHiddenSize, target.Hidden))
	}
	if target.Vocab > 0 && assistant.Arch.Vocab > 0 && target.Vocab != assistant.Arch.Vocab {
		return core.NewError(core.Sprintf("gemma4.assistant vocab_size = %d, want target vocab_size %d", assistant.Arch.Vocab, target.Vocab))
	}
	return validateNativeGemma4AssistantTargetTypes(target, assistant)
}

func validateNativeGemma4AssistantTargetTypes(target model.Arch, assistant *Gemma4AssistantModel) error {
	targetTypes := map[string]int{}
	for _, layer := range target.Layer {
		layerType := nativeGemma4LayerType(layer)
		if layerType != "" {
			if _, ok := targetTypes[layerType]; !ok {
				targetTypes[layerType] = layer.HeadDim
			}
		}
	}
	if len(targetTypes) == 0 {
		return core.NewError("gemma4.assistant pair target layer types are unavailable")
	}
	for idx, layer := range assistant.Arch.Layer {
		layerType := nativeGemma4AssistantLayerType(assistant, idx, layer)
		if _, ok := targetTypes[layerType]; !ok {
			return core.NewError(core.Sprintf("gemma4.assistant layer %d type %q has no target K/V stream", idx, layerType))
		}
		wantHeadDim := targetTypes[layerType]
		if wantHeadDim > 0 && layer.HeadDim != wantHeadDim {
			return core.NewError(core.Sprintf("gemma4.assistant layer %d head_dim = %d, want target %s head_dim %d", idx, layer.HeadDim, layerType, wantHeadDim))
		}
	}
	return nil
}

func nativeGemma4AssistantLayerType(assistant *Gemma4AssistantModel, idx int, layer model.LayerSpec) string {
	if assistant != nil && idx >= 0 && idx < len(assistant.Config.TextConfig.LayerTypes) {
		if layerType := assistant.Config.TextConfig.LayerTypes[idx]; layerType != "" {
			return layerType
		}
	}
	return nativeGemma4LayerType(layer)
}

func nativeGemma4LayerType(layer model.LayerSpec) string {
	if layer.Attention == model.SlidingAttention {
		return "sliding_attention"
	}
	return "full_attention"
}

func validateNativeGemma4AssistantModel(m *Gemma4AssistantModel) error {
	if m == nil {
		return core.NewError("gemma4.assistant model is nil")
	}
	var missing []string
	addMissing := func(name string) {
		t, ok := m.Tensors[name]
		if !ok || t.Dtype == "" || len(t.Data) == 0 {
			missing = append(missing, name)
		}
	}
	addAnyMissing := func(label string, names ...string) {
		for _, name := range names {
			t, ok := m.Tensors[name]
			if ok && t.Dtype != "" && len(t.Data) > 0 {
				return
			}
		}
		missing = append(missing, label)
	}
	addLinearMissing := func(name string) { addMissing(name + ".weight") }
	addNormMissing := func(name string) { addMissing(name + ".weight") }

	addMissing("model.embed_tokens.weight")
	addNormMissing("model.norm")
	addLinearMissing("pre_projection")
	addLinearMissing("post_projection")
	if m.UseOrderedEmbeddings {
		addLinearMissing("masked_embedding.centroids")
		addMissing("masked_embedding.token_ordering")
	}
	for i := range m.Arch.Layer {
		prefix := core.Sprintf("model.layers.%d", i)
		addNormMissing(prefix + ".input_layernorm")
		addNormMissing(prefix + ".post_attention_layernorm")
		addNormMissing(prefix + ".pre_feedforward_layernorm")
		addNormMissing(prefix + ".post_feedforward_layernorm")
		addAnyMissing(prefix+".layer_scalar", prefix+".layer_scalar", prefix+".layer_scalar.weight")
		addLinearMissing(prefix + ".self_attn.q_proj")
		addLinearMissing(prefix + ".self_attn.o_proj")
		addNormMissing(prefix + ".self_attn.q_norm")
		addLinearMissing(prefix + ".mlp.gate_proj")
		addLinearMissing(prefix + ".mlp.up_proj")
		addLinearMissing(prefix + ".mlp.down_proj")
	}
	if len(missing) > 0 {
		return core.NewError("missing required tensors: " + core.Join(", ", missing...))
	}
	if err := validateNativeGemma4AssistantProjectionShapes(m); err != nil {
		return err
	}
	if err := validateNativeGemma4AssistantOrderedEmbeddingShape(m); err != nil {
		return err
	}
	return nil
}

func validateNativeGemma4AssistantProjectionShapes(m *Gemma4AssistantModel) error {
	if err := validateNativeGemma4AssistantLinearShape(m, "pre_projection", m.Arch.Hidden, m.BackboneHiddenSize*2); err != nil {
		return err
	}
	if err := validateNativeGemma4AssistantLinearShape(m, "post_projection", m.BackboneHiddenSize, m.Arch.Hidden); err != nil {
		return err
	}
	if m.UseOrderedEmbeddings {
		if err := validateNativeGemma4AssistantLinearShape(m, "masked_embedding.centroids", m.NumCentroids, m.Arch.Hidden); err != nil {
			return err
		}
	}
	return nil
}

func validateNativeGemma4AssistantLinearShape(m *Gemma4AssistantModel, name string, out, in int) error {
	t, ok := m.Tensors[name+".weight"]
	if !ok {
		return nil
	}
	if len(t.Shape) < 2 {
		return core.NewError(name + ".weight has invalid rank")
	}
	gotOut := t.Shape[len(t.Shape)-2]
	gotIn := t.Shape[len(t.Shape)-1]
	if out > 0 && gotOut != out {
		return core.NewError(core.Sprintf("%s.weight output dim = %d, want %d", name, gotOut, out))
	}
	if in > 0 && !nativeGemma4AssistantLinearInputMatches(m, name, gotIn, in) {
		return core.NewError(core.Sprintf("%s.weight input dim = %d, want %d", name, gotIn, in))
	}
	return nil
}

func nativeGemma4AssistantLinearInputMatches(m *Gemma4AssistantModel, name string, gotIn, wantIn int) bool {
	if gotIn == wantIn {
		return true
	}
	quant := m.Config.TextConfig.ResolvedQuant()
	if quant == nil {
		return false
	}
	_, bits := quant.For(name)
	if bits <= 0 {
		return false
	}
	if _, ok := m.Tensors[name+".scales"]; !ok {
		return false
	}
	packFactor := 32 / bits
	if packFactor > 0 && wantIn%packFactor == 0 && gotIn == wantIn/packFactor {
		return true
	}
	return gotIn == (wantIn*bits+31)/32
}

func validateNativeGemma4AssistantOrderedEmbeddingShape(m *Gemma4AssistantModel) error {
	if !m.UseOrderedEmbeddings {
		return nil
	}
	t, ok := m.Tensors["masked_embedding.token_ordering"]
	if !ok {
		return nil
	}
	switch t.Dtype {
	case "I32", "I64":
	default:
		return core.NewError("masked_embedding.token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
	vocabSize := m.Arch.Vocab
	numCentroids := m.NumCentroids
	if vocabSize <= 0 || numCentroids <= 0 || vocabSize%numCentroids != 0 {
		return core.NewError("masked_embedding.token_ordering requires vocab_size divisible by num_centroids")
	}
	tokensPerCentroid := vocabSize / numCentroids
	if len(t.Shape) == 1 && t.Shape[0] == vocabSize {
		return nil
	}
	if len(t.Shape) == 2 && t.Shape[0] == numCentroids && t.Shape[1] == tokensPerCentroid {
		return nil
	}
	return core.NewError(core.Sprintf("masked_embedding.token_ordering shape = %v, want [%d] or [%d %d]", t.Shape, vocabSize, numCentroids, tokensPerCentroid))
}

func (m *Gemma4AssistantModel) Close() error {
	if m == nil || m.mapping == nil {
		return nil
	}
	err := m.mapping.Close()
	m.mapping = nil
	m.Tensors = nil
	return err
}

func (m *Gemma4AssistantModel) ModelType() string {
	if m == nil {
		return ""
	}
	return "gemma4_assistant"
}

func (m *Gemma4AssistantModel) Tokenizer() *tokenizer.Tokenizer {
	if m == nil {
		return nil
	}
	return m.Tok
}

func (m *Gemma4AssistantModel) NumLayers() int {
	if m == nil {
		return 0
	}
	return len(m.Arch.Layer)
}

func (m *Gemma4AssistantModel) Tensor(name string) (safetensors.Tensor, bool) {
	if m == nil {
		return safetensors.Tensor{}, false
	}
	t, ok := m.Tensors[name]
	return t, ok
}

func (pair *Gemma4AssistantPair) TargetKVByLayerType(targetKVs []Gemma4AssistantTargetKV) (Gemma4AssistantTargetKVByType, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step requires a validated pair")
	}
	var out Gemma4AssistantTargetKVByType
	for layerIdx, layer := range pair.TargetArch.Layer {
		layerType := nativeGemma4LayerType(layer)
		if layerType == "" {
			continue
		}
		ownerIdx := layerIdx
		if layer.KVShareFrom >= 0 {
			ownerIdx = layer.KVShareFrom
		}
		if ownerIdx < 0 || ownerIdx >= len(pair.TargetArch.Layer) {
			continue
		}
		cacheIdx := pair.TargetArch.Layer[ownerIdx].CacheIndex
		if cacheIdx < 0 || cacheIdx >= len(targetKVs) {
			continue
		}
		targetKV := targetKVs[cacheIdx]
		if !targetKV.HasState() {
			return Gemma4AssistantTargetKVByType{}, core.NewError(core.Sprintf("gemma4.assistant draft step target layer %d has empty K/V stream", layerIdx))
		}
		out.set(layerType, targetKV)
	}
	for idx, layer := range pair.Assistant.Arch.Layer {
		layerType := nativeGemma4AssistantLayerType(pair.Assistant, idx, layer)
		targetKV, ok := out.Get(layerType)
		if !ok || !targetKV.HasState() {
			return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step missing populated target K/V stream for " + layerType)
		}
	}
	return out, nil
}

// TargetKVByLayerTypeFromSession maps the target session's resident K/V cache
// rows to the assistant's layer-type streams. ArchSession stores K/V rows
// token-major; the assistant attention primitive consumes head-major slabs, so
// this materialises the visible cache window in assistant-ready order.
func (pair *Gemma4AssistantPair) TargetKVByLayerTypeFromSession(target *ArchSession) (Gemma4AssistantTargetKVByType, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step requires a validated pair")
	}
	if target == nil {
		return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step target session is nil")
	}
	if target.pos <= 0 {
		return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step target session cache is empty")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return Gemma4AssistantTargetKVByType{}, err
	}
	views, err := target.stateLayerViews()
	if err != nil {
		return Gemma4AssistantTargetKVByType{}, err
	}
	maxCacheIndex := -1
	for _, view := range views {
		if view.cacheIndex > maxCacheIndex {
			maxCacheIndex = view.cacheIndex
		}
	}
	if maxCacheIndex < 0 {
		return Gemma4AssistantTargetKVByType{}, core.NewError("gemma4.assistant draft step target session has no K/V cache owners")
	}
	targetKVs := make([]Gemma4AssistantTargetKV, maxCacheIndex+1)
	for _, view := range views {
		if view.cacheIndex < 0 {
			continue
		}
		start, tokenCount, err := nativeKVLayerCaptureWindow(view, target.pos)
		if err != nil {
			return Gemma4AssistantTargetKVByType{}, err
		}
		keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, target.pos)
		if err != nil {
			return Gemma4AssistantTargetKVByType{}, err
		}
		if len(keyRows) == 0 || len(valueRows) == 0 {
			return Gemma4AssistantTargetKVByType{}, core.NewError(core.Sprintf("gemma4.assistant draft step target layer %d has empty K/V stream", view.layer))
		}
		keySlab := make([]byte, len(keyRows))
		valueSlab := make([]byte, len(valueRows))
		nativeKVTokenRowsToLayerSlab(keySlab, keyRows, tokenCount, view.kvHeads, view.headDim)
		nativeKVTokenRowsToLayerSlab(valueSlab, valueRows, tokenCount, view.kvHeads, view.headDim)
		targetKVs[view.cacheIndex] = Gemma4AssistantTargetKV{
			Key:     keySlab,
			Value:   valueSlab,
			Offset:  start,
			Length:  tokenCount,
			KVHeads: view.kvHeads,
			HeadDim: view.headDim,
		}
	}
	return pair.TargetKVByLayerType(targetKVs)
}

func (pair *Gemma4AssistantPair) validateTargetSessionArch(arch model.Arch) error {
	target := pair.TargetArch
	if target.Hidden <= 0 || arch.Hidden <= 0 || target.Hidden != arch.Hidden {
		return core.NewError(core.Sprintf("gemma4.assistant target session hidden_size = %d, want %d", arch.Hidden, target.Hidden))
	}
	if target.Vocab > 0 && arch.Vocab > 0 && target.Vocab != arch.Vocab {
		return core.NewError(core.Sprintf("gemma4.assistant target session vocab_size = %d, want %d", arch.Vocab, target.Vocab))
	}
	if len(target.Layer) == 0 || len(arch.Layer) != len(target.Layer) {
		return core.NewError(core.Sprintf("gemma4.assistant target session layer count = %d, want %d", len(arch.Layer), len(target.Layer)))
	}
	for idx := range target.Layer {
		want := target.Layer[idx]
		got := arch.Layer[idx]
		if got.Attention != want.Attention || got.KVShareFrom != want.KVShareFrom || got.CacheIndex != want.CacheIndex {
			return core.NewError(core.Sprintf("gemma4.assistant target session layer %d cache topology mismatch", idx))
		}
		if want.HeadDim > 0 && got.HeadDim > 0 && got.HeadDim != want.HeadDim {
			return core.NewError(core.Sprintf("gemma4.assistant target session layer %d head_dim = %d, want %d", idx, got.HeadDim, want.HeadDim))
		}
		if want.KVHeads > 0 && got.KVHeads > 0 && got.KVHeads != want.KVHeads {
			return core.NewError(core.Sprintf("gemma4.assistant target session layer %d kv_heads = %d, want %d", idx, got.KVHeads, want.KVHeads))
		}
	}
	return nil
}

func (m *Gemma4AssistantModel) DraftInputProjection(tokenEmbedding, previousHidden []byte) ([]byte, error) {
	return m.DraftInputProjectionInto(nil, tokenEmbedding, previousHidden)
}

func (m *Gemma4AssistantModel) DraftInputProjectionInto(out []byte, tokenEmbedding, previousHidden []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("gemma4.assistant draft input model is nil")
	}
	backbone := m.BackboneHiddenSize
	hidden := m.Arch.Hidden
	if backbone <= 0 || hidden <= 0 {
		return nil, core.NewError("gemma4.assistant draft input has incomplete dimensions")
	}
	backboneBytes := backbone * bf16Size
	if len(tokenEmbedding) != backboneBytes {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft input token embedding bytes = %d, want %d", len(tokenEmbedding), backboneBytes))
	}
	if len(previousHidden) != backboneBytes {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft input previous hidden bytes = %d, want %d", len(previousHidden), backboneBytes))
	}
	weight, ok := m.Tensors["pre_projection.weight"]
	if !ok {
		return nil, core.NewError("gemma4.assistant draft input missing pre_projection.weight")
	}
	if weight.Dtype != "BF16" {
		return nil, core.NewError("gemma4.assistant draft input pre_projection.weight dtype = " + weight.Dtype + ", want BF16")
	}
	input := backbone * 2
	if len(weight.Shape) < 2 || weight.Shape[len(weight.Shape)-2] != hidden || weight.Shape[len(weight.Shape)-1] != input {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft input pre_projection.weight shape = %v, want [%d %d]", weight.Shape, hidden, input))
	}
	if len(weight.Data) != hidden*input*bf16Size {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft input pre_projection.weight bytes = %d, want %d", len(weight.Data), hidden*input*bf16Size))
	}
	combined := make([]byte, input*bf16Size)
	copy(combined, tokenEmbedding)
	copy(combined[backboneBytes:], previousHidden)
	return MatMulBF16NTInto(out, combined, weight.Data, 1, input, hidden)
}

func (pair *Gemma4AssistantPair) DraftInputProjectionForToken(targetEmbed []byte, lastToken int32, previousHidden []byte) ([]byte, error) {
	return pair.DraftInputProjectionForTokenInto(nil, targetEmbed, lastToken, previousHidden)
}

func (pair *Gemma4AssistantPair) DraftInputProjectionForTokenInto(out []byte, targetEmbed []byte, lastToken int32, previousHidden []byte) ([]byte, error) {
	target, err := pair.validateDraftInputTarget()
	if err != nil {
		return nil, err
	}
	tokenEmbedding := make([]byte, target.Hidden*bf16Size)
	if _, err := embedTokenBF16Into(tokenEmbedding, targetEmbed, lastToken, target.Vocab, target.Hidden, nativeGemma4EmbeddingScale(target)); err != nil {
		return nil, core.E("gemma4.assistant draft input", "target token embedding", err)
	}
	return pair.Assistant.DraftInputProjectionInto(out, tokenEmbedding, previousHidden)
}

func (pair *Gemma4AssistantPair) DraftInputProjectionForTokenQuant(packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte) ([]byte, error) {
	return pair.DraftInputProjectionForTokenQuantInto(nil, packed, scales, biases, groupSize, bits, lastToken, previousHidden)
}

func (pair *Gemma4AssistantPair) DraftInputProjectionForTokenQuantInto(out []byte, packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte) ([]byte, error) {
	target, err := pair.validateDraftInputTarget()
	if err != nil {
		return nil, err
	}
	tokenEmbedding := make([]byte, target.Hidden*bf16Size)
	if _, err := embedTokenQuantInto(tokenEmbedding, packed, scales, biases, lastToken, target.Vocab, target.Hidden, groupSize, bits, nativeGemma4EmbeddingScale(target)); err != nil {
		return nil, core.E("gemma4.assistant draft input", "target quant token embedding", err)
	}
	return pair.Assistant.DraftInputProjectionInto(out, tokenEmbedding, previousHidden)
}

func (pair *Gemma4AssistantPair) DraftStep(targetEmbed []byte, lastToken int32, previousHidden []byte, targetKVs Gemma4AssistantTargetKVByType, suppressTokens ...[]int32) (Gemma4AssistantDraftStepResult, error) {
	if lastToken < 0 {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step token is invalid")
	}
	projected, err := pair.DraftInputProjectionForToken(targetEmbed, lastToken, previousHidden)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	return pair.draftStepFromProjected(projected, targetKVs, suppressTokens...)
}

func (pair *Gemma4AssistantPair) DraftStepQuant(packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte, targetKVs Gemma4AssistantTargetKVByType, suppressTokens ...[]int32) (Gemma4AssistantDraftStepResult, error) {
	if lastToken < 0 {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step token is invalid")
	}
	projected, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, lastToken, previousHidden)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	return pair.draftStepFromProjected(projected, targetKVs, suppressTokens...)
}

// DraftStepFromSession drafts one assistant token from a target ArchSession
// boundary. The target session must already hold the accepted prefix in its
// resident cache and retainedHidden boundary.
func (pair *Gemma4AssistantPair) DraftStepFromSession(target *ArchSession, lastToken int32, suppressTokens ...[]int32) (Gemma4AssistantDraftStepResult, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step requires a validated pair")
	}
	if lastToken < 0 {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step token is invalid")
	}
	if target == nil {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step target session is nil")
	}
	if target.embed == nil && target.embedInto == nil {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step target session has no embedder")
	}
	targetKVs, err := pair.TargetKVByLayerTypeFromSession(target)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	previousHidden, err := target.BoundaryNormedHidden()
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, core.E("gemma4.assistant draft step", "target boundary hidden", err)
	}
	tokenEmbedding, err := target.embedID(lastToken)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, core.E("gemma4.assistant draft step", "target token embedding", err)
	}
	if len(tokenEmbedding) != pair.TargetArch.Hidden*bf16Size {
		return Gemma4AssistantDraftStepResult{}, core.NewError(core.Sprintf("gemma4.assistant draft step target token embedding bytes = %d, want %d", len(tokenEmbedding), pair.TargetArch.Hidden*bf16Size))
	}
	projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, previousHidden)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	return pair.draftStepFromProjected(projected, targetKVs, suppressTokens...)
}

// DraftBlockFromSession chains assistant draft steps from a target ArchSession
// boundary and returns CPU-visible proposed token ids. Verification is a
// separate target-session concern.
func (pair *Gemma4AssistantPair) DraftBlockFromSession(target *ArchSession, lastToken int32, maxDraftTokens int, suppressTokens ...[]int32) (Gemma4AssistantDraftBlockResult, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantDraftBlockResult{}, core.NewError("gemma4.assistant draft block requires a validated pair")
	}
	if maxDraftTokens <= 0 {
		return Gemma4AssistantDraftBlockResult{}, core.NewError("gemma4.assistant draft block maxDraftTokens must be > 0")
	}
	if lastToken < 0 {
		return Gemma4AssistantDraftBlockResult{}, core.NewError("gemma4.assistant draft step token is invalid")
	}
	if target == nil {
		return Gemma4AssistantDraftBlockResult{}, core.NewError("gemma4.assistant draft step target session is nil")
	}
	if target.embed == nil && target.embedInto == nil {
		return Gemma4AssistantDraftBlockResult{}, core.NewError("gemma4.assistant draft step target session has no embedder")
	}
	targetKVs, err := pair.TargetKVByLayerTypeFromSession(target)
	if err != nil {
		return Gemma4AssistantDraftBlockResult{}, err
	}
	currentHidden, err := target.BoundaryNormedHidden()
	if err != nil {
		return Gemma4AssistantDraftBlockResult{}, core.E("gemma4.assistant draft block", "target boundary hidden", err)
	}
	currentToken := lastToken
	tokens := make([]int32, 0, maxDraftTokens)
	for len(tokens) < maxDraftTokens {
		tokenEmbedding, err := target.embedID(currentToken)
		if err != nil {
			return Gemma4AssistantDraftBlockResult{}, core.E("gemma4.assistant draft block", "target token embedding", err)
		}
		if len(tokenEmbedding) != pair.TargetArch.Hidden*bf16Size {
			return Gemma4AssistantDraftBlockResult{}, core.NewError(core.Sprintf("gemma4.assistant draft block target token embedding bytes = %d, want %d", len(tokenEmbedding), pair.TargetArch.Hidden*bf16Size))
		}
		projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, currentHidden)
		if err != nil {
			return Gemma4AssistantDraftBlockResult{}, err
		}
		step, err := pair.draftStepFromProjected(projected, targetKVs, suppressTokens...)
		if err != nil {
			return Gemma4AssistantDraftBlockResult{}, err
		}
		tokens = append(tokens, step.Token)
		currentToken = step.Token
		currentHidden = step.Hidden
	}
	return Gemma4AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

// VerifyDraftBlockFromSession compares assistant draft tokens against the
// target session's greedy continuation, keeps the accepted prefix resident, and
// rolls back any rejected suffix. The caller commits ReplacementToken separately
// on reject, matching pkg/metal's assistant verifier contract.
func (pair *Gemma4AssistantPair) VerifyDraftBlockFromSession(target *ArchSession, draftTokens []int32, suppressTokens ...[]int32) (Gemma4AssistantVerifyResult, error) {
	if pair == nil {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant verify requires a target pair")
	}
	if target == nil {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant verify target session is nil")
	}
	if len(draftTokens) == 0 {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant verify draft tokens are required")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	var suppress []int32
	if len(suppressTokens) > 0 {
		suppress = suppressTokens[0]
	}
	boundaryHidden := append([]byte(nil), target.retainedHidden...)
	boundaryLogits, err := target.BoundaryLogits()
	if err != nil {
		return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant verify", "target boundary logits", err)
	}
	boundaryLogits = append([]byte(nil), boundaryLogits...)
	first, err := greedyBF16Suppressed(boundaryLogits, target.arch.Vocab, suppress)
	if err != nil {
		return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant verify", "target boundary token", err)
	}

	posBefore := target.pos
	result := Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), draftTokens...),
	}
	rows, hiddens, err := target.verifyGemma4AssistantDraftRows(draftTokens, suppress)
	if err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	if len(rows) < len(draftTokens) || len(hiddens) < len(draftTokens) {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant verify target rows are incomplete")
	}

	accepted := 0
	for i, draft := range draftTokens {
		targetToken := first
		if i > 0 {
			targetToken = rows[i-1]
		}
		if i == 0 {
			result.TargetTokens = append(result.TargetTokens, targetToken)
		}
		if targetToken != draft {
			break
		}
		result.AcceptedTokens = append(result.AcceptedTokens, draft)
		accepted++
	}
	result.AcceptedCount = accepted
	result.RejectedCount = len(draftTokens) - accepted
	result.AllAccepted = accepted == len(draftTokens)
	if !result.AllAccepted {
		result.RejectedTokens = append([]int32(nil), draftTokens[accepted:]...)
		result.ReplacementToken = first
		if accepted > 0 {
			result.ReplacementToken = rows[accepted-1]
		}
	}

	target.pos = posBefore + accepted
	if err := target.state.truncateDevicePagedKV(target.pos); err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	target.rememberGemma4AssistantAcceptedIDs(posBefore, result.AcceptedTokens)

	if accepted == 0 {
		target.rememberRetainedHidden(boundaryHidden)
		target.rememberRetainedLogits(boundaryLogits)
		result.Logits = append([]byte(nil), boundaryLogits...)
		return result, nil
	}

	hidden := hiddens[accepted-1]
	if len(hidden) != target.arch.Hidden*bf16Size {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant verify accepted hidden has wrong size")
	}
	logits, err := target.headLogitsScratch(hidden, false)
	if err != nil {
		return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant verify", "accepted logits", err)
	}
	result.Hidden = append([]byte(nil), hidden...)
	result.Logits = append([]byte(nil), logits...)
	target.rememberRetainedHidden(hidden)
	target.rememberRetainedLogits(result.Logits)
	return result, nil
}

// VerifyDraftBlockSampledFromSession compares assistant draft tokens against
// target-sampled decisions from the target session. When carry is true, block[0]
// is an already-emitted replacement token from the previous round and is
// accepted without consuming a sampler draw.
func (pair *Gemma4AssistantPair) VerifyDraftBlockSampledFromSession(target *ArchSession, draftTokens []int32, sampler *model.Sampler, params model.SampleParams, carry bool) (Gemma4AssistantVerifyResult, error) {
	if pair == nil {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify requires a target pair")
	}
	if target == nil {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify target session is nil")
	}
	if len(draftTokens) == 0 {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify draft tokens are required")
	}
	if sampler == nil {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify sampler is nil")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	boundaryHidden := append([]byte(nil), target.retainedHidden...)
	boundaryLogits, err := target.BoundaryLogits()
	if err != nil {
		return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant sampled verify", "target boundary logits", err)
	}
	boundaryLogits = append([]byte(nil), boundaryLogits...)

	posBefore := target.pos
	result := Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), draftTokens...),
	}
	hiddens, err := target.verifyGemma4AssistantDraftHiddens(draftTokens)
	if err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	if len(hiddens) < len(draftTokens) {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify target rows are incomplete")
	}

	accepted := 0
	for i, draft := range draftTokens {
		if i == 0 && carry {
			result.AcceptedTokens = append(result.AcceptedTokens, draft)
			accepted++
			continue
		}
		rowLogits := boundaryLogits
		if i > 0 {
			rowLogits, err = target.headLogitsScratch(hiddens[i-1], false)
			if err != nil {
				return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant sampled verify", "target row logits", err)
			}
		}
		targetToken, err := sampler.Sample(rowLogits, target.arch.Vocab, params)
		if err != nil {
			return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant sampled verify", "sample verifier row", err)
		}
		if len(result.TargetTokens) == 0 {
			result.TargetTokens = append(result.TargetTokens, targetToken)
		}
		if targetToken != draft {
			result.ReplacementToken = targetToken
			break
		}
		result.AcceptedTokens = append(result.AcceptedTokens, draft)
		accepted++
	}
	result.AcceptedCount = accepted
	result.RejectedCount = len(draftTokens) - accepted
	result.AllAccepted = accepted == len(draftTokens)
	if !result.AllAccepted {
		result.RejectedTokens = append([]int32(nil), draftTokens[accepted:]...)
	}

	target.pos = posBefore + accepted
	if err := target.state.truncateDevicePagedKV(target.pos); err != nil {
		return Gemma4AssistantVerifyResult{}, err
	}
	target.rememberGemma4AssistantAcceptedIDs(posBefore, result.AcceptedTokens)

	if accepted == 0 {
		target.rememberRetainedHidden(boundaryHidden)
		target.rememberRetainedLogits(boundaryLogits)
		result.Logits = append([]byte(nil), boundaryLogits...)
		return result, nil
	}

	hidden := hiddens[accepted-1]
	if len(hidden) != target.arch.Hidden*bf16Size {
		return Gemma4AssistantVerifyResult{}, core.NewError("gemma4.assistant sampled verify accepted hidden has wrong size")
	}
	logits, err := target.headLogitsScratch(hidden, false)
	if err != nil {
		return Gemma4AssistantVerifyResult{}, core.E("gemma4.assistant sampled verify", "accepted logits", err)
	}
	result.Hidden = append([]byte(nil), hidden...)
	result.Logits = append([]byte(nil), logits...)
	target.rememberRetainedHidden(hidden)
	target.rememberRetainedLogits(result.Logits)
	return result, nil
}

// GenerateFromSession greedily generates token ids from a native target session
// using this assistant pair for speculative proposals.
func (pair *Gemma4AssistantPair) GenerateFromSession(target *ArchSession, promptIDs []int32, maxNew, eosID, draftTokens int, suppress []int32) (Gemma4AssistantGenerateResult, error) {
	return pair.GenerateFromSessionEach(target, promptIDs, maxNew, eosID, draftTokens, suppress, nil)
}

// GenerateFromSessionEach is GenerateFromSession with per-token streaming.
func (pair *Gemma4AssistantPair) GenerateFromSessionEach(target *ArchSession, promptIDs []int32, maxNew, eosID, draftTokens int, suppress []int32, yield Gemma4AssistantTokenSink) (Gemma4AssistantGenerateResult, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation requires a validated pair")
	}
	if target == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation target session is nil")
	}
	if len(promptIDs) == 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation prompt tokens are required")
	}
	if maxNew <= 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation maxNew must be > 0")
	}
	draftTokens = nativeGemma4AssistantResolveDraftTokens(draftTokens)
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if err := target.prepareGemma4AssistantPrompt(promptIDs); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}

	result := Gemma4AssistantGenerateResult{
		PromptTokens:       len(promptIDs),
		DraftTokenSchedule: make([]int, 0, (maxNew+draftTokens-1)/draftTokens),
	}
	lastToken := promptIDs[len(promptIDs)-1]
	carryLead := int32(-1)
	stopped := false
	for len(result.Tokens) < maxNew && !stopped {
		remaining := maxNew - len(result.Tokens)
		blockSize := draftTokens
		if blockSize > remaining {
			blockSize = remaining
		}
		if (yield != nil || eosID >= 0) && blockSize > 1 {
			blockSize = 1
		}
		draft, err := pair.DraftBlockFromSession(target, lastToken, blockSize, suppress)
		if err != nil {
			return result, err
		}
		result.DraftCalls++
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		block := draft.Tokens
		carryPresent := carryLead >= 0
		if carryPresent {
			block = append([]int32{carryLead}, draft.Tokens...)
		}
		verify, err := pair.VerifyDraftBlockFromSession(target, block, suppress)
		if err != nil {
			return result, err
		}
		result.TargetVerifyCalls++
		result.TargetCalls++
		emitStart := 0
		if carryPresent && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
			emitStart = 1
			carryLead = -1
		}
		newDrafts := 0
		result.RejectedTokens += verify.RejectedCount
		for _, id := range verify.AcceptedTokens[emitStart:] {
			if nativeGemma4AssistantEmitToken(&result, id, eosID, yield) {
				stopped = true
				break
			}
			lastToken = id
			newDrafts++
		}
		result.AcceptedTokens += newDrafts
		result.TargetTokens += newDrafts
		if stopped {
			break
		}
		if len(result.Tokens) >= maxNew {
			break
		}
		if verify.AllAccepted {
			carryLead = -1
			continue
		}

		replacement := verify.ReplacementToken
		if nativeGemma4AssistantEmitToken(&result, replacement, eosID, yield) {
			stopped = true
		}
		result.TargetTokens++
		lastToken = replacement
		carryLead = replacement
	}
	if carryLead >= 0 && !stopped && yield == nil {
		if _, err := target.stepID(carryLead); err != nil {
			return result, err
		}
	}
	return result, nil
}

// GenerateSampledFromSession samples token ids from a native target session
// while using this assistant pair for speculative proposals. The target sampler
// decides every committed token; assistant proposals only affect acceptance.
func (pair *Gemma4AssistantPair) GenerateSampledFromSession(target *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	return pair.GenerateSampledFromSessionEach(target, promptIDs, maxNew, stopTokens, sampler, params, draftTokens, nil)
}

// GenerateSampledFromSessionEach is GenerateSampledFromSession with per-token
// streaming.
func (pair *Gemma4AssistantPair) GenerateSampledFromSessionEach(target *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int, yield Gemma4AssistantTokenSink) (Gemma4AssistantGenerateResult, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant sampled generation requires a validated pair")
	}
	if target == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant sampled generation target session is nil")
	}
	if sampler == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant sampled generation sampler is nil")
	}
	if len(promptIDs) == 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant sampled generation prompt tokens are required")
	}
	if maxNew <= 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant sampled generation maxNew must be > 0")
	}
	draftTokens = nativeGemma4AssistantResolveDraftTokens(draftTokens)
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if err := target.prepareGemma4AssistantPrompt(promptIDs); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}

	result := Gemma4AssistantGenerateResult{
		PromptTokens:       len(promptIDs),
		DraftTokenSchedule: make([]int, 0, (maxNew+draftTokens-1)/draftTokens),
	}
	lastToken := promptIDs[len(promptIDs)-1]
	carryLead := int32(-1)
	stopped := false
	for len(result.Tokens) < maxNew && !stopped {
		remaining := maxNew - len(result.Tokens)
		blockSize := draftTokens
		if blockSize > remaining {
			blockSize = remaining
		}
		if (yield != nil || len(stopTokens) > 0 || params.MinTokensBeforeStop > 0) && blockSize > 1 {
			blockSize = 1
		}
		pickParams := target.mtpSamplePickParams(params, stopTokens, len(result.Tokens))
		draft, err := pair.DraftBlockFromSession(target, lastToken, blockSize, pickParams.SuppressTokens)
		if err != nil {
			return result, err
		}
		result.DraftCalls++
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		block := draft.Tokens
		carryPresent := carryLead >= 0
		if carryPresent {
			block = append([]int32{carryLead}, draft.Tokens...)
		}
		verify, err := pair.VerifyDraftBlockSampledFromSession(target, block, sampler, pickParams, carryPresent)
		if err != nil {
			return result, err
		}
		result.TargetVerifyCalls++
		result.TargetCalls++
		emitStart := 0
		if carryPresent && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
			emitStart = 1
			carryLead = -1
		}
		newDrafts := 0
		result.RejectedTokens += verify.RejectedCount
		for _, id := range verify.AcceptedTokens[emitStart:] {
			if nativeGemma4AssistantEmitSampledToken(&result, id, stopTokens, yield) {
				stopped = true
				break
			}
			lastToken = id
			newDrafts++
		}
		result.AcceptedTokens += newDrafts
		result.TargetTokens += newDrafts
		if stopped {
			break
		}
		if len(result.Tokens) >= maxNew {
			break
		}
		if verify.AllAccepted {
			carryLead = -1
			continue
		}

		replacement := verify.ReplacementToken
		if nativeGemma4AssistantEmitSampledToken(&result, replacement, stopTokens, yield) {
			stopped = true
		}
		result.TargetTokens++
		lastToken = replacement
		carryLead = replacement
	}
	if carryLead >= 0 && !stopped && yield == nil {
		if _, err := target.stepID(carryLead); err != nil {
			return result, err
		}
	}
	return result, nil
}

func nativeGemma4AssistantResolveDraftTokens(draftTokens int) int {
	if draftTokens <= 0 {
		return nativeGemma4AssistantDefaultDraftTokens
	}
	return draftTokens
}

func (s *ArchSession) prepareGemma4AssistantPrompt(promptIDs []int32) error {
	if len(promptIDs) == 0 {
		return core.NewError("gemma4.assistant generation prompt tokens are required")
	}
	if len(promptIDs) > s.maxLen {
		return core.NewError("gemma4.assistant generation prompt would exceed maxLen cache rows")
	}
	if hidden := s.cachedPromptHiddenFor(promptIDs); hidden != nil {
		s.pos = len(promptIDs)
		if err := s.state.truncateDevicePagedKV(s.pos); err != nil {
			return err
		}
		resident := s.cachedIDs[:0]
		s.cachedIDs = append(resident, promptIDs...)
		s.rememberRetainedHidden(hidden)
		if logits := s.cachedPromptLogitsFor(promptIDs); logits != nil {
			s.rememberRetainedLogits(logits)
		}
		return nil
	}
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(promptIDs) {
		lcp = len(promptIDs) - 1
	}
	s.pos = lcp
	if err := s.state.truncateDevicePagedKV(s.pos); err != nil {
		return err
	}
	hidden, logits, err := s.prefillPromptCacheEntry(promptIDs[lcp:])
	if err != nil {
		s.cachedIDs = nil
		s.clearCachedPromptHidden()
		s.resetRetainedHidden()
		return err
	}
	resident := s.cachedIDs[:0]
	s.cachedIDs = append(resident, promptIDs...)
	s.rememberCachedPromptEntry(promptIDs, hidden, logits)
	s.rememberRetainedHidden(hidden)
	s.rememberRetainedLogits(logits)
	return nil
}

func nativeGemma4AssistantEmitToken(result *Gemma4AssistantGenerateResult, id int32, eosID int, yield Gemma4AssistantTokenSink) bool {
	if eosID >= 0 && int(id) == eosID {
		return true
	}
	result.Tokens = append(result.Tokens, id)
	if yield != nil && !yield(id) {
		return true
	}
	return false
}

func nativeGemma4AssistantEmitSampledToken(result *Gemma4AssistantGenerateResult, id int32, stopTokens []int32, yield Gemma4AssistantTokenSink) bool {
	result.Tokens = append(result.Tokens, id)
	return (yield != nil && !yield(id)) || nativeTokenInSet(id, stopTokens)
}

func (s *ArchSession) verifyGemma4AssistantDraftRows(draftTokens, suppress []int32) ([]int32, [][]byte, error) {
	hiddens, err := s.verifyGemma4AssistantDraftHiddens(draftTokens)
	if err != nil {
		return nil, nil, err
	}
	rows := make([]int32, len(draftTokens))
	if len(hiddens) != len(draftTokens) {
		return nil, nil, core.NewError("gemma4.assistant verify target rows are incomplete")
	}
	for i, hidden := range hiddens {
		token, err := s.greedyFromHiddenInPool(hidden, suppress)
		if err != nil {
			return nil, nil, err
		}
		rows[i] = token
	}
	return rows, hiddens, nil
}

func (s *ArchSession) verifyGemma4AssistantDraftHiddens(draftTokens []int32) ([][]byte, error) {
	hiddens, batched, err := s.verifyBatchedHiddens(draftTokens)
	if err != nil {
		return nil, err
	}
	if batched {
		if len(hiddens) != len(draftTokens) {
			return nil, core.NewError("gemma4.assistant verify batched target rows are incomplete")
		}
		return hiddens, nil
	}

	hiddens = make([][]byte, 0, len(draftTokens))
	for _, draft := range draftTokens {
		hidden, err := s.stepID(draft)
		if err != nil {
			return nil, err
		}
		hiddens = append(hiddens, append([]byte(nil), hidden...))
	}
	return hiddens, nil
}

func (s *ArchSession) rememberGemma4AssistantAcceptedIDs(posBefore int, accepted []int32) {
	if s == nil {
		return
	}
	if posBefore < 0 || len(s.cachedIDs) < posBefore {
		s.cachedIDs = nil
		return
	}
	s.cachedIDs = s.cachedIDs[:posBefore]
	s.cachedIDs = append(s.cachedIDs, accepted...)
}

func (pair *Gemma4AssistantPair) draftStepFromProjected(projected []byte, targetKVs Gemma4AssistantTargetKVByType, suppressTokens ...[]int32) (Gemma4AssistantDraftStepResult, error) {
	if pair == nil || pair.Assistant == nil {
		return Gemma4AssistantDraftStepResult{}, core.NewError("gemma4.assistant draft step requires a validated pair")
	}
	normed, hidden, err := pair.Assistant.DraftStepActivations(projected, targetKVs)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	logits, err := pair.Assistant.DraftLogits(normed)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	token, err := pair.Assistant.DraftGreedyToken(logits, suppressTokens...)
	if err != nil {
		return Gemma4AssistantDraftStepResult{}, err
	}
	return Gemma4AssistantDraftStepResult{Logits: logits, Token: token, Hidden: hidden}, nil
}

func (pair *Gemma4AssistantPair) validateDraftInputTarget() (model.Arch, error) {
	if pair == nil || pair.Assistant == nil {
		return model.Arch{}, core.NewError("gemma4.assistant draft input requires a validated pair")
	}
	target := pair.TargetArch
	if target.Hidden <= 0 || target.Vocab <= 0 {
		return model.Arch{}, core.NewError("gemma4.assistant draft input target arch is incomplete")
	}
	if pair.Assistant.BackboneHiddenSize != target.Hidden {
		return model.Arch{}, core.NewError(core.Sprintf("gemma4.assistant backbone_hidden_size = %d, want target hidden_size %d", pair.Assistant.BackboneHiddenSize, target.Hidden))
	}
	return target, nil
}

func nativeGemma4EmbeddingScale(arch model.Arch) float32 {
	if arch.Hidden <= 0 {
		return 0
	}
	return float32(math.Sqrt(float64(arch.Hidden)))
}

func (m *Gemma4AssistantModel) DraftOutputProjection(assistantHidden []byte) ([]byte, error) {
	return m.DraftOutputProjectionInto(nil, assistantHidden)
}

func (m *Gemma4AssistantModel) DraftFinalNorm(hiddenStates []byte) ([]byte, error) {
	return m.DraftFinalNormInto(nil, hiddenStates)
}

func (m *Gemma4AssistantModel) DraftAttention(layerIdx int, hiddenStates []byte, targetKV Gemma4AssistantTargetKV) ([]byte, error) {
	return m.DraftAttentionInto(nil, layerIdx, hiddenStates, targetKV)
}

func (m *Gemma4AssistantModel) DraftAttentionInto(out []byte, layerIdx int, hiddenStates []byte, targetKV Gemma4AssistantTargetKV) ([]byte, error) {
	layer, nHeads, headDim, err := m.validateDraftAttentionInput(layerIdx, hiddenStates, targetKV)
	if err != nil {
		return nil, err
	}
	kvHeads, err := nativeGemma4AssistantTargetKVHeads(targetKV, headDim)
	if err != nil {
		return nil, err
	}
	if nHeads%kvHeads != 0 {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft attention heads = %d, want multiple of target kv heads %d", nHeads, kvHeads))
	}

	prefix := core.Sprintf("model.layers.%d.self_attn.", layerIdx)
	qProj, err := nativeGemma4AssistantBF16Matrix(m, prefix+"q_proj.weight", nHeads*headDim, m.Arch.Hidden)
	if err != nil {
		return nil, err
	}
	qNorm, err := nativeGemma4AssistantBF16Vector(m, prefix+"q_norm.weight", headDim)
	if err != nil {
		return nil, err
	}
	oProj, err := nativeGemma4AssistantBF16Matrix(m, prefix+"o_proj.weight", m.Arch.Hidden, nHeads*headDim)
	if err != nil {
		return nil, err
	}

	q, err := MatVecBF16(qProj.Data, hiddenStates, nHeads*headDim, m.Arch.Hidden)
	if err != nil {
		return nil, core.E("gemma4.assistant draft attention", "q_proj", err)
	}
	q, err = RMSNormBF16(q, qNorm.Data, nHeads, headDim, m.Arch.Eps)
	if err != nil {
		return nil, core.E("gemma4.assistant draft attention", "q_norm", err)
	}
	q, err = nativeGemma4AssistantRoPE(q, m, layer, nHeads, headDim, targetKV.Offset)
	if err != nil {
		return nil, err
	}
	attn, err := SDPA(q, targetKV.Key, targetKV.Value, 1, nHeads, kvHeads, headDim, targetKV.Length, nativeGemma4AssistantAttentionScale(m))
	if err != nil {
		return nil, core.E("gemma4.assistant draft attention", "target kv sdpa", err)
	}
	return MatVecBF16Into(out, oProj.Data, attn, m.Arch.Hidden, nHeads*headDim)
}

func (m *Gemma4AssistantModel) DraftLayer(layerIdx int, hiddenStates []byte, targetKV Gemma4AssistantTargetKV) ([]byte, error) {
	return m.DraftLayerInto(nil, layerIdx, hiddenStates, targetKV)
}

func (m *Gemma4AssistantModel) DraftStepActivations(projectedHidden []byte, targetKVs Gemma4AssistantTargetKVByType) (normed []byte, targetHidden []byte, err error) {
	return m.DraftStepActivationsInto(nil, nil, projectedHidden, targetKVs)
}

func (m *Gemma4AssistantModel) DraftStepActivationsInto(normedOut, targetHiddenOut []byte, projectedHidden []byte, targetKVs Gemma4AssistantTargetKVByType) (normed []byte, targetHidden []byte, err error) {
	if m == nil {
		return nil, nil, core.NewError("gemma4.assistant draft step model is nil")
	}
	hidden := m.Arch.Hidden
	if hidden <= 0 || len(m.Arch.Layer) == 0 {
		return nil, nil, core.NewError("gemma4.assistant draft step has incomplete dimensions")
	}
	if len(projectedHidden) != hidden*bf16Size {
		return nil, nil, core.NewError(core.Sprintf("gemma4.assistant draft step projected hidden bytes = %d, want %d", len(projectedHidden), hidden*bf16Size))
	}
	h := projectedHidden
	for idx, layer := range m.Arch.Layer {
		layerType := nativeGemma4AssistantLayerType(m, idx, layer)
		targetKV, ok := targetKVs.Get(layerType)
		if !ok || !targetKV.HasState() {
			return nil, nil, core.NewError("gemma4.assistant draft step missing target K/V stream for " + layerType)
		}
		h, err = m.DraftLayer(idx, h, targetKV)
		if err != nil {
			return nil, nil, err
		}
	}
	normed, err = m.DraftFinalNormInto(normedOut, h)
	if err != nil {
		return nil, nil, err
	}
	targetHidden, err = m.DraftOutputProjectionInto(targetHiddenOut, normed)
	if err != nil {
		return nil, nil, err
	}
	return normed, targetHidden, nil
}

func (m *Gemma4AssistantModel) DraftLayerInto(out []byte, layerIdx int, hiddenStates []byte, targetKV Gemma4AssistantTargetKV) ([]byte, error) {
	hidden, dFF, err := m.validateDraftLayerInput(layerIdx, hiddenStates)
	if err != nil {
		return nil, err
	}
	prefix := core.Sprintf("model.layers.%d", layerIdx)
	inputNorm, err := nativeGemma4AssistantBF16Vector(m, prefix+".input_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	postAttnNorm, err := nativeGemma4AssistantBF16Vector(m, prefix+".post_attention_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	preFFNorm, err := nativeGemma4AssistantBF16Vector(m, prefix+".pre_feedforward_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	postFFNorm, err := nativeGemma4AssistantBF16Vector(m, prefix+".post_feedforward_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	gateProj, err := nativeGemma4AssistantBF16Matrix(m, prefix+".mlp.gate_proj.weight", dFF, hidden)
	if err != nil {
		return nil, err
	}
	upProj, err := nativeGemma4AssistantBF16Matrix(m, prefix+".mlp.up_proj.weight", dFF, hidden)
	if err != nil {
		return nil, err
	}
	downProj, err := nativeGemma4AssistantBF16Matrix(m, prefix+".mlp.down_proj.weight", hidden, dFF)
	if err != nil {
		return nil, err
	}
	layerScalar, err := nativeGemma4AssistantLayerScalar(m, prefix, hidden)
	if err != nil {
		return nil, err
	}

	normed, err := RMSNormBF16(hiddenStates, inputNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "input norm", err)
	}
	attnOut, err := m.DraftAttention(layerIdx, normed, targetKV)
	if err != nil {
		return nil, err
	}
	attnResidual, err := RMSNormBF16(attnOut, postAttnNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "post attention norm", err)
	}
	h, err := AddBF16(hiddenStates, attnResidual)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "attention residual", err)
	}

	ffIn, err := RMSNormBF16(h, preFFNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "pre feed-forward norm", err)
	}
	gate, err := MatVecBF16(gateProj.Data, ffIn, dFF, hidden)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "mlp gate projection", err)
	}
	up, err := MatVecBF16(upProj.Data, ffIn, dFF, hidden)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "mlp up projection", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "mlp gate activation", err)
	}
	ff, err := MatVecBF16(downProj.Data, gated, hidden, dFF)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "mlp down projection", err)
	}
	ffResidual, err := RMSNormBF16(ff, postFFNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "post feed-forward norm", err)
	}
	hNext, err := AddBF16(h, ffResidual)
	if err != nil {
		return nil, core.E("gemma4.assistant draft layer", "feed-forward residual", err)
	}
	if len(layerScalar) == bf16Size {
		return nativeGemma4AssistantMulScalarInto(out, hNext, layerScalar)
	}
	if len(layerScalar) == len(hNext) {
		return nativeGemma4AssistantMulVectorInto(out, hNext, layerScalar)
	}
	return nativeGemma4AssistantCopyInto(out, hNext), nil
}

func (m *Gemma4AssistantModel) validateDraftLayerInput(layerIdx int, hiddenStates []byte) (int, int, error) {
	if m == nil {
		return 0, 0, core.NewError("gemma4.assistant draft layer model is nil")
	}
	if layerIdx < 0 || layerIdx >= len(m.Arch.Layer) {
		return 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft layer index = %d, want [0,%d)", layerIdx, len(m.Arch.Layer)))
	}
	hidden := m.Arch.Hidden
	dFF := m.Arch.FF
	if hidden <= 0 || dFF <= 0 {
		return 0, 0, core.NewError("gemma4.assistant draft layer has incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft layer hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	return hidden, dFF, nil
}

func (m *Gemma4AssistantModel) validateDraftAttentionInput(layerIdx int, hiddenStates []byte, targetKV Gemma4AssistantTargetKV) (model.LayerSpec, int, int, error) {
	if m == nil {
		return model.LayerSpec{}, 0, 0, core.NewError("gemma4.assistant draft attention model is nil")
	}
	if layerIdx < 0 || layerIdx >= len(m.Arch.Layer) {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft attention layer index = %d, want [0,%d)", layerIdx, len(m.Arch.Layer)))
	}
	hidden := m.Arch.Hidden
	nHeads := m.Arch.Heads
	layer := m.Arch.Layer[layerIdx]
	headDim := layer.HeadDim
	if headDim <= 0 {
		headDim = m.Arch.HeadDim
	}
	if hidden <= 0 || nHeads <= 0 || headDim <= 0 {
		return model.LayerSpec{}, 0, 0, core.NewError("gemma4.assistant draft attention has incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft attention hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	if !targetKV.HasState() {
		return model.LayerSpec{}, 0, 0, core.NewError("gemma4.assistant draft attention target K/V stream is empty")
	}
	if targetKV.HeadDim > 0 && targetKV.HeadDim != headDim {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft attention target head_dim = %d, want %d", targetKV.HeadDim, headDim))
	}
	wantBytes := nativeGemma4AssistantTargetKVByteLen(targetKV, headDim)
	if wantBytes <= 0 {
		return model.LayerSpec{}, 0, 0, core.NewError("gemma4.assistant draft attention target K/V geometry is incomplete")
	}
	if len(targetKV.Key) != wantBytes {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft attention target key bytes = %d, want %d", len(targetKV.Key), wantBytes))
	}
	if len(targetKV.Value) != wantBytes {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("gemma4.assistant draft attention target value bytes = %d, want %d", len(targetKV.Value), wantBytes))
	}
	return layer, nHeads, headDim, nil
}

func nativeGemma4AssistantTargetKVHeads(kv Gemma4AssistantTargetKV, headDim int) (int, error) {
	if kv.KVHeads > 0 {
		return kv.KVHeads, nil
	}
	if kv.Length <= 0 || headDim <= 0 {
		return 0, core.NewError("gemma4.assistant draft attention target K/V geometry is incomplete")
	}
	denom := kv.Length * headDim * bf16Size
	if denom <= 0 || len(kv.Key)%denom != 0 {
		return 0, core.NewError("gemma4.assistant draft attention cannot infer target kv heads")
	}
	return len(kv.Key) / denom, nil
}

func nativeGemma4AssistantTargetKVByteLen(kv Gemma4AssistantTargetKV, headDim int) int {
	kvHeads := kv.KVHeads
	if kvHeads <= 0 && kv.Length > 0 && headDim > 0 {
		denom := kv.Length * headDim * bf16Size
		if denom > 0 && len(kv.Key)%denom == 0 {
			kvHeads = len(kv.Key) / denom
		}
	}
	if kvHeads <= 0 || kv.Length <= 0 || headDim <= 0 {
		return 0
	}
	return kvHeads * kv.Length * headDim * bf16Size
}

func nativeGemma4AssistantRoPE(q []byte, m *Gemma4AssistantModel, layer model.LayerSpec, nHeads, headDim, offset int) ([]byte, error) {
	rotaryDim := nativeGemma4AssistantLayerRotaryDim(m, layer, headDim)
	scale := m.Arch.RopeScale
	if scale == 0 {
		scale = 1
	}
	if len(m.Arch.RopeFreqs) > 0 {
		out, err := RoPEFreqsBF16(q, 1, nHeads, headDim, rotaryDim, m.Arch.RopeFreqs, scale, offset, false)
		if err != nil {
			return nil, core.E("gemma4.assistant draft attention", "q_rope", err)
		}
		return out, nil
	}
	base := nativeGemma4AssistantLayerRopeBase(m, layer)
	out, err := RoPEDimsBF16(q, 1, nHeads, headDim, rotaryDim, base, scale, offset, false)
	if err != nil {
		return nil, core.E("gemma4.assistant draft attention", "q_rope", err)
	}
	return out, nil
}

func nativeGemma4AssistantLayerRotaryDim(m *Gemma4AssistantModel, layer model.LayerSpec, headDim int) int {
	rotaryDim := m.Arch.RotaryDim
	if layer.Attention == model.SlidingAttention && m.Arch.RotaryDimLocal > 0 {
		rotaryDim = m.Arch.RotaryDimLocal
	}
	if rotaryDim <= 0 || rotaryDim > headDim {
		rotaryDim = headDim
	}
	return rotaryDim
}

func nativeGemma4AssistantLayerRopeBase(m *Gemma4AssistantModel, layer model.LayerSpec) float32 {
	if layer.Attention == model.SlidingAttention && m.Arch.RopeLocalBase > 0 {
		return m.Arch.RopeLocalBase
	}
	if m.Arch.RopeBase > 0 {
		return m.Arch.RopeBase
	}
	return 10000
}

func nativeGemma4AssistantAttentionScale(m *Gemma4AssistantModel) float32 {
	if m == nil || m.Arch.AttnScale == 0 {
		return 1
	}
	return m.Arch.AttnScale
}

func (m *Gemma4AssistantModel) DraftFinalNormInto(out []byte, hiddenStates []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("gemma4.assistant draft final norm model is nil")
	}
	hidden := m.Arch.Hidden
	if hidden <= 0 {
		return nil, core.NewError("gemma4.assistant draft final norm hidden_size is invalid")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft final norm hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	weight, ok := m.Tensors["model.norm.weight"]
	if !ok {
		return nil, core.NewError("gemma4.assistant draft final norm missing model.norm.weight")
	}
	if weight.Dtype != "BF16" {
		return nil, core.NewError("gemma4.assistant draft final norm model.norm.weight dtype = " + weight.Dtype + ", want BF16")
	}
	if len(weight.Shape) != 1 || weight.Shape[0] != hidden {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft final norm model.norm.weight shape = %v, want [%d]", weight.Shape, hidden))
	}
	if len(weight.Data) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft final norm model.norm.weight bytes = %d, want %d", len(weight.Data), hidden*bf16Size))
	}
	return RMSNormBF16Into(out, hiddenStates, weight.Data, 1, hidden, m.Arch.Eps)
}

func (m *Gemma4AssistantModel) DraftOutputProjectionInto(out []byte, assistantHidden []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("gemma4.assistant draft output model is nil")
	}
	hidden := m.Arch.Hidden
	backbone := m.BackboneHiddenSize
	if hidden <= 0 || backbone <= 0 {
		return nil, core.NewError("gemma4.assistant draft output has incomplete dimensions")
	}
	hiddenBytes := hidden * bf16Size
	if len(assistantHidden) != hiddenBytes {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft output assistant hidden bytes = %d, want %d", len(assistantHidden), hiddenBytes))
	}
	weight, ok := m.Tensors["post_projection.weight"]
	if !ok {
		return nil, core.NewError("gemma4.assistant draft output missing post_projection.weight")
	}
	if weight.Dtype != "BF16" {
		return nil, core.NewError("gemma4.assistant draft output post_projection.weight dtype = " + weight.Dtype + ", want BF16")
	}
	if len(weight.Shape) < 2 || weight.Shape[len(weight.Shape)-2] != backbone || weight.Shape[len(weight.Shape)-1] != hidden {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft output post_projection.weight shape = %v, want [%d %d]", weight.Shape, backbone, hidden))
	}
	if len(weight.Data) != backbone*hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft output post_projection.weight bytes = %d, want %d", len(weight.Data), backbone*hidden*bf16Size))
	}
	return MatMulBF16NTInto(out, assistantHidden, weight.Data, 1, hidden, backbone)
}

func (m *Gemma4AssistantModel) DraftLogits(hiddenStates []byte) ([]byte, error) {
	return m.DraftLogitsInto(nil, hiddenStates)
}

func (m *Gemma4AssistantModel) DraftLogitsInto(out []byte, hiddenStates []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("gemma4.assistant logits model is nil")
	}
	hidden := m.Arch.Hidden
	vocab := m.Arch.Vocab
	if hidden <= 0 || vocab <= 0 {
		return nil, core.NewError("gemma4.assistant logits have incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("gemma4.assistant logits hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	if m.UseOrderedEmbeddings {
		return m.draftOrderedLogitsInto(out, hiddenStates)
	}
	embed, err := nativeGemma4AssistantBF16Matrix(m, "model.embed_tokens.weight", vocab, hidden)
	if err != nil {
		return nil, err
	}
	outLen := vocab * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	for tokenID := 0; tokenID < vocab; tokenID++ {
		sum := nativeGemma4AssistantDotBF16Row(hiddenStates, embed.Data, tokenID, hidden)
		h := f32ToBF16(sum)
		off := tokenID * bf16Size
		out[off] = byte(h)
		out[off+1] = byte(h >> 8)
	}
	return out, nil
}

func (m *Gemma4AssistantModel) draftOrderedLogitsInto(out []byte, hiddenStates []byte) ([]byte, error) {
	hidden := m.Arch.Hidden
	vocab := m.Arch.Vocab
	numCentroids := m.NumCentroids
	topK := m.CentroidIntermediateTopK
	if numCentroids <= 0 || topK <= 0 || topK > numCentroids {
		return nil, core.NewError("gemma4.assistant ordered embeddings centroid_intermediate_top_k is invalid")
	}
	if vocab%numCentroids != 0 {
		return nil, core.NewError("gemma4.assistant token_ordering requires vocab_size divisible by num_centroids")
	}
	embed, err := nativeGemma4AssistantBF16Matrix(m, "model.embed_tokens.weight", vocab, hidden)
	if err != nil {
		return nil, err
	}
	centroids, err := nativeGemma4AssistantBF16Matrix(m, "masked_embedding.centroids.weight", numCentroids, hidden)
	if err != nil {
		return nil, err
	}
	ordering, ok := m.Tensors["masked_embedding.token_ordering"]
	if !ok {
		return nil, core.NewError("gemma4.assistant ordered embeddings require masked_embedding.token_ordering")
	}
	vocabPerCentroid := vocab / numCentroids
	if err := nativeGemma4AssistantValidateOrdering(ordering, vocab, numCentroids, vocabPerCentroid); err != nil {
		return nil, err
	}

	scores := make([]float32, numCentroids)
	for c := 0; c < numCentroids; c++ {
		scores[c] = nativeGemma4AssistantDotBF16Row(hiddenStates, centroids.Data, c, hidden)
	}
	selected := nativeGemma4AssistantTopK(scores, topK)

	outLen := vocab * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	floor := f32ToBF16(nativeGemma4AssistantLogitsFloor)
	for i := 0; i < vocab; i++ {
		out[i*bf16Size] = byte(floor)
		out[i*bf16Size+1] = byte(floor >> 8)
	}
	for _, centroid := range selected {
		for pos := 0; pos < vocabPerCentroid; pos++ {
			tokenID, err := nativeGemma4AssistantOrderingToken(ordering, centroid, pos, vocabPerCentroid)
			if err != nil {
				return nil, err
			}
			if tokenID < 0 || int(tokenID) >= vocab {
				return nil, core.NewError(core.Sprintf("gemma4.assistant token_ordering token id = %d, want [0,%d)", tokenID, vocab))
			}
			sum := nativeGemma4AssistantDotBF16Row(hiddenStates, embed.Data, int(tokenID), hidden)
			h := f32ToBF16(sum)
			off := int(tokenID) * bf16Size
			out[off] = byte(h)
			out[off+1] = byte(h >> 8)
		}
	}
	return out, nil
}

func nativeGemma4AssistantBF16Matrix(m *Gemma4AssistantModel, name string, rows, cols int) (safetensors.Tensor, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return safetensors.Tensor{}, core.NewError("gemma4.assistant missing " + name)
	}
	if t.Dtype != "BF16" {
		return safetensors.Tensor{}, core.NewError("gemma4.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
	}
	if len(t.Shape) < 2 || t.Shape[len(t.Shape)-2] != rows || t.Shape[len(t.Shape)-1] != cols {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("gemma4.assistant %s shape = %v, want [%d %d]", name, t.Shape, rows, cols))
	}
	if len(t.Data) != rows*cols*bf16Size {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("gemma4.assistant %s bytes = %d, want %d", name, len(t.Data), rows*cols*bf16Size))
	}
	return t, nil
}

func nativeGemma4AssistantBF16Vector(m *Gemma4AssistantModel, name string, elems int) (safetensors.Tensor, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return safetensors.Tensor{}, core.NewError("gemma4.assistant missing " + name)
	}
	if t.Dtype != "BF16" {
		return safetensors.Tensor{}, core.NewError("gemma4.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
	}
	if len(t.Shape) != 1 || t.Shape[0] != elems {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("gemma4.assistant %s shape = %v, want [%d]", name, t.Shape, elems))
	}
	if len(t.Data) != elems*bf16Size {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("gemma4.assistant %s bytes = %d, want %d", name, len(t.Data), elems*bf16Size))
	}
	return t, nil
}

func nativeGemma4AssistantLayerScalar(m *Gemma4AssistantModel, prefix string, hidden int) ([]byte, error) {
	for _, name := range []string{prefix + ".layer_scalar", prefix + ".layer_scalar.weight"} {
		t, ok := m.Tensors[name]
		if !ok || len(t.Data) == 0 {
			continue
		}
		if t.Dtype != "BF16" {
			return nil, core.NewError("gemma4.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
		}
		if len(t.Shape) == 1 && t.Shape[0] == 1 && len(t.Data) == bf16Size {
			return t.Data, nil
		}
		if len(t.Shape) == 1 && t.Shape[0] == hidden && len(t.Data) == hidden*bf16Size {
			return t.Data, nil
		}
		return nil, core.NewError(core.Sprintf("gemma4.assistant %s shape = %v, want [1] or [%d]", name, t.Shape, hidden))
	}
	return nil, nil
}

func nativeGemma4AssistantMulScalarInto(out []byte, in, scalar []byte) ([]byte, error) {
	if cap(out) >= len(in) {
		out = out[:len(in)]
		if err := MulScalarBF16Into(out, in, scalar); err != nil {
			return nil, err
		}
		return out, nil
	}
	return MulScalarBF16(in, scalar)
}

func nativeGemma4AssistantMulVectorInto(out []byte, in, vec []byte) ([]byte, error) {
	if cap(out) >= len(in) {
		out = out[:len(in)]
		if err := MulBF16Into(out, in, vec); err != nil {
			return nil, err
		}
		return out, nil
	}
	return MulBF16(in, vec)
}

func nativeGemma4AssistantCopyInto(out []byte, in []byte) []byte {
	if cap(out) < len(in) {
		return in
	}
	out = out[:len(in)]
	copy(out, in)
	return out
}

func nativeGemma4AssistantValidateOrdering(t safetensors.Tensor, vocab, numCentroids, vocabPerCentroid int) error {
	switch t.Dtype {
	case "I32":
		if len(t.Data) != vocab*4 {
			return core.NewError(core.Sprintf("gemma4.assistant token_ordering bytes = %d, want %d", len(t.Data), vocab*4))
		}
	case "I64":
		if len(t.Data) != vocab*8 {
			return core.NewError(core.Sprintf("gemma4.assistant token_ordering bytes = %d, want %d", len(t.Data), vocab*8))
		}
	default:
		return core.NewError("gemma4.assistant token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
	if len(t.Shape) == 1 && t.Shape[0] == vocab {
		return nil
	}
	if len(t.Shape) == 2 && t.Shape[0] == numCentroids && t.Shape[1] == vocabPerCentroid {
		return nil
	}
	return core.NewError(core.Sprintf("gemma4.assistant token_ordering shape = %v, want [%d] or [%d %d]", t.Shape, vocab, numCentroids, vocabPerCentroid))
}

func nativeGemma4AssistantOrderingToken(t safetensors.Tensor, centroid, pos, vocabPerCentroid int) (int32, error) {
	idx := centroid*vocabPerCentroid + pos
	switch t.Dtype {
	case "I32":
		off := idx * 4
		return int32(binary.LittleEndian.Uint32(t.Data[off:])), nil
	case "I64":
		off := idx * 8
		v := int64(binary.LittleEndian.Uint64(t.Data[off:]))
		if v < -2147483648 || v > 2147483647 {
			return 0, core.NewError(core.Sprintf("gemma4.assistant token_ordering token id = %d, want int32 range", v))
		}
		return int32(v), nil
	default:
		return 0, core.NewError("gemma4.assistant token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
}

func nativeGemma4AssistantDotBF16Row(vec, rows []byte, row, cols int) float32 {
	base := row * cols * bf16Size
	var sum float32
	for i := 0; i < cols; i++ {
		vo := i * bf16Size
		wo := base + i*bf16Size
		sum += bf16ToF32(vec[vo], vec[vo+1]) * bf16ToF32(rows[wo], rows[wo+1])
	}
	return sum
}

func nativeGemma4AssistantTopK(scores []float32, k int) []int {
	selected := make([]int, 0, k)
	for idx, score := range scores {
		pos := len(selected)
		for pos > 0 && score > scores[selected[pos-1]] {
			pos--
		}
		if pos >= k {
			continue
		}
		selected = append(selected, 0)
		copy(selected[pos+1:], selected[pos:len(selected)-1])
		selected[pos] = idx
		if len(selected) > k {
			selected = selected[:k]
		}
	}
	return selected
}

func (m *Gemma4AssistantModel) DraftGreedyToken(logits []byte, suppressTokens ...[]int32) (int32, error) {
	if m == nil {
		return 0, core.NewError("gemma4.assistant greedy token model is nil")
	}
	vocab := m.Arch.Vocab
	if vocab <= 0 {
		return 0, core.NewError("gemma4.assistant greedy token vocab_size is invalid")
	}
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError(core.Sprintf("gemma4.assistant greedy token logits bytes = %d, want %d", len(logits), vocab*bf16Size))
	}
	var suppressed []int32
	if len(suppressTokens) > 0 {
		suppressed = suppressTokens[0]
	}
	var bestID int32 = -1
	var best float32
	for id := 0; id < vocab; id++ {
		if nativeGemma4AssistantSuppressed(int32(id), suppressed) {
			continue
		}
		v := bf16ToF32(logits[id*bf16Size], logits[id*bf16Size+1])
		if bestID < 0 || v > best {
			bestID = int32(id)
			best = v
		}
	}
	if bestID < 0 {
		return 0, core.NewError("gemma4.assistant greedy token produced no token")
	}
	return bestID, nil
}

func nativeGemma4AssistantSuppressed(id int32, suppressTokens []int32) bool {
	for _, suppressed := range suppressTokens {
		if suppressed >= 0 && suppressed == id {
			return true
		}
	}
	return false
}

func (pair *Gemma4AssistantPair) Close() error {
	if pair == nil || pair.Assistant == nil {
		return nil
	}
	err := pair.Assistant.Close()
	pair.Assistant = nil
	return err
}
