// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

const (
	// StateBundleVersion is the portable model-state bundle schema version.
	StateBundleVersion = 1
	// StateBundleKind identifies go-mlx state-bundle JSON payloads.
	StateBundleKind = "go-mlx/state-bundle"
	// StateBundleRefMemvid identifies a memvid cold-storage reference.
	StateBundleRefMemvid = "memvid"
)

// StateBundleOptions labels a state bundle with caller-owned provenance.
type StateBundleOptions struct {
	Model     string
	ModelPath string
	ModelInfo ModelInfo
	Prompt    string
	Tokenizer StateBundleTokenizer
	Runtime   StateBundleRuntime
	Adapter   StateBundleAdapter
	// AdapterPath is retained for callers that do not need the richer adapter identity.
	AdapterPath string
	KVPath      string
	Sampler     GenerateConfig
	Analysis    *KVAnalysis
	SAMI        *SAMIResult
	Refs        []StateBundleRef
	MemvidRefs  []memvid.ChunkRef
	Meta        map[string]string
}

// StateBundle is a portable, strict model-state artifact.
type StateBundle struct {
	Version   int                  `json:"version"`
	Kind      string               `json:"kind"`
	Model     StateBundleModel     `json:"model"`
	Prompt    StateBundlePrompt    `json:"prompt"`
	Tokenizer StateBundleTokenizer `json:"tokenizer"`
	Runtime   StateBundleRuntime   `json:"runtime"`
	Adapter   StateBundleAdapter   `json:"adapter,omitempty"`
	Sampler   StateBundleSampler   `json:"sampler"`
	KV        *KVSnapshot          `json:"kv,omitempty"`
	KVPath    string               `json:"kv_path,omitempty"`
	KVHash    string               `json:"kv_hash"`
	Analysis  *KVAnalysis          `json:"analysis,omitempty"`
	SAMI      *SAMIResult          `json:"sami,omitempty"`
	Refs      []StateBundleRef     `json:"refs,omitempty"`
	Meta      map[string]string    `json:"meta,omitempty"`
}

// StateBundleModel identifies the model expected by the bundle.
type StateBundleModel struct {
	Name          string `json:"name,omitempty"`
	Path          string `json:"path,omitempty"`
	Architecture  string `json:"architecture"`
	VocabSize     int    `json:"vocab_size,omitempty"`
	NumLayers     int    `json:"num_layers,omitempty"`
	HiddenSize    int    `json:"hidden_size,omitempty"`
	QuantBits     int    `json:"quant_bits,omitempty"`
	QuantGroup    int    `json:"quant_group,omitempty"`
	ContextLength int    `json:"context_length,omitempty"`
	Hash          string `json:"hash,omitempty"`
}

// StateBundlePrompt identifies the prompt/token state captured by the bundle.
type StateBundlePrompt struct {
	Text        string `json:"text,omitempty"`
	Hash        string `json:"hash,omitempty"`
	TokenCount  int    `json:"token_count"`
	TokenOffset int    `json:"token_offset"`
}

// StateBundleTokenizer identifies tokenizer and chat-template compatibility.
type StateBundleTokenizer struct {
	Kind             string `json:"kind,omitempty"`
	Path             string `json:"path,omitempty"`
	Version          string `json:"version,omitempty"`
	Hash             string `json:"hash,omitempty"`
	VocabSize        int    `json:"vocab_size,omitempty"`
	BOS              int32  `json:"bos,omitempty"`
	EOS              int32  `json:"eos,omitempty"`
	ChatTemplate     string `json:"chat_template,omitempty"`
	ChatTemplateHash string `json:"chat_template_hash,omitempty"`
}

// StateBundleRuntime identifies the go-mlx runtime that created the bundle.
type StateBundleRuntime struct {
	Name     string `json:"name,omitempty"`
	Version  string `json:"version,omitempty"`
	Build    string `json:"build,omitempty"`
	Platform string `json:"platform,omitempty"`
}

// StateBundleAdapter identifies an optional LoRA adapter applied to the model.
type StateBundleAdapter struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// StateBundleSampler stores generation settings needed for reproducible replay.
type StateBundleSampler struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k"`
	TopP          float32 `json:"top_p"`
	MinP          float32 `json:"min_p"`
	StopTokens    []int32 `json:"stop_tokens,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty"`
}

// StateBundleRef links external cold-storage artifacts such as memvid chunks.
type StateBundleRef struct {
	Kind   string          `json:"kind"`
	URI    string          `json:"uri"`
	Hash   string          `json:"hash,omitempty"`
	Title  string          `json:"title,omitempty"`
	Track  string          `json:"track,omitempty"`
	Memvid memvid.ChunkRef `json:"memvid,omitempty"`
}

// NewStateBundle builds a portable state bundle around a restorable KV snapshot.
func NewStateBundle(snapshot *KVSnapshot, opts StateBundleOptions) (*StateBundle, error) {
	if snapshot == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	kv := snapshot.Clone()
	normalizeBundleSnapshot(kv)
	kvHash, err := hashKVSnapshot(kv)
	if err != nil {
		return nil, err
	}
	analysis := opts.Analysis
	if analysis == nil {
		analysis = AnalyzeKV(kv)
	}
	sami := opts.SAMI
	if sami == nil {
		result := SAMIFromKV(kv, analysis, SAMIOptions{Model: opts.Model, Prompt: opts.Prompt})
		sami = &result
	}
	model := stateBundleModel(kv, opts)
	tokenizer := stateBundleTokenizer(opts.Tokenizer)
	runtime := stateBundleRuntime(opts.Runtime)
	adapter := stateBundleAdapter(opts.Adapter, opts.AdapterPath, opts.ModelInfo.Adapter)
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		Model:   model,
		Prompt: StateBundlePrompt{
			Text:        opts.Prompt,
			Hash:        stateHash(opts.Prompt),
			TokenCount:  len(kv.Tokens),
			TokenOffset: kv.TokenOffset,
		},
		Tokenizer: tokenizer,
		Runtime:   runtime,
		Adapter:   adapter,
		Sampler:   stateSamplerFromGenerateConfig(opts.Sampler),
		KV:        kv,
		KVPath:    opts.KVPath,
		KVHash:    kvHash,
		Analysis:  analysis,
		SAMI:      sami,
		Refs:      stateBundleRefs(opts.Refs, opts.MemvidRefs),
		Meta:      cloneStateBundleMeta(opts.Meta),
	}
	if stateBundleAdapterEmpty(bundle.Adapter) {
		bundle.Adapter = StateBundleAdapter{}
	}
	return bundle, nil
}

// ExportBundle captures a live session and returns a portable state bundle.
func (s *ModelSession) ExportBundle(opts StateBundleOptions) (*StateBundle, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return NewStateBundle(snapshot, opts)
}

// Save writes the state bundle as stable JSON.
func (b *StateBundle) Save(path string) error {
	if err := b.Validate(); err != nil {
		return err
	}
	data := core.JSONMarshalIndent(b, "", "  ")
	if !data.OK {
		return core.E("StateBundle.Save", "marshal bundle", stateBundleResultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.E("StateBundle.Save", "write bundle", stateBundleResultError(result))
	}
	return nil
}

// LoadStateBundle reads a bundle saved by (*StateBundle).Save.
func LoadStateBundle(path string) (*StateBundle, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("LoadStateBundle", "read bundle", stateBundleResultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("LoadStateBundle", "read bundle returned non-byte data", nil)
	}
	var bundle StateBundle
	if result := core.JSONUnmarshal(data, &bundle); !result.OK {
		return nil, core.E("LoadStateBundle", "parse bundle", stateBundleResultError(result))
	}
	if err := bundle.Validate(); err != nil {
		return nil, err
	}
	return &bundle, nil
}

// Snapshot returns a defensive KV snapshot copy, loading KVPath when needed.
func (b *StateBundle) Snapshot() (*KVSnapshot, error) {
	if b == nil {
		return nil, core.NewError("mlx: state bundle is nil")
	}
	if b.KV != nil {
		return b.KV.Clone(), nil
	}
	if b.KVPath == "" {
		return nil, core.NewError("mlx: state bundle has no KV snapshot")
	}
	snapshot, err := LoadKVSnapshot(b.KVPath)
	if err != nil {
		return nil, err
	}
	if b.KVHash != "" {
		got, hashErr := hashKVSnapshot(snapshot)
		if hashErr != nil {
			return nil, hashErr
		}
		if got != b.KVHash {
			return nil, core.NewError("mlx: state bundle KV hash mismatch")
		}
	}
	return snapshot, nil
}

// Validate checks schema version, kind, and embedded KV hash integrity.
func (b *StateBundle) Validate() error {
	if b == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if b.Version <= 0 || b.Version > StateBundleVersion {
		return core.NewError("mlx: unsupported state bundle version")
	}
	if b.Kind != StateBundleKind {
		return core.NewError("mlx: invalid state bundle kind")
	}
	if b.KV == nil && b.KVPath == "" {
		return core.NewError("mlx: state bundle has no KV snapshot")
	}
	if b.KV != nil && b.KVHash != "" {
		got, err := hashKVSnapshot(b.KV)
		if err != nil {
			return err
		}
		if got != b.KVHash {
			return core.NewError("mlx: state bundle KV hash mismatch")
		}
	}
	return nil
}

// CheckStateBundleCompatibility verifies that a loaded model can safely restore a bundle.
func CheckStateBundleCompatibility(info ModelInfo, bundle *StateBundle) error {
	if bundle == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if err := bundle.Validate(); err != nil {
		return err
	}
	if bundle.Model.Architecture != "" && info.Architecture != "" && bundle.Model.Architecture != info.Architecture {
		return core.NewError("mlx: state bundle model architecture mismatch")
	}
	if bundle.Model.NumLayers > 0 && info.NumLayers > 0 && bundle.Model.NumLayers != info.NumLayers {
		return core.NewError("mlx: state bundle model layer mismatch")
	}
	return checkStateBundleAdapterCompatibility(info.Adapter, bundle.Adapter)
}

func stateSamplerFromGenerateConfig(cfg GenerateConfig) StateBundleSampler {
	return StateBundleSampler{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

// StateBundleFileHash hashes an external file for strict bundle metadata.
func StateBundleFileHash(path string) (string, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return "", core.E("StateBundleFileHash", "read file", stateBundleResultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return "", core.E("StateBundleFileHash", "read file returned non-byte data", nil)
	}
	return core.SHA256Hex(data), nil
}

func stateBundleModel(snapshot *KVSnapshot, opts StateBundleOptions) StateBundleModel {
	info := opts.ModelInfo
	arch := info.Architecture
	if arch == "" && snapshot != nil {
		arch = snapshot.Architecture
	}
	numLayers := info.NumLayers
	if numLayers == 0 && snapshot != nil {
		numLayers = snapshot.NumLayers
	}
	model := StateBundleModel{
		Name:          opts.Model,
		Path:          opts.ModelPath,
		Architecture:  arch,
		VocabSize:     info.VocabSize,
		NumLayers:     numLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
	model.Hash = stateHash(core.Join("\n", model.Name, model.Path, model.Architecture, core.Sprintf("%d", model.VocabSize), core.Sprintf("%d", model.NumLayers), core.Sprintf("%d", model.QuantBits), core.Sprintf("%d", model.ContextLength)))
	return model
}

func stateBundleTokenizer(tokenizer StateBundleTokenizer) StateBundleTokenizer {
	if tokenizer.Hash == "" && tokenizer.Path != "" {
		tokenizer.Hash = stateHash(tokenizer.Path)
	}
	if tokenizer.ChatTemplateHash == "" && tokenizer.ChatTemplate != "" {
		tokenizer.ChatTemplateHash = stateHash(tokenizer.ChatTemplate)
	}
	return tokenizer
}

func stateBundleRuntime(runtime StateBundleRuntime) StateBundleRuntime {
	if runtime.Name == "" {
		runtime.Name = "go-mlx"
	}
	return runtime
}

func stateBundleAdapter(adapter StateBundleAdapter, adapterPath string, info LoRAAdapterInfo) StateBundleAdapter {
	if stateBundleAdapterEmpty(adapter) && !loraAdapterInfoEmpty(info) {
		adapter = stateBundleAdapterFromInfo(info)
	}
	if adapter.Path == "" {
		adapter.Path = adapterPath
	}
	if adapter.Hash == "" {
		adapter.Hash = stateHash(core.Join("\n", adapter.Name, adapter.Path, core.Sprintf("%d", adapter.Rank), core.Sprintf("%f", adapter.Alpha), core.Sprintf("%f", adapter.Scale), core.Join(",", adapter.TargetKeys...)))
	}
	if adapter.Path == "" && adapter.Name == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0 {
		adapter.Hash = ""
	}
	adapter.TargetKeys = append([]string(nil), adapter.TargetKeys...)
	return adapter
}

func stateBundleAdapterEmpty(adapter StateBundleAdapter) bool {
	return adapter.Name == "" && adapter.Path == "" && adapter.Hash == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0
}

func stateBundleAdapterFromInfo(info LoRAAdapterInfo) StateBundleAdapter {
	return StateBundleAdapter{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: append([]string(nil), info.TargetKeys...),
	}
}

func stateBundleAdapterToInfo(adapter StateBundleAdapter) LoRAAdapterInfo {
	return LoRAAdapterInfo{
		Name:       adapter.Name,
		Path:       adapter.Path,
		Hash:       adapter.Hash,
		Rank:       adapter.Rank,
		Alpha:      adapter.Alpha,
		Scale:      adapter.Scale,
		TargetKeys: append([]string(nil), adapter.TargetKeys...),
	}
}

func checkStateBundleAdapterCompatibility(active LoRAAdapterInfo, expected StateBundleAdapter) error {
	if stateBundleAdapterEmpty(expected) {
		return nil
	}
	if loraAdapterInfoEmpty(active) {
		return core.NewError("mlx: state bundle requires a LoRA adapter but model has none")
	}
	want := stateBundleAdapterToInfo(expected)
	if want.Hash != "" && active.Hash != "" && want.Hash != active.Hash {
		return core.NewError("mlx: state bundle LoRA adapter hash mismatch")
	}
	if want.Path != "" && active.Path != "" && want.Path != active.Path && (want.Hash == "" || active.Hash == "") {
		return core.NewError("mlx: state bundle LoRA adapter path mismatch")
	}
	if want.Rank > 0 && active.Rank > 0 && want.Rank != active.Rank {
		return core.NewError("mlx: state bundle LoRA adapter rank mismatch")
	}
	if want.Alpha != 0 && active.Alpha != 0 && want.Alpha != active.Alpha {
		return core.NewError("mlx: state bundle LoRA adapter alpha mismatch")
	}
	return nil
}

func stateBundleRefs(refs []StateBundleRef, memvidRefs []memvid.ChunkRef) []StateBundleRef {
	if len(refs) == 0 && len(memvidRefs) == 0 {
		return nil
	}
	out := make([]StateBundleRef, 0, len(refs)+len(memvidRefs))
	for _, ref := range refs {
		out = append(out, ref)
	}
	for _, ref := range memvidRefs {
		out = append(out, StateBundleRef{
			Kind:   StateBundleRefMemvid,
			URI:    stateMemvidURI(ref),
			Hash:   stateHash(stateMemvidURI(ref)),
			Memvid: ref,
		})
	}
	return out
}

func stateMemvidURI(ref memvid.ChunkRef) string {
	if ref.Segment != "" {
		return core.Sprintf("memvid://%s#chunk=%d", ref.Segment, ref.ChunkID)
	}
	return core.Sprintf("memvid://chunk/%d", ref.ChunkID)
}

func cloneStateBundleMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(meta))
	for key, value := range meta {
		cloned[key] = value
	}
	return cloned
}

func normalizeBundleSnapshot(snapshot *KVSnapshot) {
	if snapshot == nil {
		return
	}
	if snapshot.Version == 0 {
		snapshot.Version = KVSnapshotVersion
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
}

func hashKVSnapshot(snapshot *KVSnapshot) (string, error) {
	if snapshot == nil {
		return "", core.NewError("mlx: KV snapshot is nil")
	}
	cloned := snapshot.Clone()
	normalizeBundleSnapshot(cloned)
	data, err := cloned.bytes()
	if err != nil {
		return "", err
	}
	return core.SHA256Hex(data), nil
}

func stateHash(value string) string {
	if value == "" {
		return ""
	}
	return core.SHA256HexString(value)
}

func stateBundleResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	if text, ok := result.Value.(string); ok {
		return core.NewError(text)
	}
	return core.NewError("core result failed")
}
