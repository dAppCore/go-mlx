// SPDX-Licence-Identifier: EUPL-1.2

// Package bundle is the portable model-state artifact for go-mlx
// sessions: a kv.Snapshot plus the tokenizer, runtime, adapter, and
// sampler identity needed to safely replay it on a different host.
//
//	b, err := bundle.New(snapshot, bundle.Options{
//	    Model: "gemma4-e4b", ModelPath: "/models/gemma4",
//	    Source: bundle.ModelInfo{Architecture: "gemma4_text", NumLayers: 32},
//	})
package bundle

import (
	"context"
	"strconv"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/lora"
)

const (
	// Version is the portable bundle schema version.
	Version = 1
	// Kind identifies go-mlx state-bundle JSON payloads.
	Kind = "go-mlx/state-bundle"
	// RefState identifies a State cold-storage reference.
	RefState = "state"
	// RefMemvid identifies an old memvid cold-storage reference.
	//
	// Deprecated: use RefState.
	RefMemvid = "memvid"
)

// Options labels a bundle with caller-owned provenance.
type Options struct {
	Model       string
	ModelPath   string
	Source      ModelInfo
	Prompt      string
	Tokenizer   Tokenizer
	Runtime     Runtime
	Adapter     Adapter
	AdapterPath string
	KVPath      string
	Sampler     Sampler
	Analysis    *kv.Analysis
	SAMI        *SAMIResult
	Refs        []Ref
	StateRefs   []state.ChunkRef
	// Deprecated: use StateRefs.
	MemvidRefs []state.ChunkRef
	Meta       map[string]string
}

// ModelInfo describes the model expected by a bundle. Mirrors the
// mlx-root ModelInfo struct; converters at the boundary keep the two in
// sync.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	QuantBits     int
	QuantGroup    int
	ContextLength int
	Adapter       lora.AdapterInfo
}

// Bundle is a portable, strict model-state artifact.
type Bundle struct {
	Version   int               `json:"version"`
	Kind      string            `json:"kind"`
	Model     Model             `json:"model"`
	Prompt    Prompt            `json:"prompt"`
	Tokenizer Tokenizer         `json:"tokenizer"`
	Runtime   Runtime           `json:"runtime"`
	Adapter   Adapter           `json:"adapter,omitempty"`
	Sampler   Sampler           `json:"sampler"`
	KV        *kv.Snapshot      `json:"kv,omitempty"`
	KVPath    string            `json:"kv_path,omitempty"`
	KVHash    string            `json:"kv_hash"`
	Analysis  *kv.Analysis      `json:"analysis,omitempty"`
	SAMI      *SAMIResult       `json:"sami,omitempty"`
	Refs      []Ref             `json:"refs,omitempty"`
	Meta      map[string]string `json:"meta,omitempty"`
}

// Model identifies the model captured by the bundle.
type Model struct {
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

// Prompt identifies the prompt/token state captured by the bundle.
type Prompt struct {
	Text        string `json:"text,omitempty"`
	Hash        string `json:"hash,omitempty"`
	TokenCount  int    `json:"token_count"`
	TokenOffset int    `json:"token_offset"`
}

// Tokenizer identifies tokenizer and chat-template compatibility.
type Tokenizer struct {
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

// Runtime identifies the go-mlx runtime that created the bundle.
type Runtime struct {
	Name     string `json:"name,omitempty"`
	Version  string `json:"version,omitempty"`
	Build    string `json:"build,omitempty"`
	Platform string `json:"platform,omitempty"`
}

// Adapter identifies an optional LoRA adapter applied to the model.
type Adapter struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// Sampler stores generation settings needed for reproducible replay.
type Sampler struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k"`
	TopP          float32 `json:"top_p"`
	MinP          float32 `json:"min_p"`
	StopTokens    []int32 `json:"stop_tokens,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty"`
}

// Ref links external cold-storage artifacts such as State chunks.
type Ref struct {
	Kind   string         `json:"kind"`
	URI    string         `json:"uri"`
	Hash   string         `json:"hash,omitempty"`
	Title  string         `json:"title,omitempty"`
	Track  string         `json:"track,omitempty"`
	State  state.ChunkRef `json:"state,omitempty"`
	Memvid state.ChunkRef `json:"memvid,omitempty"`
}

// New builds a portable bundle around a restorable kv.Snapshot.
//
//	b, err := bundle.New(snapshot, bundle.Options{Model: "gemma4-e4b"})
func New(snapshot *kv.Snapshot, opts Options) (*Bundle, error) {
	if snapshot == nil {
		return nil, core.NewError("bundle: KV snapshot is nil")
	}
	snap := snapshot.Clone()
	if snap.Version == 0 {
		snap.Version = kv.SnapshotVersion
	}
	if snap.TokenOffset == 0 {
		snap.TokenOffset = len(snap.Tokens)
	}
	kvHash, err := kv.HashSnapshot(snap)
	if err != nil {
		return nil, err
	}
	analysis := opts.Analysis
	if analysis == nil {
		analysis = kv.Analyze(snap)
	}
	sami := opts.SAMI
	if sami == nil {
		result := SAMIFromKV(snap, analysis, SAMIOptions{Model: opts.Model, Prompt: opts.Prompt})
		sami = &result
	}
	model := buildModel(snap, opts)
	tokenizer := NormaliseTokenizer(opts.Tokenizer)
	runtime := normaliseRuntime(opts.Runtime)
	adapter := buildAdapter(opts.Adapter, opts.AdapterPath, opts.Source.Adapter)
	b := &Bundle{
		Version: Version,
		Kind:    Kind,
		Model:   model,
		Prompt: Prompt{
			Text:        opts.Prompt,
			Hash:        HashString(opts.Prompt),
			TokenCount:  len(snap.Tokens),
			TokenOffset: snap.TokenOffset,
		},
		Tokenizer: tokenizer,
		Runtime:   runtime,
		Adapter:   adapter,
		Sampler:   opts.Sampler,
		KV:        snap,
		KVPath:    opts.KVPath,
		KVHash:    kvHash,
		Analysis:  analysis,
		SAMI:      sami,
		Refs:      buildRefs(opts.Refs, append(append([]state.ChunkRef(nil), opts.StateRefs...), opts.MemvidRefs...)),
		Meta:      cloneMeta(opts.Meta),
	}
	if AdapterEmpty(b.Adapter) {
		b.Adapter = Adapter{}
	}
	return b, nil
}

// Save writes the bundle as stable indented JSON.
//
//	if err := b.Save(path); err != nil { … }
func (b *Bundle) Save(path string) error {
	if err := b.Validate(); err != nil {
		return err
	}
	data := core.JSONMarshalIndent(b, "", "  ")
	if !data.OK {
		return core.E("bundle.Save", "marshal bundle", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.E("bundle.Save", "write bundle", resultError(result))
	}
	return nil
}

// Load reads a bundle saved by (*Bundle).Save.
//
//	b, err := bundle.Load(path)
func Load(path string) (*Bundle, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("bundle.Load", "read bundle", resultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("bundle.Load", "read bundle returned non-byte data", nil)
	}
	var b Bundle
	if result := core.JSONUnmarshal(data, &b); !result.OK {
		return nil, core.E("bundle.Load", "parse bundle", resultError(result))
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return &b, nil
}

// Snapshot returns a defensive kv.Snapshot copy, loading KVPath when needed.
//
//	snap, err := b.Snapshot()
func (b *Bundle) Snapshot() (*kv.Snapshot, error) {
	if b == nil {
		return nil, core.NewError("bundle: state bundle is nil")
	}
	if b.KV != nil {
		return b.KV.Clone(), nil
	}
	if b.KVPath == "" {
		return nil, core.NewError("bundle: state bundle has no KV snapshot")
	}
	snapshot, err := kv.Load(b.KVPath)
	if err != nil {
		return nil, err
	}
	if b.KVHash != "" {
		got, hashErr := kv.HashSnapshot(snapshot)
		if hashErr != nil {
			return nil, hashErr
		}
		if got != b.KVHash {
			return nil, core.NewError("bundle: state bundle KV hash mismatch")
		}
	}
	return snapshot, nil
}

// SnapshotFromState resolves a State-backed KV snapshot.
//
//	snap, err := b.SnapshotFromState(ctx, store)
func (b *Bundle) SnapshotFromState(ctx context.Context, store state.Store) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if b == nil {
		return nil, core.NewError("bundle: state bundle is nil")
	}
	if b.KV != nil || b.KVPath != "" {
		return b.Snapshot()
	}
	ref, ok := b.stateRef()
	if !ok {
		return nil, core.NewError("bundle: state bundle has no State KV snapshot")
	}
	snapshot, err := kv.LoadFromState(ctx, store, ref)
	if err != nil {
		return nil, err
	}
	if b.KVHash != "" {
		got, hashErr := kv.HashSnapshot(snapshot)
		if hashErr != nil {
			return nil, hashErr
		}
		if got != b.KVHash {
			return nil, core.NewError("bundle: state bundle KV hash mismatch")
		}
	}
	return snapshot, nil
}

// SnapshotFromMemvid resolves an old memvid-backed KV snapshot.
//
// Deprecated: use SnapshotFromState.
func (b *Bundle) SnapshotFromMemvid(ctx context.Context, store state.Store) (*kv.Snapshot, error) {
	return b.SnapshotFromState(ctx, store)
}

func (b *Bundle) stateRef() (state.ChunkRef, bool) {
	if b == nil {
		return state.ChunkRef{}, false
	}
	for _, ref := range b.Refs {
		if ref.Kind == RefState && ref.State.ChunkID != 0 {
			return ref.State, true
		}
		if ref.Kind == RefState && ref.Memvid.ChunkID != 0 {
			return ref.Memvid, true
		}
		if ref.Kind == RefMemvid {
			return ref.Memvid, true
		}
	}
	return state.ChunkRef{}, false
}

// Validate checks schema version, kind, and embedded KV hash integrity.
//
//	if err := b.Validate(); err != nil { … }
func (b *Bundle) Validate() error {
	if b == nil {
		return core.NewError("bundle: state bundle is nil")
	}
	if b.Version <= 0 || b.Version > Version {
		return core.NewError("bundle: unsupported state bundle version")
	}
	if b.Kind != Kind {
		return core.NewError("bundle: invalid state bundle kind")
	}
	if b.KV == nil && b.KVPath == "" {
		if _, ok := b.stateRef(); !ok {
			return core.NewError("bundle: state bundle has no KV snapshot")
		}
		return nil
	}
	if b.KV != nil && b.KVHash != "" {
		got, err := kv.HashSnapshot(b.KV)
		if err != nil {
			return err
		}
		if got != b.KVHash {
			return core.NewError("bundle: state bundle KV hash mismatch")
		}
	}
	return nil
}

// CheckCompatibility verifies that a loaded model can safely restore a bundle.
//
//	if err := bundle.CheckCompatibility(modelInfo, b); err != nil { … }
func CheckCompatibility(info ModelInfo, b *Bundle) error {
	if b == nil {
		return core.NewError("bundle: state bundle is nil")
	}
	if err := b.Validate(); err != nil {
		return err
	}
	if b.Model.Architecture != "" && info.Architecture != "" && b.Model.Architecture != info.Architecture {
		return core.NewError("bundle: state bundle model architecture mismatch")
	}
	if b.Model.NumLayers > 0 && info.NumLayers > 0 && b.Model.NumLayers != info.NumLayers {
		return core.NewError("bundle: state bundle model layer mismatch")
	}
	return checkAdapterCompatibility(info.Adapter, b.Adapter)
}

// FileHash hashes an external file for strict bundle metadata.
//
//	hash, err := bundle.FileHash(path)
func FileHash(path string) (string, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return "", core.E("bundle.FileHash", "read file", resultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return "", core.E("bundle.FileHash", "read file returned non-byte data", nil)
	}
	return core.SHA256Hex(data), nil
}

// NormaliseTokenizer fills missing Tokenizer hash fields based on
// Path / ChatTemplate values.
//
//	t := bundle.NormaliseTokenizer(t)
func NormaliseTokenizer(tokenizer Tokenizer) Tokenizer {
	if tokenizer.Hash == "" && tokenizer.Path != "" {
		tokenizer.Hash = HashString(tokenizer.Path)
	}
	if tokenizer.ChatTemplateHash == "" && tokenizer.ChatTemplate != "" {
		tokenizer.ChatTemplateHash = HashString(tokenizer.ChatTemplate)
	}
	return tokenizer
}

// AdapterEmpty reports whether the adapter has no meaningful fields set.
//
//	if bundle.AdapterEmpty(a) { … }
func AdapterEmpty(adapter Adapter) bool {
	return adapter.Name == "" && adapter.Path == "" && adapter.Hash == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0
}

// AdapterFromInfo lifts a lora.AdapterInfo into an Adapter.
//
//	a := bundle.AdapterFromInfo(info)
func AdapterFromInfo(info lora.AdapterInfo) Adapter {
	return Adapter{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: append([]string(nil), info.TargetKeys...),
	}
}

// AdapterToInfo lowers an Adapter to a lora.AdapterInfo.
//
//	info := bundle.AdapterToInfo(a)
func AdapterToInfo(adapter Adapter) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       adapter.Name,
		Path:       adapter.Path,
		Hash:       adapter.Hash,
		Rank:       adapter.Rank,
		Alpha:      adapter.Alpha,
		Scale:      adapter.Scale,
		TargetKeys: append([]string(nil), adapter.TargetKeys...),
	}
}

// HashString returns the SHA-256 hex of a string, or empty for empty input.
//
//	h := bundle.HashString("hello")
func HashString(value string) string {
	if value == "" {
		return ""
	}
	return core.SHA256HexString(value)
}

// StateURI renders a State chunk reference as a state:// URI.
//
//	uri := bundle.StateURI(ref)
func StateURI(ref state.ChunkRef) string {
	if ref.Segment != "" {
		return core.Sprintf("state://%s#chunk=%d", ref.Segment, ref.ChunkID)
	}
	return core.Sprintf("state://chunk/%d", ref.ChunkID)
}

func buildModel(snapshot *kv.Snapshot, opts Options) Model {
	src := opts.Source
	arch := src.Architecture
	if arch == "" && snapshot != nil {
		arch = snapshot.Architecture
	}
	numLayers := src.NumLayers
	if numLayers == 0 && snapshot != nil {
		numLayers = snapshot.NumLayers
	}
	model := Model{
		Name:          opts.Model,
		Path:          opts.ModelPath,
		Architecture:  arch,
		VocabSize:     src.VocabSize,
		NumLayers:     numLayers,
		HiddenSize:    src.HiddenSize,
		QuantBits:     src.QuantBits,
		QuantGroup:    src.QuantGroup,
		ContextLength: src.ContextLength,
	}
	// Hand-built hash payload — avoids 4× Sprintf("%d") boxing and a
	// 7-arg Join intermediate slice. Capacity sized for typical
	// model.Path + Architecture strings plus 4 small integer fields.
	buf := make([]byte, 0, len(model.Name)+len(model.Path)+len(model.Architecture)+48)
	buf = append(buf, model.Name...)
	buf = append(buf, '\n')
	buf = append(buf, model.Path...)
	buf = append(buf, '\n')
	buf = append(buf, model.Architecture...)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.VocabSize), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.NumLayers), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.QuantBits), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.ContextLength), 10)
	model.Hash = HashString(core.AsString(buf))
	return model
}

func normaliseRuntime(runtime Runtime) Runtime {
	if runtime.Name == "" {
		runtime.Name = "go-mlx"
	}
	return runtime
}

func buildAdapter(adapter Adapter, adapterPath string, info lora.AdapterInfo) Adapter {
	if AdapterEmpty(adapter) && !info.IsEmpty() {
		adapter = AdapterFromInfo(info)
	}
	if adapter.Path == "" {
		adapter.Path = adapterPath
	}
	if adapter.Hash == "" {
		// Hand-built hash payload — avoids Sprintf("%d") + 2× Sprintf("%f")
		// boxing and a 6-arg Join intermediate. Float formatting matches
		// fmt's default %f precision (6 decimals).
		keyCommas := 0
		if n := len(adapter.TargetKeys); n > 1 {
			keyCommas = n - 1
		}
		keyBytes := 0
		for _, key := range adapter.TargetKeys {
			keyBytes += len(key)
		}
		buf := make([]byte, 0, len(adapter.Name)+len(adapter.Path)+keyBytes+keyCommas+48)
		buf = append(buf, adapter.Name...)
		buf = append(buf, '\n')
		buf = append(buf, adapter.Path...)
		buf = append(buf, '\n')
		buf = strconv.AppendInt(buf, int64(adapter.Rank), 10)
		buf = append(buf, '\n')
		buf = strconv.AppendFloat(buf, float64(adapter.Alpha), 'f', 6, 32)
		buf = append(buf, '\n')
		buf = strconv.AppendFloat(buf, float64(adapter.Scale), 'f', 6, 32)
		buf = append(buf, '\n')
		for i, key := range adapter.TargetKeys {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, key...)
		}
		adapter.Hash = HashString(core.AsString(buf))
	}
	if adapter.Path == "" && adapter.Name == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0 {
		adapter.Hash = ""
	}
	adapter.TargetKeys = append([]string(nil), adapter.TargetKeys...)
	return adapter
}

func checkAdapterCompatibility(active lora.AdapterInfo, expected Adapter) error {
	if AdapterEmpty(expected) {
		return nil
	}
	if active.IsEmpty() {
		return core.NewError("bundle: state bundle requires a LoRA adapter but model has none")
	}
	want := AdapterToInfo(expected)
	if want.Hash != "" && active.Hash != "" && want.Hash != active.Hash {
		return core.NewError("bundle: state bundle LoRA adapter hash mismatch")
	}
	if want.Path != "" && active.Path != "" && want.Path != active.Path && (want.Hash == "" || active.Hash == "") {
		return core.NewError("bundle: state bundle LoRA adapter path mismatch")
	}
	if want.Rank > 0 && active.Rank > 0 && want.Rank != active.Rank {
		return core.NewError("bundle: state bundle LoRA adapter rank mismatch")
	}
	if want.Alpha != 0 && active.Alpha != 0 && want.Alpha != active.Alpha {
		return core.NewError("bundle: state bundle LoRA adapter alpha mismatch")
	}
	return nil
}

// MemvidURI renders an old memvid chunk reference as a memvid:// URI.
//
// Deprecated: use StateURI.
func MemvidURI(ref state.ChunkRef) string {
	if ref.Segment != "" {
		return core.Sprintf("memvid://%s#chunk=%d", ref.Segment, ref.ChunkID)
	}
	return core.Sprintf("memvid://chunk/%d", ref.ChunkID)
}

func buildRefs(refs []Ref, stateRefs []state.ChunkRef) []Ref {
	if len(refs) == 0 && len(stateRefs) == 0 {
		return nil
	}
	out := make([]Ref, 0, len(refs)+len(stateRefs))
	out = append(out, refs...)
	for _, ref := range stateRefs {
		uri := StateURI(ref)
		out = append(out, Ref{
			Kind:  RefState,
			URI:   uri,
			Hash:  HashString(uri),
			State: ref,
		})
	}
	return out
}

func cloneMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(meta))
	for key, value := range meta {
		cloned[key] = value
	}
	return cloned
}

func resultError(result core.Result) error {
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
