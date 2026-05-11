// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
)

const (
	// KVSnapshotMemvidBundleIndexKind identifies a memvid-stored lookup index
	// for named spans inside one or more KV block bundles.
	KVSnapshotMemvidBundleIndexKind = "go-mlx/kv-snapshot-bundle-index"
	// KVSnapshotMemvidBundleIndexVersion is the bundle-index schema version.
	KVSnapshotMemvidBundleIndexVersion = 1
)

// KVSnapshotMemvidBundleIndexOptions configures a durable index for named KV
// bundle spans such as chapters, sections, or checkpointed agent states.
type KVSnapshotMemvidBundleIndexOptions struct {
	BundleURI string
	Title     string
	Model     string
	ModelPath string
	ModelInfo ModelInfo
	Tokenizer StateBundleTokenizer
	Entries   []KVSnapshotMemvidBundleIndexEntry
}

// KVSnapshotMemvidBundleIndex records model identity and named token spans for
// restoring partial prefixes from a larger memvid KV block bundle.
type KVSnapshotMemvidBundleIndex struct {
	Version      int                                `json:"version"`
	Kind         string                             `json:"kind"`
	BundleURI    string                             `json:"bundle_uri,omitempty"`
	SnapshotHash string                             `json:"snapshot_hash,omitempty"`
	KVEncoding   KVSnapshotEncoding                 `json:"kv_encoding,omitempty"`
	TokenCount   int                                `json:"token_count,omitempty"`
	BlockSize    int                                `json:"block_size,omitempty"`
	Model        StateBundleModel                   `json:"model"`
	Tokenizer    StateBundleTokenizer               `json:"tokenizer"`
	Entries      []KVSnapshotMemvidBundleIndexEntry `json:"entries,omitempty"`
	Hash         string                             `json:"hash,omitempty"`
}

// KVSnapshotMemvidBundleIndexEntry names one logical span in a KV bundle. The
// current wake path restores the prefix ending at TokenStart+TokenCount.
type KVSnapshotMemvidBundleIndexEntry struct {
	URI        string            `json:"uri"`
	BundleURI  string            `json:"bundle_uri,omitempty"`
	Title      string            `json:"title,omitempty"`
	TokenStart int               `json:"token_start"`
	TokenCount int               `json:"token_count"`
	ByteStart  int64             `json:"byte_start,omitempty"`
	ByteCount  int64             `json:"byte_count,omitempty"`
	Hash       string            `json:"hash,omitempty"`
	Labels     []string          `json:"labels,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

// NewKVSnapshotMemvidBundleIndex builds an index around a memvid KV block
// bundle. When no entries are supplied, it creates one full-bundle entry.
func NewKVSnapshotMemvidBundleIndex(bundle *KVSnapshotMemvidBlockBundle, opts KVSnapshotMemvidBundleIndexOptions) (*KVSnapshotMemvidBundleIndex, error) {
	if err := validateKVSnapshotMemvidBlockBundle(bundle); err != nil {
		return nil, err
	}
	index := &KVSnapshotMemvidBundleIndex{
		Version:      KVSnapshotMemvidBundleIndexVersion,
		Kind:         KVSnapshotMemvidBundleIndexKind,
		BundleURI:    core.Trim(opts.BundleURI),
		SnapshotHash: bundle.SnapshotHash,
		KVEncoding:   bundle.KVEncoding,
		TokenCount:   bundle.TokenCount,
		BlockSize:    bundle.BlockSize,
		Model:        kvSnapshotMemvidIndexModel(bundle, opts),
		Tokenizer:    stateBundleTokenizer(opts.Tokenizer),
		Entries:      cloneKVSnapshotMemvidBundleIndexEntries(opts.Entries),
	}
	if len(index.Entries) == 0 {
		index.Entries = []KVSnapshotMemvidBundleIndexEntry{{
			URI:        firstNonEmpty(index.BundleURI, "mlx://kv/full"),
			BundleURI:  index.BundleURI,
			Title:      firstNonEmpty(opts.Title, "full bundle"),
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
		}}
	}
	for i := range index.Entries {
		if index.Entries[i].BundleURI == "" {
			index.Entries[i].BundleURI = index.BundleURI
		}
		fillKVSnapshotMemvidBundleIndexEntryByteSpan(&index.Entries[i], bundle)
		if index.Entries[i].Hash == "" {
			index.Entries[i].Hash = kvSnapshotMemvidBundleIndexEntryHash(index.Entries[i])
		}
	}
	index.Hash = kvSnapshotMemvidBundleIndexHash(index)
	if err := index.Validate(); err != nil {
		return nil, err
	}
	return index, nil
}

// Validate checks schema, model identity, and indexed span bounds.
func (index *KVSnapshotMemvidBundleIndex) Validate() error {
	if index == nil {
		return core.NewError("mlx: memvid KV bundle index is nil")
	}
	if index.Version <= 0 || index.Version > KVSnapshotMemvidBundleIndexVersion {
		return core.NewError("mlx: unsupported memvid KV bundle index version")
	}
	if index.Kind != KVSnapshotMemvidBundleIndexKind {
		return core.NewError("mlx: invalid memvid KV bundle index kind")
	}
	if index.TokenCount <= 0 {
		return core.NewError("mlx: memvid KV bundle index token count is empty")
	}
	if len(index.Entries) == 0 {
		return core.NewError("mlx: memvid KV bundle index has no entries")
	}
	seen := map[string]bool{}
	for _, entry := range index.Entries {
		if err := index.validateEntry(entry); err != nil {
			return err
		}
		if seen[entry.URI] {
			return core.NewError("mlx: duplicate memvid KV bundle index URI")
		}
		seen[entry.URI] = true
	}
	if index.Hash != "" && index.Hash != kvSnapshotMemvidBundleIndexHash(index) {
		return core.NewError("mlx: memvid KV bundle index hash mismatch")
	}
	return nil
}

func (index *KVSnapshotMemvidBundleIndex) validateEntry(entry KVSnapshotMemvidBundleIndexEntry) error {
	if core.Trim(entry.URI) == "" {
		return core.NewError("mlx: memvid KV bundle index entry URI is required")
	}
	if core.Trim(entry.BundleURI) == "" && core.Trim(index.BundleURI) == "" {
		return core.NewError("mlx: memvid KV bundle index entry bundle URI is required")
	}
	if entry.TokenStart < 0 {
		return core.NewError("mlx: memvid KV bundle index entry token start is invalid")
	}
	if entry.TokenCount <= 0 {
		return core.NewError("mlx: memvid KV bundle index entry token count is empty")
	}
	if entry.TokenStart+entry.TokenCount > index.TokenCount {
		return core.NewError("mlx: memvid KV bundle index entry exceeds bundle token count")
	}
	if entry.ByteStart < 0 || entry.ByteCount < 0 {
		return core.NewError("mlx: memvid KV bundle index entry byte span is invalid")
	}
	if entry.Hash != "" && entry.Hash != kvSnapshotMemvidBundleIndexEntryHash(entry) {
		return core.NewError("mlx: memvid KV bundle index entry hash mismatch")
	}
	return nil
}

// Entry returns a defensive copy of the entry with URI.
func (index *KVSnapshotMemvidBundleIndex) Entry(uri string) (KVSnapshotMemvidBundleIndexEntry, bool) {
	if index == nil {
		return KVSnapshotMemvidBundleIndexEntry{}, false
	}
	for _, entry := range index.Entries {
		if entry.URI == uri {
			return cloneKVSnapshotMemvidBundleIndexEntry(entry), true
		}
	}
	return KVSnapshotMemvidBundleIndexEntry{}, false
}

// RequiredContextLength reports the largest prefix length needed by any entry.
func (index *KVSnapshotMemvidBundleIndex) RequiredContextLength() int {
	if index == nil {
		return 0
	}
	required := 0
	for _, entry := range index.Entries {
		if end := entry.PrefixTokens(); end > required {
			required = end
		}
	}
	return required
}

// PrefixTokens reports the prefix length needed to restore this entry.
func (entry KVSnapshotMemvidBundleIndexEntry) PrefixTokens() int {
	return entry.TokenStart + entry.TokenCount
}

// SaveKVSnapshotMemvidBundleIndex stores the index JSON in the same memvid
// store as its referenced bundle manifests.
func SaveKVSnapshotMemvidBundleIndex(ctx context.Context, store memvid.Writer, index *KVSnapshotMemvidBundleIndex, uri string) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid store is nil")
	}
	if core.Trim(uri) == "" {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid KV bundle index URI is required")
	}
	if err := index.Validate(); err != nil {
		return memvid.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(index), memvid.PutOptions{
		URI:    uri,
		Title:  "go-mlx KV bundle index",
		Kind:   KVSnapshotMemvidBundleIndexKind,
		Track:  "session-kv-index",
		Labels: []string{"go-mlx", "kv-snapshot-bundle-index"},
	})
	if err != nil {
		return memvid.ChunkRef{}, core.E("KVSnapshot.SaveMemvidBundleIndex", "write memvid bundle index", err)
	}
	return ref, nil
}

// LoadKVSnapshotMemvidBundleIndex restores an index by URI from a memvid store.
func LoadKVSnapshotMemvidBundleIndex(ctx context.Context, store memvid.Store, uri string) (*KVSnapshotMemvidBundleIndex, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	if core.Trim(uri) == "" {
		return nil, core.NewError("mlx: memvid KV bundle index URI is required")
	}
	chunk, err := memvid.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadKVSnapshotMemvidBundleIndex", "resolve memvid bundle index", err)
	}
	var index KVSnapshotMemvidBundleIndex
	if result := core.JSONUnmarshalString(chunk.Text, &index); !result.OK {
		return nil, core.E("LoadKVSnapshotMemvidBundleIndex", "parse bundle index", kvSnapshotResultError(result))
	}
	if err := index.Validate(); err != nil {
		return nil, err
	}
	return &index, nil
}

// LoadKVSnapshotPrefixFromMemvidBundleIndex resolves entryURI through index,
// loads its referenced block bundle, and restores only the prefix required by
// that entry.
func LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx context.Context, store memvid.Store, index *KVSnapshotMemvidBundleIndex, entryURI string, opts KVSnapshotLoadOptions) (*KVSnapshot, KVSnapshotMemvidBundleIndexEntry, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, core.NewError("mlx: memvid store is nil")
	}
	if err := index.Validate(); err != nil {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, err
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, core.NewError("mlx: memvid KV bundle index entry not found")
	}
	bundleURI := entry.BundleURI
	if bundleURI == "" {
		bundleURI = index.BundleURI
	}
	bundle, err := LoadKVSnapshotMemvidBlockBundle(ctx, store, bundleURI)
	if err != nil {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, err
	}
	prefixTokens := entry.PrefixTokens()
	if prefixTokens <= 0 || prefixTokens > bundle.TokenCount {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, core.NewError("mlx: memvid KV bundle index prefix is invalid")
	}
	snapshot, err := LoadKVSnapshotPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, opts)
	if err != nil {
		return nil, KVSnapshotMemvidBundleIndexEntry{}, err
	}
	return snapshot, entry, nil
}

// CheckKVSnapshotMemvidBundleIndexCompatibility verifies model and tokenizer
// identity before restoring indexed KV state into a loaded model.
func CheckKVSnapshotMemvidBundleIndexCompatibility(info ModelInfo, tokenizer StateBundleTokenizer, index *KVSnapshotMemvidBundleIndex) error {
	if err := index.Validate(); err != nil {
		return err
	}
	if index.Model.Architecture != "" && info.Architecture != "" && index.Model.Architecture != info.Architecture {
		return core.NewError("mlx: memvid KV bundle index model architecture mismatch")
	}
	if index.Model.NumLayers > 0 && info.NumLayers > 0 && index.Model.NumLayers != info.NumLayers {
		return core.NewError("mlx: memvid KV bundle index model layer mismatch")
	}
	if index.Model.QuantBits > 0 && info.QuantBits > 0 && index.Model.QuantBits != info.QuantBits {
		return core.NewError("mlx: memvid KV bundle index model quantization mismatch")
	}
	if index.Model.Hash != "" && index.Model.Name == "" && index.Model.Path == "" && kvSnapshotMemvidModelHashComparable(info, index.Model) {
		active := kvSnapshotMemvidIndexModel(nil, KVSnapshotMemvidBundleIndexOptions{ModelInfo: info})
		if active.Hash != "" && active.Hash != index.Model.Hash {
			return core.NewError("mlx: memvid KV bundle index model hash mismatch")
		}
	}
	if info.ContextLength > 0 && index.RequiredContextLength() > info.ContextLength {
		return core.NewError("mlx: memvid KV bundle index exceeds model context length")
	}
	if index.Tokenizer.Hash != "" && tokenizer.Hash != "" && index.Tokenizer.Hash != tokenizer.Hash {
		return core.NewError("mlx: memvid KV bundle index tokenizer hash mismatch")
	}
	if index.Tokenizer.ChatTemplateHash != "" && tokenizer.ChatTemplateHash != "" && index.Tokenizer.ChatTemplateHash != tokenizer.ChatTemplateHash {
		return core.NewError("mlx: memvid KV bundle index chat template hash mismatch")
	}
	return nil
}

func kvSnapshotMemvidModelHashComparable(info ModelInfo, model StateBundleModel) bool {
	if model.Architecture != "" && info.Architecture == "" {
		return false
	}
	if model.VocabSize > 0 && info.VocabSize == 0 {
		return false
	}
	if model.NumLayers > 0 && info.NumLayers == 0 {
		return false
	}
	if model.QuantBits > 0 && info.QuantBits == 0 {
		return false
	}
	if model.ContextLength > 0 && info.ContextLength == 0 {
		return false
	}
	return true
}

func kvSnapshotMemvidIndexModel(bundle *KVSnapshotMemvidBlockBundle, opts KVSnapshotMemvidBundleIndexOptions) StateBundleModel {
	info := opts.ModelInfo
	if info.Architecture == "" && bundle != nil {
		info.Architecture = bundle.Architecture
	}
	model := StateBundleModel{
		Name:          opts.Model,
		Path:          opts.ModelPath,
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
	model.Hash = stateHash(core.Join("\n", model.Name, model.Path, model.Architecture, core.Sprintf("%d", model.VocabSize), core.Sprintf("%d", model.NumLayers), core.Sprintf("%d", model.QuantBits), core.Sprintf("%d", model.ContextLength)))
	return model
}

func fillKVSnapshotMemvidBundleIndexEntryByteSpan(entry *KVSnapshotMemvidBundleIndexEntry, bundle *KVSnapshotMemvidBlockBundle) {
	if entry == nil || bundle == nil || len(bundle.Blocks) == 0 {
		return
	}
	if entry.ByteStart != 0 || entry.ByteCount != 0 {
		return
	}
	spanStart := entry.TokenStart
	spanEnd := entry.TokenStart + entry.TokenCount
	if spanEnd <= spanStart {
		return
	}
	var (
		byteStartSet bool
		byteStart    int64
		byteCount    int64
	)
	for _, ref := range bundle.Blocks {
		refStart := ref.TokenStart
		refEnd := ref.TokenStart + ref.TokenCount
		if refEnd <= spanStart || refStart >= spanEnd {
			continue
		}
		if !byteStartSet && ref.Memvid.HasFrameOffset && ref.Memvid.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(ref.Memvid.FrameOffset)
			byteStartSet = true
		}
		if ref.PayloadByteCount > 0 {
			byteCount += int64(ref.PayloadByteCount)
		}
	}
	if entry.ByteStart == 0 && byteStartSet {
		entry.ByteStart = byteStart
	}
	if entry.ByteCount == 0 && byteCount > 0 {
		entry.ByteCount = byteCount
	}
}

func kvSnapshotMemvidBundleIndexHash(index *KVSnapshotMemvidBundleIndex) string {
	if index == nil {
		return ""
	}
	builder := core.NewBuilder()
	builder.WriteString(index.Kind)
	builder.WriteString("|")
	builder.WriteString(index.BundleURI)
	builder.WriteString("|")
	builder.WriteString(index.SnapshotHash)
	builder.WriteString("|")
	builder.WriteString(string(index.KVEncoding))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(index.TokenCount))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(index.BlockSize))
	builder.WriteString("|")
	builder.WriteString(index.Model.Hash)
	builder.WriteString("|")
	builder.WriteString(index.Tokenizer.Hash)
	builder.WriteString("|")
	builder.WriteString(index.Tokenizer.ChatTemplateHash)
	for _, entry := range index.Entries {
		builder.WriteString("|")
		builder.WriteString(kvSnapshotMemvidBundleIndexEntryHash(entry))
	}
	return core.SHA256HexString(builder.String())
}

func kvSnapshotMemvidBundleIndexEntryHash(entry KVSnapshotMemvidBundleIndexEntry) string {
	builder := core.NewBuilder()
	builder.WriteString(entry.URI)
	builder.WriteString("|")
	builder.WriteString(entry.BundleURI)
	builder.WriteString("|")
	builder.WriteString(entry.Title)
	builder.WriteString("|")
	builder.WriteString(core.Itoa(entry.TokenStart))
	builder.WriteString("|")
	builder.WriteString(core.Itoa(entry.TokenCount))
	builder.WriteString("|")
	builder.WriteString(core.FormatInt(entry.ByteStart, 10))
	builder.WriteString("|")
	builder.WriteString(core.FormatInt(entry.ByteCount, 10))
	for _, label := range entry.Labels {
		builder.WriteString("|")
		builder.WriteString(label)
	}
	if len(entry.Meta) > 0 {
		keys := make([]string, 0, len(entry.Meta))
		for key := range entry.Meta {
			keys = append(keys, key)
		}
		core.SliceSort(keys)
		for _, key := range keys {
			builder.WriteString("|")
			builder.WriteString(key)
			builder.WriteString("=")
			builder.WriteString(entry.Meta[key])
		}
	}
	return core.SHA256HexString(builder.String())
}

func cloneKVSnapshotMemvidBundleIndexEntries(entries []KVSnapshotMemvidBundleIndexEntry) []KVSnapshotMemvidBundleIndexEntry {
	if len(entries) == 0 {
		return nil
	}
	out := make([]KVSnapshotMemvidBundleIndexEntry, len(entries))
	for i, entry := range entries {
		out[i] = cloneKVSnapshotMemvidBundleIndexEntry(entry)
	}
	return out
}

func cloneKVSnapshotMemvidBundleIndexEntry(entry KVSnapshotMemvidBundleIndexEntry) KVSnapshotMemvidBundleIndexEntry {
	entry.Labels = append([]string(nil), entry.Labels...)
	if len(entry.Meta) > 0 {
		meta := make(map[string]string, len(entry.Meta))
		for key, value := range entry.Meta {
			meta[key] = value
		}
		entry.Meta = meta
	}
	return entry
}
