// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"crypto/sha256"
	"hash"
	"strconv"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
)

const (
	// StateIndexKind identifies a State-stored lookup index
	// for named spans inside one or more KV block bundles.
	StateIndexKind = "go-mlx/kv-snapshot-bundle-index"
	// KVSnapshotStateBundleIndexVersion is the bundle-index schema version.
	KVSnapshotStateBundleIndexVersion = 1
	// MemvidIndexKind identifies an old memvid-named lookup index for named
	// spans inside one or more KV block bundles.
	//
	// Deprecated: use StateIndexKind.
	MemvidIndexKind = StateIndexKind
	// KVSnapshotMemvidBundleIndexVersion is the bundle-index schema version.
	//
	// Deprecated: use KVSnapshotStateBundleIndexVersion.
	KVSnapshotMemvidBundleIndexVersion = KVSnapshotStateBundleIndexVersion
)

// stateIndexPutLabels is the canonical label set attached to every
// SaveStateIndex Put call. Package-scoped so each call shares one backing
// array instead of allocating a fresh slice literal per save.
var stateIndexPutLabels = []string{"go-mlx", "kv-snapshot-bundle-index"}

// StateIndexOptions configures a durable index for named State
// spans such as chapters, sections, or checkpointed agent states.
type StateIndexOptions struct {
	BundleURI string
	Title     string
	Model     string
	ModelPath string
	ModelInfo memory.ModelInfo
	Tokenizer bundle.Tokenizer
	Entries   []StateIndexEntry
}

// MemvidIndexOptions configures a durable index for old memvid-named KV
// bundle spans such as chapters, sections, or checkpointed agent states.
//
// Deprecated: use StateIndexOptions.
type MemvidIndexOptions = StateIndexOptions

// StateIndex records model identity and named token spans for restoring
// partial prefixes from a larger durable State block bundle.
type StateIndex struct {
	Version      int               `json:"version"`
	Kind         string            `json:"kind"`
	BundleURI    string            `json:"bundle_uri,omitempty"`
	SnapshotHash string            `json:"snapshot_hash,omitempty"`
	KVEncoding   kv.Encoding       `json:"kv_encoding,omitempty"`
	TokenCount   int               `json:"token_count,omitempty"`
	BlockSize    int               `json:"block_size,omitempty"`
	Model        bundle.Model      `json:"model"`
	Tokenizer    bundle.Tokenizer  `json:"tokenizer"`
	Entries      []StateIndexEntry `json:"entries,omitempty"`
	Hash         string            `json:"hash,omitempty"`
}

// MemvidIndex records model identity and named token spans for restoring
// partial prefixes from a larger old memvid-named KV block bundle.
//
// Deprecated: use StateIndex.
type MemvidIndex = StateIndex

// StateIndexEntry names one logical span in a State bundle. The current wake
// path restores the prefix ending at TokenStart+TokenCount.
type StateIndexEntry struct {
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

// MemvidIndexEntry names one logical span in an old memvid-named KV bundle.
//
// Deprecated: use StateIndexEntry.
type MemvidIndexEntry = StateIndexEntry

// NewStateIndex builds an index around a durable State block bundle. When no
// entries are supplied, it creates one full-bundle entry.
func NewStateIndex(bundle *kv.StateBlockBundle, opts StateIndexOptions) (*StateIndex, error) {
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	index := &StateIndex{
		Version:      KVSnapshotStateBundleIndexVersion,
		Kind:         StateIndexKind,
		BundleURI:    core.Trim(opts.BundleURI),
		SnapshotHash: bundle.SnapshotHash,
		KVEncoding:   bundle.KVEncoding,
		TokenCount:   bundle.TokenCount,
		BlockSize:    bundle.BlockSize,
		Model:        indexModel(bundle, opts),
		Tokenizer:    stateBundleTokenizer(opts.Tokenizer),
		Entries:      cloneIndexEntries(opts.Entries),
	}
	if len(index.Entries) == 0 {
		index.Entries = []StateIndexEntry{{
			URI:        firstNonEmpty(index.BundleURI, "mlx://kv/full"),
			BundleURI:  index.BundleURI,
			Title:      firstNonEmpty(opts.Title, "full bundle"),
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
		}}
	}
	sortedBlocks := stateBlockRefsSortedByTokenStart(bundle.Blocks)
	for i := range index.Entries {
		if index.Entries[i].BundleURI == "" {
			index.Entries[i].BundleURI = index.BundleURI
		}
		if sortedBlocks {
			fillIndexEntryByteSpanSorted(&index.Entries[i], bundle)
		} else {
			fillIndexEntryByteSpan(&index.Entries[i], bundle)
		}
		if index.Entries[i].Hash == "" {
			index.Entries[i].Hash = indexEntryHash(index.Entries[i])
		} else if index.Entries[i].Hash != indexEntryHash(index.Entries[i]) {
			return nil, core.NewError("mlx: State index entry hash mismatch")
		}
	}
	index.Hash = indexHash(index)
	if err := index.validate(false); err != nil {
		return nil, err
	}
	return index, nil
}

// NewMemvidIndex builds an index around an old memvid-named KV block bundle. When no
// entries are supplied, it creates one full-bundle entry.
//
// Deprecated: use NewStateIndex.
func NewMemvidIndex(bundle *kv.MemvidBlockBundle, opts MemvidIndexOptions) (*MemvidIndex, error) {
	return NewStateIndex(bundle, opts)
}

// Validate checks schema, model identity, and indexed span bounds.
func (index *StateIndex) Validate() error {
	return index.validate(true)
}

func (index *StateIndex) validate(checkHashes bool) error {
	if index == nil {
		return core.NewError("mlx: State index is nil")
	}
	if index.Version <= 0 || index.Version > KVSnapshotStateBundleIndexVersion {
		return core.NewError("mlx: unsupported State index version")
	}
	if index.Kind != StateIndexKind {
		return core.NewError("mlx: invalid State index kind")
	}
	if index.TokenCount <= 0 {
		return core.NewError("mlx: State index token count is empty")
	}
	if len(index.Entries) == 0 {
		return core.NewError("mlx: State index has no entries")
	}
	seen := make(map[string]bool, len(index.Entries))
	indexBundleURIEmpty := core.Trim(index.BundleURI) == ""
	for _, entry := range index.Entries {
		if err := index.validateEntry(entry, checkHashes, indexBundleURIEmpty); err != nil {
			return err
		}
		if seen[entry.URI] {
			return core.NewError("mlx: duplicate State index URI")
		}
		seen[entry.URI] = true
	}
	if checkHashes && index.Hash != "" && index.Hash != indexHash(index) {
		return core.NewError("mlx: State index hash mismatch")
	}
	return nil
}

func (index *StateIndex) validateEntry(entry StateIndexEntry, checkHash, indexBundleURIEmpty bool) error {
	if core.Trim(entry.URI) == "" {
		return core.NewError("mlx: State index entry URI is required")
	}
	if indexBundleURIEmpty && core.Trim(entry.BundleURI) == "" {
		return core.NewError("mlx: State index entry bundle URI is required")
	}
	if entry.TokenStart < 0 {
		return core.NewError("mlx: State index entry token start is invalid")
	}
	if entry.TokenCount <= 0 {
		return core.NewError("mlx: State index entry token count is empty")
	}
	if entry.TokenStart+entry.TokenCount > index.TokenCount {
		return core.NewError("mlx: State index entry exceeds bundle token count")
	}
	if entry.ByteStart < 0 || entry.ByteCount < 0 {
		return core.NewError("mlx: State index entry byte span is invalid")
	}
	if checkHash && entry.Hash != "" && entry.Hash != indexEntryHash(entry) {
		return core.NewError("mlx: State index entry hash mismatch")
	}
	return nil
}

// Entry returns a defensive copy of the entry with URI.
func (index *StateIndex) Entry(uri string) (StateIndexEntry, bool) {
	if index == nil {
		return StateIndexEntry{}, false
	}
	for _, entry := range index.Entries {
		if entry.URI == uri {
			return cloneIndexEntry(entry), true
		}
	}
	return StateIndexEntry{}, false
}

// RequiredContextLength reports the largest prefix length needed by any entry.
func (index *StateIndex) RequiredContextLength() int {
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
func (entry StateIndexEntry) PrefixTokens() int {
	return entry.TokenStart + entry.TokenCount
}

// SaveStateIndex stores the index JSON in the same State store as its
// referenced bundle manifests.
func SaveStateIndex(ctx context.Context, store state.Writer, index *StateIndex, uri string) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return state.ChunkRef{}, core.NewError("mlx: state store is nil")
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, core.NewError("mlx: State index URI is required")
	}
	if err := index.Validate(); err != nil {
		return state.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(index), state.PutOptions{
		URI:    uri,
		Title:  "go-mlx State index",
		Kind:   StateIndexKind,
		Track:  "session-kv-index",
		Labels: stateIndexPutLabels,
	})
	if err != nil {
		return state.ChunkRef{}, core.E("kv.Snapshot.SaveStateIndex", "write State index", err)
	}
	return ref, nil
}

// SaveMemvidIndex stores the index JSON in the same old memvid-named store as its
// referenced bundle manifests.
//
// Deprecated: use SaveStateIndex.
func SaveMemvidIndex(ctx context.Context, store state.Writer, index *MemvidIndex, uri string) (state.ChunkRef, error) {
	return SaveStateIndex(ctx, store, index, uri)
}

// LoadStateIndex restores an index by URI from a State store.
func LoadStateIndex(ctx context.Context, store state.Store, uri string) (*StateIndex, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: state store is nil")
	}
	if core.Trim(uri) == "" {
		return nil, core.NewError("mlx: State index URI is required")
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadStateIndex", "resolve State index", err)
	}
	var index StateIndex
	if result := core.JSONUnmarshalString(chunk.Text, &index); !result.OK {
		return nil, core.E("LoadStateIndex", "parse State index", kv.ResultError(result))
	}
	if err := index.Validate(); err != nil {
		return nil, err
	}
	return &index, nil
}

// LoadMemvidIndex restores an index by URI from an old memvid-named store.
//
// Deprecated: use LoadStateIndex.
func LoadMemvidIndex(ctx context.Context, store state.Store, uri string) (*MemvidIndex, error) {
	return LoadStateIndex(ctx, store, uri)
}

// LoadPrefixFromStateIndex resolves entryURI through index,
// loads its referenced block bundle, and restores only the prefix required by
// that entry.
func LoadPrefixFromStateIndex(ctx context.Context, store state.Store, index *StateIndex, entryURI string, opts kv.LoadOptions) (*kv.Snapshot, StateIndexEntry, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, StateIndexEntry{}, core.NewError("mlx: state store is nil")
	}
	if err := index.Validate(); err != nil {
		return nil, StateIndexEntry{}, err
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, StateIndexEntry{}, core.NewError("mlx: State index entry not found")
	}
	bundleURI := entry.BundleURI
	if bundleURI == "" {
		bundleURI = index.BundleURI
	}
	bundle, err := kv.LoadStateBlockBundle(ctx, store, bundleURI)
	if err != nil {
		return nil, StateIndexEntry{}, err
	}
	prefixTokens := entry.PrefixTokens()
	if prefixTokens <= 0 || prefixTokens > bundle.TokenCount {
		return nil, StateIndexEntry{}, core.NewError("mlx: State index prefix is invalid")
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, opts)
	if err != nil {
		return nil, StateIndexEntry{}, err
	}
	return snapshot, entry, nil
}

// LoadPrefixFromMemvidIndex resolves entryURI through index, loads its
// referenced block bundle, and restores only the prefix required by that entry.
//
// Deprecated: use LoadPrefixFromStateIndex.
func LoadPrefixFromMemvidIndex(ctx context.Context, store state.Store, index *MemvidIndex, entryURI string, opts kv.LoadOptions) (*kv.Snapshot, MemvidIndexEntry, error) {
	return LoadPrefixFromStateIndex(ctx, store, index, entryURI, opts)
}

// CheckStateIndexCompatibility verifies model and tokenizer identity before
// restoring indexed State into a loaded model.
func CheckStateIndexCompatibility(info memory.ModelInfo, tokenizer bundle.Tokenizer, index *StateIndex) error {
	if err := index.Validate(); err != nil {
		return err
	}
	if index.Model.Architecture != "" && info.Architecture != "" && index.Model.Architecture != info.Architecture {
		return core.NewError("mlx: State index model architecture mismatch")
	}
	if index.Model.NumLayers > 0 && info.NumLayers > 0 && index.Model.NumLayers != info.NumLayers {
		return core.NewError("mlx: State index model layer mismatch")
	}
	if index.Model.QuantBits > 0 && info.QuantBits > 0 && index.Model.QuantBits != info.QuantBits {
		return core.NewError("mlx: State index model quantization mismatch")
	}
	if index.Model.Hash != "" && index.Model.Name == "" && index.Model.Path == "" && modelHashComparable(info, index.Model) {
		active := indexModel(nil, StateIndexOptions{ModelInfo: info})
		if active.Hash != "" && active.Hash != index.Model.Hash {
			return core.NewError("mlx: State index model hash mismatch")
		}
	}
	if info.ContextLength > 0 && index.RequiredContextLength() > info.ContextLength {
		return core.NewError("mlx: State index exceeds model context length")
	}
	if index.Tokenizer.Hash != "" && tokenizer.Hash != "" && index.Tokenizer.Hash != tokenizer.Hash {
		return core.NewError("mlx: State index tokenizer hash mismatch")
	}
	if index.Tokenizer.ChatTemplateHash != "" && tokenizer.ChatTemplateHash != "" && index.Tokenizer.ChatTemplateHash != tokenizer.ChatTemplateHash {
		return core.NewError("mlx: State index chat template hash mismatch")
	}
	return nil
}

// CheckMemvidIndexCompatibility verifies model and tokenizer
// identity before restoring indexed KV state into a loaded model.
//
// Deprecated: use CheckStateIndexCompatibility.
func CheckMemvidIndexCompatibility(info memory.ModelInfo, tokenizer bundle.Tokenizer, index *MemvidIndex) error {
	return CheckStateIndexCompatibility(info, tokenizer, index)
}

func modelHashComparable(info memory.ModelInfo, model bundle.Model) bool {
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

func indexModel(blk *kv.StateBlockBundle, opts StateIndexOptions) bundle.Model {
	info := opts.ModelInfo
	if info.Architecture == "" && blk != nil {
		info.Architecture = blk.Architecture
	}
	model := bundle.Model{
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
	builder := core.NewBuilder()
	builder.WriteString(model.Name)
	builder.WriteByte('\n')
	builder.WriteString(model.Path)
	builder.WriteByte('\n')
	builder.WriteString(model.Architecture)
	builder.WriteByte('\n')
	var intBuf [20]byte
	builder.Write(strconv.AppendInt(intBuf[:0], int64(model.VocabSize), 10))
	builder.WriteByte('\n')
	builder.Write(strconv.AppendInt(intBuf[:0], int64(model.NumLayers), 10))
	builder.WriteByte('\n')
	builder.Write(strconv.AppendInt(intBuf[:0], int64(model.QuantBits), 10))
	builder.WriteByte('\n')
	builder.Write(strconv.AppendInt(intBuf[:0], int64(model.ContextLength), 10))
	model.Hash = stateHash(builder.String())
	return model
}

func fillIndexEntryByteSpan(entry *StateIndexEntry, bundle *kv.StateBlockBundle) {
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
		chunk := kv.StateBlockChunkRef(ref)
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
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

func fillIndexEntryByteSpanSorted(entry *StateIndexEntry, bundle *kv.StateBlockBundle) {
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
	blocks := bundle.Blocks
	lo, hi := 0, len(blocks)
	for lo < hi {
		mid := lo + (hi-lo)/2
		ref := blocks[mid]
		if ref.TokenStart+ref.TokenCount <= spanStart {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	var (
		byteStartSet bool
		byteStart    int64
		byteCount    int64
	)
	for i := lo; i < len(blocks); i++ {
		ref := blocks[i]
		if ref.TokenStart >= spanEnd {
			break
		}
		chunk := kv.StateBlockChunkRef(ref)
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
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

func stateBlockRefsSortedByTokenStart(blocks []kv.StateBlockRef) bool {
	for i := 1; i < len(blocks); i++ {
		prev := blocks[i-1]
		current := blocks[i]
		if current.TokenStart < prev.TokenStart {
			return false
		}
		if current.TokenStart == prev.TokenStart && current.Index < prev.Index {
			return false
		}
	}
	return true
}

func indexHash(index *StateIndex) string {
	if index == nil {
		return ""
	}
	hash := sha256.New()
	writeIndexHashString(hash, index.Kind)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, index.BundleURI)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, index.SnapshotHash)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, string(index.KVEncoding))
	writeIndexHashString(hash, "|")
	writeIndexHashInt(hash, index.TokenCount)
	writeIndexHashString(hash, "|")
	writeIndexHashInt(hash, index.BlockSize)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, index.Model.Hash)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, index.Tokenizer.Hash)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, index.Tokenizer.ChatTemplateHash)
	for _, entry := range index.Entries {
		writeIndexHashString(hash, "|")
		entryHash := entry.Hash
		if entryHash == "" {
			entryHash = indexEntryHash(entry)
		}
		writeIndexHashString(hash, entryHash)
	}
	return core.HexEncode(hash.Sum(nil))
}

func indexEntryHash(entry StateIndexEntry) string {
	hash := sha256.New()
	writeIndexHashString(hash, entry.URI)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, entry.BundleURI)
	writeIndexHashString(hash, "|")
	writeIndexHashString(hash, entry.Title)
	writeIndexHashString(hash, "|")
	writeIndexHashInt(hash, entry.TokenStart)
	writeIndexHashString(hash, "|")
	writeIndexHashInt(hash, entry.TokenCount)
	writeIndexHashString(hash, "|")
	writeIndexHashInt64(hash, entry.ByteStart)
	writeIndexHashString(hash, "|")
	writeIndexHashInt64(hash, entry.ByteCount)
	for _, label := range entry.Labels {
		writeIndexHashString(hash, "|")
		writeIndexHashString(hash, label)
	}
	if len(entry.Meta) == 1 {
		for key, value := range entry.Meta {
			writeIndexHashString(hash, "|")
			writeIndexHashString(hash, key)
			writeIndexHashString(hash, "=")
			writeIndexHashString(hash, value)
		}
	} else if len(entry.Meta) > 1 {
		keys := make([]string, 0, len(entry.Meta))
		for key := range entry.Meta {
			keys = append(keys, key)
		}
		core.SliceSort(keys)
		for _, key := range keys {
			writeIndexHashString(hash, "|")
			writeIndexHashString(hash, key)
			writeIndexHashString(hash, "=")
			writeIndexHashString(hash, entry.Meta[key])
		}
	}
	return core.HexEncode(hash.Sum(nil))
}

func writeIndexHashString(h hash.Hash, value string) {
	h.Write(core.AsBytes(value))
}

func writeIndexHashInt(h hash.Hash, value int) {
	writeIndexHashInt64(h, int64(value))
}

func writeIndexHashInt64(h hash.Hash, value int64) {
	var buf [20]byte
	if value == 0 {
		buf[0] = '0'
		h.Write(buf[:1])
		return
	}
	negative := value < 0
	if negative {
		value = -value
	}
	i := len(buf)
	for value > 0 {
		i--
		buf[i] = byte('0' + value%10)
		value /= 10
	}
	if negative {
		i--
		buf[i] = '-'
	}
	h.Write(buf[i:])
}

func cloneIndexEntries(entries []StateIndexEntry) []StateIndexEntry {
	if len(entries) == 0 {
		return nil
	}
	out := make([]StateIndexEntry, len(entries))
	for i, entry := range entries {
		out[i] = cloneIndexEntry(entry)
	}
	return out
}

func cloneIndexEntry(entry StateIndexEntry) StateIndexEntry {
	entry.Labels = core.SliceClone(entry.Labels)
	entry.Meta = core.MapClone(entry.Meta)
	return entry
}
