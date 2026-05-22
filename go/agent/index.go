// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"crypto/sha256"
	"hash"
	"strconv"
	"strings"

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

// Sentinel validation errors hoisted to package scope. Each previously
// triggered a fresh core.NewError allocation per error-path hit; the
// hot Validate path returns one of these on every bad entry, and
// keeping them as singletons collapses N allocs → 0 on the failure
// branches and also lets callers errors.Is them.
var (
	errStateIndexNil                  = core.NewError("mlx: State index is nil")
	errStateIndexUnsupportedVersion   = core.NewError("mlx: unsupported State index version")
	errStateIndexInvalidKind          = core.NewError("mlx: invalid State index kind")
	errStateIndexEmptyTokenCount      = core.NewError("mlx: State index token count is empty")
	errStateIndexNoEntries            = core.NewError("mlx: State index has no entries")
	errStateIndexDuplicateURI         = core.NewError("mlx: duplicate State index URI")
	errStateIndexHashMismatch         = core.NewError("mlx: State index hash mismatch")
	errStateIndexEntryURIRequired     = core.NewError("mlx: State index entry URI is required")
	errStateIndexEntryBundleRequired  = core.NewError("mlx: State index entry bundle URI is required")
	errStateIndexEntryTokenStart      = core.NewError("mlx: State index entry token start is invalid")
	errStateIndexEntryTokenCount      = core.NewError("mlx: State index entry token count is empty")
	errStateIndexEntryExceedsBundle   = core.NewError("mlx: State index entry exceeds bundle token count")
	errStateIndexEntryByteSpan        = core.NewError("mlx: State index entry byte span is invalid")
	errStateIndexEntryHashMismatch    = core.NewError("mlx: State index entry hash mismatch")
	errStateIndexEntryNotFound        = core.NewError("mlx: State index entry not found")
	errStateIndexPrefixInvalid        = core.NewError("mlx: State index prefix is invalid")
	errStateStoreNil                  = core.NewError("mlx: state store is nil")
	errStateIndexURIRequired          = core.NewError("mlx: State index URI is required")
	errStateIndexArchitectureMismatch = core.NewError("mlx: State index model architecture mismatch")
	errStateIndexLayerMismatch        = core.NewError("mlx: State index model layer mismatch")
	errStateIndexQuantMismatch        = core.NewError("mlx: State index model quantization mismatch")
	errStateIndexModelHashMismatch    = core.NewError("mlx: State index model hash mismatch")
	errStateIndexExceedsContext       = core.NewError("mlx: State index exceeds model context length")
	errStateIndexTokenizerMismatch    = core.NewError("mlx: State index tokenizer hash mismatch")
	errStateIndexChatTemplateMismatch = core.NewError("mlx: State index chat template hash mismatch")
	errStateURIRequired               = core.NewError("mlx: State URI is required")
)

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
			return nil, errStateIndexEntryHashMismatch
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
		return errStateIndexNil
	}
	if index.Version <= 0 || index.Version > KVSnapshotStateBundleIndexVersion {
		return errStateIndexUnsupportedVersion
	}
	if index.Kind != StateIndexKind {
		return errStateIndexInvalidKind
	}
	if index.TokenCount <= 0 {
		return errStateIndexEmptyTokenCount
	}
	if len(index.Entries) == 0 {
		return errStateIndexNoEntries
	}
	seen := make(map[string]bool, len(index.Entries))
	indexBundleURIEmpty := core.Trim(index.BundleURI) == ""
	for _, entry := range index.Entries {
		if err := index.validateEntry(entry, checkHashes, indexBundleURIEmpty); err != nil {
			return err
		}
		if seen[entry.URI] {
			return errStateIndexDuplicateURI
		}
		seen[entry.URI] = true
	}
	if checkHashes && index.Hash != "" && index.Hash != indexHash(index) {
		return errStateIndexHashMismatch
	}
	return nil
}

func (index *StateIndex) validateEntry(entry StateIndexEntry, checkHash, indexBundleURIEmpty bool) error {
	if core.Trim(entry.URI) == "" {
		return errStateIndexEntryURIRequired
	}
	if indexBundleURIEmpty && core.Trim(entry.BundleURI) == "" {
		return errStateIndexEntryBundleRequired
	}
	if entry.TokenStart < 0 {
		return errStateIndexEntryTokenStart
	}
	if entry.TokenCount <= 0 {
		return errStateIndexEntryTokenCount
	}
	if entry.TokenStart+entry.TokenCount > index.TokenCount {
		return errStateIndexEntryExceedsBundle
	}
	if entry.ByteStart < 0 || entry.ByteCount < 0 {
		return errStateIndexEntryByteSpan
	}
	if checkHash && entry.Hash != "" && entry.Hash != indexEntryHash(entry) {
		return errStateIndexEntryHashMismatch
	}
	return nil
}

// Entry returns a defensive copy of the entry with URI.
func (index *StateIndex) Entry(uri string) (StateIndexEntry, bool) {
	if index == nil {
		return StateIndexEntry{}, false
	}
	for i := range index.Entries {
		if index.Entries[i].URI == uri {
			return cloneIndexEntry(index.Entries[i]), true
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
	for i := range index.Entries {
		if end := index.Entries[i].PrefixTokens(); end > required {
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
		return state.ChunkRef{}, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, errStateIndexURIRequired
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
		return nil, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return nil, errStateIndexURIRequired
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
		return nil, StateIndexEntry{}, errStateStoreNil
	}
	if err := index.Validate(); err != nil {
		return nil, StateIndexEntry{}, err
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, StateIndexEntry{}, errStateIndexEntryNotFound
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
		return nil, StateIndexEntry{}, errStateIndexPrefixInvalid
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
		return errStateIndexArchitectureMismatch
	}
	if index.Model.NumLayers > 0 && info.NumLayers > 0 && index.Model.NumLayers != info.NumLayers {
		return errStateIndexLayerMismatch
	}
	if index.Model.QuantBits > 0 && info.QuantBits > 0 && index.Model.QuantBits != info.QuantBits {
		return errStateIndexQuantMismatch
	}
	if index.Model.Hash != "" && index.Model.Name == "" && index.Model.Path == "" && modelHashComparable(info, index.Model) {
		active := indexModel(nil, StateIndexOptions{ModelInfo: info})
		if active.Hash != "" && active.Hash != index.Model.Hash {
			return errStateIndexModelHashMismatch
		}
	}
	if info.ContextLength > 0 && index.RequiredContextLength() > info.ContextLength {
		return errStateIndexExceedsContext
	}
	if index.Tokenizer.Hash != "" && tokenizer.Hash != "" && index.Tokenizer.Hash != tokenizer.Hash {
		return errStateIndexTokenizerMismatch
	}
	if index.Tokenizer.ChatTemplateHash != "" && tokenizer.ChatTemplateHash != "" && index.Tokenizer.ChatTemplateHash != tokenizer.ChatTemplateHash {
		return errStateIndexChatTemplateMismatch
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
	blocks := bundle.Blocks
	for i := range blocks {
		refStart := blocks[i].TokenStart
		refEnd := refStart + blocks[i].TokenCount
		if refEnd <= spanStart || refStart >= spanEnd {
			continue
		}
		chunk := kv.StateBlockChunkRef(blocks[i])
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
			byteStartSet = true
		}
		if blocks[i].PayloadByteCount > 0 {
			byteCount += int64(blocks[i].PayloadByteCount)
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
		if blocks[mid].TokenStart+blocks[mid].TokenCount <= spanStart {
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
		if blocks[i].TokenStart >= spanEnd {
			break
		}
		chunk := kv.StateBlockChunkRef(blocks[i])
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
			byteStartSet = true
		}
		if blocks[i].PayloadByteCount > 0 {
			byteCount += int64(blocks[i].PayloadByteCount)
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
		prevStart := blocks[i-1].TokenStart
		curStart := blocks[i].TokenStart
		if curStart < prevStart {
			return false
		}
		if curStart == prevStart && blocks[i].Index < blocks[i-1].Index {
			return false
		}
	}
	return true
}

// indexHash streams the canonical input into a sha256 hasher.
// Streaming wins for the index-level hash because the per-entry
// contribution scales linearly with len(Entries); using a single
// strings.Builder would double-and-copy its backing slice up through
// hundreds of KB before the Sum, which loses to sha256's fixed-size
// block buffer at scale (1000-entry index measured at 25 µs streaming
// vs 57 µs builder-batched).
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
	for i := range index.Entries {
		writeIndexHashString(hash, "|")
		entryHash := index.Entries[i].Hash
		if entryHash == "" {
			entryHash = indexEntryHash(index.Entries[i])
		}
		writeIndexHashString(hash, entryHash)
	}
	return core.HexEncode(hash.Sum(nil))
}

func indexEntryHash(entry StateIndexEntry) string {
	b := core.NewBuilder()
	// Pre-grow to a typical entry-hash input size to amortise the
	// 64→128→256→… backing slice growth chain. A representative
	// entry (rich, sorted meta) lands around 250 bytes; 320 leaves
	// headroom for the long-tail labels without overshooting.
	b.Grow(320)
	var intBuf [20]byte
	appendHashString(b, entry.URI)
	appendHashSep(b)
	appendHashString(b, entry.BundleURI)
	appendHashSep(b)
	appendHashString(b, entry.Title)
	appendHashSep(b)
	appendHashInt(b, intBuf[:], int64(entry.TokenStart))
	appendHashSep(b)
	appendHashInt(b, intBuf[:], int64(entry.TokenCount))
	appendHashSep(b)
	appendHashInt(b, intBuf[:], entry.ByteStart)
	appendHashSep(b)
	appendHashInt(b, intBuf[:], entry.ByteCount)
	for _, label := range entry.Labels {
		appendHashSep(b)
		appendHashString(b, label)
	}
	if len(entry.Meta) == 1 {
		for key, value := range entry.Meta {
			appendHashSep(b)
			appendHashString(b, key)
			b.WriteByte('=')
			appendHashString(b, value)
		}
	} else if len(entry.Meta) > 1 {
		keys := make([]string, 0, len(entry.Meta))
		for key := range entry.Meta {
			keys = append(keys, key)
		}
		core.SliceSort(keys)
		for _, key := range keys {
			appendHashSep(b)
			appendHashString(b, key)
			b.WriteByte('=')
			appendHashString(b, entry.Meta[key])
		}
	}
	sum := sha256.Sum256(core.AsBytes(b.String()))
	return core.HexEncode(sum[:])
}

func appendHashString(b *strings.Builder, value string) {
	b.WriteString(value)
}

func appendHashSep(b *strings.Builder) {
	b.WriteByte('|')
}

// appendHashInt formats value as decimal into scratch and writes the
// resulting bytes to b. scratch is caller-owned; the slice doesn't
// escape because *strings.Builder.Write is a concrete-receiver call
// (no interface dispatch).
func appendHashInt(b *strings.Builder, scratch []byte, value int64) {
	b.Write(strconv.AppendInt(scratch[:0], value, 10))
}

// writeIndexHashString / writeIndexHashInt / writeIndexHashInt64
// are retained for backward compatibility with any external caller
// (the test file may call them); kept as thin shims over the same
// interface dispatch they always had.
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
