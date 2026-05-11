// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// AgentMemoryWakeOptions selects a durable KV prefix to restore into a live
// session. EntryURI is optional when the index has exactly one natural first
// entry.
type AgentMemoryWakeOptions struct {
	Index                  *KVSnapshotMemvidBundleIndex
	IndexURI               string
	EntryURI               string
	Tokenizer              StateBundleTokenizer
	LoadOptions            kv.LoadOptions
	SkipCompatibilityCheck bool
}

// AgentMemoryWakeReport describes the restored durable prefix.
type AgentMemoryWakeReport struct {
	IndexURI     string `json:"index_uri,omitempty"`
	EntryURI     string `json:"entry_uri,omitempty"`
	BundleURI    string `json:"bundle_uri,omitempty"`
	Title        string `json:"title,omitempty"`
	PrefixTokens int    `json:"prefix_tokens,omitempty"`
	BundleTokens int    `json:"bundle_tokens,omitempty"`
	BlockSize    int    `json:"block_size,omitempty"`
	BlocksRead   int    `json:"blocks_read,omitempty"`
	IndexHash    string `json:"index_hash,omitempty"`
	SnapshotHash string `json:"snapshot_hash,omitempty"`
}

// AgentMemorySleepOptions controls how a live session is streamed to durable
// KV block storage.
type AgentMemorySleepOptions struct {
	EntryURI          string
	BundleURI         string
	IndexURI          string
	ParentEntryURI    string
	ParentBundleURI   string
	ParentIndexURI    string
	Title             string
	Model             string
	ModelPath         string
	ModelInfo         ModelInfo
	Tokenizer         StateBundleTokenizer
	ReuseParentPrefix bool
	BlockOptions      kv.MemvidBlockOptions
	Labels            []string
	Meta              map[string]string
}

// AgentMemorySleepReport describes the durable state written by Sleep.
type AgentMemorySleepReport struct {
	IndexURI        string             `json:"index_uri,omitempty"`
	EntryURI        string             `json:"entry_uri,omitempty"`
	BundleURI       string             `json:"bundle_uri,omitempty"`
	ParentEntryURI  string             `json:"parent_entry_uri,omitempty"`
	ParentBundleURI string             `json:"parent_bundle_uri,omitempty"`
	ParentIndexURI  string             `json:"parent_index_uri,omitempty"`
	Title           string             `json:"title,omitempty"`
	TokenCount      int                `json:"token_count,omitempty"`
	BlockSize       int                `json:"block_size,omitempty"`
	BlocksWritten   int                `json:"blocks_written,omitempty"`
	BlocksReused    int                `json:"blocks_reused,omitempty"`
	KVEncoding      kv.Encoding `json:"kv_encoding,omitempty"`
	IndexHash       string             `json:"index_hash,omitempty"`
	SnapshotHash    string             `json:"snapshot_hash,omitempty"`
	BundleRef       memvid.ChunkRef    `json:"bundle_ref,omitempty"`
	IndexRef        memvid.ChunkRef    `json:"index_ref,omitempty"`
}

type agentMemoryWakePlan struct {
	Index  *KVSnapshotMemvidBundleIndex
	Entry  KVSnapshotMemvidBundleIndexEntry
	Bundle *kv.MemvidBlockBundle
	Report *AgentMemoryWakeReport
}

func loadAgentMemoryWakeSnapshot(ctx context.Context, store memvid.Store, opts AgentMemoryWakeOptions, info ModelInfo) (*kv.Snapshot, *AgentMemoryWakeReport, error) {
	plan, err := planAgentMemoryWake(ctx, store, opts, info)
	if err != nil {
		return nil, nil, err
	}
	snapshot, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, plan.Bundle, plan.Entry.PrefixTokens(), opts.LoadOptions)
	if err != nil {
		return nil, nil, err
	}
	return snapshot, plan.Report, nil
}

func planAgentMemoryWake(ctx context.Context, store memvid.Store, opts AgentMemoryWakeOptions, info ModelInfo) (*agentMemoryWakePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	index, err := loadAgentMemoryIndex(ctx, store, opts)
	if err != nil {
		return nil, err
	}
	if !opts.SkipCompatibilityCheck {
		if err := CheckKVSnapshotMemvidBundleIndexCompatibility(info, opts.Tokenizer, index); err != nil {
			return nil, err
		}
	}
	entryURI := core.Trim(opts.EntryURI)
	if entryURI == "" && len(index.Entries) > 0 {
		entryURI = index.Entries[0].URI
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, core.NewError("mlx: memvid KV bundle index entry not found")
	}
	bundleURI := firstNonEmptyString(entry.BundleURI, index.BundleURI)
	bundle, err := kv.LoadMemvidBlockBundle(ctx, store, bundleURI)
	if err != nil {
		return nil, err
	}
	prefixTokens := entry.PrefixTokens()
	if prefixTokens <= 0 || prefixTokens > bundle.TokenCount {
		return nil, core.NewError("mlx: memvid KV bundle index prefix is invalid")
	}
	report := &AgentMemoryWakeReport{
		IndexURI:     opts.IndexURI,
		EntryURI:     entry.URI,
		BundleURI:    bundleURI,
		Title:        entry.Title,
		PrefixTokens: prefixTokens,
		BundleTokens: bundle.TokenCount,
		BlockSize:    bundle.BlockSize,
		BlocksRead:   kvSnapshotMemvidBlocksNeededForPrefix(bundle, prefixTokens),
		IndexHash:    index.Hash,
		SnapshotHash: bundle.SnapshotHash,
	}
	return &agentMemoryWakePlan{
		Index:  index,
		Entry:  entry,
		Bundle: bundle,
		Report: report,
	}, nil
}

func loadAgentMemoryIndex(ctx context.Context, store memvid.Store, opts AgentMemoryWakeOptions) (*KVSnapshotMemvidBundleIndex, error) {
	if opts.Index != nil {
		if err := opts.Index.Validate(); err != nil {
			return nil, err
		}
		return opts.Index, nil
	}
	if core.Trim(opts.IndexURI) == "" {
		return nil, core.NewError("mlx: agent memory index URI is required")
	}
	return LoadKVSnapshotMemvidBundleIndex(ctx, store, opts.IndexURI)
}

func agentMemorySleepURIs(opts AgentMemorySleepOptions) (entryURI, bundleURI, indexURI string, err error) {
	entryURI = core.Trim(opts.EntryURI)
	bundleURI = core.Trim(opts.BundleURI)
	indexURI = core.Trim(opts.IndexURI)
	if entryURI == "" {
		entryURI = firstNonEmptyString(bundleURI, indexURI, "mlx://agent-memory/latest")
	}
	if bundleURI == "" {
		bundleURI = entryURI + "/bundle"
	}
	if indexURI == "" {
		indexURI = entryURI + "/index"
	}
	if entryURI == "" || bundleURI == "" || indexURI == "" {
		return "", "", "", core.NewError("mlx: agent memory URI is required")
	}
	return entryURI, bundleURI, indexURI, nil
}

func agentMemoryBlockOptions(opts AgentMemorySleepOptions, bundleURI string) kv.MemvidBlockOptions {
	blockOpts := opts.BlockOptions
	if blockOpts.KVEncoding == "" {
		blockOpts.KVEncoding = kv.EncodingNative
	}
	if blockOpts.URI == "" {
		blockOpts.URI = bundleURI + "/blocks"
	}
	if blockOpts.Title == "" {
		blockOpts.Title = firstNonEmptyString(opts.Title, "go-mlx agent memory")
	}
	blockOpts.Labels = append([]string(nil), blockOpts.Labels...)
	blockOpts.Labels = append(blockOpts.Labels, "agent-memory")
	return blockOpts
}

func newAgentMemoryBundleIndex(bundle *kv.MemvidBlockBundle, opts AgentMemorySleepOptions, entryURI, bundleURI string) (*KVSnapshotMemvidBundleIndex, error) {
	entry := KVSnapshotMemvidBundleIndexEntry{
		URI:        entryURI,
		BundleURI:  bundleURI,
		Title:      opts.Title,
		TokenStart: 0,
		TokenCount: bundle.TokenCount,
		Labels:     append([]string(nil), opts.Labels...),
		Meta:       agentMemoryEntryMeta(opts),
	}
	if entry.Title == "" {
		entry.Title = "agent memory"
	}
	return NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: bundleURI,
		Title:     opts.Title,
		Model:     opts.Model,
		ModelPath: opts.ModelPath,
		ModelInfo: opts.ModelInfo,
		Tokenizer: opts.Tokenizer,
		Entries:   []KVSnapshotMemvidBundleIndexEntry{entry},
	})
}

func agentMemoryEntryMeta(opts AgentMemorySleepOptions) map[string]string {
	meta := cloneStringMap(opts.Meta)
	if opts.ParentEntryURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_entry_uri"] = opts.ParentEntryURI
	}
	if opts.ParentBundleURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_bundle_uri"] = opts.ParentBundleURI
	}
	if opts.ParentIndexURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_index_uri"] = opts.ParentIndexURI
	}
	return meta
}

func agentMemorySleepReport(index *KVSnapshotMemvidBundleIndex, bundle *kv.MemvidBlockBundle, opts AgentMemorySleepOptions, entryURI, bundleURI, indexURI string, bundleRef, indexRef memvid.ChunkRef) *AgentMemorySleepReport {
	return &AgentMemorySleepReport{
		IndexURI:        indexURI,
		EntryURI:        entryURI,
		BundleURI:       bundleURI,
		ParentEntryURI:  opts.ParentEntryURI,
		ParentBundleURI: opts.ParentBundleURI,
		ParentIndexURI:  opts.ParentIndexURI,
		Title:           opts.Title,
		TokenCount:      bundle.TokenCount,
		BlockSize:       bundle.BlockSize,
		BlocksWritten:   len(bundle.Blocks),
		BlocksReused:    bundle.ReusedBlocks,
		KVEncoding:      bundle.KVEncoding,
		IndexHash:       index.Hash,
		SnapshotHash:    bundle.SnapshotHash,
		BundleRef:       bundleRef,
		IndexRef:        indexRef,
	}
}

func agentMemoryWakeReportFromSleep(report *AgentMemorySleepReport) *AgentMemoryWakeReport {
	if report == nil {
		return nil
	}
	return &AgentMemoryWakeReport{
		IndexURI:     report.IndexURI,
		EntryURI:     report.EntryURI,
		BundleURI:    report.BundleURI,
		Title:        report.Title,
		PrefixTokens: report.TokenCount,
		BundleTokens: report.TokenCount,
		BlockSize:    report.BlockSize,
		BlocksRead:   0,
		IndexHash:    report.IndexHash,
		SnapshotHash: report.SnapshotHash,
	}
}

func cloneAgentMemoryWakeReport(report *AgentMemoryWakeReport) *AgentMemoryWakeReport {
	if report == nil {
		return nil
	}
	cloned := *report
	return &cloned
}

func kvSnapshotMemvidBlocksNeededForPrefix(bundle *kv.MemvidBlockBundle, prefixTokens int) int {
	if bundle == nil || prefixTokens <= 0 {
		return 0
	}
	count := 0
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		count++
		if ref.TokenStart+ref.TokenCount >= prefixTokens {
			break
		}
	}
	return count
}
