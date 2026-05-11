// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
)

// Legacy aliases — the canonical agent-memory + KV bundle index
// implementation lives at dappco.re/go/mlx/agent/. mlx-root callers
// keep their AgentMemoryWake/Sleep + KVSnapshotMemvidBundleIndex
// surface via these aliases.
type (
	AgentMemoryWakeOptions             = agent.WakeOptions
	AgentMemoryWakeReport              = agent.WakeReport
	AgentMemorySleepOptions            = agent.SleepOptions
	AgentMemorySleepReport             = agent.SleepReport
	KVSnapshotMemvidBundleIndex        = agent.MemvidIndex
	KVSnapshotMemvidBundleIndexEntry   = agent.MemvidIndexEntry
	KVSnapshotMemvidBundleIndexOptions = agent.MemvidIndexOptions
)

// NewKVSnapshotMemvidBundleIndex builds a per-bundle memvid lookup index.
//
//	idx, err := mlx.NewKVSnapshotMemvidBundleIndex(bundle, opts)
func NewKVSnapshotMemvidBundleIndex(b *kv.MemvidBlockBundle, opts KVSnapshotMemvidBundleIndexOptions) (*KVSnapshotMemvidBundleIndex, error) {
	return agent.NewMemvidIndex(b, opts)
}

// SaveKVSnapshotMemvidBundleIndex writes a memvid bundle index to durable storage.
//
//	ref, err := mlx.SaveKVSnapshotMemvidBundleIndex(ctx, store, idx, uri)
func SaveKVSnapshotMemvidBundleIndex(ctx context.Context, store memvid.Writer, idx *KVSnapshotMemvidBundleIndex, uri string) (memvid.ChunkRef, error) {
	return agent.SaveMemvidIndex(ctx, store, idx, uri)
}

// LoadKVSnapshotMemvidBundleIndex reads a memvid bundle index from durable storage.
//
//	idx, err := mlx.LoadKVSnapshotMemvidBundleIndex(ctx, store, uri)
func LoadKVSnapshotMemvidBundleIndex(ctx context.Context, store memvid.Store, uri string) (*KVSnapshotMemvidBundleIndex, error) {
	return agent.LoadMemvidIndex(ctx, store, uri)
}

// LoadKVSnapshotPrefixFromMemvidBundleIndex restores the prefix for one
// named entry inside a memvid bundle index.
//
//	snap, entry, err := mlx.LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx, store, idx, entryURI, opts)
func LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx context.Context, store memvid.Store, idx *KVSnapshotMemvidBundleIndex, entryURI string, opts kv.LoadOptions) (*kv.Snapshot, KVSnapshotMemvidBundleIndexEntry, error) {
	return agent.LoadPrefixFromMemvidIndex(ctx, store, idx, entryURI, opts)
}

// CheckKVSnapshotMemvidBundleIndexCompatibility verifies model +
// tokenizer compatibility before consuming a stored index.
//
//	if err := mlx.CheckKVSnapshotMemvidBundleIndexCompatibility(info, tokenizer, idx); err != nil { … }
func CheckKVSnapshotMemvidBundleIndexCompatibility(info ModelInfo, tokenizer StateBundleTokenizer, idx *KVSnapshotMemvidBundleIndex) error {
	return agent.CheckMemvidIndexCompatibility(modelInfoToMemory(info), tokenizer, idx)
}

// KVSnapshotMemvidBundleIndexKind identifies a memvid-stored lookup
// index. Forwarded from the agent package.
const KVSnapshotMemvidBundleIndexKind = agent.MemvidIndexKind

func loadAgentMemoryWakeSnapshot(ctx context.Context, store memvid.Store, opts AgentMemoryWakeOptions, info ModelInfo) (*kv.Snapshot, *AgentMemoryWakeReport, error) {
	return agent.LoadWakeSnapshot(ctx, store, opts, modelInfoToMemory(info))
}

func planAgentMemoryWake(ctx context.Context, store memvid.Store, opts AgentMemoryWakeOptions, info ModelInfo) (*agent.WakePlan, error) {
	return agent.PlanWake(ctx, store, opts, modelInfoToMemory(info))
}

func agentMemorySleepURIs(opts AgentMemorySleepOptions) (entryURI, bundleURI, indexURI string, err error) {
	return agent.SleepURIs(opts)
}

func agentMemoryBlockOptions(opts AgentMemorySleepOptions, bundleURI string) kv.MemvidBlockOptions {
	return agent.SleepBlockOptions(opts, bundleURI)
}

func newAgentMemoryBundleIndex(bundle *kv.MemvidBlockBundle, opts AgentMemorySleepOptions, entryURI, bundleURI string) (*KVSnapshotMemvidBundleIndex, error) {
	return agent.NewSleepIndex(bundle, opts, entryURI, bundleURI)
}

func agentMemorySleepReport(index *KVSnapshotMemvidBundleIndex, bundle *kv.MemvidBlockBundle, opts AgentMemorySleepOptions, entryURI, bundleURI, indexURI string, bundleRef, indexRef memvid.ChunkRef) *AgentMemorySleepReport {
	return agent.NewSleepReport(index, bundle, opts, entryURI, bundleURI, indexURI, bundleRef, indexRef)
}

func cloneAgentMemoryWakeReport(report *AgentMemoryWakeReport) *AgentMemoryWakeReport {
	return agent.CloneWakeReport(report)
}

func agentMemoryWakeReportFromSleep(report *AgentMemorySleepReport) *AgentMemoryWakeReport {
	return agent.WakeReportFromSleep(report)
}

func modelInfoToMemory(info ModelInfo) memory.ModelInfo {
	return memory.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}
