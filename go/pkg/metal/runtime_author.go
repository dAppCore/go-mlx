// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// runtime_author.go is the metal "runtime-author" SDK surface (RFC.model-sdk).
//
// A model package (e.g. package gemma4) owns its architecture *and* its
// speculative-decode runtime, but the runtime needs to drive metal's private
// generation machinery — the prompt cache, the device guard, the parallel-slot
// gate, the metrics sink, and the deep per-type K/V cache layout. Rather than
// leak the raw fields, metal exposes a curated set of exported accessors and
// operations here. They change no behaviour — each is a documented pass-through
// to an existing internal method or field.
package metal

import (
	"context"

	core "dappco.re/go"
)

// UnderlyingModel returns the raw loaded architecture so a runtime author can
// type-assert it to a concrete model (e.g. *gemma4.Gemma4Model). Unlike
// Internal, it does not wrap the model in a device shim — the caller is
// expected to drive the device itself via WithDevice.
//
//	if g, ok := m.UnderlyingModel().(*gemma4.Gemma4Model); ok { … }
func (m *Model) UnderlyingModel() InternalModel {
	if m == nil {
		return nil
	}
	return m.model
}

// RuntimeTokenizer returns the model's tokenizer for a runtime author that needs
// to encode prompts or decode generated tokens directly.
//
//	tokens := m.RuntimeTokenizer().Encode(prompt)
func (m *Model) RuntimeTokenizer() *Tokenizer {
	if m == nil {
		return nil
	}
	return m.tokenizer
}

// RequireTextRuntime reports whether the loaded model supports native text
// decode, returning a descriptive error otherwise. operation names the caller
// for the error message.
//
//	if err := m.RequireTextRuntime("Model.GenerateAssistant"); err != nil { return err }
func (m *Model) RequireTextRuntime(operation string) error {
	return m.requireTextRuntime(operation)
}

// AcquireSlot blocks until a parallel-generation slot is free (or ctx is done)
// and returns a release func the caller must defer. Unbounded models return a
// no-op release immediately.
//
//	release, err := m.AcquireSlot(ctx); if err != nil { return err }; defer release()
func (m *Model) AcquireSlot(ctx context.Context) (func(), error) {
	return m.acquireSlot(ctx)
}

// AcquirePromptCache locks the prompt cache for the duration of one generation
// and returns the unlock func to defer. Prompt-cache-disabled models return a
// no-op.
//
//	defer m.AcquirePromptCache()()
func (m *Model) AcquirePromptCache() func() {
	return m.acquirePromptCache()
}

// WithDevice runs fn on the model's Metal device. A runtime author wraps its
// per-token graph work in this so allocations land on the correct device.
//
//	err := m.WithDevice(func() { result, err = m.decodeLoop(...) })
func (m *Model) WithDevice(fn func()) error {
	return m.withDevice(fn)
}

// NewCachesWithRequestFixedSize builds a fresh per-layer K/V cache set sized for
// a single request (prompt + max new tokens). requestFixedSize is the value from
// GenerationFixedGemma4CacheSize.
//
//	caches := m.NewCachesWithRequestFixedSize(size)
func (m *Model) NewCachesWithRequestFixedSize(requestFixedSize int) []Cache {
	return m.newCachesWithRequestFixedSize(requestFixedSize)
}

// GenerationFixedGemma4CacheSize returns the fixed K/V cache length a Gemma 4
// fixed-cache generation should preallocate for the given prompt and token
// budget (0 = grow-as-needed).
//
//	size := m.GenerationFixedGemma4CacheSize(len(promptTokens), cfg.MaxTokens)
func (m *Model) GenerationFixedGemma4CacheSize(promptTokens, maxTokens int) int {
	return m.generationFixedGemma4CacheSize(promptTokens, maxTokens)
}

// RuntimeCachesSnapshotSafe reports whether the active cache mode supports prompt
// cache snapshotting. Quantised KV modes return false.
//
//	if m.RuntimeCachesSnapshotSafe() { m.StorePromptCacheWithHidden(...) }
func (m *Model) RuntimeCachesSnapshotSafe() bool {
	return m.runtimeCachesSnapshotSafe()
}

// PromptCacheEnabled reports whether prompt caching is active for this model.
//
//	if m.PromptCacheEnabled() { … }
func (m *Model) PromptCacheEnabled() bool {
	if m == nil {
		return false
	}
	return m.promptCacheEnabled
}

// PrefillChunkSize returns the configured prompt prefill chunk size (0 = no
// chunking). A runtime author chunks long prompts at this boundary.
//
//	if cs := m.PrefillChunkSize(); cs > 0 && len(tokens) > cs { … }
func (m *Model) PrefillChunkSize() int {
	if m == nil {
		return 0
	}
	return m.prefillChunkSize
}

// PromptCacheMinimum returns the minimum prompt length (tokens) below which a
// generation will not populate the prompt cache.
//
//	if len(tokens) >= m.PromptCacheMinimum() { … }
func (m *Model) PromptCacheMinimum() int {
	return m.promptCacheMinimum()
}

// SetLastErr records the error from the most recent generation so a later
// m.Err() reports it. A runtime author calls this on the failure paths of its
// own generation entry point, mirroring metal's built-in Generate.
//
//	if err != nil { m.SetLastErr(err) }
func (m *Model) SetLastErr(err error) {
	if m == nil {
		return
	}
	m.lastErr = err
}

// SetLastMetrics records the metrics from the most recent generation so a later
// m.LastMetrics() reports them. A runtime author populates this at the end of
// its generation loop, mirroring metal's built-in Generate.
//
//	m.SetLastMetrics(metrics)
func (m *Model) SetLastMetrics(metrics Metrics) {
	if m == nil {
		return
	}
	m.lastMetrics = metrics
}

// AdapterCacheKey returns the prompt-cache key fragment that identifies the
// active LoRA adapter (empty when none). A runtime author stamps this onto any
// prompt-cache entry it stores so an adapter swap invalidates the cache.
//
//	entry.SetAdapterHash(m.AdapterCacheKey())
func (m *Model) AdapterCacheKey() string {
	return m.adapterCacheKey()
}

// PromptCacheMatchWithHidden looks up the longest cached prefix of tokens that
// also carries usable hidden state (so a hidden-state-driven runtime can resume
// from it). It returns the matched entry and the prefix length, or (nil, 0) on a
// miss.
//
//	if entry, prefixLen := m.PromptCacheMatchWithHidden(tokens); entry != nil { … }
func (m *Model) PromptCacheMatchWithHidden(tokens []int32) (*PromptCacheEntry, int) {
	return m.promptCacheMatchWithHidden(tokens)
}

// StorePromptCacheEntry installs entry as the model's prompt cache, stamping it
// with the active adapter key and dropping any previous entry. A runtime author
// hands metal a freshly built entry (see NewPromptCacheEntryWithHidden) after a
// cache-miss prefill.
//
//	m.StorePromptCacheEntry(entry)
func (m *Model) StorePromptCacheEntry(entry *PromptCacheEntry) {
	if m == nil || entry == nil {
		return
	}
	entry.adapterHash = m.adapterCacheKey()
	m.clearPromptCache()
	m.promptCache = entry
}

// Logits returns the cached last-token logits for a fully-matched prefix, or nil
// when the entry holds none.
//
//	if l := entry.Logits(); l != nil && l.Valid() { … }
func (entry *PromptCacheEntry) Logits() *Array {
	if entry == nil {
		return nil
	}
	return entry.logits
}

// Hidden returns the cached last-token hidden state for a fully-matched prefix,
// or nil when the entry holds none. Used by hidden-state-driven runtimes (e.g.
// speculative decode).
//
//	if h := entry.Hidden(); h != nil && h.Valid() { … }
func (entry *PromptCacheEntry) Hidden() *Array {
	if entry == nil {
		return nil
	}
	return entry.hidden
}

// RestoreCaches rebuilds live per-layer K/V caches from the entry's snapshot,
// keeping prefixLen tokens and presizing to requestFixedSize. It wraps
// RestorePromptCachesWithRequestFixedSize over the entry's internal snapshot so
// a runtime author never touches the snapshot type.
//
//	caches, err := entry.RestoreCaches(prefixLen, requestFixedSize)
func (entry *PromptCacheEntry) RestoreCaches(prefixLen, requestFixedSize int) ([]Cache, error) {
	if entry == nil {
		return nil, core.NewError("metal: prompt cache entry is nil")
	}
	return RestorePromptCachesWithRequestFixedSize(entry.caches, prefixLen, requestFixedSize)
}
