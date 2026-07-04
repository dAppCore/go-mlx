// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"net/http"

	mlx "dappco.re/go/mlx"
)

// adminPathServeStatus is the path of the active-config snapshot.
const adminPathServeStatus = "/v1/admin/serve/status"

// adminRuntimeMetal is the value the Runtime field carries from this
// binary. Sibling binaries (lthn-cuda, lthn-amd) populate the same
// field with "cuda" / "rocm" so consumers can branch on actual GPU
// backend without parsing the binary name.
const adminRuntimeMetal = "metal"

// adminServeStatus is the response shape for GET /v1/admin/serve/status.
// Field names stay backend-neutral so the same JSON works across the
// lthn-{mlx,cuda,amd} binary family; the Runtime field tells the
// caller which backend actually produced the snapshot.
type adminServeStatus struct {
	ModelPath    string                 `json:"model_path"`
	ProfilePath  string                 `json:"profile_path,omitempty"`
	Runtime      string                 `json:"runtime"`
	LoadedAtUnix int64                  `json:"loaded_at_unix"`
	Config       adminServeStatusConfig `json:"config"`
	Memory       adminServeStatusMemory `json:"memory"`
}

// adminServeStatusMemory is the live GPU memory snapshot, read per request
// (not at boot). ActiveBytes is what the runtime currently holds live;
// CacheBytes is the allocator's retained-but-free pool; PeakBytes is the
// high-water mark since load. The active/cache split is what tells you whether
// growth across a long generation is a real leak (active climbs) or just the
// allocator caching freed buffers (cache climbs, active flat).
type adminServeStatusMemory struct {
	ActiveBytes uint64 `json:"active_bytes"`
	CacheBytes  uint64 `json:"cache_bytes"`
	PeakBytes   uint64 `json:"peak_bytes"`
}

// adminServeStatusConfig mirrors the cross-backend LoadConfig fields
// that every GPU runtime (Metal / CUDA / ROCm) carries. Backend-only
// extras (SlidingWindow, etc.) are deliberately omitted from v1
// — add a `backend_specific` sub-object when a real consumer needs
// one. PromptCache is always rendered (true/false both meaningful).
type adminServeStatusConfig struct {
	ContextLength        int    `json:"context_length,omitempty"`
	ParallelSlots        int    `json:"parallel_slots,omitempty"`
	PromptCache          bool   `json:"prompt_cache"`
	PromptCacheMinTokens int    `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy          string `json:"cache_policy,omitempty"`
	CacheMode            string `json:"cache_mode,omitempty"`
	BatchSize            int    `json:"batch_size,omitempty"`
	PrefillChunkSize     int    `json:"prefill_chunk_size,omitempty"`
	ExpectedQuantization int    `json:"expected_quantization,omitempty"`
	MemoryLimitBytes     uint64 `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes      uint64 `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes      uint64 `json:"wired_limit_bytes,omitempty"`
	AdapterPath          string `json:"adapter_path,omitempty"`
}

// adminServeStatusHandler returns the snapshot of what serve was
// configured with at boot. Read-only, GET only. Behind Bearer auth
// like the rest of /v1/admin/*. Snapshot is captured at boot time
// rather than recomputed per request so the response shows the
// effective config at the moment of load (after profile resolution
// + --context override applied).
//
//	mux.HandleFunc(adminPathServeStatus, adminServeStatusHandler(snapshot))
func adminServeStatusHandler(snapshot adminServeStatus) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Memory is read live (the rest of the snapshot is boot-time) so a
		// caller can watch active vs cache climb across a long generation.
		snapshot.Memory = adminServeStatusMemory{
			ActiveBytes: mlx.GetActiveMemory(),
			CacheBytes:  mlx.GetCacheMemory(),
			PeakBytes:   mlx.GetPeakMemory(),
		}
		writeJSON(w, http.StatusOK, snapshot)
	}
}
