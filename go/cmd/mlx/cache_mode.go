// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
)

const cacheModeFlagUsage = "override KV cache mode: fp16, q8, k-q8-v-q4, paged, or turboquant"

func parseRuntimeCacheMode(raw string) (memory.KVCacheMode, bool) {
	trimmed := core.Trim(raw)
	if trimmed == "" {
		return memory.KVCacheModeDefault, false
	}
	return memory.KVCacheMode(trimmed), true
}

func isRuntimeCacheMode(mode memory.KVCacheMode) bool {
	return mode != memory.KVCacheModeDefault && memory.IsKnownKVCacheMode(mode)
}
