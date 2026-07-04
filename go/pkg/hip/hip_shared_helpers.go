// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import core "dappco.re/go"

// firstPositiveInt and rocmLabelUint were relocated here from model_pack.go
// during the pkg/hip quarantine landing (see the landing commit body):
// model_pack.go was excluded because it pervasively depends on the missing
// dappco.re/go/rocm/model (+ model/gemma4) packages, but these two small
// helpers have no such dependency and are still used by
// hip_gemma4_q4_layer.go, hip_gemma4_q4_prefill.go, hip_small_decode.go,
// hip_transformer_launch.go, kv_cache.go, native.go, and state_session.go.

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func rocmLabelUint(value string) uint64 {
	if value == "" {
		return 0
	}
	parsed := core.ParseInt(value, 10, 64)
	if !parsed.OK {
		return 0
	}
	if parsed.Value.(int64) < 0 {
		return 0
	}
	return uint64(parsed.Value.(int64))
}
