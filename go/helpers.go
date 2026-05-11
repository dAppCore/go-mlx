// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

// firstNonEmpty returns the first non-empty string after trimming whitespace.
// Shared across dataset_stream / kv_snapshot_index / memvid_chapter_smoke /
// model_pack and the legacy hf_fit alias surface.
//
//	value := firstNonEmpty(primary, fallback)
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

// firstPositive returns the first positive value from a list.
//
//	n := firstPositive(headDim*heads, hidden)
func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// renderTokensText concatenates Token.Text || Token.Value across a token
// slice. Used by memvid_chapter_smoke when no Text was reported.
//
//	text := renderTokensText(tokens)
func renderTokensText(tokens []Token) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(firstNonEmpty(token.Text, token.Value))
	}
	return builder.String()
}

// indexString locates substr inside s, returning its index or -1.
// Shared between hf_fit and openai.go.
//
//	pos := indexString(haystack, needle)
func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
