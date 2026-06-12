// SPDX-Licence-Identifier: EUPL-1.2

package gguf

// Metadata returns a .gguf file's key/value metadata map without loading
// any tensor data. Values arrive as the parser's native Go types (string,
// bool, uint32/uint64/int32/int64, float32/float64, []any) — callers
// coerce per key.
//
// The gemma4 assistant GGUF loader reads its drafter config from the
// gemma4-assistant.* keys this returns (#92).
//
//	meta, err := gguf.Metadata("/models/MTP/gemma-4-31B-it-Q8_0-MTP.gguf")
//	arch, _ := meta["general.architecture"].(string)
func Metadata(path string) (map[string]any, error) {
	meta, _, err := parseGGUF(path)
	return meta, err
}
