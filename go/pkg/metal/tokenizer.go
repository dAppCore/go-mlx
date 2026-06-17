// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go/mlx/pkg/tokenizer"

// The gemma4 SentencePiece/BPE tokenizer is pure Go and backend-agnostic, so it now lives
// in pkg/tokenizer — one implementation shared by every backend (the no-cgo native path and
// go-rocm can't import this cgo package). These aliases preserve the historic metal.* API so
// existing callers (backend, training, gemma4 loader, generate/session) compile unchanged.

// Tokenizer is the relocated gemma4 tokenizer (see pkg/tokenizer).
type Tokenizer = tokenizer.Tokenizer

// LoadTokenizer loads a tokenizer from a tokenizer.json path.
var LoadTokenizer = tokenizer.LoadTokenizer

// FormatGemmaPrompt wraps a prompt in the Gemma turn template.
var FormatGemmaPrompt = tokenizer.FormatGemmaPrompt

// NewForDecode builds a decode-only tokenizer from an inverse vocab (see pkg/tokenizer).
var NewForDecode = tokenizer.NewForDecode

// indexIn is the relocated substring-index helper, still used by lora.go.
var indexIn = tokenizer.IndexIn
