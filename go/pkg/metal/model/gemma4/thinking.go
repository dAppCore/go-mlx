// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Thought-channel delimiters (#99): gemma4 emits its reasoning between an
// atomic open token (<|channel>) and an atomic close token (<channel|>); the
// visible answer follows the close. Exposing the two token IDs lets the
// engine's thinking budget force the close when reasoning overruns. The IDs
// are resolved from the tokenizer (robust across snapshots) and cached.

package gemma4

import "dappco.re/go/mlx/pkg/metal"

const (
	gemma4ChannelOpenMarker  = "<|channel>"
	gemma4ChannelCloseMarker = "<channel|>"
)

// ThinkingChannelTokens returns the thought-channel open and close token IDs.
// Implements metal.ThinkingChannelModel so the decode loop can enforce a
// thinking budget without naming gemma4. Resolved per call from the live
// tokenizer (cheap — two short encodes, once per generation — and correct
// across a model reload).
func (m *Gemma4Model) ThinkingChannelTokens() (open, close int32, ok bool) {
	if m == nil || m.Tok == nil {
		return 0, 0, false
	}
	o, ook := gemma4SpecialTokenID(m.Tok, gemma4ChannelOpenMarker)
	c, cok := gemma4SpecialTokenID(m.Tok, gemma4ChannelCloseMarker)
	if !ook || !cok || o == c {
		return 0, 0, false
	}
	return o, c, true
}

// gemma4SpecialTokenID resolves a single special-token marker to its ID.
// Tokenizer.Encode may prepend BOS, so a leading BOS is stripped; the marker
// must then be exactly one token.
func gemma4SpecialTokenID(tok *metal.Tokenizer, marker string) (int32, bool) {
	ids := tok.Encode(marker)
	if tok.HasBOSToken() && len(ids) > 0 && ids[0] == tok.BOSToken() {
		ids = ids[1:]
	}
	if len(ids) != 1 {
		return 0, false
	}
	return ids[0], true
}
