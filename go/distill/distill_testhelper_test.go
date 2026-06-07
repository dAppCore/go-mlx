// SPDX-Licence-Identifier: EUPL-1.2

package distill

import core "dappco.re/go"

// fakeSFTTokenizer is the test fake carried with the package on extraction (it
// was an unexported root helper in sft_test.go, not importable across the
// package boundary). It implements mlx.TokenizerImpl and is wrapped via
// mlx.NewTokenizer in the distillation tests.
type fakeSFTTokenizer struct {
	encoded map[string][]int32
	eos     int32
}

func (t fakeSFTTokenizer) Encode(text string) []int32 {
	if tokens, ok := t.encoded[text]; ok {
		return append([]int32(nil), tokens...)
	}
	out := make([]int32, 0, len(text))
	for _, r := range text {
		out = append(out, int32(r))
	}
	return out
}

func (t fakeSFTTokenizer) Decode(tokens []int32) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(core.Sprintf("%d", token))
	}
	return builder.String()
}

func (t fakeSFTTokenizer) TokenID(text string) (int32, bool) {
	tokens := t.Encode(text)
	if len(tokens) != 1 {
		return 0, false
	}
	return tokens[0], true
}

func (t fakeSFTTokenizer) IDToken(id int32) string { return core.Sprintf("%d", id) }

func (t fakeSFTTokenizer) DecodeOne(id int32) string { return t.Decode([]int32{id}) }

func (t fakeSFTTokenizer) BOS() int32        { return 0 }
func (t fakeSFTTokenizer) EOS() int32        { return t.eos }
func (t fakeSFTTokenizer) HasBOSToken() bool { return false }
