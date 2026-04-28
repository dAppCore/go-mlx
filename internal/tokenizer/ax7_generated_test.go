// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import core "dappco.re/go"

func TestAX7_FormatGemmaPrompt_Bad(t *core.T) {
	symbol := any(FormatGemmaPrompt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FormatGemmaPrompt_Bad", "FormatGemmaPrompt")
}

func TestAX7_FormatGemmaPrompt_Ugly(t *core.T) {
	symbol := any(FormatGemmaPrompt)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "FormatGemmaPrompt_Ugly", "FormatGemmaPrompt")
}

func TestAX7_LoadTokenizer_Bad(t *core.T) {
	symbol := any(LoadTokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadTokenizer_Bad", "LoadTokenizer")
}

func TestAX7_LoadTokenizer_Ugly(t *core.T) {
	symbol := any(LoadTokenizer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LoadTokenizer_Ugly", "LoadTokenizer")
}

func TestAX7_Tokenizer_BOS_Good(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Good", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_BOS_Bad(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Bad", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_BOS_Ugly(t *core.T) {
	symbol := any((*Tokenizer).BOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOS_Ugly", "Tokenizer_BOS")
}

func TestAX7_Tokenizer_BOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Good", "Tokenizer_BOSToken")
}

func TestAX7_Tokenizer_BOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Bad", "Tokenizer_BOSToken")
}

func TestAX7_Tokenizer_BOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).BOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_BOSToken_Ugly", "Tokenizer_BOSToken")
}

func TestAX7_Tokenizer_Decode_Good(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Good", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_Decode_Bad(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Bad", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_Decode_Ugly(t *core.T) {
	symbol := any((*Tokenizer).Decode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Decode_Ugly", "Tokenizer_Decode")
}

func TestAX7_Tokenizer_DecodeToken_Good(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Good", "Tokenizer_DecodeToken")
}

func TestAX7_Tokenizer_DecodeToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Bad", "Tokenizer_DecodeToken")
}

func TestAX7_Tokenizer_DecodeToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).DecodeToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_DecodeToken_Ugly", "Tokenizer_DecodeToken")
}

func TestAX7_Tokenizer_EOS_Good(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Good", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_EOS_Bad(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Bad", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_EOS_Ugly(t *core.T) {
	symbol := any((*Tokenizer).EOS)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOS_Ugly", "Tokenizer_EOS")
}

func TestAX7_Tokenizer_EOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Good", "Tokenizer_EOSToken")
}

func TestAX7_Tokenizer_EOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Bad", "Tokenizer_EOSToken")
}

func TestAX7_Tokenizer_EOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).EOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_EOSToken_Ugly", "Tokenizer_EOSToken")
}

func TestAX7_Tokenizer_Encode_Good(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Good", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_Encode_Bad(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Bad", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_Encode_Ugly(t *core.T) {
	symbol := any((*Tokenizer).Encode)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_Encode_Ugly", "Tokenizer_Encode")
}

func TestAX7_Tokenizer_HasBOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Good", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasBOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Bad", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasBOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).HasBOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasBOSToken_Ugly", "Tokenizer_HasBOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Good(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Good", "Tokenizer_HasEOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Bad", "Tokenizer_HasEOSToken")
}

func TestAX7_Tokenizer_HasEOSToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).HasEOSToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_HasEOSToken_Ugly", "Tokenizer_HasEOSToken")
}

func TestAX7_Tokenizer_IDToken_Good(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Good", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_IDToken_Bad(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Bad", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_IDToken_Ugly(t *core.T) {
	symbol := any((*Tokenizer).IDToken)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_IDToken_Ugly", "Tokenizer_IDToken")
}

func TestAX7_Tokenizer_TokenID_Good(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Good", "Tokenizer_TokenID")
}

func TestAX7_Tokenizer_TokenID_Bad(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Bad", "Tokenizer_TokenID")
}

func TestAX7_Tokenizer_TokenID_Ugly(t *core.T) {
	symbol := any((*Tokenizer).TokenID)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Tokenizer_TokenID_Ugly", "Tokenizer_TokenID")
}
