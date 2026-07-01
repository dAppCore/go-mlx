// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	openaicompat "dappco.re/go/inference/provider/openai"
)

// The /v1/responses streaming hot path builds each "response.output_text.delta"
// frame by hand (appendJSONStringHTML into a reused buffer) instead of
// marshalling a ResponseStreamEvent struct, to keep the per-token encode off the
// reflect path. The wire format is contract -- the OpenAI SDK's SSE parser
// depends on the exact bytes -- so the hand-rolled encoder must stay
// byte-identical to core.JSONMarshalString (encoding/json). These tests lock
// that equivalence:
//
//   - appendJSONStringHTML == json.Marshal(string), including HTML-safety
//     escaping (< > &), the control-byte mnemonics, \u00XX, the line/paragraph
//     separators, and invalid UTF-8 -> the escaped replacement char.
//   - the assembled frame == writeSSEData(core.JSONMarshalString(event)) -- i.e.
//     the fixed prefix/suffix literals wrap the escaped delta exactly the way the
//     marshalled event would, so a typo in either literal is caught.

// escapeCorpus carries every input that broke an earlier draft of the escaper
// (kept as both a table for plain `go test` and as the fuzz seed set so CI,
// which does not run -fuzz, still exercises the regressions): HTML
// metacharacters, each control-byte mnemonic, a non-mnemonic control needing
// \u00XX, the Unicode line/paragraph separators, invalid UTF-8 (lone
// continuation + truncated sequence), multibyte runes, and an emoji. The
// separator entries use \u escapes so this source file stays plain ASCII.
var escapeCorpus = []string{
	"",
	"plain answer",
	"a < b && c > d",
	"<div class=\"x\">&amp;</div>",
	"</script>",
	"quote \" backslash \\ slash /",
	"tab\there",
	"newline\nhere",
	"carriage\rreturn",
	"bell\bhere",
	"formfeed\fhere",
	"vtab\x0bhere",               // 0x0b: no mnemonic, escapes via the \u00XX form
	"null\x00byte",               // 0x00: escapes via the \u00XX form
	"unit\x1fsep",                // 0x1f: escapes via the \u00XX form
	"del\x7fchar",                // 0x7f: passes through (>= 0x20, plain ASCII)
	"line sep",                   // U+2028 LINE SEPARATOR: HTML-safe escapes it
	"para sep",                   // U+2029 PARAGRAPH SEPARATOR: HTML-safe escapes it
	"café résumé",                // 2-byte runes pass through
	"日本語",                        // 3-byte runes pass through
	"emoji \U0001F600\U0001F680", // 4-byte runes pass through
	"\xb0",                       // lone continuation byte: invalid UTF-8
	"\xff\xfe",                   // invalid UTF-8, two bad bytes
	"valid\xc3then",              // truncated 2-byte sequence: invalid UTF-8
	"<|im_end|>",                 // stop-token-shaped delta
	"  ",                         // plain spaces: must NOT be treated as separators
}

func TestOpenai_appendJSONStringHTML_MatchesEncodingJSON(t *testing.T) {
	for _, s := range escapeCorpus {
		want := core.JSONMarshalString(s)
		got := string(appendJSONStringHTML(nil, s))
		if got != want {
			t.Errorf("appendJSONStringHTML(%q) = %q, want json.Marshal %q", s, got, want)
		}
	}
}

// FuzzAppendJSONStringHTML is the durable proof that the hand-rolled escaper is
// byte-identical to encoding/json's default string marshal. The oracle is
// core.JSONMarshalString -- the exact call the streaming path replaced. Plain
// `go test` runs the seed corpus (escapeCorpus); `-fuzz` explores beyond it.
func FuzzAppendJSONStringHTML(f *testing.F) {
	for _, s := range escapeCorpus {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		want := core.JSONMarshalString(s)
		got := string(appendJSONStringHTML(nil, s))
		if got != want {
			t.Fatalf("appendJSONStringHTML(%q) [% x] = %q, want %q", s, []byte(s), got, want)
		}
	})
}

// wantDeltaFrame is the byte string the old reflect path produced for one delta
// token: the marshalled event wrapped in the "data: ...\n\n" SSE framing.
func wantDeltaFrame(delta string) string {
	event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: delta}
	return core.Concat("data: ", core.JSONMarshalString(event), "\n\n")
}

func TestOpenai_writeResponseDeltaFrame_MatchesLegacy(t *testing.T) {
	for _, s := range escapeCorpus {
		if s == "" {
			// The caller never emits an empty delta (Delta is omitempty, so an
			// empty-delta event would marshal without the field). Equivalence is
			// only claimed for non-empty deltas -- see sseResponseDeltaFramePrefix.
			continue
		}
		var buf bytes.Buffer
		writeResponseDeltaFrame(&buf, nil, s)
		if got, want := buf.String(), wantDeltaFrame(s); got != want {
			t.Errorf("writeResponseDeltaFrame(%q) = %q, want legacy %q", s, got, want)
		}
	}
}

// TestOpenai_writeResponseDeltaFrame_BufferReuse proves the returned buffer is
// reused across tokens (the allocation win) and that reuse never corrupts a
// frame -- each call must overwrite, not append to, the prior contents.
func TestOpenai_writeResponseDeltaFrame_BufferReuse(t *testing.T) {
	var buf []byte
	for _, s := range []string{"first", "a < b", "third \U0001F600", "x"} {
		var sink bytes.Buffer
		buf = writeResponseDeltaFrame(&sink, buf, s)
		if got, want := sink.String(), wantDeltaFrame(s); got != want {
			t.Fatalf("reused frame for %q = %q, want %q", s, got, want)
		}
	}
}

// FuzzResponseDeltaFrame locks the end-to-end frame (prefix + escaped delta +
// suffix) against the marshalled-event-wrapped-in-SSE form. The escaper fuzz
// proves appendJSONStringHTML == json.Marshal; this proves the surrounding
// literals are correct too -- a typo in sseResponseDeltaFramePrefix/Suffix
// diverges here even when the escaper is right. Empty delta is skipped
// (omitempty).
func FuzzResponseDeltaFrame(f *testing.F) {
	for _, s := range escapeCorpus {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, delta string) {
		if delta == "" {
			return
		}
		var buf bytes.Buffer
		writeResponseDeltaFrame(&buf, nil, delta)
		if got, want := buf.String(), wantDeltaFrame(delta); got != want {
			t.Fatalf("frame(%q) [% x] = %q, want %q", delta, []byte(delta), got, want)
		}
	})
}
