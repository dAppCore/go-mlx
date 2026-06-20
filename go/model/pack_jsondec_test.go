// SPDX-Licence-Identifier: EUPL-1.2

// Differential tests for the hand-rolled sentence-transformers config
// walkers in pack_jsondec.go. Each walker replaced a core.JSONUnmarshal
// call; these assert the walker returns the identical (value, ok) the
// stdlib reflect path would have produced, across the edge cases that
// distinguish a correct port from an approximate one — top-level null,
// empty containers, unknown keys, duplicate keys, float-into-int, escapes,
// trailing garbage, and leading whitespace. Mirrors the differential
// pattern already guarding config_probe_unmarshal.go.

package model

import (
	"encoding/json"
	"testing"

	core "dappco.re/go"
)

// --- parseSingleIntField vs encoding/json into struct{int} ---

func controlSingleIntField(data []byte, key string) (int, bool) {
	// Reproduces readSentenceBertMaxSequence's original stdlib path: a
	// one-field struct whose tag is the key. (max_seq_length is the only
	// caller; the helper is generic but the control hard-codes that tag.)
	var config struct {
		MaxSequenceLength int `json:"max_seq_length"`
	}
	if key != "max_seq_length" {
		panic("control only models the max_seq_length tag")
	}
	if r := core.JSONUnmarshal(data, &config); !r.OK {
		return 0, false
	}
	return config.MaxSequenceLength, true
}

func TestParseSingleIntField_MatchesStdlib(t *testing.T) {
	cases := []string{
		`{"max_seq_length": 256}`,
		`{"max_seq_length": 0}`,
		`{"max_seq_length": -1}`,
		`{}`,
		`null`,
		`   null   `,
		`{"max_seq_length": 256, "other": "x"}`, // unknown key skipped
		`{"other": {"nested": [1,2,3]}, "max_seq_length": 7}`,
		`{"max_seq_length": 1, "max_seq_length": 2}`, // duplicate key (stdlib: last wins)
		`{"max_seq_length": 256.0}`,                  // float into int -> stdlib errors
		`{"max_seq_length": 256e2}`,                  // exp into int -> stdlib errors
		`{"max_seq_length": "256"}`,                  // string into int -> stdlib errors
		`{"max_seq_length": 01}`,                     // leading zero -> invalid
		`{"max_seq_length": null}`,                   // null field -> zero
		`{"max_seq_length": 256} trailing`,           // trailing garbage
		`  {"max_seq_length": 5}`,                    // leading whitespace
		`[]`,                                         // wrong shape (array)
		`{`,                                          // truncated
		`not json`,
		``,
	}
	for _, in := range cases {
		gotV, gotOK := parseSingleIntField([]byte(in), "max_seq_length")
		wantV, wantOK := controlSingleIntField([]byte(in), "max_seq_length")
		if gotV != wantV || gotOK != wantOK {
			t.Errorf("parseSingleIntField(%q) = (%d,%v); stdlib = (%d,%v)", in, gotV, gotOK, wantV, wantOK)
		}
	}
}

// --- parsePoolingFlags vs encoding/json into the four-bool struct ---

func controlPoolingName(data []byte) (string, bool) {
	// Reproduces readPoolingConfig's original stdlib path verbatim.
	var config struct {
		CLS          bool `json:"pooling_mode_cls_token"`
		Mean         bool `json:"pooling_mode_mean_tokens"`
		Max          bool `json:"pooling_mode_max_tokens"`
		WeightedMean bool `json:"pooling_mode_weightedmean_tokens"`
	}
	if r := core.JSONUnmarshal(data, &config); !r.OK {
		return "", false
	}
	switch {
	case config.Mean:
		return "mean", true
	case config.CLS:
		return "cls", true
	case config.Max:
		return "max", true
	case config.WeightedMean:
		return "weighted_mean", true
	}
	return "", false
}

// walkerPoolingName applies parsePoolingFlags then the same precedence
// switch readPoolingConfig uses, so the comparison is end-to-end on the
// observable (name, ok) the reader returns.
func walkerPoolingName(data []byte) (string, bool) {
	cls, mean, max, weightedMean, ok := parsePoolingFlags(data)
	if !ok {
		return "", false
	}
	switch {
	case mean:
		return "mean", true
	case cls:
		return "cls", true
	case max:
		return "max", true
	case weightedMean:
		return "weighted_mean", true
	}
	return "", false
}

func TestParsePoolingFlags_MatchesStdlib(t *testing.T) {
	cases := []string{
		`{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false}`,
		`{"pooling_mode_cls_token": true}`,
		`{"pooling_mode_max_tokens": true}`,
		`{"pooling_mode_weightedmean_tokens": true}`,
		`{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": true}`, // mean precedence
		`{}`,   // no mode -> ("",false)
		`null`, // null -> struct no-op -> ("",false)
		`   null `,
		`{"unknown": 1, "pooling_mode_mean_tokens": true}`,
		`{"pooling_mode_mean_tokens": false, "pooling_mode_mean_tokens": true}`, // duplicate, last wins
		`{"pooling_mode_mean_tokens": "true"}`,                                  // string into bool -> stdlib errors
		`{"pooling_mode_mean_tokens": 1}`,                                       // int into bool -> stdlib errors
		`{"pooling_mode_mean_tokens": null}`,                                    // null field -> false
		`{"pooling_mode_mean_tokens": true} x`,                                  // trailing garbage
		`  {"pooling_mode_cls_token": true}`,                                    // leading whitespace
		`[]`,
		`{`,
		``,
	}
	for _, in := range cases {
		gotN, gotOK := walkerPoolingName([]byte(in))
		wantN, wantOK := controlPoolingName([]byte(in))
		if gotN != wantN || gotOK != wantOK {
			t.Errorf("pooling(%q) = (%q,%v); stdlib = (%q,%v)", in, gotN, gotOK, wantN, wantOK)
		}
	}
}

// --- modulesDeclareNormalize vs encoding/json into []struct{Type,Path} ---

func controlModulesNormalize(data []byte) (bool, bool) {
	// Reproduces readSentenceTransformerNormalize's original stdlib path.
	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}
	if r := core.JSONUnmarshal(data, &modules); !r.OK {
		return false, false
	}
	for _, module := range modules {
		if containsASCIIInsensitive(module.Type, "normalize") || containsASCIIInsensitive(module.Path, "normalize") {
			return true, true
		}
	}
	return false, true
}

func TestModulesDeclareNormalize_MatchesStdlib(t *testing.T) {
	cases := []string{
		`[{"type":"sentence_transformers.models.Transformer","path":"0"},{"type":"sentence_transformers.models.Normalize","path":"2_Normalize"}]`,
		`[{"type":"X","path":"2_normalize"}]`,         // case-insensitive on path
		`[{"type":"NORMALIZE"}]`,                      // upper
		`[{"type":"Transformer"},{"type":"Pooling"}]`, // no normalize -> (false,true)
		`[]`,   // empty array -> (false,true)
		`null`, // null -> nil slice -> (false,true)
		`  null  `,
		`[{}]`, // entry with no type/path
		`[{"idx":0,"name":"0","type":"Normalize","path":""}]`, // extra keys skipped
		`[{"type":"a","type":"Normalize"}]`,                   // duplicate key (last wins -> normalize)
		`[{"type":"Normalize"}]`,                              // escape decodes to "Normalize"
		`[{"type":42}]`,                                       // int into string field -> stdlib errors
		`[{"type":"x"}] trailing`,                             // trailing garbage
		`  [{"type":"Normalize"}]`,                            // leading whitespace
		`{}`,                                                  // wrong shape (object)
		`[`,                                                   // truncated
		`[{"type":"x"}`,                                       // unterminated array
		``,
	}
	for _, in := range cases {
		gotN, gotOK := modulesDeclareNormalize([]byte(in))
		wantN, wantOK := controlModulesNormalize([]byte(in))
		if gotN != wantN || gotOK != wantOK {
			t.Errorf("modules(%q) = (%v,%v); stdlib = (%v,%v)", in, gotN, gotOK, wantN, wantOK)
		}
	}
}

// --- tokenizerJSONValidObject vs encoding/json into struct{} ---
//
// inspectModelPackTokenizer replaced a validity-only
// json.Unmarshal(data, &struct{}{}) (which allocates a fresh per-call
// scanner + decode state) with this zero-alloc gate. The control is that
// exact call, so the cases pin every boundary that distinguishes the gate
// from an approximation: malformed numbers / escapes the lenient jsonenc
// skippers would wave through, the top-level object-or-null rule a bare
// validity check would not enforce, and the trailing-junk / empty rejects.

func controlTokenizerValidObject(data []byte) bool {
	var probe struct{}
	return json.Unmarshal(data, &probe) == nil
}

func TestTokenizerJSONValidObject_MatchesStdlib(t *testing.T) {
	cases := []string{
		// Accepted: well-formed top-level objects (incl. arbitrary nested values).
		`{}`,
		`{"model":{"type":"BPE","vocab":{"h":0,"e":1}},"added_tokens":[{"id":1,"content":"<bos>","special":true}]}`,
		`{"a":[1,2,3],"b":"str","c":true,"d":null,"e":{"f":1.5e3}}`,
		`  {"a":1}  `,           // surrounding whitespace
		`{"a":"tokén"}`,         // raw multi-byte UTF-8
		"{\"a\":\"b\\u00e9c\"}", // valid \u escape (vocab byte-fallback)
		"{\"a\":\"b\\tc\"}",     // valid \t escape
		`{"a":1e3}`, `{"a":1E3}`, `{"a":1.5e-3}`, `{"a":-0}`, `{"a":0}`,
		// Accepted: top-level null (struct{} no-op).
		`null`, `   null   `,
		// Rejected: top-level non-object, non-null.
		`true`, `false`, `123`, `12.5`, `[]`, `[1,2,3]`, `"x"`, `"justastring"`,
		// Rejected: malformed numbers (lenient jsonenc skip would accept these).
		`{"a":1.2.3}`, `{"a":007}`, `{"a":1e}`, `{"a":--5}`, `{"a":1.}`,
		`{"a":.5}`, `{"a":0x10}`, `{"a":Infinity}`, `{"a":NaN}`,
		// Rejected: malformed string escapes (lenient skip would accept these).
		"{\"a\":\"b\\xc\"}",    // unknown \x escape
		"{\"a\":\"b\\u00gZ\"}", // bad \u hex
		"{\"a\":\"b\nc\"}",     // raw control char in string
		// Rejected: structural / trailing.
		`{"a":1,}`, `{"a":[1,2,]}`, `{,}`, `{"a":1} {"b":2}`, `{"a":1} trailing`,
		`{`, `not json`, `nul`, ``, `   `,
	}
	for _, in := range cases {
		got := tokenizerJSONValidObject([]byte(in))
		want := controlTokenizerValidObject([]byte(in))
		if got != want {
			t.Errorf("tokenizerJSONValidObject(%q) = %v; stdlib struct{} = %v", in, got, want)
		}
	}
}

// FuzzTokenizerJSONValidObject asserts the gate's accept/reject boundary is
// byte-identical to json.Unmarshal(data, &struct{}{}) on arbitrary bytes —
// the package's stated bar for a hand-rolled replacement of a stdlib decode.
func FuzzTokenizerJSONValidObject(f *testing.F) {
	for _, seed := range []string{
		`{}`, `null`, `{"a":1}`, `[]`, `"x"`, `123`, `true`,
		`{"a":1.2.3}`, "{\"a\":\"b\\xc\"}", `{"a":1} trailing`, ``,
		`{"model":{"type":"BPE"},"added_tokens":[]}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		got := tokenizerJSONValidObject(data)
		want := controlTokenizerValidObject(data)
		if got != want {
			t.Errorf("tokenizerJSONValidObject(%q) = %v; stdlib struct{} = %v", data, got, want)
		}
	})
}
