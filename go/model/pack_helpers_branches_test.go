// SPDX-Licence-Identifier: EUPL-1.2

// Branch coverage for the residual arms of the pack helper files —
// the chat-template resolver, the dir index, the sentence-transformers
// JSON walkers, the quantisation config inspectors, and the task-profile
// readers. The _Good suites exercise the populated happy paths; the arms
// here are the not-present / malformed / fall-through branches those
// fixtures don't reach. Most are exercised by calling the unexported
// helper directly (where Inspect can't construct the precise input) with
// a few Inspect-level cases where the branch is only reachable through
// the full pipeline.

package model

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/gguf"
	mp "dappco.re/go/mlx/pack"
)

// --- pack_chattemplate.go ----------------------------------------------

// TestReadTokenizerChatTemplate_Branches drives the readTokenizerChatTemplate
// arms the existing GoodBad test leaves: an explicit null / absent
// chat_template (the `len(raw) == 0 || == "null"` arm), a not-present file
// (IsNotExist → ok=false,nil), a directory (non-ENOENT read error), an
// empty array and a whitespace-only array (the `[` scan that finds `]`),
// and an array whose first element is a value (named templates).
func TestReadTokenizerChatTemplate_Branches(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer_config.json")

	t.Run("explicit_null", func(t *testing.T) {
		writeModelPackFile(t, path, `{"chat_template":null}`)
		template, ok, err := readTokenizerChatTemplate(path)
		if err != nil || ok || template != "" {
			t.Fatalf("null chat_template = %q/%v/%v, want empty/false/nil", template, ok, err)
		}
	})

	t.Run("absent_key", func(t *testing.T) {
		writeModelPackFile(t, path, `{"model_max_length":131072}`)
		template, ok, err := readTokenizerChatTemplate(path)
		if err != nil || ok || template != "" {
			t.Fatalf("absent chat_template = %q/%v/%v, want empty/false/nil", template, ok, err)
		}
	})

	t.Run("empty_array", func(t *testing.T) {
		writeModelPackFile(t, path, `{"chat_template":[]}`)
		template, ok, err := readTokenizerChatTemplate(path)
		if err != nil || ok || template != "" {
			t.Fatalf("empty-array chat_template = %q/%v/%v, want empty/false/nil", template, ok, err)
		}
	})

	t.Run("whitespace_then_close_array", func(t *testing.T) {
		writeModelPackFile(t, path, "{\"chat_template\":[ \t\n ]}")
		template, ok, err := readTokenizerChatTemplate(path)
		if err != nil || ok || template != "" {
			t.Fatalf("whitespace-array chat_template = %q/%v/%v, want empty/false/nil", template, ok, err)
		}
	})

	t.Run("not_present", func(t *testing.T) {
		template, ok, err := readTokenizerChatTemplate(core.PathJoin(dir, "no_such_config.json"))
		if err != nil || ok || template != "" {
			t.Fatalf("missing file = %q/%v/%v, want empty/false/nil (IsNotExist)", template, ok, err)
		}
	})

	t.Run("directory_read_error", func(t *testing.T) {
		dirPath := core.PathJoin(dir, "as_a_dir")
		if result := core.MkdirAll(dirPath, 0o755); !result.OK {
			t.Fatalf("MkdirAll: %v", result.Value)
		}
		if _, ok, err := readTokenizerChatTemplate(dirPath); ok || err == nil {
			t.Fatalf("directory read should yield ok=false + non-nil err, got ok=%v err=%v", ok, err)
		}
	})
}

// TestReadJinjaChatTemplate_Branches drives readJinjaChatTemplate's
// not-present (IsNotExist) and directory (non-ENOENT) arms, plus a
// populated success.
func TestReadJinjaChatTemplate_Branches(t *testing.T) {
	dir := t.TempDir()

	t.Run("good", func(t *testing.T) {
		path := core.PathJoin(dir, "chat_template.jinja")
		writeModelPackFile(t, path, "  {{ messages }}  ")
		template, ok, err := readJinjaChatTemplate(path)
		if err != nil || !ok || template != "{{ messages }}" {
			t.Fatalf("jinja good = %q/%v/%v", template, ok, err)
		}
	})

	t.Run("not_present", func(t *testing.T) {
		template, ok, err := readJinjaChatTemplate(core.PathJoin(dir, "missing.jinja"))
		if err != nil || ok || template != "" {
			t.Fatalf("jinja missing = %q/%v/%v, want empty/false/nil", template, ok, err)
		}
	})

	t.Run("directory_read_error", func(t *testing.T) {
		dirPath := core.PathJoin(dir, "jinja_dir")
		if result := core.MkdirAll(dirPath, 0o755); !result.OK {
			t.Fatalf("MkdirAll: %v", result.Value)
		}
		if _, ok, err := readJinjaChatTemplate(dirPath); ok || err == nil {
			t.Fatalf("jinja directory should yield ok=false + non-nil err, got ok=%v err=%v", ok, err)
		}
	})
}

// TestInspectChatTemplate_FromTokenizerConfig drives the file-sourced
// success arm of inspectModelPackChatTemplate (pack_chattemplate.go:17-22):
// a model whose tokenizer_config.json carries a chat_template wins over
// the native template, recording the file source.
func TestInspectChatTemplate_FromTokenizerConfig(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma4_text"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer_config.json"),
		`{"chat_template":"{{ messages }}"}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect error = %v", err)
	}
	if !pack.HasChatTemplate || pack.ChatTemplateSource != mp.ModelPackChatTemplateFile {
		t.Fatalf("chat template = has:%v source:%q, want file source",
			pack.HasChatTemplate, pack.ChatTemplateSource)
	}
}

// TestInspectChatTemplate_BadTokenizerConfigWarns drives the
// `err != nil` warning arm (pack_chattemplate.go:23-25): a
// tokenizer_config.json that's a directory makes readTokenizerChatTemplate
// return an error, so the inspector raises a missing-chat-template warning
// and falls through to the native template.
func TestInspectChatTemplate_BadTokenizerConfigWarns(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma4_text"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	if result := core.MkdirAll(core.PathJoin(dir, "tokenizer_config.json"), 0o755); !result.OK {
		t.Fatalf("MkdirAll tokenizer_config.json: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect error = %v", err)
	}
	// Native gemma4 template still resolves, so HasChatTemplate ends true,
	// but the warning was raised along the way.
	if !pack.HasIssue(mp.ModelPackIssueMissingChatTemplate) {
		t.Fatalf("issues = %+v, want missing-chat-template warning from the bad tokenizer_config", pack.Issues)
	}
}

// TestInspectChatTemplate_BadJinjaWarns drives the jinja `err != nil`
// warning arm (pack_chattemplate.go:36-38): no tokenizer_config, a
// chat_template.jinja that's a directory, on an architecture with no
// native template — so the jinja read errors and warns.
func TestInspectChatTemplate_BadJinjaWarns(t *testing.T) {
	dir := t.TempDir()
	// bert has no native chat template, so the jinja path is consulted.
	writeModelPackFile(t, core.PathJoin(dir, "config.json"),
		`{"architectures":["BertModel"],"model_type":"bert"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	if result := core.MkdirAll(core.PathJoin(dir, "chat_template.jinja"), 0o755); !result.OK {
		t.Fatalf("MkdirAll chat_template.jinja: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueMissingChatTemplate) {
		t.Fatalf("issues = %+v, want missing-chat-template warning from the bad jinja", pack.Issues)
	}
}

// --- pack_dirindex.go --------------------------------------------------

// TestModelPackDirIndex_Branches drives the dir-index arms the pipeline
// fixtures don't pin: the nil/unpopulated `has` short-circuit, the
// unrecognised-name fall-through (`return true`), the nil-receiver
// `record` no-op, and a record/has round-trip for every recognised name.
func TestModelPackDirIndex_Branches(t *testing.T) {
	t.Run("nil_receiver", func(t *testing.T) {
		var d *modelPackDirIndex
		if !d.has("tokenizer.json") {
			t.Errorf("nil index has() should return true (fall through to ReadFile)")
		}
		d.record("tokenizer.json") // must not panic
	})

	t.Run("unpopulated", func(t *testing.T) {
		d := &modelPackDirIndex{} // populated=false
		if !d.has("tokenizer.json") {
			t.Errorf("unpopulated index has() should return true")
		}
	})

	t.Run("unknown_name_returns_true", func(t *testing.T) {
		d := &modelPackDirIndex{populated: true}
		if !d.has("something_unlisted.json") {
			t.Errorf("populated index has(unknown) should return true (defer to normal probe)")
		}
		d.record("something_unlisted.json") // no-op, must not panic
	})

	t.Run("record_then_has", func(t *testing.T) {
		d := &modelPackDirIndex{populated: true}
		names := []string{
			"jang_config.json", "auto_round_config.json", "quantization_config.json",
			"codebook_config.json", "tokenizer_config.json", "tokenizer.json",
			"chat_template.jinja", "sentence_bert_config.json", "modules.json",
		}
		for _, name := range names {
			if d.has(name) {
				t.Errorf("has(%q) before record should be false", name)
			}
			d.record(name)
			if !d.has(name) {
				t.Errorf("has(%q) after record should be true", name)
			}
		}
	})
}

// --- pack_jsondec.go ---------------------------------------------------

// TestParseSingleIntField_Branches drives parseSingleIntField across its
// arms: top-level null, a present null value, a missing key (zero, ok),
// a populated value, an unknown key skipped, and the malformed-body /
// trailing-junk ok=false returns.
func TestParseSingleIntField_Branches(t *testing.T) {
	cases := []struct {
		name    string
		json    string
		want    int
		wantOK  bool
		wantKey string
	}{
		{"top_null", `null`, 0, true, "max_seq_length"},
		{"value_null", `{"max_seq_length":null}`, 0, true, "max_seq_length"},
		{"missing_key", `{"other":3}`, 0, true, "max_seq_length"},
		{"populated", `{"max_seq_length":256}`, 256, true, "max_seq_length"},
		{"unknown_skipped", `{"foo":{"a":1},"max_seq_length":128}`, 128, true, "max_seq_length"},
		{"bad_value", `{"max_seq_length":"x"}`, 0, false, "max_seq_length"},
		{"not_object", `5`, 0, false, "max_seq_length"},
		{"trailing_junk", `{"max_seq_length":256} extra`, 0, false, "max_seq_length"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseSingleIntField([]byte(tc.json), tc.wantKey)
			if ok != tc.wantOK || (ok && got != tc.want) {
				t.Fatalf("parseSingleIntField(%q) = (%d,%v), want (%d,%v)", tc.json, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestParsePoolingFlags_Branches drives parsePoolingFlags: top-level null,
// each recognised flag, a null flag value, an unknown key skipped, and the
// malformed / non-bool / trailing-junk ok=false returns.
func TestParsePoolingFlags_Branches(t *testing.T) {
	t.Run("top_null", func(t *testing.T) {
		_, _, _, _, ok := parsePoolingFlags([]byte(`null`))
		if !ok {
			t.Fatalf("top-level null should be ok=true")
		}
	})
	t.Run("mean_set", func(t *testing.T) {
		cls, mean, _, _, ok := parsePoolingFlags([]byte(`{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":false}`))
		if !ok || !mean || cls {
			t.Fatalf("mean flags = cls:%v mean:%v ok:%v", cls, mean, ok)
		}
	})
	t.Run("weighted_mean_set", func(t *testing.T) {
		_, _, _, wm, ok := parsePoolingFlags([]byte(`{"pooling_mode_weightedmean_tokens":true}`))
		if !ok || !wm {
			t.Fatalf("weighted-mean = %v ok:%v", wm, ok)
		}
	})
	t.Run("null_flag_value", func(t *testing.T) {
		_, _, _, _, ok := parsePoolingFlags([]byte(`{"pooling_mode_cls_token":null}`))
		if !ok {
			t.Fatalf("null flag value should parse ok=true")
		}
	})
	t.Run("unknown_skipped", func(t *testing.T) {
		_, mean, _, _, ok := parsePoolingFlags([]byte(`{"word_embedding_dimension":384,"pooling_mode_mean_tokens":true}`))
		if !ok || !mean {
			t.Fatalf("unknown-skipped = mean:%v ok:%v", mean, ok)
		}
	})
	for _, bad := range []string{`{"pooling_mode_cls_token":"yes"}`, `5`, `{"pooling_mode_cls_token":true} junk`} {
		t.Run("bad_"+bad, func(t *testing.T) {
			if _, _, _, _, ok := parsePoolingFlags([]byte(bad)); ok {
				t.Fatalf("parsePoolingFlags(%q) ok=true, want false", bad)
			}
		})
	}
}

// TestModulesDeclareNormalize_Branches drives modulesDeclareNormalize:
// top-level null, an empty array, an empty array with trailing junk, a
// null type/path value, a normalize hit, a non-normalize array, and the
// malformed-shape ok=false returns (not an array, bad element, missing
// comma, trailing junk after close).
func TestModulesDeclareNormalize_Branches(t *testing.T) {
	t.Run("top_null", func(t *testing.T) {
		norm, ok := modulesDeclareNormalize([]byte(`null`))
		if !ok || norm {
			t.Fatalf("top null = norm:%v ok:%v, want false/true", norm, ok)
		}
	})
	t.Run("empty_array", func(t *testing.T) {
		norm, ok := modulesDeclareNormalize([]byte(`[]`))
		if !ok || norm {
			t.Fatalf("empty array = norm:%v ok:%v, want false/true", norm, ok)
		}
	})
	t.Run("empty_array_trailing_junk", func(t *testing.T) {
		if _, ok := modulesDeclareNormalize([]byte(`[] junk`)); ok {
			t.Fatalf("trailing junk after empty array should be ok=false")
		}
	})
	t.Run("null_field_value", func(t *testing.T) {
		// type:null is consumed via the IsJSONNull arm, path carries the hit.
		norm, ok := modulesDeclareNormalize([]byte(`[{"type":null,"path":"2_Normalize"}]`))
		if !ok || !norm {
			t.Fatalf("null type + normalize path = norm:%v ok:%v", norm, ok)
		}
	})
	t.Run("normalize_hit", func(t *testing.T) {
		norm, ok := modulesDeclareNormalize([]byte(`[{"type":"sentence_transformers.models.Normalize","path":""}]`))
		if !ok || !norm {
			t.Fatalf("normalize module = norm:%v ok:%v", norm, ok)
		}
	})
	t.Run("no_normalize", func(t *testing.T) {
		norm, ok := modulesDeclareNormalize([]byte(`[{"type":"sentence_transformers.models.Transformer","path":""}]`))
		if !ok || norm {
			t.Fatalf("non-normalize module = norm:%v ok:%v, want false/true", norm, ok)
		}
	})
	for _, bad := range []string{
		`{"type":"x"}`,             // not an array
		`[{"type":5}]`,             // non-string element value
		`[{"type":"a"} {"path":1}]`, // missing comma between elements
		`[{"type":"x"}] trailing`,   // trailing junk after close
	} {
		t.Run("bad_"+bad, func(t *testing.T) {
			if _, ok := modulesDeclareNormalize([]byte(bad)); ok {
				t.Fatalf("modulesDeclareNormalize(%q) ok=true, want false", bad)
			}
		})
	}
}

// TestWalkFlatObject_ErrorArms drives the two walkFlatObject error arms
// the other walkers don't hit through their callers: a malformed key
// string (ParseJSONStringRaw error) and a missing colon after the key.
// Driven through parseSingleIntField, whose body is walkFlatObject.
func TestWalkFlatObject_ErrorArms(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"bad_key_escape", `{"max\xseq":5}`},
		{"missing_colon", `{"max_seq_length" 5}`},
		{"eof_after_key", `{"max_seq_length"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, ok := parseSingleIntField([]byte(tc.json), "max_seq_length"); ok {
				t.Fatalf("parseSingleIntField(%q) ok=true, want false", tc.json)
			}
		})
	}
}

// --- pack_quantinspect.go ----------------------------------------------

// TestInspectQuantConfigs_ParseErrors drives the parse-error arms of the
// three quant-config inspectors via Inspect: a malformed jang_config.json
// (JANG warning), a malformed auto_round_config.json (AutoRound error),
// and a malformed codebook_config.json (codebook error).
func TestInspectQuantConfigs_ParseErrors(t *testing.T) {
	t.Run("jang", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "jang_config.json"), "{ not json")
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
		pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect error = %v", err)
		}
		if !pack.HasIssue(mp.ModelPackIssueQuantizationMismatch) {
			t.Fatalf("issues = %+v, want jang parse warning", pack.Issues)
		}
	})

	t.Run("auto_round", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "auto_round_config.json"), "{ not json")
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
		pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect error = %v", err)
		}
		if !pack.HasIssue(mp.ModelPackIssueUnsupportedAutoRound) {
			t.Fatalf("issues = %+v, want auto-round parse error", pack.Issues)
		}
	})

	t.Run("codebook", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "codebook_config.json"), "{ not json")
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
		pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect error = %v", err)
		}
		if !pack.HasIssue(mp.ModelPackIssueUnsupportedCodebook) {
			t.Fatalf("issues = %+v, want codebook parse error", pack.Issues)
		}
	})
}

// TestCloneGGUFQuantizationInfo_Empty drives cloneGGUFQuantizationInfo's
// empty-input nil return (pack_quantinspect.go:113-115): a zero-value
// QuantizationInfo clones to nil rather than an empty allocation, while a
// populated one deep-clones its tensor-type slice.
func TestCloneGGUFQuantizationInfo_Empty(t *testing.T) {
	if got := cloneGGUFQuantizationInfo(gguf.QuantizationInfo{}); got != nil {
		t.Fatalf("clone of empty info = %+v, want nil", got)
	}
	src := gguf.QuantizationInfo{Type: "Q4_K", Family: "gguf", Bits: 4}
	got := cloneGGUFQuantizationInfo(src)
	if got == nil || got.Type != "Q4_K" || got.Bits != 4 {
		t.Fatalf("clone of populated info = %+v, want a Q4_K copy", got)
	}
}

// --- pack_taskprofiles.go ----------------------------------------------

// TestInspectModelPackTaskProfiles_NilPack drives the `pack == nil` guard
// (pack_taskprofiles.go:11-13) and modelPackCapabilities' own nil guard
// (pack.go:389) — both must no-op / return nil rather than panic.
func TestInspectModelPackTaskProfiles_NilPack(t *testing.T) {
	inspectModelPackTaskProfiles(nil, "", nil) // must not panic
	if got := modelPackCapabilities(nil); got != nil {
		t.Fatalf("modelPackCapabilities(nil) = %v, want nil", got)
	}
}

// TestInspectModelPackEmbeddingProfile_EmptyRoot drives the `root == ""`
// early return in inspectModelPackEmbeddingProfile (pack_taskprofiles.go:41-43)
// — with no root the transformers defaults stand, no sentence-transformers
// files are read.
func TestInspectModelPackEmbeddingProfile_EmptyRoot(t *testing.T) {
	pack := &mp.ModelPack{HiddenSize: 384, ContextLength: 512}
	profile := inspectModelPackEmbeddingProfile(pack, "", nil)
	if profile.Source != "transformers" || profile.Dimension != 384 || profile.Pooling != "cls" {
		t.Fatalf("empty-root embedding profile = %+v, want transformers defaults", profile)
	}
}

// TestInspectModelPackRerankProfile_WithSentenceBert drives the
// sentence-transformers max-sequence arm in inspectModelPackRerankProfile
// (pack_taskprofiles.go:66-69): a cross-encoder whose sentence_bert_config
// supplies a max_seq_length overrides the context length and flips the
// source. Built via a bert_rerank pack so the rerank inspector fires.
func TestInspectModelPackRerankProfile_WithSentenceBert(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "sentence_bert_config.json"), `{"max_seq_length": 384}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect error = %v", err)
	}
	if pack.Rerank == nil {
		t.Fatalf("Rerank = nil, want a cross-encoder profile")
	}
	if pack.Rerank.MaxSequenceLength != 384 || pack.Rerank.Source != "sentence-transformers" {
		t.Fatalf("Rerank = %+v, want max-seq 384 from sentence-transformers", pack.Rerank)
	}
}

// TestReadSentenceTransformerNormalize_NotPresentAndBad drives
// readSentenceTransformerNormalize's not-present (no modules.json →
// dir.has false) and unreadable (directory) arms (pack_taskprofiles.go:154-156),
// called directly so the (false,false) boundary is pinned without a full
// embedding pack.
func TestReadSentenceTransformerNormalize_NotPresentAndBad(t *testing.T) {
	dir := t.TempDir()

	t.Run("not_listed", func(t *testing.T) {
		// populated index that does NOT record modules.json → has() false.
		idx := &modelPackDirIndex{populated: true}
		if norm, ok := readSentenceTransformerNormalize(dir, idx); ok || norm {
			t.Fatalf("absent modules.json = norm:%v ok:%v, want false/false", norm, ok)
		}
	})

	t.Run("unreadable", func(t *testing.T) {
		// modules.json is a directory → ReadFile fails → (false,false).
		if result := core.MkdirAll(core.PathJoin(dir, "modules.json"), 0o755); !result.OK {
			t.Fatalf("MkdirAll modules.json: %v", result.Value)
		}
		idx := &modelPackDirIndex{populated: true}
		idx.record("modules.json")
		if norm, ok := readSentenceTransformerNormalize(dir, idx); ok || norm {
			t.Fatalf("directory modules.json = norm:%v ok:%v, want false/false", norm, ok)
		}
	})
}
