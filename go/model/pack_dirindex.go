// SPDX-Licence-Identifier: EUPL-1.2

package model

// modelPackDirIndex caches presence of the specific optional-config
// filenames the inspect pipeline probes downstream — built from the
// same single PathGlob the weight inspector already runs, so this is
// opportunistic and adds no extra syscall. The index records the seven
// basenames we'd otherwise ReadFile-then-IsNotExist for, in fixed bool
// fields, so populating + querying is zero-alloc.
//
// The `populated` flag lets callers distinguish "no listing available"
// (single-file resolvedPath) from "listed but file absent" — the
// former falls through to the regular ReadFile probe so semantics for
// the single-file entry path stay unchanged.
//
// tokenizer.json is included so inspectModelPackTokenizer can skip a
// ReadFile + IsNotExist round-trip when the model directory has no
// tokenizer — the missing-tokenizer error path runs on every Inspect
// against a partial download or weights-only pack.
type modelPackDirIndex struct {
	populated         bool
	jangConfig        bool
	autoRoundConfig   bool
	quantConfig       bool
	codebookConfig    bool
	tokenizerConfig   bool
	tokenizerJSON     bool
	chatTemplateJinja bool
	sentenceBert      bool
	modulesJSON       bool
	// poolingDir holds the basename of the first sentence-transformers
	// "*_Pooling" subdirectory seen in the weight glob, so the embedding
	// inspector can read its config.json by direct path instead of
	// re-walking the directory tree with a second `*_Pooling/config.json`
	// glob. Empty when no such directory was listed.
	poolingDir string
}

// has reports whether the named direct child of root is present in the
// pre-fetched listing. Returns true if the index is empty (no listing
// available) so callers fall through to the existing ReadFile probe —
// the precise root-stat is preserved in that path. The name argument
// is one of the seven recognised optional-config filenames; anything
// else returns true (let the caller perform the normal probe).
func (d *modelPackDirIndex) has(name string) bool {
	if d == nil || !d.populated {
		return true
	}
	switch name {
	case "jang_config.json":
		return d.jangConfig
	case "auto_round_config.json":
		return d.autoRoundConfig
	case "quantization_config.json":
		return d.quantConfig
	case "codebook_config.json":
		return d.codebookConfig
	case "tokenizer_config.json":
		return d.tokenizerConfig
	case "tokenizer.json":
		return d.tokenizerJSON
	case "chat_template.jinja":
		return d.chatTemplateJinja
	case "sentence_bert_config.json":
		return d.sentenceBert
	case "modules.json":
		return d.modulesJSON
	}
	return true
}

// record marks the matching field when basename is one of the
// recognised optional-config filenames; otherwise it's a no-op.
func (d *modelPackDirIndex) record(basename string) {
	if d == nil {
		return
	}
	switch basename {
	case "jang_config.json":
		d.jangConfig = true
	case "auto_round_config.json":
		d.autoRoundConfig = true
	case "quantization_config.json":
		d.quantConfig = true
	case "codebook_config.json":
		d.codebookConfig = true
	case "tokenizer_config.json":
		d.tokenizerConfig = true
	case "tokenizer.json":
		d.tokenizerJSON = true
	case "chat_template.jinja":
		d.chatTemplateJinja = true
	case "sentence_bert_config.json":
		d.sentenceBert = true
	case "modules.json":
		d.modulesJSON = true
	}
}
