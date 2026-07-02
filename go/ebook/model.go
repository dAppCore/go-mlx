// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import (
	core "dappco.re/go"
	"dappco.re/go/inference/modelmgmt"
)

// ModelBookOptions configures BuildModelBook. A transparent alias for
// modelmgmt.ModelBookOptions — pure data, field-for-field identical to this
// package's original definition.
type ModelBookOptions = modelmgmt.ModelBookOptions

// DIVERGENCE (kept local — see go/ebook symbol-disposition report):
// modelmgmt is an engine-agnostic library with no CLI of its own, so its
// colophon credits its own entry point ("modelmgmt.BuildModelBook"). This
// repo's actual ebook command is `lthn-mlx ebook` (see cmd/mlx/ebook.go) —
// the line a reader of the rendered book should see credited — so
// BuildModelBook below patches the colophon after delegating. Candidate
// upstream fix: a GeneratorCredit field on modelmgmt.ModelBookOptions, which
// would let this patch retire.
const (
	sharedGeneratorCredit = "<code>modelmgmt.BuildModelBook</code>"
	localGeneratorCredit  = "<code>lthn-mlx ebook</code>"
	colophonChapterID     = "ch999-colophon" // mirrors modelmgmt's internal ID; always the last chapter appended
)

// BuildModelBook reads a model directory and assembles it as an authored
// book: title + licence, the foreword (README — the human-speech anchor), the
// method section (architecture + inventory + this-book-in-numbers), and —
// when IncludeWeights is set — the weights as base64 plates plus a decode
// recipe so the book reconstructs into a runnable model. No model is loaded;
// this reads bytes and arranges them.
//
// Delegates to modelmgmt.BuildModelBook, adapting its core.Result back to
// this package's original (*Book, error) signature and re-crediting the
// colophon to the local CLI verb (see the DIVERGENCE note above).
func BuildModelBook(opts ModelBookOptions) (*Book, error) {
	r := modelmgmt.BuildModelBook(opts)
	if !r.OK {
		return nil, resultErr(r)
	}
	mb := r.Value.(*modelmgmt.Book)
	patchColophonCredit(mb)
	return (*Book)(mb), nil
}

// patchColophonCredit rewrites the colophon's generator credit from
// modelmgmt's library entry point to this repo's actual CLI command. Scoped
// to the one (small) colophon chapter, deliberately — a substring scan over
// every chapter would walk multi-gigabyte weight plates for no reason. See
// the DIVERGENCE note on BuildModelBook.
func patchColophonCredit(b *modelmgmt.Book) {
	if n := len(b.Chapters); n > 0 {
		last := &b.Chapters[n-1]
		if last.ID == colophonChapterID {
			last.Body = core.Replace(last.Body, sharedGeneratorCredit, localGeneratorCredit)
		}
	}
}
