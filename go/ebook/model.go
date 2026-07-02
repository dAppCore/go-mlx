// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import "dappco.re/go/inference/modelmgmt"

// ModelBookOptions configures BuildModelBook. A transparent alias for
// modelmgmt.ModelBookOptions — pure data, field-for-field identical to this
// package's original definition.
type ModelBookOptions = modelmgmt.ModelBookOptions

// localGeneratorCredit names this repo's actual CLI command (see
// cmd/mlx/ebook.go) — the line a reader of the rendered book should see
// credited, rather than modelmgmt's own library entry point
// ("modelmgmt.BuildModelBook").
const localGeneratorCredit = "lthn-mlx ebook"

// BuildModelBook reads a model directory and assembles it as an authored
// book: title + licence, the foreword (README — the human-speech anchor), the
// method section (architecture + inventory + this-book-in-numbers), and —
// when IncludeWeights is set — the weights as base64 plates plus a decode
// recipe so the book reconstructs into a runnable model. No model is loaded;
// this reads bytes and arranges them.
//
// Delegates to modelmgmt.BuildModelBook, adapting its core.Result back to
// this package's original (*Book, error) signature. GeneratorCredit is
// always set to this repo's CLI verb so the rendered colophon credits the
// command a reader actually ran.
func BuildModelBook(opts ModelBookOptions) (*Book, error) {
	opts.GeneratorCredit = localGeneratorCredit
	r := modelmgmt.BuildModelBook(opts)
	if !r.OK {
		return nil, resultErr(r)
	}
	mb := r.Value.(*modelmgmt.Book)
	return (*Book)(mb), nil
}
