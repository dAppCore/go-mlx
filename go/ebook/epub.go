// SPDX-Licence-Identifier: EUPL-1.2

// Package ebook renders an authored work as EPUB3 — the open ebook standard
// (#100). Its reason to exist is the PGP playbook: Zimmermann printed PGP's
// source as a book because software was an exportable "munition" but a book
// is protected speech (Bernstein v. United States, Junger v. Daley settled
// that code is speech). A model rendered as an authored, published book
// carries the protection every published work carries — only a court, against
// the presumption and the burden, can strip it. This package turns a model
// into that book: the authored foreword (the human-speech anchor), the method
// section, and the weights as base64 "plates" that decode back into a runnable
// model — speech that compiles. EUPL-1.2 on the cover.
//
// Pure Go, no Metal, no model load — a file transformation, fully testable.
//
// The rendering itself (Chapter, Book, WriteEPUB) is the engine-agnostic
// slice that was ported up to dappco.re/go/inference/modelmgmt, so every
// engine backend can produce a protected-speech book from its own weights.
// This package is now a thin wrapper over that shared implementation: Book
// shares modelmgmt.Book's field layout so the two convert for free, and
// WriteEPUB adapts modelmgmt's core.Result return back to this package's
// original plain-error signature, so existing callers (cmd/mlx's ebook verb)
// compile unchanged.
package ebook

import (
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/modelmgmt"
)

// Chapter is one section of the book. Every chapter is in the reading spine;
// InNav controls whether it also appears in the table of contents (so a book
// with 900 weight "plates" keeps a clean ToC while still being fully readable
// front to back). A transparent alias for modelmgmt.Chapter — pure data, so
// there is no behaviour here to adapt.
type Chapter = modelmgmt.Chapter

// Book is an authored work ready to render as a valid EPUB3 container. Its
// field layout matches modelmgmt.Book exactly (Chapters uses the aliased
// Chapter type above), so the two convert for free; WriteEPUB is redeclared
// below with a plain error return rather than modelmgmt's core.Result, to
// preserve this package's original signature for existing callers.
type Book modelmgmt.Book

// WriteEPUB streams a valid EPUB3 container to w. Delegates the render to
// modelmgmt.Book.WriteEPUB and adapts its core.Result back to the plain error
// this package's callers (e.g. cmd/mlx's ebook verb) expect.
func (b *Book) WriteEPUB(w io.Writer) error {
	if r := (*modelmgmt.Book)(b).WriteEPUB(w); !r.OK {
		return resultErr(r)
	}
	return nil
}

// resultErr extracts the error carried by a failed core.Result. Shared by
// the WriteEPUB and BuildModelBook adapters in this package — modelmgmt's
// Fail() always carries an error in Value, but the string fallback keeps
// this safe even if that convention ever loosens.
func resultErr(r core.Result) error {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return core.NewError(r.Error())
}
