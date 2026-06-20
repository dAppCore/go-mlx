// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleBackendQuant shows the (backend,kind) cross-section: a backend registers its
// quant compute under its own name, and the engine resolves a weight's matvec by the
// backend it loaded + the kind the model declares. Two backends can register the SAME
// kind ("affine") without colliding — the fact that distinguishes this registry from
// pkg/scheme's kind-only one. (fakeQuant stands in for a real impl, which lives in
// pkg/native / pkg/metal; no Output: directive, matching the repo's example convention.)
func ExampleBackendQuant() {
	// a backend's init() does this once — here two backends, same kind.
	RegisterBackendQuant("native", fakeQuant{kind: "affine", bits: 4, tag: 0x1})
	RegisterBackendQuant("metal", fakeQuant{kind: "affine", bits: 4, tag: 0x2})

	// the engine resolves by (loaded backend, declared kind):
	q, ok := BackendQuant("native", "affine")
	core.Println(ok) // true — native registered "affine"
	if ok {
		core.Println(q.Bits()) // 4 — the nominal width of the resolved impl
	}

	// an unregistered (backend,kind) is detectable, not a panic:
	_, missing := BackendQuant("rocm", "affine")
	core.Println(missing) // false — no rocm backend has registered yet
}
