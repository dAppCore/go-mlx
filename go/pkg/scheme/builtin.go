// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import shared "dappco.re/go/inference/scheme"

// BFloat16, Float32 alias the shared catalogue's activation/compute dtypes —
// see dappco.re/go/inference/scheme.BFloat16 / .Float32. Exported (unlike the
// rest of the builtin seed) because they are compile-time foundational to
// every backend's elementwise ops, like the StateKind constants.
//
// The builtin catalogue itself — the softmax-hybrid mixer, the KV-cache
// modes, the affine quant — is seeded exactly once, by the shared package's
// own init(), which runs automatically because scheme.go imports it. This
// file used to carry a second, local init() that re-registered the identical
// seed under the same names; dappco.re/go's Registry.Set() overwrites an
// existing key rather than erroring or duplicating it, so that double
// registration was harmless, but it was also dead weight duplicating the
// shared source of truth — removed rather than kept as a belt-and-braces
// copy. See TestBuiltinSeedNotDuplicated_Good.
var (
	BFloat16 DType = shared.BFloat16
	Float32  DType = shared.Float32
)
