// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
)

// Example_parseRuntimeCacheMode shows how the --cache-mode flag value is
// turned into a typed KV cache mode. A non-empty value reports present=true
// so serve knows to override the engine default; the value is trimmed but
// not validated here (isRuntimeCacheMode does that).
//
//	lthn-mlx serve -model ... -cache-mode q8
func Example_parseRuntimeCacheMode() {
	mode, present := parseRuntimeCacheMode("  q8 ")
	core.Println(string(mode), present)

	_, blank := parseRuntimeCacheMode("")
	core.Println(blank)
	// Output:
	// q8 true
	// false
}

// Example_isRuntimeCacheMode shows the validation gate the parsed mode
// passes through before serve applies it: known modes engage, the default
// (empty) and any typo are rejected so a bad --cache-mode never silently
// changes the cache layout.
func Example_isRuntimeCacheMode() {
	mode, _ := parseRuntimeCacheMode("paged")
	core.Println(isRuntimeCacheMode(mode))

	typo, _ := parseRuntimeCacheMode("pagd")
	core.Println(isRuntimeCacheMode(typo))
	// Output:
	// true
	// false
}
