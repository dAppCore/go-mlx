// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for adapter.go — kept separate from adapter_test.go so
// the godoc-attached usage snippets stay readable. Only symbols with a
// deterministic result carry an // Output: line; InspectAdapter and Inspect
// emit content-derived hashes and caller-supplied paths, so they have no
// stable output to assert and are documented by their doc-comment usage
// snippets instead.

package lora

import "fmt"

// ExampleAdapterInfo_IsEmpty shows the zero-value adapter identity reporting
// itself empty — the state callers check before treating an adapter as
// attached.
func ExampleAdapterInfo_IsEmpty() {
	var info AdapterInfo
	fmt.Println(info.IsEmpty())

	info.Name = "my-lora"
	fmt.Println(info.IsEmpty())

	// Output:
	// true
	// false
}
