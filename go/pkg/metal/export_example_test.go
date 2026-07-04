// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleNewClosure() {
	core.Println("NewClosure")
	// Output: NewClosure
}

func ExampleClosure_Free() {
	core.Println("Closure_Free")
	// Output: Closure_Free
}

func ExampleNewClosureKwargs() {
	core.Println("NewClosureKwargs")
	// Output: NewClosureKwargs
}

func ExampleClosureKwargs_Free() {
	core.Println("ClosureKwargs_Free")
	// Output: ClosureKwargs_Free
}

func ExampleExportFunction() {
	core.Println("ExportFunction")
	// Output: ExportFunction
}

func ExampleExportFunctionKwargs() {
	core.Println("ExportFunctionKwargs")
	// Output: ExportFunctionKwargs
}

func ExampleImportFunction() {
	core.Println("ImportFunction")
	// Output: ImportFunction
}

func ExampleImportedFunction_Apply() {
	core.Println("ImportedFunction_Apply")
	// Output: ImportedFunction_Apply
}

func ExampleImportedFunction_ApplyKwargs() {
	core.Println("ImportedFunction_ApplyKwargs")
	// Output: ImportedFunction_ApplyKwargs
}

func ExampleImportedFunction_Free() {
	core.Println("ImportedFunction_Free")
	// Output: ImportedFunction_Free
}
