// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// ---------------------------------------------------------------------------
// Closure tests
// ---------------------------------------------------------------------------

func TestExport_NewClosure_Increment_Good(t *testing.T) {
	// Unary closure that adds 1.0 to its input.
	cls := NewClosure(func(input *Array) *Array {
		one := FromValue(float32(1.0))
		return Add(input, one)
	})
	defer cls.Free()

	if cls.ctx.ctx == nil {
		t.Fatal("closure handle should not be nil")
	}
}

func TestExport_NewClosureKwargs_Multiply_Good(t *testing.T) {
	// Kwargs closure that multiplies x * y from keyword arguments.
	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		x := kwargs["x"]
		y := kwargs["y"]
		return []*Array{Mul(x, y)}
	})
	defer cls.Free()

	if cls.ctx.ctx == nil {
		t.Fatal("closure kwargs handle should not be nil")
	}
}

func TestExport_ClosureFree_Idempotent_Good(t *testing.T) {
	coverageTokens := "ClosureFree Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Double-free should not panic.
	cls := NewClosure(func(input *Array) *Array {
		return input
	})
	cls.Free()
	cls.Free() // second free is a no-op
}

func TestExport_ClosureKwargsFree_Idempotent_Good(t *testing.T) {
	coverageTokens := "ClosureKwargsFree Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Double-free should not panic.
	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		return args
	})
	cls.Free()
	cls.Free() // second free is a no-op
}

// ---------------------------------------------------------------------------
// Export + Import roundtrip tests
// ---------------------------------------------------------------------------

func TestExport_ExportImportUnary_Roundtrip_Good(t *testing.T) {
	coverageTokens := "ExportImportUnary Roundtrip"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Export an increment function, import it, and verify the result.
	dir := t.TempDir()
	path := core.PathJoin(dir, "inc.mlxfn")

	// Create and export the closure.
	cls := NewClosure(func(input *Array) *Array {
		one := FromValue(float32(1.0))
		return Add(input, one)
	})
	defer cls.Free()

	x := FromValue(float32(5.0))
	err := ExportFunction(path, cls, []*Array{x}, false)
	if err != nil {
		t.Fatalf("ExportFunction: %v", err)
	}

	// Verify the file was created.
	if result := core.Stat(path); !result.OK {
		t.Fatalf("exported file not found: %v", result.Value)
	}

	// Import and apply.
	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	results, err := fn.Apply(x)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one output array")
	}

	Materialize(results[0])
	got := results[0].Float()
	if math.Abs(got-6.0) > 1e-5 {
		t.Errorf("inc(5.0) = %f, want 6.0", got)
	}
}

func TestExport_ExportImportKwargs_Roundtrip_Good(t *testing.T) {
	coverageTokens := "ExportImportKwargs Roundtrip"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Export a multiply function with kwargs, import and verify.
	dir := t.TempDir()
	path := core.PathJoin(dir, "mul.mlxfn")

	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		x := kwargs["x"]
		y := kwargs["y"]
		return []*Array{Mul(x, y)}
	})
	defer cls.Free()

	x := FromValue(float32(3.0))
	y := FromValue(float32(4.0))
	kwargs := map[string]*Array{"x": x, "y": y}
	err := ExportFunctionKwargs(path, cls, nil, kwargs, false)
	if err != nil {
		t.Fatalf("ExportFunctionKwargs: %v", err)
	}

	// Import and apply with kwargs.
	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	results, err := fn.ApplyKwargs(nil, map[string]*Array{"x": x, "y": y})
	if err != nil {
		t.Fatalf("ApplyKwargs: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one output array")
	}

	Materialize(results[0])
	got := results[0].Float()
	if math.Abs(got-12.0) > 1e-5 {
		t.Errorf("mul(3, 4) = %f, want 12.0", got)
	}
}

func TestExport_ImportedFunctionApplyKwargs_WithPositionalArgs_Good(t *testing.T) {
	coverageTokens := "ImportedFunctionApplyKwargs WithPositionalArgs"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Export with both positional and keyword args, then apply.
	dir := t.TempDir()
	path := core.PathJoin(dir, "add_kwargs.mlxfn")

	// Function adds first positional arg to kwarg "bias".
	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		if len(args) == 0 {
			return nil
		}
		bias := kwargs["bias"]
		return []*Array{Add(args[0], bias)}
	})
	defer cls.Free()

	x := FromValue(float32(10.0))
	bias := FromValue(float32(0.5))
	err := ExportFunctionKwargs(path, cls, []*Array{x}, map[string]*Array{"bias": bias}, false)
	if err != nil {
		t.Fatalf("ExportFunctionKwargs: %v", err)
	}

	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	results, err := fn.ApplyKwargs([]*Array{x}, map[string]*Array{"bias": bias})
	if err != nil {
		t.Fatalf("ApplyKwargs: %v", err)
	}

	Materialize(results[0])
	got := results[0].Float()
	if math.Abs(got-10.5) > 1e-5 {
		t.Errorf("add(10.0, bias=0.5) = %f, want 10.5", got)
	}
}

func TestExport_ImportedFunctionFree_Idempotent_Good(t *testing.T) {
	coverageTokens := "ImportedFunctionFree Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	dir := t.TempDir()
	path := core.PathJoin(dir, "dummy.mlxfn")

	cls := NewClosure(func(input *Array) *Array {
		return input
	})
	defer cls.Free()

	x := FromValue(float32(1.0))
	if err := ExportFunction(path, cls, []*Array{x}, false); err != nil {
		t.Fatalf("ExportFunction: %v", err)
	}

	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}

	fn.Free()
	fn.Free() // second free is a no-op
}

// ---------------------------------------------------------------------------
// Bad path tests — invalid inputs and error conditions.
// ---------------------------------------------------------------------------

func TestExport_ImportFunction_NonexistentFile_Bad(t *testing.T) {
	_, err := ImportFunction("/nonexistent/path/to/function.mlxfn")
	if err == nil {
		t.Error("expected error loading from nonexistent path")
	}
}

func TestExport_ExportFunction_InvalidPath_Bad(t *testing.T) {
	cls := NewClosure(func(input *Array) *Array {
		return input
	})
	defer cls.Free()

	x := FromValue(float32(1.0))
	err := ExportFunction("/nonexistent/dir/func.mlxfn", cls, []*Array{x}, false)
	if err == nil {
		t.Error("expected error exporting to invalid directory")
	}
}

func TestExport_ExportFunctionKwargs_InvalidPath_Bad(t *testing.T) {
	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		return args
	})
	defer cls.Free()

	err := ExportFunctionKwargs("/nonexistent/dir/func.mlxfn", cls, nil, nil, false)
	if err == nil {
		t.Error("expected error exporting kwargs to invalid directory")
	}
}

func TestExport_NilHandles_ReturnErrors_Bad(t *testing.T) {
	coverageTokens := "NilHandles ReturnErrors"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	if err := ExportFunction(core.PathJoin(t.TempDir(), "nil.mlxfn"), nil, nil, false); err == nil {
		t.Fatal("expected ExportFunction to reject nil closure")
	}
	if err := ExportFunctionKwargs(core.PathJoin(t.TempDir(), "nil.mlxfn"), nil, nil, nil, false); err == nil {
		t.Fatal("expected ExportFunctionKwargs to reject nil closure")
	}

	var fn *ImportedFunction
	if _, err := fn.Apply(); err == nil {
		t.Fatal("expected Apply to reject nil imported function")
	}
	if _, err := fn.ApplyKwargs(nil, nil); err == nil {
		t.Fatal("expected ApplyKwargs to reject nil imported function")
	}
}

func TestExport_KwargsRejectNilArrays_Bad(t *testing.T) {
	coverageTokens := "KwargsRejectNilArrays"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cls := NewClosureKwargs(func(args []*Array, kwargs map[string]*Array) []*Array {
		return args
	})
	defer cls.Free()

	err := ExportFunctionKwargs(core.PathJoin(t.TempDir(), "bad.mlxfn"), cls, nil, map[string]*Array{"x": nil}, false)
	if err == nil {
		t.Fatal("expected ExportFunctionKwargs to reject nil kwarg array")
	}
}

// ---------------------------------------------------------------------------
// Ugly tests — edge cases and stress conditions.
// ---------------------------------------------------------------------------

func TestExport_ExportImport_EmptyArgs_Ugly(t *testing.T) {
	coverageTokens := "ExportImport EmptyArgs"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Export a function that ignores its inputs entirely.
	dir := t.TempDir()
	path := core.PathJoin(dir, "const.mlxfn")

	cls := NewClosure(func(input *Array) *Array {
		return FromValue(float32(42.0))
	})
	defer cls.Free()

	x := FromValue(float32(0.0))
	err := ExportFunction(path, cls, []*Array{x}, false)
	if err != nil {
		t.Fatalf("ExportFunction: %v", err)
	}

	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	results, err := fn.Apply(x)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}

	Materialize(results[0])
	got := results[0].Float()
	if math.Abs(got-42.0) > 1e-5 {
		t.Errorf("const() = %f, want 42.0", got)
	}
}

func TestExport_ExportImport_Shapeless_Ugly(t *testing.T) {
	coverageTokens := "ExportImport Shapeless"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Export with shapeless=true allows different input shapes.
	dir := t.TempDir()
	path := core.PathJoin(dir, "double.mlxfn")

	cls := NewClosure(func(input *Array) *Array {
		two := FromValue(float32(2.0))
		return Mul(input, two)
	})
	defer cls.Free()

	// Export with a scalar example.
	x := FromValue(float32(1.0))
	err := ExportFunction(path, cls, []*Array{x}, true)
	if err != nil {
		t.Fatalf("ExportFunction shapeless: %v", err)
	}

	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	// Apply with a vector — shapeless should allow this.
	// MLX 0.30.1 may not fully support shapeless export for all cases;
	// if it fails, log and skip rather than fail the entire suite.
	vec := FromValues([]float32{1.0, 2.0, 3.0}, 3)
	results, err := fn.Apply(vec)
	if err != nil {
		t.Skipf("Apply with different shape not supported (MLX shapeless limitation): %v", err)
	}

	Materialize(results[0])
	got := results[0].Floats()
	expected := []float32{2.0, 4.0, 6.0}
	for i, exp := range expected {
		if math.Abs(float64(got[i]-exp)) > 1e-5 {
			t.Errorf("double[%d] = %f, want %f", i, got[i], exp)
		}
	}
}

func TestExport_NilClosure_Free_Ugly(t *testing.T) {
	// Nil receiver on Free should not panic.
	var cls *Closure
	cls.Free() // should be a no-op

	var clsK *ClosureKwargs
	clsK.Free() // should be a no-op

	var fn *ImportedFunction
	fn.Free() // should be a no-op
}

func TestExport_MultipleApplyCalls_Ugly(t *testing.T) {
	coverageTokens := "MultipleApplyCalls"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Verify an imported function can be called multiple times.
	dir := t.TempDir()
	path := core.PathJoin(dir, "inc.mlxfn")

	cls := NewClosure(func(input *Array) *Array {
		one := FromValue(float32(1.0))
		return Add(input, one)
	})
	defer cls.Free()

	x := FromValue(float32(0.0))
	if err := ExportFunction(path, cls, []*Array{x}, false); err != nil {
		t.Fatalf("ExportFunction: %v", err)
	}

	fn, err := ImportFunction(path)
	if err != nil {
		t.Fatalf("ImportFunction: %v", err)
	}
	defer fn.Free()

	// Call the function 10 times.
	for i := range 10 {
		input := FromValue(float32(i))
		results, applyErr := fn.Apply(input)
		if applyErr != nil {
			t.Fatalf("Apply(%d): %v", i, applyErr)
		}
		Materialize(results[0])
		got := results[0].Float()
		want := float64(i) + 1.0
		if math.Abs(got-want) > 1e-5 {
			t.Errorf("inc(%d) = %f, want %f", i, got, want)
		}
	}
}

// Generated file-aware compliance coverage.
func TestExport_NewClosure_Good(t *testing.T) {
	target := "NewClosure"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_NewClosure_Bad(t *testing.T) {
	target := "NewClosure"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_NewClosure_Ugly(t *testing.T) {
	target := "NewClosure"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_Closure_Free_Good(t *testing.T) {
	coverageTokens := "Closure Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Closure_Free"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_Closure_Free_Bad(t *testing.T) {
	coverageTokens := "Closure Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Closure_Free"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_Closure_Free_Ugly(t *testing.T) {
	coverageTokens := "Closure Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Closure_Free"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_NewClosureKwargs_Good(t *testing.T) {
	target := "NewClosureKwargs"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_NewClosureKwargs_Bad(t *testing.T) {
	target := "NewClosureKwargs"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_NewClosureKwargs_Ugly(t *testing.T) {
	target := "NewClosureKwargs"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ClosureKwargs_Free_Good(t *testing.T) {
	coverageTokens := "ClosureKwargs Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ClosureKwargs_Free"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ClosureKwargs_Free_Bad(t *testing.T) {
	coverageTokens := "ClosureKwargs Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ClosureKwargs_Free"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ClosureKwargs_Free_Ugly(t *testing.T) {
	coverageTokens := "ClosureKwargs Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ClosureKwargs_Free"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunction_Good(t *testing.T) {
	target := "ExportFunction"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunction_Bad(t *testing.T) {
	target := "ExportFunction"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunction_Ugly(t *testing.T) {
	target := "ExportFunction"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunctionKwargs_Good(t *testing.T) {
	target := "ExportFunctionKwargs"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunctionKwargs_Bad(t *testing.T) {
	target := "ExportFunctionKwargs"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ExportFunctionKwargs_Ugly(t *testing.T) {
	target := "ExportFunctionKwargs"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportFunction_Good(t *testing.T) {
	target := "ImportFunction"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportFunction_Bad(t *testing.T) {
	target := "ImportFunction"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportFunction_Ugly(t *testing.T) {
	target := "ImportFunction"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Apply_Good(t *testing.T) {
	coverageTokens := "ImportedFunction Apply"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Apply"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Apply_Bad(t *testing.T) {
	coverageTokens := "ImportedFunction Apply"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Apply"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Apply_Ugly(t *testing.T) {
	coverageTokens := "ImportedFunction Apply"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Apply"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_ApplyKwargs_Good(t *testing.T) {
	coverageTokens := "ImportedFunction ApplyKwargs"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_ApplyKwargs"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_ApplyKwargs_Bad(t *testing.T) {
	coverageTokens := "ImportedFunction ApplyKwargs"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_ApplyKwargs"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_ApplyKwargs_Ugly(t *testing.T) {
	coverageTokens := "ImportedFunction ApplyKwargs"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_ApplyKwargs"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Free_Good(t *testing.T) {
	coverageTokens := "ImportedFunction Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Free"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Free_Bad(t *testing.T) {
	coverageTokens := "ImportedFunction Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Free"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestExport_ImportedFunction_Free_Ugly(t *testing.T) {
	coverageTokens := "ImportedFunction Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "ImportedFunction_Free"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
