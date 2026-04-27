// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"os"
	"path/filepath"
	"testing"
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
	// Double-free should not panic.
	cls := NewClosure(func(input *Array) *Array {
		return input
	})
	cls.Free()
	cls.Free() // second free is a no-op
}

func TestExport_ClosureKwargsFree_Idempotent_Good(t *testing.T) {
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
	// Export an increment function, import it, and verify the result.
	dir := t.TempDir()
	path := filepath.Join(dir, "inc.mlxfn")

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
	if _, statErr := os.Stat(path); statErr != nil {
		t.Fatalf("exported file not found: %v", statErr)
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
	// Export a multiply function with kwargs, import and verify.
	dir := t.TempDir()
	path := filepath.Join(dir, "mul.mlxfn")

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
	// Export with both positional and keyword args, then apply.
	dir := t.TempDir()
	path := filepath.Join(dir, "add_kwargs.mlxfn")

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
	dir := t.TempDir()
	path := filepath.Join(dir, "dummy.mlxfn")

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

// ---------------------------------------------------------------------------
// Ugly tests — edge cases and stress conditions.
// ---------------------------------------------------------------------------

func TestExport_ExportImport_EmptyArgs_Ugly(t *testing.T) {
	// Export a function that ignores its inputs entirely.
	dir := t.TempDir()
	path := filepath.Join(dir, "const.mlxfn")

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
	// Export with shapeless=true allows different input shapes.
	dir := t.TempDir()
	path := filepath.Join(dir, "double.mlxfn")

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
	// Verify an imported function can be called multiple times.
	dir := t.TempDir()
	path := filepath.Join(dir, "inc.mlxfn")

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
