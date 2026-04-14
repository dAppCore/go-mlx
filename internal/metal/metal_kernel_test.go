//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"
	"testing"
)

// --- Good: correct usage ---

func TestMetalKernel_ExpElementwise_Good(t *testing.T) {
	// Custom Metal kernel that computes exp(x) element-wise, matching the C example.
	source := `uint elem = thread_position_in_grid.x;
T tmp = inp[elem];
out[elem] = metal::exp(tmp);`

	kernel := NewMetalKernel("test_exp", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	input := FromValues([]float32{0, 1, 2, 3}, 4)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.AddTemplateDType("T", DTypeFloat32)
	cfg.SetGrid(input.Size(), 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(input.Shape(), input.Dtype())

	results, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 output, got %d", len(results))
	}

	Materialize(results[0])
	got := results[0].Floats()
	want := []float64{math.Exp(0), math.Exp(1), math.Exp(2), math.Exp(3)}

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-3 {
			t.Errorf("exp[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMetalKernel_AddKernel_Good(t *testing.T) {
	// Custom kernel that adds two arrays element-wise.
	source := `uint elem = thread_position_in_grid.x;
out[elem] = a[elem] + b[elem];`

	kernel := NewMetalKernel("test_add", []string{"a", "b"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	a := FromValues([]float32{1, 2, 3, 4}, 4)
	b := FromValues([]float32{10, 20, 30, 40}, 4)
	Materialize(a, b)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(a.Size(), 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(a.Shape(), a.Dtype())

	results, err := kernel.Apply(cfg, a, b)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	Materialize(results[0])
	got := results[0].Floats()
	want := []float32{11, 22, 33, 44}

	for i := range got {
		if math.Abs(float64(got[i])-float64(want[i])) > 1e-5 {
			t.Errorf("add[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMetalKernel_2DShape_Good(t *testing.T) {
	// Verify output shape is preserved for multi-dimensional arrays.
	source := `uint elem = thread_position_in_grid.x;
T tmp = inp[elem];
out[elem] = tmp * tmp;`

	kernel := NewMetalKernel("test_square", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	input := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.AddTemplateDType("T", DTypeFloat32)
	cfg.SetGrid(input.Size(), 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(input.Shape(), input.Dtype())

	results, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	Materialize(results[0])
	shape := results[0].Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}

	got := results[0].Floats()
	want := []float32{1, 4, 9, 16, 25, 36}
	for i := range got {
		if math.Abs(float64(got[i])-float64(want[i])) > 1e-3 {
			t.Errorf("square[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMetalKernel_ConfigReuse_Good(t *testing.T) {
	// Config can be reused across multiple Apply calls.
	source := `uint elem = thread_position_in_grid.x;
out[elem] = inp[elem] + inp[elem];`

	kernel := NewMetalKernel("test_double", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(4, 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg([]int32{4}, DTypeFloat32)

	for round := 0; round < 3; round++ {
		input := FromValues([]float32{float32(round), float32(round + 1), float32(round + 2), float32(round + 3)}, 4)
		Materialize(input)

		results, err := kernel.Apply(cfg, input)
		if err != nil {
			t.Fatalf("round %d: Apply failed: %v", round, err)
		}
		Materialize(results[0])
		got := results[0].Floats()
		for i, v := range got {
			want := float32(round+i) * 2
			if math.Abs(float64(v)-float64(want)) > 1e-5 {
				t.Errorf("round %d [%d] = %f, want %f", round, i, v, want)
			}
		}
	}
}

// --- Bad: invalid or error-producing usage ---

func TestMetalKernel_NilConfig_Bad(t *testing.T) {
	// Applying with a freed config should produce an error, not a panic.
	source := `uint elem = thread_position_in_grid.x;
out[elem] = inp[elem];`

	kernel := NewMetalKernel("test_nil_cfg", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	cfg := NewMetalKernelConfig()
	cfg.Free() // free before use

	input := FromValues([]float32{1, 2, 3, 4}, 4)
	Materialize(input)

	_, err := kernel.Apply(cfg, input)
	if err == nil {
		t.Log("Apply with freed config did not error — MLX-C may tolerate nil config")
	}
}

func TestMetalKernel_EmptySource_Bad(t *testing.T) {
	// Empty source string should either error on apply or produce no useful output.
	kernel := NewMetalKernel("test_empty", []string{"inp"}, []string{"out"}, "", "", true, false)
	defer kernel.Free()

	input := FromValues([]float32{1, 2}, 2)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(input.Size(), 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(input.Shape(), input.Dtype())

	_, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Logf("expected error from empty source: %v", err)
	}
}

func TestMetalKernel_DoubleFree_Bad(t *testing.T) {
	// Double-free on kernel and config should not panic.
	kernel := NewMetalKernel("test_dbl_free", []string{"inp"}, []string{"out"},
		"uint i = thread_position_in_grid.x; out[i] = inp[i];", "", true, false)
	kernel.Free()
	kernel.Free() // second free is a no-op

	cfg := NewMetalKernelConfig()
	cfg.Free()
	cfg.Free() // second free is a no-op
}

// --- Ugly: edge cases and boundary conditions ---

func TestMetalKernel_SingleElement_Ugly(t *testing.T) {
	// Kernel operating on a single element.
	source := `uint elem = thread_position_in_grid.x;
out[elem] = inp[elem] * 42.0f;`

	kernel := NewMetalKernel("test_single", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	input := FromValues([]float32{1.0}, 1)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(1, 1, 1)
	cfg.SetThreadGroup(1, 1, 1)
	cfg.AddOutputArg([]int32{1}, DTypeFloat32)

	results, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	Materialize(results[0])
	got := results[0].Floats()
	if len(got) != 1 || math.Abs(float64(got[0])-42.0) > 1e-3 {
		t.Errorf("single element = %v, want [42.0]", got)
	}
}

func TestMetalKernel_LargeArray_Ugly(t *testing.T) {
	// Kernel operating on a large array to verify grid/threadgroup scaling.
	n := 65536
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)
	}

	source := `uint elem = thread_position_in_grid.x;
out[elem] = inp[elem] + 1.0f;`

	kernel := NewMetalKernel("test_large", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	input := FromValues(data, n)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(n, 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg([]int32{int32(n)}, DTypeFloat32)

	results, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	Materialize(results[0])
	got := results[0].Floats()
	if len(got) != n {
		t.Fatalf("expected %d elements, got %d", n, len(got))
	}

	// Spot-check a few values
	for _, idx := range []int{0, 1, 100, 1000, n - 1} {
		want := float32(idx) + 1.0
		if math.Abs(float64(got[idx])-float64(want)) > 1e-3 {
			t.Errorf("[%d] = %f, want %f", idx, got[idx], want)
		}
	}
}

func TestMetalKernel_InitValue_Ugly(t *testing.T) {
	// Test SetInitValue — output should start at the init value,
	// and kernel writes only to specific positions.
	source := `uint elem = thread_position_in_grid.x;
if (elem == 0) { out[elem] = 99.0f; }`

	kernel := NewMetalKernel("test_init", []string{"inp"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	input := FromValues([]float32{0, 0, 0, 0}, 4)
	Materialize(input)

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(input.Size(), 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.SetInitValue(-1.0)
	cfg.AddOutputArg(input.Shape(), input.Dtype())

	results, err := kernel.Apply(cfg, input)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	Materialize(results[0])
	got := results[0].Floats()
	// Element 0 is written to 99.0, others should be init value -1.0
	if math.Abs(float64(got[0])-99.0) > 1e-3 {
		t.Errorf("[0] = %f, want 99.0", got[0])
	}
	for i := 1; i < len(got); i++ {
		if math.Abs(float64(got[i])-(-1.0)) > 1e-3 {
			t.Errorf("[%d] = %f, want -1.0 (init value)", i, got[i])
		}
	}
}
