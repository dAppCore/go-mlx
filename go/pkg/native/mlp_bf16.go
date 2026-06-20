// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// bfloat16 MLP pieces: the bf16 elementwise ops and the composed bf16 GELU gate
// that a bf16 feed-forward block needs. Like the other bf16 ops these take/return
// raw bf16 []byte. GELU is composed (no metallib kernel) — in bf16 the question
// is whether the separately-run, each-step-rounded composition matches mlx's
// mlx_compile-fused gelu; the parity test answers it with data.

// f32ToBF16 converts a float32 to bfloat16 bits with round-to-nearest-even —
// matching mlx's AsType(..., bfloat16) so a constant produced here equals the
// same constant mlx would broadcast.
func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff
	return uint16((bits + rounding) >> 16)
}

// bf16ConstBytes returns n copies of v as bf16 bytes — a broadcast scalar operand
// for the contiguous bf16 binary kernels.
// bf16ConstKey identifies a materialised bf16 broadcast-scalar operand by length
// and value, so identical (n, v) requests share one immutable backing buffer.
type bf16ConstKey struct {
	n int
	v float32
}

// bf16ConstCache memoises the dense bf16 scalar operands bf16ConstBytes builds.
// The composed GeluBF16 fires the same four compile-time constants at a fixed
// decode width every call; caching collapses the per-call make([]byte, n*2) to a
// one-time fill. Entries are read-only kernel operands, so the shared buffer is
// byte-identical to a freshly filled one.
var (
	bf16ConstMu    sync.Mutex
	bf16ConstCache = map[bf16ConstKey][]byte{}
)

func bf16ConstBytes(n int, v float32) []byte {
	if n == 0 {
		return nil
	}
	key := bf16ConstKey{n: n, v: v}
	bf16ConstMu.Lock()
	defer bf16ConstMu.Unlock()
	if s, ok := bf16ConstCache[key]; ok {
		return s
	}
	h := f32ToBF16(v)
	out := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	bf16ConstCache[key] = out
	return out
}

// runBinaryBF16 drives a contiguous bf16 binary kernel (vv_<Op>bfloat16) over two
// equal-length bf16 byte buffers. name e.g. "vv_Multiplybfloat16".
func runBinaryBF16(name string, a, b []byte) ([]byte, error) {
	out := make([]byte, len(a))
	if err := runBinaryBF16Into(name, a, b, out); err != nil {
		return nil, err
	}
	return out, nil
}

// runBinaryBF16Into is runBinaryBF16 writing into the caller-supplied out
// (len(out) must equal len(a)) instead of allocating a fresh buffer, so a
// composed bf16 op (GeluBF16) can ping-pong reusable scratch across its chain.
// Same kernel and inputs as runBinaryBF16 — the bytes written are identical.
func runBinaryBF16Into(name string, a, b, out []byte) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(a) != len(b) {
		return core.NewError("native.runBinaryBF16Into: a and b must be the same length")
	}
	if len(a)%bf16Size != 0 {
		return core.NewError("native.runBinaryBF16Into: byte length must be a multiple of 2")
	}
	if len(out) != len(a) {
		return core.NewError("native.runBinaryBF16Into: out must be the same length as a")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	n := len(a) / bf16Size
	if n == 0 {
		return nil
	}
	withAutoreleasePool(func() {
		aBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&a[0]), uint(len(a)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(a)), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
		setEncInt32(enc, int32(n), 3)
		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(a)))
	})
	return nil
}

// MulBF16 is the bf16 sibling of Mul: element-wise a[i]*b[i] over bf16 bytes
// (kernel vv_Multiplybfloat16) — the MLP gate·up step in the decode dtype.
func MulBF16(a, b []byte) ([]byte, error) { return runBinaryBF16("vv_Multiplybfloat16", a, b) }

// TanhBF16 is the bf16 sibling of Tanh: element-wise tanh over bf16 bytes (kernel
// v_Tanhbfloat16bfloat16) — the nonlinearity inside the gelu approximation.
func TanhBF16(x []byte) ([]byte, error) {
	out := make([]byte, len(x))
	if err := tanhBF16Into(x, out); err != nil {
		return nil, err
	}
	return out, nil
}

// tanhBF16Into is TanhBF16 writing into the caller-supplied out (len(out) must
// equal len(x)) instead of allocating, so GeluBF16 can keep the tanh step on its
// ping-pong scratch. Same kernel and input as TanhBF16 — bytes are identical.
func tanhBF16Into(x, out []byte) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(x)%bf16Size != 0 {
		return core.NewError("native.tanhBF16Into: byte length must be a multiple of 2")
	}
	if len(out) != len(x) {
		return core.NewError("native.tanhBF16Into: out must be the same length as x")
	}
	pso, err := pipelineFor("v_Tanhbfloat16bfloat16")
	if err != nil {
		return err
	}
	n := len(x) / bf16Size
	if n == 0 {
		return nil
	}
	withAutoreleasePool(func() {
		inBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(x)), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(inBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		cnt := uint32(n)
		enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&cnt)), 4), 4, 2)
		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(x)))
	})
	return nil
}

// GeluBF16 is the bf16 sibling of Gelu: the tanh-approximation GELU composed from
// the bf16 primitives, each intermediate rounded to bf16 (the gelu_approx graph
// MLX would run un-fused in bf16). Input/output raw bf16 bytes.
func GeluBF16(x []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	// Validate byte alignment before the empty short-circuit, matching the
	// per-primitive path (an odd length reached MulBF16's check in the old
	// composition, so an odd 1-byte input must still error rather than return).
	if len(x)%bf16Size != 0 {
		return nil, core.NewError("native.GeluBF16: byte length must be a multiple of 2")
	}
	n := len(x) / bf16Size
	out := make([]byte, len(x))
	if n == 0 {
		return out, nil
	}
	// Two reusable bf16 scratch buffers ping-pong the chain exactly as the
	// float32 Gelu does: each step reads the previous output (in the other
	// buffer) plus x or a cached const, so two buffers carry the whole graph and
	// onePlus/halfX land in the two different buffers for the final multiply.
	// Reusing scratch instead of a fresh buffer per primitive removes the
	// dominant B/op; kernels and inputs are unchanged, so output is identical.
	a, b := make([]byte, len(x)), make([]byte, len(x))
	const (
		mul = "vv_Multiplybfloat16"
		add = "vv_Addbfloat16"
	)
	c044 := bf16ConstBytes(n, 0.044715)
	c079 := bf16ConstBytes(n, 0.7978845608028654)
	c1 := bf16ConstBytes(n, 1.0)
	c05 := bf16ConstBytes(n, 0.5)
	// x2=x·x→a; x3=a·x→b; x3s=b·c044→a; inner=x+a→b; scaled=b·c079→a;
	// t=tanh(a)→b; onePlus=b+c1→a; halfX=x·c05→b; gelu=b·a→out
	for _, step := range []struct {
		name    string
		x, y, z []byte
	}{
		{mul, x, x, a},
		{mul, a, x, b},
		{mul, b, c044, a},
		{add, x, a, b},
		{mul, b, c079, a},
	} {
		if err := runBinaryBF16Into(step.name, step.x, step.y, step.z); err != nil {
			return nil, err
		}
	}
	if err := tanhBF16Into(a, b); err != nil { // t = tanh(scaled)
		return nil, err
	}
	if err := runBinaryBF16Into(add, b, c1, a); err != nil { // onePlus = t + 1
		return nil, err
	}
	if err := runBinaryBF16Into(mul, x, c05, b); err != nil { // halfX = 0.5·x
		return nil, err
	}
	if err := runBinaryBF16Into(mul, b, a, out); err != nil { // gelu = halfX·onePlus
		return nil, err
	}
	return out, nil
}

// GeluGateMulBF16 computes gelu(gate)·up in bf16 — gemma's MLP gate in the decode
// dtype. Uses the fused kernel (fp32-internal, one dispatch) when the custom kernels
// metallib is loaded, else the composed bf16 primitive chain. Parity in parity_test.go.
func GeluGateMulBF16(gate, up []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(up) != len(gate) {
		return nil, core.NewError("native.GeluGateMulBF16: gate/up length mismatch")
	}
	if gpuHasGeluKernel() {
		return geluGateMulFused(gate, up, len(gate)/bf16Size)
	}
	g, err := GeluBF16(gate)
	if err != nil {
		return nil, err
	}
	return MulBF16(g, up)
}
