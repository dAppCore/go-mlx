// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Weight-quant loader registry — quant kind → loader, mirroring
// model_registry.go. A format (affine, q4_0, mxfp4, nvfp4, …) self-registers
// from its own package init(); the Linear-loading path looks it up by the kind
// the model config declares, with no central switch and no model edit. A
// consumer (a model loader, a mixer builder) only ever receives the finished
// *Linear, so model architecture is independent of quant format.
type QuantLoader func(t QuantTensors) (*Linear, error)

// QuantTensors is the raw quantized-weight set a loader assembles into a Linear:
// the packed weight, the per-group scales/biases, an optional bias, and the
// group/bit geometry. A loader reads only what its format needs.
type QuantTensors struct {
	Weight, Scales, Biases, Bias *Array
	GroupSize, Bits              int
}

var quantLoaders = map[string]QuantLoader{}

// RegisterQuantLoader registers fn for a quant kind; a later call overrides.
// Call from init() so a format self-registers. No-op on empty kind or nil fn.
func RegisterQuantLoader(kind string, fn QuantLoader) {
	if kind == "" || fn == nil {
		return
	}
	quantLoaders[kind] = fn
}

// lookupQuantLoader returns the loader for kind, or nil if none is registered.
func lookupQuantLoader(kind string) QuantLoader { return quantLoaders[kind] }
