// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// FFNMemoryAugmenter is the model-neutral hook for hierarchical memory
// pretraining. Implementations add retrieved memory into a feed-forward output
// using the MLP input that produced it.
type FFNMemoryAugmenter interface {
	AugmentFFNMemory(layerID int32, ffnOutput, mlpInput *Array) (*Array, bool, error)
}

// ApplyFFNMemoryAugmenter runs a feed-forward memory hook and normalises the
// no-op cases so model packages can compose the feature without knowing the
// concrete memory-bank implementation.
func ApplyFFNMemoryAugmenter(augmenter FFNMemoryAugmenter, layerID int32, ffnOutput, mlpInput *Array) (*Array, bool, error) {
	if augmenter == nil {
		return ffnOutput, false, nil
	}
	if ffnOutput == nil || !ffnOutput.Valid() {
		return nil, false, core.NewError("mlx: FFN memory output is invalid")
	}
	if mlpInput == nil || !mlpInput.Valid() {
		return nil, false, core.NewError("mlx: FFN memory input is invalid")
	}
	out, applied, err := augmenter.AugmentFFNMemory(layerID, ffnOutput, mlpInput)
	if err != nil {
		return nil, false, err
	}
	if !applied {
		return ffnOutput, false, nil
	}
	if out == nil || !out.Valid() {
		return nil, false, core.NewError("mlx: FFN memory augmenter returned invalid output")
	}
	return out, true, nil
}
