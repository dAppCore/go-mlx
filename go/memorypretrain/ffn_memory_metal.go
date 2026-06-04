// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package memorypretrain

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

var _ metal.FFNMemoryAugmenter = (*MetalFFNMemoryAugmenter)(nil)

// MetalFFNMemoryAugmenter adapts an offline FFN memory bank to the neutral
// metal.FFNMemoryAugmenter hook used by model decoder layers.
type MetalFFNMemoryAugmenter struct {
	Memory     *FFNMemoryBank
	ClusterIDs []int
}

// NewMetalFFNMemoryAugmenter creates a model-facing FFN memory hook. Passing no
// cluster IDs selects the upstream generic-memory fallback slot.
func NewMetalFFNMemoryAugmenter(memory *FFNMemoryBank, clusterIDs []int) (*MetalFFNMemoryAugmenter, error) {
	if memory == nil {
		return nil, core.NewError("memorypretrain: FFN memory bank is nil")
	}
	augmenter := &MetalFFNMemoryAugmenter{Memory: memory}
	if err := augmenter.SetClusterIDs(clusterIDs); err != nil {
		return nil, err
	}
	return augmenter, nil
}

// SetClusterIDs updates the selected memory route for subsequent decoder-layer
// calls. Passing no IDs restores the generic-memory fallback.
func (augmenter *MetalFFNMemoryAugmenter) SetClusterIDs(clusterIDs []int) error {
	if augmenter == nil {
		return core.NewError("memorypretrain: metal FFN memory augmenter is nil")
	}
	if augmenter.Memory == nil {
		return core.NewError("memorypretrain: FFN memory bank is nil")
	}
	if len(clusterIDs) == 0 {
		ids, err := augmenter.Memory.GenericClusterIDs()
		if err != nil {
			return err
		}
		augmenter.ClusterIDs = ids
		return nil
	}
	ids, err := padClusterIDsWithGenericFallback(clusterIDs, augmenter.Memory.ClusterCounts())
	if err != nil {
		return err
	}
	augmenter.ClusterIDs = append(augmenter.ClusterIDs[:0], ids...)
	return nil
}

// AugmentFFNMemory implements metal.FFNMemoryAugmenter.
func (augmenter *MetalFFNMemoryAugmenter) AugmentFFNMemory(layerID int32, ffnOutput, mlpInput *metal.Array) (*metal.Array, bool, error) {
	if augmenter == nil {
		return nil, false, core.NewError("memorypretrain: metal FFN memory augmenter is nil")
	}
	if augmenter.Memory == nil {
		return nil, false, core.NewError("memorypretrain: FFN memory bank is nil")
	}
	if ffnOutput == nil || !ffnOutput.Valid() {
		return nil, false, core.NewError("memorypretrain: FFN output array is invalid")
	}
	if mlpInput == nil || !mlpInput.Valid() {
		return nil, false, core.NewError("memorypretrain: MLP input array is invalid")
	}
	if ffnOutput.Size() != mlpInput.Size() {
		return nil, false, core.Errorf("memorypretrain: FFN output size %d does not match MLP input size %d", ffnOutput.Size(), mlpInput.Size())
	}
	hiddenSize := augmenter.Memory.HiddenSize
	if hiddenSize <= 0 {
		return nil, false, core.NewError("memorypretrain: FFN memory hidden size must be positive")
	}
	if ffnOutput.Size()%hiddenSize != 0 {
		return nil, false, core.Errorf("memorypretrain: FFN output size %d is not divisible by hidden size %d", ffnOutput.Size(), hiddenSize)
	}
	clusterIDs := augmenter.ClusterIDs
	if len(clusterIDs) == 0 {
		ids, err := augmenter.Memory.GenericClusterIDs()
		if err != nil {
			return nil, false, err
		}
		clusterIDs = ids
	} else {
		ids, err := padClusterIDsWithGenericFallback(clusterIDs, augmenter.Memory.ClusterCounts())
		if err != nil {
			return nil, false, err
		}
		clusterIDs = ids
	}
	ffnValues := ffnOutput.Floats()
	mlpValues := mlpInput.Floats()
	outValues := make([]float32, len(ffnValues))
	for start := 0; start < len(ffnValues); start += hiddenSize {
		end := start + hiddenSize
		if _, _, err := augmenter.Memory.AddToFFNOutput(outValues[start:end], ffnValues[start:end], mlpValues[start:end], int(layerID), clusterIDs); err != nil {
			return nil, false, err
		}
	}
	return metal.FromValues(outValues, metalShapeAsInts(ffnOutput)...), true, nil
}

func metalShapeAsInts(arr *metal.Array) []int {
	var scratch [metal.MaxTensorRank]int32
	shape := arr.ShapeInto(scratch[:0])
	out := make([]int, len(shape))
	for i, dim := range shape {
		out[i] = int(dim)
	}
	return out
}
