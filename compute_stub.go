//go:build !(darwin && arm64) || nomlx

package mlx

import "errors"

var defaultComputeBackend Compute = unavailableCompute{}

// DefaultCompute returns the package's default stub compute backend.
func DefaultCompute() Compute { return defaultComputeBackend }

// NewSession returns an availability error on unsupported builds.
func NewSession(opts ...SessionOption) (Session, error) {
	return defaultComputeBackend.NewSession(opts...)
}

type unavailableCompute struct{}

func (unavailableCompute) Available() bool        { return false }
func (unavailableCompute) DeviceInfo() DeviceInfo { return DeviceInfo{} }
func (unavailableCompute) NewSession(...SessionOption) (Session, error) {
	return nil, errors.New("mlx: Metal compute is unavailable in this build")
}
