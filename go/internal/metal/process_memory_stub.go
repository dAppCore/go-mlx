// SPDX-Licence-Identifier: EUPL-1.2

//go:build !darwin || !arm64

package metal

// ProcessMemory reports process-level memory counters where available.
type ProcessMemory struct {
	VirtualMemoryBytes      uint64
	ResidentMemoryBytes     uint64
	PeakResidentMemoryBytes uint64
}

// GetProcessMemory returns zero counters on unsupported platforms.
func GetProcessMemory() ProcessMemory {
	return ProcessMemory{}
}
