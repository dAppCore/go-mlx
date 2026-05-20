// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <mach/mach.h>
#include <mach/task_info.h>
#include <stdint.h>

typedef struct go_mlx_process_memory_info_ {
	uint64_t virtual_size;
	uint64_t resident_size;
	uint64_t resident_size_max;
} go_mlx_process_memory_info;

static int go_mlx_process_memory(go_mlx_process_memory_info* out) {
	if (out == NULL) {
		return -1;
	}
	mach_task_basic_info_data_t info;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	kern_return_t kr = task_info(
		mach_task_self(),
		MACH_TASK_BASIC_INFO,
		(task_info_t)&info,
		&count);
	if (kr != KERN_SUCCESS) {
		return (int)kr;
	}
	out->virtual_size = (uint64_t)info.virtual_size;
	out->resident_size = (uint64_t)info.resident_size;
	out->resident_size_max = (uint64_t)info.resident_size_max;
	return 0;
}
*/
import "C"

// ProcessMemory reports process-level memory counters from mach_task_self.
type ProcessMemory struct {
	VirtualMemoryBytes      uint64
	ResidentMemoryBytes     uint64
	PeakResidentMemoryBytes uint64
}

// GetProcessMemory returns current process virtual and resident memory.
func GetProcessMemory() ProcessMemory {
	var info C.go_mlx_process_memory_info
	if C.go_mlx_process_memory(&info) != 0 {
		return ProcessMemory{}
	}
	return ProcessMemory{
		VirtualMemoryBytes:      uint64(info.virtual_size),
		ResidentMemoryBytes:     uint64(info.resident_size),
		PeakResidentMemoryBytes: uint64(info.resident_size_max),
	}
}
