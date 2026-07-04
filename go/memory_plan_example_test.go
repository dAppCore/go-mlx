// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/memory"
)

func ExamplePlanMemory() {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 14 * memory.GiB,
		},
	})
	core.Println(plan.MachineClass, plan.ContextLength, plan.CachePolicy, plan.PromptCache)
	// Output: apple-silicon-16gb 8192 rotating false
}
