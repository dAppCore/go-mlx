// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExamplePlanMemory() {
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{
			MemorySize:                   16 * MemoryGiB,
			MaxRecommendedWorkingSetSize: 14 * MemoryGiB,
		},
	})
	core.Println(plan.MachineClass, plan.ContextLength, plan.CachePolicy, plan.PromptCache)
	// Output: apple-silicon-16gb 8192 rotating false
}
