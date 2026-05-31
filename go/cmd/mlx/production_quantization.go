// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

type productionQuantizationReport struct {
	Version int `json:"version"`

	Policy      mlx.ProductionQuantizationPolicy     `json:"policy"`
	SourceLocks []mlx.OfficialGemma4E2BLock          `json:"official_source_locks"`
	Platform    []mlx.OfficialPlatformAPILock        `json:"platform_api_locks"`
	PackLocks   []mlx.ProductionQuantizationPackLock `json:"quantized_target_locks"`
	MTPPolicy   mlx.ProductionMTPPolicy              `json:"mtp_policy"`
	TurboQuant  mlx.ProductionTurboQuantPolicy       `json:"turboquant_policy"`
	Input       productionQuantizationInputReport    `json:"input"`
	Choice      mlx.ProductionQuantizationChoice     `json:"choice"`
	Command     string                               `json:"command,omitempty"`
}

type productionQuantizationInputReport struct {
	Device              productionQuantizationDeviceReport `json:"device"`
	ContextLength       int                                `json:"context_length,omitempty"`
	QualityFirst        bool                               `json:"quality_first,omitempty"`
	ConstrainedFallback bool                               `json:"constrained_fallback,omitempty"`
}

type productionQuantizationDeviceReport struct {
	Architecture                 string `json:"architecture,omitempty"`
	MaxBufferLength              uint64 `json:"max_buffer_length,omitempty"`
	MaxRecommendedWorkingSetSize uint64 `json:"max_recommended_working_set_size,omitempty"`
	MemorySize                   uint64 `json:"memory_size,omitempty"`
}

func runProductionQuantizationCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-quantization"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON quantisation selection report")
	contextLen := fs.Int("context", mlx.ProductionLaneLongContextLength, "requested retained-context length")
	qualityFirst := fs.Bool("quality", false, "prefer the q8 quality tier when memory headroom allows it")
	constrainedFallback := fs.Bool("constrained", false, "force the q4 constrained fallback tier")
	memoryGiB := fs.Uint64("memory-gib", 0, "override device memory in GiB for planning")
	workingSetGiB := fs.Uint64("working-set-gib", 0, "override recommended working set in GiB for planning")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-quantization [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Select the app-facing Gemma 4 E2B quantisation tier without\n")
		core.WriteString(stderr, "loading a model. The default context is the opencode-sized\n")
		core.WriteString(stderr, "retained workflow target. The ladder is q8 quality, q6\n")
		core.WriteString(stderr, "default, q4 constrained fallback/control.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s production-quantization -json\n", name))
		core.WriteString(stderr, core.Sprintf("    # select for this machine\n"))
		core.WriteString(stderr, core.Sprintf("  %s production-quantization -quality -context 32768\n", name))
		core.WriteString(stderr, core.Sprintf("    # prefer q8 for a long retained context when it fits\n"))
		core.WriteString(stderr, core.Sprintf("  %s production-quantization -memory-gib 16 -working-set-gib 13 -context 32768\n", name))
		core.WriteString(stderr, core.Sprintf("    # simulate a constrained machine\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s production-quantization: expected no positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}
	if *contextLen < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-quantization: context must be >= 0\n", cliName()))
		return 2
	}

	device := productionQuantizationDevice(*memoryGiB, *workingSetGiB)
	input := mlx.ProductionQuantizationSelectionInput{
		Device:              device,
		ContextLength:       *contextLen,
		QualityFirst:        *qualityFirst,
		ConstrainedFallback: *constrainedFallback,
	}
	report := productionQuantizationReport{
		Version:     1,
		Policy:      mlx.DefaultProductionQuantizationPolicy(),
		SourceLocks: mlx.DefaultOfficialGemma4E2BLocks(),
		Platform:    mlx.DefaultOfficialPlatformAPILocks(),
		PackLocks:   mlx.DefaultProductionQuantizationPackLocks(),
		MTPPolicy:   mlx.DefaultProductionMTPPolicy(),
		TurboQuant:  mlx.DefaultProductionTurboQuantPolicy(),
		Input:       productionQuantizationInput(input),
		Choice:      mlx.SelectProductionQuantizationTier(input),
		Command:     "production-quantization",
	}

	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-quantization: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionQuantizationReport(stdout, report)
	return 0
}

func productionQuantizationDevice(memoryGiB, workingSetGiB uint64) memory.DeviceInfo {
	if memoryGiB == 0 && workingSetGiB == 0 {
		return productionQuantizationMemoryDevice(runGetDeviceInfo())
	}
	device := memory.DeviceInfo{}
	if memoryGiB > 0 {
		device.MemorySize = memoryGiB * memory.GiB
	}
	if workingSetGiB > 0 {
		device.MaxRecommendedWorkingSetSize = workingSetGiB * memory.GiB
	}
	return device
}

func productionQuantizationMemoryDevice(device mlx.DeviceInfo) memory.DeviceInfo {
	return memory.DeviceInfo{
		Architecture:                 device.Architecture,
		MaxBufferLength:              device.MaxBufferLength,
		MaxRecommendedWorkingSetSize: device.MaxRecommendedWorkingSetSize,
		MemorySize:                   device.MemorySize,
	}
}

func productionQuantizationInput(input mlx.ProductionQuantizationSelectionInput) productionQuantizationInputReport {
	return productionQuantizationInputReport{
		Device: productionQuantizationDeviceReport{
			Architecture:                 input.Device.Architecture,
			MaxBufferLength:              input.Device.MaxBufferLength,
			MaxRecommendedWorkingSetSize: input.Device.MaxRecommendedWorkingSetSize,
			MemorySize:                   input.Device.MemorySize,
		},
		ContextLength:       input.ContextLength,
		QualityFirst:        input.QualityFirst,
		ConstrainedFallback: input.ConstrainedFallback,
	}
}

func printProductionQuantizationReport(stdout io.Writer, report productionQuantizationReport) {
	choice := report.Choice
	core.WriteString(stdout, core.Sprintf(
		"production quantisation: q%d %s (%s)\n",
		choice.Tier.Bits,
		choice.Tier.Name,
		choice.Tier.ModelID,
	))
	core.WriteString(stdout, core.Sprintf("  fits: %v, reason: %s\n", choice.Fits, choice.Reason))
	core.WriteString(stdout, core.Sprintf(
		"  working set: %d bytes, required: %d bytes, context: %d\n",
		choice.WorkingSetBytes,
		choice.RequiredWorkingSet,
		report.Input.ContextLength,
	))
	if choice.Tier.ActiveWeightReadBytesPerToken > 0 {
		core.WriteString(stdout, core.Sprintf(
			"  active weight read: %d bytes/token (%s)\n",
			choice.Tier.ActiveWeightReadBytesPerToken,
			report.Policy.DecodeThroughputEstimate,
		))
	}
	core.WriteString(stdout, "  ladder: q8 quality, q6 default, q4 constrained fallback\n")
	for _, lock := range report.SourceLocks {
		core.WriteString(stdout, core.Sprintf("  official source: %s %s@%s\n", lock.Role, lock.ModelID, lock.Revision))
	}
	for _, lock := range report.Platform {
		core.WriteString(stdout, core.Sprintf("  platform api: %s %s (%s)\n", lock.MinimumOS, lock.Name, lock.SourceURL))
	}
	for _, lock := range report.PackLocks {
		core.WriteString(stdout, core.Sprintf("  locked pack: q%d %s@%s\n", lock.QuantBits, lock.ModelID, lock.Revision))
	}
	core.WriteString(stdout, core.Sprintf(
		"  mtp: default=%v, draft_tokens=%d, promotion=%d retained turns plus side-by-side greedy-parity win\n",
		report.MTPPolicy.EnabledByDefault,
		report.MTPPolicy.DefaultDraftTokens,
		report.MTPPolicy.MinimumRetainedTurns,
	))
	core.WriteString(stdout, core.Sprintf(
		"  turboquant: default=%v, cache=%s, explicit_opt_in=%v, stress_ctx=%d\n",
		report.TurboQuant.EnabledByDefault,
		report.TurboQuant.CacheMode,
		report.TurboQuant.RequiresExplicitOptIn,
		report.TurboQuant.StressContextLength,
	))
}
