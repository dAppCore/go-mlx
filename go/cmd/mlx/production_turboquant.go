// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

type productionTurboQuantReport struct {
	Version int                            `json:"version"`
	Kind    string                         `json:"kind"`
	Command string                         `json:"command,omitempty"`
	Policy  mlx.ProductionTurboQuantPolicy `json:"policy"`
}

func runProductionTurboQuantCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-turboquant"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON TurboQuant KV-cache promotion policy")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-turboquant [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Print the explicit TurboQuant KV-cache promotion policy. This is\n")
		core.WriteString(stderr, "a research/validation lane, not a weight quantisation selector and\n")
		core.WriteString(stderr, "not a default runtime mode.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s production-turboquant: expected no positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}

	report := productionTurboQuantReport{
		Version: 1,
		Kind:    "production-turboquant-policy",
		Command: "production-turboquant",
		Policy:  mlx.DefaultProductionTurboQuantPolicy(),
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-turboquant: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionTurboQuantReport(stdout, report)
	return 0
}

func printProductionTurboQuantReport(stdout io.Writer, report productionTurboQuantReport) {
	policy := report.Policy
	core.WriteString(stdout, core.Sprintf(
		"production TurboQuant: mode=%s cache=%s target=%s\n",
		policy.Mode,
		policy.CacheMode,
		policy.TargetModelID,
	))
	core.WriteString(stdout, core.Sprintf(
		"  default: %v, explicit opt-in: %v, target bits: %.3f/channel\n",
		policy.EnabledByDefault,
		policy.RequiresExplicitOptIn,
		float64(policy.TargetEffectiveBitsMilli)/1000,
	))
	core.WriteString(stdout, core.Sprintf(
		"  validation: %d retained turns, normal ctx %d, stress ctx %d, quality parity %v\n",
		policy.MinimumRetainedTurns,
		policy.NormalContextLength,
		policy.StressContextLength,
		policy.RequiresQualityParity,
	))
	core.WriteString(stdout, core.Sprintf("  compare modes: %s\n", core.Join(", ", productionTurboQuantModeStrings(policy.CompareAgainstCacheModes)...)))
}

func productionTurboQuantModeStrings(modes []memory.KVCacheMode) []string {
	out := make([]string, 0, len(modes))
	for _, mode := range modes {
		out = append(out, string(mode))
	}
	return out
}
