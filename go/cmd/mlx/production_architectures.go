// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

type productionArchitecturesReport struct {
	Version int                                    `json:"version"`
	Status  mlx.ProductionArchitectureStatusReport `json:"status"`
	Command string                                 `json:"command,omitempty"`
}

func runProductionArchitecturesCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-architectures"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON architecture status report")
	gapsOnly := fs.Bool("gaps-only", false, "plain output only: list metadata-only native gaps")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-architectures [flags]\n", name))
		core.WriteString(stderr, "\n")
	core.WriteString(stderr, "Report the production native-runtime architecture matrix without\n")
	core.WriteString(stderr, "loading a model. All architectures are now native; the go/mlxlm\n")
	core.WriteString(stderr, "Python fallback has been removed.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s production-architectures -json\n", name))
		core.WriteString(stderr, core.Sprintf("    # machine-readable native gap report\n"))
		core.WriteString(stderr, core.Sprintf("  %s production-architectures -gaps-only\n", name))
		core.WriteString(stderr, core.Sprintf("    # concise remaining feature list\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s production-architectures: expected no positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}

	report := productionArchitecturesReport{
		Version: 1,
		Status:  mlx.DefaultProductionArchitectureStatus(),
		Command: "production-architectures",
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-architectures: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionArchitecturesReport(stdout, report, *gapsOnly)
	return 0
}

func printProductionArchitecturesReport(stdout io.Writer, report productionArchitecturesReport, gapsOnly bool) {
	status := report.Status
	if !gapsOnly {
			core.WriteString(stdout, core.Sprintf(
			"production architectures: %d native, %d metadata-only, %d total\n",
			status.NativeArchitectures,
			status.MetadataOnlyArchitectures,
			status.TotalArchitectures,
		))
	}
	if len(status.RemainingGaps) == 0 {
		core.WriteString(stdout, "  native gaps: none\n")
		return
	}
	core.WriteString(stdout, "  native gaps:\n")
	for _, gap := range status.RemainingGaps {
		core.WriteString(stdout, core.Sprintf("    %s: %s", gap.ID, gap.MissingNative))
		if gap.MoE {
			core.WriteString(stdout, " [moe]")
		}
		if gap.Embeddings {
			core.WriteString(stdout, " [embeddings]")
		}
		if gap.Rerank {
			core.WriteString(stdout, " [rerank]")
		}
		core.WriteString(stdout, "\n")
		if len(gap.NextWork) > 0 {
			core.WriteString(stdout, core.Sprintf("      next: %s\n", core.Join(", ", gap.NextWork...)))
		}
	}
}
