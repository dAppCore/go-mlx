// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

type productionCombinedMTPAndTurboQuantCompareReport struct {
	Version              int                                            `json:"version"`
	Command              string                                         `json:"command,omitempty"`
	MTPReportPath        string                                         `json:"mtp_report_path"`
	TurboQuantReportPath string                                         `json:"turboquant_report_path"`
	Policy               mlx.ProductionCombinedMTPAndTurboQuantPolicy   `json:"policy"`
	MTPEvidence          mlx.ProductionMTPPromotionEvidence             `json:"mtp_evidence"`
	TurboQuantEvidence   mlx.ProductionTurboQuantPromotionEvidence      `json:"turboquant_evidence"`
	MTPDecision          mlx.ProductionMTPPromotionDecision             `json:"mtp_decision"`
	TurboQuantDecision   mlx.ProductionTurboQuantPromotionDecision      `json:"turboquant_decision"`
	Decision             mlx.ProductionCombinedMTPAndTurboQuantDecision `json:"decision"`
}

func runProductionCombinedMTPAndTurboQuantCompareCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-mtp-turboquant-compare"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON combined MTP plus TurboQuant promotion report")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-mtp-turboquant-compare [flags] MTP_COMPARE.json TURBOQUANT_COMPARE.json\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Combine existing production-mtp-compare and production-turboquant-compare\n")
		core.WriteString(stderr, "reports and apply the joint promotion policy. Rejection is an auditable\n")
		core.WriteString(stderr, "report, not a command failure.\n")
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
	if fs.NArg() != 2 {
		core.WriteString(stderr, core.Sprintf("%s production-mtp-turboquant-compare: expected MTP and TurboQuant comparison JSON paths\n", cliName()))
		fs.Usage()
		return 2
	}

	mtpPath, turboPath := fs.Arg(0), fs.Arg(1)
	mtp, err := readProductionCombinedMTPReport(mtpPath)
	if err != nil {
		core.Print(stderr, "%s production-mtp-turboquant-compare: read MTP report: %v", cliName(), err)
		return 1
	}
	turbo, err := readProductionCombinedTurboQuantReport(turboPath)
	if err != nil {
		core.Print(stderr, "%s production-mtp-turboquant-compare: read TurboQuant report: %v", cliName(), err)
		return 1
	}
	report := newProductionCombinedMTPAndTurboQuantCompareReport(mtpPath, mtp, turboPath, turbo)
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-mtp-turboquant-compare: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionCombinedMTPAndTurboQuantCompareReport(stdout, report)
	return 0
}

func readProductionCombinedMTPReport(path string) (productionMTPCompareReport, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return productionMTPCompareReport{}, core.Errorf("read %s: %v", path, read.Value)
	}
	var report productionMTPCompareReport
	if result := core.JSONUnmarshal(read.Value.([]byte), &report); !result.OK {
		return productionMTPCompareReport{}, core.Errorf("decode %s: %v", path, result.Value)
	}
	return report, nil
}

func readProductionCombinedTurboQuantReport(path string) (productionTurboQuantCompareReport, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return productionTurboQuantCompareReport{}, core.Errorf("read %s: %v", path, read.Value)
	}
	var report productionTurboQuantCompareReport
	if result := core.JSONUnmarshal(read.Value.([]byte), &report); !result.OK {
		return productionTurboQuantCompareReport{}, core.Errorf("decode %s: %v", path, result.Value)
	}
	return report, nil
}

func newProductionCombinedMTPAndTurboQuantCompareReport(mtpPath string, mtp productionMTPCompareReport, turboPath string, turbo productionTurboQuantCompareReport) productionCombinedMTPAndTurboQuantCompareReport {
	policy := mlx.DefaultProductionCombinedMTPAndTurboQuantPolicy()
	return productionCombinedMTPAndTurboQuantCompareReport{
		Version:              1,
		Command:              "production-mtp-turboquant-compare",
		MTPReportPath:        mtpPath,
		TurboQuantReportPath: turboPath,
		Policy:               policy,
		MTPEvidence:          mtp.Evidence,
		TurboQuantEvidence:   turbo.Evidence,
		MTPDecision:          mtp.Decision,
		TurboQuantDecision:   turbo.Decision,
		Decision:             mlx.EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtp.Evidence, turbo.Evidence),
	}
}

func printProductionCombinedMTPAndTurboQuantCompareReport(stdout io.Writer, report productionCombinedMTPAndTurboQuantCompareReport) {
	decision := report.Decision
	core.WriteString(stdout, core.Sprintf(
		"mtp+turboquant: candidate=%v, default=%v, reason=%s\n",
		decision.ProductionCandidate,
		decision.EnableByDefault,
		decision.Reason,
	))
	core.WriteString(stdout, core.Sprintf(
		"  mtp eligible=%v, turboquant eligible=%v, cache=%s\n",
		decision.MTPEligible,
		decision.TurboQuantEligible,
		report.Policy.CacheMode,
	))
	if decision.MTPAcceptanceRate > 0 || decision.TurboQuantMemorySavingsRatio > 0 {
		core.WriteString(stdout, core.Sprintf(
			"  mtp acceptance=%.3f, turboquant memory savings=%.3f, energy savings=%.3f\n",
			decision.MTPAcceptanceRate,
			decision.TurboQuantMemorySavingsRatio,
			decision.TurboQuantEnergySavingsRatio,
		))
	}
}
