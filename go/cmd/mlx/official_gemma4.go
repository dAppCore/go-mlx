// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	mp "dappco.re/go/mlx/pack"
)

var officialGemma4VerifyLockByRole = mlx.OfficialGemma4E2BLockByRole
var officialGemma4PairInspect = func(targetDir, assistantDir string) (mlx.OfficialGemma4E2BPairReport, error) {
	return mlx.InspectOfficialGemma4E2BPairSnapshots(targetDir, assistantDir)
}
var officialGemma4ControlCompare = func(targetDir, controlDir string) (mlx.OfficialGemma4E2BControlComparison, error) {
	return mlx.CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir, mlx.OfficialGemma4E2BTargetLock())
}

type officialGemma4LocksReport struct {
	Version              int                                  `json:"version"`
	Kind                 string                               `json:"kind"`
	SourceCheckedAt      string                               `json:"source_checked_at,omitempty"`
	Locks                []mlx.OfficialGemma4E2BLock          `json:"locks"`
	QuantizedTargetLocks []mlx.ProductionQuantizationPackLock `json:"quantized_target_locks"`
	PlatformAPILocks     []mlx.OfficialPlatformAPILock        `json:"platform_api_locks"`
}

type officialGemma4VerifyReport struct {
	Version              int          `json:"version"`
	SnapshotDir          string       `json:"snapshot_dir"`
	Role                 string       `json:"role"`
	ModelID              string       `json:"model_id"`
	Revision             string       `json:"revision"`
	ExpectedArchitecture string       `json:"expected_architecture,omitempty"`
	ArchitectureOK       bool         `json:"architecture_ok"`
	Verified             bool         `json:"verified"`
	Pack                 mp.ModelPack `json:"pack"`
	Error                string       `json:"error,omitempty"`
}

func runOfficialGemma4LocksCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("official-gemma4-locks", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON source-lock report")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s official-gemma4-locks: expected no positional arguments", cliName())
		return 2
	}
	report := officialGemma4LocksReportFromDefaults()
	if *jsonOut {
		return writeOfficialGemma4LocksJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B locks checked %s\n", report.SourceCheckedAt))
	for _, lock := range report.Locks {
		core.WriteString(stdout, core.Sprintf("  %s: %s @ %s (%s, gated=%t)\n", lock.Role, lock.ModelID, lock.Revision, lock.Licence, lock.Gated))
	}
	for _, lock := range report.QuantizedTargetLocks {
		core.WriteString(stdout, core.Sprintf("  q%d target: %s @ %s (%s)\n", lock.QuantBits, lock.ModelID, lock.Revision, lock.ConversionTool))
	}
	for _, lock := range report.PlatformAPILocks {
		core.WriteString(stdout, core.Sprintf("  platform: %s %s (%s)\n", lock.MinimumOS, lock.Name, lock.SourceURL))
	}
	return 0
}

func officialGemma4LocksReportFromDefaults() officialGemma4LocksReport {
	locks := mlx.DefaultOfficialGemma4E2BLocks()
	sourceCheckedAt := ""
	if len(locks) > 0 {
		sourceCheckedAt = locks[0].SourceCheckedAt
	}
	return officialGemma4LocksReport{
		Version:              1,
		Kind:                 "official-gemma4-e2b-source-lock",
		SourceCheckedAt:      sourceCheckedAt,
		Locks:                locks,
		QuantizedTargetLocks: mlx.DefaultProductionQuantizationPackLocks(),
		PlatformAPILocks:     mlx.DefaultOfficialPlatformAPILocks(),
	}
}

func writeOfficialGemma4LocksJSON(stdout, stderr io.Writer, report officialGemma4LocksReport) int {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		core.Print(stderr, "%s official-gemma4-locks: marshal report failed", cliName())
		return 1
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return 0
}

func runOfficialGemma4VerifyCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("official-gemma4-verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	role := fs.String("role", mlx.OfficialGemma4E2BRoleTarget, "official Gemma 4 E2B role: target or assistant")
	jsonOut := fs.Bool("json", false, "write JSON verification report")
	includeChatTemplate := fs.Bool("include-chat-template", false, "include raw chat template bodies in JSON reports")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s official-gemma4-verify: expected one snapshot directory", cliName())
		return 2
	}
	lock, ok := officialGemma4VerifyLockByRole(core.Trim(*role))
	if !ok {
		core.Print(stderr, "%s official-gemma4-verify: unknown official Gemma 4 E2B role %q", cliName(), *role)
		return 2
	}
	snapshotDir := fs.Arg(0)
	preflight, err := lock.InspectLocalSnapshot(snapshotDir)
	report := officialGemma4VerifyReportFromPreflight(snapshotDir, preflight, *includeChatTemplate)
	if err != nil {
		if report.Error == "" {
			report.Error = err.Error()
		}
		if *jsonOut {
			writeOfficialGemma4VerifyJSON(stdout, stderr, report)
			return 1
		}
		core.Print(stderr, "%s official-gemma4-verify: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeOfficialGemma4VerifyJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B %s verified: %s\n", report.Role, report.SnapshotDir))
	return 0
}

func officialGemma4VerifyReportFromPreflight(snapshotDir string, preflight mlx.OfficialGemma4E2BSnapshotReport, includeChatTemplate bool) officialGemma4VerifyReport {
	if preflight.SnapshotDir != "" {
		snapshotDir = preflight.SnapshotDir
	}
	return officialGemma4VerifyReport{
		Version:              1,
		SnapshotDir:          snapshotDir,
		Role:                 preflight.Role,
		ModelID:              preflight.ModelID,
		Revision:             preflight.Revision,
		ExpectedArchitecture: preflight.ExpectedArchitecture,
		ArchitectureOK:       preflight.ArchitectureOK,
		Verified:             preflight.Verified,
		Pack:                 officialGemma4PackForReport(preflight.Pack, includeChatTemplate),
		Error:                preflight.Error,
	}
}

func writeOfficialGemma4VerifyJSON(stdout, stderr io.Writer, report officialGemma4VerifyReport) int {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		core.Print(stderr, "%s official-gemma4-verify: marshal report failed", cliName())
		return 1
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return 0
}

func runOfficialGemma4PairVerifyCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("official-gemma4-pair-verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON pair verification report")
	includeChatTemplate := fs.Bool("include-chat-template", false, "include raw chat template bodies in JSON reports")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 2 {
		core.Print(stderr, "%s official-gemma4-pair-verify: expected target and assistant snapshot directories", cliName())
		return 2
	}
	targetDir, assistantDir := fs.Arg(0), fs.Arg(1)
	report, err := officialGemma4PairInspect(targetDir, assistantDir)
	if err != nil {
		if report.Error == "" {
			report.Error = err.Error()
		}
		if *jsonOut {
			writeOfficialGemma4PairVerifyJSON(stdout, stderr, officialGemma4PairReportForOutput(report, *includeChatTemplate))
			return 1
		}
		core.Print(stderr, "%s official-gemma4-pair-verify: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeOfficialGemma4PairVerifyJSON(stdout, stderr, officialGemma4PairReportForOutput(report, *includeChatTemplate))
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B target+assistant pair verified: %s %s\n", targetDir, assistantDir))
	return 0
}

func officialGemma4PairReportForOutput(report mlx.OfficialGemma4E2BPairReport, includeChatTemplate bool) mlx.OfficialGemma4E2BPairReport {
	report.Target.Pack = officialGemma4PackForReport(report.Target.Pack, includeChatTemplate)
	report.Assistant.Pack = officialGemma4PackForReport(report.Assistant.Pack, includeChatTemplate)
	return report
}

func officialGemma4PackForReport(pack mp.ModelPack, includeChatTemplate bool) mp.ModelPack {
	if !includeChatTemplate {
		pack.ChatTemplate = ""
	}
	return pack
}

func writeOfficialGemma4PairVerifyJSON(stdout, stderr io.Writer, report mlx.OfficialGemma4E2BPairReport) int {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		core.Print(stderr, "%s official-gemma4-pair-verify: marshal report failed", cliName())
		return 1
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return 0
}

func runOfficialGemma4ControlCompareCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("official-gemma4-control-compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON control comparison report")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 2 {
		core.Print(stderr, "%s official-gemma4-control-compare: expected target and q4 control snapshot directories", cliName())
		return 2
	}
	targetDir, controlDir := fs.Arg(0), fs.Arg(1)
	report, err := officialGemma4ControlCompare(targetDir, controlDir)
	if err != nil {
		if *jsonOut {
			writeOfficialGemma4ControlCompareJSON(stdout, stderr, report)
			return 1
		}
		core.Print(stderr, "%s official-gemma4-control-compare: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeOfficialGemma4ControlCompareJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B target matches archived q4 control metadata: %s %s\n", targetDir, controlDir))
	return 0
}

func writeOfficialGemma4ControlCompareJSON(stdout, stderr io.Writer, report mlx.OfficialGemma4E2BControlComparison) int {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		core.Print(stderr, "%s official-gemma4-control-compare: marshal report failed", cliName())
		return 1
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return 0
}
