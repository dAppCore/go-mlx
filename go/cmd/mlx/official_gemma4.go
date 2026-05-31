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

func runOfficialGemma4VerifyCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("official-gemma4-verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	role := fs.String("role", mlx.OfficialGemma4E2BRoleTarget, "official Gemma 4 E2B role: target or assistant")
	jsonOut := fs.Bool("json", false, "write JSON verification report")
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
	report := officialGemma4VerifyReportFromPreflight(snapshotDir, preflight)
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

func officialGemma4VerifyReportFromPreflight(snapshotDir string, preflight mlx.OfficialGemma4E2BSnapshotReport) officialGemma4VerifyReport {
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
		Pack:                 preflight.Pack,
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
			writeOfficialGemma4PairVerifyJSON(stdout, stderr, report)
			return 1
		}
		core.Print(stderr, "%s official-gemma4-pair-verify: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeOfficialGemma4PairVerifyJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B target+assistant pair verified: %s %s\n", targetDir, assistantDir))
	return 0
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
