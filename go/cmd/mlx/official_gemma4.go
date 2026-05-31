// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

var officialGemma4VerifyLockByRole = mlx.OfficialGemma4E2BLockByRole

type officialGemma4VerifyReport struct {
	Version     int    `json:"version"`
	SnapshotDir string `json:"snapshot_dir"`
	Role        string `json:"role"`
	ModelID     string `json:"model_id"`
	Revision    string `json:"revision"`
	Verified    bool   `json:"verified"`
	Error       string `json:"error,omitempty"`
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
	report := officialGemma4VerifyReport{
		Version:     1,
		SnapshotDir: snapshotDir,
		Role:        lock.Role,
		ModelID:     lock.ModelID,
		Revision:    lock.Revision,
	}
	if err := lock.VerifyLocalSnapshot(snapshotDir); err != nil {
		report.Error = err.Error()
		if *jsonOut {
			writeOfficialGemma4VerifyJSON(stdout, stderr, report)
			return 1
		}
		core.Print(stderr, "%s official-gemma4-verify: %v", cliName(), err)
		return 1
	}
	report.Verified = true
	if *jsonOut {
		return writeOfficialGemma4VerifyJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, core.Sprintf("official Gemma 4 E2B %s verified: %s\n", report.Role, snapshotDir))
	return 0
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
