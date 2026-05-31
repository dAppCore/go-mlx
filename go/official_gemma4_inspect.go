// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	modelinspect "dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
)

// OfficialGemma4E2BSnapshotReport ties official snapshot identity checks to
// model-pack inspection. It is intentionally metadata-only: callers can run it
// before a heavyweight native load.
type OfficialGemma4E2BSnapshotReport struct {
	SnapshotDir          string                `json:"snapshot_dir"`
	Role                 string                `json:"role"`
	ModelID              string                `json:"model_id"`
	Revision             string                `json:"revision"`
	ExpectedArchitecture string                `json:"expected_architecture,omitempty"`
	ArchitectureOK       bool                  `json:"architecture_ok"`
	Verified             bool                  `json:"verified"`
	Lock                 OfficialGemma4E2BLock `json:"lock"`
	Pack                 mp.ModelPack          `json:"pack"`
	Error                string                `json:"error,omitempty"`
}

// InspectLocalSnapshot verifies and inspects a downloaded official Gemma 4 E2B
// snapshot using this lock.
func (lock OfficialGemma4E2BLock) InspectLocalSnapshot(snapshotDir string, opts ...mp.ModelPackOption) (OfficialGemma4E2BSnapshotReport, error) {
	return InspectOfficialGemma4E2BLocalSnapshot(snapshotDir, lock, opts...)
}

// InspectOfficialGemma4E2BSnapshot verifies and inspects a downloaded official
// Gemma 4 E2B snapshot by package role.
func InspectOfficialGemma4E2BSnapshot(snapshotDir, role string, opts ...mp.ModelPackOption) (OfficialGemma4E2BSnapshotReport, error) {
	role = core.Trim(role)
	if role == "" {
		return OfficialGemma4E2BSnapshotReport{}, core.NewError("mlx: official Gemma 4 E2B snapshot role is empty")
	}
	lock, ok := OfficialGemma4E2BLockByRole(role)
	if !ok {
		return OfficialGemma4E2BSnapshotReport{}, core.NewError(core.Sprintf("mlx: official Gemma 4 E2B snapshot role %q is not locked", role))
	}
	return InspectOfficialGemma4E2BLocalSnapshot(snapshotDir, lock, opts...)
}

// InspectOfficialGemma4E2BLocalSnapshot fails closed when a local snapshot does
// not match both the pinned official identity and the expected native metadata
// shape for its role.
func InspectOfficialGemma4E2BLocalSnapshot(snapshotDir string, lock OfficialGemma4E2BLock, opts ...mp.ModelPackOption) (OfficialGemma4E2BSnapshotReport, error) {
	report := OfficialGemma4E2BSnapshotReport{
		Role:                 lock.Role,
		ModelID:              lock.ModelID,
		Revision:             lock.Revision,
		ExpectedArchitecture: officialGemma4ExpectedPackArchitecture(lock),
		Lock:                 lock,
	}

	resolvedDir, err := ResolveOfficialGemma4E2BLocalSnapshot(snapshotDir, lock)
	if err != nil {
		return officialGemma4SnapshotReportError(report, err)
	}
	snapshotDir = resolvedDir
	report.SnapshotDir = snapshotDir

	pack, err := modelinspect.Inspect(snapshotDir, opts...)
	report.Pack = pack
	if err != nil {
		return officialGemma4SnapshotReportError(report, err)
	}
	if pack.HasErrorIssue() {
		return officialGemma4SnapshotReportError(report, core.NewError("mlx: official Gemma 4 E2B snapshot pack invalid: "+pack.IssueSummary()))
	}
	report.ArchitectureOK = report.ExpectedArchitecture == "" || pack.Architecture == report.ExpectedArchitecture
	if !report.ArchitectureOK {
		return officialGemma4SnapshotReportError(report, core.NewError(core.Sprintf(
			"mlx: official Gemma 4 E2B snapshot architecture = %q, want %q",
			pack.Architecture,
			report.ExpectedArchitecture,
		)))
	}
	if err := lock.VerifyLocalSnapshot(snapshotDir); err != nil {
		return officialGemma4SnapshotReportError(report, err)
	}
	report.Verified = true
	return report, nil
}

func officialGemma4ExpectedPackArchitecture(lock OfficialGemma4E2BLock) string {
	switch lock.Role {
	case OfficialGemma4E2BRoleTarget:
		return ProductionLaneArchitecture
	case OfficialGemma4E2BRoleAssistant:
		return "gemma4_assistant"
	default:
		return ""
	}
}

func officialGemma4SnapshotReportError(report OfficialGemma4E2BSnapshotReport, err error) (OfficialGemma4E2BSnapshotReport, error) {
	if err != nil {
		report.Verified = false
		report.Error = err.Error()
	}
	return report, err
}
