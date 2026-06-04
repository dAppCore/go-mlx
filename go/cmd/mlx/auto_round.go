// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/quant/autoround"
)

type autoRoundProfilesReport struct {
	Version          int                      `json:"version"`
	Kind             string                   `json:"kind"`
	NoPython         bool                     `json:"no_python"`
	Source           string                   `json:"source"`
	Profiles         []autoRoundProfileReport `json:"profiles"`
	Schemes          []autoround.SchemeInfo   `json:"schemes"`
	PackSidecars     []string                 `json:"pack_sidecars"`
	AlgorithmProfile profile.AlgorithmProfile `json:"algorithm_profile"`
	Notes            []string                 `json:"notes,omitempty"`
}

type autoRoundProfileReport struct {
	autoround.Profile
	GroupScheme        autoround.SchemeInfo        `json:"group_scheme"`
	Config             autoround.QuantizeConfig    `json:"config"`
	CalibrationDefault autoround.CalibrationConfig `json:"calibration_default"`
}

func runAutoRoundCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("auto-round", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON profile report")
	profileID := fs.String("profile", "", "profile id to report: auto-round, auto-round-best, or auto-round-light")
	fs.Usage = func() {
		name := cliCommandName("auto-round")
		core.WriteString(stderr, core.Sprintf("Usage: %s [-json] [-profile auto-round]\n", name))
		core.WriteString(stderr, "Report native AutoRound profile defaults and supported schemes.\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s auto-round: expected no positional arguments", cliName())
		return 2
	}
	report, err := autoRoundReport(*profileID)
	if err != nil {
		core.Print(stderr, "%s auto-round: %v", cliName(), err)
		return 2
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s auto-round: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	core.WriteString(stdout, "AutoRound profiles\n")
	for _, profileReport := range report.Profiles {
		p := profileReport.Profile
		core.WriteString(stdout, core.Sprintf("  %s: scheme=%s format=%s iters=%d nsamples=%d seqlen=%d group_size=%d\n",
			p.ID,
			p.Scheme,
			p.ExportFormat,
			p.Iters,
			p.NSamples,
			p.SeqLen,
			p.GroupSize,
		))
	}
	return 0
}

func autoRoundReport(selected string) (autoRoundProfilesReport, error) {
	algorithm, _ := profile.LookupAlgorithmProfile(inference.CapabilityQuantization)
	report := autoRoundProfilesReport{
		Version:          1,
		Kind:             "auto-round-profiles",
		NoPython:         true,
		Source:           "https://github.com/intel/auto-round",
		Schemes:          autoRoundSchemeReports(),
		PackSidecars:     []string{autoround.PackConfigFileAutoRound, autoround.PackConfigFileQuantization},
		AlgorithmProfile: algorithm,
		Notes: []string{
			"Profiles expose native Go metadata, RTN/SignRound primitives, and packed CPU/Metal dequant/projection helpers; no Python runtime is used.",
			"Model-pack sidecar recognition, native tensor-map validation/loading, native pack sidecar + safetensors export, validated tensor-map inspection, Metal fused projection consumption, and calibration planning are available; GGUF export orchestration, gradient capture, and model generate validation are follow-up work.",
			"Use iters=0 through QuantizeConfig for an RTN baseline.",
		},
	}
	if core.Trim(selected) != "" {
		p, ok := autoround.LookupProfile(autoround.ProfileID(core.Trim(selected)))
		if !ok {
			return report, core.Errorf("unknown profile %q", selected)
		}
		report.Profiles = []autoRoundProfileReport{autoRoundProfileReportFromProfile(p)}
		return report, nil
	}
	for _, p := range autoround.BuiltinProfiles() {
		report.Profiles = append(report.Profiles, autoRoundProfileReportFromProfile(p))
	}
	return report, nil
}

func autoRoundProfileReportFromProfile(p autoround.Profile) autoRoundProfileReport {
	return autoRoundProfileReport{
		Profile:            p,
		GroupScheme:        p.GroupScheme(),
		Config:             autoround.ConfigFromProfile(p),
		CalibrationDefault: autoround.CalibrationConfigFromProfile(p),
	}
}

func autoRoundSchemeReports() []autoround.SchemeInfo {
	schemes := []autoround.Scheme{
		autoround.SchemeW2A16,
		autoround.SchemeW4A16,
		autoround.SchemeW8A16,
		autoround.SchemeMXFP4,
		autoround.SchemeNVFP4,
		autoround.SchemeFP8Static,
		autoround.SchemeGGUFQ4KM,
	}
	out := make([]autoround.SchemeInfo, 0, len(schemes))
	for _, scheme := range schemes {
		if info, ok := autoround.ResolveScheme(scheme); ok {
			out = append(out, info)
		}
	}
	return out
}
