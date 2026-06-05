// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

// Features is the Gemma 4 architecture's feature surface: what the engine reads
// off a loaded config to configure itself. It is deliberately NOT a list of
// models — there are hundreds of Gemma 4 builds across orgs, quants, and
// fine-tunes, and the engine reacts to what a config declares, never to a model
// name or quant. Adding a new member of the family is "load its config"; the
// engine asks FeaturesOf and reacts, with no code change.
//
//	f := gemma4.FeaturesOf(model.Cfg)
//	if f.Mixture { /* route through the MoE experts path */ }
//	if f.Multimodal() { /* load the vision / audio towers */ }
type Features struct {
	Mixture     bool // mixture-of-experts block active (vs a dense MLP)
	NumExperts  int  // total experts when Mixture, 0 when dense
	TopKExperts int  // experts routed per token when Mixture, 0 when dense
	Vision      bool // vision encoder present
	Audio       bool // audio encoder present
}

// Multimodal reports whether the model carries any non-text encoder.
func (f Features) Multimodal() bool { return f.Vision || f.Audio }

// FeaturesOf reads the feature surface from a loaded Gemma 4 config. A nil config
// reports the zero surface (dense, text-only). This is the single place that
// answers "what is this model" from its settings, so callers react to the
// returned Features rather than poking config fields — a new family member then
// needs no engine change, only a config.
func FeaturesOf(cfg *Gemma4TextConfig) Features {
	if cfg == nil {
		return Features{}
	}
	f := Features{
		Vision: cfg.VisionConfig != nil,
		Audio:  cfg.AudioConfig != nil,
	}
	experts := 0
	if cfg.NumExperts != nil {
		experts = int(*cfg.NumExperts)
	}
	if cfg.EnableMoEBlock || experts > 0 {
		f.Mixture = true
		f.NumExperts = experts
		if cfg.TopKExperts != nil {
			f.TopKExperts = int(*cfg.TopKExperts)
		}
	}
	return f
}
