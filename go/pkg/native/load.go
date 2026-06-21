// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/profile"
)

// load.go is the native backend's REACTIVE directory loader — it reads a checkpoint's config.json,
// resolves its architecture, and dispatches to the registered model.Loader (model.LookupLoader), the
// same registry-driven shape pkg/metal uses (probe model_type → profile.ResolveArchitecture →
// loader). The backend holds no per-model knowledge: a model package (pkg/model/gemma4, /mistral, …)
// owns its tensor-name mapping and registers its Load from init(); adding an arch is a new package +
// an init() call, with no edit here. The chosen loader returns the neutral model.LoadedModel; the
// generic loadedToBF16/loadedToQuant turn it into the native decode structs (quant vs bf16 read from
// the loaded weights, not a re-parse of the config).

// LoadDir loads any registered architecture's checkpoint directory into a persistent decode session
// with maxLen cache rows — the one-call path from an on-disk checkpoint to a ready-to-Generate
// ArchSession. Zero-copy: the weights view the shard mmap, held on the session via shardBuffers for
// its life (Close unmaps).
func LoadDir(dir string, maxLen int) (*ArchSession, error) {
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		return nil, err
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		_ = dm.Close()
		return nil, err
	}
	var sess *ArchSession
	if quantised(lm) {
		qm, qerr := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
		if qerr != nil {
			_ = sb.Close()
			return nil, qerr
		}
		sess, err = newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	} else {
		sess, err = newArchSessionShards(loadedToBF16(lm), lm.Arch, maxLen, sb)
	}
	if err != nil {
		_ = sb.Close()
		return nil, err
	}
	sess.shards = sb
	return sess, nil
}

// LoadTokenModelDir loads any registered architecture's checkpoint directory as a model.TokenModel —
// the backend-agnostic token-loop contract the serve adapter drives (model.Generate over the returned
// SessionModel). The quant/bf16 path is read from the loaded weights; the per-token serve head is
// built once (buildHeadEncoder) to kill the per-token re-upload.
func LoadTokenModelDir(dir string, maxLen int) (model.TokenModel, error) {
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		return nil, err
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		_ = dm.Close()
		return nil, err
	}
	arch := lm.Arch
	if quantised(lm) {
		gs, bits := lm.Embed.GroupSize, lm.Embed.Bits
		g, qerr := loadedToQuant(lm, gs, bits)
		if qerr != nil {
			_ = sb.Close()
			return nil, qerr
		}
		tm, terr := NewQuantTokenModel(g, arch, maxLen)
		if terr != nil {
			_ = sb.Close()
			return nil, terr
		}
		tm.shards = sb
		he, herr := buildHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
		if herr != nil {
			_ = sb.Close()
			return nil, herr
		}
		tm.headEnc = he
		return tm, nil
	}
	g := loadedToBF16(lm)
	tm, terr := NewBF16TokenModel(g, arch, maxLen)
	if terr != nil {
		_ = sb.Close()
		return nil, terr
	}
	tm.shards = sb
	he, herr := buildHeadEncoder(sb, g.FinalNorm, g.LMHead, nil, nil, arch.Hidden, arch.Vocab, 0, 0, arch.Eps, arch.SoftCap, false)
	if herr != nil {
		_ = sb.Close()
		return nil, herr
	}
	tm.headEnc = he
	return tm, nil
}

// loadRegistered reads config.json, resolves the architecture (profile.ResolveArchitecture — the metal
// path), and dispatches to the registered model.Loader. The shared front half of every directory load;
// returns the neutral LoadedModel + the DirMapping its byte views reference (the caller binds device
// buffers from it, then it lives on the session/token-model via shardBuffers).
func loadRegistered(dir string) (*model.LoadedModel, *safetensors.DirMapping, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, nil, core.E("native.LoadDir", "read config.json", err)
	}
	arch, rawType, err := probeArch([]byte(cfgStr))
	if err != nil {
		return nil, nil, err
	}
	fn := model.LookupLoader(arch)
	if fn == nil && rawType != arch { // fall back to the raw model_type id if the resolved one has no loader
		fn = model.LookupLoader(rawType)
	}
	if fn == nil {
		return nil, nil, core.NewError("native.LoadDir: no loader registered for architecture " + arch + " (model_type " + rawType + ")")
	}
	return fn(dir)
}

// probeArch peeks config.json for the architecture id, mirroring metal's probeModelType: the
// top-level model_type, the nested text_config.model_type (multimodal wrappers), and architectures,
// resolved by profile.ResolveArchitecture. Returns the resolved id + the raw model_type (a fallback
// lookup key when a registration uses the raw id).
func probeArch(data []byte) (resolved, rawType string, err error) {
	var probe struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return "", "", core.NewError("native.LoadDir: config.json model_type parse failed")
	}
	return profile.ResolveArchitecture(probe.ModelType, probe.TextConfig.ModelType, probe.Architectures), probe.ModelType, nil
}

// quantised reports whether the loaded model's weights are quantised — read from the assembled
// embedding (it carries scales) rather than re-parsing the config's quant block. The model-wide
// group-size/bits the native quant structs use come from the same weight (uniform across the pack).
func quantised(m *model.LoadedModel) bool {
	return m != nil && m.Embed != nil && m.Embed.Quantised()
}

// buildHeadEncoder wraps newHeadEncoder in an autorelease pool — the 4-bit head uploads its weight
// once into retained owned buffers, which must be created inside a pool (they survive it, retained).
// The shared constructor for the directory token-model loaders.
func buildHeadEncoder(sb *shardBuffers, finalNormW, weight, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32, quant bool) (*headEncoder, error) {
	var he *headEncoder
	var err error
	withAutoreleasePool(func() {
		he, err = newHeadEncoder(sb, finalNormW, weight, scales, biases, dModel, vocab, groupSize, bits, eps, softCap, quant)
	})
	return he, err
}

// buildShardBuffers wraps each shard's page-aligned mmap in a no-copy Metal buffer inside an
// autorelease pool (the buffers are objc-retained, so they survive the pool — the Go reference on
// the returned shardBuffers keeps them alive). The shared constructor for both directory holders.
func buildShardBuffers(dm *safetensors.DirMapping) (*shardBuffers, error) {
	var sb *shardBuffers
	var err error
	withAutoreleasePool(func() {
		sb, err = newShardBuffers(dm)
	})
	return sb, err
}
