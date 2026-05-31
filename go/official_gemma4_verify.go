// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/bundle"
)

const officialGemma4SafetensorsIndexFile = "model.safetensors.index.json"

// VerifyLocalSnapshot checks a downloaded HF snapshot directory or cache root
// against the exact official Gemma 4 E2B target/assistant lock.
func (lock OfficialGemma4E2BLock) VerifyLocalSnapshot(snapshotDir string) error {
	return VerifyOfficialGemma4E2BLocalSnapshot(snapshotDir, lock)
}

// ResolveOfficialGemma4E2BLocalSnapshot returns the concrete locked HF snapshot
// directory from either an exact snapshot path, a snapshots directory, or a HF
// cache repo root such as models--google--gemma-4-E2B-it.
func ResolveOfficialGemma4E2BLocalSnapshot(snapshotDir string, lock OfficialGemma4E2BLock) (string, error) {
	snapshotDir = core.Trim(snapshotDir)
	if snapshotDir == "" {
		return "", core.NewError("mlx: official Gemma 4 E2B snapshot directory is empty")
	}
	if lock.Revision == "" {
		return "", core.NewError("mlx: official Gemma 4 E2B lock revision is empty")
	}
	if core.PathBase(snapshotDir) == lock.Revision {
		return snapshotDir, nil
	}

	for _, candidate := range officialGemma4LocalSnapshotCandidates(snapshotDir, lock.Revision) {
		if result := core.Stat(candidate); result.OK {
			return candidate, nil
		}
	}
	return "", core.NewError(core.Sprintf("mlx: official Gemma 4 E2B locked snapshot %q not found below %q", lock.Revision, snapshotDir))
}

func officialGemma4LocalSnapshotCandidates(snapshotDir, revision string) []string {
	if core.PathBase(snapshotDir) == "snapshots" {
		return []string{core.PathJoin(snapshotDir, revision)}
	}
	return []string{core.PathJoin(snapshotDir, "snapshots", revision)}
}

// VerifyOfficialGemma4E2BLocalSnapshot fails closed when a local official
// Google E2B target or assistant snapshot does not match the pinned artefact
// hashes. This is a lightweight identity gate; it does not prove native-load
// semantics or benchmark quality.
func VerifyOfficialGemma4E2BLocalSnapshot(snapshotDir string, lock OfficialGemma4E2BLock) error {
	resolvedDir, err := ResolveOfficialGemma4E2BLocalSnapshot(snapshotDir, lock)
	if err != nil {
		return err
	}
	snapshotDir = resolvedDir
	if core.PathBase(snapshotDir) != lock.Revision {
		return core.NewError(core.Sprintf("mlx: official Gemma 4 E2B snapshot revision = %q, want %q", core.PathBase(snapshotDir), lock.Revision))
	}

	for _, file := range officialGemma4RequiredHashFiles(lock) {
		if file.sha256 == "" {
			continue
		}
		if err := verifyOfficialGemma4SnapshotFile(snapshotDir, file.name, file.sha256); err != nil {
			return err
		}
	}
	if lock.WeightFile == "" || lock.WeightSHA256 == "" {
		return core.NewError("mlx: official Gemma 4 E2B lock weight identity is incomplete")
	}
	if err := verifyOfficialGemma4SnapshotFile(snapshotDir, lock.WeightFile, lock.WeightSHA256); err != nil {
		return err
	}
	if lock.WeightBytes > 0 {
		if err := verifyOfficialGemma4SnapshotFileSize(snapshotDir, lock.WeightFile, lock.WeightBytes); err != nil {
			return err
		}
	}
	return verifyOfficialGemma4SafetensorsIndex(snapshotDir, lock)
}

type officialGemma4HashFile struct {
	name   string
	sha256 string
}

func officialGemma4RequiredHashFiles(lock OfficialGemma4E2BLock) []officialGemma4HashFile {
	return []officialGemma4HashFile{
		{name: "config.json", sha256: lock.ConfigSHA256},
		{name: "tokenizer.json", sha256: lock.TokenizerSHA256},
		{name: "tokenizer_config.json", sha256: lock.TokenizerConfigSHA256},
		{name: "generation_config.json", sha256: lock.GenerationConfigSHA256},
		{name: "chat_template.jinja", sha256: lock.ChatTemplateSHA256},
	}
}

func verifyOfficialGemma4SnapshotFile(snapshotDir, name, wantSHA256 string) error {
	path := core.PathJoin(snapshotDir, name)
	got, err := bundle.FileHash(path)
	if err != nil {
		return core.E("mlx: official Gemma 4 E2B snapshot", "hash "+name, err)
	}
	if got != wantSHA256 {
		return core.NewError(core.Sprintf("mlx: official Gemma 4 E2B snapshot %s SHA-256 mismatch: got %s want %s", name, got, wantSHA256))
	}
	return nil
}

func verifyOfficialGemma4SnapshotFileSize(snapshotDir, name string, wantBytes uint64) error {
	info := core.Stat(core.PathJoin(snapshotDir, name))
	if !info.OK {
		return core.E("mlx: official Gemma 4 E2B snapshot", "stat "+name, officialGemma4ResultError(info))
	}
	stat, ok := info.Value.(core.FsFileInfo)
	if !ok {
		return core.NewError("mlx: official Gemma 4 E2B snapshot stat returned non-file info")
	}
	gotBytes := uint64(stat.Size())
	if gotBytes != wantBytes {
		return core.NewError(core.Sprintf("mlx: official Gemma 4 E2B snapshot %s size = %d, want %d", name, gotBytes, wantBytes))
	}
	return nil
}

func verifyOfficialGemma4SafetensorsIndex(snapshotDir string, lock OfficialGemma4E2BLock) error {
	if lock.SafetensorsIndexPresent {
		if lock.SafetensorsIndexSHA256 == "" {
			return core.NewError("mlx: official Gemma 4 E2B safetensors index hash is missing")
		}
		return verifyOfficialGemma4SnapshotFile(snapshotDir, officialGemma4SafetensorsIndexFile, lock.SafetensorsIndexSHA256)
	}
	if result := core.Stat(core.PathJoin(snapshotDir, officialGemma4SafetensorsIndexFile)); result.OK {
		return core.NewError("mlx: official Gemma 4 E2B snapshot unexpectedly contains " + officialGemma4SafetensorsIndexFile)
	}
	return nil
}

func officialGemma4ResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	if result.Value == nil {
		return nil
	}
	return core.NewError(core.Sprintf("%v", result.Value))
}
