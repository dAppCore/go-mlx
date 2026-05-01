// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	stdio "io"
	"io/fs"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// LoadModelFromMedium stages model files from an io.Medium before loading them.
//
//	model, err := mlx.LoadModelFromMedium(medium, "models/gemma-3-1b", mlx.WithContextLength(8192))
func LoadModelFromMedium(medium coreio.Medium, modelPath string, opts ...LoadOption) (*Model, error) {
	return LoadModel(modelPath, append(opts, WithMedium(medium))...)
}

func stageModelFromMedium(medium coreio.Medium, modelPath string) (string, func() error, error) {
	return stagePathFromMedium(medium, modelPath)
}

func stagePathFromMedium(medium coreio.Medium, path string) (string, func() error, error) {
	if medium == nil {
		return "", nil, core.E("mlx.stagePathFromMedium", "medium is nil", nil)
	}

	root := mediumModelRoot(path)
	if root != "" && !medium.Exists(root) {
		return "", nil, core.E("mlx.stagePathFromMedium", "path not found in medium: "+root, nil)
	}

	stageDirResult := core.MkdirTemp("", "mlx-medium-*")
	if !stageDirResult.OK {
		return "", nil, core.E("mlx.stagePathFromMedium", "create staging dir", stageDirResult.Value.(error))
	}
	stageDir := stageDirResult.Value.(string)

	cleanup := func() error {
		if r := core.RemoveAll(stageDir); !r.OK {
			return r.Value.(error)
		}
		return nil
	}

	if err := copyMediumTree(medium, root, stageDir); err != nil {
		if cleanupErr := cleanup(); cleanupErr != nil {
			core.Warn("mlx: cleanup staging dir after copy failure", "error", cleanupErr)
		}
		return "", nil, core.E("mlx.stagePathFromMedium", "stage path tree", err)
	}

	relative := mediumRelativePath(root, cleanMediumPath(path))
	if relative == "" {
		return stageDir, cleanup, nil
	}
	return core.PathJoin(stageDir, fromSlashPath(relative)), cleanup, nil
}

func mediumModelRoot(modelPath string) string {
	cleaned := cleanMediumPath(modelPath)
	switch {
	case core.HasSuffix(cleaned, ".gguf"), core.HasSuffix(cleaned, ".safetensors"):
		return cleanMediumPath(core.PathDir(cleaned))
	default:
		return cleaned
	}
}

func cleanMediumPath(p string) string {
	cleaned := core.CleanPath(core.Trim(p), "/")
	if cleaned == "." {
		return ""
	}
	return cleaned
}

func mediumRelativePath(root, target string) string {
	if target == "" {
		return ""
	}
	if root == "" {
		return core.TrimPrefix(target, "/")
	}
	// Forward-slash paths are POSIX; compute relative via filepath.Rel and
	// convert back to slash form so callers receive consistent separators.
	relativeResult := core.PathRel(fromSlashPath(root), fromSlashPath(target))
	if !relativeResult.OK || relativeResult.Value.(string) == "." {
		return ""
	}
	return core.PathToSlash(relativeResult.Value.(string))
}

func copyMediumTree(medium coreio.Medium, sourceRoot, destinationRoot string) error {
	if sourceRoot != "" && !medium.IsDir(sourceRoot) {
		return core.E("mlx.copyMediumTree", "source root is not a directory: "+sourceRoot, nil)
	}
	if r := core.MkdirAll(destinationRoot, 0o755); !r.OK {
		return core.E("mlx.copyMediumTree", "create destination root", r.Value.(error))
	}
	return walkMedium(medium, sourceRoot, func(sourcePath string, entry fs.DirEntry) error {
		relative := mediumRelativePath(sourceRoot, sourcePath)
		destinationPath := destinationRoot
		if relative != "" {
			destinationPath = core.PathJoin(destinationRoot, fromSlashPath(relative))
		}
		if entry.IsDir() {
			if r := core.MkdirAll(destinationPath, 0o755); !r.OK {
				return core.E("mlx.copyMediumTree", "create directory", r.Value.(error))
			}
			return nil
		}
		return copyMediumFile(medium, sourcePath, destinationPath)
	})
}

func walkMedium(medium coreio.Medium, root string, visit func(string, fs.DirEntry) error) error {
	entries, err := medium.List(root)
	if err != nil {
		return core.E("mlx.walkMedium", "list "+root, err)
	}
	for _, entry := range entries {
		entryPath := entry.Name()
		if root != "" {
			entryPath = core.PathJoin(root, entry.Name())
		}
		if err := visit(entryPath, entry); err != nil {
			return err
		}
		if entry.IsDir() {
			if err := walkMedium(medium, entryPath, visit); err != nil {
				return err
			}
		}
	}
	return nil
}

func copyMediumFile(medium coreio.Medium, sourcePath, destinationPath string) error {
	reader, err := medium.ReadStream(sourcePath)
	if err != nil {
		return core.E("mlx.copyMediumFile", "open "+sourcePath, err)
	}
	defer reader.Close()

	mode := fs.FileMode(0o644)
	if info, err := medium.Stat(sourcePath); err == nil {
		mode = info.Mode()
	}

	if r := core.MkdirAll(core.PathDir(destinationPath), 0o755); !r.OK {
		return core.E("mlx.copyMediumFile", "create parent directories", r.Value.(error))
	}

	writerResult := core.OpenFile(destinationPath, core.O_CREATE|core.O_TRUNC|core.O_WRONLY, mode)
	if !writerResult.OK {
		return core.E("mlx.copyMediumFile", "create "+destinationPath, writerResult.Value.(error))
	}
	writer := writerResult.Value.(*core.OSFile)
	defer writer.Close()

	if _, err := stdio.Copy(writer, reader); err != nil {
		return core.E("mlx.copyMediumFile", "copy "+sourcePath, err)
	}
	return nil
}

func fromSlashPath(path string) string {
	return core.Replace(path, "/", string(core.PathSeparator))
}
