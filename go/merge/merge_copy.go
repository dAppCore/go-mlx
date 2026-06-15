// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	core "dappco.re/go"
)

func samePath(a, b string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

// samePathResolved is the per-source-loop variant where the right-hand
// side is already absolute. Saves a core.PathAbs call (and any associated
// filesystem inspection) per iteration.
func samePathResolved(a, absB string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	return absA == absB
}

// modelPackMetadataSuffixes is the canonical metadata-extension list.
// Matching is case-sensitive to mirror the filepath.Glob("*.json"/...)
// behaviour the directory scan replaced (filepath.Match is case-sensitive,
// so e.g. Config.JSON was never copied — preserve that).
var modelPackMetadataSuffixes = [...]string{".json", ".model", ".txt"}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	// One directory read replaces the three same-directory globs (one per
	// *.json / *.model / *.txt pattern). Each filepath.Glob opened + read
	// the whole source directory — three readdirs (and three os.newDirFile
	// allocs) per merge, the dominant Packs alloc count at -alloc_objects
	// (os.newFile via os.newDirFile ~38% on the 32-tensor bench). A single
	// ReadDir lists the directory once and the suffix match runs in memory.
	// Iterating distinct entries also makes the previous seen-map basename
	// dedup unnecessary.
	listed := core.ReadDir(core.DirFS(sourceRoot), ".")
	if !listed.OK {
		// A missing/unreadable source metadata dir is not fatal — the merge
		// still produced weights. Mirror the previous glob form, which
		// silently matched nothing on a bad pattern/dir.
		return nil
	}
	entries, ok := listed.Value.([]core.FsDirEntry)
	if !ok {
		return nil
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !hasModelPackMetadataSuffix(name) {
			continue
		}
		if isModelWeightMetadataCopySkip(name) {
			continue
		}
		if err := copyModelPackLocalFile(core.PathJoin(sourceRoot, name), core.PathJoin(outputRoot, name)); err != nil {
			return err
		}
	}
	return nil
}

// hasModelPackMetadataSuffix reports whether name carries a metadata
// extension. Case-sensitive on purpose (see modelPackMetadataSuffixes).
func hasModelPackMetadataSuffix(name string) bool {
	for _, suffix := range modelPackMetadataSuffixes {
		if core.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

func isModelWeightMetadataCopySkip(name string) bool {
	// Two prior issues in this predicate:
	//
	// 1. core.Lower(name) allocated a fresh copy of every filename even
	//    though most metadata filenames are already lowercase.
	// 2. The Contains(".safetensors")|HasSuffix(".safetensors") pair is
	//    redundant — HasSuffix is a strict subset of Contains for the same
	//    suffix. Same for ".gguf". Drop the HasSuffix legs entirely.
	//
	// We keep the Contains semantics (legacy: filters anything *named*
	// with .safetensors in its path, e.g. .safetensors.index.json) by
	// using a case-folding containsFold helper.
	if equalFold(name, "adapter_provenance.json") {
		return true
	}
	return containsFold(name, ".safetensors") || containsFold(name, ".gguf")
}

func copyModelPackLocalFile(sourcePath, destinationPath string) error {
	// Stream source -> destination instead of core.ReadFile (whole file
	// into a heap []byte) + core.WriteFile. Metadata files are small for
	// the config, but tokenizer.json is multiple MB on real checkpoints,
	// so the slurp dominated Packs B/op at -alloc_space (one full-file
	// buffer per copied file). io.Copy uses a fixed ~32 KiB staging buffer
	// regardless of file size. The destination is opened with the same
	// O_WRONLY|O_CREATE|O_TRUNC flags and 0o644 mode core.WriteFile used,
	// so the written bytes and file mode are identical.
	srcOpen := core.Open(sourcePath)
	if !srcOpen.OK {
		return modelPackCopyResultError(srcOpen)
	}
	src := srcOpen.Value.(*core.OSFile)
	defer src.Close()
	dstOpen := core.OpenFile(destinationPath, core.O_WRONLY|core.O_CREATE|core.O_TRUNC, 0o644)
	if !dstOpen.OK {
		return modelPackCopyResultError(dstOpen)
	}
	dst := dstOpen.Value.(*core.OSFile)
	if result := core.Copy(dst, src); !result.OK {
		_ = dst.Close()
		return modelPackCopyResultError(result)
	}
	if err := dst.Close(); err != nil {
		return err
	}
	return nil
}

func modelPackCopyResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errPackMetadataCopy
}

func hashFile(path string) (string, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return "", resultError(read)
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return "", errReadNonByteData
	}
	return core.SHA256Hex(data), nil
}
