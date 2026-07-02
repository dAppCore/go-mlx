// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	core "dappco.re/go"
	ilora "dappco.re/go/inference/lora"
)

// AdapterInfo is the reproducible identity for an active inference adapter.
//
// Delegated onto inference/lora.AdapterInfo: go-mlx, go-rocm, and go-cpu
// share one adapter-identity shape and one Inspect implementation instead
// of each engine maintaining a byte-identical copy. The alias is a
// zero-cost, wire-identical substitution (same fields, same JSON tags) —
// AdapterInfo{...} literals, IsEmpty(), and Identity() (the bridge into
// state.AdapterIdentity) keep working unchanged for every caller in this
// module, including fuse.go's FuseOptions/FuseProvenance.
type AdapterInfo = ilora.AdapterInfo

// InspectAdapter reads adapter_config.json and hashes adapter files.
//
//	info, err := lora.InspectAdapter("/path/to/adapter")
func InspectAdapter(path string) (AdapterInfo, error) {
	return ilora.InspectAdapter(path)
}

// Inspect reads adapter_config.json at path and records identityPath as the
// user-facing path (which may differ from path when the adapter was staged
// from a Medium).
//
//	info, err := lora.Inspect(stagedPath, originalPath)
func Inspect(path string, identityPath string) (AdapterInfo, error) {
	return ilora.Inspect(path, identityPath)
}

// errResultFailed is the fallback sentinel returned by resultError when
// a core.Result reports !OK but its Value is not an error. Hoisted to a
// package var to avoid allocating on this rare-but-hot helper path.
var errResultFailed = core.NewError("core result failed")

// joinDirChildPattern concatenates a directory path with a relative
// child segment, collapsing the duplicate separator when dir already
// ends in '/'. Skips the filepath.Clean trip core.PathJoin takes; the
// pack directory paths fuse.go feeds in are already canonical (PathAbs +
// MkdirAll output, or caller-supplied non-empty roots validated
// upstream), so the only normalisation needed is the trailing-slash
// collapse rule. An empty dir falls back to a bare child segment to
// preserve PathJoin's "empty root = relative result" semantics.
//
// Lives in adapter.go (universal build) so fuse.go's darwin/arm64-only
// native-fusion path can route through it. Adapter inspection itself no
// longer needs this helper — Inspect/InspectAdapter now delegate to
// inference/lora — but fuse.go (kept byte-identical; see fuse.go's own
// header) still calls it directly, so it stays here rather than moving.
func joinDirChildPattern(dir, child string) string {
	if dir == "" {
		return child
	}
	if dir[len(dir)-1] == '/' {
		return dir + child
	}
	return dir + "/" + child
}

// resultError unwraps a core.Result into a plain error for core.E
// chaining. Kept here (rather than deleted alongside the rest of the old
// local Inspect implementation) because fuse.go's native-fusion path
// still calls it on every core.Result-returning I/O step.
func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errResultFailed
}
