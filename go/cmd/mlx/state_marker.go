// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	"dappco.re/go/inference/state/agent"
)

// The session-state compact-marker helpers below were recovered from the deleted
// state-wake-profile bench command, which had co-located them with its profiler.
// They are real session-state code: state-pack reads a compact marker (a pointer
// to where a folded session's KV state lives) to pack that state into a portable
// KV container. stateRampFoldMarker itself lives in main.go.

// stateWakeProfileMarkerFile is the on-disk JSON a compact/fold marker is read
// from — either a flat marker (store_path + index_uri) or a nested fold.
type stateWakeProfileMarkerFile struct {
	StorePath string                      `json:"store_path,omitempty"`
	IndexURI  string                      `json:"index_uri,omitempty"`
	EntryURI  string                      `json:"entry_uri,omitempty"`
	BundleURI string                      `json:"bundle_uri,omitempty"`
	Fold      *stateWakeProfileMarkerFold `json:"fold,omitempty"`
}

// stateWakeProfileMarkerFold is the nested form: an explicit compact marker, or
// a folded sleep report to derive one from.
type stateWakeProfileMarkerFold struct {
	StorePath     string               `json:"store_path,omitempty"`
	CompactMarker *stateRampFoldMarker `json:"compact_marker,omitempty"`
	Folded        *agent.SleepReport   `json:"folded,omitempty"`
}

// stateWakeProfileCompactMarkerFromFile reads a marker file and resolves it to a
// compact marker, erroring if neither a flat marker nor a fold yields an index.
func stateWakeProfileCompactMarkerFromFile(path string) (stateRampFoldMarker, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return stateRampFoldMarker{}, read.Value.(error)
	}
	var payload stateWakeProfileMarkerFile
	if result := core.JSONUnmarshal(read.Value.([]byte), &payload); !result.OK {
		return stateRampFoldMarker{}, result.Value.(error)
	}
	if marker := stateWakeProfileCompactMarkerFromPayload(payload); marker.IndexURI != "" {
		return marker, nil
	}
	return stateRampFoldMarker{}, core.NewError("State compact marker missing store_path or index_uri")
}

// stateWakeProfileCompactMarkerFromPayload derives a compact marker from a parsed
// marker file: a flat marker wins, else an explicit fold marker, else a folded
// sleep report.
func stateWakeProfileCompactMarkerFromPayload(payload stateWakeProfileMarkerFile) stateRampFoldMarker {
	if payload.IndexURI != "" {
		return stateRampFoldMarker{
			StorePath: payload.StorePath,
			IndexURI:  payload.IndexURI,
			EntryURI:  payload.EntryURI,
			BundleURI: payload.BundleURI,
		}
	}
	if payload.Fold == nil {
		return stateRampFoldMarker{}
	}
	if marker := payload.Fold.CompactMarker; marker != nil && marker.IndexURI != "" {
		return *marker
	}
	if payload.Fold.Folded == nil || payload.Fold.Folded.IndexURI == "" {
		return stateRampFoldMarker{}
	}
	return stateRampFoldMarker{
		StorePath:  payload.Fold.StorePath,
		IndexURI:   payload.Fold.Folded.IndexURI,
		EntryURI:   payload.Fold.Folded.EntryURI,
		BundleURI:  payload.Fold.Folded.BundleURI,
		TokenCount: payload.Fold.Folded.TokenCount,
	}
}
