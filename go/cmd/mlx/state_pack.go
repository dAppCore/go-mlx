// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	trix "forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

const (
	stateKVContainerMagic       = "KVST"
	stateKVContainerContentType = "application/vnd.go-mlx.state-log"
	stateKVContainerKind        = "go-mlx/state-kv"
)

type statePackOptions struct {
	MarkerFile     string
	StateStorePath string
	OutputPath     string
}

type statePackReport struct {
	Version        int                    `json:"version"`
	Magic          string                 `json:"magic"`
	TrixVersion    int                    `json:"trix_version"`
	MarkerFile     string                 `json:"marker_file"`
	StateStorePath string                 `json:"state_store_path"`
	OutputPath     string                 `json:"output_path"`
	PayloadBytes   int64                  `json:"payload_bytes"`
	ContainerBytes int64                  `json:"container_bytes,omitempty"`
	Marker         stateRampFoldMarker    `json:"marker"`
	Header         map[string]interface{} `json:"header,omitempty"`
}

type stateWakeProfileMarkerSource struct {
	Marker        stateRampFoldMarker
	SegmentAlias  string
	PayloadOffset int64
	PayloadBytes  int64
	Cleanup       func()
}

func runStatePackCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("state-pack"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "print JSON report")
	markerFile := fs.String("marker-file", "", "state-ramp-profile report or compact marker JSON")
	stateStorePath := fs.String("state-store", "", "State .mvlog path; defaults to the marker store_path")
	outputPath := fs.String("output", "", "output .kv container path")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s state-pack -marker-file <path> -output <path.kv> [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Pack a State marker + its binary .mvlog payload into a Trix .kv\n")
		core.WriteString(stderr, "container — a single portable file that state-wake-profile (or any\n")
		core.WriteString(stderr, "consumer of the State wake API) can restore in one read. The marker\n")
		core.WriteString(stderr, "file is typically a state-ramp-profile JSON report; the binary\n")
		core.WriteString(stderr, "store path defaults to the store_path the marker records.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Output format: 4-byte magic (KVST) + 1-byte version + 4-byte\n")
		core.WriteString(stderr, "header length + JSON header + raw State payload. Streams the\n")
		core.WriteString(stderr, "payload via io.Copy — no full-file bytes loaded into memory.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.PrintDefaults()
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s state-pack -marker-file ~/runs/state-ramp-r10.json -output ~/sessions/r10.kv\n", name))
		core.WriteString(stderr, core.Sprintf("    # pack the State from a state-ramp-profile run into a portable .kv\n"))
		core.WriteString(stderr, core.Sprintf("  %s state-pack -marker-file ~/marker.json -state-store ~/custom.mvlog -output ~/out.kv\n", name))
		core.WriteString(stderr, core.Sprintf("    # explicit binary store path (overrides what the marker records)\n"))
		core.WriteString(stderr, core.Sprintf("  %s state-pack -json -marker-file ~/m.json -output ~/o.kv\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON report (payload bytes, output path) — for pipelines\n"))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Next: feed the .kv to `state-wake-profile -state-index <path>` to measure\n")
		core.WriteString(stderr, "wake-from-snapshot latency, or to any process that opens the State wake API.\n")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s state-pack: expected no positional arguments\n", cliName()))
		return 2
	}
	if core.Trim(*markerFile) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-pack: marker file is required\n", cliName()))
		return 2
	}
	if core.Trim(*outputPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-pack: output path is required\n", cliName()))
		return 2
	}
	report, err := runStatePack(ctx, statePackOptions{
		MarkerFile:     *markerFile,
		StateStorePath: *stateStorePath,
		OutputPath:     *outputPath,
	})
	if err != nil {
		core.Print(stderr, "%s state-pack: %v", cliName(), err)
		return 1
	}
	if *jsonOutput {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s state-pack: marshal report failed", cliName())
			return 1
		}
		if _, err := stdout.Write(data.Value.([]byte)); err != nil {
			core.Print(stderr, "%s state-pack: write JSON report: %v", cliName(), err)
			return 1
		}
		core.WriteString(stdout, "\n")
		return 0
	}
	core.WriteString(stdout, core.Sprintf("packed %s (%d payload bytes) into %s\n", report.StateStorePath, report.PayloadBytes, report.OutputPath))
	return 0
}

var runStatePack = defaultRunStatePack

func defaultRunStatePack(_ context.Context, opts statePackOptions) (*statePackReport, error) {
	opts.MarkerFile = core.Trim(opts.MarkerFile)
	opts.StateStorePath = core.Trim(opts.StateStorePath)
	opts.OutputPath = core.Trim(opts.OutputPath)
	marker, err := stateWakeProfileCompactMarkerFromFile(opts.MarkerFile)
	if err != nil {
		return nil, err
	}
	if opts.StateStorePath == "" {
		opts.StateStorePath = marker.StorePath
	}
	if opts.StateStorePath == "" {
		return nil, core.NewError("State store path is required")
	}
	stat := core.Stat(opts.StateStorePath)
	if !stat.OK {
		return nil, stat.Value.(error)
	}
	payloadBytes := stat.Value.(core.FsFileInfo).Size()
	header := stateKVContainerHeader(opts, marker, payloadBytes)
	written, err := stateKVContainerEncode(opts.OutputPath, header, opts.StateStorePath)
	if err != nil {
		return nil, err
	}
	report := &statePackReport{
		Version:        1,
		Magic:          stateKVContainerMagic,
		TrixVersion:    trix.Version,
		MarkerFile:     opts.MarkerFile,
		StateStorePath: opts.StateStorePath,
		OutputPath:     opts.OutputPath,
		PayloadBytes:   written,
		Marker:         marker,
		Header:         header,
	}
	if stat := core.Stat(opts.OutputPath); stat.OK {
		report.ContainerBytes = stat.Value.(core.FsFileInfo).Size()
	}
	return report, nil
}

func stateKVContainerHeader(opts statePackOptions, marker stateRampFoldMarker, payloadBytes int64) map[string]interface{} {
	return map[string]interface{}{
		"kind":                 stateKVContainerKind,
		"content_type":         stateKVContainerContentType,
		"payload_file":         core.PathBase(opts.StateStorePath),
		"payload_bytes":        payloadBytes,
		"marker_file":          opts.MarkerFile,
		"state_store_path":     opts.StateStorePath,
		"index_uri":            marker.IndexURI,
		"entry_uri":            marker.EntryURI,
		"bundle_uri":           marker.BundleURI,
		"token_count":          marker.TokenCount,
		"created_at_unix_nano": time.Now().UTC().UnixNano(),
	}
}

func stateKVContainerEncode(outputPath string, header map[string]interface{}, payloadPath string) (int64, error) {
	outputPath = core.Trim(outputPath)
	dir := core.PathDir(outputPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return 0, core.Errorf("create output directory: %v", result.Value)
		}
	}
	payloadFileResult := core.Open(payloadPath)
	if !payloadFileResult.OK {
		return 0, payloadFileResult.Value.(error)
	}
	payloadFile := payloadFileResult.Value.(*core.OSFile)
	defer payloadFile.Close()

	fileResult := core.OpenFile(outputPath, core.O_CREATE|core.O_TRUNC|core.O_WRONLY, 0o600)
	if !fileResult.OK {
		return 0, fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	return trix.EncodeStream(header, stateKVContainerMagic, payloadFile, file)
}

func stateWakeProfileMarkerSourceFromFile(path string) (stateWakeProfileMarkerSource, error) {
	isStateKV, err := stateKVContainerFileHasMagic(path)
	if err != nil {
		return stateWakeProfileMarkerSource{}, err
	}
	if isStateKV {
		return stateKVContainerMarkerSourceFromFile(path)
	}
	read := core.ReadFile(path)
	if !read.OK {
		return stateWakeProfileMarkerSource{}, read.Value.(error)
	}
	data := read.Value.([]byte)
	var payload stateWakeProfileMarkerFile
	if result := core.JSONUnmarshal(data, &payload); !result.OK {
		return stateWakeProfileMarkerSource{}, result.Value.(error)
	}
	marker := stateWakeProfileCompactMarkerFromPayload(payload)
	if marker.IndexURI == "" {
		return stateWakeProfileMarkerSource{}, core.NewError("State compact marker missing store_path or index_uri")
	}
	return stateWakeProfileMarkerSource{Marker: marker}, nil
}

func stateKVContainerFileHasMagic(path string) (bool, error) {
	fileResult := core.Open(path)
	if !fileResult.OK {
		return false, fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()
	var magic [4]byte
	n, err := io.ReadFull(file, magic[:])
	if err != nil {
		if n == 0 || err == io.EOF || err == io.ErrUnexpectedEOF {
			return false, nil
		}
		return false, err
	}
	return string(magic[:]) == stateKVContainerMagic, nil
}

func stateKVContainerMarkerSourceFromFile(containerPath string) (stateWakeProfileMarkerSource, error) {
	fileResult := core.Open(containerPath)
	if !fileResult.OK {
		return stateWakeProfileMarkerSource{}, fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	info, err := trix.ReadHeaderInfo(file, stateKVContainerMagic)
	if err != nil {
		return stateWakeProfileMarkerSource{}, err
	}
	marker, err := stateKVContainerMarkerFromHeader(info.Header, info.PayloadBytes)
	if err != nil {
		return stateWakeProfileMarkerSource{}, err
	}
	segmentAlias := marker.StorePath
	marker.StorePath = containerPath
	return stateWakeProfileMarkerSource{
		Marker:        marker,
		SegmentAlias:  segmentAlias,
		PayloadOffset: info.PayloadOffset,
		PayloadBytes:  info.PayloadBytes,
	}, nil
}

func stateKVContainerMarkerFromHeader(header map[string]interface{}, actualPayloadBytes int64) (stateRampFoldMarker, error) {
	if kind := stateKVHeaderString(header, "kind"); kind != stateKVContainerKind {
		return stateRampFoldMarker{}, core.Errorf("State KV container kind = %q, want %q", kind, stateKVContainerKind)
	}
	if contentType := stateKVHeaderString(header, "content_type"); contentType != stateKVContainerContentType {
		return stateRampFoldMarker{}, core.Errorf("State KV content type = %q, want %q", contentType, stateKVContainerContentType)
	}
	if expectedPayloadBytes := stateKVHeaderInt64(header, "payload_bytes"); expectedPayloadBytes > 0 && expectedPayloadBytes != actualPayloadBytes {
		return stateRampFoldMarker{}, core.Errorf("State KV payload bytes = %d, want %d", actualPayloadBytes, expectedPayloadBytes)
	}
	marker := stateRampFoldMarker{
		StorePath:  stateKVHeaderString(header, "state_store_path"),
		IndexURI:   stateKVHeaderString(header, "index_uri"),
		EntryURI:   stateKVHeaderString(header, "entry_uri"),
		BundleURI:  stateKVHeaderString(header, "bundle_uri"),
		TokenCount: int(stateKVHeaderInt64(header, "token_count")),
	}
	if marker.IndexURI == "" {
		return stateRampFoldMarker{}, core.NewError("State KV container missing index_uri")
	}
	return marker, nil
}

func stateKVHeaderString(header map[string]interface{}, key string) string {
	value, ok := header[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return text
}

func stateKVHeaderInt64(header map[string]interface{}, key string) int64 {
	value, ok := header[key]
	if !ok {
		return 0
	}
	switch n := value.(type) {
	case int:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	default:
		return 0
	}
}
