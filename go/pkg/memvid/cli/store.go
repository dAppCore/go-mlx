// SPDX-Licence-Identifier: EUPL-1.2

package cli

import (
	"context"
	"os/exec"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

const envBinary = "MEMVID_CLI_BIN"

var (
	errNilStore       = core.NewError("memvid cli store is nil")
	errPathRequired   = core.NewError("memvid cli store path is required")
	errBinaryRequired = core.NewError("memvid cli binary is required")
	errNoFrameID      = core.NewError("memvid put did not report a frame id")
	errResultFailed   = core.NewError("core result failed")
)

type Store struct {
	path      string
	bin       string
	rawWrites bool
	runner    commandRunner
}

type Option func(*Store)

type commandRunner func(ctx context.Context, input []byte, bin string, args ...string) ([]byte, string, string, error)

func WithBinary(path string) Option {
	return func(s *Store) {
		s.bin = path
	}
}

func WithRawWrites(enabled bool) Option {
	return func(s *Store) {
		s.rawWrites = enabled
	}
}

func withRunner(runner commandRunner) Option {
	return func(s *Store) {
		s.runner = runner
	}
}

type PutOptions = memvid.PutOptions

type SearchHit struct {
	Rank  int
	Score float64
	URI   string
	Title string
	Chunk memvid.Chunk
}

type CommandError struct {
	Args   []string
	Stdout string
	Stderr string
	Err    error
}

func (e *CommandError) Error() string {
	detail := core.Trim(e.Stderr)
	if detail == "" {
		detail = core.Trim(e.Stdout)
	}
	if detail == "" && e.Err != nil {
		detail = e.Err.Error()
	}
	if detail == "" {
		detail = "unknown error"
	}
	// Single-Builder build: avoids the intermediate Join allocation
	// that the previous Concat(prefix, Join, suffix, detail) form
	// produced. Pre-size to the exact final length so the underlying
	// buffer never grows. 2 allocs → 1 alloc on the hot error path.
	const prefix = "memvid-cli "
	const suffix = " failed: "
	n := len(prefix) + len(suffix) + len(detail)
	if argc := len(e.Args); argc > 0 {
		n += argc - 1
		for _, a := range e.Args {
			n += len(a)
		}
	}
	b := core.NewBuilder()
	b.Grow(n)
	b.WriteString(prefix)
	for i, a := range e.Args {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(a)
	}
	b.WriteString(suffix)
	b.WriteString(detail)
	return b.String()
}

func (e *CommandError) Unwrap() error {
	return e.Err
}

func LookPath() (string, error) {
	if path := core.Trim(core.Env(envBinary)); path != "" {
		return path, nil
	}
	path, err := exec.LookPath("memvid")
	if err != nil {
		return "", core.Errorf("memvid-cli not found: install memvid or set %s: %w", envBinary, err)
	}
	return path, nil
}

func Open(path string, opts ...Option) (*Store, error) {
	if core.Trim(path) == "" {
		return nil, errPathRequired
	}
	store := &Store{
		path:      path,
		rawWrites: true,
		runner:    defaultRunner,
	}
	for _, opt := range opts {
		opt(store)
	}
	if core.Trim(store.bin) == "" {
		bin, err := LookPath()
		if err != nil {
			return nil, err
		}
		store.bin = bin
	}
	return store, nil
}

func Create(ctx context.Context, path string, opts ...Option) (*Store, error) {
	store, err := Open(path, opts...)
	if err != nil {
		return nil, err
	}
	if _, err := store.run(ctx, "create", store.path); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *Store) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

func (s *Store) Binary() string {
	if s == nil {
		return ""
	}
	return s.bin
}

func (s *Store) Get(ctx context.Context, chunkID int) (string, error) {
	// Resolve builds a full Chunk just so we can read .Text; viewFrame
	// returns the underlying viewResponse directly. Skip the Chunk +
	// ChunkRef construction entirely on the Get path.
	view, err := s.viewFrame(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return view.text(), nil
}

func (s *Store) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	view, err := s.viewFrame(ctx, chunkID)
	if err != nil {
		return memvid.Chunk{}, err
	}
	// chunkID is the caller's authority — view.Frame.ID is what the
	// store happens to have returned, but the contract is "the chunk
	// you asked for". If they disagree the store is wrong, not the
	// caller; carry the asked-for ID through to the Chunk.Ref so
	// downstream code matches the user's mental model. (The frame
	// offset still carries view.Frame.ID — that's the on-disk seek
	// hint, separate concern.)
	return memvid.Chunk{
		Ref: memvid.ChunkRef{
			ChunkID:        chunkID,
			FrameOffset:    view.Frame.ID,
			HasFrameOffset: true,
			Codec:          memvid.CodecQRVideo,
			Segment:        s.path,
		},
		Text: view.text(),
	}, nil
}

func (s *Store) ResolveURI(ctx context.Context, uri string) (memvid.Chunk, error) {
	if core.Trim(uri) == "" {
		return memvid.Chunk{}, &memvid.URIChunkNotFoundError{URI: uri}
	}
	view, err := s.viewURI(ctx, uri)
	if err != nil {
		return memvid.Chunk{}, err
	}
	return memvid.Chunk{
		Ref: memvid.ChunkRef{
			ChunkID:        int(view.Frame.ID),
			FrameOffset:    view.Frame.ID,
			HasFrameOffset: true,
			Codec:          memvid.CodecQRVideo,
			Segment:        s.path,
		},
		Text: view.text(),
	}, nil
}

func (s *Store) Put(ctx context.Context, text string, opts memvid.PutOptions) (memvid.ChunkRef, error) {
	if err := s.ready(); err != nil {
		return memvid.ChunkRef{}, err
	}
	// 5 fixed flags + worst-case option flags (1 raw + 2 per uri/title/
	// kind/track + 2 per tag + 2 per label). Pre-sized so subsequent
	// appends never grow the backing array.
	args := make([]string, 0, 14+2*(len(opts.Tags)+len(opts.Labels)))
	args = append(args, "put", s.path, "--json", "--no-embedding", "--no-enrich")
	if s.rawWrites {
		args = append(args, "--raw")
	}
	if opts.URI != "" {
		args = append(args, "--uri", opts.URI)
	}
	if opts.Title != "" {
		args = append(args, "--title", opts.Title)
	}
	if opts.Kind != "" {
		args = append(args, "--kind", opts.Kind)
	}
	if opts.Track != "" {
		args = append(args, "--track", opts.Track)
	}
	if len(opts.Tags) > 0 {
		keys := make([]string, 0, len(opts.Tags))
		for key := range opts.Tags {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		for _, key := range keys {
			args = append(args, "--tag", key+"="+opts.Tags[key])
		}
	}
	for _, label := range opts.Labels {
		args = append(args, "--label", label)
	}

	// Zero-copy view of text — runInput passes the bytes through
	// core.NewBuffer into cmd.Stdin which only reads from them. text
	// outlives the synchronous cmd.Run inside defaultRunner, and the
	// caller's payload is never mutated, so the view is safe.
	out, err := s.runInput(ctx, core.AsBytes(text), args...)
	if err != nil {
		return memvid.ChunkRef{}, err
	}
	var put putResponse
	if r := core.JSONUnmarshal(out, &put); !r.OK {
		return memvid.ChunkRef{}, core.E("memvid.Store.Put", "parse memvid put JSON", resultError(r))
	}
	id, err := s.putFrameID(ctx, put)
	if err != nil {
		return memvid.ChunkRef{}, err
	}
	return memvid.ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          memvid.CodecQRVideo,
		Segment:        s.path,
	}, nil
}

func (s *Store) Search(ctx context.Context, query string, topK int) ([]SearchHit, error) {
	if err := s.ready(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		topK = 8
	}
	out, err := s.run(ctx,
		"find", s.path,
		"--query", query,
		"--top-k", core.Itoa(topK),
		"--json",
		"--mode", "lex",
		"--no-adaptive",
	)
	if err != nil {
		return nil, err
	}
	var found findResponse
	if r := core.JSONUnmarshal(out, &found); !r.OK {
		return nil, core.E("memvid.Store.Search", "parse memvid find JSON", resultError(r))
	}
	hits := make([]SearchHit, 0, len(found.Hits))
	// Index iteration avoids the per-iter struct copy of the response
	// hit (6 fields, 56 bytes) — load-bearing when topK is large and
	// Search is on the per-query hot path.
	for i := range found.Hits {
		hit := &found.Hits[i]
		chunk, err := s.Resolve(ctx, int(hit.FrameID))
		if err != nil {
			return nil, err
		}
		if chunk.Text == "" {
			chunk.Text = hit.Text
		}
		hits = append(hits, SearchHit{
			Rank:  hit.Rank,
			Score: hit.Score,
			URI:   hit.URI,
			Title: hit.Title,
			Chunk: chunk,
		})
	}
	return hits, nil
}

func (s *Store) putFrameID(ctx context.Context, put putResponse) (int, error) {
	// Index iteration; report struct is small but the pattern matches
	// the rest of this package and avoids an unnecessary 16-byte copy
	// each iteration.
	for i := range put.Reports {
		uri := put.Reports[i].URI
		if uri == "" {
			continue
		}
		view, err := s.viewURI(ctx, uri)
		if err == nil {
			return int(view.Frame.ID), nil
		}
		if !isChunkNotFound(err) {
			return 0, err
		}
	}
	if put.Memory.FrameCount > 0 {
		return int(put.Memory.FrameCount - 1), nil
	}
	return 0, errNoFrameID
}

func (s *Store) viewFrame(ctx context.Context, chunkID int) (viewResponse, error) {
	if chunkID < 0 {
		return viewResponse{}, &memvid.ChunkNotFoundError{ID: chunkID}
	}
	return s.view(ctx, "--frame-id", core.Itoa(chunkID), chunkID)
}

func (s *Store) viewURI(ctx context.Context, uri string) (viewResponse, error) {
	return s.view(ctx, "--uri", uri, 0)
}

func (s *Store) view(ctx context.Context, selector string, value string, chunkID int) (viewResponse, error) {
	if err := s.ready(); err != nil {
		return viewResponse{}, err
	}
	out, err := s.run(ctx, "view", s.path, selector, value, "--json")
	if err != nil {
		if commandLooksNotFound(err) {
			return viewResponse{}, &memvid.ChunkNotFoundError{ID: chunkID}
		}
		return viewResponse{}, err
	}
	var view viewResponse
	if r := core.JSONUnmarshal(out, &view); !r.OK {
		return viewResponse{}, core.E("memvid.Store.Resolve", "parse memvid view JSON", resultError(r))
	}
	return view, nil
}

func (s *Store) run(ctx context.Context, args ...string) ([]byte, error) {
	return s.runInput(ctx, nil, args...)
}

func (s *Store) runInput(ctx context.Context, input []byte, args ...string) ([]byte, error) {
	if err := s.ready(); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	stdout, stdoutText, stderr, err := s.runner(ctx, input, s.bin, args...)
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, &CommandError{
			Args:   core.SliceClone(args),
			Stdout: limitOutput(stdoutText),
			Stderr: limitOutput(stderr),
			Err:    err,
		}
	}
	return stdout, nil
}

func (s *Store) ready() error {
	if s == nil {
		return errNilStore
	}
	if core.Trim(s.path) == "" {
		return errPathRequired
	}
	if core.Trim(s.bin) == "" {
		return errBinaryRequired
	}
	if s.runner == nil {
		s.runner = defaultRunner
	}
	return nil
}

func defaultRunner(ctx context.Context, input []byte, bin string, args ...string) ([]byte, string, string, error) {
	cmd := exec.CommandContext(ctx, bin, args...)
	if input != nil {
		cmd.Stdin = core.NewBuffer(input)
	}
	stdout := core.NewBuffer()
	stderr := core.NewBuffer()
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	// stdoutText is only consumed by the error path (limitOutput). Skip
	// the stdout.String() copy on success — callers use stdout.Bytes()
	// for the payload, and the textual form is never read.
	if err == nil {
		return stdout.Bytes(), "", stderr.String(), nil
	}
	return stdout.Bytes(), stdout.String(), stderr.String(), err
}

func commandLooksNotFound(err error) bool {
	// Direct type assertion: this helper is only ever called with the
	// error returned by Store.run/runInput — that's either *CommandError
	// (unwrapped, freshly constructed) or a context error. errors.As
	// walks the unwrap chain reflectively and boxes the type pointer,
	// which costs an alloc per call; the type assertion is free.
	cmdErr, ok := err.(*CommandError)
	if !ok {
		return false
	}
	// "was not found" contains "not found" — one needle is enough.
	// Lower each stream independently to skip the joined "stdout\nstderr"
	// allocation, and short-circuit the second Lower when stdout matches.
	if core.Contains(core.Lower(cmdErr.Stdout), "not found") {
		return true
	}
	return core.Contains(core.Lower(cmdErr.Stderr), "not found")
}

func isChunkNotFound(err error) bool {
	return core.Is(err, memvid.ErrChunkNotFound)
}

func limitOutput(out string) string {
	const max = 4096
	out = core.Trim(out)
	if len(out) <= max {
		return out
	}
	return out[:max] + "...(truncated)"
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errResultFailed
}

type putResponse struct {
	Memory struct {
		FrameCount uint64 `json:"frame_count"`
	} `json:"memory"`
	Reports []struct {
		URI string `json:"uri"`
	} `json:"reports"`
}

type viewResponse struct {
	Frame struct {
		ID          uint64 `json:"id"`
		URI         string `json:"uri"`
		Title       string `json:"title"`
		SearchText  string `json:"search_text"`
		PayloadSize uint64 `json:"payload_length"`
		Metadata    struct {
			Caption string `json:"caption"`
		} `json:"metadata"`
	} `json:"frame"`
	Content string `json:"content"`
}

// text resolves the chunk payload from the view response, falling
// back through Content → Caption → SearchText. Pointer receiver
// avoids copying the 96-byte viewResponse struct on every Search hit
// (Search calls Resolve N times per query, each call ends in text()).
func (v *viewResponse) text() string {
	if v.Content != "" {
		return v.Content
	}
	if v.Frame.Metadata.Caption != "" {
		return v.Frame.Metadata.Caption
	}
	return v.Frame.SearchText
}

type findResponse struct {
	Hits []struct {
		Rank    int     `json:"rank"`
		Score   float64 `json:"score"`
		FrameID uint64  `json:"frame_id"`
		URI     string  `json:"uri"`
		Title   string  `json:"title"`
		Text    string  `json:"text"`
	} `json:"hits"`
}
