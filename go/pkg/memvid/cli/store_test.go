// SPDX-Licence-Identifier: EUPL-1.2

package cli

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

type fakeRunCall struct {
	Bin   string
	Args  []string
	Input string
}

func TestStore_PutResolveSearch_Good(t *testing.T) {
	var calls []fakeRunCall
	runner := func(_ context.Context, input []byte, bin string, args ...string) ([]byte, string, string, error) {
		calls = append(calls, fakeRunCall{Bin: bin, Args: append([]string(nil), args...), Input: string(input)})
		switch args[0] {
		case "put":
			return []byte(`{"memory":{"frame_count":1},"reports":[]}`), "", "", nil
		case "view":
			return []byte(`{"frame":{"id":0,"uri":"mlx://chunk/0","title":"trace","search_text":"fallback","metadata":{"caption":"caption"}},"content":"payload"}`), "", "", nil
		case "find":
			return []byte(`{"hits":[{"rank":1,"score":0.75,"frame_id":0,"uri":"mlx://chunk/0","title":"trace","text":"payload"}]}`), "", "", nil
		default:
			return nil, "", "bad command", core.NewError("bad command")
		}
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	ref, err := store.Put(context.Background(), "payload", memvid.PutOptions{
		URI:   "mlx://chunk/0",
		Title: "trace",
		Tags:  map[string]string{"b": "2", "a": "1"},
	})

	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 0 || ref.Codec != memvid.CodecQRVideo || ref.Segment != "/tmp/trace.mv2" {
		t.Fatalf("Put() ref = %#v", ref)
	}
	chunk, err := store.Resolve(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Text != "payload" || chunk.Ref.FrameOffset != 0 {
		t.Fatalf("Resolve() chunk = %#v", chunk)
	}
	byURI, err := store.ResolveURI(context.Background(), "mlx://chunk/0")
	if err != nil {
		t.Fatalf("ResolveURI() error = %v", err)
	}
	if byURI.Text != "payload" || byURI.Ref.ChunkID != 0 {
		t.Fatalf("ResolveURI() chunk = %#v", byURI)
	}
	hits, err := store.Search(context.Background(), "payload", 3)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) != 1 || hits[0].Chunk.Text != "payload" || hits[0].Score != 0.75 {
		t.Fatalf("Search() hits = %#v", hits)
	}
	if len(calls) < 3 {
		t.Fatalf("calls = %d, want at least 3", len(calls))
	}
	if calls[0].Bin != "/bin/memvid" || calls[0].Input != "payload" {
		t.Fatalf("put call = %#v", calls[0])
	}
	if got := core.Join(" ", calls[0].Args...); !core.Contains(got, "--tag a=1 --tag b=2") {
		t.Fatalf("put args = %q, want sorted tags", got)
	}
}

func TestStore_Open_Bad(t *testing.T) {
	_, err := Open("", WithBinary("/bin/memvid"))

	if err == nil {
		t.Fatal("expected missing path error")
	}
}

func TestStore_LookPathEnv_Good(t *testing.T) {
	t.Setenv(envBinary, " /custom/memvid ")

	path, err := LookPath()
	if err != nil {
		t.Fatalf("LookPath() error = %v", err)
	}
	if path != "/custom/memvid" {
		t.Fatalf("LookPath() = %q, want env binary", path)
	}
	store, err := Open("/tmp/trace.mv2")
	if err != nil {
		t.Fatalf("Open(env binary) error = %v", err)
	}
	if store.Binary() != "/custom/memvid" {
		t.Fatalf("Open(env binary) bin = %q", store.Binary())
	}
}

func TestStore_MissingChunk_Ugly(t *testing.T) {
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return nil, "", "frame was not found", core.NewError("exit 1")
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	_, err = store.Resolve(context.Background(), 99)

	if !core.Is(err, memvid.ErrChunkNotFound) {
		t.Fatalf("Resolve() error = %v, want ErrChunkNotFound", err)
	}
}

func TestStore_ResolveInputErrors_Bad(t *testing.T) {
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return nil, "", "", nil
	}))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if _, err := store.Resolve(context.Background(), -1); !core.Is(err, memvid.ErrChunkNotFound) {
		t.Fatalf("Resolve(negative) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := store.ResolveURI(context.Background(), ""); !core.Is(err, memvid.ErrChunkNotFound) {
		t.Fatalf("ResolveURI(empty) error = %v, want ErrChunkNotFound", err)
	}
}

func TestStore_CreateGetAndAccessors_Good(t *testing.T) {
	var calls []fakeRunCall
	runner := func(_ context.Context, input []byte, bin string, args ...string) ([]byte, string, string, error) {
		calls = append(calls, fakeRunCall{Bin: bin, Args: append([]string(nil), args...), Input: string(input)})
		switch args[0] {
		case "create":
			return []byte(`{"ok":true}`), "", "", nil
		case "view":
			return []byte(`{"frame":{"id":3,"uri":"mlx://chunk/3","search_text":"search text","metadata":{"caption":"caption"}}}`), "", "", nil
		default:
			return nil, "", "bad command", core.NewError("bad command")
		}
	}

	store, err := Create(context.Background(), "/tmp/trace.mv2", WithBinary("/bin/memvid"), WithRawWrites(false), withRunner(runner))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if store.Path() != "/tmp/trace.mv2" || store.Binary() != "/bin/memvid" {
		t.Fatalf("accessors path=%q binary=%q", store.Path(), store.Binary())
	}
	text, err := store.Get(context.Background(), 3)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if text != "caption" {
		t.Fatalf("Get() = %q, want caption fallback", text)
	}
	if len(calls) != 2 || calls[0].Args[0] != "create" || calls[1].Args[0] != "view" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestStore_CreateError_Bad(t *testing.T) {
	_, err := Create(context.Background(), "/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return nil, "", "create failed", core.NewError("exit 1")
	}))

	if err == nil {
		t.Fatal("Create() error = nil, want command failure")
	}
}

func TestStore_PutUsesReportedURIFrame_Good(t *testing.T) {
	runner := func(_ context.Context, _ []byte, _ string, args ...string) ([]byte, string, string, error) {
		switch args[0] {
		case "put":
			return []byte(`{"memory":{"frame_count":10},"reports":[{"uri":"mlx://chunk/new"}]}`), "", "", nil
		case "view":
			return []byte(`{"frame":{"id":7,"uri":"mlx://chunk/new"},"content":"new payload"}`), "", "", nil
		default:
			return nil, "", "bad command", core.NewError("bad command")
		}
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	ref, err := store.Put(context.Background(), "payload", memvid.PutOptions{URI: "mlx://chunk/new"})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 7 || ref.FrameOffset != 7 {
		t.Fatalf("Put() ref = %+v, want frame 7 from URI lookup", ref)
	}
}

func TestStore_PutURIReportViewError_Bad(t *testing.T) {
	runner := func(_ context.Context, _ []byte, _ string, args ...string) ([]byte, string, string, error) {
		switch args[0] {
		case "put":
			return []byte(`{"memory":{"frame_count":10},"reports":[{"uri":"mlx://chunk/new"}]}`), "", "", nil
		case "view":
			return nil, "", "permission denied", core.NewError("exit 1")
		default:
			return nil, "", "bad command", core.NewError("bad command")
		}
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	if _, err := store.Put(context.Background(), "payload", memvid.PutOptions{URI: "mlx://chunk/new"}); err == nil {
		t.Fatal("Put() error = nil, want URI view failure")
	}
}

func TestStore_ReadyAndCommandErrors_Bad(t *testing.T) {
	if (*Store)(nil).Path() != "" || (*Store)(nil).Binary() != "" {
		t.Fatal("nil accessors should return empty strings")
	}
	if err := (*Store)(nil).ready(); err == nil {
		t.Fatal("expected nil store ready error")
	}
	store := &Store{path: "/tmp/trace.mv2"}
	if err := store.ready(); err == nil {
		t.Fatal("expected missing binary error")
	}
	readyStore := &Store{path: "/tmp/trace.mv2", bin: "/bin/memvid"}
	if err := readyStore.ready(); err != nil || readyStore.runner == nil {
		t.Fatalf("ready() = %v runner nil=%v, want default runner", err, readyStore.runner == nil)
	}

	cmdErr := &CommandError{Args: []string{"view"}, Stdout: " out ", Err: errors.New("exit 1")}
	if !core.Contains(cmdErr.Error(), "out") || !errors.Is(cmdErr, cmdErr.Err) {
		t.Fatalf("CommandError = %q unwrap=%v", cmdErr.Error(), errors.Unwrap(cmdErr))
	}
	for _, cmdErr := range []*CommandError{
		{Args: []string{"put"}, Stderr: " err "},
		{Args: []string{"put"}, Err: errors.New("exit 2")},
		{Args: []string{"put"}},
	} {
		if !core.Contains(cmdErr.Error(), "memvid-cli put failed:") {
			t.Fatalf("CommandError.Error() = %q", cmdErr.Error())
		}
	}
	if !commandLooksNotFound(&CommandError{Stdout: "not found"}) {
		t.Fatal("expected commandLooksNotFound(stdout)")
	}
	if commandLooksNotFound(core.NewError("not found")) {
		t.Fatal("plain errors should not be treated as command not found")
	}
	if !isChunkNotFound(&memvid.ChunkNotFoundError{ID: 1}) {
		t.Fatal("expected isChunkNotFound for ChunkNotFoundError")
	}
	builder := core.NewBuilder()
	for range 4100 {
		builder.WriteString("x")
	}
	long := builder.String()
	if got := limitOutput(long); len(got) <= 4096 || !core.Contains(got, "...(truncated)") {
		t.Fatalf("limitOutput(long) len=%d value suffix missing", len(got))
	}
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v, want nil", err)
	}
	var view viewResponse
	view.Frame.SearchText = "search fallback"
	if got := view.text(); got != "search fallback" {
		t.Fatalf("viewResponse.text() = %q, want search fallback", got)
	}
}

func TestStore_RunInputAndParseErrors_Ugly(t *testing.T) {
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(func(ctx context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		<-ctx.Done()
		return nil, "", "", ctx.Err()
	}))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.runInput(ctx, nil, "view"); err == nil {
		t.Fatal("expected context cancellation error")
	}

	badJSONStore, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(func(_ context.Context, _ []byte, _ string, args ...string) ([]byte, string, string, error) {
		return []byte("{broken"), "", "", nil
	}))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if _, err := badJSONStore.Search(context.Background(), "x", 0); err == nil {
		t.Fatal("expected Search JSON parse error")
	}
	if _, err := badJSONStore.Put(context.Background(), "x", memvid.PutOptions{}); err == nil {
		t.Fatal("expected Put JSON parse error")
	}

	emptyPutStore, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"memory":{"frame_count":0},"reports":[]}`), "", "", nil
	}))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if _, err := emptyPutStore.Put(context.Background(), "x", memvid.PutOptions{}); err == nil {
		t.Fatal("expected missing frame id error")
	}
}
