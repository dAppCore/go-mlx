// SPDX-Licence-Identifier: EUPL-1.2

package cli

import (
	"context"
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
