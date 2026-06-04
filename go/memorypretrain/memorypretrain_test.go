// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestBuildBank_RetrieveRoutesToNearestCluster_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if bank.Dimension != 2 || len(bank.Nodes) < 3 {
		t.Fatalf("bank = %+v, want dimension and child clusters", bank)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 2 || got[0].BlockID != "go-1" || got[1].BlockID != "go-2" {
		t.Fatalf("Retrieve() = %+v, want Go cluster ordered by score", got)
	}
	scratch := make([]Retrieval, 0, 2)
	reused, err := bank.RetrieveInto(scratch, []float32{0, 1}, 2)
	if err != nil {
		t.Fatalf("RetrieveInto() error = %v", err)
	}
	if len(reused) != 2 || reused[0].BlockID != "poem-1" || cap(reused) != cap(scratch) {
		t.Fatalf("RetrieveInto() = %+v cap=%d, want poem cluster in caller storage cap=%d", reused, cap(reused), cap(scratch))
	}
}

func TestBuildBank_ClonesInputAndValidatesDimensions_Bad(t *testing.T) {
	blocks := []Block{{ID: "a", Embedding: []float32{1, 0}, Meta: map[string]string{"source": "unit"}}}
	bank, err := BuildBank(blocks, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	blocks[0].Embedding[0] = 0
	blocks[0].Meta["source"] = "mutated"
	if bank.Blocks[0].Embedding[0] != 1 || bank.Blocks[0].Meta["source"] != "unit" {
		t.Fatalf("bank block aliased input: %+v", bank.Blocks[0])
	}
	if _, err := BuildBank([]Block{{Embedding: []float32{1}}, {Embedding: []float32{1, 2}}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBank() dimension mismatch error = nil")
	}
}

func TestBuildBankFromCorpus_EmbedsRecords_Good(t *testing.T) {
	records := []CorpusRecord{
		{ID: "go", Text: "Go memory planning", Meta: map[string]string{"source": "docs"}},
		{ID: "poem", Text: "winter proof poem", Meta: map[string]string{"source": "creative"}},
	}
	embedder := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
	bank, err := BuildBankFromCorpus(context.Background(), embedder, records, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBankFromCorpus() error = %v", err)
	}
	if bank.Dimension != 2 || len(bank.Blocks) != 2 {
		t.Fatalf("bank dimension=%d blocks=%d, want embedded records", bank.Dimension, len(bank.Blocks))
	}
	records[0].Meta["source"] = "mutated"
	if bank.Blocks[0].ID != "go" || bank.Blocks[0].Text != "Go memory planning" || bank.Blocks[0].Meta["source"] != "docs" {
		t.Fatalf("bank block = %+v, want cloned corpus metadata", bank.Blocks[0])
	}
	got, err := bank.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 1 || got[0].BlockID != "go" {
		t.Fatalf("Retrieve() = %+v, want embedded Go record", got)
	}
}

func TestBuildBankFromCorpus_Validation_Bad(t *testing.T) {
	if _, err := BuildBankFromCorpus(context.Background(), nil, []CorpusRecord{{Text: "x"}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBankFromCorpus(nil embedder) error = nil")
	}
	if _, err := BuildBankFromCorpus(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), nil, BuildConfig{}); err == nil {
		t.Fatal("BuildBankFromCorpus(empty records) error = nil")
	}
	wantErr := errors.New("anchor unavailable")
	if _, err := BuildBankFromCorpus(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, wantErr
	}), []CorpusRecord{{Text: "x"}}, BuildConfig{}); err == nil || !strings.Contains(err.Error(), "embed record 0") {
		t.Fatalf("BuildBankFromCorpus(embed error) error = %v, want record context", err)
	}
	if _, err := (EmbedFunc)(nil).Embed(context.Background(), "x"); err == nil {
		t.Fatal("EmbedFunc(nil).Embed() error = nil")
	}
}

func TestBuildBankFromCorpus_ContextCancelled_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls := 0
	_, err := BuildBankFromCorpus(ctx, EmbedFunc(func(context.Context, string) ([]float32, error) {
		calls++
		return []float32{1}, nil
	}), []CorpusRecord{{Text: "x"}}, BuildConfig{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("BuildBankFromCorpus(cancelled) error = %v, want context.Canceled", err)
	}
	if calls != 0 {
		t.Fatalf("embed calls = %d, want cancellation before embedding", calls)
	}
}

func TestRetrieve_Validation_Ugly(t *testing.T) {
	if _, err := (*Bank)(nil).Retrieve([]float32{1}, 1); err == nil {
		t.Fatal("Retrieve(nil) error = nil")
	}
	bank, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, err := bank.Retrieve([]float32{1}, 1); err == nil {
		t.Fatal("Retrieve(wrong dim) error = nil")
	}
	if _, err := bank.Retrieve([]float32{1, 0}, 0); err == nil {
		t.Fatal("Retrieve(k=0) error = nil")
	}
}

func TestInjectAdditive_AddsRetrievedMemory_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "near", Embedding: []float32{1, 0}},
		{ID: "far", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	hidden := []float32{0.25, 0.5}
	dst := make([]float32, 0, 2)
	scratch := make([]Retrieval, 0, 2)
	out, retrievals, stats, err := bank.InjectAdditive(dst, hidden, []float32{1, 0}, scratch, InjectionConfig{TopK: 1, Scale: 0.5, PositiveScoresOnly: true})
	if err != nil {
		t.Fatalf("InjectAdditive() error = %v", err)
	}
	if len(retrievals) != 1 || retrievals[0].BlockID != "near" {
		t.Fatalf("retrievals = %+v, want nearest memory block", retrievals)
	}
	if !stats.Applied || stats.Retrieved != 1 || stats.Scale != 0.5 {
		t.Fatalf("stats = %+v, want applied injection", stats)
	}
	if len(out) != 2 || out[0] != 0.75 || out[1] != 0.5 || cap(out) != cap(dst) {
		t.Fatalf("out = %+v cap=%d, want hidden plus scaled memory in caller buffer cap=%d", out, cap(out), cap(dst))
	}
}

func TestInjectAdditive_Validation_Bad(t *testing.T) {
	bank, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, _, _, err := bank.InjectAdditive(nil, []float32{1}, []float32{1, 0}, nil, InjectionConfig{TopK: 1}); err == nil {
		t.Fatal("InjectAdditive(hidden dim mismatch) error = nil")
	}
	if _, _, _, err := bank.InjectAdditive(nil, []float32{1, 0}, []float32{1}, nil, InjectionConfig{TopK: 1}); err == nil {
		t.Fatal("InjectAdditive(query dim mismatch) error = nil")
	}
}
