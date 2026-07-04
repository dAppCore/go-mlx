// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
)

func inspectModelPackTaskProfiles(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if pack == nil {
		return
	}
	// inspectModelPackArchitecture already resolved + cached the
	// profile pointer (or left it nil for unsupported architectures);
	// consult it directly rather than re-entering
	// LookupArchitectureProfileRef which would just repeat the same
	// negative lookup on every unsupported pack.
	arch := pack.ArchitectureProfile
	if arch == nil {
		return
	}
	if arch.Embeddings {
		embedding := inspectModelPackEmbeddingProfile(pack, root, dir)
		pack.Embedding = &embedding
	}
	if arch.Rerank {
		rerank := inspectModelPackRerankProfile(pack, root, dir)
		pack.Rerank = &rerank
	}
	pack.Capabilities = modelPackCapabilities(pack)
}

func inspectModelPackEmbeddingProfile(pack *mp.ModelPack, root string, dir *modelPackDirIndex) mp.ModelEmbeddingProfile {
	profile := mp.ModelEmbeddingProfile{
		Dimension:         pack.HiddenSize,
		Pooling:           "cls",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root == "" {
		return profile
	}
	if maxSeq, ok := readSentenceBertMaxSequence(root, dir); ok {
		profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
		profile.Source = "sentence-transformers"
	}
	if pooling, ok := readSentenceTransformerPooling(root, dir); ok {
		profile.Pooling = pooling
		profile.Source = "sentence-transformers"
	}
	if normalize, ok := readSentenceTransformerNormalize(root, dir); ok {
		profile.Normalize = normalize
		profile.Source = "sentence-transformers"
	}
	return profile
}

func inspectModelPackRerankProfile(pack *mp.ModelPack, root string, dir *modelPackDirIndex) mp.ModelRerankProfile {
	profile := mp.ModelRerankProfile{
		Method:            "cross-encoder",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root != "" {
		if maxSeq, ok := readSentenceBertMaxSequence(root, dir); ok {
			profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
			profile.Source = "sentence-transformers"
		}
	}
	return profile
}

func readSentenceBertMaxSequence(root string, dir *modelPackDirIndex) (int, bool) {
	if !dir.has("sentence_bert_config.json") {
		return 0, false
	}
	read := core.ReadFile(core.PathJoin(root, "sentence_bert_config.json"))
	if !read.OK {
		return 0, false
	}
	// Hand-rolled single-int walk instead of core.JSONUnmarshal: the
	// stdlib entry point allocates a scanner parse-state stack
	// (checkValid pre-scan) plus a decode-state per call — pure overhead
	// for one int field. parseSingleIntField skips that scan and applies
	// the same strict whole-input contract, so the (value, ok) boundary
	// is byte-identical (malformed body still yields ok=false).
	maxSeq, ok := parseSingleIntField(read.Value.([]byte), "max_seq_length")
	if !ok {
		return 0, false
	}
	return maxSeq, maxSeq > 0
}

func readSentenceTransformerPooling(root string, dir *modelPackDirIndex) (string, bool) {
	// Fast path — the weight glob already listed the "*_Pooling" subdir,
	// so read its config.json by direct path and skip a second
	// directory-tree walk (the `*_Pooling/config.json` glob below costs
	// an os.readdir per Inspect on every sentence-transformers model).
	if dir != nil && dir.populated && dir.poolingDir != "" {
		if pooling, ok := readPoolingConfig(core.PathJoin(root, dir.poolingDir, "config.json")); ok {
			return pooling, true
		}
	}
	// Glob fallback — single-file entry path (no listing) or the rare
	// model that ships more than one "*_Pooling" dir where the first did
	// not resolve. PathGlob (filepath.Glob) returns lexically sorted
	// results, so the explicit sort.Strings was redundant work on every
	// embedding inspect.
	paths := core.PathGlob(core.PathJoin(root, "*_Pooling", "config.json"))
	for _, path := range paths {
		if pooling, ok := readPoolingConfig(path); ok {
			return pooling, true
		}
	}
	return "", false
}

// readPoolingConfig parses a sentence-transformers Pooling config.json
// and maps its mode flags to the canonical pooling name. Returns ok=false
// when the file is unreadable, unparseable, or declares no recognised
// mode — letting the caller fall through to the next candidate.
func readPoolingConfig(path string) (string, bool) {
	read := core.ReadFile(path)
	if !read.OK {
		return "", false
	}
	// Hand-rolled four-bool walk instead of core.JSONUnmarshal — skips the
	// stdlib checkValid parse-state-stack + decode-state allocs for a tiny
	// fixed shape. parsePoolingFlags enforces the same strict whole-input
	// contract, so a malformed body still yields ok=false here.
	cls, mean, max, weightedMean, ok := parsePoolingFlags(read.Value.([]byte))
	if !ok {
		return "", false
	}
	switch {
	case mean:
		return "mean", true
	case cls:
		return "cls", true
	case max:
		return "max", true
	case weightedMean:
		return "weighted_mean", true
	}
	return "", false
}

func readSentenceTransformerNormalize(root string, dir *modelPackDirIndex) (bool, bool) {
	if !dir.has("modules.json") {
		return false, false
	}
	read := core.ReadFile(core.PathJoin(root, "modules.json"))
	if !read.OK {
		return false, false
	}
	// Hand-rolled array walk instead of core.JSONUnmarshal — skips the
	// stdlib checkValid parse-state-stack + decode-state + the
	// []struct{Type,Path} backing slice entirely. modulesDeclareNormalize
	// folds the ASCII-insensitive "normalize" membership test into the
	// walk (decoding each type/path string the same way the reflect path
	// did) so the (value, ok) boundary stays byte-identical: a malformed
	// body yields ok=false, a valid body yields the same found flag.
	return modulesDeclareNormalize(read.Value.([]byte))
}
