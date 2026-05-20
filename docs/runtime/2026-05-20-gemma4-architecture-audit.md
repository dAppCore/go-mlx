<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 Architecture Audit

This note records the implementation check prompted by the Gemma 3/4
architecture review. It is an audit artefact, not production benchmark
evidence.

## Findings

- Hybrid attention is model-driven, not generic LLaMA-style. `Gemma4TextConfig`
  reads `layer_types`; the loader marks each layer as `sliding_attention` or
  `full_attention`, and `Gemma4Model.NewCache` allocates `RotatingKVCache` for
  sliding layers and unbounded `KVCache` for global layers. Fixed-cache context
  replacement preserves the sliding window cap through `replacementCacheMaxSize`.
- The fallback Gemma 4 layer map was wrong. The code used a default pattern of
  `5`, which creates four sliding layers followed by one global layer, and it
  also defaulted missing `num_kv_shared_layers` to `20`. Current Transformers
  defaults are a pattern of `6` for five local layers followed by one global
  layer, a forced final global layer, and `num_kv_shared_layers=0` unless the
  config says otherwise. The fallback path now matches that contract. Current
  cached E2B, E4B, 26B, 31B, and `lthn/lemer-mlx` configs already carry
  explicit `layer_types` and sharing counts, so this patch protects future or
  reduced configs rather than explaining previous benchmark deltas.
- The ratio must stay metadata-driven. The cached E2B 4bit config declares a
  four-sliding/one-full pattern with full layers at indexes
  `4,9,14,19,24,29,34`, while cached E4B and 31B configs declare the
  five-sliding/one-full pattern. The loader therefore preserves explicit
  `layer_types` and uses the fallback pattern only when a config omits them.
- Dual RoPE is already represented. Sliding layers use the `sliding_attention`
  rope parameters, while full layers use `full_attention`; proportional RoPE is
  precomputed into `Gemma4Attention.RopeFreqs` for full-attention layers rather
  than using one unified RoPE base.
- Cross-layer KV sharing is already modelled. `buildGemma4CacheLayout` maps
  shared layers to the most recent owning layer of the same attention type and
  allocates caches only for owners. This matches the current Transformers
  `shared_kv_states[layer_type]` design.
- Gemma 4 RMSNorm should not be changed to Gemma 3's zero-centred `1 + weight`
  convention. Current Transformers `Gemma4RMSNorm` initialises weights to ones
  and multiplies by `weight` directly; the existing go-mlx
  `TestGemma4_PrecomputeNormWeightsUsesDirectScale_Good` covers that direct
  scale path. Gemma 3 remains the `1 + weight` path in this repo.
- Per-layer embeddings are now retained but lazy at load time. The model still
  keeps `embed_tokens_per_layer` arrays alive for the full model lifetime, but
  they are excluded from the initial retained-weight `Materialize` pass so the
  forward path can gather and dequantise only the token rows it needs.

## Remaining Targets

- The `.mp4` state restore path now streams KV blocks and pins raw block bytes,
  but true file-backed mmap into MLX still needs an explicit mapping lifetime
  contract and Metal-aligned payload format.
- Long-context attention remains the measured boundary after the sliding-cache
  fixes; future benchmarks should continue to separate local sliding cache
  storage, full-attention cache storage, restore time, and raw decode.
