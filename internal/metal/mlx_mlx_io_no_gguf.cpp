// GGUF support is enabled for go-mlx, so the real gguf.cpp translation unit is
// compiled. Do not also compile MLX's no_gguf.cpp fallback, which defines the
// same load_gguf/save_gguf symbols.
