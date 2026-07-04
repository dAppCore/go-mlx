#include <exception>
#include <string>
#include <unordered_map>

#include "mlx/c/error.h"
#include "mlx/c/map.h"
#include "mlx/c/private/mlx.h"
#include "mlx/io.h"

extern "C" int mlx_load_gguf_arrays(
    mlx_map_string_to_array* res,
    const char* file,
    const mlx_stream s) {
  try {
    auto [weights, metadata] =
        mlx::core::load_gguf(std::string(file), mlx_stream_get_(s));
    (void)metadata;
    mlx_map_string_to_array_set_(*res, weights);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_save_gguf_arrays(
    const char* file,
    const mlx_map_string_to_array param) {
  try {
    mlx::core::save_gguf(
        std::string(file),
        mlx_map_string_to_array_get_(param),
        std::unordered_map<std::string, mlx::core::GGUFMetaData>{});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
