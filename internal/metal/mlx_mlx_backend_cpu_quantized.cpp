#if __has_include("../../lib/mlx/mlx/backend/cpu/quantized.cpp")
#include "../../lib/mlx/mlx/backend/cpu/quantized.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/backend/cpu/quantized.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
