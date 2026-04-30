#if defined(MLX_ENABLE_DISTRIBUTED) && !MLX_ENABLE_DISTRIBUTED
// MLX distributed support is disabled for this build.
#else
#if defined(__has_include) && __has_include("../../lib/mlx/mlx/distributed/nccl/no_nccl.cpp")
#include "../../lib/mlx/mlx/distributed/nccl/no_nccl.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/distributed/nccl/no_nccl.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
#endif
