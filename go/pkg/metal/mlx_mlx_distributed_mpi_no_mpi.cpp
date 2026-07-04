#if defined(MLX_ENABLE_DISTRIBUTED) && !MLX_ENABLE_DISTRIBUTED
// MLX distributed support is disabled for this build.
#else
#if defined(__has_include) && __has_include("../../lib/mlx/mlx/distributed/mpi/no_mpi.cpp")
#include "../../lib/mlx/mlx/distributed/mpi/no_mpi.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/distributed/mpi/no_mpi.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
#endif
