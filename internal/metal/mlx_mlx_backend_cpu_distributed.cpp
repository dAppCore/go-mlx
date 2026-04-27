#if __has_include("../../lib/mlx/mlx/backend/cpu/distributed.cpp")
#include "../../lib/mlx/mlx/backend/cpu/distributed.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/backend/cpu/distributed.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
