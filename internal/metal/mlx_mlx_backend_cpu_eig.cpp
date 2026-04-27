#if __has_include("../../lib/mlx/mlx/backend/cpu/eig.cpp")
#include "../../lib/mlx/mlx/backend/cpu/eig.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/backend/cpu/eig.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
