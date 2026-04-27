#if __has_include("../../lib/mlx/mlx/backend/metal/custom_kernel.cpp")
#include "../../lib/mlx/mlx/backend/metal/custom_kernel.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/backend/metal/custom_kernel.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
