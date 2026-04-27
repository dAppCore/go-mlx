#if __has_include("../../lib/mlx/mlx/backend/metal/logsumexp.cpp")
#include "../../lib/mlx/mlx/backend/metal/logsumexp.cpp"
#else
#error "Missing forwarded source: ../../lib/mlx/mlx/backend/metal/logsumexp.cpp. Initialise submodules with git submodule update --init --recursive or fix the forwarding include path."
#endif
