#include "mlx/mlx.h"
#include "mlx/c/array.h"

extern "C" void mlx_array_detach_impl(mlx_array arr) {
    if (arr.ctx) {
        static_cast<mlx::core::array*>(arr.ctx)->detach();
    }
}
