// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cuda.h"
#include "mlx/fast.h"

namespace mlx::core {

namespace cu {

bool is_available() {
  return false;
}

} // namespace cu

namespace fast {

CustomKernelFunction cuda_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    int) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

} // namespace fast

} // namespace mlx::core
