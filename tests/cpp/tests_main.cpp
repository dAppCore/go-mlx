// SPDX-Licence-Identifier: EUPL-1.2
//
// doctest entry point for the go-mlx native-kernel test target.
//
// This is a go-mlx-OWNED test harness (not the vendored lib/mlx/tests suite):
// it exercises the custom Metal kernels go-mlx adds on top of upstream MLX —
// the fused/compiled decode and fused-quantised lm-head bridges under
// go/pkg/metal/*_bridge.cpp. The vendored MLX primitives are out of scope here
// (they have their own suite in lib/mlx/tests).
//
// Mirrors lib/mlx/tests/tests.cpp: pick the GPU device by default (the kernels
// are Metal-only), honour DEVICE=cpu for the bits that can run on CPU.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdlib>
#include <string>

#include "mlx/c/error.h"
#include "mlx/compile_impl.h"  // detail::compile_clear_cache (teardown order)
#include "mlx/mlx.h"

using namespace mlx::core;

// The bridges report contract violations by calling mlx_error() and returning
// non-zero. mlx-c's DEFAULT error handler aborts the process — which would kill
// the test on the first error-path case. We install a non-aborting handler so
// the bridge's return code is what the tests assert on (a thrown contract error
// is an expected outcome, not a crash). The message is discarded; the rc is the
// signal.
static void noabort_error_handler(const char* /*msg*/, void* /*data*/) {}

int main(int argc, char** argv) {
  mlx_set_error_handler(noabort_error_handler, nullptr, nullptr);

  doctest::Context context;

  const char* device = std::getenv("DEVICE");
  if (device != nullptr && std::string(device) == "cpu") {
    set_default_device(Device::cpu);
  } else if (is_available(Device::gpu)) {
    set_default_device(Device::gpu);
  }

  context.applyCommandLine(argc, argv);
  const int res = context.run();

  // The compiled decode/paged graphs (decode_bridge.cpp) register entries in
  // MLX's GLOBAL detail::CompilerCache — including the paged path's per-shape
  // std::map of compiled functions. At process exit that cache's destructor
  // runs against already-torn-down MLX globals and null-derefs inside
  // __hash_table::__erase_unique (observed: EXC_BAD_ACCESS at 0x0). Clearing
  // the cache HERE, while MLX is still alive, makes teardown deterministic.
  // This is an exit-order fix only — every test above has already run and
  // passed; no kernel is touched. (Vendored MLX programs clear the cache the
  // same way; the cgo runtime never hits this because the statics live for the
  // whole process and Go's exit path differs.)
  detail::compile_clear_cache();

  return res;
}
