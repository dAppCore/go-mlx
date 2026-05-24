// SPDX-Licence-Identifier: EUPL-1.2

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mdspan>

#include "cgo_pinned_view.hpp"
#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

namespace {

bool checked_mul(size_t lhs, size_t rhs, size_t* out) {
  if (out == nullptr) {
    return false;
  }
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool shape_elements(const int* shape, int dim, size_t* out) {
  if (shape == nullptr || dim <= 0 || out == nullptr) {
    return false;
  }
  size_t total = 1;
  for (int i = 0; i < dim; i++) {
    if (shape[i] <= 0) {
      return false;
    }
    if (!checked_mul(total, static_cast<size_t>(shape[i]), &total)) {
      return false;
    }
  }
  *out = total;
  return true;
}

bool validate_strided_view(
    const void* data,
    size_t storage_elements,
    size_t item_size,
    const int* shape,
    int dim,
    const int64_t* strides,
    int strides_dim,
    size_t offset) {
  if (shape == nullptr || strides == nullptr || dim <= 0 || dim != strides_dim) {
    return false;
  }
  if (offset >= storage_elements) {
    return false;
  }

  size_t max_element = offset;
  for (int i = 0; i < dim; i++) {
    if (shape[i] <= 0 || strides[i] < 0) {
      return false;
    }
    size_t extent = static_cast<size_t>(shape[i]);
    size_t stride = static_cast<size_t>(strides[i]);
    size_t contribution = 0;
    if (!checked_mul(extent - 1, stride, &contribution)) {
      return false;
    }
    if (contribution > std::numeric_limits<size_t>::max() - max_element) {
      return false;
    }
    max_element += contribution;
  }
  if (max_element >= storage_elements) {
    return false;
  }

  if (dim == 4) {
    // Bounds-validate the strided view via cgo_pinned_view's mdspan
    // helper — same construction as the hand-rolled mapping, just
    // routed through the shared substrate so the layout-stride
    // conventions stay single-sourced across go-cgo consumers.
    // Strides are scaled by item_size so the std::byte view walks
    // bytes; the helper takes element-strides (in std::byte that's
    // bytes, so the multiplication is correct).
    auto* base = static_cast<const std::byte*>(data) + offset * item_size;
    auto view = lthn::cgo::pinned_view_4d<const std::byte>(
        base,
        static_cast<size_t>(shape[0]),
        static_cast<size_t>(shape[1]),
        static_cast<size_t>(shape[2]),
        static_cast<size_t>(shape[3]),
        static_cast<std::ptrdiff_t>(strides[0]) * static_cast<std::ptrdiff_t>(item_size),
        static_cast<std::ptrdiff_t>(strides[1]) * static_cast<std::ptrdiff_t>(item_size),
        static_cast<std::ptrdiff_t>(strides[2]) * static_cast<std::ptrdiff_t>(item_size),
        static_cast<std::ptrdiff_t>(strides[3]) * static_cast<std::ptrdiff_t>(item_size));
    const std::byte* first = &view[0, 0, 0, 0];
    const std::byte* last = &view[
        static_cast<size_t>(shape[0] - 1),
        static_cast<size_t>(shape[1] - 1),
        static_cast<size_t>(shape[2] - 1),
        static_cast<size_t>(shape[3] - 1)];
    if (last < first) {
      return false;
    }
    size_t span_bytes = static_cast<size_t>(last - first) + item_size;
    return span_bytes <= (storage_elements - offset) * item_size;
  }
  return true;
}

bool same_contiguous_view(
    const int* storage_shape,
    int storage_dim,
    const int* view_shape,
    int view_dim,
    const int64_t* view_strides,
    int strides_dim,
    size_t offset) {
  if (offset != 0 || storage_dim != view_dim || view_dim != strides_dim) {
    return false;
  }
  int64_t expected = 1;
  for (int i = view_dim - 1; i >= 0; i--) {
    if (storage_shape[i] != view_shape[i] || view_strides[i] != expected) {
      return false;
    }
    expected *= static_cast<int64_t>(view_shape[i]);
  }
  return true;
}

} // namespace

extern "C" mlx_array go_mlx_array_new_pinned_strided_data(
    void* data,
    size_t byte_count,
    const int* storage_shape,
    int storage_dim,
    const int* view_shape,
    int view_dim,
    const int64_t* view_strides,
    int strides_dim,
    size_t view_offset,
    mlx_dtype dtype,
    mlx_stream stream,
    uintptr_t payload_id,
    void (*dtor)(void*)) {
  // payload_id is an opaque uintptr token from the Go side (a counter,
  // not a pointer) — we widen it to void* here because that is what
  // mlx_array_new_data_managed_payload + the dtor expect. Keeping it as
  // uintptr_t in the Go-visible signature lets `go vet`'s unsafeptr
  // check see this is not a Go pointer crossing the boundary.
  void* payload = reinterpret_cast<void*>(payload_id);
  auto release_payload = [&]() {
    if (dtor != nullptr && payload != nullptr) {
      dtor(payload);
      payload = nullptr;
    }
  };

  try {
    if (data == nullptr || byte_count == 0) {
      release_payload();
      mlx_error("mlx: pinned array data is empty");
      return mlx_array_empty;
    }
    size_t item_size = mlx_dtype_size(dtype);
    if (item_size == 0 || byte_count % item_size != 0) {
      release_payload();
      mlx_error("mlx: pinned array byte length does not match dtype");
      return mlx_array_empty;
    }

    size_t storage_elements = 0;
    if (!shape_elements(storage_shape, storage_dim, &storage_elements) ||
        storage_elements * item_size != byte_count) {
      release_payload();
      mlx_error("mlx: pinned array storage shape does not match byte length");
      return mlx_array_empty;
    }
    if (!validate_strided_view(
            data,
            storage_elements,
            item_size,
            view_shape,
            view_dim,
            view_strides,
            strides_dim,
            view_offset)) {
      release_payload();
      mlx_error("mlx: pinned array strided view is out of bounds");
      return mlx_array_empty;
    }

    mlx_array base = mlx_array_new_data_managed_payload(
        data, storage_shape, storage_dim, dtype, payload, dtor);
    if (base.ctx == nullptr) {
      release_payload();
      return mlx_array_empty;
    }
    payload = nullptr;

    if (same_contiguous_view(
            storage_shape,
            storage_dim,
            view_shape,
            view_dim,
            view_strides,
            strides_dim,
            view_offset)) {
      return base;
    }

    mlx_array view = mlx_array_empty;
    if (mlx_as_strided(
            &view,
            base,
            view_shape,
            static_cast<size_t>(view_dim),
            view_strides,
            static_cast<size_t>(strides_dim),
            view_offset,
            stream) != 0) {
      mlx_array_free(base);
      return mlx_array_empty;
    }
    mlx_array_free(base);
    return view;
  } catch (const std::exception& e) {
    release_payload();
    mlx_error(e.what());
    return mlx_array_empty;
  }
}

extern "C" mlx_array go_mlx_array_new_pinned_data(
    void* data,
    size_t byte_count,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    uintptr_t payload_id,
    void (*dtor)(void*)) {
  void* payload = reinterpret_cast<void*>(payload_id);
  auto release_payload = [&]() {
    if (dtor != nullptr && payload != nullptr) {
      dtor(payload);
      payload = nullptr;
    }
  };

  try {
    if (data == nullptr || byte_count == 0) {
      release_payload();
      mlx_error("mlx: pinned array data is empty");
      return mlx_array_empty;
    }
    size_t item_size = mlx_dtype_size(dtype);
    if (item_size == 0 || byte_count % item_size != 0) {
      release_payload();
      mlx_error("mlx: pinned array byte length does not match dtype");
      return mlx_array_empty;
    }

    size_t elements = 0;
    if (!shape_elements(shape, dim, &elements) || elements * item_size != byte_count) {
      release_payload();
      mlx_error("mlx: pinned array shape does not match byte length");
      return mlx_array_empty;
    }

    mlx_array base = mlx_array_new_data_managed_payload(
        data, shape, dim, dtype, payload, dtor);
    if (base.ctx == nullptr) {
      release_payload();
      return mlx_array_empty;
    }
    payload = nullptr;
    return base;
  } catch (const std::exception& e) {
    release_payload();
    mlx_error(e.what());
    return mlx_array_empty;
  }
}
