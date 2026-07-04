# SPDX-Licence-Identifier: EUPL-1.2

option(MLX_USE_CCACHE "Use CCache for compilation cache when available" ON)

if(MLX_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
  else()
    message(STATUS "CCache requested but not found")
  endif()
endif()
