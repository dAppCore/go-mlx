cmake_minimum_required(VERSION 3.24)
project(go-mlx-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Fetch mlx-c v0.4.1 — same version as the Go side
include(FetchContent)
FetchContent_Declare(
    mlx-c
    GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
    GIT_TAG        v0.4.1
)

set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(mlx-c)
