package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
