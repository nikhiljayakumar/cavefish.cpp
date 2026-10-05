add_library(cavefish_core kv_cache.cpp kv_cache.h)
target_include_directories(cavefish_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(kernels)
