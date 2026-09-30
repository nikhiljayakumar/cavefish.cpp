add_library(cavefish_core engine.cpp)
target_include_directories(cavefish_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(kernels)
