cmake_minimum_required(VERSION 3.21)
project(cavefish.cpp LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_HIP_ARCHITECTURES "gfx1030")

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(hip REQUIRED)
add_compile_definitions(__HIP_PLATFORM_AMD__)

add_subdirectory(common)
add_subdirectory(src)
add_subdirectory(benchmarks)
add_subdirectory(cli)
add_subdirectory(tests)