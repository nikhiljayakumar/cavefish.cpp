add_library(cavefish_kernels OBJECT gemv.hip gemv.h gemv_q8_0.hip gemv_q8_0.h)
target_link_libraries(cavefish_kernels PRIVATE common)
target_include_directories(cavefish_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})