add_library(cavefish_kernels OBJECT 
    gemv.hip gemv.h 
    gemv_q8_0.hip gemv_q8_0.h 
    rms_norm.hip rms_norm.h
    residual_add.hip residual_add.h
    silu_mul.hip silu_mul.h
    kv_view.hpp
    rope_table.cpp rope_table.hpp
)
target_link_libraries(cavefish_kernels PRIVATE common)
target_include_directories(cavefish_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})