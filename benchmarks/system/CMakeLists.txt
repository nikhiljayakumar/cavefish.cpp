add_executable(cavefish_system_benchmark 
    main.cpp
    memory_read.hip
    device_info.hip
    h2d_copy.hip
)
target_link_libraries(cavefish_system_benchmark PRIVATE common benchmark_common)
# add assembly files
#target_compile_options(cavefish_system_benchmark PRIVATE --save-temps)