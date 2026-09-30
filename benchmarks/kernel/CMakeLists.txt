function(add_cavefish_kernel_benchmark name)
    add_executable(bench_${name} bench_${name}.cpp)
    target_link_libraries(bench_${name} PRIVATE benchmark_common common cavefish_kernels cavefish_core)
endfunction()

add_cavefish_kernel_benchmark(gemv)
add_cavefish_kernel_benchmark(gemv_q8_0)