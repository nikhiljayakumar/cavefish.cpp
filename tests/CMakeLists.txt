function(add_cavefish_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE common cavefish_kernels cavefish_core)
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

add_cavefish_test(gemv)
add_cavefish_test(gemv_q8_0)
# add_cavefish_test(attention)   # later
# add_cavefish_test(rmsnorm)     # later