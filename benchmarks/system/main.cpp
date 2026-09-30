#include "system_benchmarks.hpp"
#include <iostream>

int main(){
    std::cout << "------ Running Memory Read Benchmark ------\n";
    run_memory_read_benchmark();
    std::cout << "------ Running Device Info ------\n";
    run_device_info();
    std::cout << "------ Running Host to Device Copy Benchmark ------\n";
    run_h2d_copy_benchmark();
}