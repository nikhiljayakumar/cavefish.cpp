#include <hip/hip_runtime.h>
#include <iostream>
#include "system_benchmarks.hpp"
#include "common.hpp"

void run_device_info(){
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "GCN Arch Name: " << prop.gcnArchName << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Total Global Mem (bytes): " << prop.totalGlobalMem << std::endl;
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "Memory Clock Rate (kHz): " << prop.memoryClockRate << std::endl;

    // GDDR6 has 16n prefetch so 16 bits transferred per pin per clock cycle
    double theoretical_bw_gbps =
        (prop.memoryClockRate * 1000.0 * 16.0 * (prop.memoryBusWidth / 8.0)) / 1.0e9;
    std::cout << "Theoretical Peak Bandwidth (GB/s): " << theoretical_bw_gbps << std::endl;
}