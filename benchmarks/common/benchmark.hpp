#include <hip/hip_runtime.h>
#include <vector>
#include <ranges>
#include <algorithm>
#include <numeric>
#include "common.hpp"

struct BenchmarkResult {
    const int trials;
    double avg_time_ms;
    double median_time_ms;

    BenchmarkResult() : trials(5) {}
};

template <typename Fn>
BenchmarkResult run_benchmark(Fn func){
    // run warm-up
    func();

    BenchmarkResult res;
    hipEvent_t start, end;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    std::vector<float> time_trials(res.trials);
    for (int t = 0; t < res.trials; t++){
        HIP_CHECK(hipEventRecord(start, 0));
        func();
        HIP_CHECK(hipEventRecord(end, 0));
        HIP_CHECK(hipEventSynchronize(end));
        HIP_CHECK(hipEventElapsedTime(&time_trials[t], start, end));
    }
    res.avg_time_ms = static_cast<double>(std::accumulate(time_trials.begin(), time_trials.end(), 0.0) / res.trials);
    std::ranges::sort(time_trials);
    int mid = res.trials / 2;
    if (time_trials.size() % 2 == 0) {
        res.median_time_ms = (time_trials[mid - 1] + time_trials[mid]) / 2.0;
    } else {
        res.median_time_ms = time_trials[mid];
    }
    return res;
}