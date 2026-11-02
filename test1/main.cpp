#include "vector_sum.cuh"

#include <cstdio>
#include <numeric>
#include <vector>

int main() {
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    constexpr int benchmark_iterations = 1000;
    const float gpu_sum = sum_vector_cuda(values);
    const float expected_sum = std::accumulate(values.begin(), values.end(), 0.0f);
    const float average_kernel_ms = benchmark_sum_kernel_cuda(values, benchmark_iterations);

    std::printf("Suma obliczona na GPU: %.1f\n", gpu_sum);
    std::printf("Suma oczekiwana:       %.1f\n", expected_sum);
    std::printf("Sredni czas kernela:   %.6f ms (%d powtorzen)\n",
                average_kernel_ms,
                benchmark_iterations);

    return gpu_sum == expected_sum ? 0 : 1;
}