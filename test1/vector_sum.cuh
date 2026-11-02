#pragma once

#include <vector>

float sum_vector_cuda(const std::vector<float>& values);

float benchmark_sum_kernel_cuda(const std::vector<float>& values,
								int repetitions = 1000,
								int warmup_iterations = 20);