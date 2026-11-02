#include "vector_sum.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void sum_kernel(const float* values, size_t count, float* result) {
    __shared__ float partial_sums[kThreadsPerBlock];

    const unsigned int thread_id = threadIdx.x;
    float thread_sum = 0.0f;
    for (size_t index = thread_id; index < count; index += blockDim.x) {
        thread_sum += values[index];
    }

    partial_sums[thread_id] = thread_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            partial_sums[thread_id] += partial_sums[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        *result = partial_sums[0];
    }
}

} // namespace

float sum_vector_cuda(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }

    float* device_values = nullptr;
    float* device_result = nullptr;
    float result = 0.0f;

    check_cuda(cudaMalloc(&device_values, values.size() * sizeof(float)), "cudaMalloc values");
    try {
        check_cuda(cudaMalloc(&device_result, sizeof(float)), "cudaMalloc result");
        check_cuda(cudaMemcpy(device_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");

        sum_kernel<<<1, kThreadsPerBlock>>>(device_values, values.size(), device_result);
        check_cuda(cudaGetLastError(), "sum_kernel launch");
        check_cuda(cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
    } catch (...) {
        cudaFree(device_result);
        cudaFree(device_values);
        throw;
    }

    check_cuda(cudaFree(device_result), "cudaFree result");
    check_cuda(cudaFree(device_values), "cudaFree values");
    return result;
}

float benchmark_sum_kernel_cuda(const std::vector<float>& values,
                                int repetitions,
                                int warmup_iterations) {
    if (repetitions <= 0 || warmup_iterations < 0) {
        throw std::invalid_argument("repetitions must be positive and warmup_iterations non-negative");
    }
    if (values.empty()) {
        return 0.0f;
    }

    float* device_values = nullptr;
    float* device_result = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float elapsed_ms = 0.0f;

    const auto cleanup = [&]() {
        if (stop_event != nullptr) {
            cudaEventDestroy(stop_event);
        }
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (device_result != nullptr) {
            cudaFree(device_result);
        }
        if (device_values != nullptr) {
            cudaFree(device_values);
        }
    };

    try {
        check_cuda(cudaMalloc(&device_values, values.size() * sizeof(float)), "cudaMalloc values");
        check_cuda(cudaMalloc(&device_result, sizeof(float)), "cudaMalloc result");
        check_cuda(cudaMemcpy(device_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");

        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            sum_kernel<<<1, kThreadsPerBlock>>>(device_values, values.size(), device_result);
        }
        check_cuda(cudaGetLastError(), "sum_kernel warmup launch");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        for (int iteration = 0; iteration < repetitions; ++iteration) {
            sum_kernel<<<1, kThreadsPerBlock>>>(device_values, values.size(), device_result);
        }
        check_cuda(cudaGetLastError(), "sum_kernel benchmark launch");
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return elapsed_ms / repetitions;
}