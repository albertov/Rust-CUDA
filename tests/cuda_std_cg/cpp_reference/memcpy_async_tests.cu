#include "common.h"
#include "json_output.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Test 1: Basic memcpy_async from global to shared to global
extern "C" __global__ void test_memcpy_async_basic_cpp(const int *src, int *dst,
                                                       size_t size) {
  __shared__ int shared_buffer[256];
  auto block = cg::this_thread_block();

  // Async copy from global to shared
  cg::memcpy_async(block, shared_buffer, src, size * sizeof(int));
  cg::wait(block);

  // Sync copy from shared to global
  if (block.thread_rank() < size) {
    dst[block.thread_rank()] = shared_buffer[block.thread_rank()];
  }
}

// Test 2: Pipeline pattern with double buffering and wait_prior
extern "C" __global__ void test_pipeline_pattern_cpp(const int *src, int *dst,
                                                     int stages) {
  __shared__ int buffer[2][128]; // Double buffering
  auto block = cg::this_thread_block();

  if (stages == 0)
    return;

  // Load first stage
  cg::memcpy_async(block, buffer[0], src, 128 * sizeof(int));
  cg::commit_group();

  for (int i = 1; i < stages; i++) {
    int current = i % 2;
    int prev = (i - 1) % 2;

    // Load next stage
    cg::memcpy_async(block, buffer[current], src + i * 128, 128 * sizeof(int));
    cg::commit_group();

    // Wait for previous stage to complete (all but last 1 group)
    cg::wait_prior<1>(block);

    // Process previous stage
    if (block.thread_rank() < 128) {
      dst[(i - 1) * 128 + block.thread_rank()] =
          buffer[prev][block.thread_rank()];
    }
  }

  // Process final stage
  cg::wait(block);
  int final_idx = (stages - 1) % 2;
  if (block.thread_rank() < 128) {
    dst[(stages - 1) * 128 + block.thread_rank()] =
        buffer[final_idx][block.thread_rank()];
  }
}

// Test 3: Multiple async copies with commit_group batching
extern "C" __global__ void
test_commit_group_batching_cpp(const int *src, int *dst, int batch_count) {
  __shared__ int buffers[4][64]; // 4 buffers of 64 elements
  auto block = cg::this_thread_block();

  // Issue batch_count async copies
  for (int i = 0; i < batch_count && i < 4; i++) {
    cg::memcpy_async(block, buffers[i], src + i * 64, 64 * sizeof(int));
  }

  // Commit all as one group
  cg::commit_group();

  // Wait for all
  cg::wait(block);

  // Copy results back
  for (int i = 0; i < batch_count && i < 4; i++) {
    if (block.thread_rank() < 64) {
      dst[i * 64 + block.thread_rank()] = buffers[i][block.thread_rank()];
    }
  }
}

// Test 4: Large copy (1024 elements)
extern "C" __global__ void test_memcpy_async_large_cpp(const int *src,
                                                       int *dst) {
  __shared__ int shared_buffer[1024];
  auto block = cg::this_thread_block();

  // Async copy 1024 elements (4KB)
  cg::memcpy_async(block, shared_buffer, src, 1024 * sizeof(int));
  cg::wait(block);

  // Copy back in parallel
  if (block.thread_rank() < 1024) {
    dst[block.thread_rank()] = shared_buffer[block.thread_rank()];
  }
}

// Host test harness
int main() {
  print_device_info();

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Check for SM 8.0+ support (required for memcpy_async)
  if (prop.major < 8) {
    fprintf(stderr, "memcpy_async requires SM 8.0+ (Ampere or newer)\n");
    fprintf(stderr, "Current device is SM %d.%d\n", prop.major, prop.minor);
    return 1;
  }

  JSONOutput json("cpp_memcpy_async_tests");

  // Test 1: Basic copy
  {
    const int size = 256;
    int *d_src, *d_dst;
    int h_src[size], h_dst[size];

    for (int i = 0; i < size; i++)
      h_src[i] = i;

    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(int)));
    CUDA_CHECK(
        cudaMemcpy(d_src, h_src, size * sizeof(int), cudaMemcpyHostToDevice));

    test_memcpy_async_basic_cpp<<<1, 256>>>(d_src, d_dst, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_dst, d_dst, size * sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < size; i++) {
      if (h_dst[i] != h_src[i]) {
        passed = false;
        break;
      }
    }

    json.add_test("memcpy_async_basic", passed);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
  }

  // Test 2: Pipeline pattern
  {
    const int stages = 4;
    const int total_size = stages * 128;
    int *d_src, *d_dst;
    int h_src[total_size], h_dst[total_size];

    for (int i = 0; i < total_size; i++)
      h_src[i] = i + 1000;

    CUDA_CHECK(cudaMalloc(&d_src, total_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, total_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, total_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    test_pipeline_pattern_cpp<<<1, 128>>>(d_src, d_dst, stages);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, total_size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < total_size; i++) {
      if (h_dst[i] != h_src[i]) {
        passed = false;
        break;
      }
    }

    json.add_test("pipeline_pattern", passed);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
  }

  // Test 3: Commit group batching
  {
    const int batch_count = 4;
    const int total_size = batch_count * 64;
    int *d_src, *d_dst;
    int h_src[total_size], h_dst[total_size];

    for (int i = 0; i < total_size; i++)
      h_src[i] = i * 2;

    CUDA_CHECK(cudaMalloc(&d_src, total_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, total_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, total_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    test_commit_group_batching_cpp<<<1, 64>>>(d_src, d_dst, batch_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, total_size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < total_size; i++) {
      if (h_dst[i] != h_src[i]) {
        passed = false;
        break;
      }
    }

    json.add_test("commit_group_batching", passed);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
  }

  // Test 4: Large copy
  {
    const int size = 1024;
    int *d_src, *d_dst;
    int h_src[size], h_dst[size];

    for (int i = 0; i < size; i++)
      h_src[i] = i * 3;

    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(int)));
    CUDA_CHECK(
        cudaMemcpy(d_src, h_src, size * sizeof(int), cudaMemcpyHostToDevice));

    test_memcpy_async_large_cpp<<<1, 1024>>>(d_src, d_dst);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_dst, d_dst, size * sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < size; i++) {
      if (h_dst[i] != h_src[i]) {
        passed = false;
        break;
      }
    }

    json.add_test("memcpy_async_large", passed);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
  }

  json.write_to_file("cpp_memcpy_async_results.json");

  return 0;
}
