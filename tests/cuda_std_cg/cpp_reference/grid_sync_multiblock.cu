#include "common.h"
#include "json_output.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * Multi-block coordination test kernel.
 *
 * Tests grid synchronization with varying block counts (2, 4, 8, 16 blocks).
 * Each test:
 * 1. Each thread writes its block ID to a unique array position
 * 2. Grid-wide synchronization
 * 3. All threads read from all array positions and compute sum
 * 4. Block 0 thread 0 accumulates global sum for verification
 *
 * This validates that grid.sync() works correctly regardless of block count
 * and that memory writes from all blocks are visible after synchronization.
 */
extern "C" __global__
__launch_bounds__(256, 1) void grid_sync_multiblock_kernel(int *data,
                                                           int array_size,
                                                           int *global_sum) {
  cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_size = gridDim.x * blockDim.x;

  // Phase 1: Write unique block ID to array positions
  for (int i = tid; i < array_size; i += grid_size) {
    data[i] = blockIdx.x;
  }

  // Sync: Ensure all writes are visible to all threads
  grid.sync();

  // Phase 2: Read all values and compute local sum
  int local_sum = 0;
  for (int i = tid; i < array_size; i += grid_size) {
    local_sum += data[i];
  }

  // Each block contributes its partial sum
  __shared__ int block_sum;
  if (threadIdx.x == 0) {
    block_sum = 0;
  }
  __syncthreads();

  atomicAdd(&block_sum, local_sum);
  __syncthreads();

  // Block 0 accumulates the global sum
  if (threadIdx.x == 0) {
    atomicAdd(global_sum, block_sum);
  }
}

int main() {
  // Check device capabilities
  if (!check_cooperative_launch_support()) {
    return EXIT_FAILURE;
  }

  print_device_info();

  bool overall_success = true;

  printf("[\n");

  // Test with different block counts
  int block_counts[] = {2, 4, 8, 16};
  int num_tests = sizeof(block_counts) / sizeof(block_counts[0]);

  for (int test_idx = 0; test_idx < num_tests; test_idx++) {
    int num_blocks = block_counts[test_idx];
    int threads_per_block = 256;
    int array_size = num_blocks * threads_per_block;

    // Allocate device memory
    int *d_data;
    int *d_global_sum;
    CUDA_CHECK(cudaMalloc(&d_data, array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global_sum, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, array_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(int)));

    // Launch cooperatively
    void *args[] = {&d_data, &array_size, &d_global_sum};

    CUDA_CHECK(cudaLaunchCooperativeKernel((void *)grid_sync_multiblock_kernel,
                                           dim3(num_blocks),
                                           dim3(threads_per_block), args));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    int *h_data = new int[array_size];
    int h_global_sum;
    CUDA_CHECK(cudaMemcpy(h_data, d_data, array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_global_sum, d_global_sum, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Calculate expected sum: sum of block IDs across all positions
    // Each position should contain its block ID
    int expected_sum = 0;
    for (int i = 0; i < array_size; i++) {
      int expected_block_id = i / threads_per_block;
      expected_sum += expected_block_id;
    }

    bool success = (h_global_sum == expected_sum);
    if (!success) {
      fprintf(stderr, "VERIFICATION FAILED: blocks=%d, sum=%d, expected=%d\n",
              num_blocks, h_global_sum, expected_sum);
      overall_success = false;
    }

    // Output JSON for this test
    printf("  {\n");
    json_print_field_int("blocks", num_blocks);
    json_print_field_int("threads_per_block", threads_per_block);
    json_print_field_int("array_size", array_size);
    json_print_field_int("global_sum", h_global_sum);
    json_print_field_int("expected_sum", expected_sum);
    json_print_field_bool("success", success, true);
    printf("  }%s\n", (test_idx < num_tests - 1) ? "," : "");

    // Cleanup
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_global_sum));
  }

  printf("]\n");

  return overall_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
