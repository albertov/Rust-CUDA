#include "common.h"
#include "json_output.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * Basic grid synchronization test kernel.
 *
 * Tests a simple two-phase pattern:
 * 1. Each block increments its own counter atomically
 * 2. Grid-wide synchronization
 * 3. Thread 0 verifies all counters are visible and equal to 1
 *
 * This validates that grid.sync() ensures memory visibility across all blocks.
 */
extern "C" __global__ void grid_sync_basic_kernel(int *counters,
                                                  int num_blocks) {
  cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Phase 1: Each block increments its own counter
  if (threadIdx.x == 0) {
    atomicAdd(&counters[blockIdx.x], 1);
  }

  // Synchronize all blocks - ensure all increments are visible
  grid.sync();

  // Phase 2: Verify all counters are visible with value 1
  if (tid == 0) {
    for (int i = 0; i < num_blocks; i++) {
      int value = counters[i];
      if (value != 1) {
        printf("ERROR: Counter[%d] = %d (expected 1)\n", i, value);
      }
    }
  }
}

int main() {
  // Check device capabilities
  if (!check_cooperative_launch_support()) {
    return EXIT_FAILURE;
  }

  print_device_info();

  int num_blocks = 8;
  int threads_per_block = 256;

  // Allocate device memory
  int *d_counters;
  CUDA_CHECK(cudaMalloc(&d_counters, num_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_counters, 0, num_blocks * sizeof(int)));

  // Launch cooperatively
  void *args[] = {&d_counters, &num_blocks};

  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)grid_sync_basic_kernel,
                                         dim3(num_blocks),
                                         dim3(threads_per_block), args,
                                         0, // shared memory
                                         0  // stream
                                         ));

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  int *h_counters = new int[num_blocks];
  CUDA_CHECK(cudaMemcpy(h_counters, d_counters, num_blocks * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Verify results
  bool success = true;
  for (int i = 0; i < num_blocks; i++) {
    if (h_counters[i] != 1) {
      success = false;
      fprintf(stderr, "VERIFICATION FAILED: Counter[%d] = %d (expected 1)\n", i,
              h_counters[i]);
    }
  }

  // Output JSON
  printf("{\n");
  json_print_field_string("test", "grid_sync_basic");
  json_print_field_int("blocks", num_blocks);
  json_print_field_int("threads_per_block", threads_per_block);
  json_print_array_int("counters", h_counters, num_blocks);
  printf(",\n");
  json_print_field_int("expected", 1);
  json_print_field_bool("success", success, true);
  printf("}\n");

  // Cleanup
  delete[] h_counters;
  CUDA_CHECK(cudaFree(d_counters));

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
