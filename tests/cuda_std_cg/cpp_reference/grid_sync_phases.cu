#include "common.h"
#include "json_output.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * Multi-phase synchronization test kernel.
 *
 * Tests multiple sequential grid synchronizations in a single kernel launch.
 * Each phase:
 * 1. Each block atomically increments the phase counter
 * 2. Grid-wide synchronization
 * 3. Block 0 verifies the counter equals the number of blocks
 * 4. Grid-wide synchronization before next phase
 *
 * This validates that:
 * - Multiple grid.sync() calls work correctly in sequence
 * - Each phase completes fully before the next begins
 * - Memory visibility is maintained across multiple synchronization points
 */
extern "C" __global__ void grid_sync_phases_kernel(int *phase_counters,
                                                   int num_phases) {
  cg::grid_group grid = cg::this_grid();

  for (int phase = 0; phase < num_phases; phase++) {
    // Each block contributes to the phase counter
    if (threadIdx.x == 0) {
      atomicAdd(&phase_counters[phase], 1);
    }

    // Synchronize: ensure all blocks have incremented
    grid.sync();

    // Block 0 verifies all blocks reached this phase
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      int expected = gridDim.x;
      int actual = phase_counters[phase];
      if (actual != expected) {
        printf("ERROR: Phase %d counter = %d (expected %d)\n", phase, actual,
               expected);
      }
    }

    // Synchronize before next iteration to ensure verification completes
    grid.sync();
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
  int num_phases = 5;

  // Allocate device memory
  int *d_phase_counters;
  CUDA_CHECK(cudaMalloc(&d_phase_counters, num_phases * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_phase_counters, 0, num_phases * sizeof(int)));

  // Launch cooperatively
  void *args[] = {&d_phase_counters, &num_phases};

  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)grid_sync_phases_kernel,
                                         dim3(num_blocks),
                                         dim3(threads_per_block), args));

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  int *h_phase_counters = new int[num_phases];
  CUDA_CHECK(cudaMemcpy(h_phase_counters, d_phase_counters,
                        num_phases * sizeof(int), cudaMemcpyDeviceToHost));

  // Verify results
  bool success = true;
  for (int i = 0; i < num_phases; i++) {
    if (h_phase_counters[i] != num_blocks) {
      success = false;
      fprintf(stderr,
              "VERIFICATION FAILED: Phase %d counter = %d (expected %d)\n", i,
              h_phase_counters[i], num_blocks);
    }
  }

  // Output JSON
  printf("{\n");
  json_print_field_string("test", "grid_sync_phases");
  json_print_field_int("blocks", num_blocks);
  json_print_field_int("threads_per_block", threads_per_block);
  json_print_field_int("num_phases", num_phases);
  json_print_array_int("phase_counters", h_phase_counters, num_phases);
  printf(",\n");
  json_print_field_int("expected_per_phase", num_blocks);
  json_print_field_bool("success", success, true);
  printf("}\n");

  // Cleanup
  delete[] h_phase_counters;
  CUDA_CHECK(cudaFree(d_phase_counters));

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
