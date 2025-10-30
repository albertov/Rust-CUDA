#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Helper to get the actual lane mask (not packed)
// Uses __ballot_sync with 0xFFFFFFFF to get the full warp mask
__device__ unsigned int get_actual_mask() { return __activemask(); }

/**
 * Test coalesced group with divergent execution (if/else split).
 *
 * Threads split into two groups:
 * - Threads 0-15: take the if branch
 * - Threads 16-31: take the else branch
 *
 * Each group creates a coalesced_threads() handle and writes:
 * - rank: thread rank within coalesced group (should be 0-15)
 * - size: number of threads in group (should be 16)
 * - mask: bitmask of active threads
 */
extern "C" __global__ void
test_coalesced_divergent_cpp(unsigned int *output_ranks,
                             unsigned int *output_sizes,
                             unsigned int *output_masks) {
  unsigned int tid = threadIdx.x;

  // Divergent execution based on thread ID
  if (tid < 16) {
    // First half of warp
    cg::coalesced_group group = cg::coalesced_threads();

    unsigned int rank = group.thread_rank();
    unsigned int size = group.size();
    // Get actual warp lane mask (not the packed ballot result)
    unsigned int mask = get_actual_mask();

    output_ranks[tid] = rank;
    output_sizes[tid] = size;
    output_masks[tid] = mask;

    group.sync();
  } else {
    // Second half of warp
    cg::coalesced_group group = cg::coalesced_threads();

    unsigned int rank = group.thread_rank();
    unsigned int size = group.size();
    // Get actual warp lane mask (not the packed ballot result)
    unsigned int mask = get_actual_mask();

    output_ranks[tid] = rank;
    output_sizes[tid] = size;
    output_masks[tid] = mask;

    group.sync();
  }
}

/**
 * Test coalesced group with sparse threads (every other thread).
 *
 * Threads split by odd/even:
 * - Even threads (0,2,4,...,30) form one group
 * - Odd threads (1,3,5,...,31) form another group
 *
 * Each group should have 16 threads with ranks 0-15.
 */
extern "C" __global__ void
test_coalesced_sparse_cpp(unsigned int *output_ranks,
                          unsigned int *output_sizes,
                          unsigned int *output_masks) {
  unsigned int tid = threadIdx.x;

  // Sparse divergence - every other thread
  if (tid % 2 == 0) {
    // Even threads: 0, 2, 4, ..., 30
    cg::coalesced_group group = cg::coalesced_threads();

    unsigned int rank = group.thread_rank();
    unsigned int size = group.size();
    // Get actual warp lane mask (not the packed ballot result)
    unsigned int mask = get_actual_mask();

    output_ranks[tid] = rank;
    output_sizes[tid] = size;
    output_masks[tid] = mask;

    group.sync();
  } else {
    // Odd threads: 1, 3, 5, ..., 31
    cg::coalesced_group group = cg::coalesced_threads();

    unsigned int rank = group.thread_rank();
    unsigned int size = group.size();
    // Get actual warp lane mask (not the packed ballot result)
    unsigned int mask = get_actual_mask();

    output_ranks[tid] = rank;
    output_sizes[tid] = size;
    output_masks[tid] = mask;

    group.sync();
  }
}

/**
 * Test shuffle operations with coalesced group (sparse threads).
 *
 * Only even threads participate (0,2,4,...,30).
 * Tests:
 * 1. shfl() - broadcast from rank 0 to all threads
 * 2. shfl_down() - each thread gets value from next rank
 * 3. shfl_up() - each thread gets value from previous rank
 */
extern "C" __global__ void test_coalesced_shuffle_cpp(int *shfl_results,
                                                      int *shfl_down_results,
                                                      int *shfl_up_results) {
  unsigned int tid = threadIdx.x;

  // Only even threads participate
  if (tid % 2 == 0) {
    cg::coalesced_group group = cg::coalesced_threads();

    // Each thread has its lane ID as value
    int my_value = (int)tid;

    // Test 1: Broadcast from rank 0 (which is lane 0)
    int shfl_value = group.shfl(my_value, 0);
    shfl_results[tid] = shfl_value;

    // Test 2: Shift down by 1 rank
    int shfl_down_value = group.shfl_down(my_value, 1);
    shfl_down_results[tid] = shfl_down_value;

    // Test 3: Shift up by 1 rank
    int shfl_up_value = group.shfl_up(my_value, 1);
    shfl_up_results[tid] = shfl_up_value;

    group.sync();
  }
}
