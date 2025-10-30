// C++ Reference Implementation for Scan Collective Tests
// This file contains NVIDIA's reference implementation for scan operations
// to compare PTX generation with Rust implementation.
//
// Focus: Addition operator (plus) which is the primary use case in tests

#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

// ============================================================================
// Inclusive Scan Tests - Addition
// ============================================================================

// Test 1: Basic inclusive scan with addition (tile32)
extern "C" __global__ void test_inclusive_scan_tile32_cpp(int *input,
                                                          int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 2: Inclusive scan with tile16
extern "C" __global__ void test_inclusive_scan_tile16_cpp(int *input,
                                                          int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<16>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 3: Inclusive scan with tile8
extern "C" __global__ void test_inclusive_scan_tile8_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<8>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 4: Inclusive scan with tile4
extern "C" __global__ void test_inclusive_scan_tile4_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 5: Inclusive scan with tile2
extern "C" __global__ void test_inclusive_scan_tile2_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<2>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 6: Inclusive scan with default operator (should use plus<int>)
extern "C" __global__ void test_inclusive_scan_default_cpp(int *input,
                                                           int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  int result = cg::inclusive_scan(tile, value);
  output[tile.thread_rank()] = result;
}

// ============================================================================
// Exclusive Scan Tests - Addition
// ============================================================================

// Test 7: Basic exclusive scan with addition (tile32)
extern "C" __global__ void test_exclusive_scan_tile32_cpp(int *input,
                                                          int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 8: Exclusive scan with tile16
extern "C" __global__ void test_exclusive_scan_tile16_cpp(int *input,
                                                          int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<16>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 9: Exclusive scan with tile8
extern "C" __global__ void test_exclusive_scan_tile8_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<8>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 10: Exclusive scan with tile4
extern "C" __global__ void test_exclusive_scan_tile4_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 11: Exclusive scan with tile2
extern "C" __global__ void test_exclusive_scan_tile2_cpp(int *input,
                                                         int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<2>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value, cg::plus<int>());
  output[tile.thread_rank()] = result;
}

// Test 12: Exclusive scan with default operator
extern "C" __global__ void test_exclusive_scan_default_cpp(int *input,
                                                           int *output) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  int result = cg::exclusive_scan(tile, value);
  output[tile.thread_rank()] = result;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// Test 13: Scan with sequential input [1,2,3,...,32]
extern "C" __global__ void test_scan_sequential_cpp(int *input, int *output_inc,
                                                    int *output_exc) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  output_inc[tile.thread_rank()] =
      cg::inclusive_scan(tile, value, cg::plus<int>());
  output_exc[tile.thread_rank()] =
      cg::exclusive_scan(tile, value, cg::plus<int>());
}

// Test 14: Scan with all zeros
extern "C" __global__ void test_scan_all_zeros_cpp(int *output_inc,
                                                   int *output_exc) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = 0;
  output_inc[tile.thread_rank()] =
      cg::inclusive_scan(tile, value, cg::plus<int>());
  output_exc[tile.thread_rank()] =
      cg::exclusive_scan(tile, value, cg::plus<int>());
}

// Test 15: Scan with negative numbers
extern "C" __global__ void test_scan_negative_cpp(int *input, int *output_inc,
                                                  int *output_exc) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int value = input[tile.thread_rank()];
  output_inc[tile.thread_rank()] =
      cg::inclusive_scan(tile, value, cg::plus<int>());
  output_exc[tile.thread_rank()] =
      cg::exclusive_scan(tile, value, cg::plus<int>());
}

// Test 16: Scan with single thread (tile1)
extern "C" __global__ void test_scan_tile1_cpp(int *input, int *output_inc,
                                               int *output_exc) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<1>(block);

  int value = input[threadIdx.x];
  if (threadIdx.x == 0) {
    output_inc[0] = cg::inclusive_scan(tile, value, cg::plus<int>());
    output_exc[0] = cg::exclusive_scan(tile, value, cg::plus<int>());
  }
}
