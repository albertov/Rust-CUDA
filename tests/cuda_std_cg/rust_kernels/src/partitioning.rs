//! Kernel implementations for labeled_partition and binary_partition tests.

#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]

#[cfg(target_os = "cuda")]
use cuda_std::*;

/// Test basic labeled_partition with uniform groups.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_labeled_partition_uniform(
    output_size: *mut u32,
    output_rank: *mut u32,
    output_mask: *mut u32,
) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    // Create 4 groups: threads 0-7, 8-15, 16-23, 24-31
    let label = tid / 8;
    let group = cooperative_groups::labeled_partition(&block, label);

    // Write results
    *output_size.add(tid as usize) = group.size();
    *output_rank.add(tid as usize) = group.thread_rank();
    *output_mask.add(tid as usize) = group.mask();
}

/// Test labeled_partition with sparse/uneven groups.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_labeled_partition_sparse(
    output_size: *mut u32,
    output_rank: *mut u32,
    output_mask: *mut u32,
) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    let label = tid % 3;
    let group = cooperative_groups::labeled_partition(&block, label);

    *output_size.add(tid as usize) = group.size();
    *output_rank.add(tid as usize) = group.thread_rank();
    *output_mask.add(tid as usize) = group.mask();
}

/// Test binary_partition with even/odd split.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_binary_partition_even_odd(
    output_size: *mut u32,
    output_rank: *mut u32,
    output_mask: *mut u32,
) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    let is_even = tid % 2 == 0;
    let group = cooperative_groups::binary_partition(&block, is_even);

    *output_size.add(tid as usize) = group.size();
    *output_rank.add(tid as usize) = group.thread_rank();
    *output_mask.add(tid as usize) = group.mask();
}

/// Test binary_partition with threshold condition.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_binary_partition_threshold(
    output_size: *mut u32,
    output_rank: *mut u32,
    output_mask: *mut u32,
) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    let below_threshold = tid < 20;
    let group = cooperative_groups::binary_partition(&block, below_threshold);

    *output_size.add(tid as usize) = group.size();
    *output_rank.add(tid as usize) = group.thread_rank();
    *output_mask.add(tid as usize) = group.mask();
}

/// Test partitioning from TiledGroup parent.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_partition_from_tile(
    output_size: *mut u32,
    output_rank: *mut u32,
    output_mask: *mut u32,
) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    // Create 16-thread tiles
    let tile = cooperative_groups::tiled_partition::<16>(&block);

    // Partition each tile by even/odd
    let is_even = (tid % 16) % 2 == 0;
    let group = cooperative_groups::binary_partition(&tile, is_even);

    *output_size.add(tid as usize) = group.size();
    *output_rank.add(tid as usize) = group.thread_rank();
    *output_mask.add(tid as usize) = group.mask();
}

/// Test that partitioned groups support shuffle operations.
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn test_partition_operations(output: *mut i32) {
    let block = cooperative_groups::this_thread_block();
    let tid = thread::thread_idx_x();

    // Partition by even/odd
    let is_even = tid % 2 == 0;
    let group = cooperative_groups::binary_partition(&block, is_even);

    // Each thread has its tid as value
    let value = tid as i32;

    // Shuffle down by 1 (get value from next rank)
    let neighbor = group.shfl_down(value, 1);

    // Store result
    *output.add(tid as usize) = neighbor;
}
