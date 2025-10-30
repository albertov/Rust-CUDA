use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Test inclusive_scan_add for TiledGroup<32>
///
/// Input: [1, 1, 1, ..., 1] (32 ones)
/// Expected: [1, 2, 3, ..., 32]
#[kernel]
pub unsafe fn test_inclusive_scan_add_tile32(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = 1; // Each thread contributes 1
    let prefix_sum = tile.inclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}

/// Test exclusive_scan_add for TiledGroup<32>
///
/// Input: [1, 1, 1, ..., 1] (32 ones)
/// Expected: [0, 1, 2, ..., 31]
#[kernel]
pub unsafe fn test_exclusive_scan_add_tile32(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = 1; // Each thread contributes 1
    let prefix_sum = tile.exclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}

/// Test inclusive_scan_add with sequential values
///
/// Input: [1, 2, 3, ..., 32]
/// Expected: [1, 3, 6, 10, ..., 528] (triangular numbers)
#[kernel]
pub unsafe fn test_inclusive_scan_sequential(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = (tile.thread_rank() + 1) as i32; // 1, 2, 3, ..., 32
    let prefix_sum = tile.inclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}

/// Test exclusive_scan_add with sequential values
///
/// Input: [1, 2, 3, ..., 32]
/// Expected: [0, 1, 3, 6, ..., 496]
#[kernel]
pub unsafe fn test_exclusive_scan_sequential(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = (tile.thread_rank() + 1) as i32; // 1, 2, 3, ..., 32
    let prefix_sum = tile.exclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}

/// Test inclusive_scan_min and inclusive_scan_max
/// DISABLED: scan_min/max not yet implemented
/*
#[kernel]
pub unsafe fn test_scan_min_max_tile32(min_out: *mut i32, max_out: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Each thread has a different value
    let value = tile.thread_rank() as i32;

    let running_min = tile.inclusive_scan_min(value);
    let running_max = tile.inclusive_scan_max(value);

    min_out.add(tile.thread_rank() as usize).write(running_min);
    max_out.add(tile.thread_rank() as usize).write(running_max);
}
*/

/// Test exclusive_scan_add for memory allocation pattern
///
/// Simulates allocating variable-size chunks
#[kernel]
pub unsafe fn test_exclusive_scan_allocation(
    sizes: *const i32,
    offsets: *mut i32,
    total: *mut i32,
) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let my_size = *sizes.add(tile.thread_rank() as usize);
    let my_offset = tile.exclusive_scan_add(my_size);

    offsets.add(tile.thread_rank() as usize).write(my_offset);

    // Thread 31 (last thread) computes total
    if tile.thread_rank() == 31 {
        let total_size = my_offset + my_size;
        *total = total_size;
    }
}

/// Test inclusive_scan_add for tile size 16
#[kernel]
pub unsafe fn test_inclusive_scan_add_tile16(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let value = 1;
    let prefix_sum = tile.inclusive_scan_add(value);

    output
        .add(thread::thread_idx_x() as usize)
        .write(prefix_sum);
}

/// Test exclusive_scan_add for tile size 16
#[kernel]
pub unsafe fn test_exclusive_scan_add_tile16(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let value = 1;
    let prefix_sum = tile.exclusive_scan_add(value);

    output
        .add(thread::thread_idx_x() as usize)
        .write(prefix_sum);
}

/// Test inclusive_scan_add for tile size 8
#[kernel]
pub unsafe fn test_inclusive_scan_add_tile8(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<8>(&block);

    let value = 1;
    let prefix_sum = tile.inclusive_scan_add(value);

    output
        .add(thread::thread_idx_x() as usize)
        .write(prefix_sum);
}

/// Test with DynamicTiledGroup
/// DISABLED: DynamicTiledGroup scan not yet implemented
/*
#[kernel]
pub unsafe fn test_scan_dynamic_tile(output: *mut i32, tile_size: u32) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let value = 1;
    let inclusive = tile.inclusive_scan_add(value);
    let exclusive = tile.exclusive_scan_add(value);

    // Store both results (inclusive in lower half, exclusive in upper half)
    let idx = thread::thread_idx_x() as usize;
    output.add(idx).write(inclusive);
    output.add(idx + 32).write(exclusive);
}
*/

/// Test scan with unsigned values
#[kernel]
pub unsafe fn test_scan_unsigned(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = (tile.thread_rank() + 1) as i32;
    let prefix_sum = tile.inclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}

/// Test scan correctness with varying input patterns
#[kernel]
pub unsafe fn test_scan_pattern(input: *const i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = *input.add(tile.thread_rank() as usize);
    let prefix_sum = tile.inclusive_scan_add(value);

    output.add(tile.thread_rank() as usize).write(prefix_sum);
}
