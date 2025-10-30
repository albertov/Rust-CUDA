use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Test reduce_add for TiledGroup<32>
///
/// Expected: sum of 0..31 = 496
#[kernel]
pub unsafe fn test_reduce_add_tile32(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = tile.thread_rank() as i32;
    let sum = tile.reduce_add(value);

    // All threads get the result with redux.sync
    output.add(tile.thread_rank() as usize).write(sum);
}

/// Test reduce_min and reduce_max for TiledGroup<32>
#[kernel]
pub unsafe fn test_reduce_min_max_tile32(min_out: *mut i32, max_out: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = tile.thread_rank() as i32;
    let min = tile.reduce_min(value);
    let max = tile.reduce_max(value);

    min_out.add(tile.thread_rank() as usize).write(min);
    max_out.add(tile.thread_rank() as usize).write(max);
}

/// Test bitwise reductions (AND, OR, XOR) for TiledGroup<32>
#[kernel]
pub unsafe fn test_reduce_bitwise_tile32(and_out: *mut u32, or_out: *mut u32, xor_out: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = if tile.thread_rank() % 2 == 0 {
        0xFFFFFFFF
    } else {
        0x00000000
    };

    let and_result = tile.reduce_and(value);
    let or_result = tile.reduce_or(value);
    let xor_result = tile.reduce_xor(value);

    and_out.add(tile.thread_rank() as usize).write(and_result);
    or_out.add(tile.thread_rank() as usize).write(or_result);
    xor_out.add(tile.thread_rank() as usize).write(xor_result);
}

/// Test reduce_add for smaller tile sizes
#[kernel]
pub unsafe fn test_reduce_add_tile16(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let value = tile.thread_rank() as i32;
    let sum = tile.reduce_add(value);

    output.add(thread::thread_idx_x() as usize).write(sum);
}

/// Test reduce_add for tile size 8
#[kernel]
pub unsafe fn test_reduce_add_tile8(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<8>(&block);

    let value = tile.thread_rank() as i32;
    let sum = tile.reduce_add(value);

    output.add(thread::thread_idx_x() as usize).write(sum);
}

/// Test reduce_min with unsigned values
#[kernel]
pub unsafe fn test_reduce_min_unsigned(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Each thread contributes a different value
    let value = (tile.thread_rank() * 10) as u32;
    let min = tile.reduce_min(value);

    output.add(tile.thread_rank() as usize).write(min);
}

/// Test reduce_max with unsigned values
#[kernel]
pub unsafe fn test_reduce_max_unsigned(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Each thread contributes a different value
    let value = (tile.thread_rank() * 10) as u32;
    let max = tile.reduce_max(value);

    output.add(tile.thread_rank() as usize).write(max);
}

/// Test reduce_and with specific bit patterns
#[kernel]
pub unsafe fn test_reduce_and_pattern(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // All threads have bit 0 set, only even threads have bit 1 set
    let value = if tile.thread_rank() % 2 == 0 {
        0b11u32
    } else {
        0b01u32
    };

    let and_result = tile.reduce_and(value);

    output.add(tile.thread_rank() as usize).write(and_result);
}

/// Test reduce_or with specific bit patterns
#[kernel]
pub unsafe fn test_reduce_or_pattern(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Thread N sets bit N
    let value = 1u32 << tile.thread_rank();

    let or_result = tile.reduce_or(value);

    output.add(tile.thread_rank() as usize).write(or_result);
}

/// Test reduce_xor for parity calculation
#[kernel]
pub unsafe fn test_reduce_xor_parity(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Count number of threads with rank divisible by 3
    let value = if tile.thread_rank() % 3 == 0 {
        1u32
    } else {
        0u32
    };

    let xor_result = tile.reduce_xor(value);

    output.add(tile.thread_rank() as usize).write(xor_result);
}

/// Test reduce_add with DynamicTiledGroup
#[kernel]
pub unsafe fn test_reduce_add_dynamic_tile(output: *mut i32, tile_size: u32) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let value = tile.thread_rank() as i32;
    let sum = tile.reduce_add(value);

    output.add(thread::thread_idx_x() as usize).write(sum);
}

/// Test reduce operations with CoalescedGroup
#[kernel]
pub unsafe fn test_reduce_add_coalesced(output: *mut i32) {
    let thread_idx = thread::thread_idx_x();
    let is_even = thread_idx % 2 == 0;

    if is_even {
        let coalesced = coalesced_threads();
        let value = thread_idx as i32;
        let sum = coalesced.reduce_add(value);
        output.add(thread_idx as usize).write(sum);
    } else {
        output.add(thread_idx as usize).write(-1);
    }
}
