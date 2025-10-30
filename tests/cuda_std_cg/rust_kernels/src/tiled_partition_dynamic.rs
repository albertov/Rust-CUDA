use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Test dynamic tile creation with runtime size selection
#[kernel]
pub unsafe fn test_dynamic_tile_creation(ranks: *mut u32, sizes: *mut u32, tile_size: u32) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Sync the tile
    tile.sync();

    // Write rank and size to output
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);

    // Second sync to ensure no deadlock
    tile.sync();
}

/// Test dynamic shuffle operations
#[kernel]
pub unsafe fn test_dynamic_shuffle(results: *mut i32, tile_size: u32) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank() as i32;

    // Test shfl - broadcast rank 0's value to all threads
    let shfl_result = tile.shfl(rank, 0);

    // Test shfl_down - threads get value from higher lanes
    let shfl_down_result = tile.shfl_down(rank, 1);

    // Test shfl_up - threads get value from lower lanes
    let shfl_up_result = tile.shfl_up(rank, 1);

    // Test shfl_xor - butterfly exchange
    let shfl_xor_result = tile.shfl_xor(rank, 1);

    // Write results (4 values per thread)
    let base_idx = (thread_idx * 4) as usize;
    results.add(base_idx).write(shfl_result);
    results.add(base_idx + 1).write(shfl_down_result);
    results.add(base_idx + 2).write(shfl_up_result);
    results.add(base_idx + 3).write(shfl_xor_result);
}

/// Test dynamic vote operations
#[kernel]
pub unsafe fn test_dynamic_vote(
    any_results: *mut u32,
    all_results: *mut u32,
    ballot_results: *mut u32,
    tile_size: u32,
) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Predicate: true only for thread rank 0 in each tile
    let is_rank_zero = rank == 0;

    // Test any - should be true (at least one thread has rank 0)
    let any_result = tile.any(is_rank_zero);

    // Test all - should be false (not all threads have rank 0)
    let all_result = tile.all(is_rank_zero);

    // Test ballot - bit 0 should be set in each tile
    let ballot_result = tile.ballot(is_rank_zero);

    // Write results
    any_results
        .add(thread_idx as usize)
        .write(if any_result { 1 } else { 0 });
    all_results
        .add(thread_idx as usize)
        .write(if all_result { 1 } else { 0 });
    ballot_results.add(thread_idx as usize).write(ballot_result);
}

/// Test dynamic match operations
#[kernel]
pub unsafe fn test_dynamic_match(
    match_any_results: *mut u32,
    match_all_mask: *mut u32,
    match_all_pred: *mut u32,
    tile_size: u32,
) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank() as i32;

    // All threads in tile have same value (rank) - so all should match
    let match_any_result = tile.match_any(rank);

    // match_all should return (mask, true) since all threads have different values
    // Actually, each thread has its own rank, so match_all should show only self
    let (mask, all_match) = tile.match_all(rank);

    // Write results
    match_any_results
        .add(thread_idx as usize)
        .write(match_any_result);
    match_all_mask.add(thread_idx as usize).write(mask);
    match_all_pred
        .add(thread_idx as usize)
        .write(if all_match { 1 } else { 0 });
}

/// Test mixed static and dynamic tiles in same kernel
#[kernel]
pub unsafe fn test_mixed_static_dynamic(
    static_sizes: *mut u32,
    dynamic_sizes: *mut u32,
    dynamic_tile_size: u32,
) {
    let block = this_thread_block();

    // Create static tile (compile-time size)
    let static_tile = tiled_partition::<32>(&block);

    // Create dynamic tile (runtime size)
    let dynamic_tile = tiled_partition_dynamic(&block, dynamic_tile_size);

    let thread_idx = thread::thread_idx_x();

    // Sync both tiles
    static_tile.sync();
    dynamic_tile.sync();

    // Write sizes
    static_sizes
        .add(thread_idx as usize)
        .write(static_tile.size());
    dynamic_sizes
        .add(thread_idx as usize)
        .write(dynamic_tile.size());

    // Sync again
    static_tile.sync();
    dynamic_tile.sync();
}
