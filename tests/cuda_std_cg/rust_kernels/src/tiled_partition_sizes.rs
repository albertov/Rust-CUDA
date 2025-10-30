use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Test tile creation for SIZE=16 (half warp).
///
/// This kernel tests that TiledGroup<16> correctly creates two independent tiles
/// per warp:
/// - Threads 0-15: tile 0
/// - Threads 16-31: tile 1
///
/// Each tile should have:
/// - size() = 16
/// - thread_rank() in [0, 16)
/// - Independent synchronization
#[kernel]
pub unsafe fn test_tile16_creation(ranks: *mut u32, sizes: *mut u32, tile_ids: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Calculate which tile this thread belongs to
    let lane_id = thread_idx % 32;
    let tile_id = lane_id / 16;

    // Sync tile
    tile.sync();

    // Write outputs
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);
    tile_ids.add(thread_idx as usize).write(tile_id);

    // Second sync
    tile.sync();
}

/// Test tile creation for SIZE=8 (quarter warp).
///
/// This kernel tests that TiledGroup<8> correctly creates four independent tiles
/// per warp:
/// - Threads 0-7: tile 0
/// - Threads 8-15: tile 1
/// - Threads 16-23: tile 2
/// - Threads 24-31: tile 3
#[kernel]
pub unsafe fn test_tile8_creation(ranks: *mut u32, sizes: *mut u32, tile_ids: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<8>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Calculate which tile this thread belongs to
    let lane_id = thread_idx % 32;
    let tile_id = lane_id / 8;

    // Sync tile
    tile.sync();

    // Write outputs
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);
    tile_ids.add(thread_idx as usize).write(tile_id);

    // Second sync
    tile.sync();
}

/// Test tile creation for SIZE=4 (eighth warp).
///
/// This kernel tests that TiledGroup<4> correctly creates eight independent tiles
/// per warp.
#[kernel]
pub unsafe fn test_tile4_creation(ranks: *mut u32, sizes: *mut u32, tile_ids: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<4>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Calculate which tile this thread belongs to
    let lane_id = thread_idx % 32;
    let tile_id = lane_id / 4;

    // Sync tile
    tile.sync();

    // Write outputs
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);
    tile_ids.add(thread_idx as usize).write(tile_id);

    // Second sync
    tile.sync();
}

/// Test tile creation for SIZE=2 (pairs).
///
/// This kernel tests that TiledGroup<2> correctly creates sixteen pair tiles
/// per warp.
#[kernel]
pub unsafe fn test_tile2_creation(ranks: *mut u32, sizes: *mut u32, tile_ids: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<2>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Calculate which tile this thread belongs to
    let lane_id = thread_idx % 32;
    let tile_id = lane_id / 2;

    // Sync tile
    tile.sync();

    // Write outputs
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);
    tile_ids.add(thread_idx as usize).write(tile_id);

    // Second sync
    tile.sync();
}

/// Test tile creation for SIZE=1 (individual threads).
///
/// This kernel tests that TiledGroup<1> treats each thread as its own tile.
/// This is a trivial case where:
/// - Each thread is its own tile
/// - thread_rank() always = 0
/// - size() always = 1
/// - sync() is a no-op (no other threads in tile)
#[kernel]
pub unsafe fn test_tile1_creation(ranks: *mut u32, sizes: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<1>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Sync (no-op for single-thread tile)
    tile.sync();

    // Write outputs
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);

    // Second sync
    tile.sync();
}

/// Test shuffle operations with SIZE=16.
///
/// Tests that shuffle operations stay within tile boundaries:
/// - Tiles 0 (threads 0-15) and 1 (threads 16-31) are independent
/// - shfl_down(1) should not cross tile boundary at thread 15/16
#[kernel]
pub unsafe fn test_tile16_shuffle(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Shift down by 1: thread N receives from thread N+1 within tile
    let shifted_value = tile.shfl_down(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(shifted_value);

    tile.sync();
}

/// Test vote operations with SIZE=16.
///
/// Tests that vote operations only consider threads within the tile:
/// - ballot() should return 16-bit mask for each tile
/// - any() should only see threads in same tile
#[kernel]
pub unsafe fn test_tile16_vote(ballot_output: *mut u32, any_output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Even ranks have true, odd ranks have false (within tile)
    let my_predicate = (rank % 2) == 0;

    // Ballot should only collect votes from threads in this tile
    let ballot_mask = tile.ballot(my_predicate);

    // Any should only check threads in this tile
    let any_true = tile.any(my_predicate);

    // Write results
    ballot_output.add(thread_idx as usize).write(ballot_mask);
    any_output
        .add(thread_idx as usize)
        .write(if any_true { 1 } else { 0 });

    tile.sync();
}

/// Test shuffle operations with SIZE=8.
///
/// Tests that shuffle operations stay within tile boundaries for quarter warp.
#[kernel]
pub unsafe fn test_tile8_shuffle(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<8>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Shift down by 1 within tile
    let shifted_value = tile.shfl_down(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(shifted_value);

    tile.sync();
}

/// Test shuffle operations with SIZE=4.
///
/// Tests shuffle with very small tiles (4 threads).
#[kernel]
pub unsafe fn test_tile4_shuffle(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<4>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Shift down by 1 within tile
    let shifted_value = tile.shfl_down(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(shifted_value);

    tile.sync();
}

/// Test shuffle operations with SIZE=2 (pairs).
///
/// Tests that threads can exchange with their partner.
#[kernel]
pub unsafe fn test_tile2_shuffle(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<2>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value (0 or 1)
    let my_value = rank as i32;

    // XOR with 1: swap with partner (0↔1)
    let partner_value = tile.shfl_xor(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(partner_value);

    tile.sync();
}

/// Test ThreadGroup trait polymorphism.
///
/// This kernel uses TiledGroup through the ThreadGroup trait interface
/// to verify zero-cost abstraction.
#[kernel]
pub unsafe fn test_threadgroup_trait(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    // Access through inherent methods
    let rank_inherent = tile.thread_rank();
    let size_inherent = tile.size();

    // Access through ThreadGroup trait (should be identical)
    fn get_rank_via_trait<T: ThreadGroup>(group: &T) -> u32 {
        group.thread_rank()
    }

    fn get_size_via_trait<T: ThreadGroup>(group: &T) -> u32 {
        group.size()
    }

    let rank_trait = get_rank_via_trait(&tile);
    let size_trait = get_size_via_trait(&tile);

    // Verify they match
    let rank_match = if rank_inherent == rank_trait { 1 } else { 0 };
    let size_match = if size_inherent == size_trait { 1 } else { 0 };

    let thread_idx = thread::thread_idx_x();
    output.add(thread_idx as usize).write(rank_match);
    output.add((thread_idx + 32) as usize).write(size_match);

    tile.sync();
}

/// Test reduction with SIZE=16 tiles.
///
/// This tests that reductions work correctly within smaller tiles.
#[kernel]
pub unsafe fn test_generic_reduction_tile16(input: *const i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<16>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Load input value
    let mut value = input.add(thread_idx as usize).read();

    // Reduction within tile using inherent methods
    // (ThreadGroup trait doesn't include shuffle operations)
    let mut offset = tile.size() / 2;
    while offset > 0 {
        value += tile.shfl_down(value, offset);
        offset /= 2;
    }

    // First thread in each tile writes result
    if rank == 0 {
        let warp_id = thread_idx / 16;
        output.add(warp_id as usize).write(value);
    }

    tile.sync();
}

/// Test reduction with SIZE=8 tiles.
#[kernel]
pub unsafe fn test_generic_reduction_tile8(input: *const i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<8>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Load input value
    let mut value = input.add(thread_idx as usize).read();

    // Reduction within tile
    let mut offset = tile.size() / 2;
    while offset > 0 {
        value += tile.shfl_down(value, offset);
        offset /= 2;
    }

    // First thread in each tile writes result
    if rank == 0 {
        let warp_id = thread_idx / 8;
        output.add(warp_id as usize).write(value);
    }

    tile.sync();
}
