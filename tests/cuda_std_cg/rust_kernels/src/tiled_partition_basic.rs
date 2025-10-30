use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Basic tile creation and synchronization test.
///
/// This kernel tests:
/// 1. Creating a TiledGroup<32> from a ThreadBlock
/// 2. Verifying tile.size() returns 32
/// 3. Verifying tile.thread_rank() is in range [0, 32)
/// 4. Calling tile.sync() multiple times without deadlock
///
/// Each thread writes its tile rank and size to the output arrays to verify
/// correct rank assignment.
#[kernel]
pub unsafe fn test_tile_creation_and_sync(ranks: *mut u32, sizes: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // First sync
    tile.sync();

    // Write rank and size to output
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);

    // Second sync to ensure no deadlock
    tile.sync();

    // Third sync for good measure
    tile.sync();
}

/// Test multiple tiles within a block.
///
/// This kernel launches with multiple warps per block and verifies that
/// each tile operates independently within its warp.
#[kernel]
pub unsafe fn test_multiple_tiles_per_block(ranks: *mut u32, sizes: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();
    let size = tile.size();

    // Each thread writes its rank and tile size
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);

    // Synchronize tile
    tile.sync();
}

/// Test tile synchronization with computation.
///
/// This kernel performs a simple warp-level computation that requires
/// synchronization between phases.
#[kernel]
pub unsafe fn test_tile_sync_with_computation(input: *const u32, output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();

    // Phase 1: Load data
    let value = input.add(thread_idx as usize).read();

    // Sync to ensure all threads loaded
    tile.sync();

    // Phase 2: Simple computation (double the value)
    let result = value * 2;

    // Sync before writing
    tile.sync();

    // Phase 3: Write result
    output.add(thread_idx as usize).write(result);

    // Final sync
    tile.sync();
}

/// Test tile.shfl() broadcast operation.
///
/// This kernel tests the indexed shuffle (broadcast) operation:
/// - Thread 0 has value 42
/// - All other threads have value 0
/// - All threads call tile.shfl(my_value, 0)
/// - Result: All threads should receive 42 (broadcast from thread 0)
#[kernel]
pub unsafe fn test_tile_shfl_broadcast(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Thread 0 has value 42, all others have 0
    let my_value = if rank == 0 { 42 } else { 0 };

    // Broadcast thread 0's value to all threads in tile
    let broadcast_value = tile.shfl(my_value, 0);

    // Write result
    output.add(thread_idx as usize).write(broadcast_value);

    // Sync to ensure all writes complete
    tile.sync();
}

/// Test tile.shfl_down() shift operation.
///
/// This kernel tests the downward shuffle operation:
/// - Each thread has value = rank
/// - All threads call tile.shfl_down(my_rank, 1)
/// - Result: Thread N receives value from thread N+1
///   (thread 31 receives its own value - no thread 32)
#[kernel]
pub unsafe fn test_tile_shfl_down(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Shift down by 1: receive value from lane rank-1
    let shifted_value = tile.shfl_down(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(shifted_value);

    tile.sync();
}

/// Test tile.shfl_up() shift operation.
///
/// This kernel tests the upward shuffle operation:
/// - Each thread has value = rank
/// - All threads call tile.shfl_up(my_rank, 1)
/// - Result: Thread N receives value from thread N-1
///   (thread 0 receives its own value - no thread -1)
#[kernel]
pub unsafe fn test_tile_shfl_up(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Shift up by 1: receive value from lane rank-1
    let shifted_value = tile.shfl_up(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(shifted_value);

    tile.sync();
}

/// Test tile.shfl_xor() butterfly operation.
///
/// This kernel tests the butterfly shuffle operation:
/// - Each thread has value = rank
/// - All threads call tile.shfl_xor(my_rank, 1)
/// - Result: Even threads swap with odd neighbors (0↔1, 2↔3, etc.)
#[kernel]
pub unsafe fn test_tile_shfl_xor(output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Each thread has its rank as value
    let my_value = rank as i32;

    // Butterfly XOR with mask=1: swap even/odd pairs
    let xor_value = tile.shfl_xor(my_value, 1);

    // Write result
    output.add(thread_idx as usize).write(xor_value);

    tile.sync();
}

/// Test tile.any() vote operation.
///
/// This kernel tests the any() vote operation with divergent execution:
/// - Threads 0-15: predicate = true
/// - Threads 16-31: predicate = false
/// - Result: any() should return true for ALL threads
///   (because at least one thread has true)
#[kernel]
pub unsafe fn test_tile_any(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Lower half has true, upper half has false
    let my_predicate = rank < 16;

    // Vote: any thread has true?
    let any_true = tile.any(my_predicate);

    // Convert bool to u32 for output
    let result = if any_true { 1 } else { 0 };

    // Write result
    output.add(thread_idx as usize).write(result);

    tile.sync();
}

/// Test tile.all() vote operation.
///
/// This kernel tests the all() vote operation with divergent execution:
/// - Test 1 (output[0..32]): All threads have true → all() returns true
/// - Test 2 (output[32..64]): Thread 0 has false → all() returns false
#[kernel]
pub unsafe fn test_tile_all(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Test 1: All threads have true predicate
    let unanimous_true = true;
    let all_true = tile.all(unanimous_true);
    let result1 = if all_true { 1 } else { 0 };
    output.add(thread_idx as usize).write(result1);

    tile.sync();

    // Test 2: Only thread 0 has false predicate
    let divergent = rank != 0;
    let all_true2 = tile.all(divergent);
    let result2 = if all_true2 { 1 } else { 0 };
    output.add((thread_idx + 32) as usize).write(result2);

    tile.sync();
}

/// Test tile.ballot() vote operation.
///
/// This kernel tests the ballot() operation which collects votes as a bitmask:
/// - Even threads (0,2,4,...,30): predicate = true
/// - Odd threads (1,3,5,...,31): predicate = false
/// - Result: ballot() returns 0xAAAAAAAA (alternating bits: 10101010...)
#[kernel]
pub unsafe fn test_tile_ballot(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Even ranks have true, odd ranks have false
    let my_predicate = (rank % 2) == 0;

    // Collect votes as bitmask
    let ballot_mask = tile.ballot(my_predicate);

    // All threads write the same mask
    output.add(thread_idx as usize).write(ballot_mask);

    tile.sync();
}

/// Test tile.match_any() operation (SM 7.0+).
///
/// This kernel tests the match_any() operation which finds threads with matching values:
/// - Threads 0-15: value = 100
/// - Threads 16-31: value = 200
/// - Threads in group with value=100 get mask 0x0000FFFF (lower half)
/// - Threads in group with value=200 get mask 0xFFFF0000 (upper half)
#[kernel]
pub unsafe fn test_tile_match_any(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Lower half has value 100, upper half has value 200
    let my_value = if rank < 16 { 100 } else { 200 };

    // Find threads with matching values
    let match_mask = tile.match_any(my_value);

    // Write mask for this thread
    output.add(thread_idx as usize).write(match_mask);

    tile.sync();
}

/// Test tile.match_all() operation (SM 7.0+).
///
/// This kernel tests match_all() which returns (mask, all_match):
/// - Test 1 (output[0..64]): All threads value=42 → (0xFFFFFFFF, true)
/// - Test 2 (output[64..128]): Thread 0 value=99, rest=42 → (varies, false)
#[kernel]
pub unsafe fn test_tile_match_all(output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let thread_idx = thread::thread_idx_x();
    let rank = tile.thread_rank();

    // Test 1: All threads have same value (42)
    let unanimous_value = 42i32;
    let (mask1, all_match1) = tile.match_all(unanimous_value);
    output.add(thread_idx as usize).write(mask1);
    output
        .add((thread_idx + 32) as usize)
        .write(if all_match1 { 1 } else { 0 });

    tile.sync();

    // Test 2: Thread 0 has different value (99), rest have 42
    let divergent_value = if rank == 0 { 99i32 } else { 42i32 };
    let (mask2, all_match2) = tile.match_all(divergent_value);
    output.add((thread_idx + 64) as usize).write(mask2);
    output
        .add((thread_idx + 96) as usize)
        .write(if all_match2 { 1 } else { 0 });

    tile.sync();
}
