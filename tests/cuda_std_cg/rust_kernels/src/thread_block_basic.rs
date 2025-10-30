use cuda_std::cooperative_groups::this_thread_block;
use cuda_std::prelude::*;

/// Test kernel for basic ThreadBlock functionality.
///
/// This kernel verifies:
/// 1. this_thread_block() factory creates valid handle
/// 2. sync() successfully synchronizes all threads in block
/// 3. size() returns correct thread count
/// 4. thread_rank() returns unique values for each thread
/// 5. Block-local behavior (different blocks operate independently)
///
/// Layout:
/// - Each block processes a chunk of data
/// - All threads write their rank to output array
/// - Sync between write and read phases
/// - Verify all threads see consistent data after sync
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn thread_block_basic_test(
    output_ranks: *mut u32,
    output_size: *mut u32,
    sync_flag: *mut u32,
) {
    let block = this_thread_block();

    // Get block and thread information
    let block_idx = thread::block_idx().x;
    let threads_per_block = block.size();
    let thread_rank = block.thread_rank();

    // Calculate global offset for this block
    let block_offset = (block_idx * threads_per_block) as usize;

    // Phase 1: Each thread writes its rank
    *output_ranks.add(block_offset + thread_rank as usize) = thread_rank;
    *output_size.add(block_offset + thread_rank as usize) = threads_per_block;

    // CRITICAL: Block-level sync ensures all threads finished writing
    block.sync();

    // Phase 2: Set sync flag to indicate sync completed
    // All threads should see Phase 1 writes after sync
    if thread_rank == 0 {
        *sync_flag.add(block_idx as usize) = 1;
    }

    // Another sync to ensure flag is visible to all threads
    block.sync();
}

/// Test kernel for ThreadBlock dimension accessors.
///
/// Verifies that dimension and index methods return correct values.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn thread_block_dimensions_test(
    output_dim_x: *mut u32,
    output_dim_y: *mut u32,
    output_dim_z: *mut u32,
    output_idx_x: *mut u32,
    output_idx_y: *mut u32,
    output_idx_z: *mut u32,
) {
    let block = this_thread_block();

    // Get global thread index
    let global_idx = thread::index();

    // Write dimension information
    *output_dim_x.add(global_idx as usize) = block.dim_x();
    *output_dim_y.add(global_idx as usize) = block.dim_y();
    *output_dim_z.add(global_idx as usize) = block.dim_z();

    // Write thread index information
    *output_idx_x.add(global_idx as usize) = block.thread_index_x();
    *output_idx_y.add(global_idx as usize) = block.thread_index_y();
    *output_idx_z.add(global_idx as usize) = block.thread_index_z();
}

/// Test kernel for multiple syncs in sequence.
///
/// Verifies that multiple sync calls work correctly without deadlock.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn thread_block_multi_sync_test(counter: *mut u32) {
    let block = this_thread_block();
    let thread_rank = block.thread_rank();
    let block_idx = thread::block_idx().x;
    let block_offset = (block_idx * block.size()) as usize;

    // Phase 1: Initialize
    *counter.add(block_offset + thread_rank as usize) = 0;
    block.sync();

    // Phase 2: Increment
    *counter.add(block_offset + thread_rank as usize) = 1;
    block.sync();

    // Phase 3: Read neighbor and add
    let neighbor_rank = (thread_rank + 1) % block.size();
    let neighbor_value = *counter.add(block_offset + neighbor_rank as usize);
    *counter.add(block_offset + thread_rank as usize) = 1 + neighbor_value;
    block.sync();

    // All values should now be 2
}
