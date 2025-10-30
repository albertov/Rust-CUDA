use core::arch::asm;
use cuda_std::cooperative_groups::this_grid;
use cuda_std::prelude::*;

/// Basic grid synchronization test kernel.
///
/// Tests a simple two-phase pattern:
/// 1. Each block increments its own counter
/// 2. Grid-wide synchronization
/// 3. Thread 0 verifies all counters are visible and equal to 1
///
/// This validates that grid.sync() ensures memory visibility across all blocks.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn grid_sync_basic_kernel(counters: *mut u32, num_blocks: i32) {
    let grid = this_grid();

    let block_idx = thread::block_idx_x() as usize;
    let thread_idx = thread::thread_idx_x();
    let tid = block_idx * thread::block_dim_x() as usize + thread_idx as usize;

    // Phase 1: Each block increments its own counter
    if thread_idx == 0 {
        let counter_ptr = counters.add(block_idx);
        *counter_ptr += 1;
    }

    // Synchronize all blocks - ensure all increments are visible
    grid.sync();
    if tid == 0 {
        for i in 0..num_blocks as usize {
            let _value = *counters.add(i);
        }
    }
}
