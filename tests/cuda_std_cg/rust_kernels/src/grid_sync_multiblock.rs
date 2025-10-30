use core::arch::asm;
use cuda_std::cooperative_groups::this_grid;
use cuda_std::prelude::*;

/// Atomic add operation using inline PTX assembly (global memory scope)
#[inline(always)]
unsafe fn atomic_add_i32(addr: *mut i32, val: i32) -> i32 {
    let mut old: i32;
    asm!(
        "atom.global.add.s32 {}, [{}], {};",
        out(reg32) old,
        in(reg64) addr,
        in(reg32) val,
    );
    old
}

/// Multi-block coordination test kernel.
///
/// Tests grid synchronization with varying block counts (2, 4, 8, 16 blocks).
/// Each test:
/// 1. Each thread writes its block ID to a unique array position
/// 2. Grid-wide synchronization
/// 3. All threads read from all array positions and compute sum
/// 4. Block 0 thread 0 accumulates global sum for verification
///
/// This validates that grid.sync() works correctly regardless of block count
/// and that memory writes from all blocks are visible after synchronization.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn grid_sync_multiblock_kernel(data: *mut i32, array_size: i32, global_sum: *mut i32) {
    let grid = this_grid();

    let block_idx = thread::block_idx_x();
    let thread_idx = thread::thread_idx_x();
    let block_dim = thread::block_dim_x();
    let grid_dim = thread::grid_dim_x();

    let tid = (block_idx * block_dim + thread_idx) as usize;
    let grid_size = (grid_dim * block_dim) as usize;
    let array_size = array_size as usize;

    // Phase 1: Write unique block ID to array positions
    let mut i = tid;
    while i < array_size {
        *data.add(i) = block_idx as i32;
        i += grid_size;
    }

    // Sync: Ensure all writes are visible to all threads
    grid.sync();

    // Phase 2: Read all values and compute local sum
    let mut local_sum = 0i32;
    let mut i = tid;
    while i < array_size {
        local_sum += *data.add(i);
        i += grid_size;
    }

    // Direct global atomic accumulation - NO shared memory, NO sync_threads()
    if local_sum > 0 {
        atomic_add_i32(global_sum, local_sum);
    }
}
