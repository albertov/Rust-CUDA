use core::arch::asm;
use cuda_std::cooperative_groups::this_grid;
use cuda_std::prelude::*;

/// Atomic add operation using inline PTX assembly with GPU-wide visibility
#[inline(always)]
unsafe fn atomic_add_i32(addr: *mut i32, val: i32) -> i32 {
    let mut old: i32;
    asm!(
        "atom.add.release.gpu.u32 {}, [{}], {};",
        out(reg32) old,
        in(reg64) addr,
        in(reg32) val,
    );
    old
}

/// Multi-phase synchronization test kernel.
///
/// Tests multiple sequential grid synchronizations in a single kernel launch.
/// Each phase:
/// 1. Each block atomically increments the phase counter
/// 2. Grid-wide synchronization
/// 3. Block 0 verifies the counter equals the number of blocks
/// 4. Grid-wide synchronization before next phase
///
/// This validates that:
/// - Multiple grid.sync() calls work correctly in sequence
/// - Each phase completes fully before the next begins
/// - Memory visibility is maintained across multiple synchronization points
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn grid_sync_phases_kernel(phase_counters: *mut i32, num_phases: i32) {
    let grid = this_grid();

    let block_idx = thread::block_idx_x();
    let thread_idx_x = thread::thread_idx_x();
    let grid_dim = thread::grid_dim_x();

    for phase in 0..num_phases {
        // Each block contributes to the phase counter
        if thread_idx_x == 0 {
            let counter_ptr = phase_counters.add(phase as usize);
            atomic_add_i32(counter_ptr, 1);
        }

        // Synchronize: ensure all blocks have incremented
        grid.sync();

        // Block 0 verifies all blocks reached this phase
        if block_idx == 0 && thread_idx_x == 0 {
            let expected = grid_dim as i32;
            let actual = *phase_counters.add(phase as usize);
            if actual != expected {
                // Note: printf equivalent would need to be added if available in cuda_std
                // For now we rely on host-side verification
            }
        }

        // Synchronize before next iteration to ensure verification completes
        grid.sync();
    }
}
