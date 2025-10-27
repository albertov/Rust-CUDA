//! Low-level grid synchronization intrinsics for CUDA cooperative groups.
//!
//! This module provides the foundation for multi-block (grid-wide) synchronization
//! primitives. Grid synchronization requires special kernel launch parameters
//! (`cudaLaunchCooperativeKernel`) and device support queries at runtime.
//!
//! # Architecture Support
//!
//! - **SM 6.0-6.9 (Pascal, Volta pre-7.0)**: Uses `fence.sc.gpu` + atomic operations
//! - **SM 7.0+ (Volta 7.0+, Turing, Ampere, etc.)**: Uses acquire/release atomic semantics
//!
//! # Safety Requirements
//!
//! All functions in this module are `unsafe` because:
//! - They must only be called from kernels launched with `cudaLaunchCooperativeKernel`
//! - The arrival counter must be properly allocated in device or managed memory
//! - Only the CTA (block) master thread should call arrive, but all threads must call wait
//! - Memory ordering semantics must be carefully respected
//!
//! # References
//!
//! - NVIDIA Cooperative Groups Documentation
//! - PTX ISA: Section on Memory Consistency Model
//! - CUDA Programming Guide: Appendix on Cooperative Groups

use crate::gpu_only;

/// Grid workspace structure allocated by the CUDA driver.
///
/// This structure is automatically allocated when using `cudaLaunchCooperativeKernel`
/// and its address is passed to the kernel via environment registers.
///
/// # Layout
/// - `ws_size`: Workspace size in bytes (for validation)
/// - `barrier`: Atomic counter for grid-wide synchronization
///
/// # Note
/// This structure is internal and should not be directly accessed by users.
/// Use `GridGroup::sync()` instead.
#[repr(C)]
pub struct GridWorkspace {
    ws_size: u32,
    barrier: u32,
}

/// Reads a 64-bit value from device environment registers.
///
/// Environment registers are special hardware registers that the CUDA driver
/// uses to pass implicit parameters to kernels. For cooperative kernels,
/// registers 1 and 2 contain the grid workspace pointer.
///
/// # Parameters
/// - `REG_HIGH`: Environment register number for high 32 bits (typically 1)
/// - `REG_LOW`: Environment register number for low 32 bits (typically 2)
///
/// # Returns
/// 64-bit value composed from the two environment registers
///
/// # Safety
/// This function reads from hardware registers. It is only valid when called
/// from within a cooperatively-launched kernel.
#[gpu_only]
#[inline(always)]
unsafe fn load_env_reg64<const REG_HIGH: u32, const REG_LOW: u32>() -> u64 {
    use core::arch::asm;

    let high: u32;
    let low: u32;

    // Read high 32 bits from envreg<REG_HIGH>
    asm!(
        "mov.u32 {reg}, %envreg{num};",
        reg = out(reg32) high,
        num = const REG_HIGH,
        options(nostack, preserves_flags)
    );

    // Read low 32 bits from envreg<REG_LOW>
    asm!(
        "mov.u32 {reg}, %envreg{num};",
        reg = out(reg32) low,
        num = const REG_LOW,
        options(nostack, preserves_flags)
    );

    // Combine into 64-bit value
    ((high as u64) << 32) | (low as u64)
}

/// Gets the grid workspace pointer from environment registers.
///
/// The CUDA driver automatically loads the workspace address into
/// environment registers 1 and 2 during cooperative kernel launch.
///
/// # Returns
/// Pointer to the grid workspace structure, or null if not cooperatively launched
///
/// # Safety
/// This function is only safe when called from within a cooperatively-launched kernel.
#[gpu_only]
#[inline(always)]
pub unsafe fn get_grid_workspace() -> *mut GridWorkspace {
    // NVIDIA uses envreg1 for high 32 bits, envreg2 for low 32 bits
    let addr = load_env_reg64::<1, 2>();
    addr as *mut GridWorkspace
}

/// Checks if the current thread is the master thread of its CTA (Cooperative Thread Array/block).
///
/// The CTA master is defined as the thread with indices (0, 0, 0) within the block.
/// This is used to ensure only one thread per block participates in grid-level arrival counting.
///
/// # Returns
///
/// `true` if this is thread (0,0,0) in the block, `false` otherwise.
#[inline(always)]
pub fn is_cta_master() -> bool {
    use crate::thread::{thread_idx_x, thread_idx_y, thread_idx_z};
    thread_idx_x() == 0 && thread_idx_y() == 0 && thread_idx_z() == 0
}

/// Detects if a synchronization barrier has "flipped" by comparing arrival counts.
///
/// Grid synchronization uses a wrapping counter that flips between phases. The algorithm
/// correctly handles 32-bit unsigned wraparound. This function implements NVIDIA's
/// barrier flip detection logic.
///
/// # Algorithm
///
/// The flip is detected when `(current - old) & 0x80000000 != 0`, which means:
/// - The most significant bit differs between old and current
/// - At least half the counter range has been traversed
/// - The barrier has entered a new synchronization phase
///
/// # Parameters
///
/// - `old_arrive`: The arrival count captured before incrementing
/// - `current_arrive`: The current arrival count being polled
///
/// # Returns
///
/// `true` if the barrier has flipped to a new phase, `false` otherwise.
#[inline(always)]
fn bar_has_flipped(old_arrive: u32, current_arrive: u32) -> bool {
    // NVIDIA's flip detection algorithm: checks if MSB differs
    // This handles unsigned wraparound correctly
    ((current_arrive ^ old_arrive) & 0x80000000) != 0
}

/// Records a grid barrier arrival and returns the old counter value.
///
/// This function implements the arrival phase of grid-wide synchronization:
/// 1. All threads in the block sync at block level first
/// 2. The CTA master thread atomically increments the global arrival counter
/// 3. The GPU master (block 0,0,0) uses a special increment to flip the barrier phase
///
/// # SM Architecture Differences
///
/// **SM 7.0+ (Volta 7.0+, Turing, Ampere, Ada, Hopper)**:
/// - Uses `atom.add.release.gpu.u32` with hardware release semantics
/// - Ensures all prior memory operations visible before arrival
/// - Single atomic operation, more efficient
///
/// **SM 6.0-6.9 (Pascal, Volta pre-7.0)**:
/// - Uses `fence.sc.gpu` followed by plain atomic add
/// - Fence ensures memory ordering before atomic
/// - Two-instruction sequence for compatibility
///
/// # Parameters
///
/// - `arrived`: Pointer to the global arrival counter (must be device/managed memory)
///
/// # Returns
///
/// The arrival counter value **before** this block's increment. Only valid for CTA master.
/// Non-master threads return 0.
///
/// # Safety
///
/// - Must be called from a kernel launched with `cudaLaunchCooperativeKernel`
/// - `arrived` must point to valid device-accessible memory
/// - Pointer must be naturally aligned for u32
/// - Must be called by all threads in all blocks participating in the grid sync
/// - Memory ordering must be respected (no races with other grid sync operations)
#[gpu_only]
#[inline(always)]
pub unsafe fn sync_grids_arrive(arrived: *mut u32) -> u32 {
    use crate::thread::{block_idx, grid_dim, sync_threads};

    let mut old_arrive: u32 = 0;

    // Step 1: Block-level synchronization first
    // All threads in this block must arrive before we increment the grid counter
    sync_threads();

    // Step 2: CTA master atomically increments the arrival counter
    if is_cta_master() {
        let grid = grid_dim();
        let expected = grid.x * grid.y * grid.z;
        let block = block_idx();
        let gpu_master = block.x == 0 && block.y == 0 && block.z == 0;

        // GPU master uses special value to flip barrier phase
        // Other blocks increment by 1
        let nb = if gpu_master {
            // When GPU master arrives, it flips the MSB to signal phase change
            // 0x80000000 - (expected - 1) ensures proper flip semantics
            0x80000000u32.wrapping_sub(expected - 1)
        } else {
            1u32
        };

        // Architecture-specific atomic operation
        #[cfg(target_arch_sm = "sm_70")]
        {
            // SM 7.0+: Use release atomic for efficient memory ordering
            // PTX: atom.add.release.gpu.u32 old,[arrived],nb;
            //
            // The release semantic ensures all prior memory writes are visible
            // before the arrival counter is incremented.
            asm!(
                "atom.add.release.gpu.u32 {old},[{addr}],{val};",
                old = out(reg32) old_arrive,
                addr = in(reg64) arrived,
                val = in(reg32) nb,
                options(nostack)
            );
        }

        #[cfg(not(target_arch_sm = "sm_70"))]
        {
            // SM 6.0-6.9: Use fence + relaxed atomic
            // PTX: fence.sc.gpu; followed by atom.add.relaxed.gpu.u32
            //
            // The fence provides memory ordering, then atomic increments.
            // This is less efficient but compatible with older architectures.
            use crate::atomic::intrinsics::{atomic_fetch_add_relaxed_u32_device, fence_sc_device};

            fence_sc_device();
            old_arrive = atomic_fetch_add_relaxed_u32_device(arrived, nb);
        }
    }

    // Step 3: Return the old arrival value (only meaningful for CTA master)
    old_arrive
}

/// Waits for the grid barrier to flip, indicating all blocks have arrived.
///
/// This function implements the waiting phase of grid-wide synchronization:
/// 1. CTA master polls the arrival counter until barrier flip is detected
/// 2. Block-level sync ensures all threads in the block wait together
///
/// # SM Architecture Differences
///
/// **SM 7.0+ (Volta 7.0+, Turing, Ampere, Ada, Hopper)**:
/// - Uses `ld.acquire.gpu.u32` with hardware acquire semantics
/// - Ensures memory visibility after barrier flip detected
/// - Single load operation, more efficient
///
/// **SM 6.0-6.9 (Pascal, Volta pre-7.0)**:
/// - Uses `ld.acquire.gpu.u32` followed by `fence.sc.gpu`
/// - Acquire load ensures cache coherency across SMs
/// - Fence provides additional memory ordering after barrier flip
/// - Two-instruction sequence for compatibility
///
/// # Parameters
///
/// - `old_arrive`: The value returned by `sync_grids_arrive` (only CTA master needs valid value)
/// - `arrived`: Pointer to the global arrival counter (must be same as in arrive call)
///
/// # Safety
///
/// - Must be called after `sync_grids_arrive` with the same `arrived` pointer
/// - `old_arrive` must be the value returned by the corresponding arrive call
/// - Must be called by all threads in all blocks participating in the grid sync
/// - Memory ordering must be respected (acquire semantics required)
#[gpu_only]
#[inline(always)]
pub unsafe fn sync_grids_wait(old_arrive: u32, arrived: *const u32) {
    use crate::thread::sync_threads;

    // Step 1: CTA master waits for barrier flip
    if is_cta_master() {
        #[cfg(target_arch_sm = "sm_70")]
        {
            // SM 7.0+: Use acquire load for efficient memory ordering
            // PTX: ld.acquire.gpu.u32 current,[arrived];
            //
            // The acquire semantic ensures all memory writes from other blocks
            // are visible after the load completes.
            let mut current_arrive: u32;
            loop {
                asm!(
                    "ld.acquire.gpu.u32 {current},[{addr}];",
                    current = out(reg32) current_arrive,
                    addr = in(reg64) arrived,
                    options(nostack, readonly)
                );

                if bar_has_flipped(old_arrive, current_arrive) {
                    break;
                }
            }
        }

        #[cfg(not(target_arch_sm = "sm_70"))]
        {
            // SM 6.0-6.9: Use acquire load + fence
            // PTX: ld.acquire.gpu.u32 followed by fence.sc.gpu
            //
            // Acquire load ensures cache coherency across SMs (prevents spinning on stale L1 cache).
            // Additional fence after loop provides extra memory ordering guarantee.
            use crate::atomic::intrinsics::{atomic_load_acquire_32_device, fence_sc_device};

            loop {
                let current_arrive = atomic_load_acquire_32_device(arrived);

                if bar_has_flipped(old_arrive, current_arrive) {
                    break;
                }
            }

            // Fence after detecting flip ensures memory visibility
            fence_sc_device();
        }
    }

    // Step 2: Block-level sync ensures all threads wait together
    sync_threads();
}
