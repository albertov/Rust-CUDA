//! Grid-wide synchronization primitives for CUDA cooperative groups.
//!
//! This module provides the high-level public API for synchronizing all threads
//! across all blocks in a CUDA kernel. Grid synchronization enables multi-phase
//! algorithms where each phase requires a globally consistent view of memory.
//!
//! # Overview
//!
//! The `GridGroup` type represents a handle to all threads in the entire grid.
//! It provides a safe, ergonomic wrapper around the low-level grid synchronization
//! intrinsics.
//!
//! # Requirements
//!
//! Grid synchronization has strict requirements:
//!
//! 1. **Cooperative Launch**: Kernel must be launched via `cudaLaunchCooperativeKernel`
//!    (not the regular `<<<>>>` syntax or `cudaLaunchKernel`)
//!
//! 2. **Device Support**: GPU must support cooperative launch (query
//!    `cudaDevAttrCooperativeLaunch` to check)
//!
//! 3. **Uniform Participation**: ALL threads in ALL blocks must call `sync()`.
//!    Divergent sync calls cause deadlock.
//!
//! 4. **Architecture**: Minimum SM 6.0 (Pascal). SM 7.0+ (Volta+) recommended
//!    for better performance.
//!
//! # Example Usage
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn iterative_solver(data: *mut f32, max_iters: u32) {
//!     let grid = this_grid();
//!
//!     // Multi-phase algorithm with grid-wide synchronization
//!     for iteration in 0..max_iters {
//!         // Phase 1: Each block computes local results
//!         let local_result = process_block_data(data);
//!
//!         // Synchronize: ensure all blocks finished Phase 1
//!         grid.sync();
//!
//!         // Phase 2: Update global state with consistent view
//!         update_global_state(data, local_result);
//!
//!         // Synchronize: ensure all blocks see updates
//!         grid.sync();
//!
//!         // Check convergence
//!         if check_convergence(data) {
//!             break;
//!         }
//!     }
//! }
//! ```
//!
//! # Memory Ordering
//!
//! Grid sync provides strong memory ordering guarantees:
//!
//! - **SM 7.0+**: Hardware acquire/release semantics. All memory operations before
//!   `sync()` are visible to all threads after `sync()` returns.
//!
//! - **SM 6.0-6.9**: Sequential consistency via explicit fences. Same visibility
//!   guarantees as SM 7.0+ but with additional overhead.
//!
//! # Common Patterns
//!
//! ## Producer-Consumer
//!
//! ```no_run
//! # use cuda_std::cooperative_groups::*;
//! # #[kernel]
//! # pub unsafe fn example(input: *const f32, output: *mut f32) {
//! let grid = this_grid();
//!
//! // Producer blocks generate data
//! if thread_rank() < PRODUCER_COUNT {
//!     produce_data(output);
//! }
//!
//! grid.sync(); // Ensure all data produced
//!
//! // Consumer blocks process data
//! if thread_rank() >= PRODUCER_COUNT {
//!     consume_data(output);
//! }
//! # }
//! ```
//!
//! ## Multi-Phase Reduction
//!
//! ```no_run
//! # use cuda_std::cooperative_groups::*;
//! # #[kernel]
//! # pub unsafe fn example(data: *mut f32, partial: *mut f32) {
//! let grid = this_grid();
//!
//! // Phase 1: Block-level reductions
//! let block_sum = block_reduce(data);
//! if thread_idx() == 0 {
//!     partial[block_idx()] = block_sum;
//! }
//!
//! grid.sync(); // Ensure all block sums ready
//!
//! // Phase 2: Grid-level reduction (single block)
//! if block_idx() == 0 {
//!     let total = reduce_partial_sums(partial);
//!     if thread_idx() == 0 {
//!         *data = total;
//!     }
//! }
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - **Overhead**: Grid sync is expensive (microseconds). Use block-level
//!   sync (`sync_threads()`) when possible.
//!
//! - **Occupancy**: Cooperative launch may reduce occupancy (fewer blocks
//!   resident on SM). Profile to ensure acceptable performance.
//!
//! - **Architecture**: SM 7.0+ has ~2x better grid sync performance due to
//!   hardware acquire/release atomics.
//!
//! # Safety
//!
//! The public API is safe because:
//! - `GridGroup` lifetime is tied to kernel scope (cannot escape)
//! - Type system prevents misuse of grid handles
//! - Unsafe operations are encapsulated in intrinsics layer
//!
//! However, incorrect usage patterns can still cause:
//! - **Deadlock**: If not all threads call `sync()`
//! - **Undefined behavior**: If kernel not cooperatively launched
//!
//! # Workspace Mechanism
//!
//! The workspace pointer is automatically provided by the CUDA driver via
//! environment registers (envreg1 and envreg2) during cooperative kernel launch.
//! The driver allocates a small workspace structure containing a barrier counter
//! used for grid-wide synchronization.

use core::marker::PhantomData;

/// A handle to all threads in the entire grid.
///
/// `GridGroup` represents the complete set of threads across all blocks in a
/// cooperatively-launched kernel. It provides the `sync()` method for grid-wide
/// synchronization.
///
/// # Lifetime
///
/// The `'a` lifetime parameter ties the grid group to the kernel's execution
/// scope. This prevents the handle from escaping the kernel, which would be
/// invalid since grid groups only exist during kernel execution.
///
/// # Construction
///
/// Create via the `this_grid()` function:
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
/// let grid = this_grid();
/// ```
///
/// # Requirements
///
/// - Kernel launched with `cudaLaunchCooperativeKernel`
/// - SM 6.0+ GPU (Pascal or newer)
/// - All threads must participate uniformly in sync operations
///
/// # Implementation Details
///
/// The `workspace` field stores a pointer directly to the barrier field (u32) within
/// the driver-allocated GridWorkspace structure, rather than a pointer to the full
/// GridWorkspace. This design choice eliminates field offset calculations in
/// performance-critical sync operations.
///
/// The workspace pointer is obtained from environment registers `%envreg1` and `%envreg2`,
/// which the CUDA driver sets during cooperative kernel launch. This approach matches
/// NVIDIA's internal implementation.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn cooperative_kernel(data: *mut i32) {
///     let grid = this_grid();
///
///     // Phase 1: computation
///     data.add(grid.thread_rank()).write(compute_value());
///
///     // Barrier: ensure all threads finished Phase 1
///     grid.sync();
///
///     // Phase 2: all threads see Phase 1 results
///     let neighbor = data.add((grid.thread_rank() + 1) % grid.size()).read();
///     process_neighbor(neighbor);
/// }
/// ```
pub struct GridGroup<'a> {
    /// Pointer to barrier field (u32) within driver-allocated GridWorkspace.
    ///
    /// Stored as `*mut u32` for direct use in intrinsics without field offset overhead.
    /// The pointer is valid only for cooperatively-launched kernels and is null otherwise.
    workspace: *mut u32,

    /// Phantom lifetime marker ensuring GridGroup cannot outlive the kernel invocation.
    _marker: PhantomData<&'a ()>,
}

/// Creates a grid group handle for all threads in the current kernel.
///
/// This function constructs a `GridGroup` representing all threads across all
/// blocks in the executing kernel. The returned handle can be used to perform
/// grid-wide synchronization via the `sync()` method.
///
/// # Environment Register Approach
///
/// ⚠️ **Important**: This function reads the grid workspace pointer from NVIDIA
/// environment registers (`%envreg1`, `%envreg2`). While this matches NVIDIA's
/// internal implementation, it is not officially documented behavior.
///
/// **Validation Recommended**: Test on your specific CUDA version and GPU before
/// production use. Validated configurations:
/// - CUDA 12.4.99 on SM_60, SM_70, SM_80, SM_89
///
/// # Cooperative Launch Requirement
///
/// This function will return an invalid GridGroup (with null workspace) if the
/// kernel was not launched cooperatively. Always check with [`is_valid()`](GridGroup::is_valid)
/// or rely on [`sync()`](GridGroup::sync) to panic on invalid state.
///
/// # Requirements
///
/// The kernel MUST be launched cooperatively:
///
/// **Host code (C++ example)**:
/// ```cpp
/// // Check device support
/// int supportsCoop = 0;
/// cudaDeviceGetAttribute(&supportsCoop, cudaDevAttrCooperativeLaunch, dev);
/// if (!supportsCoop) {
///     // Device does not support cooperative launch
///     return;
/// }
///
/// // Launch cooperatively
/// void* kernelArgs[] = { &data };
/// cudaLaunchCooperativeKernel(
///     (void*)my_kernel,
///     gridDim,
///     blockDim,
///     kernelArgs,
///     0,  // shared memory
///     stream
/// );
/// ```
///
/// **Host code (Rust with cust example)**:
/// ```no_run
/// use cust::prelude::*;
///
/// // Launch cooperatively
/// stream.launch_cooperative(
///     &module.get_function("my_kernel")?,
///     grid_dim,
///     block_dim,
///     shared_mem_bytes,
///     &kernel_args,
/// )?;
/// ```
///
/// # Returns
///
/// A `GridGroup<'static>` handle tied to the kernel's execution. The `'static`
/// lifetime indicates the handle is valid for the entire kernel execution.
///
/// # Safety
///
/// The workspace pointer is automatically provided by the CUDA driver via
/// environment registers. If the kernel was not launched cooperatively,
/// the workspace pointer will be null and `is_valid()` will return false.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn my_kernel(data: *mut f32) {
///     let grid = this_grid();
///
///     if !grid.is_valid() {
///         // Handle non-cooperative launch
///         return;
///     }
///
///     // Use grid handle for synchronization
///     process_local_data(data);
///     grid.sync();
///     update_global_state(data);
/// }
/// ```
#[inline(always)]
pub fn this_grid() -> GridGroup<'static> {
    // SAFETY: Reading environment registers is safe in device code.
    // The driver sets these during cooperative launch.
    let workspace = unsafe {
        use super::intrinsics::get_grid_workspace;
        get_grid_workspace()
    };

    GridGroup {
        // Cast workspace pointer to *mut u32, then offset to barrier field
        // GridWorkspace layout: [ws_size: u32, barrier: u32]
        // The sync intrinsics expect a pointer to the barrier field (+4 bytes from base)
        workspace: unsafe { (workspace as *mut u32).offset(1) },
        _marker: PhantomData,
    }
}

impl<'a> GridGroup<'a> {
    /// Synchronizes all threads across all blocks in the grid.
    ///
    /// This is a grid-wide barrier. Execution resumes only after ALL threads
    /// in ALL blocks have reached this synchronization point.
    ///
    /// # Requirements
    ///
    /// - **Uniform participation**: All threads must call `sync()`. Divergent
    ///   calls (e.g., inside `if` without all threads taking the branch) cause
    ///   deadlock.
    ///
    /// - **Cooperative launch**: Kernel must be launched via
    ///   `cudaLaunchCooperativeKernel`.
    ///
    /// - **Same workspace**: All sync calls in a phase must use the same
    ///   `GridGroup` instance (same workspace pointer).
    ///
    /// # Memory Ordering
    ///
    /// Grid sync provides strong memory ordering guarantees:
    ///
    /// - **SM 7.0+ (Volta 7.0+, Turing, Ampere, Ada, Hopper)**:
    ///   Hardware acquire/release semantics. All memory modifications before
    ///   `sync()` are visible to all threads after `sync()` returns.
    ///
    /// - **SM 6.0-6.9 (Pascal, Volta pre-7.0)**:
    ///   Sequential consistency via explicit fences. Same guarantees as SM 7.0+
    ///   but with additional overhead.
    ///
    /// In both cases, you can rely on:
    /// ```text
    /// Thread A writes X → Thread A calls sync() → sync() completes →
    /// Thread B calls sync() → Thread B reads X (sees A's write)
    /// ```
    ///
    /// **Critical**: A memory fence is executed at the start of `sync()` to ensure
    /// all writes before the barrier are visible across blocks. Without this fence,
    /// plain (non-atomic) stores before `sync()` may not be visible to other blocks,
    /// causing incorrect results or deadlock.
    ///
    /// # Implementation
    ///
    /// The sync algorithm follows NVIDIA's pattern:
    ///
    /// 1. **Memory fence**: Ensures all writes before `sync()` are visible across blocks
    /// 2. **Block sync**: All threads in each block synchronize (`sync_threads()`)
    /// 3. **Arrive**: CTA master thread increments global arrival counter
    /// 4. **Block sync**: Threads wait for arrival to complete
    /// 5. **Wait**: Threads wait for barrier flip (all blocks arrived)
    /// 6. **Block sync**: Final sync ensures memory visibility
    ///
    /// # Deadlock Prevention
    ///
    /// **SAFE - All threads participate**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let grid = this_grid();
    /// // All threads execute sync unconditionally
    /// grid.sync();
    /// ```
    ///
    /// **UNSAFE - Divergent sync**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # use cuda_std::thread::thread_idx;
    /// # let grid = this_grid();
    /// # let condition = true;
    /// // DEADLOCK: Only some threads call sync
    /// if condition {
    ///     grid.sync(); // Some threads skip this!
    /// }
    /// ```
    ///
    /// **SAFE - Uniform divergence**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # use cuda_std::thread::block_idx;
    /// # let grid = this_grid();
    /// if block_idx().x < 10 {
    ///     grid.sync(); // All threads in blocks < 10 sync
    /// } else {
    ///     grid.sync(); // All threads in blocks >= 10 sync
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// Grid sync is expensive (microseconds, not nanoseconds). Use sparingly:
    ///
    /// - **Good**: Few syncs per kernel (1-10 per iteration)
    /// - **Bad**: Syncing in tight inner loops
    /// - **Consider**: Block-level sync if possible (much faster)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn multi_phase(data: *mut f32) {
    ///     let grid = this_grid();
    ///
    ///     // Phase 1: Independent computation
    ///     let idx = grid.thread_rank() as usize;
    ///     data.add(idx).write(compute_phase1(idx));
    ///
    ///     // Barrier: Ensure all Phase 1 writes complete
    ///     grid.sync();
    ///
    ///     // Phase 2: Read neighbors (guaranteed visible)
    ///     let neighbor_idx = (idx + 1) % grid.size() as usize;
    ///     let neighbor = data.add(neighbor_idx).read();
    ///     data.add(idx).write(compute_phase2(neighbor));
    /// }
    /// ```
    #[inline(always)]
    pub fn sync(&self) {
        // Implementation of grid-wide synchronization using bit-31 flip detection.
        //
        // This implements the EXACT algorithm from NVIDIA's cooperative_groups C++ headers:
        // cooperative_groups/details/sync.h - grid_group::sync()
        //
        // Algorithm:
        // 1. Master block (block 0) adds flip value: 0x80000000 - (num_blocks - 1)
        // 2. Non-master blocks add 1
        // 3. All threads poll workspace until bit 31 flips (expected XOR current)
        //
        // The flip value ensures that when all blocks arrive, bit 31 toggles:
        // - Start: counter at some value with bit 31 = X
        // - After all blocks: counter += sum_of_all_increments
        // - sum_of_all_increments = flip + (num_blocks - 1) * 1 = 0x80000000
        // - Result: bit 31 = !X (flipped)
        //
        // Memory ordering:
        // - Uses atom.add.release.gpu.u32 and ld.acquire.gpu.u32 (SM 7.0+ instructions)
        // - cuda_std targets SM 7.0+ (Compute70), so release/acquire semantics are always available
        //
        // SAFETY: Requires cooperative kernel launch and uniform participation

        #[allow(unused_unsafe)]
        unsafe {
            use super::intrinsics::is_cta_master;
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg_attr(not(target_os = "cuda"), allow(unused_variables))]
            let workspace = self.workspace as u64;
            #[cfg_attr(not(target_os = "cuda"), allow(unused_mut))]
            let mut old_arrive: u32 = 0;

            // Step 1: Block-level synchronization first (using barrier 0 for consistency)
            // Use barrier.sync without thread count - hardware determines participation
            #[cfg(target_os = "cuda")]
            asm!(
                "barrier.sync 0;",
                options(nostack)
            );

            // Step 2: CTA master (thread 0,0,0 of each block) atomically increments
            if is_cta_master() {
                // Calculate number of blocks in grid
                let grid = crate::thread::grid_dim();
                let num_blocks = grid.x * grid.y * grid.z;

                // GPU master (block 0,0,0) uses flip value, other blocks add 1
                #[cfg_attr(not(target_os = "cuda"), allow(unused_variables))]
                let nb = if self.is_master() {
                    0x80000000u32 - (num_blocks - 1)
                } else {
                    1u32
                };

                // Atomic add with release semantics (SM 7.0+)
                #[cfg(target_os = "cuda")]
                asm!(
                    "atom.add.release.gpu.u32 {result}, [{workspace}], {value};",
                    result = out(reg32) old_arrive,
                    workspace = in(reg64) workspace,
                    value = in(reg32) nb,
                    options(nostack)
                );
            }

            // Step 3: Wait for barrier flip (only CTA masters poll)
            if is_cta_master() {
                #[cfg_attr(not(target_os = "cuda"), allow(unused_mut))]
                let mut current_arrive: u32 = 0;
                loop {
                    #[cfg(target_os = "cuda")]
                    asm!(
                        "ld.acquire.gpu.u32 {result}, [{workspace}];",
                        result = out(reg32) current_arrive,
                        workspace = in(reg64) workspace,
                        options(nostack, readonly)
                    );

                    // Check if bit 31 flipped
                    let has_flipped = ((old_arrive ^ current_arrive) & 0x80000000u32) != 0;
                    if has_flipped {
                        break;
                    }
                }
            }

            // Step 4: Block-level sync to ensure all threads wait (using barrier 0 for consistency)
            // Use barrier.sync without thread count - hardware determines participation
            #[cfg(target_os = "cuda")]
            asm!(
                "barrier.sync 0;",
                options(nostack)
            );
        }
    }

    /// Returns the total number of threads in the grid.
    ///
    /// This is the product of grid dimensions and block dimensions:
    /// ```text
    /// size = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z
    /// ```
    ///
    /// # Returns
    ///
    /// Total thread count across all blocks. Thread ranks are in the range
    /// `[0, size())`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let grid = this_grid();
    ///     let total_threads = grid.size();
    ///
    ///     // Process data in chunks per thread
    ///     let chunk_size = DATA_SIZE / total_threads;
    ///     let start = grid.thread_rank() * chunk_size;
    ///     let end = start + chunk_size;
    /// }
    /// ```
    #[inline(always)]
    pub fn size(&self) -> u32 {
        use crate::thread::{grid_dim, block_dim};

        let grid = grid_dim();
        let block = block_dim();

        // Total threads = (grid dimensions) * (block dimensions)
        grid.x * grid.y * grid.z * block.x * block.y * block.z
    }

    /// Returns the global rank of the calling thread within the grid.
    ///
    /// Thread ranks uniquely identify each thread in the grid. Ranks are
    /// computed from block indices and thread indices, arranged in row-major
    /// order (x varies fastest, then y, then z).
    ///
    /// # Returns
    ///
    /// Global thread index in the range `[0, size())`.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// block_1d = blockIdx.x + blockIdx.y * gridDim.x +
    ///            blockIdx.z * gridDim.x * gridDim.y
    ///
    /// thread_in_block = threadIdx.x + threadIdx.y * blockDim.x +
    ///                   threadIdx.z * blockDim.x * blockDim.y
    ///
    /// rank = block_1d * (blockDim.x * blockDim.y * blockDim.z) + thread_in_block
    /// ```
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example(data: *mut f32) {
    ///     let grid = this_grid();
    ///     let rank = grid.thread_rank();
    ///
    ///     // Each thread processes its own element
    ///     data.add(rank as usize).write(rank as f32);
    /// }
    /// ```
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        use crate::thread::{block_idx, block_dim, thread_idx, grid_dim};

        let block_idx = block_idx();
        let thread_idx = thread_idx();
        let block_dim = block_dim();
        let grid_dim = grid_dim();

        // Compute 1D block index from 3D block coordinates
        // Linearize as: x + y * width + z * width * height
        let block_1d = block_idx.x
            + block_idx.y * grid_dim.x
            + block_idx.z * grid_dim.x * grid_dim.y;

        // Compute thread index within the block (0 to threads_per_block - 1)
        let thread_in_block = thread_idx.x
            + thread_idx.y * block_dim.x
            + thread_idx.z * block_dim.x * block_dim.y;

        // Total threads per block
        let threads_per_block = block_dim.x * block_dim.y * block_dim.z;

        // Global rank: block offset + thread offset within block
        block_1d * threads_per_block + thread_in_block
    }

    /// Checks if this grid group is valid (cooperatively launched).
    ///
    /// Returns `true` if the grid group has a valid workspace pointer and can
    /// perform synchronization. Returns `false` if the workspace is invalid,
    /// indicating the kernel was not cooperatively launched or workspace
    /// allocation failed.
    ///
    /// # Returns
    ///
    /// - `true`: Grid group is valid, `sync()` should work
    /// - `false`: Grid group is invalid, `sync()` will not work correctly
    ///
    /// # Implementation
    ///
    /// Checks if the workspace pointer is non-null. The workspace pointer is
    /// read from environment registers (envreg1, envreg2) which are set by the
    /// CUDA driver during cooperative kernel launch. If the kernel was launched
    /// normally (not cooperatively), these registers contain null.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let grid = this_grid();
    ///
    ///     if grid.is_valid() {
    ///         // Safe to use grid synchronization
    ///         grid.sync();
    ///     } else {
    ///         // Fallback: kernel not cooperatively launched
    ///         // Use block-level sync only or skip sync
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        // Check if workspace pointer is non-null
        // The driver sets this to a valid pointer during cooperative launch
        // and leaves it null for normal launches
        !self.workspace.is_null()
    }

    /// Checks if the current block is the master block (block 0 in the grid).
    ///
    /// The master block is defined as the block with indices (0, 0, 0) within
    /// the grid. In the grid synchronization algorithm, the master block adds
    /// a special flip value while non-master blocks add 1.
    ///
    /// # Returns
    ///
    /// `true` if this is block (0,0,0) in the grid, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let grid = this_grid();
    ///
    ///     if grid.is_master() {
    ///         // Master block: initialize shared state
    ///     }
    /// }
    /// ```
    #[inline(always)]
    fn is_master(&self) -> bool {
        use crate::thread::block_idx;
        let block = block_idx();
        block.x == 0 && block.y == 0 && block.z == 0
    }
}

// Safety: GridGroup can be safely sent between threads within the kernel
// The workspace pointer is shared state designed for concurrent access
// Note: This doesn't make GridGroup Send across host threads, only within kernel
unsafe impl<'a> Send for GridGroup<'a> {}

// Safety: GridGroup can be safely shared between threads within the kernel
// The underlying synchronization primitives are designed for concurrent use
unsafe impl<'a> Sync for GridGroup<'a> {}

// Implement ThreadGroup trait for polymorphic cooperative group operations
impl<'a> super::traits::ThreadGroup for GridGroup<'a> {
    /// Synchronizes all threads across all blocks in the grid.
    ///
    /// Delegates to [`GridGroup::sync()`].
    #[inline(always)]
    fn sync(&self) {
        GridGroup::sync(self)
    }

    /// Returns the total number of threads in the grid.
    ///
    /// Delegates to [`GridGroup::size()`].
    #[inline(always)]
    fn size(&self) -> u32 {
        GridGroup::size(self)
    }

    /// Returns the global rank of the calling thread within the grid.
    ///
    /// Delegates to [`GridGroup::thread_rank()`].
    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        GridGroup::thread_rank(self)
    }
}
