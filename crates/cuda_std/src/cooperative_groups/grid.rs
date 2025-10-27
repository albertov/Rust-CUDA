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
    /// Workspace pointer for barrier coordination.
    ///
    /// This pointer is used by the grid synchronization intrinsics to coordinate
    /// arrival counts across blocks. It points to a GridWorkspace structure
    /// allocated by the CUDA driver during cooperative kernel launch.
    ///
    /// The driver automatically loads this pointer into environment registers
    /// (envreg1 for high 32 bits, envreg2 for low 32 bits) before kernel execution.
    workspace: *mut u32,

    /// Phantom data to tie lifetime to kernel scope.
    ///
    /// Prevents `GridGroup` from escaping the kernel, ensuring it's only used
    /// during valid kernel execution. The lifetime is typically `'static` since
    /// `this_grid()` returns `GridGroup<'static>`, but the type parameter allows
    /// for future flexibility.
    _marker: PhantomData<&'a ()>,
}

/// Creates a grid group handle for all threads in the current kernel.
///
/// This function constructs a `GridGroup` representing all threads across all
/// blocks in the executing kernel. The returned handle can be used to perform
/// grid-wide synchronization via the `sync()` method.
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
        workspace: workspace as *mut u32, // Cast to u32 for barrier field access
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
    /// # Implementation
    ///
    /// The sync algorithm follows NVIDIA's pattern:
    ///
    /// 1. **Block sync**: All threads in each block synchronize (`sync_threads()`)
    /// 2. **Arrive**: CTA master thread increments global arrival counter
    /// 3. **Block sync**: Threads wait for arrival to complete
    /// 4. **Wait**: Threads wait for barrier flip (all blocks arrived)
    /// 5. **Block sync**: Final sync ensures memory visibility
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
        use crate::thread::sync_threads;
        use super::intrinsics::{sync_grids_arrive, sync_grids_wait, is_cta_master};

        // Step 1: Block-level sync ensures all threads in block are ready
        // This prevents races between threads in the same block
        sync_threads();

        // Step 2: One thread per block arrives at the grid barrier
        // The CTA master atomically increments the global arrival counter
        // Non-master threads don't participate in arrival
        let old_arrive = if is_cta_master() {
            unsafe { sync_grids_arrive(self.workspace) }
        } else {
            // Non-master threads return dummy value (not used in wait)
            0
        };

        // Step 3: Block-level sync to share the arrival state
        // Ensures the old_arrive value is visible to all threads in the block
        // (Though in practice, only the CTA master's value matters)
        sync_threads();

        // Step 4: Wait for the grid barrier to flip
        // All threads wait, though the intrinsic only polls in the CTA master
        // This ensures uniform control flow across the block
        unsafe { sync_grids_wait(old_arrive, self.workspace) };

        // Step 5: Block-level sync after barrier ensures memory visibility
        // Guarantees that all memory writes from other blocks are visible
        sync_threads();
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
}

// Safety: GridGroup can be safely sent between threads within the kernel
// The workspace pointer is shared state designed for concurrent access
// Note: This doesn't make GridGroup Send across host threads, only within kernel
unsafe impl<'a> Send for GridGroup<'a> {}

// Safety: GridGroup can be safely shared between threads within the kernel
// The underlying synchronization primitives are designed for concurrent use
unsafe impl<'a> Sync for GridGroup<'a> {}
