//! Tiled partition synchronization primitives for warp-level operations.
//!
//! This module provides thread block tiles for fine-grained synchronization within
//! warps. Tiled groups enable sub-block cooperation patterns at warp granularity.
//!
//! # Overview
//!
//! The `TiledGroup<SIZE>` type represents a tile of threads partitioned from a parent
//! thread block. Tiles enable warp-level synchronization and communication primitives
//! like shuffle operations.
//!
//! # Warp Tiles
//!
//! A warp tile is a group of threads that can synchronize independently of the full
//! thread block. The most common tile size is 32 threads (full warp).
//!
//! # Requirements
//!
//! Tiled synchronization has requirements:
//!
//! 1. **Uniform Participation**: ALL threads in the tile must call `sync()`.
//!    Divergent sync calls cause deadlock.
//!
//! 2. **Warp-Level Scope**: `TiledGroup<32>` synchronizes threads within the
//!    same warp. Threads in different warps execute independently.
//!
//! # Example Usage
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn warp_reduction(data: *mut f32, output: *mut f32) {
//!     let block = this_thread_block();
//!     let tile = tiled_partition::<32>(&block);
//!
//!     // Phase 1: Each thread loads its data
//!     let thread_data = *data.add(tile.thread_rank() as usize);
//!
//!     // Synchronize: ensure all threads in tile loaded data
//!     tile.sync();
//!
//!     // Phase 2: Warp-level reduction
//!     let result = warp_reduce(thread_data, &tile);
//!
//!     // First thread in tile writes result
//!     if tile.thread_rank() == 0 {
//!         *output.add((thread::index() / 32) as usize) = result;
//!     }
//! }
//! ```
//!
//! # Memory Ordering
//!
//! Warp sync provides memory ordering guarantees:
//!
//! - All memory operations before `sync()` are visible to all threads in the tile
//!   after `sync()` returns.
//! - Uses warp-level barrier instruction (`bar.warp.sync`)
//!
//! # Common Patterns
//!
//! ## Warp-Level Reduction
//!
//! ```no_run
//! # use cuda_std::cooperative_groups::*;
//! # #[kernel]
//! # pub unsafe fn example(data: *const f32, output: *mut f32) {
//! let block = this_thread_block();
//! let tile = tiled_partition::<32>(&block);
//!
//! // Load data for this thread
//! let mut value = *data.add(thread::index() as usize);
//!
//! // Warp reduction using tile sync
//! let mut offset = tile.size() / 2;
//! while offset > 0 {
//!     tile.sync();
//!     // Shuffle operations would go here
//!     offset /= 2;
//! }
//!
//! // First thread writes result
//! if tile.thread_rank() == 0 {
//!     *output.add((thread::index() / 32) as usize) = value;
//! }
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - **Very fast synchronization**: Warp-level sync is faster than block-level sync
//! - **No shared memory**: Warp operations can use registers and shuffle instead
//! - **Lock-step execution**: Threads in a warp execute in lock-step on most operations
//!
//! # Safety
//!
//! The public API is safe because:
//! - `TiledGroup` lifetime is tied to kernel scope (cannot escape)
//! - Type system prevents misuse of tile handles
//! - Unsafe operations are encapsulated
//!
//! However, incorrect usage patterns can still cause:
//! - **Deadlock**: If not all threads in tile call `sync()`
//! - **Race conditions**: If sync is placed incorrectly

use super::thread_block::ThreadBlock;

/// A handle to a tile of threads partitioned from a thread block.
///
/// `TiledGroup<SIZE>` represents a subset of threads within a thread block,
/// typically sized to match warp boundaries (SIZE = 32). It provides the
/// `sync()` method for warp-level synchronization.
///
/// # Type Parameter
///
/// - `SIZE`: The number of threads in the tile. Must be a power of 2 and
///   divide evenly into the warp size. Common values: 32 (full warp).
///
/// # Construction
///
/// Create via the `tiled_partition::<SIZE>()` function:
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
/// let block = this_thread_block();
/// let tile = tiled_partition::<32>(&block);
/// ```
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn cooperative_warp_kernel(data: *mut i32) {
///     let block = this_thread_block();
///     let tile = tiled_partition::<32>(&block);
///
///     // Phase 1: computation
///     data.add(tile.thread_rank() as usize).write(compute_value());
///
///     // Barrier: ensure all threads in tile finished Phase 1
///     tile.sync();
///
///     // Phase 2: all threads in tile see Phase 1 results
///     let neighbor = data.add(((tile.thread_rank() + 1) % tile.size()) as usize).read();
///     process_neighbor(neighbor);
/// }
/// ```
#[repr(C)]
pub struct TiledGroup<const SIZE: u32> {
    /// Thread participation mask for this tile.
    /// For SIZE=32 (full warp), mask = 0xffffffff.
    mask: u32,

    /// Thread rank within the tile [0, SIZE).
    /// Computed as lane_id % SIZE.
    rank: u32,
}

/// Creates a tiled partition of threads from a parent thread block.
///
/// This function partitions the parent thread block into tiles of SIZE threads.
/// Threads within the same tile can synchronize independently using warp-level
/// primitives.
///
/// # Type Parameter
///
/// - `SIZE`: Number of threads per tile. Must be a power of 2 and divide
///   evenly into the warp size (32). Currently only SIZE=32 is implemented.
///
/// # Arguments
///
/// - `parent`: The parent thread block to partition
///
/// # Returns
///
/// A `TiledGroup<SIZE>` handle representing this thread's tile.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn my_kernel(data: *mut f32) {
///     let block = this_thread_block();
///     let tile = tiled_partition::<32>(&block);
///
///     // Use tile for warp-level synchronization
///     process_local_data(data);
///     tile.sync();
///     update_shared_state(data);
/// }
/// ```
#[inline(always)]
pub fn tiled_partition<const SIZE: u32>(_parent: &ThreadBlock) -> TiledGroup<SIZE> {
    use crate::thread::thread_idx_x;

    // Get lane ID within warp (0-31)
    let lane_id = thread_idx_x() % 32;

    // For SIZE=32 (full warp), all threads participate
    // mask = 0xffffffff, rank = lane_id
    let mask = if SIZE == 32 {
        0xffffffff_u32
    } else {
        // Future: Support other tile sizes (16, 8, 4, 2, 1)
        // For now, panic on unsupported sizes
        panic!("Only SIZE=32 is currently supported")
    };

    let rank = lane_id % SIZE;

    TiledGroup { mask, rank }
}

impl<const SIZE: u32> TiledGroup<SIZE> {
    /// Synchronizes all threads within the tile.
    ///
    /// This is a warp-level barrier. Execution resumes only after ALL threads
    /// in the tile have reached this synchronization point.
    ///
    /// # Requirements
    ///
    /// - **Uniform participation**: All threads in the tile must call `sync()`.
    ///   Divergent calls cause deadlock.
    ///
    /// # Memory Ordering
    ///
    /// Warp sync provides strong memory ordering guarantees:
    /// - All memory modifications before `sync()` are visible to all threads
    ///   in the tile after `sync()` returns.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `bar.warp.sync` instruction with the tile's participation mask.
    ///
    /// CRITICAL DIFFERENCE from block sync:
    /// - Block uses: `bar.sync 0, {thread_count}`
    /// - Warp uses: `bar.warp.sync {mask}`
    ///
    /// The bar.warp.sync instruction uses a mask to identify which threads must
    /// participate in the synchronization.
    ///
    /// # Deadlock Prevention
    ///
    /// **SAFE - All threads in tile participate**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let block = this_thread_block();
    /// # let tile = tiled_partition::<32>(&block);
    /// // All threads in tile execute sync unconditionally
    /// tile.sync();
    /// ```
    ///
    /// **UNSAFE - Divergent sync**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let block = this_thread_block();
    /// # let tile = tiled_partition::<32>(&block);
    /// # let condition = true;
    /// // DEADLOCK: Only some threads in tile call sync
    /// if condition {
    ///     tile.sync(); // Some threads skip this!
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// Warp-level sync is extremely fast (sub-nanosecond). Use it freely.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn warp_kernel(data: *mut f32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     // Load data
    ///     let value = *data.add(tile.thread_rank() as usize);
    ///
    ///     // Barrier: ensure all loads complete
    ///     tile.sync();
    ///
    ///     // Process with tile-level operations
    ///     let result = process(value);
    ///     *data.add(tile.thread_rank() as usize) = result;
    /// }
    /// ```
    #[inline(always)]
    pub fn sync(&self) {
        // Warp-level synchronization using bar.warp.sync instruction.
        //
        // CRITICAL DIFFERENCE from block.sync():
        // - Block uses: bar.sync 0, {thread_count}
        // - Warp uses: bar.warp.sync {mask}
        //
        // The bar.warp.sync instruction requires a mask parameter that identifies
        // which threads must participate in the synchronization. For a full warp
        // tile (SIZE=32), the mask is 0xffffffff.
        //
        // PTX Instruction:
        //   bar.warp.sync mask;
        //
        // Where mask identifies the participating threads in the warp.

        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            // Execute warp-level barrier with mask
            #[cfg(target_os = "cuda")]
            asm!(
                "bar.warp.sync {mask};",
                mask = in(reg32) self.mask,
                options(nostack)
            );
        }
    }

    /// Returns the total number of threads in the tile.
    ///
    /// # Returns
    ///
    /// The tile size (SIZE constant). Thread ranks are in the range [0, SIZE).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     assert_eq!(tile.size(), 32);
    /// }
    /// ```
    #[inline(always)]
    pub fn size(&self) -> u32 {
        SIZE
    }

    /// Returns the rank of the calling thread within the tile.
    ///
    /// Thread ranks uniquely identify each thread within the tile. Ranks are
    /// computed from the lane ID within the warp.
    ///
    /// # Returns
    ///
    /// Thread index within the tile in the range [0, SIZE).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let rank = tile.thread_rank();
    ///     // rank is in range [0, 32)
    /// }
    /// ```
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        self.rank
    }
}

// Safety: TiledGroup can be safely sent between threads within the kernel
// The tile handle contains no mutable state
unsafe impl<const SIZE: u32> Send for TiledGroup<SIZE> {}

// Safety: TiledGroup can be safely shared between threads within the kernel
// The underlying synchronization primitives are designed for concurrent use
unsafe impl<const SIZE: u32> Sync for TiledGroup<SIZE> {}

// Implement ThreadGroup trait for polymorphic cooperative group operations
impl<const SIZE: u32> super::traits::ThreadGroup for TiledGroup<SIZE> {
    /// Synchronizes all threads within the tile.
    ///
    /// Delegates to [`TiledGroup::sync()`].
    #[inline(always)]
    fn sync(&self) {
        TiledGroup::sync(self)
    }

    /// Returns the total number of threads in the tile.
    ///
    /// Delegates to [`TiledGroup::size()`].
    #[inline(always)]
    fn size(&self) -> u32 {
        TiledGroup::size(self)
    }

    /// Returns the rank of the calling thread within the tile.
    ///
    /// Delegates to [`TiledGroup::thread_rank()`].
    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        TiledGroup::thread_rank(self)
    }
}
