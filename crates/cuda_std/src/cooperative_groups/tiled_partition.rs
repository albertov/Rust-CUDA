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

/// Computes the thread participation mask for a tile within a warp.
///
/// For a tile of size N at position tile_id within a 32-thread warp,
/// this function generates a bitmask where bits corresponding to threads
/// in the tile are set to 1.
///
/// # Arguments
///
/// - `size`: Number of threads per tile (must be power of 2, ≤ 32)
/// - `tile_id`: Which tile within the warp (0 to 32/size - 1)
///
/// # Returns
///
/// A u32 bitmask for threads participating in this tile.
///
/// # Examples
///
/// ```ignore
/// compute_tile_mask(32, 0) = 0xFFFFFFFF  // All threads (full warp)
/// compute_tile_mask(16, 0) = 0x0000FFFF  // Threads 0-15
/// compute_tile_mask(16, 1) = 0xFFFF0000  // Threads 16-31
/// compute_tile_mask(8, 2)  = 0x00FF0000  // Threads 16-23
/// compute_tile_mask(4, 3)  = 0x0000F000  // Threads 12-15
/// compute_tile_mask(2, 7)  = 0x0000C000  // Threads 14-15
/// compute_tile_mask(1, 15) = 0x00008000  // Thread 15 only
/// ```
#[inline(always)]
const fn compute_tile_mask(size: u32, tile_id: u32) -> u32 {
    // Special case: full warp (size=32) must return 0xFFFFFFFF
    // Cannot compute via (1u32 << 32) - 1 because shifting by 32 bits is UB
    if size == 32 {
        return 0xFFFFFFFF;
    }

    // Create N consecutive 1-bits: (1 << N) - 1
    // For size=8: (1 << 8) - 1 = 0xFF
    let ones = (1u32 << size) - 1;

    // Shift to correct position within warp
    // For size=8, tile_id=2: 0xFF << 16 = 0x00FF0000
    ones << (tile_id * size)
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
///   evenly into the warp size (32). Supported sizes: 1, 2, 4, 8, 16, 32.
///
/// # Arguments
///
/// - `parent`: The parent thread block to partition
///
/// # Returns
///
/// A `TiledGroup<SIZE>` handle representing this thread's tile.
///
/// # Panics
///
/// Panics (in debug builds) if SIZE is not a power of 2 or does not divide 32.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn my_kernel(data: *mut f32) {
///     let block = this_thread_block();
///
///     // Full warp tile
///     let tile32 = tiled_partition::<32>(&block);
///
///     // Half warp tiles
///     let tile16 = tiled_partition::<16>(&block);
///
///     // Quarter warp tiles
///     let tile8 = tiled_partition::<8>(&block);
///
///     // Use tile for warp-level synchronization
///     process_local_data(data);
///     tile32.sync();
///     update_shared_state(data);
/// }
/// ```
#[inline(always)]
pub fn tiled_partition<const SIZE: u32>(_parent: &ThreadBlock) -> TiledGroup<SIZE> {
    use crate::thread::thread_idx_x;

    // Validate SIZE at compile-time where possible, runtime otherwise
    debug_assert!(SIZE > 0, "Tile size must be greater than 0");
    debug_assert!(SIZE <= 32, "Tile size cannot exceed warp size (32)");
    debug_assert!(SIZE.is_power_of_two(), "Tile size must be a power of 2");
    debug_assert!(32 % SIZE == 0, "Tile size must divide warp size (32)");

    // Get lane ID within warp (0-31)
    let lane_id = thread_idx_x() % 32;

    // Calculate which tile this thread belongs to
    let tile_id = lane_id / SIZE;

    // Calculate rank within tile
    let rank = lane_id % SIZE;

    // Compute participation mask for this tile
    let mask = compute_tile_mask(SIZE, tile_id);

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

    /// Broadcasts a value from a specific source lane to all threads in the tile.
    ///
    /// This shuffle operation reads the value of `var` from the thread with rank
    /// `src_lane` and returns it to all threads in the tile. This is useful for
    /// broadcasting a single value computed by one thread to all threads.
    ///
    /// # Arguments
    ///
    /// - `var`: The value to broadcast (from the source lane's perspective)
    /// - `src_lane`: The rank of the thread whose value will be broadcast [0, SIZE)
    ///
    /// # Returns
    ///
    /// The value of `var` from the thread with rank `src_lane`.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `shfl.sync.idx.b32` instruction for indexed shuffle.
    /// The `.sync` variant ensures proper synchronization across the warp.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn broadcast_example(data: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     // Thread 0 computes a value
    ///     let my_value = if tile.thread_rank() == 0 {
    ///         42
    ///     } else {
    ///         0
    ///     };
    ///
    ///     // Broadcast thread 0's value to all threads
    ///     let broadcast_value = tile.shfl(my_value, 0);
    ///
    ///     // All threads now have 42
    ///     *data.add(tile.thread_rank() as usize) = broadcast_value;
    /// }
    /// ```
    #[inline(always)]
    pub fn shfl(&self, var: i32, src_lane: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Max lane parameter should be SIZE - 1 for proper tile boundaries
                let max_lane = SIZE - 1;
                asm!(
                    "shfl.sync.idx.b32 {result}, {var}, {src}, {max_lane}, {mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    src = in(reg32) src_lane,
                    max_lane = in(reg32) max_lane,
                    mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Shifts values down within the tile (threads receive from higher lanes).
    ///
    /// Each thread receives the value of `var` from the thread `delta` lanes
    /// higher (rank + delta). Threads at the high end receive their own value
    /// when rank + delta >= SIZE. This is useful for implementing reduction
    /// patterns and prefix sums.
    ///
    /// # Arguments
    ///
    /// - `var`: The value to shift
    /// - `delta`: Number of lanes to shift by [0, SIZE)
    ///
    /// # Returns
    ///
    /// The value of `var` from thread (rank + delta), or own value if
    /// rank + delta >= SIZE.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `shfl.sync.down.b32` instruction for downward shuffle.
    /// The `.sync` variant ensures proper synchronization across the warp.
    ///
    /// # Example - Warp Reduction
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn warp_sum(data: *const i32, output: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     // Each thread loads its value
    ///     let mut value = *data.add(tile.thread_rank() as usize);
    ///
    ///     // Reduction: sum across warp using shfl_down
    ///     let mut offset = tile.size() / 2;
    ///     while offset > 0 {
    ///         value += tile.shfl_down(value, offset);
    ///         offset /= 2;
    ///     }
    ///
    ///     // Thread 0 writes the sum
    ///     if tile.thread_rank() == 0 {
    ///         *output = value;
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub fn shfl_down(&self, var: i32, delta: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Max lane parameter should be SIZE - 1 for proper tile boundaries
                let max_lane = SIZE - 1;
                asm!(
                    "shfl.sync.down.b32 {result}, {var}, {delta}, {max_lane}, {mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    delta = in(reg32) delta,
                    max_lane = in(reg32) max_lane,
                    mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Shifts values up within the tile (threads receive from lower lanes).
    ///
    /// Each thread receives the value of `var` from the thread `delta` lanes
    /// lower (rank - delta). Threads at the low end receive their own value
    /// when rank - delta < 0. This operation is the complement of shfl_down.
    ///
    /// # Arguments
    ///
    /// - `var`: The value to shift
    /// - `delta`: Number of lanes to shift by [0, SIZE)
    ///
    /// # Returns
    ///
    /// The value of `var` from thread (rank - delta), or own value if
    /// rank < delta.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `shfl.sync.up.b32` instruction for upward shuffle.
    /// The `.sync` variant ensures proper synchronization across the warp.
    /// Note: Uses max offset 0x0 (different from down/idx which use 0x1f).
    ///
    /// # Example - Prefix Sum
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn prefix_sum(data: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let rank = tile.thread_rank();
    ///     let mut value = *data.add(rank as usize);
    ///
    ///     // Parallel prefix sum using shfl_up
    ///     let mut offset = 1;
    ///     while offset < tile.size() {
    ///         let neighbor = tile.shfl_up(value, offset);
    ///         if rank >= offset {
    ///             value += neighbor;
    ///         }
    ///         offset *= 2;
    ///     }
    ///
    ///     *data.add(rank as usize) = value;
    /// }
    /// ```
    #[inline(always)]
    pub fn shfl_up(&self, var: i32, delta: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            asm!(
                "shfl.sync.up.b32 {result}, {var}, {delta}, 0x0, {mask};",
                result = out(reg32) result,
                var = in(reg32) var,
                delta = in(reg32) delta,
                mask = in(reg32) self.mask,
                options(nostack, nomem)
            );

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Butterfly shuffle using XOR-based lane addressing.
    ///
    /// Each thread receives the value of `var` from the thread at rank XOR lane_mask.
    /// This creates a butterfly communication pattern useful for parallel algorithms
    /// like FFT, bitonic sort, and tree-based reductions.
    ///
    /// # Arguments
    ///
    /// - `var`: The value to exchange
    /// - `lane_mask`: XOR mask to compute target lane [0, SIZE)
    ///
    /// # Returns
    ///
    /// The value of `var` from thread (rank XOR lane_mask).
    ///
    /// # Implementation
    ///
    /// Uses the PTX `shfl.sync.bfly.b32` instruction for butterfly shuffle.
    /// The `.sync` variant ensures proper synchronization across the warp.
    ///
    /// # Example - Butterfly Exchange
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn butterfly_exchange(data: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let rank = tile.thread_rank();
    ///     let mut value = *data.add(rank as usize);
    ///
    ///     // Exchange with XOR neighbor (mask=1 swaps even/odd pairs)
    ///     // Thread 0 ↔ Thread 1, Thread 2 ↔ Thread 3, etc.
    ///     let neighbor_value = tile.shfl_xor(value, 1);
    ///
    ///     // Process exchanged values
    ///     value = (value + neighbor_value) / 2;
    ///
    ///     *data.add(rank as usize) = value;
    /// }
    /// ```
    ///
    /// # Butterfly Patterns
    ///
    /// Different masks create different exchange patterns:
    /// - mask=1: Swap adjacent pairs (0↔1, 2↔3, 4↔5, ...)
    /// - mask=2: Swap pairs separated by 2 (0↔2, 1↔3, 4↔6, ...)
    /// - mask=4: Swap pairs separated by 4 (0↔4, 1↔5, 2↔6, ...)
    /// - mask=16: Swap halves of warp (0↔16, 1↔17, ..., 15↔31)
    #[inline(always)]
    pub fn shfl_xor(&self, var: i32, lane_mask: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Max lane parameter should be SIZE - 1 for proper tile boundaries
                let max_lane = SIZE - 1;
                asm!(
                    "shfl.sync.bfly.b32 {result}, {var}, {mask}, {max_lane}, {thread_mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    mask = in(reg32) lane_mask,
                    max_lane = in(reg32) max_lane,
                    thread_mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Checks if ANY thread in the tile has a true predicate.
    ///
    /// This is a warp-level vote operation that returns true if at least one
    /// thread in the tile has `predicate` set to true. All threads receive
    /// the same result regardless of their individual predicate values.
    ///
    /// # Arguments
    ///
    /// - `predicate`: Boolean value to test for this thread
    ///
    /// # Returns
    ///
    /// `true` if ANY thread in the tile has `predicate == true`,
    /// `false` if ALL threads have `predicate == false`.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `vote.sync.any.pred` instruction which operates on
    /// predicate registers. The implementation converts the Rust bool to
    /// a PTX predicate using `setp.ne.u32`, performs the vote, then
    /// converts the result predicate back to a bool.
    ///
    /// PTX sequence:
    /// ```ptx
    /// setp.ne.u32 %p1, {pred_val}, 0;      // Convert bool to predicate
    /// vote.sync.any.pred %p2, %p1, {mask}; // Vote across warp
    /// selp.u32 {result}, 1, 0, %p2;        // Convert predicate to bool
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Early exit detection**: Check if any thread needs more work
    /// - **Convergence testing**: Detect if any thread failed to converge
    /// - **Error propagation**: Check if any thread encountered an error
    /// - **Divergence analysis**: Detect warp divergence patterns
    ///
    /// # Example - Early Exit
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn iterative_solver(data: *mut f32, threshold: f32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     loop {
    ///         // Each thread does local work
    ///         let error = compute_error(*data.add(tile.thread_rank() as usize));
    ///
    ///         // Check if ANY thread needs more iterations
    ///         let needs_more_work = error > threshold;
    ///         if !tile.any(needs_more_work) {
    ///             // All threads converged - exit loop
    ///             break;
    ///         }
    ///
    ///         // Continue iterating
    ///         update_value(data.add(tile.thread_rank() as usize));
    ///     }
    /// }
    /// ```
    ///
    /// # Example - Divergent Execution
    ///
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let block = this_thread_block();
    /// let tile = tiled_partition::<32>(&block);
    ///
    /// // Only thread 0 has true predicate
    /// let my_predicate = tile.thread_rank() == 0;
    ///
    /// // All threads get true (because thread 0 has true)
    /// let any_true = tile.any(my_predicate);
    /// assert!(any_true);
    /// # }
    /// ```
    #[inline(always)]
    pub fn any(&self, predicate: bool) -> bool {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Convert bool to u32 for predicate conversion
                let pred_val: u32 = if predicate { 1 } else { 0 };

                // PTX vote.sync.any.pred requires predicate registers
                // 1. Convert bool → predicate using setp
                // 2. Vote across warp with vote.sync.any.pred
                // 3. Convert result predicate → bool using selp
                asm!(
                    "{{",
                    ".reg .pred %p_input, %p_result;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.any.pred %p_result, %p_input, {mask};",
                    "selp.u32 {result}, 1, 0, %p_result;",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result != 0
    }

    /// Checks if ALL threads in the tile have a true predicate.
    ///
    /// This is a warp-level vote operation that returns true if and only if
    /// ALL threads in the tile have `predicate` set to true. All threads
    /// receive the same result regardless of their individual predicate values.
    ///
    /// # Arguments
    ///
    /// - `predicate`: Boolean value to test for this thread
    ///
    /// # Returns
    ///
    /// `true` if ALL threads in the tile have `predicate == true`,
    /// `false` if ANY thread has `predicate == false`.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `vote.sync.all.pred` instruction which operates on
    /// predicate registers. The implementation converts the Rust bool to
    /// a PTX predicate using `setp.ne.u32`, performs the vote, then
    /// converts the result predicate back to a bool.
    ///
    /// PTX sequence:
    /// ```ptx
    /// setp.ne.u32 %p1, {pred_val}, 0;      // Convert bool to predicate
    /// vote.sync.all.pred %p2, %p1, {mask}; // Vote across warp
    /// selp.u32 {result}, 1, 0, %p2;        // Convert predicate to bool
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Convergence detection**: Check if all threads have converged
    /// - **Unanimous agreement**: Ensure all threads agree on a condition
    /// - **Completion detection**: Verify all threads finished their work
    /// - **Error checking**: Confirm no threads encountered errors
    ///
    /// # Example - Convergence Testing
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn iterative_algorithm(data: *mut f32, epsilon: f32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let mut converged = false;
    ///     while !converged {
    ///         // Each thread does local work
    ///         let error = update_value(data.add(tile.thread_rank() as usize));
    ///
    ///         // Check if ALL threads have converged
    ///         let local_converged = error < epsilon;
    ///         converged = tile.all(local_converged);
    ///
    ///         // Continue if any thread still needs more work
    ///     }
    /// }
    /// ```
    ///
    /// # Example - Unanimous Decision
    ///
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(values: *const f32) {
    /// let block = this_thread_block();
    /// let tile = tiled_partition::<32>(&block);
    ///
    /// // Each thread checks its value
    /// let my_value = *values.add(tile.thread_rank() as usize);
    /// let is_valid = my_value > 0.0;
    ///
    /// // All threads must agree (all values positive)
    /// if tile.all(is_valid) {
    ///     // Proceed with algorithm - all values are valid
    /// } else {
    ///     // At least one thread has invalid value
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn all(&self, predicate: bool) -> bool {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Convert bool to u32 for predicate conversion
                let pred_val: u32 = if predicate { 1 } else { 0 };

                // PTX vote.sync.all.pred requires predicate registers
                // 1. Convert bool → predicate using setp
                // 2. Vote across warp with vote.sync.all.pred
                // 3. Convert result predicate → bool using selp
                asm!(
                    "{{",
                    ".reg .pred %p_input, %p_result;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.all.pred %p_result, %p_input, {mask};",
                    "selp.u32 {result}, 1, 0, %p_result;",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result != 0
    }

    /// Collects votes from all threads as a bitmask.
    ///
    /// This is a warp-level vote operation that returns a 32-bit mask where
    /// bit N is set if thread N has `predicate == true`. All threads receive
    /// the same bitmask result.
    ///
    /// # Arguments
    ///
    /// - `predicate`: Boolean value to vote for this thread
    ///
    /// # Returns
    ///
    /// A `u32` bitmask where bit N is 1 if thread N has `predicate == true`,
    /// and 0 otherwise. For a 32-thread tile, all 32 bits may be used.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `vote.sync.ballot.b32` instruction which collects
    /// predicate values from all threads into a bitmask. The implementation
    /// converts the Rust bool to a PTX predicate, performs the ballot, and
    /// returns the u32 result directly.
    ///
    /// PTX sequence:
    /// ```ptx
    /// setp.ne.u32 %p1, {pred_val}, 0;         // Convert bool to predicate
    /// vote.sync.ballot.b32 {result}, %p1, {mask}; // Collect votes to bitmask
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Divergence analysis**: Identify which threads took which branch
    /// - **Compaction**: Build masks for stream compaction algorithms
    /// - **Warp-level reduction**: Count true predicates via popcount
    /// - **Bit-parallel algorithms**: Collect boolean arrays efficiently
    ///
    /// # Example - Count True Predicates
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn count_valid(data: *const f32, output: *mut u32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     // Each thread checks if its value is valid
    ///     let my_value = *data.add(tile.thread_rank() as usize);
    ///     let is_valid = my_value > 0.0;
    ///
    ///     // Collect all votes as bitmask
    ///     let ballot_mask = tile.ballot(is_valid);
    ///
    ///     // Thread 0 counts the valid values
    ///     if tile.thread_rank() == 0 {
    ///         let valid_count = ballot_mask.count_ones();
    ///         *output = valid_count;
    ///     }
    /// }
    /// ```
    ///
    /// # Example - Stream Compaction
    ///
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(data: *const i32, output: *mut i32) {
    /// let block = this_thread_block();
    /// let tile = tiled_partition::<32>(&block);
    ///
    /// let rank = tile.thread_rank();
    /// let value = *data.add(rank as usize);
    /// let is_valid = value != 0;
    ///
    /// // Get bitmask of valid threads
    /// let valid_mask = tile.ballot(is_valid);
    ///
    /// if is_valid {
    ///     // Count valid threads before this one (for output position)
    ///     let threads_before = valid_mask & ((1u32 << rank) - 1);
    ///     let output_pos = threads_before.count_ones();
    ///     *output.add(output_pos as usize) = value;
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn ballot(&self, predicate: bool) -> u32 {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // Convert bool to u32 for predicate conversion
                let pred_val: u32 = if predicate { 1 } else { 0 };

                // PTX vote.sync.ballot.b32 collects predicates to bitmask
                // 1. Convert bool → predicate using setp
                // 2. Collect votes across warp with vote.sync.ballot.b32
                // Result is u32 register (not predicate), so no selp needed
                asm!(
                    "{{",
                    ".reg .pred %p_input;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.ballot.b32 {result}, %p_input, {mask};",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result
    }

    /// Finds threads with matching values (SM 7.0+).
    ///
    /// This is a warp-level match operation that returns a bitmask indicating
    /// which threads have the same value as the calling thread. Each thread
    /// may receive a different result based on its value.
    ///
    /// # Arguments
    ///
    /// - `value`: The value to match against other threads
    ///
    /// # Returns
    ///
    /// A `u32` bitmask where bit M is set if thread M has the same value
    /// as the calling thread. The calling thread's own bit is always set.
    ///
    /// # Requirements
    ///
    /// - Requires SM 7.0+ (Volta architecture or newer)
    /// - On older architectures, this function will not be available
    ///
    /// # Implementation
    ///
    /// Uses the PTX `match.sync.any.b32` instruction which compares values
    /// across threads and returns a bitmask of matching threads.
    ///
    /// PTX instruction:
    /// ```ptx
    /// match.sync.any.b32 {result}, {value}, {mask};
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Value grouping**: Group threads with same values
    /// - **Deduplication**: Find unique values within a warp
    /// - **Consensus detection**: Identify threads with common values
    /// - **Divergence analysis**: Analyze value distributions
    ///
    /// # Example - Value Grouping
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn group_by_value(values: *const i32, output: *mut u32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let rank = tile.thread_rank();
    ///     let my_value = *values.add(rank as usize);
    ///
    ///     // Find all threads with the same value as this thread
    ///     let match_mask = tile.match_any(my_value);
    ///
    ///     // Thread with lowest rank in each group becomes leader
    ///     let is_leader = match_mask.trailing_zeros() == rank;
    ///
    ///     if is_leader {
    ///         let group_size = match_mask.count_ones();
    ///         *output.add(rank as usize) = group_size;
    ///     }
    /// }
    /// ```
    ///
    /// # Example - Find Unique Values
    ///
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(values: *const i32, unique: *mut i32, count: *mut u32) {
    /// let block = this_thread_block();
    /// let tile = tiled_partition::<32>(&block);
    ///
    /// let rank = tile.thread_rank();
    /// let my_value = *values.add(rank as usize);
    /// let match_mask = tile.match_any(my_value);
    ///
    /// // Only the lowest-ranked thread in each group writes
    /// if match_mask.trailing_zeros() == rank {
    ///     // This thread represents a unique value
    ///     let ballot = tile.ballot(true);
    ///     let unique_idx = (ballot & ((1u32 << rank) - 1)).count_ones();
    ///     *unique.add(unique_idx as usize) = my_value;
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn match_any(&self, value: i32) -> u32 {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // PTX match.sync.any.b32 compares values across warp
                // Returns bitmask of threads with matching values
                asm!(
                    "match.sync.any.b32 {result}, {value}, {mask};",
                    result = out(reg32) result,
                    value = in(reg32) value,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                // Fallback: only this thread matches
                result = 1u32 << self.rank;
            }
        }
        result
    }

    /// Finds threads with matching values and checks for unanimity (SM 7.0+).
    ///
    /// This is a warp-level match operation that returns both a bitmask of
    /// threads with matching values AND a boolean indicating whether ALL
    /// threads in the warp have the same value.
    ///
    /// # Arguments
    ///
    /// - `value`: The value to match against other threads
    ///
    /// # Returns
    ///
    /// A tuple `(mask, all_match)` where:
    /// - `mask`: u32 bitmask where bit M is set if thread M has the same value
    ///   as the calling thread
    /// - `all_match`: bool that is true if ALL threads in the warp have the
    ///   same value (unanimous), false otherwise
    ///
    /// # Requirements
    ///
    /// - Requires SM 7.0+ (Volta architecture or newer)
    /// - On older architectures, this function will not be available
    ///
    /// # Implementation
    ///
    /// Uses the PTX `match.sync.all.b32` instruction which compares values
    /// across threads and returns both a match mask and an all-match predicate.
    ///
    /// PTX instruction:
    /// ```ptx
    /// match.sync.all.b32 {mask}, {value}, {all_pred}, {thread_mask};
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Consensus detection**: Check if all threads agree on a value
    /// - **Early termination**: Exit early if all threads converged
    /// - **Value validation**: Verify unanimous agreement before proceeding
    /// - **Warp-level reduction**: Detect when reduction is complete
    ///
    /// # Example - Consensus Detection
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn wait_for_consensus(values: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let rank = tile.thread_rank();
    ///     let mut my_value = *values.add(rank as usize);
    ///
    ///     loop {
    ///         // Check for unanimous agreement
    ///         let (match_mask, all_agree) = tile.match_all(my_value);
    ///
    ///         if all_agree {
    ///             // All threads have the same value - we're done!
    ///             break;
    ///         }
    ///
    ///         // Update value and try again
    ///         my_value = update_consensus(my_value, match_mask);
    ///         *values.add(rank as usize) = my_value;
    ///     }
    /// }
    /// ```
    ///
    /// # Example - Leader Election with Validation
    ///
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(leader_id: *mut u32) {
    /// let block = this_thread_block();
    /// let tile = tiled_partition::<32>(&block);
    ///
    /// // Each thread proposes itself as leader (using rank)
    /// let my_proposal = tile.thread_rank() as i32;
    ///
    /// // Find minimum rank (leader)
    /// let leader_rank = warp_reduce_min(my_proposal, &tile);
    ///
    /// // Verify all threads agree on the leader
    /// let (match_mask, unanimous) = tile.match_all(leader_rank);
    ///
    /// if unanimous && tile.thread_rank() == 0 {
    ///     *leader_id = leader_rank as u32;
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn match_all(&self, value: i32) -> (u32, bool) {
        let mask: u32;
        let all_match: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                // PTX match.all.sync.b32 compares values and checks unanimity
                // Uses pipe syntax: dest_reg|dest_pred for composite output
                // Instruction format: match.all.sync.b32 %r|%p, value, mask;
                // Returns:
                //   - mask: bitmask of threads with matching values (in %r)
                //   - predicate: true if ALL threads have the same value (in %p)
                asm!(
                    "{{",
                    ".reg .pred %p_all;",
                    "match.all.sync.b32 {mask}|%p_all, {value}, {thread_mask};",
                    "selp.u32 {all_match}, 1, 0, %p_all;",
                    "}}",
                    mask = out(reg32) mask,
                    all_match = out(reg32) all_match,
                    value = in(reg32) value,
                    thread_mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                // Fallback: only this thread matches, not unanimous
                mask = 1u32 << self.rank;
                all_match = 0;
            }
        }
        (mask, all_match != 0)
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

/// A handle to a dynamically-sized tile of threads partitioned from a thread block.
///
/// `DynamicTiledGroup` represents a subset of threads within a thread block where
/// the tile size is determined at runtime rather than compile-time. This is useful
/// when the tile size depends on kernel parameters or runtime configuration.
///
/// # Comparison with TiledGroup<SIZE>
///
/// - `TiledGroup<SIZE>`: Size known at compile-time (const generic)
/// - `DynamicTiledGroup`: Size determined at runtime
///
/// Both provide identical operations and semantics, just with different
/// compile-time vs runtime trade-offs.
///
/// # Construction
///
/// Create via the `tiled_partition_dynamic()` function:
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
/// let block = this_thread_block();
/// let tile = tiled_partition_dynamic(&block, 32);
/// ```
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn adaptive_kernel(data: *mut i32, tile_size: u32) {
///     let block = this_thread_block();
///
///     // Tile size determined at runtime from kernel parameter
///     let tile = tiled_partition_dynamic(&block, tile_size);
///
///     // Use tile for warp-level operations
///     data.add(tile.thread_rank() as usize).write(compute_value());
///     tile.sync();
///     let neighbor = data.add(((tile.thread_rank() + 1) % tile.size()) as usize).read();
/// }
/// ```
#[repr(C)]
pub struct DynamicTiledGroup {
    /// Thread participation mask for this tile.
    mask: u32,

    /// Thread rank within the tile [0, size).
    rank: u32,

    /// Tile size (number of threads in tile).
    /// Must be power of 2, divide 32, and be in range [1, 32].
    size: u32,
}

/// Creates a tiled partition with runtime size selection.
///
/// This function partitions the parent thread block into tiles where the size
/// is determined at runtime. This is useful when tile size depends on kernel
/// parameters or runtime configuration.
///
/// # Arguments
///
/// - `parent`: The parent thread block to partition
/// - `size`: Tile size (must be power of 2, divide 32, range: 1-32)
///
/// # Returns
///
/// A `DynamicTiledGroup` handle representing this thread's tile.
///
/// # Panics
///
/// Panics (in debug builds) if size is not a power of 2 or does not divide 32.
/// In release builds, invalid sizes may cause undefined behavior.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn runtime_tile_size(data: *mut f32, size: u32) {
///     let block = this_thread_block();
///     let tile = tiled_partition_dynamic(&block, size);
///
///     // Use tile with runtime-determined size
///     process_local_data(data);
///     tile.sync();
///     update_shared_state(data);
/// }
/// ```
#[inline(always)]
pub fn tiled_partition_dynamic(_parent: &ThreadBlock, size: u32) -> DynamicTiledGroup {
    use crate::thread::thread_idx_x;

    // Validate size at runtime (debug mode only for performance)
    debug_assert!(size > 0, "Tile size must be greater than 0");
    debug_assert!(size <= 32, "Tile size cannot exceed warp size (32)");
    debug_assert!(size.is_power_of_two(), "Tile size must be a power of 2");
    debug_assert!(32 % size == 0, "Tile size must divide warp size (32)");

    // Get lane ID within warp (0-31)
    let lane_id = thread_idx_x() % 32;

    // Calculate which tile this thread belongs to
    let tile_id = lane_id / size;

    // Calculate rank within tile
    let rank = lane_id % size;

    // Compute participation mask for this tile
    let mask = compute_tile_mask(size, tile_id);

    DynamicTiledGroup { mask, rank, size }
}

impl DynamicTiledGroup {
    /// Synchronizes all threads within the tile.
    ///
    /// Identical semantics to `TiledGroup::sync()`, but uses runtime size.
    /// See [`TiledGroup::sync()`] for detailed documentation.
    #[inline(always)]
    pub fn sync(&self) {
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

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
    /// The tile size (determined at runtime). Thread ranks are in the range [0, size).
    #[inline(always)]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns the rank of the calling thread within the tile.
    ///
    /// # Returns
    ///
    /// Thread index within the tile in the range [0, size).
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        self.rank
    }

    /// Broadcasts a value from a specific source lane to all threads in the tile.
    ///
    /// Identical semantics to `TiledGroup::shfl()`, but uses runtime size.
    /// See [`TiledGroup::shfl()`] for detailed documentation.
    #[inline(always)]
    pub fn shfl(&self, var: i32, src_lane: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let max_lane = self.size - 1;
                asm!(
                    "shfl.sync.idx.b32 {result}, {var}, {src}, {max_lane}, {mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    src = in(reg32) src_lane,
                    max_lane = in(reg32) max_lane,
                    mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Shifts values down within the tile.
    ///
    /// Identical semantics to `TiledGroup::shfl_down()`, but uses runtime size.
    /// See [`TiledGroup::shfl_down()`] for detailed documentation.
    #[inline(always)]
    pub fn shfl_down(&self, var: i32, delta: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let max_lane = self.size - 1;
                asm!(
                    "shfl.sync.down.b32 {result}, {var}, {delta}, {max_lane}, {mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    delta = in(reg32) delta,
                    max_lane = in(reg32) max_lane,
                    mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Shifts values up within the tile.
    ///
    /// Identical semantics to `TiledGroup::shfl_up()`, but uses runtime size.
    /// See [`TiledGroup::shfl_up()`] for detailed documentation.
    #[inline(always)]
    pub fn shfl_up(&self, var: i32, delta: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            asm!(
                "shfl.sync.up.b32 {result}, {var}, {delta}, 0x0, {mask};",
                result = out(reg32) result,
                var = in(reg32) var,
                delta = in(reg32) delta,
                mask = in(reg32) self.mask,
                options(nostack, nomem)
            );

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Butterfly shuffle using XOR-based lane addressing.
    ///
    /// Identical semantics to `TiledGroup::shfl_xor()`, but uses runtime size.
    /// See [`TiledGroup::shfl_xor()`] for detailed documentation.
    #[inline(always)]
    pub fn shfl_xor(&self, var: i32, lane_mask: u32) -> i32 {
        let result: i32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let max_lane = self.size - 1;
                asm!(
                    "shfl.sync.bfly.b32 {result}, {var}, {mask}, {max_lane}, {thread_mask};",
                    result = out(reg32) result,
                    var = in(reg32) var,
                    mask = in(reg32) lane_mask,
                    max_lane = in(reg32) max_lane,
                    thread_mask = in(reg32) self.mask,
                    options(nostack, nomem)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = var;
            }
        }
        result
    }

    /// Checks if ANY thread in the tile has a true predicate.
    ///
    /// Identical semantics to `TiledGroup::any()`, but uses runtime size.
    /// See [`TiledGroup::any()`] for detailed documentation.
    #[inline(always)]
    pub fn any(&self, predicate: bool) -> bool {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let pred_val: u32 = if predicate { 1 } else { 0 };
                asm!(
                    "{{",
                    ".reg .pred %p_input, %p_result;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.any.pred %p_result, %p_input, {mask};",
                    "selp.u32 {result}, 1, 0, %p_result;",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result != 0
    }

    /// Checks if ALL threads in the tile have a true predicate.
    ///
    /// Identical semantics to `TiledGroup::all()`, but uses runtime size.
    /// See [`TiledGroup::all()`] for detailed documentation.
    #[inline(always)]
    pub fn all(&self, predicate: bool) -> bool {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let pred_val: u32 = if predicate { 1 } else { 0 };
                asm!(
                    "{{",
                    ".reg .pred %p_input, %p_result;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.all.pred %p_result, %p_input, {mask};",
                    "selp.u32 {result}, 1, 0, %p_result;",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result != 0
    }

    /// Collects votes from all threads as a bitmask.
    ///
    /// Identical semantics to `TiledGroup::ballot()`, but uses runtime size.
    /// See [`TiledGroup::ballot()`] for detailed documentation.
    #[inline(always)]
    pub fn ballot(&self, predicate: bool) -> u32 {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                let pred_val: u32 = if predicate { 1 } else { 0 };
                asm!(
                    "{{",
                    ".reg .pred %p_input;",
                    "setp.ne.u32 %p_input, {pred_val}, 0;",
                    "vote.sync.ballot.b32 {result}, %p_input, {mask};",
                    "}}",
                    result = out(reg32) result,
                    pred_val = in(reg32) pred_val,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = if predicate { 1 } else { 0 };
            }
        }
        result
    }

    /// Finds threads with matching values (SM 7.0+).
    ///
    /// Identical semantics to `TiledGroup::match_any()`, but uses runtime size.
    /// See [`TiledGroup::match_any()`] for detailed documentation.
    #[inline(always)]
    pub fn match_any(&self, value: i32) -> u32 {
        let result: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                asm!(
                    "match.sync.any.b32 {result}, {value}, {mask};",
                    result = out(reg32) result,
                    value = in(reg32) value,
                    mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                result = 1u32 << self.rank;
            }
        }
        result
    }

    /// Finds threads with matching values and checks for unanimity (SM 7.0+).
    ///
    /// Identical semantics to `TiledGroup::match_all()`, but uses runtime size.
    /// See [`TiledGroup::match_all()`] for detailed documentation.
    #[inline(always)]
    pub fn match_all(&self, value: i32) -> (u32, bool) {
        let mask: u32;
        let all_match: u32;
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            #[cfg(target_os = "cuda")]
            {
                asm!(
                    "{{",
                    ".reg .pred %p_all;",
                    "match.all.sync.b32 {mask}|%p_all, {value}, {thread_mask};",
                    "selp.u32 {all_match}, 1, 0, %p_all;",
                    "}}",
                    mask = out(reg32) mask,
                    all_match = out(reg32) all_match,
                    value = in(reg32) value,
                    thread_mask = in(reg32) self.mask,
                    options(nostack)
                );
            }

            #[cfg(not(target_os = "cuda"))]
            {
                mask = 1u32 << self.rank;
                all_match = 0;
            }
        }
        (mask, all_match != 0)
    }
}

// Safety: DynamicTiledGroup can be safely sent between threads within the kernel
unsafe impl Send for DynamicTiledGroup {}

// Safety: DynamicTiledGroup can be safely shared between threads within the kernel
unsafe impl Sync for DynamicTiledGroup {}

// Implement ThreadGroup trait for polymorphic cooperative group operations
impl super::traits::ThreadGroup for DynamicTiledGroup {
    #[inline(always)]
    fn sync(&self) {
        DynamicTiledGroup::sync(self)
    }

    #[inline(always)]
    fn size(&self) -> u32 {
        DynamicTiledGroup::size(self)
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        DynamicTiledGroup::thread_rank(self)
    }
}
