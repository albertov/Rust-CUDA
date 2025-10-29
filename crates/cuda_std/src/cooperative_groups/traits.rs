//! Trait abstractions for polymorphic cooperative group operations.
//!
//! This module provides trait definitions that enable generic algorithms to work
//! across different synchronization scopes (grid-level, block-level, and future
//! tile-level groups).

/// A trait for cooperative groups that support synchronization and thread identification.
///
/// `ThreadGroup` provides a common interface for synchronization operations across
/// different group types (grid, thread block, tiles, etc.). This enables writing
/// generic algorithms that work at any synchronization scope.
///
/// # Implementors
///
/// - [`GridGroup`](crate::cooperative_groups::GridGroup) - Grid-wide synchronization
/// - [`ThreadBlock`](crate::cooperative_groups::ThreadBlock) - Block-level synchronization
///
/// # Example: Generic Algorithm
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// // Generic function that works with any thread group
/// fn cooperative_sum<G: ThreadGroup>(group: &G, value: u32) -> u32 {
///     // Each thread contributes its value
///     let local_value = value;
///
///     // Synchronize to ensure all threads ready
///     group.sync();
///
///     // Simple sum (real implementation would use reduction)
///     let sum = local_value * group.size();
///     sum
/// }
///
/// #[kernel]
/// pub unsafe fn kernel_using_generic(data: *mut u32) {
///     let block = this_thread_block();
///     let result = cooperative_sum(&block, 42);
///
///     // Can also use with grid groups
///     let grid = this_grid();
///     if grid.is_valid() {
///         let grid_result = cooperative_sum(&grid, 17);
///     }
/// }
/// ```
///
/// # Design Rationale
///
/// The trait provides only the minimal common interface:
/// - `sync()`: Universal barrier operation
/// - `size()`: Total thread count in group
/// - `thread_rank()`: Unique thread identifier within group
///
/// Group-specific operations (like `dim_x()` for ThreadBlock or `is_valid()`
/// for GridGroup) remain on the concrete types.
///
/// # Performance
///
/// All trait methods are marked `#[inline(always)]` to ensure zero-cost abstraction.
/// Generic functions using this trait will be specialized at compile time with
/// no runtime overhead.
pub trait ThreadGroup {
    /// Synchronizes all threads in this group.
    ///
    /// This is a barrier operation. Execution resumes only after ALL threads
    /// in the group have reached this synchronization point.
    ///
    /// # Requirements
    ///
    /// - **Uniform participation**: All threads in the group must call `sync()`.
    ///   Divergent calls cause deadlock.
    ///
    /// # Memory Ordering
    ///
    /// Provides strong memory ordering guarantees:
    /// - All memory modifications before `sync()` are visible to all threads
    ///   in the group after `sync()` returns.
    ///
    /// # Deadlock Prevention
    ///
    /// **SAFE - All threads participate**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let group = this_thread_block();
    /// // All threads execute sync unconditionally
    /// group.sync();
    /// ```
    ///
    /// **UNSAFE - Divergent sync**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let group = this_thread_block();
    /// # let condition = true;
    /// // DEADLOCK: Only some threads call sync
    /// if condition {
    ///     group.sync(); // Some threads skip this!
    /// }
    /// ```
    fn sync(&self);

    /// Returns the total number of threads in this group.
    ///
    /// # Returns
    ///
    /// Total thread count. Thread ranks are in the range `[0, size())`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// fn process_with_group<G: ThreadGroup>(group: &G, data: *mut u32) {
    ///     let total = group.size();
    ///     let rank = group.thread_rank();
    ///
    ///     // Each thread processes its portion
    ///     let chunk_size = DATA_SIZE / total;
    ///     let start = rank * chunk_size;
    /// }
    /// ```
    fn size(&self) -> u32;

    /// Returns the rank of the calling thread within this group.
    ///
    /// Thread ranks uniquely identify each thread in the group, ranging
    /// from 0 to `size() - 1`.
    ///
    /// # Returns
    ///
    /// Unique thread identifier in the range `[0, size())`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// fn write_rank<G: ThreadGroup>(group: &G, output: *mut u32) {
    ///     let rank = group.thread_rank();
    ///     unsafe {
    ///         *output.add(rank as usize) = rank;
    ///     }
    /// }
    /// ```
    fn thread_rank(&self) -> u32;

    /// Returns the 32-bit thread participation mask for this group.
    ///
    /// The mask is a 32-bit value where bit N is set if lane N (thread with
    /// `threadIdx.x % 32 == N`) participates in this group.
    ///
    /// # Returns
    ///
    /// 32-bit participation mask for threads in this group.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// fn check_participation<G: ThreadGroup>(group: &G) {
    ///     let mask = group.mask();
    ///     if mask == 0xFFFFFFFF {
    ///         // All 32 lanes in warp are active
    ///     }
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// For groups larger than 32 threads (like ThreadBlock), this returns
    /// the mask for the current thread's warp (0xFFFFFFFF for full warp participation).
    fn mask(&self) -> u32;

    /// Performs a reduction across all threads in the group using addition.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the sum of all values. Uses hardware-accelerated `redux.sync.add`
    /// instruction on SM 8.0+ (Ampere and newer).
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The sum of all values from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn sum_kernel(output: *mut i32) {
    ///     let block = this_thread_block();
    ///     let tile = tiled_partition::<32>(&block);
    ///
    ///     let value = tile.thread_rank() as i32; // Each thread contributes its rank
    ///     let sum = tile.reduce_add(value);      // Sum across all threads
    ///
    ///     if tile.thread_rank() == 0 {
    ///         *output = sum; // Thread 0 writes result (sum = 0+1+2+...+31 = 496)
    ///     }
    /// }
    /// ```
    fn reduce_add<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::Add) }
    }

    /// Performs a reduction across all threads in the group using minimum.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the minimum value. Uses hardware-accelerated `redux.sync.min`
    /// instruction on SM 8.0+.
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The minimum value from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    fn reduce_min<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::Min) }
    }

    /// Performs a reduction across all threads in the group using maximum.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the maximum value. Uses hardware-accelerated `redux.sync.max`
    /// instruction on SM 8.0+.
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The maximum value from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    fn reduce_max<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::Max) }
    }

    /// Performs a bitwise AND reduction across all threads in the group.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the bitwise AND of all values. Uses hardware-accelerated
    /// `redux.sync.and` instruction on SM 8.0+.
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The bitwise AND of all values from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    fn reduce_and<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::And) }
    }

    /// Performs a bitwise OR reduction across all threads in the group.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the bitwise OR of all values. Uses hardware-accelerated
    /// `redux.sync.or` instruction on SM 8.0+.
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The bitwise OR of all values from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    fn reduce_or<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::Or) }
    }

    /// Performs a bitwise XOR reduction across all threads in the group.
    ///
    /// All threads in the group contribute their `value`, and the operation
    /// returns the bitwise XOR of all values. Uses hardware-accelerated
    /// `redux.sync.xor` instruction on SM 8.0+.
    ///
    /// # Arguments
    ///
    /// - `value`: The value contributed by this thread
    ///
    /// # Returns
    ///
    /// The bitwise XOR of all values from all threads in the group.
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - All threads in the group must call this function uniformly
    fn reduce_xor<T: crate::warp::WarpReduceValue>(&self, value: T) -> T {
        unsafe { crate::warp::warp_reduce(self.mask(), value, crate::warp::WarpReductionOp::Xor) }
    }
}
