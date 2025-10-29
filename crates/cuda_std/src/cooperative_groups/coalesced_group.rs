//! Coalesced group primitives for divergent execution.
//!
//! This module provides coalesced thread groups that represent the set of currently-active
//! threads within a warp. Unlike tiled groups which have static membership, coalesced groups
//! have dynamic membership determined at runtime based on execution divergence.
//!
//! # Overview
//!
//! The `CoalescedGroup` type represents threads that are active at a specific point in code,
//! even when other threads in the warp have diverged. This enables cooperation among threads
//! that took the same code path through conditional branches.
//!
//! # Use Cases
//!
//! Coalesced groups are essential for algorithms where:
//! - Threads diverge based on runtime conditions (if/else, loops)
//! - Only active threads need to synchronize or communicate
//! - Thread participation varies dynamically
//!
//! # Example: Divergent Execution
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn conditional_processing(data: *mut f32, flags: *const bool) {
//!     let idx = thread::index() as usize;
//!
//!     // Threads diverge based on flag
//!     if *flags.add(idx) {
//!         // Only threads with flag=true execute here
//!         let group = coalesced_threads();
//!
//!         // Synchronize only active threads
//!         group.sync();
//!
//!         // Cooperate among active threads
//!         let neighbor_data = group.shfl_down(data[idx] as i32, 1);
//!
//!         if group.thread_rank() == 0 {
//!             // First active thread does work
//!         }
//!     }
//! }
//! ```
//!
//! # Key Concepts
//!
//! ## Active Mask
//! The `activemask.b32` PTX instruction returns a 32-bit mask where each bit represents
//! whether the corresponding lane (thread) is active at that point. All active threads
//! receive the same mask.
//!
//! ## Rank Calculation
//! Each thread's rank within the coalesced group is determined by counting how many
//! active threads have lower lane IDs. This means rank is relative to active threads,
//! not absolute lane position.
//!
//! Example with mask 0x0000FF00 (lanes 8-15 active):
//! - Lane 8 has rank 0
//! - Lane 9 has rank 1
//! - Lane 15 has rank 7
//!
//! ## Sparse Thread Handling
//! Shuffle operations must map relative ranks (0 to size-1) to absolute lane IDs.
//! The `nth_active_lane()` helper performs this mapping by counting set bits in the mask.
//!
//! # Requirements
//!
//! 1. **Uniform Call**: All active threads at a divergence point must call `coalesced_threads()`
//! 2. **Same Warp**: Only threads within the same warp can form a coalesced group
//! 3. **Consistent Mask**: All operations must use the mask captured at group creation time
//!
//! # Known Limitations
//!
//! ## Rust Compiler Optimization Affecting Divergence Detection
//!
//! **Critical Issue**: Due to Rust compiler optimizations, the `activemask.b32` instruction
//! may not correctly detect branch divergence when called from Rust-compiled kernels. The
//! compiler can eliminate branches it considers side-effect-free, causing `activemask.b32`
//! to always return 0xFFFFFFFF (all threads active) even when logical divergence exists.
//!
//! **Impact**:
//! - Divergent execution patterns (if/else splits, modulo conditions) may appear convergent
//! - `coalesced_threads()` may report incorrect group sizes and masks
//! - Rank calculations based on divergence will be incorrect
//!
//! **Working Scenarios**:
//! - Convergent execution (all threads same path) works correctly
//! - Vote operations (any, all, ballot) work correctly
//! - Calling from C++ or hand-written PTX works correctly
//!
//! **Workarounds**:
//! 1. Use C++ cooperative_groups for divergent kernels (see tests/cuda_std_cg_tests/cpp_reference/coalesced_tests.cu)
//! 2. Call Rust coalesced group functions from C++-compiled kernels
//! 3. Use explicit warp intrinsics (__ballot_sync, __activemask) if available in your Rust context
//!
//! **Example of Issue**:
//! ```no_run
//! // This Rust code may NOT detect divergence correctly:
//! if thread_id < 16 {
//!     let group = coalesced_threads();  // May report size=32 instead of 16
//! }
//! ```
//!
//! For production use requiring correct divergence detection, consider calling from C++
//! or verifying behavior with actual GPU execution.
//!
//! # Performance Considerations
//!
//! - Very lightweight group creation (single PTX instruction)
//! - Synchronization and shuffle operations work efficiently with sparse threads
//! - No shared memory required
//! - Rank calculation uses efficient bit manipulation
//!
//! # Safety
//!
//! The public API is safe because:
//! - Group lifetime tied to kernel scope
//! - Mask captured atomically at creation
//! - Type system prevents mask inconsistencies

use crate::thread;
use super::traits::ThreadGroup;
use core::arch::asm;

/// Dynamic group of currently-active threads within a warp.
///
/// Created by [`coalesced_threads()`], represents threads that are active at the
/// same point in code despite divergent execution paths.
///
/// # Fields
/// - `mask`: 32-bit mask where bit N indicates if lane N is active
/// - `rank`: This thread's 0-indexed rank among active threads
/// - `size`: Total number of active threads (popcount of mask)
///
/// # Example
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// let group = coalesced_threads();
/// println!("Group has {} threads, I am rank {}", group.size(), group.thread_rank());
/// # }
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CoalescedGroup {
    mask: u32,
    rank: u32,
    size: u32,
}

impl CoalescedGroup {
    /// Synchronize all active threads in this coalesced group.
    ///
    /// Uses the group's mask to synchronize only threads that were active when
    /// the group was created. Threads not in the mask are not affected.
    ///
    /// # Requirements
    /// All threads in the group must call `sync()`. Divergent calls cause deadlock.
    ///
    /// # Memory Ordering
    /// All memory operations before `sync()` are visible to all threads in the group
    /// after `sync()` returns.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(data: *mut i32) {
    /// let group = coalesced_threads();
    ///
    /// // Write to memory
    /// *data.add(thread::index() as usize) = 42;
    ///
    /// // Ensure all threads in group see the write
    /// group.sync();
    /// # }
    /// ```
    #[inline(always)]
    pub fn sync(&self) {
        unsafe {
            asm!(
                "bar.warp.sync {mask};",
                mask = in(reg32) self.mask,
                options(nostack)
            );
        }
    }

    /// Returns the number of threads in this coalesced group.
    ///
    /// This is the population count (number of set bits) of the active mask.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// if group.size() == 32 {
    ///     // All threads in warp are active
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns this thread's rank within the coalesced group (0-indexed).
    ///
    /// Rank is determined by counting active threads with lower lane IDs.
    /// The thread with the lowest active lane ID has rank 0.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(output: *mut i32) {
    /// let group = coalesced_threads();
    /// if group.thread_rank() == 0 {
    ///     // First active thread writes result
    ///     *output = 42;
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        self.rank
    }

    /// Returns the 32-bit mask of active threads in this group.
    ///
    /// Bit N in the mask is set if lane N (thread with `threadIdx.x % 32 == N`)
    /// is active in this group.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let mask = group.mask();
    /// if mask == 0xFFFFFFFF {
    ///     // All 32 lanes are active
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// Broadcast a value from source rank to all threads in the group.
    ///
    /// # Arguments
    /// - `var`: Value to broadcast (only used from thread with rank `src_rank`)
    /// - `src_rank`: Rank of source thread (0 to size-1)
    ///
    /// # Returns
    /// The value from the thread with rank `src_rank`.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let value = thread::index() as i32;
    ///
    /// // All threads receive value from rank 0
    /// let broadcast = group.shfl(value, 0);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shfl(&self, var: i32, src_rank: u32) -> i32 {
        // Map rank to actual lane ID
        let src_lane = nth_active_lane(self.mask, src_rank);

        let result: i32;
        unsafe {
            asm!(
                "shfl.sync.idx.b32 {out}, {in}, {src}, 0x1f, {mask};",
                out = out(reg32) result,
                in = in(reg32) var,
                src = in(reg32) src_lane,
                mask = in(reg32) self.mask,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// Shift value down by `delta` ranks within the group.
    ///
    /// Each thread receives the value from the thread with rank `thread_rank() + delta`.
    /// Threads with `thread_rank() + delta >= size()` receive undefined values.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let value = thread::index() as i32;
    ///
    /// // Each thread gets value from next thread
    /// let neighbor = group.shfl_down(value, 1);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shfl_down(&self, var: i32, delta: u32) -> i32 {
        let target_rank = self.rank + delta;
        if target_rank < self.size {
            self.shfl(var, target_rank)
        } else {
            var // Out of bounds, return own value
        }
    }

    /// Shift value up by `delta` ranks within the group.
    ///
    /// Each thread receives the value from the thread with rank `thread_rank() - delta`.
    /// Threads with `thread_rank() < delta` receive undefined values.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let value = thread::index() as i32;
    ///
    /// // Each thread gets value from previous thread
    /// let neighbor = group.shfl_up(value, 1);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shfl_up(&self, var: i32, delta: u32) -> i32 {
        if self.rank >= delta {
            self.shfl(var, self.rank - delta)
        } else {
            var // Out of bounds, return own value
        }
    }

    /// XOR shuffle: exchange values with thread at rank `(thread_rank() ^ mask)`.
    ///
    /// Useful for butterfly communication patterns in reductions.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let value = thread::index() as i32;
    ///
    /// // Butterfly exchange
    /// let exchanged = group.shfl_xor(value, 1);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shfl_xor(&self, var: i32, mask: u32) -> i32 {
        let target_rank = self.rank ^ mask;
        if target_rank < self.size {
            self.shfl(var, target_rank)
        } else {
            var
        }
    }

    /// Returns true if any thread in the group has `predicate == true`.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let found_error = thread::index() == 42;
    ///
    /// if group.any(found_error) {
    ///     // At least one thread found an error
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn any(&self, predicate: bool) -> bool {
        let pred_val: u32 = if predicate { 1 } else { 0 };
        let result: u32;
        unsafe {
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
                options(pure, nomem, nostack)
            );
        }
        result != 0
    }

    /// Returns true if all threads in the group have `predicate == true`.
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example(data: *const i32) {
    /// let group = coalesced_threads();
    /// let idx = thread::index() as usize;
    /// let is_valid = unsafe { *data.add(idx) > 0 };
    ///
    /// if group.all(is_valid) {
    ///     // All threads have valid data
    /// }
    /// # }
    /// ```
    #[inline(always)]
    pub fn all(&self, predicate: bool) -> bool {
        let pred_val: u32 = if predicate { 1 } else { 0 };
        let result: u32;
        unsafe {
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
                options(pure, nomem, nostack)
            );
        }
        result != 0
    }

    /// Returns a ballot of predicate values across all threads in the group.
    ///
    /// Bit N in the result is set if the thread with rank N has `predicate == true`.
    /// Only considers threads in this group (ranks 0 to size-1).
    ///
    /// # Example
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # #[kernel]
    /// # pub unsafe fn example() {
    /// let group = coalesced_threads();
    /// let condition = thread::index() < 16;
    ///
    /// let ballot = group.ballot(condition);
    /// // ballot has bits set for threads where condition is true
    /// # }
    /// ```
    #[inline(always)]
    pub fn ballot(&self, predicate: bool) -> u32 {
        let pred_val: u32 = if predicate { 1 } else { 0 };
        let result: u32;
        unsafe {
            asm!(
                "{{",
                ".reg .pred %p_input;",
                "setp.ne.u32 %p_input, {pred_val}, 0;",
                "vote.sync.ballot.b32 {result}, %p_input, {mask};",
                "}}",
                result = out(reg32) result,
                pred_val = in(reg32) pred_val,
                mask = in(reg32) self.mask,
                options(pure, nomem, nostack)
            );
        }
        // Mask result to only include threads in this group
        result & self.mask
    }
}

impl ThreadGroup for CoalescedGroup {
    #[inline(always)]
    fn sync(&self) {
        CoalescedGroup::sync(self)
    }

    #[inline(always)]
    fn size(&self) -> u32 {
        self.size
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        self.rank
    }
}

/// Returns a coalesced group of currently-active threads in this warp.
///
/// Uses the `activemask.b32` PTX instruction to detect which threads are active
/// at the call site. All active threads receive the same mask but different ranks.
///
/// # Active Mask
/// The mask captures which threads execute this instruction. Threads that diverged
/// earlier (took different branches) are not included.
///
/// # Rank Assignment
/// Rank is calculated by counting active threads with lower lane IDs. This ensures
/// ranks are contiguous (0 to size-1) even when threads are sparse in the warp.
///
/// # Example: Convergent Execution
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// // All threads in warp execute this
/// let group = coalesced_threads();
/// // group.size() == 32, ranks 0-31
/// # }
/// ```
///
/// # Example: Divergent Execution
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// if thread::index() < 16 {
///     // Only first 16 threads execute this
///     let group = coalesced_threads();
///     // group.size() == 16, ranks 0-15
/// }
/// # }
/// ```
///
/// # Example: Sparse Threads
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// if thread::index() % 2 == 0 {
///     // Every other thread executes this
///     let group = coalesced_threads();
///     // group.size() == 16, but threads are lanes 0,2,4,...,30
///     // Still get ranks 0-15 in order
/// }
/// # }
/// ```
#[inline(always)]
pub fn coalesced_threads() -> CoalescedGroup {
    let mask: u32;
    unsafe {
        asm!(
            "activemask.b32 {mask};",
            mask = out(reg32) mask,
            options(pure, nomem, nostack)
        );
    }

    let lane_id = thread::thread_idx_x() % 32;

    // Rank = number of active threads with lower lane IDs
    let rank = count_active_below(mask, lane_id);

    // Size = total number of active threads
    let size = mask.count_ones();

    CoalescedGroup { mask, rank, size }
}

/// Count how many bits are set in mask below the given position.
///
/// Used to calculate a thread's rank within the coalesced group.
///
/// # Arguments
/// - `mask`: 32-bit mask of active threads
/// - `position`: Lane ID (0-31) to count below
///
/// # Returns
/// Number of set bits in positions [0, position)
///
/// # Example
/// ```
/// let mask = 0xFF00; // Lanes 8-15 active
/// assert_eq!(count_active_below(mask, 8), 0);  // No active lanes below 8
/// assert_eq!(count_active_below(mask, 10), 2); // Lanes 8,9 below 10
/// assert_eq!(count_active_below(mask, 16), 8); // All 8 active lanes below 16
/// ```
#[inline(always)]
fn count_active_below(mask: u32, position: u32) -> u32 {
    let below_mask = (1u32 << position).wrapping_sub(1);
    (mask & below_mask).count_ones()
}

/// Find the lane ID of the Nth active thread (0-indexed).
///
/// Used to map relative ranks to absolute lane IDs for shuffle operations.
///
/// # Arguments
/// - `mask`: 32-bit mask of active threads
/// - `n`: Rank to find (0 to popcount-1)
///
/// # Returns
/// Lane ID of the Nth active thread, or 31 if not found
///
/// # Example
/// ```
/// let mask = 0xFF00; // Lanes 8-15 active
/// assert_eq!(nth_active_lane(mask, 0), 8);  // First active thread is lane 8
/// assert_eq!(nth_active_lane(mask, 1), 9);  // Second active thread is lane 9
/// assert_eq!(nth_active_lane(mask, 7), 15); // Eighth active thread is lane 15
/// ```
#[inline(always)]
fn nth_active_lane(mask: u32, n: u32) -> u32 {
    let mut count = 0u32;
    for lane in 0..32 {
        if (mask & (1 << lane)) != 0 {
            if count == n {
                return lane;
            }
            count += 1;
        }
    }
    31 // Fallback (should not reach if n < popcount)
}
