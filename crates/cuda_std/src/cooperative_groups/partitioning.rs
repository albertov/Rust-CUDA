//! Thread group partitioning functions.
//!
//! Provides functions to create coalesced subgroups from parent groups
//! based on thread-specific criteria (labels or predicates).
//!
//! # Overview
//!
//! Partitioning functions create dynamic [`CoalescedGroup`] instances by grouping
//! threads that share a common label or predicate value. Unlike static partitioning
//! with [`tiled_partition`], these functions create groups based on runtime values.
//!
//! # Functions
//!
//! - [`labeled_partition`] - Partition by integer label (threads with same label form group)
//! - [`binary_partition`] - Partition by boolean predicate (true/false groups)
//!
//! # Examples
//!
//! ## Labeled Partition
//!
//! Create groups based on integer labels:
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn process_by_category(data: *mut f32, categories: *const u32) {
//!     let block = this_thread_block();
//!     let tid = thread::index() as usize;
//!
//!     // Partition by data category
//!     let label = *categories.add(tid);
//!     let group = labeled_partition(&block, label);
//!
//!     // Cooperate with threads in same category
//!     let sum = group_reduce_sum(&group, *data.add(tid));
//!
//!     if group.thread_rank() == 0 {
//!         // First thread in each category writes result
//!     }
//! }
//! ```
//!
//! ## Binary Partition
//!
//! Split threads into two groups based on a condition:
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn process_valid_data(data: *const f32, valid: *const bool) {
//!     let tile = tiled_partition::<32>(this_thread_block());
//!     let tid = thread::index() as usize;
//!
//!     // Partition into valid/invalid groups
//!     let is_valid = *valid.add(tid);
//!     let group = binary_partition(&tile, is_valid);
//!
//!     // Valid threads process data independently
//!     if is_valid {
//!         let value = *data.add(tid);
//!         // Process with other valid threads
//!     }
//! }
//! ```
//!
//! # Algorithm
//!
//! Both functions use the `match.any.sync` PTX instruction:
//!
//! 1. Each thread provides a label/predicate value
//! 2. The hardware finds all threads with the same value
//! 3. A mask is returned identifying matching threads
//! 4. A [`CoalescedGroup`] is created from this mask
//! 5. Rank is calculated by counting active threads with lower lane IDs
//!
//! # Performance
//!
//! - Single PTX instruction (`match.any.sync`) for label matching
//! - O(1) group formation regardless of label distribution
//! - Efficient even with sparse or uneven group sizes
//! - Works with any parent group (ThreadBlock, TiledGroup, etc.)

use super::coalesced_group::CoalescedGroup;
use super::traits::ThreadGroup;
use crate::gpu_only;

/// Partition threads into coalesced groups based on integer labels.
///
/// Creates a [`CoalescedGroup`] containing all threads from the parent group
/// that have the same `label` value. Each unique label value forms a separate
/// group, though only the calling thread's group is returned.
///
/// # Arguments
///
/// - `parent`: The parent group to partition (typically a ThreadBlock or TiledGroup)
/// - `label`: Integer label identifying which group this thread belongs to
///
/// # Returns
///
/// A [`CoalescedGroup`] containing all threads with the same label value.
///
/// # Requirements
///
/// - All threads in the parent group must call this function
/// - Threads must be in the same warp (32-thread hardware unit)
/// - Parent group mask determines eligible threads
///
/// # Examples
///
/// ## Uniform Partitioning
///
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// let block = this_thread_block();
/// let tid = thread::thread_idx_x();
///
/// // Create 4 groups of 8 threads each
/// let label = tid / 8;
/// let group = labeled_partition(&block, label);
///
/// // Threads 0-7 have label 0 (one group)
/// // Threads 8-15 have label 1 (another group)
/// // etc.
/// # }
/// ```
///
/// ## Dynamic Partitioning
///
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example(categories: *const u32) {
/// let block = this_thread_block();
/// let tid = thread::thread_idx_x() as usize;
///
/// // Partition by runtime data
/// let category = unsafe { *categories.add(tid) };
/// let group = labeled_partition(&block, category);
///
/// // Group size varies based on category distribution
/// # }
/// ```
///
/// # Algorithm
///
/// Uses the `match.any.sync` PTX instruction to find threads with matching labels:
///
/// 1. Get parent group's active mask
/// 2. Use `match.any.sync` with this thread's label
/// 3. Receive mask of all threads with same label
/// 4. Calculate rank by counting active threads with lower lane IDs
/// 5. Calculate size as population count of mask
///
/// # PTX Generated
///
/// ```ptx
/// match.any.sync.b32 %r_mask, %r_label, %r_parent_mask;
/// popc.b32 %r_size, %r_mask;
/// ```
#[inline(always)]
pub fn labeled_partition<G: ThreadGroup>(parent: &G, label: u32) -> CoalescedGroup {
    // Get parent's active mask to constrain matching
    let parent_mask = parent.mask();

    // Use match.any to find threads with same label
    let mask = unsafe { match_any_32(parent_mask, label) };

    // Construct CoalescedGroup from mask
    CoalescedGroup::from_mask(mask)
}

/// Partition threads into two coalesced groups based on a boolean predicate.
///
/// Creates a [`CoalescedGroup`] containing all threads from the parent group
/// where the `predicate` has the same boolean value (true or false). This is
/// equivalent to `labeled_partition(parent, predicate as u32)`.
///
/// # Arguments
///
/// - `parent`: The parent group to partition (typically a ThreadBlock or TiledGroup)
/// - `predicate`: Boolean condition determining group membership
///
/// # Returns
///
/// A [`CoalescedGroup`] containing all threads with the same predicate value.
///
/// # Requirements
///
/// - All threads in the parent group must call this function
/// - Threads must be in the same warp (32-thread hardware unit)
/// - Parent group mask determines eligible threads
///
/// # Examples
///
/// ## Even/Odd Split
///
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example() {
/// let block = this_thread_block();
/// let tid = thread::thread_idx_x();
///
/// // Split into even and odd threads
/// let is_even = tid % 2 == 0;
/// let group = binary_partition(&block, is_even);
///
/// // Even threads: group.size() == 16, mask = 0x55555555
/// // Odd threads:  group.size() == 16, mask = 0xAAAAAAAA
/// # }
/// ```
///
/// ## Threshold Split
///
/// ```no_run
/// # use cuda_std::cooperative_groups::*;
/// # #[kernel]
/// # pub unsafe fn example(data: *const f32) {
/// let tile = tiled_partition::<32>(this_thread_block());
/// let tid = thread::thread_idx_x() as usize;
///
/// // Split based on data threshold
/// let value = unsafe { *data.add(tid) };
/// let above_threshold = value > 0.5;
/// let group = binary_partition(&tile, above_threshold);
///
/// // Group size varies based on data distribution
/// # }
/// ```
///
/// # Algorithm
///
/// Converts the boolean predicate to a label (0 or 1) and uses [`labeled_partition`]:
///
/// 1. Convert predicate to u32 label (false=0, true=1)
/// 2. Call `labeled_partition` with the label
/// 3. Return the resulting coalesced group
///
/// This creates exactly two groups: one for `predicate==false` and one for `predicate==true`.
#[inline(always)]
pub fn binary_partition<G: ThreadGroup>(parent: &G, predicate: bool) -> CoalescedGroup {
    let label = if predicate { 1u32 } else { 0u32 };
    labeled_partition(parent, label)
}

/// LLVM intrinsic for match.any.sync.i32 instruction.
///
/// Finds all threads in `mask` that have the same `value`.
///
/// # Arguments
///
/// - `mask`: 32-bit mask of threads to consider
/// - `value`: Value to match against
///
/// # Returns
///
/// 32-bit mask where bit N is set if thread N has the same value as the calling thread.
///
/// # PTX Instruction
///
/// ```ptx
/// match.any.sync.b32 dest, value, mask;
/// ```
#[gpu_only]
#[inline(always)]
unsafe fn match_any_32(mask: u32, value: u32) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.match.any.sync.i32"]
        fn __nvvm_warp_match_any_32(mask: u32, value: u32) -> u32;
    }
    unsafe { __nvvm_warp_match_any_32(mask, value) }
}

