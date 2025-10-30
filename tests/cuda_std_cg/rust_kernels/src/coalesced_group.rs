use cuda_std::cooperative_groups::*;
use cuda_std::prelude::*;

/// Test coalesced group creation with all threads active (convergent case).
///
/// This kernel tests:
/// 1. Creating a CoalescedGroup with coalesced_threads()
/// 2. Verifying all 32 threads are active (size = 32)
/// 3. Verifying thread ranks are 0-31
/// 4. Verifying mask is 0xFFFFFFFF
#[kernel]
pub unsafe fn test_coalesced_all_active(ranks: *mut u32, sizes: *mut u32, masks: *mut u32) {
    let thread_idx = thread::thread_idx_x();

    // All threads execute this - no divergence
    let group = coalesced_threads();

    let rank = group.thread_rank();
    let size = group.size();
    let mask = group.mask();

    // Write results
    ranks.add(thread_idx as usize).write(rank);
    sizes.add(thread_idx as usize).write(size);
    masks.add(thread_idx as usize).write(mask);

    // Sync all active threads
    group.sync();
}

/// Test coalesced group with divergent execution (if/else).
///
/// This kernel tests:
/// 1. Threads split into two groups (0-15 and 16-31)
/// 2. Each group creates a coalesced_threads() handle
/// 3. Verifying each group has 16 threads
/// 4. Verifying ranks are 0-15 within each group
/// 5. Verifying masks reflect the correct thread sets
#[kernel]
pub unsafe fn test_coalesced_divergent(ranks: *mut u32, sizes: *mut u32, masks: *mut u32) {
    let thread_idx = thread::thread_idx_x();

    // Divergent execution
    // Write to local memory to prevent compiler optimization
    let mut local_idx = thread_idx;
    core::ptr::write_volatile(&mut local_idx, thread_idx);
    let condition = core::ptr::read_volatile(&local_idx) < 16;

    if condition {
        // First half of warp
        let group = coalesced_threads();

        let rank = group.thread_rank();
        let size = group.size();
        let mask = group.mask();

        ranks.add(thread_idx as usize).write(rank);
        sizes.add(thread_idx as usize).write(size);
        masks.add(thread_idx as usize).write(mask);

        group.sync();
    } else {
        // Second half of warp
        let group = coalesced_threads();

        let rank = group.thread_rank();
        let size = group.size();
        let mask = group.mask();

        ranks.add(thread_idx as usize).write(rank);
        sizes.add(thread_idx as usize).write(size);
        masks.add(thread_idx as usize).write(mask);

        group.sync();
    }
}

/// Test coalesced group with sparse threads (every other thread).
///
/// This kernel tests:
/// 1. Threads split by odd/even
/// 2. Only even threads (0,2,4,...,30) form one group
/// 3. Only odd threads (1,3,5,...,31) form another group
/// 4. Each group has 16 threads with ranks 0-15
/// 5. Masks show non-contiguous thread patterns
#[kernel]
pub unsafe fn test_coalesced_sparse(ranks: *mut u32, sizes: *mut u32, masks: *mut u32) {
    let thread_idx = thread::thread_idx_x();

    // Sparse divergence - every other thread
    // Write to local memory to prevent compiler optimization
    let mut local_idx = thread_idx;
    core::ptr::write_volatile(&mut local_idx, thread_idx);
    let is_even = core::ptr::read_volatile(&local_idx) % 2 == 0;

    if is_even {
        // Even threads: 0, 2, 4, ..., 30
        let group = coalesced_threads();

        let rank = group.thread_rank();
        let size = group.size();
        let mask = group.mask();

        ranks.add(thread_idx as usize).write(rank);
        sizes.add(thread_idx as usize).write(size);
        masks.add(thread_idx as usize).write(mask);

        group.sync();
    } else {
        // Odd threads: 1, 3, 5, ..., 31
        let group = coalesced_threads();

        let rank = group.thread_rank();
        let size = group.size();
        let mask = group.mask();

        ranks.add(thread_idx as usize).write(rank);
        sizes.add(thread_idx as usize).write(size);
        masks.add(thread_idx as usize).write(mask);

        group.sync();
    }
}

/// Test shuffle operations with coalesced group (sparse threads).
///
/// This kernel tests:
/// 1. shfl() - broadcast from rank 0 to all threads
/// 2. shfl_down() - each thread gets value from next rank
/// 3. shfl_up() - each thread gets value from previous rank
///
/// Only even threads participate (0,2,4,...,30).
#[kernel]
pub unsafe fn test_coalesced_shuffle(
    shfl_results: *mut i32,
    shfl_down_results: *mut i32,
    shfl_up_results: *mut i32,
) {
    let thread_idx = thread::thread_idx_x();

    // Only even threads participate
    // Write to local memory to prevent compiler optimization
    let mut local_idx = thread_idx;
    core::ptr::write_volatile(&mut local_idx, thread_idx);
    let condition = core::ptr::read_volatile(&local_idx) % 2 == 0;

    if condition {
        let group = coalesced_threads();

        // Each thread has its lane ID as value
        let my_value = thread_idx as i32;

        // Test 1: Broadcast from rank 0 (lane 0)
        let shfl_value = group.shfl(my_value, 0);
        shfl_results.add(thread_idx as usize).write(shfl_value);

        // Test 2: Shift down by 1 rank
        let shfl_down_value = group.shfl_down(my_value, 1);
        shfl_down_results
            .add(thread_idx as usize)
            .write(shfl_down_value);

        // Test 3: Shift up by 1 rank
        let shfl_up_value = group.shfl_up(my_value, 1);
        shfl_up_results
            .add(thread_idx as usize)
            .write(shfl_up_value);

        group.sync();
    }
}

/// Test vote operations (any, all, ballot) with coalesced group.
///
/// This kernel tests:
/// 1. any() - returns true if any thread has predicate true
/// 2. all() - returns true if all threads have predicate true
/// 3. ballot() - returns mask of threads with predicate true
///
/// Only even threads participate (0,2,4,...,30) with predicate based on rank < 8.
#[kernel]
pub unsafe fn test_coalesced_vote(
    any_results: *mut u32,
    all_results: *mut u32,
    ballot_results: *mut u32,
) {
    let thread_idx = thread::thread_idx_x();

    // Only even threads participate
    // Write to local memory to prevent compiler optimization
    let mut local_idx = thread_idx;
    core::ptr::write_volatile(&mut local_idx, thread_idx);
    let condition = core::ptr::read_volatile(&local_idx) % 2 == 0;

    if condition {
        let group = coalesced_threads();

        let rank = group.thread_rank();

        // Predicate: rank < 8 (true for first 8 even threads: 0,2,4,...,14)
        let predicate = rank < 8;

        // Test 1: any() - should be true (some threads have rank < 8)
        let any_result = if group.any(predicate) { 1 } else { 0 };
        any_results.add(thread_idx as usize).write(any_result);

        // Test 2: all() - should be false (not all threads have rank < 8)
        let all_result = if group.all(predicate) { 1 } else { 0 };
        all_results.add(thread_idx as usize).write(all_result);

        // Test 3: ballot() - should show first 8 ranks (lanes 0,2,4,...,14)
        let ballot_result = group.ballot(predicate);
        ballot_results.add(thread_idx as usize).write(ballot_result);

        group.sync();
    }
}
