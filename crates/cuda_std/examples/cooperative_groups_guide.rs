//! Comprehensive guide to using cooperative groups in Rust-CUDA
//!
//! This example demonstrates all major cooperative group features including:
//! - Grid-wide synchronization
//! - Thread block operations
//! - Warp-level tiles (static and dynamic)
//! - Coalesced groups for divergent execution
//! - Shuffle operations
//! - Vote/ballot operations
//! - Reduction collectives (SM 8.0+)
//! - Scan operations (prefix sums)
//! - Async memory operations (SM 8.0+)
//! - Thread block clusters (SM 9.0+)
//!
//! Note: This is a documentation example showing API usage patterns.
//! The functions are not meant to be compiled as-is.

use cuda_std::*;

/// Example 1: Grid-wide synchronization
///
/// Synchronize all threads across all blocks in a kernel launch.
/// Requires cooperative kernel launch via cudaLaunchCooperativeKernel.
///
/// # Use Case
/// Multi-phase algorithms where each phase requires a globally consistent
/// view of memory across all blocks.
#[kernel]
pub unsafe fn grid_sync_example(data: *mut i32, num_phases: u32) {
    let grid = this_grid();

    for phase in 0..num_phases {
        // Phase 1: Each block writes its data
        if thread::thread_idx_x() == 0 {
            let block_id = thread::block_idx_x() as usize;
            *data.add(block_id) = phase as i32;
        }

        // Synchronize across entire grid
        // All blocks wait here until all have completed Phase 1
        grid.sync();

        // Phase 2: All blocks can now read all data with consistency
        // The grid sync ensures all writes from Phase 1 are visible
        let total_blocks = thread::grid_dim_x() as usize;
        let mut sum = 0i32;
        for i in 0..total_blocks {
            sum += *data.add(i);
        }
    }
}

/// Example 2: Thread block operations
///
/// Synchronize all threads within a single block using shared memory.
///
/// # Use Case
/// Block-level reductions, shared memory cooperation, multi-phase
/// algorithms within a block.
#[kernel]
pub unsafe fn thread_block_example(input: *const f32, output: *mut f32) {
    let block = this_thread_block();
    let shared = shared_array![f32; 256];

    let tid = block.thread_rank() as usize;

    // Phase 1: Load from global memory to shared memory
    shared[tid] = *input.add(thread::index() as usize);

    // Synchronize all threads in block
    block.sync();

    // Phase 2: Each thread can now access all shared memory
    // Perform a simple neighbor average
    let left = if tid > 0 { shared[tid - 1] } else { shared[tid] };
    let right = if tid < block.size() as usize - 1 { shared[tid + 1] } else { shared[tid] };
    let avg = (left + shared[tid] + right) / 3.0;

    // Synchronize before writing back
    block.sync();

    shared[tid] = avg;

    // Final sync before writing to global memory
    block.sync();

    *output.add(thread::index() as usize) = shared[tid];
}

/// Example 3: Warp-level operations with static tiles
///
/// Use compile-time sized tiles for efficient warp-level operations.
///
/// # Use Case
/// Warp-level reductions, shuffle-based algorithms, efficient
/// fine-grained parallelism without shared memory.
#[kernel]
pub unsafe fn warp_operations_example(data: *mut i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Load data for this thread
    let tid = tile.thread_rank() as usize;
    let value = *data.add(tid);

    // Shuffle down: each thread gets value from thread+1
    let neighbor = tile.shfl_down(value, 1);

    // Vote operations: check if all values are positive
    let all_positive = tile.all(value > 0);

    // Ballot: collect votes as bitmask
    let positive_mask = tile.ballot(value > 0);

    // Warp reduction: sum across all threads in tile
    let mut sum = value;
    let mut offset = tile.size() / 2;
    while offset > 0 {
        sum += tile.shfl_down(sum, offset);
        offset /= 2;
    }

    // Thread 0 writes the warp sum
    if tile.thread_rank() == 0 {
        *output = sum;
    }
}

/// Example 4: Dynamic tile sizes
///
/// Use runtime-determined tile sizes for flexible algorithms.
///
/// # Use Case
/// When tile size depends on kernel parameters or runtime configuration.
#[kernel]
pub unsafe fn dynamic_tile_example(data: *mut i32, tile_size: u32) {
    let block = this_thread_block();
    let tile = tiled_partition_dynamic(&block, tile_size);

    let value = *data.add(tile.thread_rank() as usize);

    // Use tile operations with runtime size
    tile.sync();

    let broadcast_value = tile.shfl(value, 0);

    *data.add(tile.thread_rank() as usize) = broadcast_value;
}

/// Example 5: Coalesced groups for divergent execution
///
/// Handle divergent code paths where only some threads are active.
///
/// # Use Case
/// Algorithms with conditional execution where only active threads
/// need to synchronize or communicate.
///
/// # Note
/// The Rust compiler may optimize away divergence, causing activemask
/// to incorrectly report all threads active. For production divergent
/// code, consider calling from C++. See module documentation for details.
#[kernel]
pub unsafe fn coalesced_example(data: *mut i32, flags: *const bool) {
    let idx = thread::index() as usize;

    // Threads diverge based on flag
    if *flags.add(idx) {
        // Only threads with flag=true execute this branch
        let active_threads = coalesced_threads();

        // Synchronize only active threads
        active_threads.sync();

        // Get size and rank among active threads only
        let active_count = active_threads.size();
        let active_rank = active_threads.thread_rank();

        // Cooperate among active threads using shuffle
        let value = *data.add(idx);
        let neighbor = active_threads.shfl_down(value, 1);

        // Thread 0 among active threads does special work
        if active_rank == 0 {
            *data.add(idx) = active_count as i32;
        }
    }
}

/// Example 6: Hardware-accelerated reductions (SM 8.0+)
///
/// Use hardware instructions for efficient reductions.
///
/// # Requirements
/// - SM 8.0+ (Ampere architecture or newer)
///
/// # Use Case
/// High-performance reductions for sum, min, max, and bitwise operations.
#[cfg(target_feature = "sm_80")]
#[kernel]
pub unsafe fn reduction_example(data: *const i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = *data.add(tile.thread_rank() as usize);

    // Hardware-accelerated reductions (single instruction)
    let sum = tile.reduce_add(value);
    let min_val = tile.reduce_min(value);
    let max_val = tile.reduce_max(value);
    let and_val = tile.reduce_and(value);
    let or_val = tile.reduce_or(value);
    let xor_val = tile.reduce_xor(value);

    // Thread 0 writes results
    if tile.thread_rank() == 0 {
        *output.add(0) = sum;
        *output.add(1) = min_val;
        *output.add(2) = max_val;
    }
}

/// Example 7: Scan operations (prefix sums)
///
/// Compute parallel prefix sums across threads.
///
/// # Use Case
/// Stream compaction, parallel prefix algorithms, work allocation.
#[kernel]
pub unsafe fn scan_example(data: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = *data.add(tile.thread_rank() as usize);

    // Inclusive scan (prefix sum)
    // Each thread gets sum of all values from threads [0..rank] (inclusive)
    let inclusive_sum = tile.inclusive_scan_add(value);

    // Exclusive scan
    // Each thread gets sum of all values from threads [0..rank) (exclusive)
    // Thread 0 gets 0
    let exclusive_sum = tile.exclusive_scan_add(value);

    // Write results
    *data.add(tile.thread_rank() as usize) = inclusive_sum;
    *data.add(tile.thread_rank() as usize + 32) = exclusive_sum;
}

/// Example 8: Async memory operations (SM 8.0+)
///
/// Pipeline asynchronous memory copies for better performance.
///
/// # Requirements
/// - SM 8.0+ (Ampere architecture or newer)
///
/// # Use Case
/// Overlapping computation with memory transfers, reducing latency.
#[cfg(target_feature = "sm_80")]
#[kernel]
pub unsafe fn async_memory_example(src: *const i32, dst: *mut i32) {
    let block = this_thread_block();
    let shared = shared_array![i32; 256];

    // Async copy from global to shared memory
    block.memcpy_async(shared.as_mut_ptr(), src, 256);

    // Can do other work here while copy is in progress
    // ...

    // Wait for async copy to complete
    block.wait();

    // Process data in shared memory
    let tid = block.thread_rank() as usize;
    let value = shared[tid] * 2;

    block.sync();

    shared[tid] = value;

    block.sync();

    // Write back to global memory
    *dst.add(tid) = shared[tid];
}

/// Example 9: Partitioning groups
///
/// Split groups by labels or binary predicates.
///
/// # Use Case
/// Group threads with similar characteristics for specialized processing.
#[kernel]
pub unsafe fn partitioning_example(data: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = *data.add(tile.thread_rank() as usize);

    // Binary partition: split into two groups based on predicate
    let even_group = binary_partition(&tile, value % 2 == 0);

    // Synchronize within the partition
    even_group.sync();

    // Each partition can work independently
    let partition_sum = even_group.reduce_add(value);

    // Labeled partition: split into multiple groups by label
    let label = (value / 10) as u32;  // Group by tens digit
    let labeled_group = labeled_partition(&tile, label);

    labeled_group.sync();
}

/// Example 10: Thread block clusters (SM 9.0+)
///
/// Coordinate across multiple thread blocks within a cluster.
///
/// # Requirements
/// - SM 9.0+ (Hopper architecture or newer)
/// - Special launch configuration
///
/// # Use Case
/// Multi-block cooperation, distributed algorithms, locality optimization.
///
/// # Note
/// Clusters require cudaLaunchKernelEx which is not yet in rust-cuda.
/// This example shows the API but may not be testable on current hardware.
#[cfg(target_feature = "sm_90")]
#[kernel]
pub unsafe fn cluster_example(data: *mut i32) {
    let cluster = this_cluster();

    // Query cluster dimensions
    let cluster_size = cluster.num_blocks();
    let block_rank = cluster.block_rank();

    // Synchronize all blocks in the cluster
    cluster.sync();

    // Each block in cluster can coordinate
    if thread::thread_idx_x() == 0 {
        let value = block_rank as i32;
        *data.add(block_rank as usize) = value;
    }

    cluster.sync();
}

/// Example 11: Vote operations
///
/// Warp-level voting for predicates across threads.
///
/// # Use Case
/// Early exit detection, convergence testing, divergence analysis.
#[kernel]
pub unsafe fn vote_operations_example(data: *const f32, threshold: f32, output: *mut bool) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let value = *data.add(tile.thread_rank() as usize);
    let above_threshold = value > threshold;

    // Check if ANY thread has value above threshold
    let any_above = tile.any(above_threshold);

    // Check if ALL threads have value above threshold
    let all_above = tile.all(above_threshold);

    // Get bitmask of which threads are above threshold
    let ballot_mask = tile.ballot(above_threshold);
    let count_above = ballot_mask.count_ones();

    if tile.thread_rank() == 0 {
        *output.add(0) = any_above;
        *output.add(1) = all_above;
        *output.add(2) = count_above > 16;
    }
}

/// Example 12: Match operations (SM 7.0+)
///
/// Find threads with matching values.
///
/// # Requirements
/// - SM 7.0+ (Volta architecture or newer)
///
/// # Use Case
/// Value grouping, deduplication, consensus detection.
#[cfg(target_feature = "sm_70")]
#[kernel]
pub unsafe fn match_operations_example(values: *const i32, output: *mut u32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    let my_value = *values.add(tile.thread_rank() as usize);

    // Find all threads with the same value as this thread
    let match_mask = tile.match_any(my_value);

    // Check if ALL threads have the same value (unanimous)
    let (match_mask_all, all_agree) = tile.match_all(my_value);

    // Leader election: lowest rank in each group becomes leader
    let is_leader = match_mask.trailing_zeros() == tile.thread_rank();

    if is_leader {
        let group_size = match_mask.count_ones();
        *output.add(tile.thread_rank() as usize) = group_size;
    }
}

/// Example 13: Generic algorithms using ThreadGroup trait
///
/// Write algorithms that work with any group type.
///
/// # Use Case
/// Reusable algorithms that work at any synchronization scope.
fn generic_cooperative_algorithm<G: ThreadGroup>(
    group: &G,
    data: *mut i32,
) -> i32 {
    let rank = group.thread_rank() as usize;

    // Each thread contributes its data
    let value = unsafe { *data.add(rank) };

    // Synchronize to ensure all threads ready
    group.sync();

    // Perform reduction using group's reduction operation
    let sum = group.reduce_add(value);

    sum
}

#[kernel]
pub unsafe fn generic_example(data: *mut i32, output: *mut i32) {
    let block = this_thread_block();
    let tile = tiled_partition::<32>(&block);

    // Same algorithm works with both block and tile
    let block_sum = generic_cooperative_algorithm(&block, data);
    let tile_sum = generic_cooperative_algorithm(&tile, data.add(256));

    if block.thread_rank() == 0 {
        *output.add(0) = block_sum;
    }
    if tile.thread_rank() == 0 {
        *output.add(1) = tile_sum;
    }
}
