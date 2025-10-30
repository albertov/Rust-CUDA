//! Test kernels for thread block cluster cooperative groups (SM 9.0+)
//!
//! These kernels test cluster-level synchronization and query operations.
//! They require SM 9.0+ (Hopper architecture) and cluster launch configuration.

#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]

#[cfg(not(target_os = "cuda"))]
use cuda_std_macros::kernel;

#[cfg(target_os = "cuda")]
use cuda_std::prelude::*;

/// Test basic cluster query operations
///
/// This kernel tests:
/// - this_cluster() factory function
/// - num_blocks() - total blocks in cluster
/// - block_rank() - this block's rank
/// - dim_blocks_x/y/z() - cluster dimensions
///
/// Expected behavior:
/// - Each block writes its cluster rank to output[block_rank]
/// - Values should be unique and in range [0, num_blocks)
#[kernel]
#[no_mangle]
pub unsafe fn test_cluster_basic(output: *mut i32) {
    #[cfg(target_os = "cuda")]
    {
        use cuda_std::cooperative_groups::*;

        let cluster = this_cluster();
        let block = this_thread_block();

        // Only thread 0 in each block writes output
        if block.thread_rank() == 0 {
            let rank = cluster.block_rank();
            let num_blocks = cluster.num_blocks();

            // Write cluster info
            // Format: [block_rank, num_blocks, dim_x, dim_y, dim_z, ...]
            let base = rank as usize * 5;
            *output.add(base + 0) = rank as i32;
            *output.add(base + 1) = num_blocks as i32;
            *output.add(base + 2) = cluster.dim_blocks_x() as i32;
            *output.add(base + 3) = cluster.dim_blocks_y() as i32;
            *output.add(base + 4) = cluster.dim_blocks_z() as i32;
        }
    }
}

/// Test cluster synchronization
///
/// This kernel tests:
/// - cluster.sync() - synchronize all threads in all blocks
///
/// Expected behavior:
/// - Phase 1: Each block writes its rank
/// - cluster.sync()
/// - Phase 2: All blocks can read all ranks
/// - Verification: sum of all ranks should equal sum(0..num_blocks-1)
#[kernel]
#[no_mangle]
pub unsafe fn test_cluster_sync(data: *mut i32, output: *mut i32) {
    #[cfg(target_os = "cuda")]
    {
        use cuda_std::cooperative_groups::*;

        let cluster = this_cluster();
        let block = this_thread_block();

        // Phase 1: Each block writes its rank
        if block.thread_rank() == 0 {
            let rank = cluster.block_rank();
            *data.add(rank as usize) = rank as i32;
        }

        // Synchronize across entire cluster
        cluster.sync();

        // Phase 2: All blocks can read all ranks
        // Block 0, thread 0 computes sum and writes result
        if cluster.block_rank() == 0 && block.thread_rank() == 0 {
            let num_blocks = cluster.num_blocks();
            let mut sum: i32 = 0;

            for i in 0..num_blocks {
                sum += *data.add(i as usize);
            }

            // Expected sum: 0 + 1 + 2 + ... + (num_blocks-1) = num_blocks * (num_blocks-1) / 2
            let expected = (num_blocks * (num_blocks - 1)) / 2;

            *output.add(0) = sum;
            *output.add(1) = expected as i32;
        }
    }
}

/// Test ThreadGroup trait implementation for ThreadBlockCluster
///
/// This kernel tests:
/// - size() via ThreadGroup trait
/// - thread_rank() via ThreadGroup trait
/// - sync() via ThreadGroup trait
/// - mask() via ThreadGroup trait
///
/// Expected behavior:
/// - cluster.size() == num_blocks * threads_per_block
/// - thread_rank ranges from 0 to size()-1
/// - All threads can sync via trait method
#[kernel]
#[no_mangle]
pub unsafe fn test_cluster_trait(output: *mut i32) {
    #[cfg(target_os = "cuda")]
    {
        use cuda_std::cooperative_groups::*;

        let cluster = this_cluster();
        let block = this_thread_block();

        // Use cluster as ThreadGroup trait object
        fn test_generic_group<G: ThreadGroup>(group: &G) -> (u32, u32, u32) {
            let size = group.size();
            let rank = group.thread_rank();
            let mask = group.mask();
            (size, rank, mask)
        }

        let (size, thread_rank, mask) = test_generic_group(&cluster);

        // Each thread writes its results
        if block.thread_rank() == 0 {
            let rank = cluster.block_rank();
            let base = rank as usize * 4;

            *output.add(base + 0) = size as i32;
            *output.add(base + 1) = thread_rank as i32;
            *output.add(base + 2) = mask as i32;
            *output.add(base + 3) = cluster.num_blocks() as i32;
        }

        // Sync via trait
        ThreadGroup::sync(&cluster);
    }
}

/// Test multi-phase cluster synchronization
///
/// This kernel tests multiple sync points in sequence:
/// - Phase 1: Write data
/// - cluster.sync()
/// - Phase 2: Read and transform data
/// - cluster.sync()
/// - Phase 3: Verify transformations
///
/// Expected behavior:
/// - All phases complete correctly
/// - All blocks see consistent data at each sync point
#[kernel]
#[no_mangle]
pub unsafe fn test_cluster_multi_phase(data: *mut i32, output: *mut i32) {
    #[cfg(target_os = "cuda")]
    {
        use cuda_std::cooperative_groups::*;

        let cluster = this_cluster();
        let block = this_thread_block();
        let rank = cluster.block_rank();

        // Phase 1: Each block writes initial value
        if block.thread_rank() == 0 {
            *data.add(rank as usize) = (rank * 10) as i32;
        }
        cluster.sync();

        // Phase 2: Each block reads neighbor and adds to it
        if block.thread_rank() == 0 {
            let num_blocks = cluster.num_blocks();
            let next_rank = (rank + 1) % num_blocks;
            let neighbor_value = *data.add(next_rank as usize);
            *data.add(rank as usize) = neighbor_value + rank as i32;
        }
        cluster.sync();

        // Phase 3: Block 0 verifies all values
        if cluster.block_rank() == 0 && block.thread_rank() == 0 {
            let num_blocks = cluster.num_blocks();
            let mut all_correct = 1i32; // 1 = true, 0 = false

            for i in 0..num_blocks {
                let prev_rank = if i == 0 { num_blocks - 1 } else { i - 1 };
                let expected = (prev_rank * 10) + i;
                let actual = *data.add(i as usize);

                if actual != expected as i32 {
                    all_correct = 0;
                }
            }

            *output = all_correct;
        }
    }
}
