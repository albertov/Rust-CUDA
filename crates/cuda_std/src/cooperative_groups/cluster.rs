//! Thread block cluster synchronization primitives for CUDA cooperative groups.
//!
//! This module provides support for thread block clusters, a feature available on
//! SM 9.0+ (Hopper architecture: H100, H200). Clusters are groups of thread blocks
//! that are co-scheduled on the same streaming multiprocessor (SM) and can cooperate
//! via fast synchronization and distributed shared memory.
//!
//! # Overview
//!
//! A thread block cluster is a group of thread blocks (typically configured as
//! 2x2x1, 4x1x1, etc.) that are guaranteed to execute concurrently on the same SM.
//! This enables:
//!
//! - **Fast inter-block synchronization**: Much faster than grid-level sync
//! - **Distributed shared memory**: Access shared memory from other blocks in cluster
//! - **Guaranteed co-scheduling**: All blocks in cluster run simultaneously
//!
//! # Hardware Requirements
//!
//! - **GPU**: SM 9.0+ (Hopper architecture: H100, H200)
//! - **PTX**: 7.8+
//! - **CUDA**: 11.8+
//! - **Launch**: Requires `cudaLaunchKernelEx` with cluster configuration
//!
//! # Launch Configuration
//!
//! Cluster kernels cannot be launched with standard `cudaLaunchKernel`. They require
//! special configuration via `cudaLaunchKernelEx`:
//!
//! ```c
//! // C/C++ example (not Rust)
//! cudaLaunchConfig_t config = {0};
//! config.gridDim = dim3(blocks_x, blocks_y, blocks_z);
//! config.blockDim = dim3(threads_x, threads_y, threads_z);
//!
//! cudaLaunchAttribute attrs[1];
//! attrs[0].id = cudaLaunchAttributeClusterDimension;
//! attrs[0].val.clusterDim.x = 2;  // 2x2x1 cluster
//! attrs[0].val.clusterDim.y = 2;
//! attrs[0].val.clusterDim.z = 1;
//!
//! config.attrs = attrs;
//! config.numAttrs = 1;
//!
//! cudaLaunchKernelEx(&config, kernel, args...);
//! ```
//!
//! # Example Usage
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn cluster_kernel(data: *mut i32) {
//!     let cluster = this_cluster();
//!     let block = this_thread_block();
//!
//!     // Each block writes its cluster rank
//!     if block.thread_rank() == 0 {
//!         *data.add(cluster.block_rank() as usize) = cluster.block_rank() as i32;
//!     }
//!
//!     // Synchronize across all blocks in the cluster
//!     cluster.sync();
//!
//!     // Now all blocks can safely read data from all other blocks
//!     let first_block_data = *data.add(0);
//! }
//! ```
//!
//! # Performance Considerations
//!
//! - **Cluster sync**: Faster than grid sync, but slower than block sync
//! - **Cluster size**: Typical sizes are 2x2x1 (4 blocks) or 4x1x1 (4 blocks)
//! - **SM occupancy**: Clusters consume entire SM, affecting occupancy
//!
//! # Distributed Shared Memory
//!
//! Hopper GPUs support accessing shared memory from other blocks within the same
//! cluster. This requires additional PTX instructions (`mapa.shared::cluster`) and
//! is not yet implemented in this version.
//!
//! # Safety
//!
//! The public API is safe because:
//! - `ThreadBlockCluster` lifetime is tied to kernel scope (cannot escape)
//! - Type system prevents misuse of cluster handles
//! - Unsafe operations are encapsulated
//!
//! However, incorrect usage patterns can still cause:
//! - **Deadlock**: If not all threads in all blocks call `sync()`
//! - **Runtime errors**: If kernel not launched with cluster configuration
//!
//! # Compatibility Notes
//!
//! This module will compile on all CUDA targets but will only function correctly
//! on SM 9.0+ hardware with proper cluster launch configuration. Using cluster
//! operations on older hardware or without cluster launch will result in undefined
//! behavior or runtime errors.

use core::marker::PhantomData;

/// A handle to all blocks within a thread block cluster.
///
/// `ThreadBlockCluster` represents the set of thread blocks within the current
/// cluster. A cluster is a group of blocks co-scheduled on the same SM that can
/// cooperate via fast synchronization and distributed shared memory.
///
/// # Requirements
///
/// - **Hardware**: SM 9.0+ (Hopper architecture: H100, H200)
/// - **Launch**: Must use `cudaLaunchKernelEx` with cluster configuration
/// - **Uniform participation**: ALL threads in ALL blocks must call `sync()`
///
/// # Lifetime
///
/// The `'a` lifetime parameter ties the cluster handle to the kernel's execution
/// scope. This prevents the handle from escaping the kernel, which would be
/// invalid since clusters only exist during kernel execution.
///
/// # Construction
///
/// Create via the `this_cluster()` function:
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
/// let cluster = this_cluster();
/// ```
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn multi_block_reduction(data: *mut i32, output: *mut i32) {
///     let cluster = this_cluster();
///     let block = this_thread_block();
///
///     // Phase 1: Each block computes local reduction
///     let block_result = compute_block_reduction(data);
///
///     // Store block result
///     if block.thread_rank() == 0 {
///         *output.add(cluster.block_rank() as usize) = block_result;
///     }
///
///     // Synchronize across entire cluster
///     cluster.sync();
///
///     // Phase 2: Block 0 computes cluster-wide reduction
///     if cluster.block_rank() == 0 && block.thread_rank() == 0 {
///         let mut cluster_result = 0;
///         for i in 0..cluster.num_blocks() {
///             cluster_result += *output.add(i as usize);
///         }
///         *output = cluster_result;
///     }
/// }
/// ```
#[repr(C)]
pub struct ThreadBlockCluster<'a> {
    /// Phantom lifetime marker ensuring ThreadBlockCluster cannot outlive the kernel invocation.
    _marker: PhantomData<&'a ()>,
}

/// Creates a thread block cluster handle for the current cluster.
///
/// This function constructs a `ThreadBlockCluster` representing all blocks within
/// the executing cluster. The returned handle can be used to perform cluster-level
/// synchronization via the `sync()` method.
///
/// # Requirements
///
/// - **Hardware**: SM 9.0+ (Hopper architecture)
/// - **Launch**: Kernel must be launched with cluster configuration
///
/// # Returns
///
/// A `ThreadBlockCluster<'static>` handle tied to the kernel's execution. The
/// `'static` lifetime indicates the handle is valid for the entire kernel execution.
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn my_kernel(data: *mut i32) {
///     let cluster = this_cluster();
///
///     // Use cluster handle for synchronization
///     process_local_data(data);
///     cluster.sync();
///     update_shared_state(data);
/// }
/// ```
#[inline(always)]
pub fn this_cluster() -> ThreadBlockCluster<'static> {
    ThreadBlockCluster {
        _marker: PhantomData,
    }
}

impl<'a> ThreadBlockCluster<'a> {
    /// Synchronizes all threads in all blocks within the cluster.
    ///
    /// This is a cluster-level barrier. Execution resumes only after ALL threads
    /// in ALL blocks in the cluster have reached this synchronization point.
    ///
    /// # Requirements
    ///
    /// - **Hardware**: SM 9.0+ (Hopper architecture)
    /// - **Launch**: Kernel must be launched with cluster configuration
    /// - **Uniform participation**: ALL threads in ALL blocks must call `sync()`
    ///
    /// # Memory Ordering
    ///
    /// Cluster sync provides strong memory ordering guarantees:
    /// - All memory modifications before `sync()` are visible to all threads
    ///   in all blocks in the cluster after `sync()` returns.
    /// - This includes shared memory, global memory, and distributed shared memory.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `barrier.cluster.aligned` instruction for SM 9.0+.
    /// This instruction synchronizes all threads across all blocks in the cluster.
    ///
    /// # Deadlock Prevention
    ///
    /// **SAFE - All threads participate**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let cluster = this_cluster();
    /// // All threads in all blocks execute sync unconditionally
    /// cluster.sync();
    /// ```
    ///
    /// **UNSAFE - Divergent sync**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let cluster = this_cluster();
    /// # let condition = true;
    /// // DEADLOCK: Only some blocks call sync
    /// if condition {
    ///     cluster.sync(); // Some blocks skip this!
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// Cluster-level sync is faster than grid-level sync but slower than block-level
    /// sync. Typical latency is in the range of tens to hundreds of nanoseconds.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn cluster_sync_kernel(data: *mut i32) {
    ///     let cluster = this_cluster();
    ///     let block = this_thread_block();
    ///
    ///     // Each block writes its rank
    ///     if block.thread_rank() == 0 {
    ///         *data.add(cluster.block_rank() as usize) = cluster.block_rank() as i32;
    ///     }
    ///
    ///     // Barrier: ensure all blocks wrote their data
    ///     cluster.sync();
    ///
    ///     // All blocks can now safely read all data
    ///     let first_block_data = *data.add(0);
    /// }
    /// ```
    #[inline(always)]
    pub fn sync(&self) {
        // Cluster-level synchronization using barrier.cluster.aligned instruction.
        //
        // This instruction is available on SM 9.0+ (Hopper architecture) and
        // synchronizes all threads across all blocks in the cluster.
        //
        // PTX Instruction:
        //   barrier.cluster.aligned;
        //
        // Unlike block-level barriers (bar.sync), cluster barriers do not require
        // an explicit thread count parameter. The hardware knows the cluster size
        // from the launch configuration.
        //
        // IMPORTANT: The kernel must be launched with cudaLaunchKernelEx and
        // cluster configuration. Using this instruction without proper launch
        // configuration results in undefined behavior.

        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            // Execute cluster-level barrier (SM 9.0+)
            #[cfg(target_os = "cuda")]
            asm!(
                "barrier.cluster.aligned;",
                options(nostack)
            );
        }
    }

    /// Returns the total number of blocks in the cluster.
    ///
    /// This is the product of cluster dimensions:
    /// ```text
    /// num_blocks = cluster_dim.x * cluster_dim.y * cluster_dim.z
    /// ```
    ///
    /// # Returns
    ///
    /// Total block count in the cluster. Block ranks are in the range
    /// `[0, num_blocks())`.
    ///
    /// # PTX Implementation
    ///
    /// Reads from special register `%nclusterid` which contains the total
    /// number of blocks in the cluster.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let cluster = this_cluster();
    ///     let total_blocks = cluster.num_blocks();
    ///
    ///     // Distribute work across blocks
    ///     if cluster.block_rank() < total_blocks / 2 {
    ///         // First half of blocks
    ///     } else {
    ///         // Second half of blocks
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub fn num_blocks(&self) -> u32 {
        // Read total number of blocks in cluster from %nclusterid
        #[cfg(target_os = "cuda")]
        unsafe {
            use core::arch::asm;
            let mut size: u32;
            asm!(
                "mov.u32 {size}, %nclusterid;",
                size = out(reg32) size,
                options(nostack)
            );
            size
        }
        #[cfg(not(target_os = "cuda"))]
        {
            1 // Default for non-CUDA compilation
        }
    }

    /// Returns the rank of the calling block within the cluster.
    ///
    /// Block ranks uniquely identify each block within the cluster. Ranks are
    /// computed from block indices within the cluster, arranged in row-major order.
    ///
    /// # Returns
    ///
    /// Block index within the cluster in the range `[0, num_blocks())`.
    ///
    /// # PTX Implementation
    ///
    /// Reads from special register `%clusterid` which contains the block's
    /// rank within the cluster.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example(output: *mut i32) {
    ///     let cluster = this_cluster();
    ///     let block = this_thread_block();
    ///
    ///     // Each block writes its cluster rank
    ///     if block.thread_rank() == 0 {
    ///         *output.add(cluster.block_rank() as usize) = cluster.block_rank() as i32;
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub fn block_rank(&self) -> u32 {
        // Read block rank within cluster from %clusterid
        #[cfg(target_os = "cuda")]
        unsafe {
            use core::arch::asm;
            let mut rank: u32;
            asm!(
                "mov.u32 {rank}, %clusterid;",
                rank = out(reg32) rank,
                options(nostack)
            );
            rank
        }
        #[cfg(not(target_os = "cuda"))]
        {
            0 // Default for non-CUDA compilation
        }
    }

    /// Returns the x-dimension of the cluster (blocks in x-direction).
    ///
    /// # Returns
    ///
    /// Number of blocks in the x-dimension of the cluster.
    ///
    /// # PTX Implementation
    ///
    /// Reads from special register `%cluster_nctaid.x`.
    #[inline(always)]
    pub fn dim_blocks_x(&self) -> u32 {
        #[cfg(target_os = "cuda")]
        unsafe {
            use core::arch::asm;
            let mut dim: u32;
            asm!(
                "mov.u32 {dim}, %cluster_nctaid.x;",
                dim = out(reg32) dim,
                options(nostack)
            );
            dim
        }
        #[cfg(not(target_os = "cuda"))]
        {
            1 // Default for non-CUDA compilation
        }
    }

    /// Returns the y-dimension of the cluster (blocks in y-direction).
    ///
    /// # Returns
    ///
    /// Number of blocks in the y-dimension of the cluster.
    ///
    /// # PTX Implementation
    ///
    /// Reads from special register `%cluster_nctaid.y`.
    #[inline(always)]
    pub fn dim_blocks_y(&self) -> u32 {
        #[cfg(target_os = "cuda")]
        unsafe {
            use core::arch::asm;
            let mut dim: u32;
            asm!(
                "mov.u32 {dim}, %cluster_nctaid.y;",
                dim = out(reg32) dim,
                options(nostack)
            );
            dim
        }
        #[cfg(not(target_os = "cuda"))]
        {
            1 // Default for non-CUDA compilation
        }
    }

    /// Returns the z-dimension of the cluster (blocks in z-direction).
    ///
    /// # Returns
    ///
    /// Number of blocks in the z-dimension of the cluster.
    ///
    /// # PTX Implementation
    ///
    /// Reads from special register `%cluster_nctaid.z`.
    #[inline(always)]
    pub fn dim_blocks_z(&self) -> u32 {
        #[cfg(target_os = "cuda")]
        unsafe {
            use core::arch::asm;
            let mut dim: u32;
            asm!(
                "mov.u32 {dim}, %cluster_nctaid.z;",
                dim = out(reg32) dim,
                options(nostack)
            );
            dim
        }
        #[cfg(not(target_os = "cuda"))]
        {
            1 // Default for non-CUDA compilation
        }
    }
}

// Safety: ThreadBlockCluster can be safely sent between threads within the kernel
// The cluster handle contains no mutable state
unsafe impl<'a> Send for ThreadBlockCluster<'a> {}

// Safety: ThreadBlockCluster can be safely shared between threads within the kernel
// The underlying synchronization primitives are designed for concurrent use
unsafe impl<'a> Sync for ThreadBlockCluster<'a> {}

// Implement ThreadGroup trait for polymorphic cooperative group operations
impl<'a> super::traits::ThreadGroup for ThreadBlockCluster<'a> {
    /// Synchronizes all threads in all blocks within the cluster.
    ///
    /// Delegates to [`ThreadBlockCluster::sync()`].
    #[inline(always)]
    fn sync(&self) {
        ThreadBlockCluster::sync(self)
    }

    /// Returns the total number of threads in the cluster.
    ///
    /// This is computed as:
    /// ```text
    /// size = num_blocks * threads_per_block
    /// ```
    #[inline(always)]
    fn size(&self) -> u32 {
        let num_blocks = self.num_blocks();
        let block = super::this_thread_block();
        num_blocks * block.size()
    }

    /// Returns the global rank of the calling thread within the cluster.
    ///
    /// This is computed as:
    /// ```text
    /// rank = block_rank * threads_per_block + thread_rank_in_block
    /// ```
    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        let block_rank = self.block_rank();
        let block = super::this_thread_block();
        block_rank * block.size() + block.thread_rank()
    }

    /// Returns the thread participation mask for the current warp.
    ///
    /// For ThreadBlockCluster, this returns 0xFFFFFFFF (full warp participation)
    /// since clusters encompass all threads in each warp across all blocks.
    #[inline(always)]
    fn mask(&self) -> u32 {
        0xFFFFFFFF
    }
}
