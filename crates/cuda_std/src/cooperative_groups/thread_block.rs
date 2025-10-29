//! Thread block synchronization primitives for CUDA cooperative groups.
//!
//! This module provides the high-level API for synchronizing all threads within
//! a single thread block. Thread block groups enable intra-block cooperation and
//! synchronization patterns.
//!
//! # Overview
//!
//! The `ThreadBlock` type represents a handle to all threads within a single thread
//! block. It provides a safe, ergonomic wrapper around block-level synchronization
//! primitives.
//!
//! # Requirements
//!
//! Thread block synchronization has requirements:
//!
//! 1. **Uniform Participation**: ALL threads in the block must call `sync()`.
//!    Divergent sync calls cause deadlock.
//!
//! 2. **Single Block Scope**: `ThreadBlock` only synchronizes threads within the
//!    same block. Different blocks execute independently.
//!
//! # Example Usage
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn block_reduction(data: *mut f32, output: *mut f32) {
//!     let block = this_thread_block();
//!
//!     // Phase 1: Each thread loads its data
//!     let thread_data = *data.add(block.thread_rank() as usize);
//!
//!     // Synchronize: ensure all threads loaded data
//!     block.sync();
//!
//!     // Phase 2: Block-level reduction
//!     let result = reduce_block(thread_data);
//!
//!     // Thread 0 writes result
//!     if block.thread_rank() == 0 {
//!         *output.add(block_idx().x as usize) = result;
//!     }
//! }
//! ```
//!
//! # Memory Ordering
//!
//! Block sync provides memory ordering guarantees:
//!
//! - All memory operations before `sync()` are visible to all threads in the block
//!   after `sync()` returns.
//! - Uses block-level barrier instruction (`bar.sync`)
//!
//! # Common Patterns
//!
//! ## Shared Memory Cooperation
//!
//! ```no_run
//! # use cuda_std::cooperative_groups::*;
//! # use cuda_std::shared_array;
//! # #[kernel]
//! # pub unsafe fn example(input: *const f32, output: *mut f32) {
//! let block = this_thread_block();
//! let shared = shared_array![f32; 256];
//!
//! // Load from global to shared memory
//! shared[block.thread_rank() as usize] = *input.add(block.thread_rank() as usize);
//!
//! block.sync(); // Ensure all loads complete
//!
//! // Process shared memory data
//! let result = compute(shared[block.thread_rank() as usize]);
//!
//! block.sync(); // Ensure all processing complete
//!
//! // Write results back
//! *output.add(block.thread_rank() as usize) = result;
//! # }
//! ```
//!
//! ## Block-Level Reduction
//!
//! ```no_run
//! # use cuda_std::cooperative_groups::*;
//! # use cuda_std::shared_array;
//! # #[kernel]
//! # pub unsafe fn example(data: *const f32, partial: *mut f32) {
//! let block = this_thread_block();
//! let shared = shared_array![f32; 256];
//!
//! // Load data into shared memory
//! shared[block.thread_rank() as usize] = *data.add(thread::index() as usize);
//! block.sync();
//!
//! // Reduction tree
//! let mut stride = block.size() / 2;
//! while stride > 0 {
//!     if block.thread_rank() < stride {
//!         shared[block.thread_rank() as usize] += shared[(block.thread_rank() + stride) as usize];
//!     }
//!     block.sync();
//!     stride /= 2;
//! }
//!
//! // Thread 0 writes result
//! if block.thread_rank() == 0 {
//!     *partial.add(block_idx().x as usize) = shared[0];
//! }
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - **Fast synchronization**: Block-level sync is much faster than grid-level sync
//!   (nanoseconds vs microseconds)
//! - **Use liberally**: Unlike grid sync, block sync has low overhead
//! - **Warp divergence**: Minimize divergent code paths to avoid serialization
//!
//! # Safety
//!
//! The public API is safe because:
//! - `ThreadBlock` lifetime is tied to kernel scope (cannot escape)
//! - Type system prevents misuse of block handles
//! - Unsafe operations are encapsulated
//!
//! However, incorrect usage patterns can still cause:
//! - **Deadlock**: If not all threads call `sync()`
//! - **Race conditions**: If sync is placed incorrectly

use core::marker::PhantomData;

/// A handle to all threads within a single thread block.
///
/// `ThreadBlock` represents the set of threads within the current block.
/// It provides the `sync()` method for block-level synchronization and
/// methods to query block dimensions and thread positions.
///
/// # Lifetime
///
/// The `'a` lifetime parameter ties the thread block to the kernel's execution
/// scope. This prevents the handle from escaping the kernel, which would be
/// invalid since thread blocks only exist during kernel execution.
///
/// # Construction
///
/// Create via the `this_thread_block()` function:
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
/// let block = this_thread_block();
/// ```
///
/// # Example
///
/// ```no_run
/// use cuda_std::cooperative_groups::*;
///
/// #[kernel]
/// pub unsafe fn cooperative_kernel(data: *mut i32) {
///     let block = this_thread_block();
///
///     // Phase 1: computation
///     data.add(block.thread_rank() as usize).write(compute_value());
///
///     // Barrier: ensure all threads finished Phase 1
///     block.sync();
///
///     // Phase 2: all threads in block see Phase 1 results
///     let neighbor = data.add(((block.thread_rank() + 1) % block.size()) as usize).read();
///     process_neighbor(neighbor);
/// }
/// ```
#[repr(C)]
pub struct ThreadBlock<'a> {
    /// Phantom lifetime marker ensuring ThreadBlock cannot outlive the kernel invocation.
    _marker: PhantomData<&'a ()>,
}

/// Creates a thread block handle for all threads in the current block.
///
/// This function constructs a `ThreadBlock` representing all threads within
/// the executing block. The returned handle can be used to perform block-level
/// synchronization via the `sync()` method.
///
/// # Returns
///
/// A `ThreadBlock<'static>` handle tied to the kernel's execution. The `'static`
/// lifetime indicates the handle is valid for the entire kernel execution.
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
///     // Use block handle for synchronization
///     process_local_data(data);
///     block.sync();
///     update_shared_state(data);
/// }
/// ```
#[inline(always)]
pub fn this_thread_block() -> ThreadBlock<'static> {
    ThreadBlock {
        _marker: PhantomData,
    }
}

impl<'a> ThreadBlock<'a> {
    /// Synchronizes all threads within the thread block.
    ///
    /// This is a block-level barrier. Execution resumes only after ALL threads
    /// in the block have reached this synchronization point.
    ///
    /// # Requirements
    ///
    /// - **Uniform participation**: All threads in the block must call `sync()`.
    ///   Divergent calls (e.g., inside `if` without all threads taking the branch)
    ///   cause deadlock.
    ///
    /// # Memory Ordering
    ///
    /// Block sync provides strong memory ordering guarantees:
    /// - All memory modifications before `sync()` are visible to all threads
    ///   in the block after `sync()` returns.
    /// - This includes shared memory and global memory operations.
    ///
    /// # Implementation
    ///
    /// Uses the PTX `bar.sync` instruction with barrier ID 0 and explicit thread count.
    /// The thread count parameter ensures proper synchronization for the block size.
    ///
    /// # Deadlock Prevention
    ///
    /// **SAFE - All threads participate**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # let block = this_thread_block();
    /// // All threads execute sync unconditionally
    /// block.sync();
    /// ```
    ///
    /// **UNSAFE - Divergent sync**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # use cuda_std::thread::thread_idx_x;
    /// # let block = this_thread_block();
    /// # let condition = true;
    /// // DEADLOCK: Only some threads call sync
    /// if condition {
    ///     block.sync(); // Some threads skip this!
    /// }
    /// ```
    ///
    /// **SAFE - Uniform divergence**:
    /// ```no_run
    /// # use cuda_std::cooperative_groups::*;
    /// # use cuda_std::thread::thread_idx_x;
    /// # let block = this_thread_block();
    /// if thread_idx_x() < 128 {
    ///     block.sync(); // All threads with threadIdx.x < 128 sync
    /// } else {
    ///     block.sync(); // All threads with threadIdx.x >= 128 sync
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// Block-level sync is fast (nanoseconds). Use it freely for correctness.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn shared_mem_kernel(data: *mut f32) {
    ///     let block = this_thread_block();
    ///     let shared = shared_array![f32; 256];
    ///
    ///     // Load into shared memory
    ///     shared[block.thread_rank() as usize] = *data.add(block.thread_rank() as usize);
    ///
    ///     // Barrier: ensure all loads complete
    ///     block.sync();
    ///
    ///     // Process with shared memory
    ///     let result = process(shared);
    ///     *data.add(block.thread_rank() as usize) = result;
    /// }
    /// ```
    #[inline(always)]
    pub fn sync(&self) {
        // Block-level synchronization using bar.sync instruction.
        //
        // CRITICAL DIFFERENCE from grid.sync():
        // - Grid uses: barrier.sync 0 (NO thread count)
        // - Block uses: bar.sync 0, {thread_count} (WITH thread count)
        //
        // The bar.sync instruction requires explicit thread count parameter
        // to know how many threads must arrive before releasing the barrier.
        //
        // PTX Instruction:
        //   bar.sync 0, {threads_per_block};
        //
        // Where threads_per_block = blockDim.x * blockDim.y * blockDim.z
        //
        // Barrier ID 0 is the default block-level barrier. All threads in the
        // block must arrive at this barrier before any can proceed.

        #[allow(unused_unsafe)]
        unsafe {
            use crate::thread::{block_dim_x, block_dim_y, block_dim_z};
            #[cfg(target_os = "cuda")]
            use core::arch::asm;

            // Calculate total threads in block
            #[cfg_attr(not(target_os = "cuda"), allow(unused_variables))]
            let threads_per_block = block_dim_x() * block_dim_y() * block_dim_z();

            // Execute block-level barrier with explicit thread count
            #[cfg(target_os = "cuda")]
            asm!(
                "bar.sync 0, {threads};",
                threads = in(reg32) threads_per_block,
                options(nostack)
            );
        }
    }

    /// Returns the total number of threads in the thread block.
    ///
    /// This is the product of block dimensions:
    /// ```text
    /// size = blockDim.x * blockDim.y * blockDim.z
    /// ```
    ///
    /// # Returns
    ///
    /// Total thread count in the block. Thread ranks are in the range
    /// `[0, size())`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let block = this_thread_block();
    ///     let block_size = block.size();
    ///
    ///     // Perform block-level algorithm
    ///     if block.thread_rank() < block_size / 2 {
    ///         // First half of threads
    ///     } else {
    ///         // Second half of threads
    ///     }
    /// }
    /// ```
    #[inline(always)]
    pub fn size(&self) -> u32 {
        use crate::thread::{block_dim_x, block_dim_y, block_dim_z};

        // Total threads = block dimensions product
        block_dim_x() * block_dim_y() * block_dim_z()
    }

    /// Returns the rank of the calling thread within the thread block.
    ///
    /// Thread ranks uniquely identify each thread within the block. Ranks are
    /// computed from thread indices, arranged in row-major order (x varies
    /// fastest, then y, then z).
    ///
    /// # Returns
    ///
    /// Thread index within the block in the range `[0, size())`.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// rank = threadIdx.x + threadIdx.y * blockDim.x +
    ///        threadIdx.z * blockDim.x * blockDim.y
    /// ```
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn example() {
    ///     let block = this_thread_block();
    ///     let shared = shared_array![f32; 256];
    ///
    ///     // Each thread accesses its unique shared memory slot
    ///     shared[block.thread_rank() as usize] = compute();
    ///     block.sync();
    ///
    ///     // Process shared data
    ///     process(shared[block.thread_rank() as usize]);
    /// }
    /// ```
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        use crate::thread::{thread_idx_x, thread_idx_y, thread_idx_z};
        use crate::thread::{block_dim_x, block_dim_y};

        // Compute thread index within block (0 to threads_per_block - 1)
        // Linearize as: x + y * width + z * width * height
        thread_idx_x()
            + thread_idx_y() * block_dim_x()
            + thread_idx_z() * block_dim_x() * block_dim_y()
    }

    /// Returns the x-dimension of the thread block.
    ///
    /// # Returns
    ///
    /// Number of threads in the x-dimension (`blockDim.x`).
    #[inline(always)]
    pub fn dim_x(&self) -> u32 {
        crate::thread::block_dim_x()
    }

    /// Returns the y-dimension of the thread block.
    ///
    /// # Returns
    ///
    /// Number of threads in the y-dimension (`blockDim.y`).
    #[inline(always)]
    pub fn dim_y(&self) -> u32 {
        crate::thread::block_dim_y()
    }

    /// Returns the z-dimension of the thread block.
    ///
    /// # Returns
    ///
    /// Number of threads in the z-dimension (`blockDim.z`).
    #[inline(always)]
    pub fn dim_z(&self) -> u32 {
        crate::thread::block_dim_z()
    }

    /// Returns the thread's x-index within the block.
    ///
    /// # Returns
    ///
    /// Thread's x-coordinate (`threadIdx.x`).
    #[inline(always)]
    pub fn thread_index_x(&self) -> u32 {
        crate::thread::thread_idx_x()
    }

    /// Returns the thread's y-index within the block.
    ///
    /// # Returns
    ///
    /// Thread's y-coordinate (`threadIdx.y`).
    #[inline(always)]
    pub fn thread_index_y(&self) -> u32 {
        crate::thread::thread_idx_y()
    }

    /// Returns the thread's z-index within the block.
    ///
    /// # Returns
    ///
    /// Thread's z-coordinate (`threadIdx.z`).
    #[inline(always)]
    pub fn thread_index_z(&self) -> u32 {
        crate::thread::thread_idx_z()
    }
}

// Safety: ThreadBlock can be safely sent between threads within the kernel
// The block handle contains no mutable state
unsafe impl<'a> Send for ThreadBlock<'a> {}

// Safety: ThreadBlock can be safely shared between threads within the kernel
// The underlying synchronization primitives are designed for concurrent use
unsafe impl<'a> Sync for ThreadBlock<'a> {}

// Implement ThreadGroup trait for polymorphic cooperative group operations
impl<'a> super::traits::ThreadGroup for ThreadBlock<'a> {
    /// Synchronizes all threads within the thread block.
    ///
    /// Delegates to [`ThreadBlock::sync()`].
    #[inline(always)]
    fn sync(&self) {
        ThreadBlock::sync(self)
    }

    /// Returns the total number of threads in the thread block.
    ///
    /// Delegates to [`ThreadBlock::size()`].
    #[inline(always)]
    fn size(&self) -> u32 {
        ThreadBlock::size(self)
    }

    /// Returns the rank of the calling thread within the thread block.
    ///
    /// Delegates to [`ThreadBlock::thread_rank()`].
    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        ThreadBlock::thread_rank(self)
    }
}
