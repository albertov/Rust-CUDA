//! Asynchronous memory copy operations for cooperative groups (SM 8.0+).
//!
//! This module provides asynchronous memory copy operations using the `cp.async`
//! PTX instructions introduced in CUDA 11.0 / PTX 7.0 for Ampere (SM 8.0+) GPUs.
//!
//! # Overview
//!
//! Async memory copy enables overlapping data movement with computation through
//! a pipeline pattern:
//!
//! 1. **Initiate async copy**: `memcpy_async()` - Start non-blocking copy from global to shared memory
//! 2. **Commit group**: `commit_group()` - Mark a batch of async operations as a group
//! 3. **Wait selectively**: `wait_prior::<N>()` - Wait for all but the last N groups
//! 4. **Process data**: Compute while next stage loads
//! 5. **Repeat**: Pipeline stages for maximum throughput
//!
//! # Hardware Requirements
//!
//! - **SM 8.0+**: Ampere architecture or newer (RTX 30xx, A100, etc.)
//! - **PTX 7.0+**: Required for cp.async instructions
//!
//! # Usage Patterns
//!
//! ## Basic Copy
//!
//! Simple global-to-shared copy with wait:
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//! use cuda_std::shared_array;
//!
//! #[kernel]
//! pub unsafe fn simple_copy(src: *const f32, dst: *mut f32) {
//!     let block = this_thread_block();
//!     let shared = shared_array![f32; 256];
//!
//!     // Async copy to shared memory
//!     block.memcpy_async(shared.as_mut_ptr(), src, 256);
//!     block.wait();
//!
//!     // Now process data in shared memory
//!     let tid = block.thread_rank();
//!     if tid < 256 {
//!         *dst.add(tid as usize) = *shared.as_ptr().add(tid as usize) * 2.0;
//!     }
//! }
//! ```
//!
//! ## Pipeline Pattern (Double Buffering)
//!
//! Overlap data loading with computation:
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//! use cuda_std::shared_array;
//!
//! #[kernel]
//! pub unsafe fn pipelined_processing(src: *const f32, dst: *mut f32, stages: u32) {
//!     let block = this_thread_block();
//!     let buffer0 = shared_array![f32; 128];
//!     let buffer1 = shared_array![f32; 128];
//!
//!     // Load first stage
//!     block.memcpy_async(buffer0.as_mut_ptr(), src, 128);
//!     block.commit_group();
//!
//!     for stage in 1..stages {
//!         let current = if stage % 2 == 0 { buffer0 } else { buffer1 };
//!         let prev = if stage % 2 == 1 { buffer0 } else { buffer1 };
//!
//!         // Load next stage (non-blocking)
//!         block.memcpy_async(current.as_mut_ptr(), src.add((stage * 128) as usize), 128);
//!         block.commit_group();
//!
//!         // Wait for previous stage to finish
//!         block.wait_prior::<1>();
//!
//!         // Process previous stage while next loads
//!         let tid = block.thread_rank();
//!         if tid < 128 {
//!             let idx = ((stage - 1) * 128 + tid) as usize;
//!             *dst.add(idx) = *prev.as_ptr().add(tid as usize) * 2.0;
//!         }
//!     }
//!
//!     // Process final stage
//!     block.wait();
//!     let final_buffer = if stages % 2 == 1 { buffer0 } else { buffer1 };
//!     let tid = block.thread_rank();
//!     if tid < 128 {
//!         let idx = ((stages - 1) * 128 + tid) as usize;
//!         *dst.add(idx) = *final_buffer.as_ptr().add(tid as usize) * 2.0;
//!     }
//! }
//! ```
//!
//! # Performance Considerations
//!
//! ## Benefits
//!
//! - **Overlapped execution**: Data transfer happens concurrently with computation
//! - **Reduced latency**: Hide global memory latency behind computation
//! - **Higher throughput**: Keep GPU cores busy during data loading
//!
//! ## Optimization Guidelines
//!
//! 1. **Alignment**: 16-byte aligned addresses perform best
//! 2. **Size**: Copy sizes of 4, 8, or 16 bytes have dedicated instructions
//! 3. **Pipeline depth**: 2-4 stages typically optimal (balance occupancy vs complexity)
//! 4. **Cache policy**: cp.async.ca uses L2 cache by default
//!
//! # Implementation Notes
//!
//! ## PTX Instructions
//!
//! - `cp.async.ca.shared.global [dst], [src], size`: Async copy with L2 cache
//! - `cp.async.commit_group`: Mark end of async operation group
//! - `cp.async.wait_group N`: Wait for all but last N groups
//! - `cp.async.wait_all`: Wait for all pending async operations
//!
//! ## Safety
//!
//! - **Shared memory destination**: Must be in shared memory address space
//! - **Global memory source**: Must be in global memory address space
//! - **Uniform participation**: All threads in group must call wait operations uniformly
//! - **Synchronization**: Proper wait/commit_group usage required to avoid race conditions

use core::arch::asm;

/// Trait providing asynchronous memory copy operations for cooperative groups.
///
/// This trait extends cooperative groups with async memory operations available
/// on SM 8.0+ (Ampere) GPUs.
pub trait AsyncMemory {
    /// Asynchronously copy memory from global to shared memory.
    ///
    /// Initiates a non-blocking copy from global memory (`src`) to shared memory
    /// (`dst`). The copy completes asynchronously while the kernel continues execution.
    /// Use `wait()` or `wait_prior()` to ensure completion before accessing data.
    ///
    /// # Arguments
    ///
    /// - `dst`: Destination pointer in shared memory
    /// - `src`: Source pointer in global memory
    /// - `count`: Number of elements of type `T` to copy
    ///
    /// # Requirements
    ///
    /// - SM 8.0+ (Ampere architecture or newer)
    /// - `dst` must point to shared memory
    /// - `src` must point to global memory
    /// - Both pointers should be 16-byte aligned for best performance
    ///
    /// # Safety
    ///
    /// - Caller must ensure `src` and `dst` are valid for `count` elements
    /// - No overlap between source and destination regions
    /// - Must call `wait()` or `wait_prior()` before accessing copied data
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn example(src: *const i32, dst: *mut i32) {
    ///     let block = this_thread_block();
    ///     let shared = shared_array![i32; 256];
    ///
    ///     block.memcpy_async(shared.as_mut_ptr(), src, 256);
    ///     block.wait();
    ///
    ///     // Data now available in shared memory
    /// }
    /// ```
    unsafe fn memcpy_async<T>(&self, dst: *mut T, src: *const T, count: usize);

    /// Wait for all pending asynchronous memory operations to complete.
    ///
    /// Blocks until all previously issued `memcpy_async()` operations have
    /// finished. This is equivalent to `wait_prior::<0>()`.
    ///
    /// # Requirements
    ///
    /// - All threads in the group must call this function uniformly
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn example(src: *const i32) {
    ///     let block = this_thread_block();
    ///     let shared = shared_array![i32; 128];
    ///
    ///     block.memcpy_async(shared.as_mut_ptr(), src, 128);
    ///     block.wait(); // Block until copy completes
    ///
    ///     // Safe to access shared memory now
    ///     let value = *shared.as_ptr();
    /// }
    /// ```
    fn wait(&self);

    /// Wait for all but the last `N` committed async operation groups.
    ///
    /// Enables pipeline patterns by allowing overlapped execution. With `N=1`,
    /// waits for all groups except the most recent, allowing the latest stage
    /// to load while processing the previous stage.
    ///
    /// # Type Parameters
    ///
    /// - `N`: Number of most recent groups to NOT wait for (typically 1-3)
    ///
    /// # Requirements
    ///
    /// - All threads in the group must call this function uniformly
    /// - Must call `commit_group()` after each batch of async operations
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn pipeline(src: *const i32, dst: *mut i32) {
    ///     let block = this_thread_block();
    ///     let buffer0 = shared_array![i32; 64];
    ///     let buffer1 = shared_array![i32; 64];
    ///
    ///     // Stage 0: Load first batch
    ///     block.memcpy_async(buffer0.as_mut_ptr(), src, 64);
    ///     block.commit_group();
    ///
    ///     // Stage 1: Load second batch, wait for first
    ///     block.memcpy_async(buffer1.as_mut_ptr(), src.add(64), 64);
    ///     block.commit_group();
    ///     block.wait_prior::<1>(); // Wait for all but last 1 group (buffer0 ready)
    ///
    ///     // Process buffer0 while buffer1 loads in background
    ///     // ...
    /// }
    /// ```
    fn wait_prior<const N: u32>(&self);

    /// Commit a group of asynchronous memory operations.
    ///
    /// Marks all `memcpy_async()` calls since the last `commit_group()` as a
    /// single unit for `wait_prior::<N>()` tracking. Required for pipeline patterns.
    ///
    /// # Requirements
    ///
    /// - All threads in the group must call this function uniformly
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuda_std::cooperative_groups::*;
    /// use cuda_std::shared_array;
    ///
    /// #[kernel]
    /// pub unsafe fn batched_loads(src: *const i32) {
    ///     let block = this_thread_block();
    ///     let buf0 = shared_array![i32; 32];
    ///     let buf1 = shared_array![i32; 32];
    ///
    ///     // Batch multiple async copies as one group
    ///     block.memcpy_async(buf0.as_mut_ptr(), src, 32);
    ///     block.memcpy_async(buf1.as_mut_ptr(), src.add(32), 32);
    ///     block.commit_group(); // Both copies are one group
    ///
    ///     block.wait();
    /// }
    /// ```
    fn commit_group(&self);
}

/// Implementation of async memory operations for ThreadBlock.
impl<'a> AsyncMemory for crate::cooperative_groups::ThreadBlock<'a> {
    #[inline(always)]
    unsafe fn memcpy_async<T>(&self, dst: *mut T, src: *const T, count: usize) {
        let size_bytes = count * core::mem::size_of::<T>();
        let thread_rank = self.thread_rank() as usize;
        let num_threads = self.size() as usize;

        // Calculate per-thread workload
        let bytes_per_thread = (size_bytes + num_threads - 1) / num_threads;
        let thread_offset = thread_rank * bytes_per_thread;

        if thread_offset < size_bytes {
            let copy_size = core::cmp::min(bytes_per_thread, size_bytes - thread_offset);

            let dst_ptr = (dst as *mut u8).add(thread_offset);
            let src_ptr = (src as *const u8).add(thread_offset);

            // Use cp.async.ca.shared.global with L2 cache policy
            // Note: cp.async requires size to be 4, 8, or 16 bytes for optimal performance
            // For arbitrary sizes, we use a loop of 16-byte copies
            let mut remaining = copy_size;
            let mut dst_cur = dst_ptr;
            let mut src_cur = src_ptr;

            while remaining >= 16 {
                asm!(
                    "cp.async.ca.shared.global [{dst}], [{src}], 16;",
                    dst = in(reg64) dst_cur,
                    src = in(reg64) src_cur,
                    options(nostack)
                );
                dst_cur = dst_cur.add(16);
                src_cur = src_cur.add(16);
                remaining -= 16;
            }

            // Handle remaining bytes with smaller cp.async sizes
            if remaining >= 8 {
                asm!(
                    "cp.async.ca.shared.global [{dst}], [{src}], 8;",
                    dst = in(reg64) dst_cur,
                    src = in(reg64) src_cur,
                    options(nostack)
                );
                dst_cur = dst_cur.add(8);
                src_cur = src_cur.add(8);
                remaining -= 8;
            }

            if remaining >= 4 {
                asm!(
                    "cp.async.ca.shared.global [{dst}], [{src}], 4;",
                    dst = in(reg64) dst_cur,
                    src = in(reg64) src_cur,
                    options(nostack)
                );
                dst_cur = dst_cur.add(4);
                src_cur = src_cur.add(4);
                remaining -= 4;
            }

            // For < 4 bytes, fall back to synchronous copy
            if remaining > 0 {
                core::ptr::copy_nonoverlapping(src_cur, dst_cur, remaining);
            }
        }
    }

    #[inline(always)]
    fn wait(&self) {
        unsafe {
            asm!(
                "cp.async.wait_all;",
                options(nostack)
            );
        }
    }

    #[inline(always)]
    fn wait_prior<const N: u32>(&self) {
        unsafe {
            asm!(
                "cp.async.wait_group {n};",
                n = const N,
                options(nostack)
            );
        }
    }

    #[inline(always)]
    fn commit_group(&self) {
        unsafe {
            asm!(
                "cp.async.commit_group;",
                options(nostack)
            );
        }
    }
}
