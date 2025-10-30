//! Asynchronous memory copy test kernels for cooperative groups (SM 8.0+).

use cuda_std::cooperative_groups::*;
use cuda_std::*;

/// Test 1: Basic memcpy_async from global to shared to global
#[kernel]
pub unsafe fn test_memcpy_async_basic(src: *const i32, dst: *mut i32, size: u32) {
    let block = this_thread_block();
    let shared = shared_array![i32; 256];

    // Async copy from global to shared
    block.memcpy_async(shared, src, size as usize);
    block.wait();

    // Sync copy from shared to global
    let tid = block.thread_rank();
    if tid < size {
        *dst.add(tid as usize) = *shared.add(tid as usize);
    }
}

/// Test 2: Pipeline pattern with double buffering and wait_prior
#[kernel]
pub unsafe fn test_pipeline_pattern(src: *const i32, dst: *mut i32, stages: u32) {
    let block = this_thread_block();
    let buffer0 = shared_array![i32; 128];
    let buffer1 = shared_array![i32; 128];

    if stages == 0 {
        return;
    }

    // Load first stage
    block.memcpy_async(buffer0, src, 128);
    block.commit_group();

    for stage in 1..stages {
        let current_buf = if stage % 2 == 0 { buffer0 } else { buffer1 };
        let prev_buf = if stage % 2 == 1 { buffer0 } else { buffer1 };

        // Load next stage (non-blocking)
        let src_offset = (stage * 128) as usize;
        block.memcpy_async(current_buf, src.add(src_offset), 128);
        block.commit_group();

        // Wait for previous stage to complete (all but last 1 group)
        block.wait_prior::<1>();

        // Process previous stage
        let tid = block.thread_rank();
        if tid < 128 {
            let dst_offset = ((stage - 1) * 128 + tid) as usize;
            *dst.add(dst_offset) = *prev_buf.add(tid as usize);
        }
    }

    // Process final stage
    block.wait();
    let final_buf = if (stages - 1) % 2 == 0 {
        buffer0
    } else {
        buffer1
    };
    let tid = block.thread_rank();
    if tid < 128 {
        let dst_offset = ((stages - 1) * 128 + tid) as usize;
        *dst.add(dst_offset) = *final_buf.add(tid as usize);
    }
}

/// Test 3: Multiple async copies with commit_group batching
#[kernel]
pub unsafe fn test_commit_group_batching(src: *const i32, dst: *mut i32, batch_count: u32) {
    let block = this_thread_block();
    let buffer0 = shared_array![i32; 64];
    let buffer1 = shared_array![i32; 64];
    let buffer2 = shared_array![i32; 64];
    let buffer3 = shared_array![i32; 64];

    // Issue batch_count async copies
    if batch_count > 0 {
        block.memcpy_async(buffer0, src, 64);
    }
    if batch_count > 1 {
        block.memcpy_async(buffer1, src.add(64), 64);
    }
    if batch_count > 2 {
        block.memcpy_async(buffer2, src.add(128), 64);
    }
    if batch_count > 3 {
        block.memcpy_async(buffer3, src.add(192), 64);
    }

    // Commit all as one group
    block.commit_group();

    // Wait for all
    block.wait();

    let tid = block.thread_rank();

    // Copy results back
    if batch_count > 0 && tid < 64 {
        *dst.add(tid as usize) = *buffer0.add(tid as usize);
    }
    if batch_count > 1 && tid < 64 {
        *dst.add(64 + tid as usize) = *buffer1.add(tid as usize);
    }
    if batch_count > 2 && tid < 64 {
        *dst.add(128 + tid as usize) = *buffer2.add(tid as usize);
    }
    if batch_count > 3 && tid < 64 {
        *dst.add(192 + tid as usize) = *buffer3.add(tid as usize);
    }
}

/// Test 4: Large copy (1024 elements)
#[kernel]
pub unsafe fn test_memcpy_async_large(src: *const i32, dst: *mut i32) {
    let block = this_thread_block();
    let shared = shared_array![i32; 1024];

    // Async copy 1024 elements (4KB)
    block.memcpy_async(shared, src, 1024);
    block.wait();

    // Copy back in parallel
    let tid = block.thread_rank();
    if tid < 1024 {
        *dst.add(tid as usize) = *shared.add(tid as usize);
    }
}
