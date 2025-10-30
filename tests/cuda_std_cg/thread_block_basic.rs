mod common;
use common::*;

/// Test basic ThreadBlock functionality
///
/// Verifies:
/// - this_thread_block() creates valid handle
/// - sync() synchronizes all threads in block
/// - size() returns correct value
/// - thread_rank() returns unique values 0..size-1
/// - Multiple blocks operate independently
#[test]
fn test_thread_block_basic() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("thread_block_basic_test")?;

    // Test configuration: 2 blocks of 128 threads each
    let blocks = 2u32;
    let threads_per_block = 128u32;
    let total_threads = blocks * threads_per_block;

    // Allocate device memory
    let output_ranks = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_size = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let sync_flag = DeviceBuffer::from_slice(&vec![0u32; blocks as usize])?;

    // Launch kernel (regular launch, not cooperative - block sync doesn't need it)
    unsafe {
        launch!(
            kernel<<<blocks, threads_per_block, 0, stream>>>(
                output_ranks.as_device_ptr(),
                output_size.as_device_ptr(),
                sync_flag.as_device_ptr()
            )
        )?;
    }

    // Synchronize and copy results
    stream.synchronize()?;

    let mut ranks_host = vec![0u32; total_threads as usize];
    let mut size_host = vec![0u32; total_threads as usize];
    let mut sync_flag_host = vec![0u32; blocks as usize];

    output_ranks.copy_to(&mut ranks_host)?;
    output_size.copy_to(&mut size_host)?;
    sync_flag.copy_to(&mut sync_flag_host)?;

    // Verify results
    for block_idx in 0..blocks {
        let block_offset = (block_idx * threads_per_block) as usize;

        // Verify size() returns correct value for all threads in block
        for thread_rank in 0..threads_per_block {
            let idx = block_offset + thread_rank as usize;
            assert_eq!(
                size_host[idx], threads_per_block,
                "Block {} thread {} reports incorrect size: got {}, expected {}",
                block_idx, thread_rank, size_host[idx], threads_per_block
            );
        }

        // Verify thread_rank() returns unique values 0..127 within each block
        let mut block_ranks: Vec<u32> =
            ranks_host[block_offset..block_offset + threads_per_block as usize].to_vec();
        block_ranks.sort();

        for (i, rank) in block_ranks.iter().enumerate() {
            assert_eq!(
                *rank, i as u32,
                "Block {} missing rank {} or has duplicate",
                block_idx, i
            );
        }

        // Verify sync completed (flag set by thread 0)
        assert_eq!(
            sync_flag_host[block_idx as usize], 1,
            "Block {} sync flag not set correctly",
            block_idx
        );
    }

    println!("✓ ThreadBlock basic test passed:");
    println!("  - {} blocks of {} threads", blocks, threads_per_block);
    println!("  - size() returned correct value");
    println!(
        "  - thread_rank() returned unique values 0..{}",
        threads_per_block - 1
    );
    println!("  - sync() completed successfully in all blocks");

    Ok(())
}

/// Test ThreadBlock dimension accessors
#[test]
fn test_thread_block_dimensions() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let kernel = module.get_function("thread_block_dimensions_test")?;

    // Test with 2D block: 16x8 = 128 threads
    let blocks = 1u32;
    let block_dim_x = 16u32;
    let block_dim_y = 8u32;
    let block_dim_z = 1u32;
    let total_threads = block_dim_x * block_dim_y * block_dim_z;

    let output_dim_x = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_dim_y = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_dim_z = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_idx_x = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_idx_y = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;
    let output_idx_z = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<blocks, (block_dim_x, block_dim_y, block_dim_z), 0, stream>>>(
                output_dim_x.as_device_ptr(),
                output_dim_y.as_device_ptr(),
                output_dim_z.as_device_ptr(),
                output_idx_x.as_device_ptr(),
                output_idx_y.as_device_ptr(),
                output_idx_z.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut dim_x_host = vec![0u32; total_threads as usize];
    let mut dim_y_host = vec![0u32; total_threads as usize];
    let mut dim_z_host = vec![0u32; total_threads as usize];
    let mut idx_x_host = vec![0u32; total_threads as usize];
    let mut idx_y_host = vec![0u32; total_threads as usize];
    let mut idx_z_host = vec![0u32; total_threads as usize];

    output_dim_x.copy_to(&mut dim_x_host)?;
    output_dim_y.copy_to(&mut dim_y_host)?;
    output_dim_z.copy_to(&mut dim_z_host)?;
    output_idx_x.copy_to(&mut idx_x_host)?;
    output_idx_y.copy_to(&mut idx_y_host)?;
    output_idx_z.copy_to(&mut idx_z_host)?;

    // Verify dimensions are correct for all threads
    for i in 0..total_threads as usize {
        assert_eq!(dim_x_host[i], block_dim_x, "dim_x mismatch at thread {}", i);
        assert_eq!(dim_y_host[i], block_dim_y, "dim_y mismatch at thread {}", i);
        assert_eq!(dim_z_host[i], block_dim_z, "dim_z mismatch at thread {}", i);
    }

    // Verify thread indices are in correct ranges
    for i in 0..total_threads as usize {
        assert!(
            idx_x_host[i] < block_dim_x,
            "idx_x out of range at thread {}",
            i
        );
        assert!(
            idx_y_host[i] < block_dim_y,
            "idx_y out of range at thread {}",
            i
        );
        assert!(
            idx_z_host[i] < block_dim_z,
            "idx_z out of range at thread {}",
            i
        );
    }

    println!("✓ ThreadBlock dimensions test passed:");
    println!(
        "  - Block dimensions: {}x{}x{}",
        block_dim_x, block_dim_y, block_dim_z
    );
    println!("  - All dimension accessors returned correct values");

    Ok(())
}

/// Test multiple syncs in sequence
#[test]
fn test_thread_block_multi_sync() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let kernel = module.get_function("thread_block_multi_sync_test")?;

    let blocks = 2u32;
    let threads_per_block = 64u32;
    let total_threads = blocks * threads_per_block;

    let counter = DeviceBuffer::from_slice(&vec![0u32; total_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<blocks, threads_per_block, 0, stream>>>(
                counter.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut counter_host = vec![0u32; total_threads as usize];
    counter.copy_to(&mut counter_host)?;

    // All values should be 2 after three phases with syncs
    for (i, &value) in counter_host.iter().enumerate() {
        assert_eq!(value, 2, "Thread {} has incorrect value: {}", i, value);
    }

    println!("✓ ThreadBlock multi-sync test passed:");
    println!("  - Multiple sync calls completed without deadlock");
    println!("  - All threads see consistent data after each sync");

    Ok(())
}
