//! Simplified multi-block grid synchronization test WITHOUT sync_threads()
//!
//! This test validates the hypothesis that grid.sync() completes but is incompatible
//! with subsequent sync_threads() calls.
//!
//! Key differences from comparison_multiblock.rs:
//! - NO sync_threads() calls after grid.sync()
//! - NO shared memory reduction
//! - Direct global atomic accumulation from all threads
//! - Results array to track block completion
//!
//! Test pattern:
//! 1. Launch simplified Rust kernel with cooperative launch
//! 2. Verify kernel completes without deadlock
//! 3. Verify results array shows all blocks completed [1,1,1,1,1,1,1,1]
//! 4. Verify global sum matches expected value
//!
//! Expected outcomes:
//! - If test PASSES: Confirms grid.sync() works but sync_threads() is incompatible
//! - If test DEADLOCKS: Rules out sync_threads() interference, indicates grid.sync() internal bug

mod common;
use common::*;
use cust::function::BlockSize;

#[test]
fn test_grid_sync_multiblock_simple_rust() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Verify device supports cooperative launch
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Load Rust module
    let rust_module = load_rust_module()?;

    // Get kernel function
    let rust_kernel = rust_module.get_function("grid_sync_multiblock_simple_kernel")?;

    // Test configuration: 8 blocks, 256 threads per block
    let num_blocks = 8u32;
    let threads_per_block = 256u32;
    let array_size = (num_blocks * threads_per_block) as usize;

    // Check kernel occupancy before cooperative launch
    println!("Checking kernel occupancy...");
    let rust_occupancy = rust_kernel.check_kernel_occupancy(BlockSize::x(threads_per_block), 0)?;
    println!(
        "Rust kernel occupancy: {} blocks/SM (max {} blocks)",
        rust_occupancy.0, rust_occupancy.1
    );

    // Calculate safe grid size for cooperative launch
    let requested_blocks = num_blocks;
    let rust_max_blocks = rust_occupancy.1 as u32;
    let rust_actual_blocks = std::cmp::min(requested_blocks, rust_max_blocks);

    println!(
        "Rust: Requested {} blocks, max cooperative {}, using {}",
        requested_blocks, rust_max_blocks, rust_actual_blocks
    );

    // Allocate device memory for Rust kernel
    let rust_data = DeviceBuffer::<i32>::zeroed(array_size)?;
    let rust_global_sum = DeviceBuffer::<i32>::zeroed(1)?;
    let rust_results = DeviceBuffer::<i32>::zeroed(num_blocks as usize)?;

    // Launch Rust kernel cooperatively
    println!("Launching simplified Rust kernel (NO sync_threads())...");
    let rust_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            rust_kernel<<<(rust_actual_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_data.as_device_ptr(),
                array_size as i32,
                rust_global_sum.as_device_ptr(),
                rust_results.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    let rust_elapsed = rust_start.elapsed();
    println!("Rust kernel completed in {:?}", rust_elapsed);

    // Copy results back to host
    let mut rust_data_host = vec![0i32; array_size];
    let mut rust_sum_host = vec![0i32; 1];
    let mut rust_results_host = vec![0i32; num_blocks as usize];
    rust_data.copy_to(&mut rust_data_host)?;
    rust_global_sum.copy_to(&mut rust_sum_host)?;
    rust_results.copy_to(&mut rust_results_host)?;

    // Calculate expected sum for validation
    // Each position should contain its block ID, sum = sum of all block IDs
    let mut expected_sum = 0i32;
    for i in 0..array_size {
        let expected_block_id = (i / threads_per_block as usize) as i32;
        expected_sum += expected_block_id;
    }

    // Print results for debugging
    println!(
        "Configuration: {} blocks × {} threads = {} total threads",
        num_blocks, threads_per_block, array_size
    );
    println!("Rust global sum:     {}", rust_sum_host[0]);
    println!("Expected sum:        {}", expected_sum);
    println!("Results array:       {:?}", rust_results_host);

    // Validate results array - all blocks should have completed
    let expected_results = vec![1i32; num_blocks as usize];
    assert_eq!(
        rust_results_host, expected_results,
        "Not all blocks completed: expected {:?}, got {:?}",
        expected_results, rust_results_host
    );

    // Validate global sum against expected value
    assert_eq!(
        rust_sum_host[0], expected_sum,
        "Rust kernel did not produce expected sum"
    );

    // Validate data array contents
    for i in 0..array_size {
        let expected_block_id = (i / threads_per_block as usize) as i32;
        assert_eq!(
            rust_data_host[i], expected_block_id,
            "Data mismatch at index {}: expected {}, got {}",
            i, expected_block_id, rust_data_host[i]
        );
    }

    println!(
        "✓ Test PASSED: Simplified multiblock kernel (NO sync_threads()) completed successfully"
    );
    println!(
        "✓ Hypothesis CONFIRMED: grid.sync() completes but is incompatible with sync_threads()"
    );

    Ok(())
}
