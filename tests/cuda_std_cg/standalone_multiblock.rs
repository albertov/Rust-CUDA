//! Standalone test for multi-block grid synchronization (Rust-only)
//!
//! This test validates that the Rust implementation of grid_sync_multiblock_kernel
//! works correctly without deadlocking. This bypasses C++ comparison infrastructure.
//!
//! Primary goal: Verify barrier fix - kernel should complete without deadlock

mod common;
use common::*;

#[test]
fn test_grid_sync_multiblock_standalone() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Verify device supports cooperative launch
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Load Rust module only
    let rust_module = load_rust_module()?;
    let rust_kernel = rust_module.get_function("grid_sync_multiblock_kernel")?;

    // Test configuration: 8 blocks, 256 threads per block
    let num_blocks = 8u32;
    let threads_per_block = 256u32;
    let array_size = (num_blocks * threads_per_block) as usize;

    // Allocate device memory
    let rust_data = DeviceBuffer::<i32>::zeroed(array_size)?;
    let rust_global_sum = DeviceBuffer::<i32>::zeroed(1)?;

    // Launch Rust kernel cooperatively
    println!("Launching Rust multiblock kernel...");
    println!(
        "Configuration: {} blocks × {} threads = {} total threads",
        num_blocks, threads_per_block, array_size
    );

    let rust_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            rust_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_data.as_device_ptr(),
                array_size as i32,
                rust_global_sum.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    let rust_elapsed = rust_start.elapsed();
    println!(
        "✓ Rust kernel completed in {:?} (NO DEADLOCK!)",
        rust_elapsed
    );

    // Copy results back to host
    let mut rust_data_host = vec![0i32; array_size];
    let mut rust_sum_host = vec![0i32; 1];
    rust_data.copy_to(&mut rust_data_host)?;
    rust_global_sum.copy_to(&mut rust_sum_host)?;

    // Calculate expected sum for validation
    // Each position should contain its block ID, sum = sum of all block IDs
    let mut expected_sum = 0i32;
    for i in 0..array_size {
        let expected_block_id = (i / threads_per_block as usize) as i32;
        expected_sum += expected_block_id;
    }

    println!("Rust global sum:     {}", rust_sum_host[0]);
    println!("Expected sum:        {}", expected_sum);

    // Validate against expected values
    assert_eq!(
        rust_sum_host[0], expected_sum,
        "Rust kernel did not produce expected sum (got {}, expected {})",
        rust_sum_host[0], expected_sum
    );

    // Verify data array has correct block IDs in each position
    for i in 0..array_size {
        let expected_block_id = (i / threads_per_block as usize) as i32;
        assert_eq!(
            rust_data_host[i], expected_block_id,
            "Position {} has wrong block ID (got {}, expected {})",
            i, rust_data_host[i], expected_block_id
        );
    }

    println!("✓ All data validated: kernel executed correctly!");
    println!("✓ BARRIER FIX VERIFIED: No deadlock occurred");

    Ok(())
}
