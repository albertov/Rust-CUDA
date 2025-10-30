//! Comparative validation test for multi-block grid synchronization
//!
//! This test validates that the Rust implementation of grid_sync_multiblock_kernel
//! produces identical results to the C++ reference implementation.
//!
//! Test pattern:
//! 1. Load both Rust and C++ PTX modules
//! 2. Allocate separate device buffers for each implementation
//! 3. Launch both kernels with IDENTICAL parameters using cooperative launch
//! 4. Compare outputs - both data array and global sum must match exactly
//!
//! Kernel behavior:
//! - Phase 1: Each thread writes its block ID to array positions
//! - Grid synchronization to ensure all writes are visible
//! - Phase 2: All threads read from all positions and compute sums
//! - Block-level reduction followed by global accumulation

mod common;
use common::*;
use cust::function::BlockSize;

#[test]
fn test_grid_sync_multiblock_rust_vs_cpp() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Verify device supports cooperative launch
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Load both modules
    let rust_module = load_rust_module()?;
    let cpp_module = load_cpp_module("multiblock")?;

    // Get kernel functions
    let rust_kernel = rust_module.get_function("grid_sync_multiblock_kernel")?;
    let cpp_kernel = cpp_module.get_function("grid_sync_multiblock_kernel")?;

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

    let cpp_occupancy = cpp_kernel.check_kernel_occupancy(BlockSize::x(threads_per_block), 0)?;
    println!(
        "C++ kernel occupancy: {} blocks/SM (max {} blocks)",
        cpp_occupancy.0, cpp_occupancy.1
    );

    // Use requested blocks directly without occupancy constraint
    // This matches C++ reference behavior which uses requested blocks directly
    // The occupancy check is kept for informational purposes only
    let requested_blocks = num_blocks;
    let rust_max_blocks = rust_occupancy.1 as u32;
    let cpp_max_blocks = cpp_occupancy.1 as u32;

    // CHANGED: Remove occupancy constraint to match C++ behavior
    // Previous code used: std::cmp::min(requested_blocks, rust/cpp_max_blocks)
    // This caused block count mismatch and broke atomic counter algorithm
    let rust_actual_blocks = requested_blocks;
    let cpp_actual_blocks = requested_blocks;

    println!(
        "Rust: Requested {} blocks, max cooperative {}, using {} (no constraint)",
        requested_blocks, rust_max_blocks, rust_actual_blocks
    );
    println!(
        "C++: Requested {} blocks, max cooperative {}, using {} (no constraint)",
        requested_blocks, cpp_max_blocks, cpp_actual_blocks
    );

    // Allocate device memory for Rust kernel
    let rust_data = DeviceBuffer::<i32>::zeroed(array_size)?;
    let rust_global_sum = DeviceBuffer::<i32>::zeroed(1)?;

    // Allocate device memory for C++ kernel
    let cpp_data = DeviceBuffer::<i32>::zeroed(array_size)?;
    let cpp_global_sum = DeviceBuffer::<i32>::zeroed(1)?;

    // Launch C++ kernel cooperatively FIRST to test if it works with launch_cooperative! macro
    println!(
        "\nLaunching C++ kernel with {} blocks...",
        cpp_actual_blocks
    );
    let cpp_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            cpp_kernel<<<(cpp_actual_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                cpp_data.as_device_ptr(),
                array_size as i32,
                cpp_global_sum.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    let cpp_elapsed = cpp_start.elapsed();
    println!("C++ kernel completed in {:?}", cpp_elapsed);

    // Launch Rust kernel cooperatively SECOND
    println!(
        "\nLaunching Rust kernel with {} blocks...",
        rust_actual_blocks
    );
    let rust_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            rust_kernel<<<(rust_actual_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_data.as_device_ptr(),
                array_size as i32,
                rust_global_sum.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    let rust_elapsed = rust_start.elapsed();
    println!("Rust kernel completed in {:?}", rust_elapsed);

    // Copy results back to host
    let mut rust_data_host = vec![0i32; array_size];
    let mut rust_sum_host = vec![0i32; 1];
    rust_data.copy_to(&mut rust_data_host)?;
    rust_global_sum.copy_to(&mut rust_sum_host)?;

    let mut cpp_data_host = vec![0i32; array_size];
    let mut cpp_sum_host = vec![0i32; 1];
    cpp_data.copy_to(&mut cpp_data_host)?;
    cpp_global_sum.copy_to(&mut cpp_sum_host)?;

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
    println!("C++ global sum:      {}", cpp_sum_host[0]);
    println!("Expected sum:        {}", expected_sum);

    // Validate against expected values
    assert_eq!(
        rust_sum_host[0], expected_sum,
        "Rust kernel did not produce expected sum"
    );
    assert_eq!(
        cpp_sum_host[0], expected_sum,
        "C++ kernel did not produce expected sum"
    );

    // Critical comparison: Rust vs C++ data arrays
    compare_outputs(&rust_data_host, &cpp_data_host, "grid_sync_multiblock_data")?;

    // Critical comparison: Rust vs C++ global sums
    compare_outputs(&rust_sum_host, &cpp_sum_host, "grid_sync_multiblock_sum")?;

    println!("✓ Test PASSED: Rust and C++ grid_sync_multiblock kernels produce identical results");

    Ok(())
}
