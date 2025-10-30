//! Comparative validation test: grid_sync_basic
//!
//! This test validates that the Rust implementation of grid_sync_basic produces
//! identical results to the C++ reference implementation when given identical inputs.
//!
//! Test Requirement: TR4 - Basic grid sync comparative validation
//!
//! Strategy:
//! 1. Load both Rust and C++ modules
//! 2. Launch both kernels with identical parameters
//! 3. Use separate output buffers to avoid cross-contamination
//! 4. Assert that outputs match exactly

mod common;
use common::*;

#[test]
fn test_grid_sync_basic_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Verify device supports cooperative launch
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Load both Rust and C++ modules
    let rust_module = load_rust_module()?;
    let cpp_module = load_cpp_module("basic")?;

    // Get kernel functions from both modules
    let rust_kernel = rust_module.get_function("grid_sync_basic_kernel")?;
    let cpp_kernel = cpp_module.get_function("grid_sync_basic_kernel")?;

    // Define identical launch parameters
    let num_blocks = 8u32;
    let threads_per_block = 256u32;

    // Allocate separate output buffers for each kernel
    let rust_output = DeviceBuffer::from_slice(&vec![0u32; num_blocks as usize])?;
    let cpp_output = DeviceBuffer::from_slice(&vec![0u32; num_blocks as usize])?;

    // Launch C++ kernel first
    println!("\nLaunching C++ kernel with {} blocks...", num_blocks);
    let cpp_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            cpp_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                cpp_output.as_device_ptr(),
                num_blocks
            )
        )?;
    }
    stream.synchronize()?;
    let cpp_elapsed = cpp_start.elapsed();
    println!("C++ kernel completed in {:?}", cpp_elapsed);

    // Launch Rust kernel
    println!("\nLaunching Rust kernel with {} blocks...", num_blocks);
    let rust_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            rust_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_output.as_device_ptr(),
                num_blocks
            )
        )?;
    }
    stream.synchronize()?;
    let rust_elapsed = rust_start.elapsed();
    println!("Rust kernel completed in {:?}", rust_elapsed);

    // Copy results to host
    let rust_result = rust_output.as_host_vec()?;
    let cpp_result = cpp_output.as_host_vec()?;

    // Compare outputs using common infrastructure helper
    compare_outputs(&rust_result, &cpp_result, "grid_sync_basic")?;

    Ok(())
}
