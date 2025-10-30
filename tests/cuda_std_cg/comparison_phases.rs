mod common;
use common::*;

/// Comparative validation test: grid_sync_phases_kernel
///
/// Tests multi-phase grid synchronization by launching both Rust and C++
/// implementations with identical inputs and asserting outputs match.
///
/// Each kernel performs 5 phases of:
/// 1. Each block atomically increments phase counter
/// 2. Grid-wide sync
/// 3. Block 0 verifies counter == num_blocks
/// 4. Grid-wide sync before next phase
///
/// This validates:
/// - Multiple grid.sync() calls work correctly in sequence
/// - Each phase completes fully before the next begins
/// - Memory visibility across multiple synchronization points
/// - Rust implementation matches NVIDIA C++ reference behavior
#[test]
fn test_grid_sync_phases_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Verify device supports cooperative launch
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Load both Rust and C++ modules
    let rust_module = load_rust_module()?;
    let cpp_module = load_cpp_module("phases")?;

    let rust_kernel = rust_module.get_function("grid_sync_phases_kernel")?;
    let cpp_kernel = cpp_module.get_function("grid_sync_phases_kernel")?;

    // Test configuration - must be identical for both kernels
    let num_blocks = 8u32;
    let threads_per_block = 256u32;
    let num_phases = 5i32;

    // Allocate separate output buffers for Rust and C++
    let rust_counters = vec![0i32; num_phases as usize];
    let cpp_counters = vec![0i32; num_phases as usize];

    let rust_output = DeviceBuffer::from_slice(&rust_counters)?;
    let cpp_output = DeviceBuffer::from_slice(&cpp_counters)?;

    // Launch C++ kernel first
    println!(
        "\nLaunching C++ kernel with {} blocks, {} phases...",
        num_blocks, num_phases
    );
    let cpp_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            cpp_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                cpp_output.as_device_ptr(),
                num_phases
            )
        )?;
    }
    stream.synchronize()?;
    let cpp_elapsed = cpp_start.elapsed();
    println!("C++ kernel completed in {:?}", cpp_elapsed);

    // Launch Rust kernel with cooperative launch
    println!(
        "\nLaunching Rust kernel with {} blocks, {} phases...",
        num_blocks, num_phases
    );
    let rust_start = std::time::Instant::now();
    unsafe {
        launch_cooperative!(
            rust_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_output.as_device_ptr(),
                num_phases
            )
        )?;
    }
    stream.synchronize()?;
    let rust_elapsed = rust_start.elapsed();
    println!("Rust kernel completed in {:?}", rust_elapsed);

    // Copy results to host
    let mut rust_result = vec![0i32; num_phases as usize];
    let mut cpp_result = vec![0i32; num_phases as usize];

    rust_output.copy_to(&mut rust_result)?;
    cpp_output.copy_to(&mut cpp_result)?;

    // Compare outputs - this is the critical validation
    compare_outputs(&rust_result, &cpp_result, "grid_sync_phases")?;

    // Additional verification: each phase counter should equal num_blocks
    for (phase, &count) in rust_result.iter().enumerate() {
        assert_eq!(
            count, num_blocks as i32,
            "Phase {} counter incorrect: {} (expected {})",
            phase, count, num_blocks
        );
    }

    Ok(())
}
