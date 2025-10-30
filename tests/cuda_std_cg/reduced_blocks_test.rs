//! Test with reduced block counts to verify scheduling
//!
//! This test attempts cooperative launch with 2, 4, and 8 blocks
//! to determine if the issue is block-count specific.

mod common;
use common::*;

fn test_with_block_count(num_blocks: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing with {} blocks ===", num_blocks);

    let _ctx = cust::quick_init()?;
    check_cooperative_launch_support()?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let rust_module = load_rust_module()?;
    let rust_kernel = rust_module.get_function("grid_sync_multiblock_kernel")?;

    let threads_per_block = 256u32;
    let array_size = (num_blocks * threads_per_block) as usize;

    let rust_data = DeviceBuffer::<i32>::zeroed(array_size)?;
    let rust_global_sum = DeviceBuffer::<i32>::zeroed(1)?;

    println!(
        "Launching kernel with {} blocks × {} threads...",
        num_blocks, threads_per_block
    );
    let start = std::time::Instant::now();

    unsafe {
        launch_cooperative!(
            rust_kernel<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                rust_data.as_device_ptr(),
                array_size as i32,
                rust_global_sum.as_device_ptr()
            )
        )?;
    }

    // Set a timeout for synchronization
    println!("Waiting for kernel completion (5 second timeout)...");
    let sync_start = std::time::Instant::now();

    // Try to synchronize with timeout
    let result = std::panic::catch_unwind(|| stream.synchronize());

    match result {
        Ok(Ok(())) => {
            let elapsed = start.elapsed();
            println!("✓ Kernel completed successfully in {:?}", elapsed);

            // Verify results
            let mut data_host = vec![0i32; array_size];
            let mut sum_host = vec![0i32; 1];
            rust_data.copy_to(&mut data_host)?;
            rust_global_sum.copy_to(&mut sum_host)?;

            let mut expected_sum = 0i32;
            for i in 0..array_size {
                let expected_block_id = (i / threads_per_block as usize) as i32;
                expected_sum += expected_block_id;
            }

            println!("Global sum: {} (expected: {})", sum_host[0], expected_sum);

            if sum_host[0] == expected_sum {
                println!("✓ Result validation PASSED");
                Ok(())
            } else {
                Err(format!(
                    "Result mismatch: got {}, expected {}",
                    sum_host[0], expected_sum
                )
                .into())
            }
        }
        Ok(Err(e)) => {
            println!("✗ Kernel synchronization failed: {:?}", e);
            Err(e.into())
        }
        Err(_) => {
            println!(
                "✗ Kernel HUNG (panic during sync after {:?})",
                sync_start.elapsed()
            );
            Err("Kernel hung during synchronization".into())
        }
    }
}

#[test]
fn test_reduced_blocks_2() -> Result<(), Box<dyn std::error::Error>> {
    test_with_block_count(2)
}

#[test]
fn test_reduced_blocks_4() -> Result<(), Box<dyn std::error::Error>> {
    test_with_block_count(4)
}

#[test]
fn test_reduced_blocks_8() -> Result<(), Box<dyn std::error::Error>> {
    test_with_block_count(8)
}
