//! Device capabilities diagnostic for cooperative launch
//!
//! This diagnostic queries CUDA device attributes to determine:
//! 1. Whether device supports cooperative launch
//! 2. Maximum number of cooperative blocks that can be scheduled
//! 3. SM count and occupancy limits
//! 4. Resource constraints (registers, shared memory)

mod common;
use common::*;

#[test]
fn test_device_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;
    let device = Device::get_device(0)?;

    println!("=== CUDA Device Capabilities Diagnostic ===\n");

    // Basic device info
    let name = device.name()?;
    println!("Device: {}", name);

    // Compute capability
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    println!("Compute Capability: {}.{}", major, minor);

    // SM count
    let num_sms = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
    println!("Number of SMs: {}", num_sms);

    // Max threads per SM
    let max_threads_per_sm = device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)?;
    println!("Max threads per SM: {}", max_threads_per_sm);

    // Max threads per block
    let max_threads_per_block = device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)?;
    println!("Max threads per block: {}", max_threads_per_block);

    // Max block dimensions
    let max_block_dim_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)?;
    let max_block_dim_y = device.get_attribute(DeviceAttribute::MaxBlockDimY)?;
    let max_block_dim_z = device.get_attribute(DeviceAttribute::MaxBlockDimZ)?;
    println!(
        "Max block dimensions: ({}, {}, {})",
        max_block_dim_x, max_block_dim_y, max_block_dim_z
    );

    // Max grid dimensions
    let max_grid_dim_x = device.get_attribute(DeviceAttribute::MaxGridDimX)?;
    let max_grid_dim_y = device.get_attribute(DeviceAttribute::MaxGridDimY)?;
    let max_grid_dim_z = device.get_attribute(DeviceAttribute::MaxGridDimZ)?;
    println!(
        "Max grid dimensions: ({}, {}, {})",
        max_grid_dim_x, max_grid_dim_y, max_grid_dim_z
    );

    // Total constant memory
    let total_const_mem = device.get_attribute(DeviceAttribute::TotalConstantMemory)?;
    println!("Total constant memory: {} bytes", total_const_mem);

    // Shared memory per block
    let shared_mem_per_block = device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)?;
    println!("Shared memory per block: {} bytes", shared_mem_per_block);

    // Registers per block
    let regs_per_block = device.get_attribute(DeviceAttribute::MaxRegistersPerBlock)?;
    println!("Registers per block: {}", regs_per_block);

    // Warp size
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)?;
    println!("Warp size: {}", warp_size);

    println!("\n=== Cooperative Launch Support ===\n");

    // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH (95)
    let coop_launch_attr: DeviceAttribute =
        unsafe { std::mem::transmute::<u32, DeviceAttribute>(95) };
    let coop_launch = device.get_attribute(coop_launch_attr)?;
    println!("CooperativeLaunch (attr 95): {}", coop_launch);

    // CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH (96)
    let coop_multi_device_attr: DeviceAttribute =
        unsafe { std::mem::transmute::<u32, DeviceAttribute>(96) };
    let coop_multi_device = device.get_attribute(coop_multi_device_attr)?;
    println!(
        "CooperativeMultiDeviceLaunch (attr 96): {}",
        coop_multi_device
    );

    println!("\n=== Test Configuration Analysis ===\n");

    // Test parameters from comparison_multiblock.rs
    let num_blocks = 8u32;
    let threads_per_block = 256u32;
    let array_size = (num_blocks * threads_per_block) as usize;

    println!("Test configuration:");
    println!("  Blocks: {}", num_blocks);
    println!("  Threads per block: {}", threads_per_block);
    println!("  Total threads: {}", array_size);

    // Calculate theoretical limits
    let max_blocks_per_sm = max_threads_per_sm / threads_per_block as i32;
    let total_theoretical_blocks = num_sms * max_blocks_per_sm;

    println!("\nTheoretical limits:");
    println!(
        "  Max blocks per SM (based on thread limit): {}",
        max_blocks_per_sm
    );
    println!(
        "  Total blocks device can handle: {}",
        total_theoretical_blocks
    );
    println!(
        "  Test needs {} blocks: {}",
        num_blocks,
        if num_blocks as i32 <= total_theoretical_blocks {
            "WITHIN LIMITS"
        } else {
            "EXCEEDS LIMITS"
        }
    );

    // Calculate occupancy for the test configuration
    let warps_per_block = threads_per_block / warp_size as u32;
    println!("\nOccupancy analysis:");
    println!("  Warps per block: {}", warps_per_block);
    println!(
        "  Active blocks for {} concurrent blocks: {}",
        num_blocks, num_blocks
    );
    println!("  Active warps: {}", num_blocks * warps_per_block);

    // Check if cooperative launch can schedule all 8 blocks concurrently
    if coop_launch == 0 {
        println!("\n⚠️  WARNING: Device reports NO cooperative launch support!");
        println!("    CooperativeLaunch attribute = 0");
        println!("    Cooperative kernels will HANG or fail!");
    } else {
        println!("\n✓ Device supports cooperative launch");

        // For cooperative kernels, ALL blocks must fit concurrently on device
        if num_blocks as i32 > total_theoretical_blocks {
            println!("\n⚠️  CRITICAL: Test requests {} blocks but device can only handle {} concurrently!",
                     num_blocks, total_theoretical_blocks);
            println!("    This will cause cooperative launch to FAIL!");
            println!("    Cooperative kernels require ALL blocks to run simultaneously.");
        } else {
            println!("\n✓ Test configuration fits within device limits");

            // Additional check: Do we have enough SMs to run all blocks?
            let min_sms_needed = (num_blocks as f32 / max_blocks_per_sm as f32).ceil() as i32;
            println!(
                "\nMinimum SMs needed: {} (available: {})",
                min_sms_needed, num_sms
            );

            if min_sms_needed > num_sms {
                println!("⚠️  WARNING: Need more SMs than available!");
                println!("    This may cause scheduling issues!");
            } else {
                println!("✓ Sufficient SMs available");
            }
        }
    }

    println!("\n=== Diagnostic Complete ===\n");

    Ok(())
}
