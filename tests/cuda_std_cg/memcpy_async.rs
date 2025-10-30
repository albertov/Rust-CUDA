//! Tests for asynchronous memory copy operations (SM 8.0+).

mod common;
use common::*;

/// Helper to check if current device supports SM 8.0+
fn check_sm80_support() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get_device(0)?;
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    if major < 8 {
        return Err(format!(
            "memcpy_async requires SM 8.0+ (Ampere or newer), found SM {}.x",
            major
        )
        .into());
    }
    Ok(())
}

#[test]
fn test_memcpy_async_basic() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = check_sm80_support() {
        println!("Skipping test: {}", e);
        return Ok(());
    }

    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_memcpy_async_basic")?;

    const SIZE: usize = 256;

    // Prepare host data
    let h_src: Vec<i32> = (0..SIZE).map(|i| i as i32).collect();

    // Allocate device memory
    let d_src = DeviceBuffer::from_slice(&h_src)?;
    let d_dst = DeviceBuffer::from_slice(&vec![0i32; SIZE])?;

    // Launch kernel: 1 block, 256 threads
    unsafe {
        launch!(
            kernel<<<1, SIZE as u32, 0, stream>>>(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr(),
                SIZE as u32
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut result = vec![0i32; SIZE];
    d_dst.copy_to(&mut result)?;

    // Verify results
    for i in 0..SIZE {
        assert_eq!(
            result[i], h_src[i],
            "Mismatch at index {}: expected {}, got {}",
            i, h_src[i], result[i]
        );
    }

    println!("✓ memcpy_async_basic test passed");
    Ok(())
}

#[test]
fn test_pipeline_pattern() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = check_sm80_support() {
        println!("Skipping test: {}", e);
        return Ok(());
    }

    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_pipeline_pattern")?;

    const STAGES: u32 = 4;
    const STAGE_SIZE: usize = 128;
    const TOTAL_SIZE: usize = (STAGES as usize) * STAGE_SIZE;

    // Prepare host data
    let h_src: Vec<i32> = (0..TOTAL_SIZE).map(|i| (i + 1000) as i32).collect();

    let d_src = DeviceBuffer::from_slice(&h_src)?;
    let d_dst = DeviceBuffer::from_slice(&vec![0i32; TOTAL_SIZE])?;

    unsafe {
        launch!(
            kernel<<<1, STAGE_SIZE as u32, 0, stream>>>(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr(),
                STAGES
            )
        )?;
    }

    stream.synchronize()?;

    let mut result = vec![0i32; TOTAL_SIZE];
    d_dst.copy_to(&mut result)?;

    // Verify results
    for i in 0..TOTAL_SIZE {
        assert_eq!(
            result[i], h_src[i],
            "Pipeline mismatch at index {}: expected {}, got {}",
            i, h_src[i], result[i]
        );
    }

    println!("✓ pipeline_pattern test passed");
    Ok(())
}

#[test]
fn test_commit_group_batching() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = check_sm80_support() {
        println!("Skipping test: {}", e);
        return Ok(());
    }

    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_commit_group_batching")?;

    const BATCH_COUNT: u32 = 4;
    const BATCH_SIZE: usize = 64;
    const TOTAL_SIZE: usize = (BATCH_COUNT as usize) * BATCH_SIZE;

    let h_src: Vec<i32> = (0..TOTAL_SIZE).map(|i| (i * 2) as i32).collect();

    let d_src = DeviceBuffer::from_slice(&h_src)?;
    let d_dst = DeviceBuffer::from_slice(&vec![0i32; TOTAL_SIZE])?;

    unsafe {
        launch!(
            kernel<<<1, BATCH_SIZE as u32, 0, stream>>>(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr(),
                BATCH_COUNT
            )
        )?;
    }

    stream.synchronize()?;

    let mut result = vec![0i32; TOTAL_SIZE];
    d_dst.copy_to(&mut result)?;

    for i in 0..TOTAL_SIZE {
        assert_eq!(
            result[i], h_src[i],
            "Batching mismatch at index {}: expected {}, got {}",
            i, h_src[i], result[i]
        );
    }

    println!("✓ commit_group_batching test passed");
    Ok(())
}

#[test]
fn test_memcpy_async_large() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = check_sm80_support() {
        println!("Skipping test: {}", e);
        return Ok(());
    }

    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_memcpy_async_large")?;

    const SIZE: usize = 1024;

    let h_src: Vec<i32> = (0..SIZE).map(|i| (i * 3) as i32).collect();

    let d_src = DeviceBuffer::from_slice(&h_src)?;
    let d_dst = DeviceBuffer::from_slice(&vec![0i32; SIZE])?;

    unsafe {
        launch!(
            kernel<<<1, SIZE as u32, 0, stream>>>(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut result = vec![0i32; SIZE];
    d_dst.copy_to(&mut result)?;

    for i in 0..SIZE {
        assert_eq!(
            result[i], h_src[i],
            "Large copy mismatch at index {}: expected {}, got {}",
            i, h_src[i], result[i]
        );
    }

    println!("✓ memcpy_async_large test passed");
    Ok(())
}
