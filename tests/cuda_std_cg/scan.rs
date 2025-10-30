mod common;
use common::*;

#[test]
fn test_inclusive_scan_add_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_inclusive_scan_add_tile32")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("inclusive_scan_add(tile32) with all 1s:");
    for (i, &value) in host_output.iter().enumerate() {
        let expected = (i + 1) as i32; // Thread i should have sum 1+1+...+1 = i+1
        println!("  Thread {}: {} (expected {})", i, value, expected);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ inclusive_scan_add tile32 test passed");
    Ok(())
}

#[test]
fn test_exclusive_scan_add_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_exclusive_scan_add_tile32")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("exclusive_scan_add(tile32) with all 1s:");
    for (i, &value) in host_output.iter().enumerate() {
        let expected = i as i32; // Thread i should have sum of previous threads = i
        println!("  Thread {}: {} (expected {})", i, value, expected);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ exclusive_scan_add tile32 test passed");
    Ok(())
}

#[test]
fn test_inclusive_scan_sequential() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_inclusive_scan_sequential")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("inclusive_scan_add(tile32) with sequential values:");
    for (i, &value) in host_output.iter().enumerate() {
        // Sum of 1+2+3+...+(i+1) = (i+1)*(i+2)/2
        let expected = ((i + 1) * (i + 2) / 2) as i32;
        println!("  Thread {}: {} (expected {})", i, value, expected);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ inclusive_scan_sequential test passed");
    Ok(())
}

#[test]
fn test_exclusive_scan_sequential() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_exclusive_scan_sequential")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("exclusive_scan_add(tile32) with sequential values:");
    for (i, &value) in host_output.iter().enumerate() {
        // Sum of 1+2+3+...+i = i*(i+1)/2
        let expected = (i * (i + 1) / 2) as i32;
        println!("  Thread {}: {} (expected {})", i, value, expected);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ exclusive_scan_sequential test passed");
    Ok(())
}

#[test]
#[ignore = "scan_min/max not yet implemented"]
fn test_scan_min_max_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_scan_min_max_tile32")?;

    let min_output = DeviceBuffer::from_slice(&vec![0i32; 32])?;
    let max_output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            min_output.as_device_ptr(),
            max_output.as_device_ptr()
        ))?;
    }

    stream.synchronize()?;

    let mut host_min = vec![0i32; 32];
    let mut host_max = vec![0i32; 32];
    min_output.copy_to(&mut host_min)?;
    max_output.copy_to(&mut host_max)?;

    println!("inclusive_scan_min/max(tile32) results:");
    for i in 0..32 {
        // Running min: should always be 0 (minimum so far)
        // Running max: should be i (maximum so far)
        println!(
            "  Thread {}: min={} (expected 0), max={} (expected {})",
            i, host_min[i], host_max[i], i
        );
        assert_eq!(
            host_min[i], 0,
            "Thread {} has incorrect min: {}",
            i, host_min[i]
        );
        assert_eq!(
            host_max[i], i as i32,
            "Thread {} has incorrect max: {}",
            i, host_max[i]
        );
    }

    println!("✓ scan_min/max tile32 test passed");
    Ok(())
}

#[test]
fn test_exclusive_scan_allocation() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_exclusive_scan_allocation")?;

    // Create varying sizes for allocation pattern
    let sizes: Vec<i32> = (0..32).map(|i| (i % 5) + 1).collect(); // Sizes: 1,2,3,4,5,1,2,3,4,5,...
    let sizes_device = DeviceBuffer::from_slice(&sizes)?;
    let offsets_device = DeviceBuffer::from_slice(&vec![0i32; 32])?;
    let total_device = DeviceBuffer::from_slice(&vec![0i32; 1])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            sizes_device.as_device_ptr(),
            offsets_device.as_device_ptr(),
            total_device.as_device_ptr()
        ))?;
    }

    stream.synchronize()?;

    let mut host_offsets = vec![0i32; 32];
    let mut host_total = vec![0i32; 1];
    offsets_device.copy_to(&mut host_offsets)?;
    total_device.copy_to(&mut host_total)?;

    // Verify offsets are correct (cumulative sum of sizes)
    let mut expected_offset = 0;
    println!("exclusive_scan_allocation results:");
    for (i, &offset) in host_offsets.iter().enumerate() {
        println!(
            "  Thread {}: size={}, offset={} (expected {})",
            i, sizes[i], offset, expected_offset
        );
        assert_eq!(
            offset, expected_offset,
            "Thread {} has incorrect offset: got {}, expected {}",
            i, offset, expected_offset
        );
        expected_offset += sizes[i];
    }

    println!(
        "  Total allocation: {} (expected {})",
        host_total[0], expected_offset
    );
    assert_eq!(
        host_total[0], expected_offset,
        "Total allocation incorrect: got {}, expected {}",
        host_total[0], expected_offset
    );

    println!("✓ exclusive_scan_allocation test passed");
    Ok(())
}

#[test]
fn test_inclusive_scan_add_tile16() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_inclusive_scan_add_tile16")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("inclusive_scan_add(tile16) results:");
    for i in 0..32 {
        // Each tile of 16: threads 0-15 get 1-16, threads 16-31 get 1-16
        let tile_rank = i % 16;
        let expected = (tile_rank + 1) as i32;
        println!("  Thread {}: {} (expected {})", i, host_output[i], expected);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ inclusive_scan_add tile16 test passed");
    Ok(())
}

#[test]
fn test_exclusive_scan_add_tile16() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_exclusive_scan_add_tile16")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("exclusive_scan_add(tile16) results:");
    for i in 0..32 {
        let tile_rank = i % 16;
        let expected = tile_rank as i32;
        println!("  Thread {}: {} (expected {})", i, host_output[i], expected);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ exclusive_scan_add tile16 test passed");
    Ok(())
}

#[test]
fn test_inclusive_scan_add_tile8() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_inclusive_scan_add_tile8")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("inclusive_scan_add(tile8) results:");
    for i in 0..32 {
        let tile_rank = i % 8;
        let expected = (tile_rank + 1) as i32;
        println!("  Thread {}: {} (expected {})", i, host_output[i], expected);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ inclusive_scan_add tile8 test passed");
    Ok(())
}

#[test]
#[ignore = "DynamicTiledGroup scan not yet implemented"]
fn test_scan_dynamic_tile() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_scan_dynamic_tile")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 64])?; // 32 for inclusive, 32 for exclusive
    let tile_size = 8u32;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            output.as_device_ptr(),
            tile_size
        ))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 64];
    output.copy_to(&mut host_output)?;

    println!("scan_dynamic_tile (size={}) results:", tile_size);
    for i in 0..32 {
        let tile_rank = i % tile_size as usize;
        let expected_inclusive = (tile_rank + 1) as i32;
        let expected_exclusive = tile_rank as i32;

        let inclusive = host_output[i];
        let exclusive = host_output[i + 32];

        println!(
            "  Thread {}: inclusive={} (expected {}), exclusive={} (expected {})",
            i, inclusive, expected_inclusive, exclusive, expected_exclusive
        );

        assert_eq!(
            inclusive, expected_inclusive,
            "Thread {} has incorrect inclusive scan",
            i
        );
        assert_eq!(
            exclusive, expected_exclusive,
            "Thread {} has incorrect exclusive scan",
            i
        );
    }

    println!("✓ scan_dynamic_tile test passed");
    Ok(())
}

#[test]
fn test_scan_unsigned() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_scan_unsigned")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    println!("scan_unsigned results:");
    for (i, &value) in host_output.iter().enumerate() {
        let expected = ((i + 1) * (i + 2) / 2) as i32;
        println!("  Thread {}: {} (expected {})", i, value, expected);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect scan: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ scan_unsigned test passed");
    Ok(())
}

#[test]
fn test_scan_pattern() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_scan_pattern")?;

    // Create an interesting pattern: alternating 1 and -1
    let input: Vec<i32> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let input_device = DeviceBuffer::from_slice(&input)?;
    let output_device = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            input_device.as_device_ptr(),
            output_device.as_device_ptr()
        ))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output_device.copy_to(&mut host_output)?;

    // Compute expected values
    let mut expected = vec![0i32; 32];
    let mut sum = 0;
    for i in 0..32 {
        sum += input[i];
        expected[i] = sum;
    }

    println!("scan_pattern (alternating 1, -1) results:");
    for i in 0..32 {
        println!(
            "  Thread {}: input={}, scan={} (expected {})",
            i, input[i], host_output[i], expected[i]
        );
        assert_eq!(
            host_output[i], expected[i],
            "Thread {} has incorrect scan: got {}, expected {}",
            i, host_output[i], expected[i]
        );
    }

    println!("✓ scan_pattern test passed");
    Ok(())
}
