mod common;
use common::*;

#[test]
fn test_reduce_add_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_add_tile32")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    // Expected sum: 0+1+2+...+31 = 31*32/2 = 496
    let expected = 496;

    println!("reduce_add(tile32) results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: {}", i, value);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect sum: got {}, expected {}",
            i, value, expected
        );
    }

    println!("✓ reduce_add tile32 test passed");
    Ok(())
}

#[test]
fn test_reduce_min_max_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_min_max_tile32")?;

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

    println!("reduce_min/max(tile32) results:");
    for i in 0..32 {
        println!("  Thread {}: min={}, max={}", i, host_min[i], host_max[i]);
        assert_eq!(
            host_min[i], 0,
            "Thread {} has incorrect min: {}",
            i, host_min[i]
        );
        assert_eq!(
            host_max[i], 31,
            "Thread {} has incorrect max: {}",
            i, host_max[i]
        );
    }

    println!("✓ reduce_min/max tile32 test passed");
    Ok(())
}

#[test]
fn test_reduce_bitwise_tile32() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_bitwise_tile32")?;

    let and_output = DeviceBuffer::from_slice(&vec![0u32; 32])?;
    let or_output = DeviceBuffer::from_slice(&vec![0u32; 32])?;
    let xor_output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            and_output.as_device_ptr(),
            or_output.as_device_ptr(),
            xor_output.as_device_ptr()
        ))?;
    }

    stream.synchronize()?;

    let mut host_and = vec![0u32; 32];
    let mut host_or = vec![0u32; 32];
    let mut host_xor = vec![0u32; 32];
    and_output.copy_to(&mut host_and)?;
    or_output.copy_to(&mut host_or)?;
    xor_output.copy_to(&mut host_xor)?;

    // Even threads: 0xFFFFFFFF, Odd threads: 0x00000000
    // AND: all must be set = 0x00000000
    // OR: any can be set = 0xFFFFFFFF
    // XOR: even number of set values (16) = 0x00000000

    println!("reduce_bitwise(tile32) results:");
    for i in 0..32 {
        println!(
            "  Thread {}: and=0x{:08X}, or=0x{:08X}, xor=0x{:08X}",
            i, host_and[i], host_or[i], host_xor[i]
        );

        assert_eq!(
            host_and[i], 0x00000000,
            "Thread {} has incorrect AND: 0x{:08X}",
            i, host_and[i]
        );
        assert_eq!(
            host_or[i], 0xFFFFFFFF,
            "Thread {} has incorrect OR: 0x{:08X}",
            i, host_or[i]
        );
        assert_eq!(
            host_xor[i], 0x00000000,
            "Thread {} has incorrect XOR: 0x{:08X}",
            i, host_xor[i]
        );
    }

    println!("✓ reduce_bitwise tile32 test passed");
    Ok(())
}

#[test]
fn test_reduce_add_tile16() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_add_tile16")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    // Expected sum for tile of 16: 0+1+2+...+15 = 15*16/2 = 120
    let expected = 120;

    println!("reduce_add(tile16) results:");
    for i in 0..32 {
        println!("  Thread {}: {}", i, host_output[i]);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect sum: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ reduce_add tile16 test passed");
    Ok(())
}

#[test]
fn test_reduce_add_tile8() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_add_tile8")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    // Expected sum for tile of 8: 0+1+2+...+7 = 7*8/2 = 28
    let expected = 28;

    println!("reduce_add(tile8) results:");
    for i in 0..32 {
        println!("  Thread {}: {}", i, host_output[i]);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect sum: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ reduce_add tile8 test passed");
    Ok(())
}

#[test]
fn test_reduce_min_unsigned() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_min_unsigned")?;

    let output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0u32; 32];
    output.copy_to(&mut host_output)?;

    // Expected min: 0 * 10 = 0
    println!("reduce_min(unsigned) results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: {}", i, value);
        assert_eq!(value, 0, "Thread {} has incorrect min: {}", i, value);
    }

    println!("✓ reduce_min unsigned test passed");
    Ok(())
}

#[test]
fn test_reduce_max_unsigned() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_max_unsigned")?;

    let output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0u32; 32];
    output.copy_to(&mut host_output)?;

    // Expected max: 31 * 10 = 310
    let expected = 310;
    println!("reduce_max(unsigned) results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: {}", i, value);
        assert_eq!(value, expected, "Thread {} has incorrect max: {}", i, value);
    }

    println!("✓ reduce_max unsigned test passed");
    Ok(())
}

#[test]
fn test_reduce_and_pattern() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_and_pattern")?;

    let output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0u32; 32];
    output.copy_to(&mut host_output)?;

    // All threads have bit 0 set, only even have bit 1 set
    // AND result should be 0b01 (only bit 0 common to all)
    let expected = 0b01u32;

    println!("reduce_and pattern results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: 0x{:08X}", i, value);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect AND: 0x{:08X}",
            i, value
        );
    }

    println!("✓ reduce_and pattern test passed");
    Ok(())
}

#[test]
fn test_reduce_or_pattern() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_or_pattern")?;

    let output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0u32; 32];
    output.copy_to(&mut host_output)?;

    // Each thread sets one bit, OR should have all 32 bits set
    let expected = 0xFFFFFFFFu32;

    println!("reduce_or pattern results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: 0x{:08X}", i, value);
        assert_eq!(
            value, expected,
            "Thread {} has incorrect OR: 0x{:08X}",
            i, value
        );
    }

    println!("✓ reduce_or pattern test passed");
    Ok(())
}

#[test]
fn test_reduce_xor_parity() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_xor_parity")?;

    let output = DeviceBuffer::from_slice(&vec![0u32; 32])?;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0u32; 32];
    output.copy_to(&mut host_output)?;

    // Threads 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 have value 1
    // That's 11 threads, odd count, so XOR = 1
    let expected = 1u32;

    println!("reduce_xor parity results:");
    for (i, &value) in host_output.iter().enumerate() {
        println!("  Thread {}: {}", i, value);
        assert_eq!(value, expected, "Thread {} has incorrect XOR: {}", i, value);
    }

    println!("✓ reduce_xor parity test passed");
    Ok(())
}

#[test]
fn test_reduce_add_dynamic_tile8() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_reduce_add_dynamic_tile")?;

    let output = DeviceBuffer::from_slice(&vec![0i32; 32])?;
    let tile_size = 8u32;

    unsafe {
        launch!(kernel<<<1, 32, 0, stream>>>(
            output.as_device_ptr(),
            tile_size
        ))?;
    }

    stream.synchronize()?;

    let mut host_output = vec![0i32; 32];
    output.copy_to(&mut host_output)?;

    // Expected sum for tile of 8: 0+1+2+...+7 = 28
    let expected = 28;

    println!("reduce_add(dynamic tile8) results:");
    for i in 0..32 {
        println!("  Thread {}: {}", i, host_output[i]);
        assert_eq!(
            host_output[i], expected,
            "Thread {} has incorrect sum: got {}, expected {}",
            i, host_output[i], expected
        );
    }

    println!("✓ reduce_add dynamic tile8 test passed");
    Ok(())
}
