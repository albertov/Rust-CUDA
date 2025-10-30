mod common;
use common::*;

/// Test basic TiledGroup<32> creation and synchronization
///
/// Verifies:
/// - tiled_partition::<32>() creates valid tile handle
/// - tile.size() returns 32
/// - tile.thread_rank() returns values in range [0, 32)
/// - tile.sync() works without deadlock
/// - Multiple warps in a block each get correct tile
#[test]
fn test_tile_creation_and_sync() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_creation_and_sync")?;

    // Test with a single block of 64 threads (2 warps)
    let num_threads = 64u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;

    // Verify results
    println!("Tile ranks for {} threads:", num_threads);
    for (thread_idx, (&rank, &size)) in ranks_host.iter().zip(sizes_host.iter()).enumerate() {
        println!("  Thread {}: rank {}, size {}", thread_idx, rank, size);

        // Each tile should have size 32
        assert_eq!(size, 32, "Thread {} has incorrect tile size", thread_idx);

        // Each thread's rank should be its lane ID (thread_idx % 32)
        let expected_rank = (thread_idx % 32) as u32;
        assert_eq!(
            rank, expected_rank,
            "Thread {} has rank {}, expected {}",
            thread_idx, rank, expected_rank
        );
    }

    // First warp (threads 0-31) should have ranks 0-31
    for i in 0..32 {
        assert_eq!(ranks_host[i], i as u32, "First warp rank mismatch");
    }

    // Second warp (threads 32-63) should have ranks 0-31
    for i in 32..64 {
        let expected_rank = (i - 32) as u32;
        assert_eq!(
            ranks_host[i], expected_rank,
            "Second warp rank mismatch at thread {}",
            i
        );
    }

    println!("✓ Tile creation and sync test passed");
    Ok(())
}

/// Test multiple tiles per block
///
/// Verifies that multiple warps in a block each get correct tile handles
/// and can synchronize independently.
#[test]
fn test_multiple_tiles_per_block() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_multiple_tiles_per_block")?;

    // Test with a block of 128 threads (4 warps)
    let num_threads = 128u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;

    // Verify results
    println!("Multiple tiles test with {} threads:", num_threads);
    for (thread_idx, (&rank, &size)) in ranks_host.iter().zip(sizes_host.iter()).enumerate() {
        // Each tile should have size 32
        assert_eq!(
            size, 32,
            "Thread {} has tile size {}, expected 32",
            thread_idx, size
        );

        // Rank should be lane ID within warp
        let expected_rank = (thread_idx % 32) as u32;
        assert_eq!(
            rank, expected_rank,
            "Thread {} has rank {}, expected {}",
            thread_idx, rank, expected_rank
        );
    }

    // Verify each warp has ranks 0-31
    for warp in 0..4 {
        let base = warp * 32;
        for lane in 0..32 {
            let idx = base + lane;
            assert_eq!(
                ranks_host[idx], lane as u32,
                "Warp {} lane {} has incorrect rank",
                warp, lane
            );
        }
        println!("  Warp {}: ranks 0-31 ✓", warp);
    }

    println!("✓ Multiple tiles per block test passed");
    Ok(())
}

/// Test tile synchronization with computation
///
/// Verifies that tile sync correctly orders computation phases.
#[test]
fn test_tile_sync_with_computation() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_sync_with_computation")?;

    // Test with a single warp
    let num_threads = 32u32;
    let input: Vec<u32> = (0..num_threads).collect();
    let d_input = DeviceBuffer::from_slice(&input)?;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_input.as_device_ptr(),
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results - each value should be doubled
    println!("Tile sync with computation test:");
    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        let expected = inp * 2;
        assert_eq!(
            out, expected,
            "Thread {} output {} != expected {}",
            i, out, expected
        );
        println!("  Thread {}: {} * 2 = {} ✓", i, inp, out);
    }

    println!("✓ Tile sync with computation test passed");
    Ok(())
}

/// Test tile.shfl() broadcast operation
///
/// Verifies that shfl() correctly broadcasts a value from a specific source
/// lane to all threads in the tile.
#[test]
fn test_tile_shfl_broadcast() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_shfl_broadcast")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0i32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results - all threads should have received 42 from thread 0
    println!("Tile shfl() broadcast test:");
    for (i, &value) in output.iter().enumerate() {
        assert_eq!(
            value, 42,
            "Thread {} received {}, expected 42 (broadcast from thread 0)",
            i, value
        );
        println!("  Thread {}: {} ✓", i, value);
    }

    println!("✓ Tile shfl() broadcast test passed");
    Ok(())
}

/// Test tile.shfl_down() shift operation
///
/// Verifies that shfl_down() correctly shifts values down within the tile,
/// with each thread receiving from the thread delta lanes lower.
#[test]
fn test_tile_shfl_down() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_shfl_down")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0i32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results
    println!("Tile shfl_down() shift test:");
    for (i, &value) in output.iter().enumerate() {
        let expected = if i >= 31 {
            // Thread 31 receives its own value (no thread 32)
            31
        } else {
            // Thread N receives value from thread N+1
            (i + 1) as i32
        };

        assert_eq!(
            value, expected,
            "Thread {} received {}, expected {}",
            i, value, expected
        );
        println!("  Thread {}: {} ✓", i, value);
    }

    println!("✓ Tile shfl_down() shift test passed");
    Ok(())
}

/// Test tile.shfl_up() shift operation
///
/// Verifies that shfl_up() correctly shifts values up within the tile,
/// with each thread receiving from the thread delta lanes lower.
#[test]
fn test_tile_shfl_up() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_shfl_up")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0i32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results
    println!("Tile shfl_up() shift test:");
    for (i, &value) in output.iter().enumerate() {
        let expected = if i == 0 {
            // Thread 0 receives its own value (no thread -1)
            0
        } else {
            // Thread N receives value from thread N-1
            (i - 1) as i32
        };

        assert_eq!(
            value, expected,
            "Thread {} received {}, expected {}",
            i, value, expected
        );
        println!("  Thread {}: {} ✓", i, value);
    }

    println!("✓ Tile shfl_up() shift test passed");
    Ok(())
}

/// Test tile.shfl_xor() butterfly operation
///
/// Verifies that shfl_xor() correctly exchanges values using XOR addressing,
/// creating butterfly communication patterns.
#[test]
fn test_tile_shfl_xor() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_shfl_xor")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0i32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results
    println!("Tile shfl_xor() butterfly test:");
    for (i, &value) in output.iter().enumerate() {
        // With XOR mask=1, thread N exchanges with thread N^1
        // Even threads (0,2,4,...) get odd values (1,3,5,...)
        // Odd threads (1,3,5,...) get even values (0,2,4,...)
        let expected = (i ^ 1) as i32;

        assert_eq!(
            value, expected,
            "Thread {} received {}, expected {} (XOR with mask=1)",
            i, value, expected
        );
        println!("  Thread {}: {} ✓", i, value);
    }

    println!("✓ Tile shfl_xor() butterfly test passed");
    Ok(())
}

/// Test tile.any() vote operation
///
/// Verifies that any() correctly detects when at least one thread
/// has a true predicate in divergent execution.
#[test]
fn test_tile_any() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_any")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Verify results - all threads should see any() == true
    // because threads 0-15 have true predicate
    println!("Tile any() vote test:");
    for (i, &value) in output.iter().enumerate() {
        assert_eq!(
            value, 1,
            "Thread {} got any()={}, expected true (1)",
            i, value
        );
        println!("  Thread {}: any() = {} ✓", i, value);
    }

    println!("✓ Tile any() vote test passed");
    Ok(())
}

/// Test tile.all() vote operation
///
/// Verifies that all() correctly detects when all threads have true predicate
/// and when at least one thread has false.
#[test]
fn test_tile_all() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_all")?;

    // Test with a single warp, need 64 outputs (2 tests)
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; 64])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; 64];
    d_output.copy_to(&mut output)?;

    // Verify Test 1 results (all threads have true) - output[0..32]
    println!("Tile all() vote test - Test 1 (all true):");
    for i in 0..32 {
        let value = output[i];
        assert_eq!(
            value, 1,
            "Test 1: Thread {} got all()={}, expected true (1)",
            i, value
        );
        println!("  Thread {}: all() = {} ✓", i, value);
    }

    // Verify Test 2 results (thread 0 has false) - output[32..64]
    println!("Tile all() vote test - Test 2 (thread 0 false):");
    for i in 0..32 {
        let value = output[32 + i];
        assert_eq!(
            value, 0,
            "Test 2: Thread {} got all()={}, expected false (0)",
            i, value
        );
        println!("  Thread {}: all() = {} ✓", i, value);
    }

    println!("✓ Tile all() vote test passed");
    Ok(())
}

/// Test tile.ballot() vote operation
///
/// Verifies that ballot() correctly collects votes from all threads as a bitmask.
#[test]
fn test_tile_ballot() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_ballot")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Expected: even threads (0,2,4,...) have true predicate
    // Thread N sets bit N, so:
    // Bit:   31 30 29 28 ... 3 2 1 0
    // Value:  0  1  0  1 ... 0 1 0 1
    // Thread 0 sets bit 0, thread 2 sets bit 2, etc.
    // This is 0x55555555 (alternating 01 pattern from LSB)
    let expected_mask = 0x55555555u32;

    // Verify results - all threads should see the same ballot mask
    println!("Tile ballot() vote test:");
    println!(
        "  Expected mask: 0x{:08X} (binary: {:032b})",
        expected_mask, expected_mask
    );
    for (i, &value) in output.iter().enumerate() {
        assert_eq!(
            value, expected_mask,
            "Thread {} got ballot()=0x{:08X}, expected 0x{:08X}",
            i, value, expected_mask
        );
        if i < 4 || i >= 28 {
            println!("  Thread {}: ballot() = 0x{:08X} ✓", i, value);
        } else if i == 4 {
            println!("  ... (threads 4-27 also 0x{:08X}) ...", expected_mask);
        }
    }

    println!("✓ Tile ballot() vote test passed");
    Ok(())
}

/// Test tile.match_any() operation (SM 7.0+)
///
/// Verifies that match_any() correctly groups threads by matching values.
#[test]
fn test_tile_match_any() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_match_any")?;

    // Test with a single warp
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; num_threads as usize];
    d_output.copy_to(&mut output)?;

    // Expected masks:
    // Threads 0-15 (value=100): 0x0000FFFF (bits 0-15 set)
    // Threads 16-31 (value=200): 0xFFFF0000 (bits 16-31 set)
    let mask_group1 = 0x0000FFFFu32;
    let mask_group2 = 0xFFFF0000u32;

    // Verify results
    println!("Tile match_any() test:");
    println!(
        "  Group 1 (threads 0-15, value=100): expect 0x{:08X}",
        mask_group1
    );
    println!(
        "  Group 2 (threads 16-31, value=200): expect 0x{:08X}",
        mask_group2
    );

    for (i, &value) in output.iter().enumerate() {
        let expected = if i < 16 { mask_group1 } else { mask_group2 };
        assert_eq!(
            value, expected,
            "Thread {} got match_any()=0x{:08X}, expected 0x{:08X}",
            i, value, expected
        );
        if i < 4 || i == 15 || i == 16 || i >= 28 {
            println!("  Thread {}: match_any() = 0x{:08X} ✓", i, value);
        } else if i == 4 {
            println!("  ... (threads 4-14 also 0x{:08X}) ...", expected);
        } else if i == 17 {
            println!("  ... (threads 17-27 also 0x{:08X}) ...", expected);
        }
    }

    println!("✓ Tile match_any() test passed");
    Ok(())
}

/// Test tile.match_all() operation (SM 7.0+)
///
/// Verifies that match_all() correctly returns both match mask and unanimity flag.
#[test]
fn test_tile_match_all() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Load Rust module
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernel function
    let kernel = module.get_function("test_tile_match_all")?;

    // Test with a single warp, need 128 outputs (2 tests × 2 outputs × 32 threads)
    let num_threads = 32u32;
    let d_output = DeviceBuffer::from_slice(&vec![0u32; 128])?;

    // Launch kernel
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                d_output.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    // Copy results back
    let mut output = vec![0u32; 128];
    d_output.copy_to(&mut output)?;

    // Test 1: All threads have value=42 (unanimous)
    // output[0..32]: match masks (should all be 0xFFFFFFFF)
    // output[32..64]: all_match flags (should all be 1)
    println!("Tile match_all() test - Test 1 (unanimous value=42):");
    for i in 0..32 {
        let mask = output[i];
        let all_match = output[32 + i];

        assert_eq!(
            mask, 0xFFFFFFFF,
            "Test 1: Thread {} mask=0x{:08X}, expected 0xFFFFFFFF",
            i, mask
        );
        assert_eq!(
            all_match, 1,
            "Test 1: Thread {} all_match={}, expected 1",
            i, all_match
        );

        if i < 4 {
            println!("  Thread {}: (0x{:08X}, {}) ✓", i, mask, all_match);
        } else if i == 4 {
            println!("  ... (threads 4-31 also same) ...");
        }
    }

    // Test 2: Thread 0 has value=99, rest have value=42 (not unanimous)
    // match.all.sync semantics: When values diverge, ALL threads get mask=0x00000000
    // This is different from match.any.sync which returns per-thread match masks
    // output[64..96]: match masks (all threads get 0x00000000 when divergent)
    // output[96..128]: all_match flags (should all be 0)
    println!("Tile match_all() test - Test 2 (thread 0 divergent):");
    for i in 0..32 {
        let mask = output[64 + i];
        let all_match = output[96 + i];

        // When values diverge, match.all.sync returns 0x00000000 for ALL threads
        let expected_mask = 0x00000000u32;

        assert_eq!(
            mask, expected_mask,
            "Test 2: Thread {} mask=0x{:08X}, expected 0x{:08X}",
            i, mask, expected_mask
        );
        assert_eq!(
            all_match, 0,
            "Test 2: Thread {} all_match={}, expected 0",
            i, all_match
        );

        if i < 4 {
            println!("  Thread {}: (0x{:08X}, {}) ✓", i, mask, all_match);
        } else if i == 4 {
            println!("  ... (threads 4-31 also have mask=0x00000000, all_match=0) ...");
        }
    }

    println!("✓ Tile match_all() test passed");
    Ok(())
}
