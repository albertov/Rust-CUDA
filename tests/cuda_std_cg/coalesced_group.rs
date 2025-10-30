mod common;
use common::*;

/// Test basic coalesced group creation with all threads active (convergent)
///
/// Verifies:
/// - coalesced_threads() creates valid group
/// - All 32 threads in warp are active
/// - Ranks are correctly assigned 0-31
/// - Size is 32 for full warp
/// - Mask is 0xFFFFFFFF
#[test]
fn test_coalesced_all_active() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_coalesced_all_active")?;

    let num_threads = 32u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;
    masks.copy_to(&mut masks_host)?;

    println!("Coalesced group (all active) for {} threads:", num_threads);
    for (thread_idx, ((&rank, &size), &mask)) in ranks_host
        .iter()
        .zip(sizes_host.iter())
        .zip(masks_host.iter())
        .enumerate()
    {
        println!(
            "  Thread {}: rank {}, size {}, mask {:08x}",
            thread_idx, rank, size, mask
        );

        // All threads active: size should be 32
        assert_eq!(size, 32, "Thread {} has incorrect size", thread_idx);

        // Rank should match lane ID
        assert_eq!(
            rank, thread_idx as u32,
            "Thread {} has rank {}, expected {}",
            thread_idx, rank, thread_idx
        );

        // Mask should be full (all 32 lanes active)
        assert_eq!(
            mask, 0xFFFFFFFF,
            "Thread {} has incorrect mask {:08x}",
            thread_idx, mask
        );
    }

    println!("✓ Coalesced all-active test passed");
    Ok(())
}

/// Test coalesced group with divergent execution (if/else split)
///
/// Verifies:
/// - Threads that take same branch form correct group
/// - Only first 16 threads active in their group
/// - Ranks are 0-15 for active threads
/// - Size is 16
/// - Mask reflects only active threads
///
/// NOTE: Uses C++ reference kernel due to Rust compiler optimizations
/// that eliminate divergence detection in Rust-compiled kernels.
#[test]
fn test_coalesced_divergent() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_cpp_module("coalesced")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_coalesced_divergent_cpp")?;

    let num_threads = 32u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;
    masks.copy_to(&mut masks_host)?;

    println!("Coalesced group (divergent) for {} threads:", num_threads);

    // First 16 threads (active branch)
    for thread_idx in 0..16 {
        let rank = ranks_host[thread_idx];
        let size = sizes_host[thread_idx];
        let mask = masks_host[thread_idx];

        println!(
            "  Thread {} (active): rank {}, size {}, mask {:08x}",
            thread_idx, rank, size, mask
        );

        // Active group should have 16 threads
        assert_eq!(
            size, 16,
            "Thread {} (active) has incorrect size",
            thread_idx
        );

        // Ranks should be 0-15
        assert_eq!(
            rank, thread_idx as u32,
            "Thread {} (active) has rank {}, expected {}",
            thread_idx, rank, thread_idx
        );

        // Mask should show lower 16 bits set (lanes 0-15)
        assert_eq!(
            mask, 0x0000FFFF,
            "Thread {} (active) has incorrect mask {:08x}",
            thread_idx, mask
        );
    }

    // Last 16 threads (inactive branch) - written by else clause
    for thread_idx in 16..32 {
        let rank = ranks_host[thread_idx];
        let size = sizes_host[thread_idx];
        let mask = masks_host[thread_idx];

        println!(
            "  Thread {} (else): rank {}, size {}, mask {:08x}",
            thread_idx, rank, size, mask
        );

        // Else group should have 16 threads
        assert_eq!(size, 16, "Thread {} (else) has incorrect size", thread_idx);

        // Ranks should be 0-15 (relative to their group)
        assert_eq!(
            rank,
            (thread_idx - 16) as u32,
            "Thread {} (else) has rank {}, expected {}",
            thread_idx,
            rank,
            thread_idx - 16
        );

        // Mask should show upper 16 bits set (lanes 16-31)
        assert_eq!(
            mask, 0xFFFF0000,
            "Thread {} (else) has incorrect mask {:08x}",
            thread_idx, mask
        );
    }

    println!("✓ Coalesced divergent test passed");
    Ok(())
}

/// Test coalesced group with sparse threads (every other thread)
///
/// Verifies:
/// - Non-contiguous threads form correct group
/// - Even threads (0,2,4,...,30) have ranks 0-15
/// - Size is 16
/// - Mask has every other bit set
///
/// NOTE: Uses C++ reference kernel due to Rust compiler optimizations
/// that eliminate divergence detection in Rust-compiled kernels.
#[test]
fn test_coalesced_sparse() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_cpp_module("coalesced")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_coalesced_sparse_cpp")?;

    let num_threads = 32u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;
    masks.copy_to(&mut masks_host)?;

    println!("Coalesced group (sparse) for {} threads:", num_threads);

    let expected_mask = 0x55555555u32; // Every other bit: 0101...0101

    // Even threads (active)
    for i in 0..16 {
        let thread_idx = i * 2;
        let rank = ranks_host[thread_idx];
        let size = sizes_host[thread_idx];
        let mask = masks_host[thread_idx];

        println!(
            "  Thread {} (even): rank {}, size {}, mask {:08x}",
            thread_idx, rank, size, mask
        );

        // Group should have 16 threads
        assert_eq!(size, 16, "Thread {} (even) has incorrect size", thread_idx);

        // Rank should be sequential 0-15
        assert_eq!(
            rank, i as u32,
            "Thread {} (even) has rank {}, expected {}",
            thread_idx, rank, i
        );

        // Mask should show every other bit set
        assert_eq!(
            mask, expected_mask,
            "Thread {} (even) has incorrect mask {:08x}",
            thread_idx, mask
        );
    }

    // Odd threads (written by else clause)
    for i in 0..16 {
        let thread_idx = i * 2 + 1;
        let rank = ranks_host[thread_idx];
        let size = sizes_host[thread_idx];
        let mask = masks_host[thread_idx];

        println!(
            "  Thread {} (odd): rank {}, size {}, mask {:08x}",
            thread_idx, rank, size, mask
        );

        // Group should have 16 threads
        assert_eq!(size, 16, "Thread {} (odd) has incorrect size", thread_idx);

        // Rank should be sequential 0-15
        assert_eq!(
            rank, i as u32,
            "Thread {} (odd) has rank {}, expected {}",
            thread_idx, rank, i
        );

        // Mask should show every other bit set (offset by 1)
        let odd_mask = 0xAAAAAAAAu32; // 1010...1010
        assert_eq!(
            mask, odd_mask,
            "Thread {} (odd) has incorrect mask {:08x}",
            thread_idx, mask
        );
    }

    println!("✓ Coalesced sparse test passed");
    Ok(())
}

/// Test shuffle operations with coalesced group
///
/// Verifies:
/// - shfl() broadcasts from source rank to all threads
/// - shfl_down() shifts values correctly
/// - shfl_up() shifts values correctly
/// - Works with sparse thread patterns
///
/// NOTE: Uses C++ reference kernel due to Rust compiler optimizations
/// that eliminate divergence detection in Rust-compiled kernels.
#[test]
fn test_coalesced_shuffle() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_cpp_module("coalesced")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_coalesced_shuffle_cpp")?;

    let num_threads = 32u32;
    let shfl_results = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;
    let shfl_down_results = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;
    let shfl_up_results = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                shfl_results.as_device_ptr(),
                shfl_down_results.as_device_ptr(),
                shfl_up_results.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut shfl_host = vec![0i32; num_threads as usize];
    let mut shfl_down_host = vec![0i32; num_threads as usize];
    let mut shfl_up_host = vec![0i32; num_threads as usize];
    shfl_results.copy_to(&mut shfl_host)?;
    shfl_down_results.copy_to(&mut shfl_down_host)?;
    shfl_up_results.copy_to(&mut shfl_up_host)?;

    println!("Coalesced shuffle results:");

    // Test shfl() - broadcast from rank 0 to all even threads
    for i in 0..16 {
        let thread_idx = i * 2;
        let value = shfl_host[thread_idx];
        println!("  Thread {} shfl: {}", thread_idx, value);

        // All even threads should receive value from rank 0 (thread 0)
        assert_eq!(value, 0, "Thread {} shfl result incorrect", thread_idx);
    }

    // Test shfl_down() - each thread gets value from next rank
    for i in 0..15 {
        let thread_idx = i * 2;
        let value = shfl_down_host[thread_idx];
        let expected = (i + 1) * 2; // Next even thread's lane ID
        println!("  Thread {} shfl_down: {}", thread_idx, value);

        assert_eq!(
            value, expected as i32,
            "Thread {} shfl_down result incorrect",
            thread_idx
        );
    }

    // Last active thread (rank 15) should get its own value
    let last_thread = 30;
    assert_eq!(
        shfl_down_host[last_thread], last_thread as i32,
        "Last thread shfl_down incorrect"
    );

    // Test shfl_up() - each thread gets value from previous rank
    for i in 1..16 {
        let thread_idx = i * 2;
        let value = shfl_up_host[thread_idx];
        let expected = (i - 1) * 2; // Previous even thread's lane ID
        println!("  Thread {} shfl_up: {}", thread_idx, value);

        assert_eq!(
            value, expected as i32,
            "Thread {} shfl_up result incorrect",
            thread_idx
        );
    }

    // First active thread (rank 0) should get its own value
    assert_eq!(shfl_up_host[0], 0, "First thread shfl_up incorrect");

    println!("✓ Coalesced shuffle test passed");
    Ok(())
}

/// Test vote operations (any, all, ballot) with coalesced group
///
/// Verifies:
/// - any() returns true if any thread has predicate true
/// - all() returns true if all threads have predicate true
/// - ballot() returns mask of threads with predicate true
#[test]
fn test_coalesced_vote() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_coalesced_vote")?;

    let num_threads = 32u32;
    let any_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let all_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ballot_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                any_results.as_device_ptr(),
                all_results.as_device_ptr(),
                ballot_results.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut any_host = vec![0u32; num_threads as usize];
    let mut all_host = vec![0u32; num_threads as usize];
    let mut ballot_host = vec![0u32; num_threads as usize];
    any_results.copy_to(&mut any_host)?;
    all_results.copy_to(&mut all_host)?;
    ballot_results.copy_to(&mut ballot_host)?;

    println!("Coalesced vote results:");

    // Only even threads participate (rank < 8 predicate)
    for i in 0..16 {
        let thread_idx = i * 2;
        let any = any_host[thread_idx];
        let all = all_host[thread_idx];
        let ballot = ballot_host[thread_idx];

        println!(
            "  Thread {}: any={}, all={}, ballot={:08x}",
            thread_idx, any, all, ballot
        );

        // any() should be true (some threads have rank < 8)
        assert_eq!(any, 1, "Thread {} any() incorrect", thread_idx);

        // all() should be false (not all threads have rank < 8)
        assert_eq!(all, 0, "Thread {} all() incorrect", thread_idx);

        // ballot should show first 8 ranks (threads 0,2,4,...,14)
        let expected_ballot = 0x00005555u32; // First 8 even lanes
        assert_eq!(
            ballot, expected_ballot,
            "Thread {} ballot incorrect: got {:08x}, expected {:08x}",
            thread_idx, ballot, expected_ballot
        );
    }

    println!("✓ Coalesced vote test passed");
    Ok(())
}
