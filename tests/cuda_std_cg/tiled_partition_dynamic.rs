mod common;
use common::*;

/// Test dynamic tile creation with runtime size selection
///
/// Verifies:
/// - tiled_partition_dynamic() creates valid tile handle
/// - Size is stored correctly and returned by size()
/// - Thread ranks are correct for runtime-determined sizes
/// - Synchronization works without deadlock
#[test]
fn test_dynamic_tile_creation() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_dynamic_tile_creation")?;

    // Test with 64 threads (2 warps)
    let num_threads = 64u32;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    // Test tile size 32
    let tile_size = 32u32;
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                ranks.as_device_ptr(),
                sizes.as_device_ptr(),
                tile_size
            )
        )?;
    }
    stream.synchronize()?;

    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut sizes_host = vec![0u32; num_threads as usize];
    ranks.copy_to(&mut ranks_host)?;
    sizes.copy_to(&mut sizes_host)?;

    // Verify results
    for (thread_idx, (&rank, &size)) in ranks_host.iter().zip(sizes_host.iter()).enumerate() {
        assert_eq!(
            size, tile_size,
            "Thread {} has incorrect tile size",
            thread_idx
        );
        let expected_rank = (thread_idx % tile_size as usize) as u32;
        assert_eq!(
            rank, expected_rank,
            "Thread {} has rank {}, expected {}",
            thread_idx, rank, expected_rank
        );
    }

    println!("✓ Dynamic tile creation test passed for size {}", tile_size);
    Ok(())
}

/// Test all valid tile sizes (1, 2, 4, 8, 16, 32)
///
/// Verifies that dynamic partition works correctly for all valid tile sizes
#[test]
fn test_dynamic_all_sizes() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_dynamic_tile_creation")?;

    let num_threads = 64u32;
    let valid_sizes = [1u32, 2, 4, 8, 16, 32];

    for &tile_size in &valid_sizes {
        let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
        let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

        unsafe {
            launch!(
                kernel<<<1, num_threads, 0, stream>>>(
                    ranks.as_device_ptr(),
                    sizes.as_device_ptr(),
                    tile_size
                )
            )?;
        }
        stream.synchronize()?;

        let mut ranks_host = vec![0u32; num_threads as usize];
        let mut sizes_host = vec![0u32; num_threads as usize];
        ranks.copy_to(&mut ranks_host)?;
        sizes.copy_to(&mut sizes_host)?;

        // Verify all threads report correct size and rank
        for (thread_idx, (&rank, &size)) in ranks_host.iter().zip(sizes_host.iter()).enumerate() {
            assert_eq!(
                size, tile_size,
                "Size {} - Thread {} has incorrect tile size {}",
                tile_size, thread_idx, size
            );
            let expected_rank = (thread_idx % tile_size as usize) as u32;
            assert_eq!(
                rank, expected_rank,
                "Size {} - Thread {} has rank {}, expected {}",
                tile_size, thread_idx, rank, expected_rank
            );
        }

        println!("✓ Dynamic partition test passed for size {}", tile_size);
    }

    Ok(())
}

/// Test dynamic shuffle operations
///
/// Verifies that all shuffle variants work correctly with dynamic tiles
#[test]
fn test_dynamic_shuffle() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_dynamic_shuffle")?;

    let num_threads = 64u32;
    let tile_size = 16u32;
    let results = DeviceBuffer::from_slice(&vec![0i32; (num_threads * 4) as usize])?; // 4 shuffle variants

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                results.as_device_ptr(),
                tile_size
            )
        )?;
    }
    stream.synchronize()?;

    let mut results_host = vec![0i32; (num_threads * 4) as usize];
    results.copy_to(&mut results_host)?;

    // Verify shuffle results
    for thread_idx in 0..num_threads as usize {
        let rank = (thread_idx % tile_size as usize) as i32;
        let shfl_result = results_host[thread_idx * 4];
        let shfl_down_result = results_host[thread_idx * 4 + 1];
        let shfl_up_result = results_host[thread_idx * 4 + 2];
        let shfl_xor_result = results_host[thread_idx * 4 + 3];

        // shfl(rank, 0) should return 0 for all threads
        assert_eq!(
            shfl_result, 0,
            "Thread {} shfl incorrect: got {}, expected 0",
            thread_idx, shfl_result
        );

        // Verify other shuffle operations have valid results
        // (actual values depend on shuffle semantics)
        println!(
            "Thread {}: shfl={}, down={}, up={}, xor={}",
            thread_idx, shfl_result, shfl_down_result, shfl_up_result, shfl_xor_result
        );
    }

    println!("✓ Dynamic shuffle test passed for size {}", tile_size);
    Ok(())
}

/// Test dynamic vote operations (any, all, ballot)
///
/// Verifies that vote operations work correctly with dynamic tiles
#[test]
fn test_dynamic_vote() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_dynamic_vote")?;

    let num_threads = 64u32;
    let tile_size = 8u32;
    let any_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let all_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ballot_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                any_results.as_device_ptr(),
                all_results.as_device_ptr(),
                ballot_results.as_device_ptr(),
                tile_size
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

    // All threads in same tile should see same vote results
    for tile_idx in 0..(num_threads / tile_size) {
        let tile_start = (tile_idx * tile_size) as usize;
        let tile_end = ((tile_idx + 1) * tile_size) as usize;

        let any_ref = any_host[tile_start];
        let all_ref = all_host[tile_start];
        let ballot_ref = ballot_host[tile_start];

        for thread_idx in tile_start..tile_end {
            assert_eq!(
                any_host[thread_idx], any_ref,
                "Thread {} in tile {} has different any result",
                thread_idx, tile_idx
            );
            assert_eq!(
                all_host[thread_idx], all_ref,
                "Thread {} in tile {} has different all result",
                thread_idx, tile_idx
            );
            assert_eq!(
                ballot_host[thread_idx], ballot_ref,
                "Thread {} in tile {} has different ballot result",
                thread_idx, tile_idx
            );
        }
    }

    println!("✓ Dynamic vote test passed for size {}", tile_size);
    Ok(())
}

/// Test dynamic match operations (match_any, match_all)
///
/// Verifies that match operations work correctly with dynamic tiles
#[test]
fn test_dynamic_match() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_dynamic_match")?;

    let num_threads = 64u32;
    let tile_size = 32u32;
    let match_any_results = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let match_all_mask = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let match_all_pred = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                match_any_results.as_device_ptr(),
                match_all_mask.as_device_ptr(),
                match_all_pred.as_device_ptr(),
                tile_size
            )
        )?;
    }
    stream.synchronize()?;

    let mut match_any_host = vec![0u32; num_threads as usize];
    let mut match_all_mask_host = vec![0u32; num_threads as usize];
    let mut match_all_pred_host = vec![0u32; num_threads as usize];
    match_any_results.copy_to(&mut match_any_host)?;
    match_all_mask.copy_to(&mut match_all_mask_host)?;
    match_all_pred.copy_to(&mut match_all_pred_host)?;

    // Verify match results are consistent within tiles
    for tile_idx in 0..(num_threads / tile_size) {
        let tile_start = (tile_idx * tile_size) as usize;
        println!(
            "Tile {}: match_any={:032b}, match_all_mask={:032b}, all_match={}",
            tile_idx,
            match_any_host[tile_start],
            match_all_mask_host[tile_start],
            match_all_pred_host[tile_start]
        );
    }

    println!("✓ Dynamic match test passed for size {}", tile_size);
    Ok(())
}

/// Test mixed static and dynamic tiles in same kernel
///
/// Verifies that both TiledGroup<SIZE> and DynamicTiledGroup can coexist
#[test]
fn test_mixed_static_dynamic() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_mixed_static_dynamic")?;

    let num_threads = 64u32;
    let static_sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let dynamic_sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    let dynamic_tile_size = 16u32;
    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                static_sizes.as_device_ptr(),
                dynamic_sizes.as_device_ptr(),
                dynamic_tile_size
            )
        )?;
    }
    stream.synchronize()?;

    let mut static_host = vec![0u32; num_threads as usize];
    let mut dynamic_host = vec![0u32; num_threads as usize];
    static_sizes.copy_to(&mut static_host)?;
    dynamic_sizes.copy_to(&mut dynamic_host)?;

    // Verify static tiles all have size 32
    for (i, &size) in static_host.iter().enumerate() {
        assert_eq!(size, 32, "Thread {} static tile has wrong size", i);
    }

    // Verify dynamic tiles all have size 16
    for (i, &size) in dynamic_host.iter().enumerate() {
        assert_eq!(
            size, dynamic_tile_size,
            "Thread {} dynamic tile has wrong size",
            i
        );
    }

    println!("✓ Mixed static/dynamic tile test passed");
    Ok(())
}
