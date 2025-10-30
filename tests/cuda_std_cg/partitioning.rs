//! Tests for labeled_partition and binary_partition functions.

mod common;
use common::*;

/// Test basic labeled_partition with uniform groups.
#[test]
fn test_labeled_partition_uniform() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_labeled_partition_uniform")?;

    let num_threads = 32u32;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                sizes.as_device_ptr(),
                ranks.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    sizes.copy_to(&mut sizes_host)?;
    ranks.copy_to(&mut ranks_host)?;
    masks.copy_to(&mut masks_host)?;

    // Verify each group has 8 threads
    for i in 0..32 {
        assert_eq!(sizes_host[i], 8, "Thread {} should have group size 8", i);
    }

    // Verify ranks 0-7 within each group
    for group_id in 0..4 {
        for offset in 0..8 {
            let tid = group_id * 8 + offset;
            assert_eq!(
                ranks_host[tid], offset as u32,
                "Thread {} should have rank {}",
                tid, offset
            );
        }
    }

    // Verify masks
    let expected_masks = [0x000000FFu32, 0x0000FF00u32, 0x00FF0000u32, 0xFF000000u32];
    for i in 0..32 {
        let group_id = i / 8;
        assert_eq!(
            masks_host[i], expected_masks[group_id],
            "Thread {} should have mask 0x{:08X}",
            i, expected_masks[group_id]
        );
    }

    Ok(())
}

/// Test labeled_partition with sparse/uneven groups.
#[test]
fn test_labeled_partition_sparse() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_labeled_partition_sparse")?;

    let num_threads = 32u32;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                sizes.as_device_ptr(),
                ranks.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    sizes.copy_to(&mut sizes_host)?;
    ranks.copy_to(&mut ranks_host)?;
    masks.copy_to(&mut masks_host)?;

    let expected_sizes = [11, 11, 10]; // Labels 0, 1, 2
    let expected_masks = [0x49249249u32, 0x92492492u32, 0x24924924u32];

    for tid in 0..32 {
        let label = tid % 3;
        assert_eq!(
            sizes_host[tid], expected_sizes[label],
            "Thread {} (label {}) should have size {}",
            tid, label, expected_sizes[label]
        );
        assert_eq!(
            masks_host[tid], expected_masks[label],
            "Thread {} (label {}) should have mask 0x{:08X}",
            tid, label, expected_masks[label]
        );
    }

    // Verify ranks
    let mut rank_counters = [0u32; 3];
    for tid in 0..32 {
        let label = tid % 3;
        let expected_rank = rank_counters[label];
        rank_counters[label] += 1;
        assert_eq!(
            ranks_host[tid], expected_rank,
            "Thread {} (label {}) should have rank {}",
            tid, label, expected_rank
        );
    }

    Ok(())
}

/// Test binary_partition with even/odd split.
#[test]
fn test_binary_partition_even_odd() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_binary_partition_even_odd")?;

    let num_threads = 32u32;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                sizes.as_device_ptr(),
                ranks.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    sizes.copy_to(&mut sizes_host)?;
    ranks.copy_to(&mut ranks_host)?;
    masks.copy_to(&mut masks_host)?;

    let even_mask = 0x55555555u32;
    let odd_mask = 0xAAAAAAAAu32;

    for i in 0..32 {
        assert_eq!(sizes_host[i], 16, "Thread {} should have size 16", i);
        let expected_mask = if i % 2 == 0 { even_mask } else { odd_mask };
        assert_eq!(
            masks_host[i], expected_mask,
            "Thread {} should have mask 0x{:08X}",
            i, expected_mask
        );
        assert_eq!(
            ranks_host[i],
            (i / 2) as u32,
            "Thread {} should have rank {}",
            i,
            i / 2
        );
    }

    Ok(())
}

/// Test binary_partition with threshold condition.
#[test]
fn test_binary_partition_threshold() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_binary_partition_threshold")?;

    let num_threads = 32u32;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                sizes.as_device_ptr(),
                ranks.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    sizes.copy_to(&mut sizes_host)?;
    ranks.copy_to(&mut ranks_host)?;
    masks.copy_to(&mut masks_host)?;

    let below_mask = 0x000FFFFFu32;
    let above_mask = 0xFFF00000u32;

    for i in 0..20 {
        assert_eq!(sizes_host[i], 20, "Thread {} should have size 20", i);
        assert_eq!(
            ranks_host[i], i as u32,
            "Thread {} should have rank {}",
            i, i
        );
        assert_eq!(
            masks_host[i], below_mask,
            "Thread {} should have below mask",
            i
        );
    }

    for i in 20..32 {
        assert_eq!(sizes_host[i], 12, "Thread {} should have size 12", i);
        assert_eq!(
            ranks_host[i],
            (i - 20) as u32,
            "Thread {} should have rank {}",
            i,
            i - 20
        );
        assert_eq!(
            masks_host[i], above_mask,
            "Thread {} should have above mask",
            i
        );
    }

    Ok(())
}

/// Test partitioning from TiledGroup parent.
#[test]
fn test_partition_from_tile() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_partition_from_tile")?;

    let num_threads = 32u32;
    let sizes = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let ranks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;
    let masks = DeviceBuffer::from_slice(&vec![0u32; num_threads as usize])?;

    unsafe {
        launch!(
            kernel<<<1, num_threads, 0, stream>>>(
                sizes.as_device_ptr(),
                ranks.as_device_ptr(),
                masks.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut sizes_host = vec![0u32; num_threads as usize];
    let mut ranks_host = vec![0u32; num_threads as usize];
    let mut masks_host = vec![0u32; num_threads as usize];
    sizes.copy_to(&mut sizes_host)?;
    ranks.copy_to(&mut ranks_host)?;
    masks.copy_to(&mut masks_host)?;

    for i in 0..32 {
        assert_eq!(sizes_host[i], 8, "Thread {} should have size 8", i);

        let tile_id = i / 16;
        let is_even = (i % 16) % 2 == 0;
        let expected_mask = match (tile_id, is_even) {
            (0, true) => 0x00005555u32,
            (0, false) => 0x0000AAAAu32,
            (1, true) => 0x55550000u32,
            (1, false) => 0xAAAA0000u32,
            _ => unreachable!(),
        };
        assert_eq!(
            masks_host[i], expected_mask,
            "Thread {} should have mask 0x{:08X}",
            i, expected_mask
        );

        let expected_rank = ((i % 16) / 2) as u32;
        assert_eq!(
            ranks_host[i], expected_rank,
            "Thread {} should have rank {}",
            i, expected_rank
        );
    }

    Ok(())
}

/// Test that partitioned groups support shuffle operations.
#[test]
fn test_partition_operations() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;
    let module = load_rust_module()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("test_partition_operations")?;

    let num_threads = 32u32;
    let output = DeviceBuffer::from_slice(&vec![0i32; num_threads as usize])?;

    unsafe {
        launch!(kernel<<<1, num_threads, 0, stream>>>(output.as_device_ptr()))?;
    }

    stream.synchronize()?;

    let mut results = vec![0i32; num_threads as usize];
    output.copy_to(&mut results)?;

    // Verify shuffle operations work
    // Most threads should get value from next rank
    for i in 0..28 {
        let is_even = i % 2 == 0;
        let rank = i / 2;
        let expected = if is_even {
            ((rank + 1) * 2) as i32
        } else {
            ((rank + 1) * 2 + 1) as i32
        };
        assert_eq!(
            results[i as usize], expected,
            "Thread {} should have shuffled value {}",
            i, expected
        );
    }

    // Threads 30 and 31 (rank 15 in their groups) should get their own values
    // since shfl_down(1) from rank 15 is out of bounds
    assert_eq!(results[30], 30, "Thread 30 should have own value");
    assert_eq!(results[31], 31, "Thread 31 should have own value");

    Ok(())
}
