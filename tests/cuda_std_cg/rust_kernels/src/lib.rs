#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

// Disabled: Cluster APIs require SM 9.0+ (Hopper), causes InvalidPtx on SM 8.0 targets
// pub mod cluster;
pub mod coalesced_group;
pub mod grid_sync_basic;
pub mod grid_sync_multiblock;
pub mod grid_sync_multiblock_simple;
pub mod grid_sync_phases;
pub mod memcpy_async;
pub mod partitioning;
pub mod reduction;
// Disabled: scan operations are incomplete (Priority 6 blocked)
pub mod scan;
pub mod thread_block_basic;
pub mod tiled_partition_basic;
pub mod tiled_partition_dynamic;
pub mod tiled_partition_sizes;
