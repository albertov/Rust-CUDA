//! Cooperative Groups API for CUDA in Rust.
//!
//! This module provides Rust bindings for CUDA's cooperative groups functionality,
//! enabling synchronization and communication patterns beyond traditional block-level
//! primitives.
//!
//! # Overview
//!
//! Cooperative groups extends CUDA's execution model to support:
//! - **Grid synchronization**: Synchronize across all blocks in a kernel launch
//! - **Thread block tiles**: Partition blocks into smaller groups for warp-level operations
//! - **Flexible synchronization**: Express complex synchronization patterns safely
//!
//! # Grid Synchronization
//!
//! Grid-wide synchronization requires:
//! 1. Device support (query `cudaDevAttrCooperativeLaunch`)
//! 2. Special launch API (`cudaLaunchCooperativeKernel`)
//! 3. Careful memory ordering (acquire/release semantics)
//!
//! # Architecture Requirements
//!
//! - **Minimum**: SM 6.0 (Pascal) for grid synchronization
//! - **Recommended**: SM 7.0+ (Volta+) for efficient acquire/release atomics
//!
//! # Safety
//!
//! Grid synchronization primitives are `unsafe` because they impose strict requirements:
//! - Must use cooperative kernel launch
//! - All participating blocks must reach the sync point
//! - Memory ordering must be correctly managed
//! - Deadlock possible if used incorrectly
//!
//! # Example (Conceptual)
//!
//! ```ignore
//! use cuda_std::cooperative_groups::intrinsics::{sync_grids_arrive, sync_grids_wait};
//!
//! #[kernel]
//! pub unsafe fn multi_phase_kernel(data: *mut u32, arrived: *mut u32) {
//!     // Phase 1: Process data
//!     *data.add(thread::index()) = thread::index() as u32;
//!
//!     // Grid-wide synchronization
//!     let old_arrive = sync_grids_arrive(arrived);
//!     sync_grids_wait(old_arrive, arrived);
//!
//!     // Phase 2: All blocks see all Phase 1 results
//!     let neighbor_data = *data.add((thread::index() + 1) % total_threads);
//! }
//! ```

pub mod intrinsics;
