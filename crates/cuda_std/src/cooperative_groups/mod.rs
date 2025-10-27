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
//! # Usage Patterns
//!
//! ## High-Level API (Recommended)
//!
//! Use the `GridGroup` type for safe, ergonomic grid synchronization:
//!
//! ```no_run
//! use cuda_std::cooperative_groups::*;
//!
//! #[kernel]
//! pub unsafe fn iterative_solver(data: *mut f32) {
//!     let grid = this_grid();
//!
//!     for iteration in 0..MAX_ITERS {
//!         // Phase 1: local computation
//!         process_local_data(data);
//!
//!         // Sync all blocks before global phase
//!         grid.sync();
//!
//!         // Phase 2: global computation with consistent view
//!         update_global_state(data);
//!         grid.sync();
//!     }
//! }
//! ```
//!
//! ## Low-Level API (Advanced)
//!
//! Use the `intrinsics` module for direct control over synchronization primitives:
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
//!
//! # Implementation Notes
//!
//! ## Workspace Allocation via Environment Registers
//!
//! Grid synchronization requires a driver-allocated workspace containing barrier state.
//! Unlike the official CUDA C++ API which uses opaque runtime integration, this Rust
//! implementation accesses the workspace pointer via NVIDIA's environment registers:
//!
//! - `%envreg1`: High 32 bits of workspace pointer
//! - `%envreg2`: Low 32 bits of workspace pointer
//!
//! This approach was discovered through analysis of NVIDIA's cooperative_groups implementation
//! in CUDA 12.4.99. While not officially documented, it matches NVIDIA's internal mechanism
//! and has been validated on:
//!
//! - CUDA 12.4.99
//! - SM architectures: 60, 70, 80, 89
//! - GPUs: RTX 4090
//!
//! ## Compatibility and Validation
//!
//! **Important**: This implementation relies on undocumented NVIDIA driver behavior.
//! Users should validate the environment register approach works on their specific
//! CUDA version and GPU architecture before production deployment.
//!
//! If the workspace pointer is null (non-cooperative launch), [`GridGroup::is_valid()`]
//! returns false and [`GridGroup::sync()`] will panic with a clear error message.
//!
//! ## Testing
//!
//! Reference C++ tests using official `cooperative_groups` API are provided in
//! `tests/cuda_std_cg_tests/cpp_reference/` and demonstrate identical behavior.
//!
//! # Safety
//!
//! Grid synchronization has strict requirements:
//! - Must use cooperative kernel launch (`cudaLaunchCooperativeKernel`)
//! - All threads in all blocks must participate uniformly in sync operations
//! - Memory ordering must be correctly managed
//! - Deadlock possible if used incorrectly (divergent sync calls)
//!
//! The high-level `GridGroup` API encapsulates unsafe operations, providing
//! a safer interface. However, incorrect usage patterns (e.g., divergent sync
//! calls) can still cause deadlock.

pub mod intrinsics;
pub mod grid;

// Re-export public API for convenience
pub use grid::{GridGroup, this_grid};
