//! Common test infrastructure for comparative validation
//!
//! This module provides shared utilities for testing Rust CUDA implementations
//! against C++ reference implementations. It enables hermetic testing by embedding
//! PTX files at compile-time and provides helpers for launching kernels and
//! comparing outputs.

use std::fmt::Debug;

// Re-export cust types for convenience
pub use cust::device::{Device, DeviceAttribute};
#[allow(unused_imports)]
pub use cust::launch_cooperative;
pub use cust::link::Linker;
pub use cust::prelude::*;

// Embed PTX files at compile time for hermetic testing
// All PTX files are built by build.rs before tests run

/// Rust implementation of cooperative groups kernels
#[allow(dead_code)]
pub static RUST_PTX: &str = include_str!("../../../../../target/cuda/cuda_std_cg_test_kernels.ptx");

/// C++ reference implementation - basic grid synchronization test
#[allow(dead_code)]
pub static CPP_BASIC_PTX: &str = include_str!("../../../../../target/cuda/grid_sync_basic.ptx");

/// C++ reference implementation - multi-block synchronization test
#[allow(dead_code)]
pub static CPP_MULTIBLOCK_PTX: &str =
    include_str!("../../../../../target/cuda/grid_sync_multiblock.ptx");

/// C++ reference implementation - phased synchronization test
#[allow(dead_code)]
pub static CPP_PHASES_PTX: &str = include_str!("../../../../../target/cuda/grid_sync_phases.ptx");

/// C++ reference implementation - coalesced group tests
#[allow(dead_code)]
pub static CPP_COALESCED_PTX: &str = include_str!("../cpp_reference/coalesced_tests.ptx");

/// Load the Rust CUDA module containing cooperative groups implementations
///
/// Uses device linker to handle any external symbols or RDC requirements.
///
/// # Returns
/// - `Ok(Module)` - Loaded CUDA module
/// - `Err(...)` - Module loading error
#[allow(dead_code)]
pub fn load_rust_module() -> Result<Module, Box<dyn std::error::Error>> {
    // EXPERIMENT: Try loading PTX directly without linker first
    eprintln!("Attempting direct PTX load without linker...");
    match Module::from_ptx(RUST_PTX, &[]) {
        Ok(module) => {
            eprintln!("SUCCESS: Direct PTX load worked!");
            return Ok(module);
        }
        Err(e) => {
            eprintln!("Direct PTX load failed: {:?}", e);
            eprintln!("Falling back to linker approach...");
        }
    }

    // Use linker to handle potential RDC (relocatable device code) requirements
    let mut linker = Linker::new().map_err(|e| {
        eprintln!("Failed to create linker for Rust PTX:");
        eprintln!("Error: {:?}", e);
        e
    })?;

    linker.add_ptx(RUST_PTX).map_err(|e| {
        eprintln!("Failed to add Rust PTX to linker:");
        eprintln!("Error: {:?}", e);
        eprintln!("PTX preview (first 500 chars):");
        eprintln!("{}", &RUST_PTX[..500.min(RUST_PTX.len())]);
        e
    })?;

    let cubin = linker.complete().map_err(|e| {
        eprintln!("Failed to complete linking of Rust PTX:");
        eprintln!("Error: {:?}", e);
        e
    })?;

    let module = Module::from_cubin(&cubin, &[])?;
    Ok(module)
}

/// Load a C++ reference CUDA module by test name
///
/// Loads PTX directly. External symbols (.extern, .weak) are resolved at runtime.
///
/// # Arguments
/// - `test_name` - Name of the test ("basic", "multiblock", "phases", or "coalesced")
///
/// # Returns
/// - `Ok(Module)` - Loaded CUDA module
/// - `Err(...)` - Module loading error or invalid test name
#[allow(dead_code)]
pub fn load_cpp_module(test_name: &str) -> Result<Module, Box<dyn std::error::Error>> {
    let ptx = match test_name {
        "basic" => CPP_BASIC_PTX,
        "multiblock" => CPP_MULTIBLOCK_PTX,
        "phases" => CPP_PHASES_PTX,
        "coalesced" => CPP_COALESCED_PTX,
        _ => {
            return Err(format!("Unknown test name: {}", test_name).into());
        }
    };

    // Load PTX directly - external symbols resolved by CUDA driver at runtime
    eprintln!("Loading C++ PTX for test: {}", test_name);
    let module = Module::from_ptx(ptx, &[]).map_err(|e| {
        eprintln!("Failed to load C++ PTX ({}):", test_name);
        eprintln!("Error: {:?}", e);
        eprintln!("PTX preview (first 500 chars):");
        eprintln!("{}", &ptx[..500.min(ptx.len())]);
        e
    })?;

    Ok(module)
}

/// Verify device supports cooperative kernel launch
///
/// Queries the CUDA device for the CooperativeLaunch attribute. This must
/// be checked before attempting any cooperative kernel launches to avoid
/// hangs or undefined behavior.
///
/// # Returns
/// - `Ok(())` - Device supports cooperative launch
/// - `Err(...)` - Device does not support cooperative launch or query failed
#[allow(dead_code)]
pub fn check_cooperative_launch_support() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get_device(0)?;

    // Query CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH (= 95)
    // This attribute is not yet in the DeviceAttribute enum, so we transmute the value
    let coop_launch_attr: DeviceAttribute =
        unsafe { std::mem::transmute::<u32, DeviceAttribute>(95) };

    let coop_launch = device.get_attribute(coop_launch_attr)?;

    if coop_launch == 0 {
        return Err(
            "Device does not support cooperative kernel launch (cudaDevAttrCooperativeLaunch = 0)"
                .into(),
        );
    }

    println!("Device supports cooperative launch: YES");
    Ok(())
}

/// Compare outputs from Rust and C++ kernel executions
///
/// This is the critical assertion that validates the Rust port matches
/// the C++ reference implementation exactly.
///
/// # Arguments
/// - `rust` - Output from Rust kernel
/// - `cpp` - Output from C++ kernel
/// - `test_name` - Name of test for error messages
///
/// # Returns
/// - `Ok(())` - Outputs match exactly
/// - `Err(...)` - Outputs differ
///
/// # Type Parameters
/// - `T` - Element type, must support equality comparison and debug printing
#[allow(dead_code)]
pub fn compare_outputs<T>(
    rust: &[T],
    cpp: &[T],
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: PartialEq + Debug,
{
    if rust != cpp {
        return Err(format!(
            "Test '{}' FAILED: Rust and C++ kernels produced different outputs!\n\
             Rust output: {:?}\n\
             C++ output:  {:?}",
            test_name, rust, cpp
        )
        .into());
    }

    println!("Test '{}' PASSED: Rust and C++ outputs match", test_name);
    Ok(())
}
