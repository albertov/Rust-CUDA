#![feature(exit_status_error)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=rust_kernels");
    println!("cargo::rerun-if-changed=cpp_reference");

    // CRITICAL FIX: Skip kernel compilation during normal builds to avoid circular dependency.
    // Kernels are only compiled when explicitly requested via CUDA_STD_BUILD_KERNELS=1.
    // This breaks the infinite recursion: cuda_std build.rs -> rust_kernels -> cuda_std -> ...
    //
    // Tests should set CUDA_STD_BUILD_KERNELS=1 before running to ensure PTX files exist.
    let should_build_kernels = env::var("CUDA_STD_BUILD_KERNELS").is_ok();

    if !should_build_kernels {
        println!("cargo:warning=Skipping test kernel compilation (set CUDA_STD_BUILD_KERNELS=1 to build)");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Navigate from tests/cuda_std_cg/ -> tests/ -> Rust-CUDA/
    let rust_cuda_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to find Rust-CUDA root");

    // Navigate from Rust-CUDA/ -> vendor/ -> workspace/
    let workspace_root = rust_cuda_root
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to find workspace root");

    let target_dir = workspace_root.join("target/cuda");
    std::fs::create_dir_all(&target_dir).unwrap();
    let dest = target_dir.join("cuda_std_cg_test_kernels.ptx");
    println!("cargo::rerun-if-changed={:?}", dest);

    // Compile Rust kernels via CudaBuilder
    // Target SM 8.0 to match C++ reference kernels
    // (SM 8.0 includes all sm_70 features plus newer reduction instructions)
    CudaBuilder::new(manifest_dir.join("rust_kernels"))
        .arch(NvvmArch::Compute80)
        .copy_to(dest)
        .build()
        .unwrap();

    // Compile C++ reference kernels via nvcc (each to separate PTX)
    let cpp_ref_dir = manifest_dir.join("cpp_reference");
    let cpp_kernels = [
        "grid_sync_basic.cu",
        "grid_sync_multiblock.cu",
        "grid_sync_phases.cu",
    ];

    for kernel in &cpp_kernels {
        let src = cpp_ref_dir.join(kernel);
        let dest = target_dir.join(kernel.replace(".cu", ".ptx"));
        println!("cargo::rerun-if-changed={:?}", dest);

        Command::new("nvcc")
            .args([
                "-arch",
                "compute_80", // SM 8.0 to match Rust kernels
                "-rdc=true",  // REQUIRED for cooperative_groups
                "--device-c", // Generate device-linkable code
                "-std=c++17",
                "-ptx",
                "--keep-device-functions", // Preserve __device__ functions in PTX
                "-I",
                cpp_ref_dir.to_str().unwrap(),
                "-o",
                dest.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .status()
            .expect("failed to find nvcc")
            .exit_ok()
            .expect("failed to execute nvcc");
    }
}
