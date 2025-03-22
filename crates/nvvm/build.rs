use std::env;
use std::path::PathBuf;

use find_cuda_helper::{find_cuda_root, find_libnvvm_bin_dir};

fn main() {
    let nvvm_header = {
        let mut cuda_root = find_cuda_root().expect("Has CUDA root");
        cuda_root.push("nvvm");
        cuda_root.push("include");
        cuda_root.push("nvvm.h");
        cuda_root.to_string_lossy().into_owned()
    };

    bindgen::Builder::default()
        .header(nvvm_header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("^nvvm.*")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("nvvm.rs"))
        .expect("Unable to write bindings to file");

    println!("cargo:rustc-link-search={}", find_libnvvm_bin_dir());
    println!("cargo:rustc-link-lib=dylib=nvvm");
}
