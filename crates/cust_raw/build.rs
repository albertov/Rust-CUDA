fn main() {
    let cuda_root = find_cuda_helper::find_cuda_root().expect("Has CUDA root");
    find_cuda_helper::include_cuda();

    let include_path = cuda_root.join("include");

    let output_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-I", &include_path.display().to_string()])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_type("^CU.*")
        .allowlist_type("^cuuint(32|64)_t")
        .allowlist_type("^cudaError_enum")
        .allowlist_type("^cu.*Complex$")
        .allowlist_type("^cuda.*")
        .allowlist_type("^libraryPropertyType.*")
        .allowlist_var("^CU.*")
        .allowlist_function("^cu.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .generate_comments(false)
        // .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(output_path.join("cuda.rs"))
        .expect("Unable to write bindings to file");
}
