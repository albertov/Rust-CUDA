fn main() {
    let cuda_root = find_cuda_helper::find_cuda_root().expect("CUDA root not found");

    let cuda_version = get_cuda_version(&cuda_root).expect("Failed to retrieve CUDA version");
    println!("Parsed CUDA version as semver: {}", cuda_version);

    let conditional_node_version = semver::VersionReq::parse(">=12.3.0").unwrap();
    if conditional_node_version.matches(&cuda_version) {
        println!("cargo:rustc-cfg=conditional_node");
    }
}

fn get_cuda_version(cuda_root: &std::path::Path) -> Option<semver::Version> {
    let versions_path = cuda_root.join("version.json");
    let versions_data = std::fs::read_to_string(&versions_path).ok()?;
    let versions_json: serde_json::Value = serde_json::from_str(&versions_data).ok()?;
    let cuda_version_str = versions_json["cuda"]["version"].as_str()?;
    semver::Version::parse(cuda_version_str).ok()
}
