//! Workspace source-size guard.
//!
//! Rust has no built-in file-length lint — `clippy::too_many_lines` counts a *function*, not
//! a file — so this build script supplies one. Any `.rs` file over [`LINE_LIMIT`] lines is
//! reported through `cargo:warning=`, which surfaces it in ordinary `cargo build` / `check` /
//! `test` output alongside compiler warnings.
//!
//! It lives in the ROOT package deliberately. Cargo only displays build-script warnings for
//! the package being built, so a guard placed in a workspace member would stay silent
//! whenever that member was compiled as a dependency of something else. The root package is
//! always the thing being built here, so its warnings always appear — and one script scanning
//! every crate beats eleven scripts each scanning one.
//!
//! **Why a warning and not an error.** The limit is a design smell, not a correctness
//! property: a 1,100-line file with one coherent responsibility is better than two 550-line
//! files that have to reach into each other. The warning starts the conversation; it does not
//! settle it.

use std::fs;
use std::path::{Path, PathBuf};

/// Lines past which a file is reported. A file this long is usually several subjects sharing
/// a filename rather than one subject that happens to be large.
const LINE_LIMIT: usize = 1000;

fn main() {
    let mut roots = vec![PathBuf::from("src")];
    if let Ok(entries) = fs::read_dir("crates") {
        for entry in entries.flatten() {
            let source_dir = entry.path().join("src");
            if source_dir.is_dir() {
                roots.push(source_dir);
            }
        }
    }

    let mut offenders: Vec<(PathBuf, usize)> = Vec::new();
    for root in &roots {
        // Re-run when anything under a source root changes, so a file that grows past the
        // limit is reported on the build that grew it.
        println!("cargo:rerun-if-changed={}", root.display());
        collect_offenders(root, &mut offenders);
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Longest first: the worst offender is the one worth carving next.
    offenders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (path, lines) in &offenders {
        println!(
            "cargo:warning={} is {lines} lines, over the {LINE_LIMIT}-line guard — consider \
             carving it into a folder module",
            path.display()
        );
    }
    if offenders.len() > 1 {
        println!(
            "cargo:warning={} files exceed the {LINE_LIMIT}-line guard",
            offenders.len()
        );
    }
}

/// Recurse a source root, recording every `.rs` file longer than the limit.
///
/// Only `src` directories are ever visited, so build artefacts and the stale worktree copies
/// under `.claude/` are out of reach by construction rather than by an exclusion list that
/// could drift.
fn collect_offenders(directory: &Path, offenders: &mut Vec<(PathBuf, usize)>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_offenders(&path, offenders);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            if let Ok(contents) = fs::read_to_string(&path) {
                let lines = contents.lines().count();
                if lines > LINE_LIMIT {
                    offenders.push((path, lines));
                }
            }
        }
    }
}
