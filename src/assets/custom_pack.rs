//! Custom-folder block source — the user-action fallback (Milestone 6).
//!
//! When auto-detection finds nothing (or the user wants a texture pack / a folder
//! the detectors don't know), the "Connect folder…" button opens the OS folder
//! picker (`rfd`) and points a [`CustomFolderSource`] at the chosen directory.
//!
//! It is a generic PNG-folder source: it walks the folder recursively for PNGs
//! and groups them with the *same* chiselable filter + variant grouping as the VS
//! source, so a pack laid out like `textures/block/**` behaves identically. The
//! relative path used for filtering/keying is taken below the picked folder.

use std::path::{Path, PathBuf};

use super::{group_block_textures, BlockGroup, BlockSource};
use super::vintage_story::scan_block_dir;

/// A block source rooted at an arbitrary folder the user picked.
pub struct CustomFolderSource {
    folder: PathBuf,
    display_name: String,
}

impl CustomFolderSource {
    /// Point a source at `folder`. If `folder` (or a descendant) is itself a
    /// `textures/block` tree, paths below it still filter correctly because the
    /// chiselable test is a substring match.
    pub fn new(folder: PathBuf) -> Self {
        let leaf = folder
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| folder.to_string_lossy().into_owned());
        Self {
            display_name: format!("Custom folder ({leaf})"),
            folder,
        }
    }

    /// The picked folder.
    pub fn folder(&self) -> &Path {
        &self.folder
    }
}

impl BlockSource for CustomFolderSource {
    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn scan(&self) -> Vec<BlockGroup> {
        // Reuse the VS walk: same PNG discovery, same chiselable filter, with the
        // relative path taken below the picked folder.
        let textures = scan_block_dir(&self.folder);
        group_block_textures(textures)
    }
}
