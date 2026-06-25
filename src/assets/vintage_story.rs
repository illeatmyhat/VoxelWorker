//! Vintage Story install detection + scanning (the first [`BlockSource`]).
//!
//! [`VintageStoryDetector`] locates VS installs on this OS with no user action
//! (DATA.md "VS install locations"): the Windows `%APPDATA%\Vintagestory\assets`
//! path is the one actually testable here; the Linux / Flatpak / macOS paths are
//! probed best-effort so the same binary auto-detects on those platforms too.
//!
//! Each detected install yields one [`VintageStorySource`] per asset domain that
//! has a `textures/block` directory (`survival`, `game`, `creative` — `survival`
//! holds the bulk). [`VintageStorySource::scan`] walks that dir with `walkdir`,
//! applies [`is_chiselable`](super::is_chiselable), and groups variants via
//! [`group_block_textures`](super::group_block_textures).

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use super::{
    group_block_textures, is_chiselable, BlockGroup, BlockSource, ScannedTexture, SourceDetector,
    MAX_TEXTURES_WALKED,
};

/// Asset domains scanned, in priority order (DATA.md: `survival` has the bulk).
const ASSET_DOMAINS: &[&str] = &["survival", "game", "creative"];

/// Detector for Vintage Story installs.
pub struct VintageStoryDetector;

impl SourceDetector for VintageStoryDetector {
    fn detect(&self) -> Vec<Box<dyn BlockSource>> {
        let mut sources: Vec<Box<dyn BlockSource>> = Vec::new();
        for assets_root in candidate_assets_roots() {
            for domain in ASSET_DOMAINS {
                let block_dir = assets_root.join(domain).join("textures").join("block");
                if block_dir.is_dir() {
                    sources.push(Box::new(VintageStorySource::new(block_dir, domain)));
                }
            }
        }
        sources
    }
}

/// All plausible `…/Vintagestory/assets` roots on this OS that actually exist.
///
/// Windows is the only path testable in this environment; the others are probed
/// best-effort (DATA.md) so the auto-detect works unchanged on Linux/macOS.
fn candidate_assets_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut push_if_exists = |path: PathBuf| {
        if path.is_dir() && !roots.contains(&path) {
            roots.push(path);
        }
    };

    // --- Windows: %APPDATA%\Vintagestory\assets ---
    if let Ok(appdata) = std::env::var("APPDATA") {
        push_if_exists(Path::new(&appdata).join("Vintagestory").join("assets"));
    }
    // Custom Windows install dir, if the env var points at one.
    if let Ok(install_dir) = std::env::var("VINTAGE_STORY") {
        push_if_exists(Path::new(&install_dir).join("assets"));
    }

    // --- Linux / macOS: derive from $HOME ---
    if let Ok(home) = std::env::var("HOME") {
        let home = Path::new(&home);
        // Linux config / data dirs.
        push_if_exists(home.join(".config").join("Vintagestory").join("assets"));
        push_if_exists(home.join(".local").join("share").join("vintagestory").join("assets"));
        // Flatpak.
        push_if_exists(
            home.join(".var")
                .join("app")
                .join("at.vintagestory.VintageStory")
                .join("current")
                .join("active")
                .join("files")
                .join("extra")
                .join("vintagestory")
                .join("assets"),
        );
        // macOS app bundle (Show Package Contents).
        push_if_exists(
            home.join("Applications")
                .join("Vintagestory.app")
                .join("Contents")
                .join("assets"),
        );
    }
    // System-wide Linux installs (Arch AUR / Debian).
    push_if_exists(Path::new("/opt/vintagestory/assets").to_path_buf());
    push_if_exists(Path::new("/usr/share/vintagestory/assets").to_path_buf());
    // macOS system Applications.
    push_if_exists(
        Path::new("/Applications/Vintagestory.app/Contents/assets").to_path_buf(),
    );

    roots
}

/// A scanned VS asset domain (its `textures/block` directory).
pub struct VintageStorySource {
    block_dir: PathBuf,
    display_name: String,
}

impl VintageStorySource {
    fn new(block_dir: PathBuf, domain: &str) -> Self {
        Self {
            block_dir,
            display_name: format!("Vintage Story ({domain})"),
        }
    }
}

impl BlockSource for VintageStorySource {
    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn scan(&self) -> Vec<BlockGroup> {
        let textures = scan_block_dir(&self.block_dir);
        group_block_textures(textures)
    }
}

/// Recursively walk a `textures/block` directory and collect the chiselable PNGs
/// (prototype `scanBlocks` walk). The `relative_path` is forward-slashed and
/// taken below `block_dir` so the substring filter behaves like the prototype.
pub(super) fn scan_block_dir(block_dir: &Path) -> Vec<ScannedTexture> {
    let mut textures = Vec::new();
    let mut walked = 0usize;
    for entry in WalkDir::new(block_dir).follow_links(false).into_iter().flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let is_png = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"));
        if !is_png {
            continue;
        }
        walked += 1;
        if walked > MAX_TEXTURES_WALKED {
            break;
        }
        let Some(relative_path) = relative_forward_slash(block_dir, path) else {
            continue;
        };
        if is_chiselable(&relative_path) {
            textures.push(ScannedTexture {
                absolute_path: path.to_path_buf(),
                relative_path,
            });
        }
    }
    textures
}

/// Path of `full` relative to `base`, as a forward-slashed lowercase-safe string.
fn relative_forward_slash(base: &Path, full: &Path) -> Option<String> {
    let relative = full.strip_prefix(base).ok()?;
    let mut joined = String::new();
    for component in relative.components() {
        if !joined.is_empty() {
            joined.push('/');
        }
        joined.push_str(&component.as_os_str().to_string_lossy());
    }
    Some(joined)
}
