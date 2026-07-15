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
//! applies [`is_chiselable`], and groups variants via
//! [`group_block_textures`].

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use walkdir::WalkDir;

use super::faces::{parse_block_type, resolve_block_faces, FaceTextures, ParsedBlockType};
use super::{
    group_block_textures, is_chiselable, BlockGroup, BlockSource, ScannedTexture, SourceDetector,
    MAX_TEXTURES_WALKED,
};

/// Safety cap on the number of blocktype JSONs parsed when building the index.
const MAX_BLOCKTYPES_PARSED: usize = 8000;

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
    /// The install's `assets` root (parent of every domain), derived from
    /// `block_dir` (`<assets>/<domain>/textures/block`). Used to find blocktype
    /// JSONs and to resolve `domain:path` texture references (M7).
    assets_root: PathBuf,
    /// The block's own asset domain (`survival`/`game`/`creative`): the first
    /// domain a bare texture reference is resolved against (M7).
    domain: String,
    display_name: String,
    /// Lazily-built blocktype index (M7): texture-stem directory → parsed
    /// blocktypes that reference it. Walking + parsing every blocktype is done
    /// once on the first `resolve_faces` call and cached here.
    blocktype_index: OnceLock<BlockTypeIndex>,
}

impl VintageStorySource {
    fn new(block_dir: PathBuf, domain: &str) -> Self {
        // `block_dir` is `<assets>/<domain>/textures/block`; the assets root is
        // four levels up.
        let assets_root = block_dir
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| block_dir.clone());
        Self {
            block_dir,
            assets_root,
            domain: domain.to_string(),
            display_name: format!("Vintage Story ({domain})"),
            blocktype_index: OnceLock::new(),
        }
    }

    /// Resolve a texture reference (`domain:path` or bare `path`) to an absolute
    /// PNG that actually exists. Tries the reference's domain (if prefixed), then
    /// the block's own domain, then `game`, then `survival` (M7 §3). The leading
    /// slash / `block/` prefix inconsistency (learned in M6) is tolerated.
    fn resolve_texture_path(&self, reference: &str) -> Option<PathBuf> {
        let (domain_opt, path) = split_domain(reference);
        // `textures/<path>.png` under each candidate domain, in priority order.
        let mut domains: Vec<&str> = Vec::new();
        if let Some(domain) = domain_opt.as_deref() {
            domains.push(domain);
        }
        domains.push(&self.domain);
        domains.push("game");
        domains.push("survival");

        for domain in domains {
            let candidate = self
                .assets_root
                .join(domain)
                .join("textures")
                .join(format!("{path}.png"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

    /// The cached blocktype index, built on first use.
    fn index(&self) -> &BlockTypeIndex {
        self.blocktype_index
            .get_or_init(|| BlockTypeIndex::build(&self.assets_root, &self.domain))
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

    fn resolve_faces(&self, group: &BlockGroup, chosen_variant: &Path) -> FaceTextures {
        // The group key (e.g. `stone/rock/granite`) is the texture stem the
        // blocktype `base` entries reference.
        let wanted_stem = group.key.as_str();
        if let Some(block) = self.index().lookup(wanted_stem) {
            return resolve_block_faces(block, wanted_stem, chosen_variant, &|reference| {
                self.resolve_texture_path(reference)
            });
        }
        // No match → uniform fallback (the M6 behaviour).
        FaceTextures::uniform(chosen_variant.to_path_buf())
    }
}

/// Split a texture reference into its optional domain prefix and the path.
/// `game:block/x` → (`Some("game")`, `block/x`); `block/x` → (`None`, `block/x`).
fn split_domain(reference: &str) -> (Option<String>, String) {
    let trimmed = reference.trim().trim_start_matches('/');
    if let Some(colon) = trimmed.find(':') {
        let domain = trimmed[..colon].to_string();
        let path = trimmed[colon + 1..].trim_start_matches('/').to_string();
        (Some(domain), path)
    } else {
        (None, trimmed.to_string())
    }
}

/// An index from a texture-stem directory (e.g. `stone/rock`) to the parsed
/// blocktypes whose `base` entries reference it. Built once per source.
struct BlockTypeIndex {
    /// Every parsed blocktype, kept alive so lookups can return a reference.
    blocks: Vec<ParsedBlockType>,
    /// directory-of-stem → indices into `blocks` that reference that directory.
    by_directory: std::collections::HashMap<String, Vec<usize>>,
}

impl BlockTypeIndex {
    /// Walk every domain's `blocktypes/**.json`, parse what M7 needs, and index
    /// each block by the directories its texture `base` entries reference.
    fn build(assets_root: &Path, primary_domain: &str) -> Self {
        let mut blocks: Vec<ParsedBlockType> = Vec::new();
        let mut by_directory: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        // Scan the block's own domain first, then the others (so a block in the
        // primary domain wins ties). De-duplicate domains.
        let mut domains: Vec<String> = vec![primary_domain.to_string()];
        for domain in ["survival", "game", "creative"] {
            if !domains.iter().any(|d| d == domain) {
                domains.push(domain.to_string());
            }
        }

        let mut parsed = 0usize;
        'outer: for domain in domains {
            let blocktypes_dir = assets_root.join(&domain).join("blocktypes");
            if !blocktypes_dir.is_dir() {
                continue;
            }
            for entry in WalkDir::new(&blocktypes_dir)
                .follow_links(false)
                .into_iter()
                .flatten()
            {
                if !entry.file_type().is_file() {
                    continue;
                }
                let is_json = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("json"));
                if !is_json {
                    continue;
                }
                parsed += 1;
                if parsed > MAX_BLOCKTYPES_PARSED {
                    break 'outer;
                }
                let Ok(raw) = std::fs::read_to_string(entry.path()) else {
                    continue;
                };
                let Some(block) = parse_block_type(&raw) else {
                    continue;
                };
                let block_index = blocks.len();
                let mut directories: Vec<String> = block
                    .referenced_bases()
                    .iter()
                    .filter_map(|base| directory_of_reference(base))
                    .collect();
                directories.sort();
                directories.dedup();
                for directory in directories {
                    by_directory.entry(directory).or_default().push(block_index);
                }
                blocks.push(block);
            }
        }

        Self { blocks, by_directory }
    }

    /// Find the best blocktype that references the directory of `wanted_stem`.
    ///
    /// Many blocks may reference a popular directory like `stone/rock` (the rock
    /// block itself, but also an anvil part that happens to texture with rock), so
    /// we score candidates and pick the highest: a block whose base IS the
    /// directory glob (`stone/rock/{rock}*` → the canonical full-block texture)
    /// outranks one that references a specific file within it, and a `code` that
    /// shares the directory's leaf name (e.g. `rock`) breaks ties. This keeps the
    /// common rock/brick cases on the right (uniform) block and avoids latching
    /// onto an unrelated per-face block by accident.
    fn lookup(&self, wanted_stem: &str) -> Option<&ParsedBlockType> {
        let directory = directory_of_reference(wanted_stem)?;
        let leaf = directory.rsplit('/').next().unwrap_or(&directory);
        let candidates = self.by_directory.get(&directory)?;
        candidates
            .iter()
            .max_by_key(|&&index| {
                let block = &self.blocks[index];
                let mut score = 0i32;
                // An exact `code == leaf` (e.g. `rock`, `drystone`) is the
                // canonical block for this directory — weight it heavily so it
                // beats unrelated blocks (anvil parts, querns) that merely
                // texture with the rock and reference the directory many times.
                if block.code == leaf {
                    score += 100;
                } else if block.code.starts_with(leaf) || leaf.starts_with(block.code.as_str()) {
                    score += 20;
                }
                // Presence (not count) of a `<dir>/{glob}` whole-block reference.
                if block.referenced_bases().iter().any(|base| {
                    directory_of_reference(base).as_deref() == Some(directory.as_str())
                        && (base.contains('{') || base.contains('*'))
                }) {
                    score += 5;
                }
                score
            })
            .map(|&index| &self.blocks[index])
    }
}

/// The cleaned directory portion of a texture reference, for indexing/lookup
/// (`block/stone/rock/{rock}*` and `stone/rock/granite` both → `stone/rock`).
///
/// A reference carrying a `{placeholder}`/`*` glob has the glob in the *filename*
/// position, so once stripped the remaining path IS the directory. A clean file
/// reference (`stone/rock/granite`) takes its parent directory.
fn directory_of_reference(reference: &str) -> Option<String> {
    let (_, path) = split_domain(reference);
    let mut path = path.strip_prefix("block/").unwrap_or(&path).to_string();
    let had_glob = path.contains('{') || path.contains('*');
    if let Some(brace) = path.find('{') {
        path.truncate(brace);
    }
    if let Some(star) = path.find('*') {
        path.truncate(star);
    }
    let path = path.trim_end_matches('/');
    if path.is_empty() {
        return None;
    }
    if had_glob {
        // The glob was the filename; what remains is the directory.
        Some(path.to_string())
    } else {
        // A clean file path: take its parent directory.
        let slash = path.rfind('/')?;
        Some(path[..slash].to_string())
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
