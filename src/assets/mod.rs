//! Pluggable block-texture sources (Milestone 6).
//!
//! The whole reason the native port exists: read the *real* Vintage Story
//! install, scan it for chiselable block textures, and surface them as a palette
//! of cube thumbnails the user can apply to the model.
//!
//! The design is deliberately pluggable so VS is just the first implementation:
//!
//!   * [`BlockGroup`] — one logical block (e.g. "Granite") with its grouped
//!     texture variants (`granite1.png`, `granite2.png`, …).
//!   * [`BlockSource`] — a scanned game install / texture pack. `scan` returns its
//!     [`BlockGroup`]s. (Milestone 7 will add `resolve_faces` here, mapping a
//!     group to its per-face textures via the block JSON.)
//!   * [`SourceDetector`] — locates installs of *one* game on this OS with no user
//!     action (the auto-detect path). Returns boxed [`BlockSource`]s.
//!   * [`registry`] runs every known detector and aggregates the sources.
//!
//! [`vintage_story`] is the first detector/source; [`custom_pack`] handles an
//! arbitrary folder the user points the OS picker at (the only user-action path).
//!
//! The chiselable filter + variant grouping (the ALLOW / EXCLUDE lists, the
//! "anything under a `/rock/` segment" accept, the trailing-digit group key and
//! Title-Case label) are transcribed from the browser prototype's
//! `isChiselable` / `scanBlocks` / `prettify` (chisel-bench-reference.html) into
//! [`is_chiselable`] / [`group_block_textures`] / [`prettify_label`] below.

use std::path::PathBuf;

pub mod custom_pack;
pub mod faces;
pub mod registry;
pub mod vintage_story;

pub use faces::{CubeFaceSlot, FaceProvenance, FaceTextures};

/// Hard cap on the number of [`BlockGroup`]s a scan returns (prototype `slice(0,90)`).
pub const MAX_BLOCK_GROUPS: usize = 90;

/// Safety cap on the number of PNGs walked per source (prototype `n>8000`).
pub const MAX_TEXTURES_WALKED: usize = 8000;

/// One logical chiselable block and its grouped texture variants.
///
/// Variants are the same block with a trailing-digit suffix (`granite1.png`,
/// `granite2.png`); applying the block picks one pseudo-randomly.
#[derive(Debug, Clone)]
pub struct BlockGroup {
    /// Title-Case display label (last path segment, `-`/`_` → space).
    pub label: String,
    /// Group key: the texture path without `.png`, trailing digits stripped.
    /// Stable identity used to de-duplicate and (M7) resolve the block JSON.
    pub key: String,
    /// Absolute paths to every texture variant in this group (≥ 1).
    pub variants: Vec<PathBuf>,
}

/// A scanned source of chiselable block textures (a game install or texture pack).
///
/// Milestone 7 will extend this with `resolve_faces(&self, group) ->
/// FaceTextures` so a block can be drawn with different top/side textures instead
/// of one texture on all six faces.
pub trait BlockSource: Send {
    /// Human-readable source name shown in the status line (e.g.
    /// "Vintage Story (survival)").
    fn display_name(&self) -> &str;

    /// Walk this source and return its grouped chiselable blocks (capped,
    /// label-sorted).
    fn scan(&self) -> Vec<BlockGroup>;

    /// Resolve a group's per-face textures (Milestone 7).
    ///
    /// `chosen_variant` is the specific PNG the palette picked for this apply
    /// (so `{rock}`/`{wood}` placeholders resolve to the right material and any
    /// face the blocktype doesn't cover falls back to it). Implementations look
    /// up the matching blocktype JSON and map each cube face to a PNG; the
    /// default returns a uniform mapping (the M6 single-texture behaviour), which
    /// is also the graceful fallback when no blocktype matches.
    fn resolve_faces(&self, _group: &BlockGroup, chosen_variant: &std::path::Path) -> FaceTextures {
        FaceTextures::uniform(chosen_variant.to_path_buf())
    }
}

/// Locates installs of *one* game on this OS, with no user action.
///
/// Each detector knows the conventional install locations for its game on every
/// platform and returns a [`BlockSource`] per install it actually finds.
pub trait SourceDetector {
    /// Return a boxed [`BlockSource`] for every install found (empty if none).
    fn detect(&self) -> Vec<Box<dyn BlockSource>>;
}

/// The chiselable ALLOW list — lowercase substrings matched against the texture
/// path (DATA.md "Scanning for chiselable blocks" / prototype `CHISEL_NAMES`).
/// All vanilla rock types plus the worked-stone / plank families.
pub const ALLOW_SUBSTRINGS: &[&str] = &[
    "granite",
    "andesite",
    "basalt",
    "chalk",
    "chert",
    "claystone",
    "conglomerate",
    "diorite",
    "dolomite",
    "dolostone",
    "sandstone",
    "limestone",
    "shale",
    "slate",
    "peridotite",
    "phyllite",
    "suevite",
    "kimberlite",
    "bauxite",
    "halite",
    "graphite",
    "obsidian",
    "marble",
    "quartzite",
    "gneiss",
    "breccia",
    "greenschist",
    "plank",
    "ashlar",
    "polished",
    "stonebrick",
    "claybrick",
    "cobblestone",
    "drystone",
];

/// The EXCLUDE list — lowercase substrings that disqualify a path *before* the
/// ALLOW / `/rock/` checks (prototype `EXCLUDE`).
///
/// The prototype tokens (`/ore`, `gravel`, `overlay`, `/soil`, `crack`,
/// `_n.png`, `-n.png`, `glow`) drop ores, gravel, decals, soil, cracked
/// overlays, normal-maps and glow maps. Two tokens are ADDED after testing
/// against the real VS 1.22.3 tree (see the M6 report): `metal/` (the `chalk`
/// ALLOW substring matches `molybdochalkos` ingots/lanterns — not chiselable
/// rock) and `painting/` (`painting/caveart/chalk`). They cleanly remove the
/// only non-stone/non-wood junk the scan otherwise pulls. (No leading slash:
/// these are top-level segments of the path *relative to* `textures/block`.)
pub const EXCLUDE_SUBSTRINGS: &[&str] = &[
    "/ore", "gravel", "overlay", "/soil", "crack", "_n.png", "-n.png", "glow", "metal/",
    "painting/",
];

/// Is the texture at `relative_path` a chiselable block? (prototype `isChiselable`)
///
/// `relative_path` is the path *below* the `textures/block` directory, using
/// forward slashes (so the `/rock/` and `/ore` segment tests work regardless of
/// the OS path separator). The matching is a case-insensitive substring test.
///
/// Order matters: EXCLUDE wins first, then anything under a `/rock/` segment is
/// always accepted, then the ALLOW list.
pub fn is_chiselable(relative_path: &str) -> bool {
    let lowercased = relative_path.to_ascii_lowercase();
    if EXCLUDE_SUBSTRINGS.iter().any(|token| lowercased.contains(token)) {
        return false;
    }
    if lowercased.contains("/rock/") {
        return true;
    }
    ALLOW_SUBSTRINGS.iter().any(|token| lowercased.contains(token))
}

/// Strip the `.png` extension and any trailing digits from a forward-slash
/// relative path, yielding the group key (prototype `key` in `scanBlocks`).
fn group_key_for(relative_path: &str) -> String {
    let without_extension = relative_path
        .strip_suffix(".png")
        .or_else(|| relative_path.strip_suffix(".PNG"))
        .unwrap_or(relative_path);
    let trimmed = without_extension.trim_end_matches(|character: char| character.is_ascii_digit());
    trimmed.to_string()
}

/// Title-Case the last path segment, with `-`/`_` → space (prototype `prettify`).
pub fn prettify_label(group_key: &str) -> String {
    let last_segment = group_key.rsplit('/').next().unwrap_or(group_key);
    let spaced: String = last_segment
        .chars()
        .map(|character| if character == '-' || character == '_' { ' ' } else { character })
        .collect();
    // Title Case: upper-case the first letter of every whitespace-delimited word.
    let mut result = String::with_capacity(spaced.len());
    let mut at_word_start = true;
    for character in spaced.trim().chars() {
        if character.is_whitespace() {
            at_word_start = true;
            result.push(character);
        } else if at_word_start {
            result.extend(character.to_uppercase());
            at_word_start = false;
        } else {
            result.push(character);
        }
    }
    result
}

/// One scanned PNG: its absolute path plus the forward-slash path relative to the
/// `textures/block` dir (used for filtering + key derivation).
pub struct ScannedTexture {
    pub absolute_path: PathBuf,
    pub relative_path: String,
}

/// Group scanned chiselable textures into [`BlockGroup`]s, transcribing the
/// prototype `scanBlocks` grouping with one documented deviation at the cap step.
///
/// Faithful to the prototype: GROUP KEY = relative path without `.png`, trailing
/// digits stripped; LABEL = Title-Cased last segment; variants accumulate per
/// key; the result is label-sorted and capped at [`MAX_BLOCK_GROUPS`].
///
/// **Deviation (reported):** the prototype keys by the *full* path, so the real
/// VS tree yields ~444 groups of which the alphabetically-first 90 are only ~18
/// distinct materials (dozens of near-identical "Andesite" tiles from different
/// brick/cobble sub-dirs). To make the 90-tile palette useful we de-duplicate by
/// LABEL at the cap, merging variants, and prefer the cleaner `/rock/` base
/// texture as the representative key. The ALLOW/EXCLUDE/key/label rules
/// themselves are unchanged.
pub fn group_block_textures(textures: Vec<ScannedTexture>) -> Vec<BlockGroup> {
    use std::collections::BTreeMap;

    // First pass: accumulate variants per full-path key (faithful key derivation).
    let mut by_key: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for texture in textures {
        let key = group_key_for(&texture.relative_path);
        by_key.entry(key).or_default().push(texture.absolute_path);
    }

    // Order keys so the cleaner representative wins when de-duping by label:
    // `/rock/` base textures first, then shallower paths, then lexicographic.
    let mut keyed: Vec<(String, Vec<PathBuf>)> = by_key.into_iter().collect();
    keyed.sort_by(|(left_key, _), (right_key, _)| {
        let left_is_rock = path_has_rock_segment(left_key);
        let right_is_rock = path_has_rock_segment(right_key);
        right_is_rock
            .cmp(&left_is_rock)
            .then_with(|| left_key.matches('/').count().cmp(&right_key.matches('/').count()))
            .then_with(|| left_key.cmp(right_key))
    });

    // De-duplicate by label, merging variants into the first (best) representative.
    let mut by_label: BTreeMap<String, BlockGroup> = BTreeMap::new();
    for (key, mut variants) in keyed {
        let label = prettify_label(&key);
        variants.sort();
        by_label
            .entry(label.clone())
            .and_modify(|group| group.variants.extend(variants.clone()))
            .or_insert(BlockGroup { label, key, variants });
    }

    // Final: label-sorted (BTreeMap already is), capped.
    by_label.into_values().take(MAX_BLOCK_GROUPS).collect()
}

/// Does the (lower-cased) key contain a `/rock/` path segment, or start with `rock/`?
fn path_has_rock_segment(key: &str) -> bool {
    let lowercased = key.to_ascii_lowercase();
    lowercased.contains("/rock/") || lowercased.starts_with("rock/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_and_rock_accept_chiselable_paths() {
        assert!(is_chiselable("stone/rock/granite1.png"));
        assert!(is_chiselable("stone/rock/limestone3.png")); // /rock/ accept
        assert!(is_chiselable("wood/planks/oak1.png")); // ALLOW "plank"
        assert!(is_chiselable("stone/drystone/andesite2.png")); // ALLOW "drystone"
    }

    #[test]
    fn exclude_wins_over_allow_and_rock() {
        // EXCLUDE runs first: ores/soil/gravel/overlays/normal-maps/glow are out
        // even under /rock/ or with an ALLOW token.
        // "/ore" matches a path segment, not a filename suffix (prototype token).
        assert!(!is_chiselable("stone/ore/granite.png"));
        assert!(!is_chiselable("overlay/rock/cracked1.png"));
        assert!(!is_chiselable("stone/rock/granite_n.png"));
        assert!(!is_chiselable("block/soil/limestone.png"));
        // Added tokens: metal ingots (chalk⊂molybdochalkos) and paintings.
        assert!(!is_chiselable("metal/ingot/molybdochalkos.png"));
        assert!(!is_chiselable("painting/caveart/chalk.png"));
    }

    #[test]
    fn group_key_strips_extension_and_trailing_digits() {
        assert_eq!(group_key_for("stone/rock/granite12.png"), "stone/rock/granite");
        assert_eq!(group_key_for("stone/rock/basalt.png"), "stone/rock/basalt");
    }

    #[test]
    fn prettify_title_cases_last_segment() {
        assert_eq!(prettify_label("stone/rock/sand-stone"), "Sand Stone");
        assert_eq!(prettify_label("wood/planks/oak_log"), "Oak Log");
        assert_eq!(prettify_label("granite"), "Granite");
    }

    #[test]
    fn grouping_dedupes_by_label_and_prefers_rock() {
        let textures = vec![
            ScannedTexture {
                absolute_path: "/x/stone/brick/granite1.png".into(),
                relative_path: "stone/brick/granite1.png".into(),
            },
            ScannedTexture {
                absolute_path: "/x/stone/rock/granite1.png".into(),
                relative_path: "stone/rock/granite1.png".into(),
            },
            ScannedTexture {
                absolute_path: "/x/stone/rock/granite2.png".into(),
                relative_path: "stone/rock/granite2.png".into(),
            },
        ];
        let groups = group_block_textures(textures);
        // One "Granite" group, representative key from /rock/, all 3 variants merged.
        let granite: Vec<_> = groups.iter().filter(|g| g.label == "Granite").collect();
        assert_eq!(granite.len(), 1);
        assert!(granite[0].key.contains("/rock/"));
        assert_eq!(granite[0].variants.len(), 3);
    }
}
