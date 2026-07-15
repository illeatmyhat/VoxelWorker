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
pub mod decode;
pub mod faces;
pub mod registry;
pub mod vintage_story;

pub use decode::{decode_rgba, DecodedRgba};
pub use faces::{CubeFaceSlot, FaceProvenance, FaceTextures};

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

/// Group scanned chiselable textures into [`BlockGroup`]s, faithfully following
/// the prototype `scanBlocks` grouping — no artificial cap, no label de-dup.
///
/// GROUP KEY = relative texture path without `.png`, trailing digits stripped, so
/// each distinct texture-set is ONE tile and numbered variants of the same stem
/// (`granite1`, `granite2`, …) stay grouped under that stem. LABEL = Title-Cased
/// last segment. Every group that passes the chiselable ALLOW/EXCLUDE filter is
/// returned; the result is key-sorted (stable, deterministic) so the palette
/// order is reproducible across runs.
///
/// Earlier builds de-duplicated by LABEL and capped the result at 90 groups to
/// keep the palette small; that hid most of the chiselable blocks (two distinct
/// texture-sets that prettify to the same label were merged, and everything past
/// the 90th label was dropped). Both the cap and the label de-dup are gone: the
/// real VS tree yields a few hundred groups and the palette shows them all.
pub fn group_block_textures(textures: Vec<ScannedTexture>) -> Vec<BlockGroup> {
    use std::collections::BTreeMap;

    // Accumulate variants per full-path key (the faithful prototype key): one
    // group per distinct texture-set, numbered variants of a stem merged in.
    let mut by_key: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for texture in textures {
        let key = group_key_for(&texture.relative_path);
        by_key.entry(key).or_default().push(texture.absolute_path);
    }

    // One BlockGroup per key, variants sorted for deterministic variant picking.
    // BTreeMap iteration is already key-sorted, so the palette order is stable.
    by_key
        .into_iter()
        .map(|(key, mut variants)| {
            variants.sort();
            let label = prettify_label(&key);
            BlockGroup { label, key, variants }
        })
        .collect()
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
    fn grouping_keeps_one_group_per_texture_set_and_merges_stem_variants() {
        let textures = vec![
            // Two numbered variants of the SAME stem → one group, two variants.
            ScannedTexture {
                absolute_path: "/x/stone/rock/granite1.png".into(),
                relative_path: "stone/rock/granite1.png".into(),
            },
            ScannedTexture {
                absolute_path: "/x/stone/rock/granite2.png".into(),
                relative_path: "stone/rock/granite2.png".into(),
            },
            // A DIFFERENT texture-set that prettifies to the same label "Granite"
            // must stay its OWN group now (no label de-dup).
            ScannedTexture {
                absolute_path: "/x/stone/brick/granite1.png".into(),
                relative_path: "stone/brick/granite1.png".into(),
            },
        ];
        let groups = group_block_textures(textures);
        // Two distinct keys → two groups, both labelled "Granite".
        assert_eq!(groups.len(), 2);
        let rock = groups
            .iter()
            .find(|g| g.key == "stone/rock/granite")
            .expect("rock granite group");
        assert_eq!(rock.variants.len(), 2, "stem variants merge into one group");
        assert!(groups.iter().any(|g| g.key == "stone/brick/granite"));
        assert!(groups.iter().all(|g| g.label == "Granite"));
    }

    #[test]
    fn grouping_has_no_artificial_cap() {
        // 200 distinct texture-sets must all survive (the old code capped at 90).
        // Use distinct non-numeric stems so trailing-digit stripping keeps them
        // separate (numbered variants of one stem would correctly merge).
        let textures: Vec<ScannedTexture> = (0..200)
            .map(|i| {
                let stem = format!("kind_{}", (b'a' + (i % 26) as u8) as char) + &i.to_string() + "x";
                ScannedTexture {
                    absolute_path: format!("/x/stone/rock/{stem}.png").into(),
                    relative_path: format!("stone/rock/{stem}.png"),
                }
            })
            .collect();
        let groups = group_block_textures(textures);
        assert_eq!(groups.len(), 200, "no 90-group cap");
    }
}
