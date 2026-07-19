//! Per-face block texture resolution (Milestone 7).
//!
//! The web prototype (and M6) put one texture on all six faces. Vintage Story
//! blocks can texture faces differently — a log shows end-grain on top/bottom and
//! bark on the sides; soil shows grass on top — and that per-face mapping lives in
//! the block's `assets/<domain>/blocktypes/**.json`.
//!
//! This module resolves a scanned [`BlockGroup`](super::BlockGroup) to a
//! [`FaceTextures`]: a PNG path for each of the six cube faces. It is the
//! native-only upgrade that is the main reason the port exists.
//!
//! ## What is handled
//!
//!   * `textures` and `texturesByType` maps (for `texturesByType`, the first
//!     variant block whose key references the wanted texture stem is used).
//!   * Face keys: `all` (all six), the explicit `up`/`down`/`north`/`south`/
//!     `east`/`west`, and the group shorthands `sides`/`horizontals` (the four
//!     horizontal faces) and `verticals` (up+down). Precedence: a specific face
//!     key overrides a group key overrides `all`.
//!   * `domain:path` → `assets/<domain>/textures/<path>.png`; a bare `path` is
//!     resolved against the block's own domain, then `game`, then `survival`,
//!     until a file exists (leading-slash tolerated like M6).
//!
//! ## What falls back to uniform (the M6 behaviour)
//!
//!   * No matching blocktype, or a parse failure (VS JSON is lenient — unquoted
//!     keys / trailing commas — and we pre-normalise it, but anything that still
//!     won't parse falls back).
//!   * `overlays` / tints / `rotation` (ignored for v1 — only `base` is read).
//!   * Unknown face keys (e.g. `inside-*` cut faces) are ignored.
//!   * Any face left unresolved inherits the group's own chosen variant PNG, so
//!     the result is always a complete six-face mapping.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The six cube faces, in the renderer's texture-array layer order
/// (slot 0..5 = East, West, Up, Down, South, North).
///
/// **Z-up world-axis mapping** (matches `cuboid_loaded.wgsl::face_layer` exactly):
/// the VERTICAL axis is Z, so `up` = +Z and `down` = -Z; the four horizontals are
/// `east` = +X, `west` = -X, `south` = -Y (the front face — front looks along +Y),
/// `north` = +Y (back). Exact compass orientation of the sides is not critical
/// Top-vs-side correctness — the `up` PNG on the +Z top, not a wall — is
/// what matters, and that is now correct under Z-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFaceSlot {
    East,
    West,
    Up,
    Down,
    South,
    North,
}

impl CubeFaceSlot {
    /// All six slots in texture-array layer order (0 East/+X, 1 West/-X, 2 Up/+Z,
    /// 3 Down/-Z, 4 South/-Y, 5 North/+Y).
    pub const ALL: [CubeFaceSlot; 6] = [
        CubeFaceSlot::East,
        CubeFaceSlot::West,
        CubeFaceSlot::Up,
        CubeFaceSlot::Down,
        CubeFaceSlot::South,
        CubeFaceSlot::North,
    ];

    /// The texture-array layer index (0..6) for this slot. Z-up: Up=+Z, Down=-Z,
    /// East=+X, West=-X, South=-Y, North=+Y.
    pub fn layer(self) -> usize {
        match self {
            CubeFaceSlot::East => 0,
            CubeFaceSlot::West => 1,
            CubeFaceSlot::Up => 2,
            CubeFaceSlot::Down => 3,
            CubeFaceSlot::South => 4,
            CubeFaceSlot::North => 5,
        }
    }
}

/// A resolved PNG path for each of the six cube faces.
///
/// `paths` is indexed by [`CubeFaceSlot::layer`]. A uniform block has the same
/// path in all six slots. [`is_uniform`](Self::is_uniform) reports whether the
/// block genuinely differs per face (used by `--list-perface`).
#[derive(Debug, Clone)]
pub struct FaceTextures {
    /// One absolute PNG path per face, in `CubeFaceSlot` layer order
    /// (East/+X, West/-X, Up/+Z, Down/-Z, South/-Y, North/+Y under Z-up).
    pub paths: [PathBuf; 6],
    /// How the mapping was obtained (for the `--list-perface` / report log).
    pub provenance: FaceProvenance,
}

/// Where a [`FaceTextures`] mapping came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaceProvenance {
    /// No blocktype matched (or it failed to parse / had no usable keys): every
    /// face is the group's own chosen variant PNG (the M6 single-texture path).
    UniformFallback,
    /// Resolved from a blocktype's `textures` map.
    Textures { block_code: String },
    /// Resolved from a blocktype's `texturesByType` map (one variant picked).
    TexturesByType { block_code: String, variant_key: String },
}

impl FaceTextures {
    /// A uniform mapping: the same `path` on all six faces (M6 behaviour).
    pub fn uniform(path: PathBuf) -> Self {
        Self {
            paths: [
                path.clone(),
                path.clone(),
                path.clone(),
                path.clone(),
                path.clone(),
                path,
            ],
            provenance: FaceProvenance::UniformFallback,
        }
    }

    /// Does every face resolve to the same PNG? `true` means a single-texture
    /// block (the common rock case); `false` means genuinely per-face.
    pub fn is_uniform(&self) -> bool {
        self.paths.iter().all(|p| *p == self.paths[0])
    }

    /// Whether the top (`up`) face differs from a representative side face — the
    /// most visible per-face distinction (`--list-perface` uses this).
    pub fn top_differs_from_side(&self) -> bool {
        let up = &self.paths[CubeFaceSlot::Up.layer()];
        let side = &self.paths[CubeFaceSlot::East.layer()];
        up != side
    }
}

// ===========================================================================
// VS blocktype JSON parsing.
// ===========================================================================

/// One texture entry. VS entries are objects `{ base: "...", overlays: [...] }`
/// OR (rarely) a bare string. We only read `base`; everything else is ignored.
#[derive(Debug, Clone)]
struct TextureEntry {
    base: String,
}

/// A `textures` / one-variant-of-`texturesByType` map: face-key → entry.
type TextureMap = BTreeMap<String, TextureEntry>;

/// A parsed blocktype, reduced to only what M7 needs: its `code`, an optional
/// `textures` map, and an optional `texturesByType` (variant-key → map).
#[derive(Debug, Clone)]
pub struct ParsedBlockType {
    pub code: String,
    textures: Option<TextureMap>,
    textures_by_type: Option<BTreeMap<String, TextureMap>>,
}

impl ParsedBlockType {
    /// Every `base` texture path this block references (across `textures` and all
    /// `texturesByType` variants). Used to index a blocktype by the stems it
    /// references so a scanned texture can find its block.
    pub fn referenced_bases(&self) -> Vec<String> {
        let mut bases = Vec::new();
        if let Some(map) = &self.textures {
            for entry in map.values() {
                bases.push(entry.base.clone());
            }
        }
        if let Some(by_type) = &self.textures_by_type {
            for map in by_type.values() {
                for entry in map.values() {
                    bases.push(entry.base.clone());
                }
            }
        }
        bases
    }
}

/// Normalise VS's lenient JSON into strict JSON `serde_json` can parse.
///
/// VS blocktype files use unquoted object keys and trailing commas (JSON5-ish).
/// We make a best-effort pass that strips `//` and `/* */` comments (rare but
/// legal), quotes bare identifier keys (`code:` → `"code":`), and removes
/// trailing commas before `}` / `]`.
///
/// String contents are preserved verbatim (we track whether we're inside a
/// string and skip transformations there). This is a pragmatic normaliser, not a
/// full JSON5 parser; anything it can't fix cleanly just fails to parse and the
/// caller falls back to uniform.
pub fn normalize_vs_json(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len() + 64);
    let mut i = 0usize;
    let mut in_string = false;
    let mut string_quote = b'"';

    while i < bytes.len() {
        let c = bytes[i];

        if in_string {
            out.push(c as char);
            if c == b'\\' && i + 1 < bytes.len() {
                // Copy the escaped character verbatim.
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == string_quote {
                in_string = false;
            }
            i += 1;
            continue;
        }

        // Not in a string.
        match c {
            b'"' | b'\'' => {
                // VS uses double quotes; tolerate single just in case by
                // emitting a double quote so serde_json accepts it.
                in_string = true;
                string_quote = c;
                out.push('"');
                i += 1;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Line comment: skip to end of line.
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                // Block comment: skip to closing */.
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            b',' => {
                // Drop a trailing comma: look ahead past whitespace/comments for
                // a closing bracket.
                let next = next_significant_byte(bytes, i + 1);
                if matches!(next, Some(b'}') | Some(b']')) {
                    // Skip this comma.
                    i += 1;
                } else {
                    out.push(',');
                    i += 1;
                }
            }
            _ if is_ident_start(c) => {
                // A bare identifier. If it is an object key (followed, after
                // whitespace, by ':'), quote it. Otherwise (true/false/null or a
                // bare value) leave it alone.
                let start = i;
                while i < bytes.len() && is_ident_part(bytes[i]) {
                    i += 1;
                }
                let ident = &source[start..i];
                let after = next_significant_byte(bytes, i);
                if after == Some(b':') {
                    out.push('"');
                    out.push_str(ident);
                    out.push('"');
                } else {
                    out.push_str(ident);
                }
            }
            _ => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// First non-whitespace, non-comment byte at or after `from`.
fn next_significant_byte(bytes: &[u8], from: usize) -> Option<u8> {
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            other => return Some(other),
        }
    }
    None
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_part(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.'
}

/// Parse a normalised blocktype JSON value into a [`ParsedBlockType`], or `None`
/// if it has neither a usable `code` nor any texture map.
pub fn parse_block_type(raw: &str) -> Option<ParsedBlockType> {
    let normalized = normalize_vs_json(raw);
    let value: serde_json::Value = serde_json::from_str(&normalized).ok()?;
    let object = value.as_object()?;

    let code = object
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let textures = object.get("textures").and_then(parse_texture_map);
    let textures_by_type = object.get("texturesByType").and_then(|v| {
        let by_type = v.as_object()?;
        let mut out: BTreeMap<String, TextureMap> = BTreeMap::new();
        for (variant_key, map_value) in by_type {
            if let Some(map) = parse_texture_map(map_value) {
                out.insert(variant_key.clone(), map);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    });

    if textures.is_none() && textures_by_type.is_none() {
        return None;
    }
    Some(ParsedBlockType {
        code,
        textures,
        textures_by_type,
    })
}

/// Parse a `textures`-shaped object into a face-key → entry map. Entries may be
/// `{ base: "..." }` objects or bare `"..."` strings; only `base` is kept.
fn parse_texture_map(value: &serde_json::Value) -> Option<TextureMap> {
    let object = value.as_object()?;
    let mut map: TextureMap = BTreeMap::new();
    for (face_key, entry_value) in object {
        let base = match entry_value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => {
                o.get("base").and_then(|b| b.as_str()).map(str::to_string)
            }
            _ => None,
        };
        if let Some(base) = base {
            map.insert(face_key.clone(), TextureEntry { base });
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

// ===========================================================================
// Resolving a chosen texture map → six face PNG paths.
// ===========================================================================

/// Resolve one `textures` map (already variant-selected) plus the chosen group
/// PNG into a full six-face mapping.
///
/// `group_png` is the variant the palette picked; it is the fallback for any
/// face the map doesn't cover, and supplies the `{placeholder}` substitution so
/// `{rock}` / `{wood}` resolve to this block's actual material. `resolve_path`
/// turns a texture reference (`domain:path` or bare) into an absolute PNG path
/// if a file exists.
fn resolve_face_map(
    map: &TextureMap,
    group_png: &Path,
    resolve_path: &dyn Fn(&str) -> Option<PathBuf>,
) -> [PathBuf; 6] {
    // Per-slot resolution with precedence: specific > group > all. We resolve
    // `all` first into every slot, then group keys, then specific keys.
    let mut slots: [Option<PathBuf>; 6] = Default::default();

    let mut apply = |face_slot: CubeFaceSlot, base: &str| {
        if let Some(path) = resolve_reference(base, group_png, resolve_path) {
            slots[face_slot.layer()] = Some(path);
        }
    };

    // `all` → every face.
    if let Some(entry) = map.get("all") {
        for slot in CubeFaceSlot::ALL {
            apply(slot, &entry.base);
        }
    }

    // Group keys (sides/horizontals → 4 horizontals; verticals → up+down).
    if let Some(entry) = map.get("sides").or_else(|| map.get("horizontals")) {
        for slot in [
            CubeFaceSlot::East,
            CubeFaceSlot::West,
            CubeFaceSlot::South,
            CubeFaceSlot::North,
        ] {
            apply(slot, &entry.base);
        }
    }
    if let Some(entry) = map.get("verticals") {
        apply(CubeFaceSlot::Up, &entry.base);
        apply(CubeFaceSlot::Down, &entry.base);
    }

    // Specific face keys (highest precedence).
    for (key, slot) in [
        ("up", CubeFaceSlot::Up),
        ("down", CubeFaceSlot::Down),
        ("north", CubeFaceSlot::North),
        ("south", CubeFaceSlot::South),
        ("east", CubeFaceSlot::East),
        ("west", CubeFaceSlot::West),
    ] {
        if let Some(entry) = map.get(key) {
            apply(slot, &entry.base);
        }
    }

    // Any face still unresolved falls back to the group PNG.
    let fallback = group_png.to_path_buf();
    CubeFaceSlot::ALL.map(|slot| slots[slot.layer()].clone().unwrap_or_else(|| fallback.clone()))
}

/// Resolve a single texture reference into an absolute PNG path.
///
/// VS bases carry a `{placeholder}` (`{rock}`/`{wood}`) for the material and
/// sometimes a trailing `*` glob. We substitute the placeholder with the chosen
/// variant's material name (the group PNG's file stem) so a shared base like
/// `block/wood/treetrunk/{wood}` points at the right per-material texture (this
/// is what makes a log's end-grain `up`/`down` differ from its bark sides). The
/// substituted reference is then resolved via [`resolve_path`]; only if that
/// finds no file on disk do we fall back to the group PNG itself.
fn resolve_reference(
    base: &str,
    group_png: &Path,
    resolve_path: &dyn Fn(&str) -> Option<PathBuf>,
) -> Option<PathBuf> {
    let substituted = substitute_placeholder(base, group_png);
    // If a clean (no-glob) reference resolves to a real file, use it.
    if !substituted.contains('{') && !substituted.contains('*') {
        if let Some(path) = resolve_path(&substituted) {
            return Some(path);
        }
    }
    // Otherwise the exact texture is variant-dependent (or missing on disk); the
    // group PNG already IS the chosen variant, so use it.
    Some(group_png.to_path_buf())
}

/// Substitute a single `{...}` placeholder in `base` with the material name taken
/// from the chosen variant PNG's file stem (e.g. `granite1.png` → `granite1`,
/// `oak.png` → `oak`). A trailing `*` after the placeholder is dropped. If `base`
/// has no placeholder it is returned unchanged.
fn substitute_placeholder(base: &str, group_png: &Path) -> String {
    if !base.contains('{') {
        return base.to_string();
    }
    let material = group_png
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut out = String::with_capacity(base.len() + material.len());
    let mut chars = base.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            // Skip to the closing brace, then emit the material name.
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
            }
            out.push_str(&material);
        } else {
            out.push(c);
        }
    }
    // A trailing `*` glob (e.g. `granite*`) is not part of any filename: drop it.
    out.replace('*', "")
}

/// Choose the best `textures`/`texturesByType` map for a block given the texture
/// stem we are looking for, and produce a [`FaceTextures`].
///
/// `wanted_stem` is the group key (e.g. `stone/rock/granite`) — the path the
/// blocktype's `base` entries reference. For `texturesByType`, the variant whose
/// any `base` references `wanted_stem` wins; otherwise the first variant is used.
pub fn resolve_block_faces(
    block: &ParsedBlockType,
    wanted_stem: &str,
    group_png: &Path,
    resolve_path: &dyn Fn(&str) -> Option<PathBuf>,
) -> FaceTextures {
    // Prefer the plain `textures` map when present.
    if let Some(map) = &block.textures {
        let paths = resolve_face_map(map, group_png, resolve_path);
        return FaceTextures {
            paths,
            provenance: FaceProvenance::Textures {
                block_code: block.code.clone(),
            },
        };
    }

    if let Some(by_type) = &block.textures_by_type {
        // Consider only variants whose any base references the wanted stem (else
        // all variants). Among those, prefer one that yields a genuinely per-face
        // mapping (top != side) so a multi-orientation block like a log resolves
        // to its most distinct variant; otherwise take the first.
        let matching: Vec<(&String, &TextureMap)> = by_type
            .iter()
            .filter(|(_, map)| map.values().any(|e| base_matches_stem(&e.base, wanted_stem)))
            .collect();
        let pool: Vec<(&String, &TextureMap)> = if matching.is_empty() {
            by_type.iter().collect()
        } else {
            matching
        };
        let chosen = pool
            .iter()
            .map(|(key, map)| (*key, *map, resolve_face_map(map, group_png, resolve_path)))
            .max_by_key(|(_, _, paths)| variant_face_score(paths));
        if let Some((variant_key, _, paths)) = chosen {
            return FaceTextures {
                paths,
                provenance: FaceProvenance::TexturesByType {
                    block_code: block.code.clone(),
                    variant_key: variant_key.clone(),
                },
            };
        }
    }

    FaceTextures::uniform(group_png.to_path_buf())
}

/// Rank a candidate `texturesByType` variant for how clearly it shows a
/// top-vs-side distinction (used to pick the most demonstrative orientation of a
/// multi-orientation block like a log). Higher is better:
///   * the top (`up`) face must differ from a representative side, and
///   * the texture on `up` should be the *less common* one across the six faces
///     (so a standing log's end-grain top is preferred over a lying log whose
///     end-grain is on the sides and whose top is plain bark).
fn variant_face_score(paths: &[PathBuf; 6]) -> i32 {
    let up = &paths[CubeFaceSlot::Up.layer()];
    let side = &paths[CubeFaceSlot::East.layer()];
    if up == side {
        return 0;
    }
    // up is distinct from the side: prefer the variant where fewer faces share
    // the up texture (i.e. up is the special/minority face → it reads as "top").
    let faces_matching_up = paths.iter().filter(|p| *p == up).count() as i32;
    // Base score 10 for being distinct, minus how widespread the up texture is.
    10 - faces_matching_up
}

/// Does a blocktype `base` reference the wanted texture stem? The base may carry
/// a `block/` prefix, `{placeholder}` and trailing `*`; we compare the cleaned
/// directory portion against the wanted stem's directory.
pub fn base_matches_stem(base: &str, wanted_stem: &str) -> bool {
    let base_clean = clean_texture_path(base);
    let wanted_clean = clean_texture_path(wanted_stem);
    if base_clean.is_empty() || wanted_clean.is_empty() {
        return false;
    }
    // The wanted stem's directory (e.g. `stone/rock` for `stone/rock/granite`).
    let wanted_dir = directory_of(&wanted_clean);
    if wanted_dir.is_empty() {
        return false;
    }
    // The base (glob already stripped) is either that directory itself (e.g.
    // `stone/rock/{rock}*` → `stone/rock`) or a file within it (e.g.
    // `stone/polishedrock/{rock}-inside` → `stone/polishedrock/...-inside`).
    base_clean == wanted_dir || base_clean.starts_with(&format!("{wanted_dir}/"))
}

/// Strip a leading `block/` (and leading slash), a domain prefix (`game:`), and a
/// trailing `{...}`/`*` glob, yielding a comparable texture path.
fn clean_texture_path(reference: &str) -> String {
    let mut s = reference.trim();
    // Drop a domain prefix like `game:`.
    if let Some(colon) = s.find(':') {
        s = &s[colon + 1..];
    }
    let s = s.trim_start_matches('/');
    let mut s = s.strip_prefix("block/").unwrap_or(s).to_string();
    // Strip a trailing glob `{...}` or `*` and anything after.
    if let Some(brace) = s.find('{') {
        s.truncate(brace);
    }
    if let Some(star) = s.find('*') {
        s.truncate(star);
    }
    s.trim_end_matches('/').to_string()
}

/// The directory portion of a `a/b/c` path (`a/b`), or empty if no slash.
fn directory_of(path: &str) -> String {
    match path.rfind('/') {
        Some(slash) => path[..slash].to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_quotes_bare_keys_and_drops_trailing_commas() {
        let raw = r#"{ code: "rock", textures: { all: { base: "x" }, }, }"#;
        let normalized = normalize_vs_json(raw);
        let value: serde_json::Value = serde_json::from_str(&normalized).expect("parses");
        assert_eq!(value["code"], "rock");
        assert_eq!(value["textures"]["all"]["base"], "x");
    }

    #[test]
    fn normalize_preserves_string_contents_with_colons() {
        let raw = r#"{ base: "block/stone:rock/{rock}*" }"#;
        let normalized = normalize_vs_json(raw);
        let value: serde_json::Value = serde_json::from_str(&normalized).expect("parses");
        assert_eq!(value["base"], "block/stone:rock/{rock}*");
    }

    #[test]
    fn parse_extracts_textures_and_by_type() {
        let raw = r#"{
            code: "log",
            texturesByType: {
                "*-ud": { all: { base: "block/wood/bark/{wood}" }, up: { base: "block/wood/treetrunk/{wood}" }, down: { base: "block/wood/treetrunk/{wood}" } }
            }
        }"#;
        let block = parse_block_type(raw).expect("parses");
        assert_eq!(block.code, "log");
        let bases = block.referenced_bases();
        assert!(bases.iter().any(|b| b.contains("bark")));
        assert!(bases.iter().any(|b| b.contains("treetrunk")));
    }

    #[test]
    fn base_matches_stem_on_directory() {
        assert!(base_matches_stem("block/stone/rock/{rock}*", "stone/rock/granite"));
        assert!(base_matches_stem("game:block/wood/bark/{wood}", "wood/bark/oak"));
        assert!(!base_matches_stem("block/wood/bark/{wood}", "stone/rock/granite"));
    }

    #[test]
    fn uniform_all_yields_six_identical_faces() {
        let group_png = Path::new("/x/granite1.png");
        let mut map: TextureMap = BTreeMap::new();
        map.insert("all".to_string(), TextureEntry { base: "block/stone/rock/{rock}*".to_string() });
        let paths = resolve_face_map(&map, group_png, &|_| None);
        assert!(paths.iter().all(|p| p == group_png));
    }

    #[test]
    fn up_down_override_all_for_distinct_faces() {
        // Resolver returns the requested clean path verbatim (file exists stub).
        let group_png = Path::new("/x/bark/oak.png");
        let resolve = |reference: &str| -> Option<PathBuf> {
            Some(PathBuf::from(format!("/assets/{}.png", reference)))
        };
        let mut map: TextureMap = BTreeMap::new();
        map.insert("all".to_string(), TextureEntry { base: "wood/bark/oak".to_string() });
        map.insert("up".to_string(), TextureEntry { base: "wood/treetrunk/oak".to_string() });
        map.insert("down".to_string(), TextureEntry { base: "wood/treetrunk/oak".to_string() });
        let paths = resolve_face_map(&map, group_png, &resolve);
        let up = &paths[CubeFaceSlot::Up.layer()];
        let east = &paths[CubeFaceSlot::East.layer()];
        assert!(up.to_string_lossy().contains("treetrunk"));
        assert!(east.to_string_lossy().contains("bark"));
        assert_ne!(up, east);
    }
}
