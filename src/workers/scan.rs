//! Background block scanning (Milestone 6).
//!
//! Detection + walkdir + PNG decode are CPU work that must NOT block startup, so
//! they run on a worker thread. The worker streams results to the main thread
//! over an `mpsc` channel; the main thread polls the channel each frame and turns
//! each message into GPU resources (a thumbnail texture, an egui `TextureId`).
//!
//! Only CPU-side work happens here (`image` decode + `walkdir`). All GPU work
//! (texture upload, the 45° thumbnail render, egui registration) stays on the
//! main thread — see `block_palette.rs`.
//!
//! Two message kinds flow over the channel:
//!   * [`ScanMessage::Group`] — one scanned [`BlockGroup`] with its first
//!     variant decoded to an RGBA buffer (the thumbnail source).
//!   * [`ScanMessage::Done`] — the scan finished; carries the total group count
//!     and the human-readable source name (or `None` if nothing was detected) so
//!     the status line can settle.

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::assets::{custom_pack::CustomFolderSource, registry::detect_all_sources, BlockGroup, BlockSource};

/// A decoded RGBA image: `(width, height, rgba_bytes)`.
pub type DecodedRgba = (u32, u32, Vec<u8>);

/// One message streamed from the scan worker to the main thread.
pub enum ScanMessage {
    /// A scanned group plus its first variant decoded to RGBA (thumbnail source).
    Group {
        group: BlockGroup,
        thumbnail_rgba: DecodedRgba,
    },
    /// The scan finished. `source_name` is `Some` when at least one source was
    /// found (the first source's display name), `None` when nothing was detected.
    Done {
        group_count: usize,
        source_name: Option<String>,
    },
}

/// A live background scan: holds the receiver the main thread polls plus the
/// worker handle (joined on drop is unnecessary — the channel close signals end).
pub struct ScanHandle {
    receiver: Receiver<ScanMessage>,
    _worker: JoinHandle<()>,
}

impl ScanHandle {
    /// Drain whatever the worker has produced so far without blocking.
    pub fn drain(&self) -> Vec<ScanMessage> {
        self.receiver.try_iter().collect()
    }
}

/// Spawn the auto-detect + scan worker. Returns immediately; results stream over
/// the channel. Detection runs every known `SourceDetector`; each found
/// source's groups are decoded and sent in order.
pub fn spawn_auto_scan() -> ScanHandle {
    spawn_with(detect_all_sources)
}

/// Spawn a scan of a single custom folder (the "Connect folder…" path).
pub fn spawn_custom_folder_scan(folder: std::path::PathBuf) -> ScanHandle {
    spawn_with(move || vec![Box::new(CustomFolderSource::new(folder)) as Box<dyn BlockSource>])
}

/// Shared spawn: run `make_sources` on the worker, scan each, decode + stream.
fn spawn_with<F>(make_sources: F) -> ScanHandle
where
    F: FnOnce() -> Vec<Box<dyn BlockSource>> + Send + 'static,
{
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("voxel-worker block scan".to_string())
        .spawn(move || {
            let sources = make_sources();
            run_scan(sources, &sender);
        })
        .expect("failed to spawn scan worker");
    ScanHandle {
        receiver,
        _worker: worker,
    }
}

/// Scan every source, decode each group's first variant, stream the results, then
/// send [`ScanMessage::Done`]. Used by both the windowed worker and the
/// synchronous CLI path (`--scan-vs`) via [`run_scan_collect`].
fn run_scan(sources: Vec<Box<dyn BlockSource>>, sender: &Sender<ScanMessage>) {
    let source_name = sources.first().map(|source| source.display_name().to_string());
    let mut group_count = 0usize;
    for source in &sources {
        for group in source.scan() {
            let Some(first_variant) = group.variants.first() else {
                continue;
            };
            let Some(thumbnail_rgba) = decode_rgba(first_variant) else {
                continue;
            };
            group_count += 1;
            // If the receiver is gone (window closed), stop early.
            if sender
                .send(ScanMessage::Group { group, thumbnail_rgba })
                .is_err()
            {
                return;
            }
        }
    }
    let _ = sender.send(ScanMessage::Done { group_count, source_name });
}

/// Run the same detect+scan+decode **synchronously** and return all groups with
/// their decoded thumbnails (the headless `--scan-vs` path). Returns the source
/// name too (for logging).
pub fn run_auto_scan_blocking() -> (Vec<(BlockGroup, DecodedRgba)>, Option<String>) {
    let sources = detect_all_sources();
    let source_name = sources.first().map(|source| source.display_name().to_string());
    let mut results = Vec::new();
    for source in &sources {
        for group in source.scan() {
            let Some(first_variant) = group.variants.first() else {
                continue;
            };
            if let Some(thumbnail_rgba) = decode_rgba(first_variant) {
                results.push((group, thumbnail_rgba));
            }
        }
    }
    (results, source_name)
}

/// Per-face texture resolver (Milestone 7).
///
/// Holds the detected [`BlockSource`]s alive so the main thread can resolve a
/// clicked block's per-face textures on demand (each VS source caches its parsed
/// blocktype index internally, so repeated lookups are cheap). Built once and
/// kept beside the palette.
pub struct FaceResolver {
    sources: Vec<Box<dyn BlockSource>>,
}

impl FaceResolver {
    /// Build a resolver from auto-detected sources (windowed + headless paths).
    pub fn auto() -> Self {
        Self {
            sources: detect_all_sources(),
        }
    }

    /// Build a resolver for a single custom folder (the "Connect folder…" path).
    pub fn custom_folder(folder: std::path::PathBuf) -> Self {
        Self {
            sources: vec![Box::new(CustomFolderSource::new(folder)) as Box<dyn BlockSource>],
        }
    }

    /// Resolve a group's per-face textures, picking the source by matching the
    /// group key against the source's scanned groups. `chosen_variant` is the
    /// specific PNG the palette picked. Falls back to a uniform mapping if no
    /// source recognises the group (the M6 behaviour).
    pub fn resolve(
        &self,
        group: &BlockGroup,
        chosen_variant: &std::path::Path,
    ) -> crate::assets::FaceTextures {
        // A single source is the common case; try each and take the first that
        // returns a genuinely per-face (non-uniform) mapping, else the first.
        let mut fallback: Option<crate::assets::FaceTextures> = None;
        for source in &self.sources {
            let faces = source.resolve_faces(group, chosen_variant);
            if !faces.is_uniform() {
                return faces;
            }
            if fallback.is_none() {
                fallback = Some(faces);
            }
        }
        fallback.unwrap_or_else(|| crate::assets::FaceTextures::uniform(chosen_variant.to_path_buf()))
    }
}

/// Decode a PNG file to a tightly-packed RGBA8 buffer (CPU work). Returns `None`
/// on any decode error (the group is skipped, matching the prototype's
/// try/catch-continue in `buildPalette`).
pub fn decode_rgba(path: &std::path::Path) -> Option<DecodedRgba> {
    let image = image::open(path).ok()?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some((width, height, rgba.into_raw()))
}
