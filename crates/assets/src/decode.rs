//! Raw image → RGBA decode.
//!
//! A single-responsibility helper: read an image file off disk and hand back its
//! pixels as a tightly-packed RGBA8 buffer plus dimensions — `(width, height,
//! rgba_bytes)`. It is the CPU decode step shared by the block-thumbnail scan
//! (the worker that streams palette entries) and the palette atlas builder (which
//! decodes each chosen face PNG). No GPU work happens here; the bytes are handed
//! upward for texture upload.

/// A decoded RGBA image: `(width, height, rgba_bytes)`.
pub type DecodedRgba = (u32, u32, Vec<u8>);

/// Decode an image file to a tightly-packed RGBA8 buffer (CPU work). Returns
/// `None` on any decode error (the caller skips the group, matching the
/// prototype's try/catch-continue in `buildPalette`).
pub fn decode_rgba(path: &std::path::Path) -> Option<DecodedRgba> {
    let image = image::open(path).ok()?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some((width, height, rgba.into_raw()))
}
