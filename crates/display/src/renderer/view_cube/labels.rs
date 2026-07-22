//! Pure-CPU face-label textures + bitmap font (no wgpu): the six stacked Signal
//! face-label RGBA8 layers and the built-in 5×7 glyph font that renders them. The
//! GPU renderer uploads the returned bytes into the label texture array.

use super::*;

/// A sRGB hex → RGBA8 (opaque) texel; the label textures are `Rgba8UnormSrgb`, so the
/// sRGB byte values are written straight through.
fn hex_texel(hex: u32) -> [u8; 4] {
    [((hex >> 16) & 0xff) as u8, ((hex >> 8) & 0xff) as u8, (hex & 0xff) as u8, 0xff]
}

/// The flat Signal fill of face `layer` (GEOMETRIC order +X,-X,+Y,-Y,+Z,-Z =
/// Right,Left,Back,Front,Top,Bottom) — distinct near-black values within
/// `#10141a`–`#1b2126` so the three visible faces (TOP lightest, FRONT mid, RIGHT
/// darker) read apart under the flat (unlit) shading.
fn face_fill_hex(layer: usize) -> u32 {
    match layer {
        0 => 0x13_18_20, // Right
        1 => 0x0f_13_18, // Left
        2 => 0x12_16_1b, // Back
        3 => 0x16_1c_22, // Front
        4 => 0x1b_21_26, // Top
        _ => 0x10_14_19, // Bottom
    }
}

/// Render the six face-label textures into one stacked RGBA8 buffer (6 layers, in
/// GEOMETRIC `materialIndex` order +X,-X,+Y,-Y,+Z,-Z). Z-up labels each geometric
/// face: +Y = BACK, −Y = FRONT, +Z = TOP, −Z = BOTTOM. Signal style: a flat
/// near-black fill (per-face, [`face_fill_hex`]), hairline `#2b3238` slice lines at
/// the 68 %-centre partition, and a monospace `#aeb9c4` label in the centre patch.
pub(super) fn generate_face_label_textures() -> Vec<u8> {
    const LABELS: [&str; 6] = ["RIGHT", "LEFT", "BACK", "FRONT", "TOP", "BOTTOM"];
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let mut all = Vec::with_capacity(size * size * 4 * 6);
    for (layer, label) in LABELS.iter().enumerate() {
        all.extend_from_slice(&render_face_label(label, face_fill_hex(layer)));
    }
    all
}

/// Render one Signal face-label texture (RGBA8, `FACE_LABEL_TEXTURE_SIZE` square):
/// flat `fill_hex` background and the monospace label. The 3×3 slice lines are NOT baked
/// here anymore (issue #91 item 3): they render as a constant-width anti-aliased SDF in
/// `viewcube.wgsl` (screen-space, so glancing angles never thin them), leaving this
/// texture as just the flat fill + centred label.
fn render_face_label(label: &str, fill_hex: u32) -> Vec<u8> {
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let background = hex_texel(fill_hex);
    let text = hex_texel(FACE_LABEL_HEX);

    let mut pixels = vec![0u8; size * size * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&background);
    }

    // Monospace label, sized to sit inside the 68 % centre patch.
    draw_centered_label(&mut pixels, size, label, text);
    pixels
}

/// Draw `label` centred using the built-in 5×7 bitmap font, scaled to fill the
/// face, into the RGBA8 `pixels` buffer.
fn draw_centered_label(pixels: &mut [u8], size: usize, label: &str, color: [u8; 4]) {
    let glyph_width = 5usize;
    let glyph_height = 7usize;
    let spacing = 1usize;
    let count = label.chars().count().max(1);
    let text_cells_wide = count * glyph_width + (count - 1) * spacing;
    // Choose an integer scale that keeps the label inside the 68 % centre patch
    // (~60% of the face width, ~34% of its height), clear of the slice lines.
    let max_scale_w = (size * 6 / 10) / text_cells_wide.max(1);
    let max_scale_h = (size * 34 / 100) / glyph_height;
    let scale = max_scale_w.min(max_scale_h).max(1);

    let text_pixel_width = text_cells_wide * scale;
    let text_pixel_height = glyph_height * scale;
    let origin_x = (size.saturating_sub(text_pixel_width)) / 2;
    let origin_y = (size.saturating_sub(text_pixel_height)) / 2;

    let mut cursor_x = origin_x;
    for ch in label.chars() {
        let glyph = glyph_bitmap(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..glyph_width {
                if (bits >> (glyph_width - 1 - col)) & 1 == 1 {
                    // Filled cell → scale×scale block.
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let x = cursor_x + col * scale + dx;
                            let y = origin_y + row * scale + dy;
                            if x < size && y < size {
                                let index = (y * size + x) * 4;
                                pixels[index..index + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += (glyph_width + spacing) * scale;
    }
}

/// A 5×7 bitmap (7 rows of 5-bit masks) for the uppercase letters used by the
/// face labels. Unknown characters render blank.
fn glyph_bitmap(ch: char) -> [u8; 7] {
    match ch {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        _ => [0; 7],
    }
}
