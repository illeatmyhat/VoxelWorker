//! `open_segment` — the real segment from the last committed vertex to the live cursor.

use egui::{Painter, Pos2};

use super::diamond::diamond;
use super::segment::segment;

/// The **open segment** — the real segment from the last committed vertex to the live cursor,
/// ending in a hollow diamond at the pointer. Solid, because it too is a real entity being placed
/// (owner ruling: a segment you are placing, not a rubber band — ADR 0028 §6), so it uses the
/// solid [`segment`], never the dashed idiom.
pub fn open_segment(painter: &Painter, from: Pos2, cursor: Pos2, diamond_half: f32) {
    segment(painter, from, cursor);
    diamond(painter, cursor, diamond_half);
}
