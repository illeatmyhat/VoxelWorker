//! `region_badge` — a derived sketch region's pick state, stamped at its centroid (ADR 0030 §3).
//!
//! Round, where every vertex mark in the set is square: a badge is a READOUT of a face, not a
//! grabbable handle, and the shape says so before the fill does. The two-tone idiom is the
//! family's — accent fill when the region is picked (it resolves as material), dark fill under an
//! accent ring when it is unpicked (a hole). A wash over the face itself was rejected: `egui`
//! fills convex polygons only, and a sketch face is frequently concave.

use egui::{Painter, Pos2, Stroke};

use super::{HANDLE_ACCENT, HANDLE_FILL, STROKE_HANDLE};

/// The badge's radius in egui points.
pub const REGION_BADGE_RADIUS: f32 = 4.5;

/// Draw a region badge at `center`: filled when `picked`, a hollow ring when not.
pub fn region_badge(painter: &Painter, center: Pos2, picked: bool) {
    let fill = if picked { HANDLE_ACCENT } else { HANDLE_FILL };
    painter.circle_filled(center, REGION_BADGE_RADIUS, fill);
    painter.circle_stroke(
        center,
        REGION_BADGE_RADIUS,
        Stroke::new(STROKE_HANDLE, HANDLE_ACCENT),
    );
}
