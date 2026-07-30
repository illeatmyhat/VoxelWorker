//! `arc_center` — an arc's DERIVED centre, plus the two radii that put it there.
//!
//! ADR 0030 §5 stores an arc as two endpoints and an included angle; the centre and the radius
//! are derived, never authored. So this is a DATUM, drawn at the guide weight in dashed accent —
//! the family's "authored, but not the thing you are looking at" — and it is deliberately not a
//! handle: there is nothing here to grab, because moving it would not be an edit the store can
//! hold.

use egui::{Painter, Pos2, Stroke};

use super::{dashed, HANDLE_ACCENT, STROKE_GUIDE};

/// The reach (egui points) of each arm of the centre's cross.
const CENTER_CROSS_REACH: f32 = 3.5;

/// Draw an arc's centre datum at `center`, with a dashed radius to each of its `endpoints` so the
/// radius is legible without a dimension.
pub fn arc_center(painter: &Painter, center: Pos2, endpoints: [Pos2; 2]) {
    let stroke = Stroke::new(STROKE_GUIDE, HANDLE_ACCENT);
    for endpoint in endpoints {
        dashed(painter, center, endpoint, stroke);
    }
    painter.line_segment(
        [
            Pos2::new(center.x - CENTER_CROSS_REACH, center.y),
            Pos2::new(center.x + CENTER_CROSS_REACH, center.y),
        ],
        stroke,
    );
    painter.line_segment(
        [
            Pos2::new(center.x, center.y - CENTER_CROSS_REACH),
            Pos2::new(center.x, center.y + CENTER_CROSS_REACH),
        ],
        stroke,
    );
}
