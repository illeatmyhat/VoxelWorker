//! `arc_center` — an arc's DERIVED centre, plus the two radii that put it there.
//!
//! ADR 0030 §5 stores an arc as two endpoints and an included angle; the centre and the radius
//! are derived, never authored. The centre itself is a REIFIED point entity, so it wears the same
//! vertex handle every other sketch point does — what is drawn here is only the pair of radii that
//! explain WHY the handle is there, at the guide weight in dashed accent: the family's "authored,
//! but not the thing you are looking at".

use egui::{Painter, Pos2, Stroke};

use super::{dashed, HANDLE_ACCENT, STROKE_GUIDE};

/// Draw the two dashed radii joining an arc's `center` to its `endpoints`, so the radius is
/// legible without a dimension. The centre's own handle is drawn by the vertex pass.
pub fn arc_center(painter: &Painter, center: Pos2, endpoints: [Pos2; 2]) {
    let stroke = Stroke::new(STROKE_GUIDE, HANDLE_ACCENT);
    for endpoint in endpoints {
        dashed(painter, center, endpoint, stroke);
    }
}
