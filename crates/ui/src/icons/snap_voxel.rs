//! `snap-voxel` — the vertex snaps to the fine lattice crossing. The default.
//!
//! The node sits ON the crossing of the two in-plane axes: whole-voxel quantization, the
//! sketch's default position snap. The 2D reuse of ADR 0027's snap (ADR 0028 §5).
//!
//! Geometry re-drawn from the sheet (owner, 2026-07-30) — the axes now run the full 2.5–15.5
//! the rest of the set uses, and the vertex is a [`Mark::Node`] rather than a hand-spelled
//! `Rect`. **The sheet's INK is deliberately not taken**, which is why this glyph is the one
//! member of the sheet still outside the parity gate: the sheet draws every mark of it in tool
//! blue, because there blue means "this is a mode". In the rail a mode says that by its button
//! being armed, so an all-accent glyph would read as *voxel snap is on* whichever of the three
//! snaps is actually active — and its two siblings, `snap-none` and `snap-block`, are line art.
//! A three-state selector with one member permanently lit is a regression, not a re-draw.

use super::{Ink, Mark};

/// The vertex locked on the crossing. The sheet's size; only the ink differs.
const VERTEX: f32 = 3.2;

pub(super) const DRAW: &[Mark] = &[
    // The in-plane axes.
    Mark::Line {
        points: &[(9.0, 2.5), (9.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 9.0), (15.5, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (9.0, 9.0),
        size: VERTEX,
        ink: Ink::SOLID,
    },
];
