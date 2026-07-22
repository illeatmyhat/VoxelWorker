//! `snap-none` — the vertex rides sub-voxel, exactly under the cursor.
//!
//! A free node inside a loose (dashed) region: no lattice holds it, and the fraction lives on
//! `offset_local`. The 2D reuse of ADR 0027's position snap for profile vertices (ADR 0028
//! §5); its glyph twin among the placement snaps.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The free vertex.
    Mark::Rect {
        a: (7.6, 7.6),
        b: (10.4, 10.4),
        ink: Ink::SOLID,
    },
    // The loose region — nothing quantises it.
    Mark::Circle {
        center: (9.0, 9.0),
        radius: 5.6,
        ink: Ink::DASHED,
    },
];
