//! `ellipse-sketch` — centre, then the two semi-axes.
//!
//! Three accented nodes: the centre and one end of each axis, which is exactly the click sequence.
//! The axes themselves are not drawn — the nodes already sit on them, and two crossing lines
//! inside a small ellipse close the shape into a blob at 16 px.
//!
//! Named `ellipse-sketch` rather than `ellipse` because the producer set already has one; this is
//! the sketch-mode tool, not the solid.

use super::{Ink, Mark};

/// The centre, and the ends the two drags land on.
const CENTRE: (f32, f32) = (9.0, 7.5);
const RX: f32 = 7.5;
const RY: f32 = 4.5;

pub(super) const DRAW: &[Mark] = &[
    // The outline is drawn a half unit inside the major handle, so the node sits ON the curve
    // rather than half a stroke outside it.
    Mark::Ellipse {
        center: CENTRE,
        rx: 7.0,
        ry: RY,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: CENTRE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (CENTRE.0, CENTRE.1 - RY),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (CENTRE.0 + RX, CENTRE.1),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
