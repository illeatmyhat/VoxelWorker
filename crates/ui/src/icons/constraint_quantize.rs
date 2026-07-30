//! `constraint-quantize` — the value is an integer multiple of a voxel.
//!
//! A DOT LATTICE, after AutoCAD, which is the only prior art anywhere for this. A magnet would say
//! "assistive while dragging"; this is persistent and the solver can see it.
//!
//! That is exactly the split from [`snap_voxel`](super::snap_voxel), which is modal and leaves no
//! record. ADR 0035 D1 insists on the distinction, and here the accent enforces it: the quantized
//! vertex is driven, the lattice around it is only a reference.
//!
//! The centre of the lattice is left to the node — a ninth dot under a square is mush at 16 px.

use super::{Ink, Mark};

/// The vertex the constraint drives, at the centre of a lattice of pitch 5.
const VERTEX: (f32, f32) = (9.0, 9.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Disc {
        center: (4.0, 4.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (4.0, 9.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (4.0, 14.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (9.0, 4.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (9.0, 14.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (14.0, 4.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (14.0, 9.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (14.0, 14.0),
        radius: 0.75,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: VERTEX,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
