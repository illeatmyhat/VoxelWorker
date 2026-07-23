//! `add-point` — insert a profile vertex, splitting an edge.
//!
//! A profile edge crossed by a `+` at its midpoint: the new point lands ON the segment,
//! splitting it. Deliberately distinct from `delete-vertex` (a node struck through with an
//! X) and `select-vertex` (an arrow carrying a node) — the `+` on the line says "a point is
//! added *here*, on this edge". Owner ruling 2026-07-22, ADR 0028 #95.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The profile edge being split — its midpoint is (9, 9).
    Mark::Line {
        points: &[(2.5, 13.0), (15.5, 5.0)],
        ink: Ink::SOLID,
    },
    // The add `+`, centred on the edge at its midpoint: the vertex being inserted.
    Mark::Line {
        points: &[(9.0, 4.5), (9.0, 13.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(4.5, 9.0), (13.5, 9.0)],
        ink: Ink::SOLID,
    },
];
