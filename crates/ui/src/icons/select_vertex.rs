//! `select-vertex` — the default sketch arrow, over the geometry it picks FROM.
//!
//! Re-drawn from the sheet (owner, 2026-07-30). The shipped glyph was an arrow carrying a node
//! at its tip, which said "pick a vertex" by decorating the cursor. This one says it by showing
//! the situation instead: a two-segment profile with three vertices, the middle one accented,
//! and the arrow beside it. The accent is doing the work — one of three identical squares is
//! lit, so the glyph means "pick ONE of these" rather than "a pick happens here".
//!
//! The picked vertex is drawn LARGER as well as accented (3.2 against 2.2). At 15 pt the accent
//! alone is two pixels of hue; the size difference is what survives, and it is the same
//! selected-handle idiom the viewport already uses. ADR 0028 slice 1, the mode's default tool.

use super::{Ink, Mark};

/// An unpicked profile vertex, and the picked one. Both sizes are the sheet's.
const VERTEX: f32 = 2.2;
const PICKED: f32 = 3.2;

pub(super) const DRAW: &[Mark] = &[
    // The profile: two segments meeting at the vertex the arrow is on.
    Mark::Line {
        points: &[(2.5, 13.0), (8.0, 4.5), (15.0, 10.0)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (2.5, 13.0),
        size: VERTEX,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (15.0, 10.0),
        size: VERTEX,
        ink: Ink::SOLID,
    },
    // The pick.
    Mark::Node {
        center: (8.0, 4.5),
        size: PICKED,
        ink: Ink::ACCENT,
    },
    // The pointer, tucked into the lower right so it never crosses the picked vertex.
    Mark::Closed {
        points: &[
            (10.0, 7.5),
            (10.0, 13.5),
            (11.5, 12.0),
            (13.0, 14.5),
            (14.0, 14.0),
            (12.5, 11.5),
            (14.5, 11.0),
        ],
        ink: Ink::SOLID,
    },
];
