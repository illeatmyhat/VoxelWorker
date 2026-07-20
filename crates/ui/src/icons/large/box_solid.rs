//! `box` (tile) — a complete iso cube with its three hidden back edges receding.
//!
//! The block IS a cube, so the tile draws one in projection rather than the rail's flat
//! silhouette. The fade on the back edges is x-ray, the same reading the app's operand
//! ghosts use — depth without a hidden-line pass.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The cube's visible silhouette.
    Mark::Closed {
        points: &[
            (13.0, 3.0),
            (23.0, 8.0),
            (23.0, 18.0),
            (13.0, 23.0),
            (3.0, 18.0),
            (3.0, 8.0),
        ],
        ink: Ink::SOLID,
    },
    // The three edges meeting at the far corner, dropped back.
    Mark::Line {
        points: &[(3.0, 8.0), (13.0, 13.0), (23.0, 8.0)],
        ink: Ink::faint(0.5),
    },
    Mark::Line {
        points: &[(13.0, 13.0), (13.0, 23.0)],
        ink: Ink::faint(0.5),
    },
];
