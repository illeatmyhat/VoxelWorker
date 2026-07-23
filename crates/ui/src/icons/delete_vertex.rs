//! `delete-vertex` — remove a sketch entity.
//!
//! A generic entity — a line **segment** with two hollow end-nodes, standing in for a point /
//! segment / arc — struck by a **single slash**: "remove the selected thing". General removal, NOT
//! specifically "remove a vertex from the closed loop", which is one verb of the entity-based
//! sketch model (ADR 0028; owner reframe 2026-07-23). One slash, not the rejected node-X (too close
//! to `cancel`, and it did not generalise to a segment or arc); distinct from `subtract` (two
//! overlapping bodies = a boolean). Chosen glyph = the "general verbs" sheet's delete v3.
//!
//! The destructive **warn** channel is NOT carried here: the icon family's second channel is
//! texture, not colour (`docs/design/colour-vocabulary.md`), so the rail glyph is monochrome in the
//! host colour. The warn-red destructive signal lives on the on-canvas delete-hover gizmo
//! (`HandleState::Marked`), where two-tone is available, not on this glyph.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The entity being removed: a segment with two hollow end-nodes (a generic point/segment/arc).
    Mark::Line {
        points: &[(4.5, 7.0), (13.0, 13.0)],
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (3.3, 5.8),
        b: (5.7, 8.2),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (11.8, 11.8),
        b: (14.2, 14.2),
        ink: Ink::SOLID,
    },
    // The removal slash — one stroke across the entity (the destructive warn hue is carried by the
    // on-canvas gizmo, not the monochrome rail glyph).
    Mark::Line {
        points: &[(14.5, 6.0), (6.5, 15.0)],
        ink: Ink::SOLID,
    },
];
