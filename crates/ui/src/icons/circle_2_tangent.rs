//! `circle-2-tangent` — a circle of a given radius, touching two picked curves.
//!
//! The two tangent shifts to the ACCENT and the circle stays line art, inverting the rest of the
//! family: here the picks are curves, not points, so there is nothing to put a node on and the
//! accent has to land on the curves themselves.
//!
//! The radius line survives because the radius is still typed — two tangencies leave one freedom,
//! and the mark has to say which one you supply.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(2.0, 2.0), (2.0, 16.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(2.0, 16.0), (16.0, 16.0)],
        ink: Ink::ACCENT,
    },
    // Centred 5.5 off each picked line, so both touches are real.
    Mark::Circle {
        center: (7.5, 10.5),
        radius: 5.5,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(7.5, 10.5), (11.39, 6.61)],
        ink: Ink::SOLID,
    },
];
