//! `constraint-horizontal-vertical` — along an axis, whichever axis the line is already nearer.
//!
//! ONE cell, TWO constraints (Fusion's arrangement): the author says "line this up with an axis"
//! and the drawing decides which axis it meant. What gets asserted is a plain
//! [`Horizontal`](super::constraint_horizontal) or [`Vertical`](super::constraint_vertical), and
//! the badge left behind carries THAT mark rather than this one — this glyph belongs to the
//! question, those belong to the answer.
//!
//! It is the only mark in the set that ASKS rather than reports, and the two-tone ink says so.
//! White is the reference: the two arms are the axes on offer, each 11 long — exactly the bar in
//! `Horizontal` and in `Vertical`, so the cell quotes the two answers it stands for. Red is the
//! driven entity: the author's own segment, drawn at exactly 45° with its midpoint on the
//! corner's bisector, because the question is symmetric between the two axes and a segment tilted
//! toward either one would already have answered it. It carries no end nodes — `Collinear` and
//! `Parallel` set the precedent that a bare run reads as a segment, and two more squares would
//! crowd the corner this has to stay clear of.
//!
//! The two bars SUPERIMPOSED were tried first and rejected: that is a plus with four nodes, which
//! is what `Snap to voxel` already is, and it needed its nodes shrunk below the pair's own to
//! survive its own crossing — a concession that says the construction was wrong.

use super::{Ink, Mark};

/// The axis corner: up the left, along the bottom, meeting at `(3.5, 14.5)`.
const AXES: &[(f32, f32)] = &[(3.5, 3.5), (3.5, 14.5), (14.5, 14.5)];
/// The author's segment, 45° across the corner, centred on `(9.0, 9.0)`.
const SEGMENT: &[(f32, f32)] = &[(5.5, 12.5), (12.5, 5.5)];

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: AXES,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: SEGMENT,
        ink: Ink::CONSTRAINT,
    },
];
