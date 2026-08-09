//! What a drawing tool's pointer is aimed at.

use super::{EntityId, SketchCurve, SketchPoint};

/// The point a drawing tool's click landed on.
///
/// A sum rather than a position carrying an optional identity beside an optional curve, because
/// most of that product means nothing. A click cannot both name a stored point and want a fresh
/// one held to a curve: pointing at a vertex means that vertex even when the vertex sits on an
/// edge. Written as three loose fields that rule has to be re-enforced by whoever reads them;
/// written this way the curve is only reachable from the arm that can act on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SketchTarget {
    /// A point the sketch already holds, at its own stored position.
    ///
    /// The position rides along so a preview needs no lookup, and so a point stacked on another
    /// still resolves to the one the pointer named rather than to whichever shares the coordinate.
    Existing { id: EntityId, at: SketchPoint },
    /// A place the sketch has no point at yet. Minting one there holds it to `onto` when the
    /// pointer was over a curve.
    Fresh {
        at: SketchPoint,
        onto: Option<SketchCurve>,
    },
}

impl SketchTarget {
    /// A bare place, with nothing under the pointer — the shape most tests and non-pointer
    /// callers want.
    #[must_use]
    pub const fn fresh(at: SketchPoint) -> Self {
        SketchTarget::Fresh { at, onto: None }
    }

    /// Where the target is. Every preview reads this whether or not a point is there yet.
    #[must_use]
    pub const fn at(self) -> SketchPoint {
        match self {
            SketchTarget::Existing { at, .. } | SketchTarget::Fresh { at, .. } => at,
        }
    }

    /// The curve a point minted here would be held to — nothing when the target already names a
    /// point, which is what the drawing highlights so the click and the highlight cannot disagree.
    #[must_use]
    pub const fn onto(self) -> Option<SketchCurve> {
        match self {
            SketchTarget::Existing { .. } => None,
            SketchTarget::Fresh { onto, .. } => onto,
        }
    }

    /// The identity, when the target already has one.
    #[must_use]
    pub const fn existing(self) -> Option<EntityId> {
        match self {
            SketchTarget::Existing { id, .. } => Some(id),
            SketchTarget::Fresh { .. } => None,
        }
    }
}
