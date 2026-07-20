//! The TILE glyph family: producers drawn large, for shape tiles and drawer thumbnails.
//!
//! ## Why this is a separate enum and not a second method on [`Icon`]
//!
//! A tile glyph is a **different drawing of the same noun**, not the rail mark scaled up.
//! The two families answer to different constraints, and the differences are the content:
//!
//! | | rail ([`Icon`]) | tile (here) |
//! |---|---|---|
//! | grid / stroke | 18 units · 1.25 | 26 units · 1.1 (proportionally lighter) |
//! | depth | flat silhouette | drawn in projection |
//! | opacity | none — at 15 pt a faded stroke reads as a *gap* | a third ink: solid = front, faded = behind |
//! | fills | none — a filled region at rail size is a blob | allowed, where the subject IS a region |
//! | content | the subject only | subject + datum + footprint |
//!
//! Shrinking a tile mark onto the rail grid destroys exactly the construction that earned
//! it, which is why the families live apart and neither is generated from the other.
//!
//! ## Which surfaces use which
//!
//! Shapes are a CLOSED set — a finite list of verbs — so they are a permanent rail of large
//! tiles; materials and saved parts grow forever, so they are browsed in the drawer. Tiles
//! are used for shape rail cells and for asset thumbnails; everything else (tool rail, the
//! browser tree, fold cards, inspector chips) stays on the rail set.
//!
//! A shape with no tile glyph falls back to its rail mark — see [`LargeIcon::for_icon`].

use egui::{Color32, Painter, Rect};

use super::{Icon, IconPainter};

mod box_solid;
mod cylinder;
mod displace;
mod extrude;
mod half_space;
mod revolve;
mod sculpt;
mod sketch;
mod sphere;
mod sweep;

/// The authoring grid every tile glyph is traced on: 26 × 26 units.
pub const GRID: f32 = 26.0;

/// The tile stroke, in design points — proportionally about half the rail weight, because a
/// large mark carries its weight through construction rather than through ink.
pub const STROKE_WIDTH: f32 = 1.1;

/// A mark drawn at tile size.
///
/// Mostly producers — the closed verb set that earns a permanent rail, and the marks whose
/// whole content is projection and depth. Two field/tool marks join them because their
/// construction is carried by OPACITY, which the rail cannot spend: `displace` needs its
/// datum visible-but-subordinate, and `sculpt` needs a reach that is felt rather than drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LargeIcon {
    /// The authoring atom: a curve with two authored, grabbable endpoints.
    Sketch,
    /// A sketch swept straight along its normal.
    Extrude,
    /// A sketch spun about an authored axis.
    Revolve,
    /// A profile carried along a path — the reserved third lift.
    Sweep,
    /// The box primitive, drawn as the cube a block actually is.
    BoxSolid,
    /// The sphere primitive.
    Sphere,
    /// The cylinder primitive.
    Cylinder,
    /// An unbounded plane; everything on one side is body.
    HalfSpace,
    /// A brush: the core it writes, inside the reach it stops at.
    ///
    /// The rail set has no generic sculpt — it splits the verb into `sculpt-add` and
    /// `carve`, because at 15 pt polarity has to be in the mark itself. At tile size the
    /// brush is the thing being CHOSEN and polarity is stated elsewhere, so the generic
    /// mark is the honest one. [`rail`](Self::rail) therefore answers `sculpt-add`, the
    /// nearest twin, and this is the one place the two families do not share a name.
    Sculpt,
    /// A surface pushed by a field, over the datum it left.
    Displace,
}

impl LargeIcon {
    /// Every tile glyph, in catalogue order — sketch first, because sketch→volume is the
    /// authoring atom and the primitives are sugar over it.
    pub const ALL: &'static [LargeIcon] = &[
        LargeIcon::Sketch,
        LargeIcon::Extrude,
        LargeIcon::Revolve,
        LargeIcon::Sweep,
        LargeIcon::BoxSolid,
        LargeIcon::Sphere,
        LargeIcon::Cylinder,
        LargeIcon::HalfSpace,
        LargeIcon::Sculpt,
        LargeIcon::Displace,
    ];

    /// The tile glyph for a rail [`Icon`], if that noun has one.
    ///
    /// `None` is the normal case, not an error: only producers have tile glyphs, so a caller
    /// drawing a shape cell falls back to `icon.draw(..)` at a smaller size. That fallback is
    /// what lets the shape rail hold a verb whose tile mark has not been drawn yet.
    pub fn for_icon(icon: Icon) -> Option<LargeIcon> {
        Some(match icon {
            Icon::Sketch => LargeIcon::Sketch,
            Icon::Extrude => LargeIcon::Extrude,
            Icon::Revolve => LargeIcon::Revolve,
            Icon::Sweep => LargeIcon::Sweep,
            Icon::BoxSolid => LargeIcon::BoxSolid,
            Icon::Sphere => LargeIcon::Sphere,
            Icon::Cylinder => LargeIcon::Cylinder,
            Icon::HalfSpace => LargeIcon::HalfSpace,
            Icon::SculptAdd => LargeIcon::Sculpt,
            Icon::Displace => LargeIcon::Displace,
            _ => return None,
        })
    }

    /// The rail glyph for the same noun. Every tile glyph has one; the reverse does not hold.
    pub fn rail(self) -> Icon {
        match self {
            LargeIcon::Sketch => Icon::Sketch,
            LargeIcon::Extrude => Icon::Extrude,
            LargeIcon::Revolve => Icon::Revolve,
            LargeIcon::Sweep => Icon::Sweep,
            LargeIcon::BoxSolid => Icon::BoxSolid,
            LargeIcon::Sphere => Icon::Sphere,
            LargeIcon::Cylinder => Icon::Cylinder,
            LargeIcon::HalfSpace => Icon::HalfSpace,
            LargeIcon::Sculpt => Icon::SculptAdd,
            LargeIcon::Displace => Icon::Displace,
        }
    }

    /// Paint the glyph into `rect` in `color`. `rect` is the full 26-unit square.
    pub fn draw(self, painter: &Painter, rect: Rect, color: Color32) {
        let g = IconPainter::new_on_grid(painter, rect, color, GRID, STROKE_WIDTH);
        match self {
            LargeIcon::Sketch => g.marks(sketch::DRAW),
            LargeIcon::Extrude => g.marks(extrude::DRAW),
            LargeIcon::Revolve => g.marks(revolve::DRAW),
            LargeIcon::Sweep => g.marks(sweep::DRAW),
            LargeIcon::BoxSolid => g.marks(box_solid::DRAW),
            LargeIcon::Sphere => g.marks(sphere::DRAW),
            LargeIcon::Cylinder => g.marks(cylinder::DRAW),
            LargeIcon::HalfSpace => g.marks(half_space::DRAW),
            LargeIcon::Sculpt => g.marks(sculpt::DRAW),
            LargeIcon::Displace => g.marks(displace::DRAW),
        }
    }

    /// The glyph's kebab-case name.
    ///
    /// Every tile mark answers to its rail twin's noun except [`Sculpt`](Self::Sculpt),
    /// which is the generic brush the rail set does not have — see that variant.
    pub fn name(self) -> &'static str {
        match self {
            LargeIcon::Sculpt => "sculpt",
            other => other.rail().name(),
        }
    }

    /// What the tile carries that its rail twin cannot.
    pub fn note(self) -> &'static str {
        match self {
            LargeIcon::Sketch => {
                "Filled endpoint handles — a sketch is a thing with grabbable control points. \
                 The rail twin goes hollow, because at 15 pt a disc and a ring are the same \
                 three pixels."
            }
            LargeIcon::Extrude => {
                "Depth drawn per vertex rather than as one ground rule, so the mark says the \
                 profile was carried, not that a datum sits under it."
            }
            LargeIcon::Revolve => {
                "Room for the profile to be a closed body rather than a hint, beside the axis \
                 it spins about."
            }
            LargeIcon::Sweep => {
                "Both profiles are squares rather than ticks; the far one stays dashed \
                 because sweep is not a body the app will build yet."
            }
            LargeIcon::BoxSolid => {
                "A real cube in projection with its back edges receding — the fade is x-ray, \
                 the same reading the operand ghosts use."
            }
            LargeIcon::Sphere => {
                "One receding equator carries the roundness. At equal weight it would be a \
                 second silhouette, and the ball would read as a lens."
            }
            LargeIcon::Cylinder => {
                "Both caps whole, the far one faded — a lathe body seen through, not a can."
            }
            LargeIcon::HalfSpace => {
                "Neither line terminates inside the box, which is how the mark says unbounded \
                 without drawing a boundary."
            }
            LargeIcon::Sculpt => {
                "A core inside the reach it stops at — the generic brush the rail set has to \
                 split into add and carve, because at 15 pt polarity must be in the mark."
            }
            LargeIcon::Displace => {
                "The datum stays visible but subordinate, so the mark says the surface MOVED \
                 rather than that it is bumpy."
            }
        }
    }
}

#[cfg(test)]
mod tests;
