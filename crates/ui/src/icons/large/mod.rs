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
mod extrude;
mod half_space;
mod revolve;
mod sketch;
mod sphere;
mod sweep;

/// The authoring grid every tile glyph is traced on: 26 × 26 units.
pub const GRID: f32 = 26.0;

/// The tile stroke, in design points — proportionally about half the rail weight, because a
/// large mark carries its weight through construction rather than through ink.
pub const STROKE_WIDTH: f32 = 1.1;

/// A producer drawn at tile size.
///
/// Only producers have tile glyphs: they are the closed verb set that earns a permanent
/// rail, and they are the marks whose whole content is projection and depth.
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
        }
    }

    /// Paint the glyph into `rect` in `color`. `rect` is the full 26-unit square.
    pub fn draw(self, painter: &Painter, rect: Rect, color: Color32) {
        let g = IconPainter::new_on_grid(painter, rect, color, GRID, STROKE_WIDTH);
        match self {
            LargeIcon::Sketch => sketch::draw(&g),
            LargeIcon::Extrude => extrude::draw(&g),
            LargeIcon::Revolve => revolve::draw(&g),
            LargeIcon::Sweep => sweep::draw(&g),
            LargeIcon::BoxSolid => box_solid::draw(&g),
            LargeIcon::Sphere => sphere::draw(&g),
            LargeIcon::Cylinder => cylinder::draw(&g),
            LargeIcon::HalfSpace => half_space::draw(&g),
        }
    }

    /// The glyph's kebab-case name — the same noun its rail twin answers to.
    pub fn name(self) -> &'static str {
        self.rail().name()
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
        }
    }
}

#[cfg(test)]
mod tests;
