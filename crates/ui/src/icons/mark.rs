//! A glyph as DATA rather than as code.
//!
//! A glyph used to be a `fn(&IconPainter)` — readable, but executable, and executable code can
//! fail to terminate. It did: a float-accumulating dash walk hung the whole reference binary on
//! a white window. A glyph is now a `&'static [Mark]`, so a glyph file contains no control flow
//! at all and the closure property is a fact about the TYPE rather than a convention someone has
//! to keep. There is no loop to write, so there is no loop to get wrong.
//!
//! [`Mark`] deliberately describes no geometry of its own: every variant dispatches to the
//! [`IconPainter`] method the imperative form already called, with the same arguments. The
//! size-adaptive sampling, the stroke floor and the per-family grid all stay exactly where they
//! were, which is what makes the migration provably a change of representation and nothing else.

use super::{IconPainter, Stroke};

/// How a mark is inked: whether it dashes, and how far it is faded back.
///
/// Dashing and fading are independent — a receding operand edge is both — so this is a product
/// and not an enum of the four combinations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ink {
    /// Whether the mark follows the set's dash rhythm (2.2 on, 1.8 off, in grid units).
    pub dashed: bool,
    /// Opacity multiplier, 1.0 being the glyph's full stroke.
    pub opacity: f32,
}

impl Ink {
    /// The glyph's own stroke, at full weight.
    pub const SOLID: Ink = Ink {
        dashed: false,
        opacity: 1.0,
    };

    /// The set's dash rhythm: "authored, but not what you are looking at" — an operand, an
    /// envelope, a fold entry that lost.
    pub const DASHED: Ink = Ink {
        dashed: true,
        opacity: 1.0,
    };

    /// Faded back to `opacity` — the set's one legitimate use of opacity, for a receding edge or
    /// a datum that must not compete with the subject.
    pub const fn faint(opacity: f32) -> Ink {
        Ink {
            dashed: false,
            opacity,
        }
    }

    /// Dashed and faded at once.
    pub const fn faint_dashed(opacity: f32) -> Ink {
        Ink {
            dashed: true,
            opacity,
        }
    }

    /// Resolve against a painter. Full opacity takes the painter's own stroke rather than a
    /// `gamma_multiply(1.0)` of it, so a solid mark is bit-for-bit what it always was.
    fn stroke(self, g: &IconPainter) -> Stroke {
        if self.opacity >= 1.0 {
            g.stroke()
        } else {
            g.faint(self.opacity)
        }
    }
}

/// One mark in a glyph. A glyph is a `&'static [Mark]`, painted in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mark {
    /// An open polyline through grid points.
    Line {
        points: &'static [(f32, f32)],
        ink: Ink,
    },
    /// A closed polygon outline.
    Closed {
        points: &'static [(f32, f32)],
        ink: Ink,
    },
    /// A filled convex polygon — a mark stating a REGION rather than an edge. Several faces of
    /// one solid at descending opacities read as a lit body, which no outline can say.
    Fill {
        points: &'static [(f32, f32)],
        opacity: f32,
    },
    /// The axis-aligned rectangle spanned by two corners.
    Rect {
        a: (f32, f32),
        b: (f32, f32),
        ink: Ink,
    },
    /// A circle outline.
    Circle {
        center: (f32, f32),
        radius: f32,
        ink: Ink,
    },
    /// A solid disc — a mark too small to be a ring. At 15 pt a two-pixel ring is mush where a
    /// two-pixel dot is crisp.
    Disc { center: (f32, f32), radius: f32 },
    /// An axis-aligned ellipse outline — the set's roundness mark.
    Ellipse {
        center: (f32, f32),
        rx: f32,
        ry: f32,
        ink: Ink,
    },
    /// An elliptical arc, angles in radians, clockwise from +x (y grows downward, matching the
    /// SVG the set was authored in).
    Arc {
        center: (f32, f32),
        rx: f32,
        ry: f32,
        from: f32,
        to: f32,
        ink: Ink,
    },
    /// A cubic Bézier through its four control points.
    Cubic {
        p0: (f32, f32),
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
        ink: Ink,
    },
}

impl Mark {
    /// Paint this mark. Every arm is a dispatch to the method the imperative form called — no
    /// arithmetic happens here, which is what keeps the two forms byte-identical.
    pub(super) fn paint(&self, g: &IconPainter) {
        match *self {
            Mark::Line { points, ink } => g.polyline_inked(points, ink),
            Mark::Closed { points, ink } => {
                let mut looped: Vec<(f32, f32)> = points.to_vec();
                if let Some(&first) = points.first() {
                    looped.push(first);
                }
                g.polyline_inked(&looped, ink);
            }
            Mark::Fill { points, opacity } => {
                if opacity >= 1.0 {
                    g.fill(points);
                } else {
                    g.fill_with(points, g.faint(opacity).color);
                }
            }
            Mark::Rect { a, b, ink } => {
                if ink.dashed {
                    g.dashed_rect_with(a, b, ink.stroke(g));
                } else {
                    g.rect_with(a, b, ink.stroke(g));
                }
            }
            // Circles and ellipses dash through `dashed_ellipse_with`, which samples more
            // finely (24..96) than the arc path (12..64). Keeping that split is deliberate:
            // it is what the imperative set already did, and parity is the point.
            Mark::Circle { center, radius, ink } => {
                if ink.dashed {
                    g.dashed_ellipse_with(center, radius, radius, ink.stroke(g));
                } else {
                    g.circle_with(center, radius, ink.stroke(g));
                }
            }
            Mark::Disc { center, radius } => g.filled_circle(center, radius),
            Mark::Ellipse {
                center,
                rx,
                ry,
                ink,
            } => {
                if ink.dashed {
                    g.dashed_ellipse_with(center, rx, ry, ink.stroke(g));
                } else {
                    g.ellipse_with(center, rx, ry, ink.stroke(g));
                }
            }
            Mark::Arc {
                center,
                rx,
                ry,
                from,
                to,
                ink,
            } => g.arc_inked(center, rx, ry, from, to, ink),
            Mark::Cubic {
                p0,
                p1,
                p2,
                p3,
                ink,
            } => g.cubic_inked(p0, p1, p2, p3, ink),
        }
    }
}

impl IconPainter<'_> {
    /// Paint a glyph: its marks, in order.
    pub fn marks(&self, marks: &[Mark]) {
        for mark in marks {
            mark.paint(self);
        }
    }

    /// Stroke a polyline solid or dashed, as the ink says.
    pub(super) fn polyline_inked(&self, points: &[(f32, f32)], ink: Ink) {
        let stroke = ink.stroke(self);
        if ink.dashed {
            self.dashed_polyline_with(points, stroke);
        } else {
            self.line_with(points, stroke);
        }
    }

    /// Stroke an arc solid or dashed.
    pub(super) fn arc_inked(
        &self,
        center: (f32, f32),
        rx: f32,
        ry: f32,
        from: f32,
        to: f32,
        ink: Ink,
    ) {
        self.polyline_inked(&self.arc_points(center, rx, ry, from, to), ink);
    }

    /// Trace a cubic Bézier solid or dashed.
    pub(super) fn cubic_inked(
        &self,
        p0: (f32, f32),
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
        ink: Ink,
    ) {
        self.polyline_inked(&self.cubic_points(p0, p1, p2, p3), ink);
    }
}
