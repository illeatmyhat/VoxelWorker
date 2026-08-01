//! Tests for the glyph families as a whole.

#![allow(
    clippy::duration_subsec,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::match_same_arms,
    clippy::panic,
    clippy::semicolon_if_nothing_returned,
    clippy::unwrap_used,
    clippy::while_float
)]

use std::sync::mpsc;
use std::time::Duration;

use super::design_reference::REFERENCE;
use super::large::LargeIcon;
use super::{Group, Icon, Ink, Mark};

/// How far a transposed coordinate may sit from the sheet's, on the 18-unit grid.
///
/// The sheet carries four decimals on a 36-unit canvas, so halving it leaves 5e-5 of rounding.
/// Anything looser than this is a typo rather than a rounding difference, which is the whole
/// distinction the gate exists to make.
const TOLERANCE: f32 = 2e-3;

/// Every glyph is data, except the three that compose rotated ellipses at paint time.
///
/// [`Icon::draw`] routes anything that is not one of those three through [`Icon::marks`], so an
/// accidentally empty slice would paint a blank button and no other test would notice — the
/// size sweep below is happy to paint nothing 133 times.
#[test]
fn glyphs_are_data() {
    let imperative = [Icon::Orbit, Icon::OrbitConstrained, Icon::OrbitFree];
    for icon in Icon::ALL {
        let expected_empty = imperative.contains(icon);
        assert_eq!(
            icon.marks().is_empty(),
            expected_empty,
            "{}: marks() is {}, which is not what this glyph's form allows",
            icon.name(),
            if expected_empty { "non-empty" } else { "empty" },
        );
    }
}

/// No shelf can be added to the set and left off the catalog.
///
/// [`Group::ALL`] is what the design_reference sheet walks, so a variant missing from it is a
/// shelf of glyphs that exist, gate green, and are invisible to the only reader of the set. Both
/// halves are checked: the match names every variant, so a new one stops this compiling, and
/// every shelf is asserted to actually hold a glyph, because an empty header says nothing.
#[test]
fn every_shelf_is_on_the_catalog_and_holds_a_glyph() {
    for group in Group::ALL {
        // Naming each variant is the compile-time half — a new shelf breaks this arm.
        let named = match group {
            Group::Navigation
            | Group::ViewerModes
            | Group::Combine
            | Group::Fields
            | Group::Producers
            | Group::Structure
            | Group::Tools
            | Group::Sketch
            | Group::SketchCreate
            | Group::SketchModify
            | Group::SketchConstraint
            | Group::SketchDimension
            | Group::SketchOperator
            | Group::Chrome => group,
        };
        assert!(
            Icon::ALL.iter().any(|icon| icon.group() == *named),
            "{} is on the catalog with nothing under it",
            named.title(),
        );
    }
    for icon in Icon::ALL {
        assert!(
            Group::ALL.contains(&icon.group()),
            "{} sits on a shelf the catalog never walks",
            icon.name(),
        );
    }
}

/// Every sketch glyph draws exactly what the design sheet resolved, mark for mark.
///
/// The set is authored twice over — as SVG on the sheet, where the geometry is argued for, and
/// as [`Mark`] data here, where the prose lives — and the second is a hand transposition of the
/// first onto half the grid. That is dozens of coordinates retyped, and a slipped digit produces
/// an icon that is wrong in a way no one notices by looking: a tangency that is nearly a
/// tangency, a node a third of a unit off its own vertex.
///
/// `design_reference.rs` is regenerated from the sheet rather than written, so this compares two
/// independent expressions of the same drawing. A failure means one of them moved.
#[test]
fn glyphs_match_the_design_sheet() {
    for (id, reference) in REFERENCE {
        let icon = Icon::ALL
            .iter()
            .find(|icon| icon.name() == *id)
            .unwrap_or_else(|| panic!("the sheet resolves `{id}`, but no glyph answers to it"));
        let drawn = icon.marks();

        assert_eq!(
            drawn.len(),
            reference.len(),
            "{id}: draws {} marks, the sheet resolves {}",
            drawn.len(),
            reference.len(),
        );
        for (index, (a, b)) in drawn.iter().zip(reference.iter()).enumerate() {
            assert!(
                same(a, b),
                "{id}: mark {index} is not the sheet's\n  glyph: {a:?}\n  sheet: {b:?}",
            );
        }
    }
}

/// Two marks agreeing to [`TOLERANCE`], including their ink — a mark drawn in the wrong ink is
/// as wrong as one drawn in the wrong place, and it is the easier of the two to mistype.
fn same(a: &Mark, b: &Mark) -> bool {
    let near = |x: f32, y: f32| (x - y).abs() <= TOLERANCE;
    let path = |p: &[(f32, f32)], q: &[(f32, f32)]| {
        p.len() == q.len()
            && p.iter()
                .zip(q)
                .all(|(a, b)| near(a.0, b.0) && near(a.1, b.1))
    };
    match (a, b) {
        (
            Mark::Line { points: p, ink: i },
            Mark::Line {
                points: q,
                ink: ink_b,
            },
        )
        | (
            Mark::Closed { points: p, ink: i },
            Mark::Closed {
                points: q,
                ink: ink_b,
            },
        ) => path(p, q) && ink(*i, *ink_b),
        (
            Mark::Rect { a: p, b: q, ink: i },
            Mark::Rect {
                a: r,
                b: s,
                ink: ink_b,
            },
        ) => path(&[*p, *q], &[*r, *s]) && ink(*i, *ink_b),
        (
            Mark::Node {
                center: p,
                size: u,
                ink: i,
            },
            Mark::Node {
                center: q,
                size: v,
                ink: ink_b,
            },
        ) => path(&[*p], &[*q]) && near(*u, *v) && ink(*i, *ink_b),
        (
            Mark::Circle {
                center: p,
                radius: u,
                ink: i,
            },
            Mark::Circle {
                center: q,
                radius: v,
                ink: ink_b,
            },
        )
        | (
            Mark::Disc {
                center: p,
                radius: u,
                ink: i,
            },
            Mark::Disc {
                center: q,
                radius: v,
                ink: ink_b,
            },
        ) => path(&[*p], &[*q]) && near(*u, *v) && ink(*i, *ink_b),
        (
            Mark::Ellipse {
                center: p,
                rx: ux,
                ry: uy,
                ink: i,
            },
            Mark::Ellipse {
                center: q,
                rx: vx,
                ry: vy,
                ink: ink_b,
            },
        ) => path(&[*p], &[*q]) && near(*ux, *vx) && near(*uy, *vy) && ink(*i, *ink_b),
        (
            Mark::Arc {
                center: p,
                rx: ux,
                ry: uy,
                from: f0,
                to: t0,
                ink: i,
            },
            Mark::Arc {
                center: q,
                rx: vx,
                ry: vy,
                from: f1,
                to: t1,
                ink: ink_b,
            },
        ) => {
            path(&[*p], &[*q])
                && near(*ux, *vx)
                && near(*uy, *vy)
                && near(*f0, *f1)
                && near(*t0, *t1)
                && ink(*i, *ink_b)
        }
        (
            Mark::Cubic {
                p0: a0,
                p1: a1,
                p2: a2,
                p3: a3,
                ink: i,
            },
            Mark::Cubic {
                p0: b0,
                p1: b1,
                p2: b2,
                p3: b3,
                ink: ink_b,
            },
        ) => path(&[*a0, *a1, *a2, *a3], &[*b0, *b1, *b2, *b3]) && ink(*i, *ink_b),
        _ => false,
    }
}

/// Ink equality. Opacity is compared loosely for the same reason coordinates are.
fn ink(a: Ink, b: Ink) -> bool {
    a.role == b.role && a.dashed == b.dashed && (a.opacity - b.opacity).abs() <= TOLERANCE
}

/// Every glyph in BOTH families paints across the whole range of sizes the app and the design
/// reference render at — and, more to the point, terminates.
///
/// The size sweep is the whole value of this test. A [`IconPainter::dash_path`] that walks a
/// polyline by advancing a cursor lands exactly on a dash boundary at every step; wherever
/// rounding then puts the phase one ULP short of the boundary, the next advance (~1e-7) is
/// smaller than the cursor's own precision, the cursor stops moving, and the walk spins forever.
/// That form reproduced at 184 of 532 simulated size/grid combinations — including a plain
/// dashed line at 18 pt, the rail's own natural size — and hung `design_reference` on a white
/// window with no output at all. A two-size smoke test sails straight past it.
///
/// The work runs on its own thread behind a watchdog because the failure mode is a HANG, not a
/// wrong pixel: an assertion at the end of a stalled loop is never reached.
#[test]
fn every_glyph_paints_at_every_size_and_terminates() {
    let (painted, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let context = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(400.0, 400.0),
            )),
            ..Default::default()
        };
        let _ = context.run_ui(raw_input, |ui| {
            for size in 8..=140 {
                let rect =
                    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(size as f32));
                for icon in Icon::ALL {
                    icon.draw(ui.painter(), rect, egui::Color32::WHITE);
                }
                for large in LargeIcon::ALL {
                    large.draw(ui.painter(), rect, egui::Color32::WHITE);
                }
            }
        });
        let _ = painted.send(());
    });

    match finished.recv_timeout(Duration::from_mins(1)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("a glyph never finished painting — the dash walk is stalling again")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("a glyph panicked while painting")
        }
    }
}
