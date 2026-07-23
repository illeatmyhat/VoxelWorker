//! `axis_guide` — the dashed extension line a vertex snapped to an in-plane axis runs along.

use egui::{Painter, Pos2, Stroke};

use super::{dashed, Axis, STROKE_GUIDE};

/// A **snap axis guide** — the dashed extension line in the [`Axis`] colour (X `#d9603f` /
/// Y `#7dba6a` / Z accent) that a vertex snapped to an in-plane axis runs along. Axis-lock and
/// equal length fall out of the same lattice, so the guide's colour IS the constraint it stands
/// in for (ADR 0028 §5) — the by-product a solver would have called an alignment constraint.
pub fn axis_guide(painter: &Painter, a: Pos2, b: Pos2, axis: Axis) {
    dashed(painter, a, b, Stroke::new(STROKE_GUIDE, axis.color()));
}
