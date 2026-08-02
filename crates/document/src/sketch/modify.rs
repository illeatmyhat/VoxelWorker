//! Topology-preserving sketch modification adapters.
//!
//! Continuous curve intersection and splitting belong to `substrate`; this module is the document
//! boundary that maps stable sketch entities into those curves and writes the resulting pieces
//! back without flattening arcs. Modifier previews and commits consume the same placement value,
//! so a hover cannot promise a different cut from the one an undoable click performs.

use super::{
    AngleMeasurement, Arc, ArcSweep, EntityId, EntityRole, Segment, Sketch, SketchCurve,
    SketchPoint, SketchSolid, ABSENT_CENTER,
};
use substrate::curve_intersection::PlanarCurve;

/// Canonical pieces a Break command will persist in place of `source`.
#[derive(Debug, Clone, PartialEq)]
pub struct BreakPlacement {
    pub source: SketchCurve,
    pub pieces: Vec<PlanarCurve>,
}

/// Why a curve cannot be broken against the rest of the sketch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakRefusal {
    UnknownCurve,
    NoInteriorIntersection,
    Unrepresentable,
}

impl SketchSolid {
    /// Resolve every intersection of `source` with the other authored curves, retaining native
    /// line/arc pieces. End-only contacts do not break an already-open curve.
    pub fn break_placement(
        &self,
        source: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Result<BreakPlacement, BreakRefusal> {
        let source_curve = self
            .sketch
            .planar_curve(source, context)
            .ok_or(BreakRefusal::UnknownCurve)?;
        let mut cuts = Vec::new();
        for other in self.sketch.curves() {
            if other == source {
                continue;
            }
            let Some(other_curve) = self.sketch.planar_curve(other, context) else {
                continue;
            };
            cuts.extend(
                source_curve
                    .crossings(&other_curve)
                    .into_iter()
                    .map(|crossing| crossing.parameter_on_first),
            );
        }
        let pieces = source_curve.split_at(&cuts);
        if pieces.len() <= 1 {
            return Err(BreakRefusal::NoInteriorIntersection);
        }
        Ok(BreakPlacement { source, pieces })
    }

    /// Atomically replace one curve by the pieces in its canonical Break placement.
    pub fn with_curve_broken(
        &self,
        source: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, BreakRefusal> {
        let placement = self.break_placement(source, context)?;
        let mut next = self.clone();
        next.sketch.apply_break(&placement)?;
        Ok(next)
    }
}

impl Sketch {
    fn curves(&self) -> impl Iterator<Item = SketchCurve> + '_ {
        self.segments
            .iter()
            .map(|segment| SketchCurve::Segment(segment.id))
            .chain(self.arcs.iter().map(|arc| SketchCurve::Arc(arc.id)))
            .chain(
                self.circles
                    .iter()
                    .map(|circle| SketchCurve::Circle(circle.id)),
            )
    }

    fn planar_curve(
        &self,
        curve: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Option<PlanarCurve> {
        match self.curve_geometry(curve, context)? {
            parametric::sketch::CurveGeometry::Segment { from, to } => Some(PlanarCurve::Segment {
                start: from,
                end: to,
            }),
            parametric::sketch::CurveGeometry::Circular(circular) => circular.arc.map_or_else(
                || Some(PlanarCurve::circle(circular.center, circular.radius)),
                |arc| {
                    let offset = [
                        arc.from[0] - circular.center[0],
                        arc.from[1] - circular.center[1],
                    ];
                    Some(PlanarCurve::Arc {
                        center: circular.center,
                        radius: circular.radius,
                        start_radians: offset[1].atan2(offset[0]),
                        sweep_radians: arc.sweep_radians,
                    })
                },
            ),
        }
    }

    fn apply_break(&mut self, placement: &BreakPlacement) -> Result<(), BreakRefusal> {
        if placement.pieces.len() <= 1 {
            return Err(BreakRefusal::NoInteriorIntersection);
        }
        match placement.source {
            SketchCurve::Segment(id) => self.break_segment(id, &placement.pieces),
            SketchCurve::Arc(id) => self.break_arc(id, &placement.pieces),
            SketchCurve::Circle(id) => self.break_circle(id, &placement.pieces),
        }?;
        self.sync_arc_centers();
        self.prune_orphan_centers();
        self.drop_dangling_constraints();
        Ok(())
    }

    fn break_segment(&mut self, id: EntityId, pieces: &[PlanarCurve]) -> Result<(), BreakRefusal> {
        let index = self
            .segments
            .iter()
            .position(|segment| segment.id == id)
            .ok_or(BreakRefusal::UnknownCurve)?;
        let source = self.segments[index];
        let boundaries = self.open_piece_boundaries(pieces, source.from, source.to, source.role)?;
        self.segments[index].to = boundaries[1];
        for pair in boundaries[1..].array_windows::<2>() {
            let id = self.alloc_id();
            self.segments.push(Segment {
                id,
                from: pair[0],
                to: pair[1],
                origin: source.origin,
                role: source.role,
            });
        }
        Ok(())
    }

    fn break_arc(&mut self, id: EntityId, pieces: &[PlanarCurve]) -> Result<(), BreakRefusal> {
        let index = self
            .arcs
            .iter()
            .position(|arc| arc.id == id)
            .ok_or(BreakRefusal::UnknownCurve)?;
        let source = self.arcs[index];
        let boundaries = self.open_piece_boundaries(pieces, source.from, source.to, source.role)?;
        let first_sweep = piece_sweep(pieces.first().ok_or(BreakRefusal::Unrepresentable)?)?;
        self.arcs[index].to = boundaries[1];
        self.arcs[index].bulge = ArcSweep::free(first_sweep);
        for (piece, pair) in pieces[1..].iter().zip(boundaries[1..].array_windows::<2>()) {
            let id = self.alloc_id();
            self.arcs.push(Arc {
                id,
                from: pair[0],
                to: pair[1],
                bulge: ArcSweep::free(piece_sweep(piece)?),
                center: ABSENT_CENTER,
                origin: source.origin,
                role: source.role,
            });
        }
        Ok(())
    }

    fn break_circle(&mut self, id: EntityId, pieces: &[PlanarCurve]) -> Result<(), BreakRefusal> {
        let index = self
            .circles
            .iter()
            .position(|circle| circle.id == id)
            .ok_or(BreakRefusal::UnknownCurve)?;
        let source = self.circles.remove(index);
        let mut boundaries = Vec::with_capacity(pieces.len());
        for piece in pieces {
            boundaries.push(self.point_for_break(piece.start(), source.role)?);
        }
        for (piece_index, piece) in pieces.iter().enumerate() {
            let next_index = piece_index.saturating_add(1) % boundaries.len();
            let id = self.alloc_id();
            self.arcs.push(Arc {
                id,
                from: boundaries[piece_index],
                to: boundaries[next_index],
                bulge: ArcSweep::free(piece_sweep(piece)?),
                center: ABSENT_CENTER,
                origin: source.origin,
                role: source.role,
            });
        }
        Ok(())
    }

    fn open_piece_boundaries(
        &mut self,
        pieces: &[PlanarCurve],
        from: EntityId,
        to: EntityId,
        role: EntityRole,
    ) -> Result<Vec<EntityId>, BreakRefusal> {
        let mut boundaries = Vec::with_capacity(pieces.len().saturating_add(1));
        boundaries.push(from);
        for piece in pieces.iter().take(pieces.len().saturating_sub(1)) {
            boundaries.push(self.point_for_break(piece.end(), role)?);
        }
        boundaries.push(to);
        Ok(boundaries)
    }

    fn point_for_break(
        &mut self,
        point: [f64; 2],
        role: EntityRole,
    ) -> Result<EntityId, BreakRefusal> {
        let point = SketchPoint::try_from_continuous(point[0], point[1])
            .map_err(|_| BreakRefusal::Unrepresentable)?;
        if let Some(existing) = self.point_at(point) {
            return Ok(existing);
        }
        let id = self.add_point(point);
        if let Some(stored) = self.points.iter_mut().find(|stored| stored.id == id) {
            stored.role = role;
        }
        Ok(id)
    }
}

fn piece_sweep(piece: &PlanarCurve) -> Result<AngleMeasurement, BreakRefusal> {
    let PlanarCurve::Arc { sweep_radians, .. } = *piece else {
        return Err(BreakRefusal::Unrepresentable);
    };
    AngleMeasurement::try_from_degrees_f64(sweep_radians.to_degrees())
        .map_err(|_| BreakRefusal::Unrepresentable)
}
