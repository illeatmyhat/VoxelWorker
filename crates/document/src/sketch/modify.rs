//! Topology-preserving sketch modification adapters.
//!
//! Continuous curve intersection and splitting belong to `substrate`; this module is the document
//! boundary that maps stable sketch entities into those curves and writes the resulting pieces
//! back without flattening arcs. Modifier previews and commits consume the same placement value,
//! so a hover cannot promise a different cut from the one an undoable click performs.

use super::{
    boxed_push, boxed_retain, AngleMeasurement, Arc, ArcSweep, ConstraintKind, EntityId,
    EntityRole, Segment, Sketch, SketchCurve, SketchLength, SketchPoint, SketchSolid,
    ABSENT_CENTER,
};
use substrate::curve_intersection::{CurveSupportCrossing, PlanarCurve};

const EXTEND_EPSILON: f64 = 1.0e-9;

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

/// Canonical result of trimming the interval nearest the click witness.
#[derive(Debug, Clone, PartialEq)]
pub struct TrimPlacement {
    pub source: SketchCurve,
    pub removed: PlanarCurve,
    pub kept: Vec<PlanarCurve>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimRefusal {
    UnknownCurve,
    Unrepresentable,
}

/// Which authored end of an open curve an Extend operation grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendEndpoint {
    Start,
    End,
}

/// Canonical native curve after extending one of `source`'s endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendPlacement {
    pub source: SketchCurve,
    pub endpoint: ExtendEndpoint,
    pub extended: PlanarCurve,
}

/// Why an authored curve cannot be extended in the requested direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendRefusal {
    UnknownCurve,
    ClosedCurve,
    NoIntersection,
    FixedSweep,
    Unrepresentable,
}

/// Canonical line-line corner replacement produced by Fillet.
#[derive(Debug, Clone, PartialEq)]
pub struct FilletPlacement {
    pub first: SketchCurve,
    pub second: SketchCurve,
    pub shortened_first: PlanarCurve,
    pub shortened_second: PlanarCurve,
    pub arc: PlanarCurve,
    corner: EntityId,
    first_endpoint: ExtendEndpoint,
    second_endpoint: ExtendEndpoint,
}

/// Why a clicked corner cannot be rounded without changing unrelated topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilletRefusal {
    UnknownCurve,
    UnsupportedCurve,
    AmbiguousCorner,
    RoleMismatch,
    RadiusOutOfRange,
    Unrepresentable,
    Constraint,
}

/// Canonical two-line corner replacement produced by any Chamfer input grammar.
#[derive(Debug, Clone, PartialEq)]
pub struct ChamferPlacement {
    pub first: SketchCurve,
    pub second: SketchCurve,
    pub shortened_first: PlanarCurve,
    pub shortened_second: PlanarCurve,
    pub connector: PlanarCurve,
    corner: EntityId,
    second_endpoint: ExtendEndpoint,
}

/// Why a chamfer could not be derived or written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChamferRefusal {
    Corner(FilletRefusal),
    DistanceOutOfRange,
    Unrepresentable,
}

/// Canonical native copy produced by offsetting one authored curve.
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetPlacement {
    pub source: SketchCurve,
    pub offset: PlanarCurve,
}

/// Why a curve has no representable offset at the supplied witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetRefusal {
    UnknownCurve,
    ZeroDistance,
    Degenerate,
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

    /// Resolve the finite interval of `source` nearest `witness` between adjacent crossings.
    /// With no crossing the whole curve is the interval, matching Fusion's delete-on-no-crossing
    /// Trim behavior.
    pub fn trim_placement(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<TrimPlacement, TrimRefusal> {
        let source_curve = self
            .sketch
            .planar_curve(source, context)
            .ok_or(TrimRefusal::UnknownCurve)?;
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
        let mut pieces = source_curve.split_at(&cuts);
        let remove_index = pieces
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                distance_to_curve(first, witness).total_cmp(&distance_to_curve(second, witness))
            })
            .map(|(index, _)| index)
            .ok_or(TrimRefusal::UnknownCurve)?;
        let removed = pieces.remove(remove_index);
        Ok(TrimPlacement {
            source,
            removed,
            kept: pieces,
        })
    }

    /// Atomically remove the clicked Trim interval and persist the remaining native pieces.
    pub fn with_curve_trimmed(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, TrimRefusal> {
        let placement = self.trim_placement(source, witness, context)?;
        let mut next = self.clone();
        next.sketch
            .replace_curve_with_pieces(placement.source, &placement.kept)
            .map_err(|_| TrimRefusal::Unrepresentable)?;
        Ok(next)
    }

    /// Grow the endpoint nearest `witness` to the first isolated meeting with another authored
    /// finite curve. A segment follows its supporting ray; an arc follows its supporting circle in
    /// its existing signed direction. Closed circles have no endpoint and cannot be extended.
    pub fn extend_placement(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<ExtendPlacement, ExtendRefusal> {
        let source_curve = self
            .sketch
            .planar_curve(source, context)
            .ok_or(ExtendRefusal::UnknownCurve)?;
        if source_curve.is_closed() {
            return Err(ExtendRefusal::ClosedCurve);
        }
        if let SketchCurve::Arc(id) = source {
            let arc = self
                .sketch
                .arcs
                .iter()
                .find(|arc| arc.id == id)
                .ok_or(ExtendRefusal::UnknownCurve)?;
            // Extending an arc necessarily changes its included angle. A measurement-backed
            // sweep owns that value, so the preview must refuse here instead of promising a
            // placement the commit would later be unable to persist.
            if arc.bulge.free_value().is_none() {
                return Err(ExtendRefusal::FixedSweep);
            }
        }
        let endpoint = if squared_distance(witness, source_curve.start())
            <= squared_distance(witness, source_curve.end())
        {
            ExtendEndpoint::Start
        } else {
            ExtendEndpoint::End
        };
        let mut candidates = Vec::new();
        for other in self.sketch.curves() {
            if other == source {
                continue;
            }
            let Some(other_curve) = self.sketch.planar_curve(other, context) else {
                continue;
            };
            candidates.extend(
                source_curve
                    .support_crossings_with(&other_curve)
                    .into_iter()
                    .filter(|crossing| !crossing.overlapping)
                    .filter_map(|crossing| extension_candidate(&source_curve, endpoint, crossing)),
            );
        }
        let (_, extended) = candidates
            .into_iter()
            .min_by(|(first, _), (second, _)| first.total_cmp(second))
            .ok_or(ExtendRefusal::NoIntersection)?;
        Ok(ExtendPlacement {
            source,
            endpoint,
            extended,
        })
    }

    /// Atomically persist the native curve described by [`Self::extend_placement`].
    pub fn with_curve_extended(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, ExtendRefusal> {
        let placement = self.extend_placement(source, witness, context)?;
        let mut next = self.clone();
        next.sketch.apply_extend(&placement)?;
        Ok(next)
    }

    /// Round the endpoint nearest `witness` where exactly two same-role line segments meet.
    /// The witness's projection down the clicked leg chooses the tangent distance and therefore
    /// the radius, so preview and commit need no hidden default length.
    pub fn fillet_placement(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<FilletPlacement, FilletRefusal> {
        let SketchCurve::Segment(source_id) = source else {
            return Err(FilletRefusal::UnsupportedCurve);
        };
        let source_segment = self
            .sketch
            .segments
            .iter()
            .find(|segment| segment.id == source_id)
            .ok_or(FilletRefusal::UnknownCurve)?;
        let source_curve = self
            .sketch
            .planar_curve(source, context)
            .ok_or(FilletRefusal::UnknownCurve)?;
        let (corner, first_endpoint) = if squared_distance(witness, source_curve.start())
            <= squared_distance(witness, source_curve.end())
        {
            (source_segment.from, ExtendEndpoint::Start)
        } else {
            (source_segment.to, ExtendEndpoint::End)
        };
        let incident: Vec<_> = self
            .sketch
            .segments
            .iter()
            .filter(|segment| segment.id != source_id)
            .filter(|segment| segment.from == corner || segment.to == corner)
            .collect();
        if incident.len() != 1 || self.sketch.non_segment_uses_point(corner) {
            return Err(FilletRefusal::AmbiguousCorner);
        }
        let second_segment = incident[0];
        if second_segment.role != source_segment.role {
            return Err(FilletRefusal::RoleMismatch);
        }
        let second = SketchCurve::Segment(second_segment.id);
        let second_curve = self
            .sketch
            .planar_curve(second, context)
            .ok_or(FilletRefusal::UnknownCurve)?;
        let second_endpoint = if second_segment.from == corner {
            ExtendEndpoint::Start
        } else {
            ExtendEndpoint::End
        };
        line_fillet_geometry(
            source,
            second,
            source_curve,
            second_curve,
            corner,
            first_endpoint,
            second_endpoint,
            witness,
        )
    }

    /// Atomically replace one two-line corner by its native tangent arc.
    pub fn with_corner_filleted(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, FilletRefusal> {
        let placement = self.fillet_placement(source, witness, context)?;
        let mut next = self.clone();
        next.sketch.apply_fillet(&placement, context)?;
        Ok(next)
    }

    /// Bevel the same two-line corner Fillet recognizes. `second_witness == None` uses the first
    /// leg's tangent distance on both legs (Equal Distance). Supplying a second witness projects it
    /// onto the other leg, which is the shared geometric result of the Two Distance and
    /// Distance/Angle input grammars.
    pub fn chamfer_placement(
        &self,
        source: SketchCurve,
        first_witness: [f64; 2],
        second_witness: Option<[f64; 2]>,
        context: parametric::EvaluationContext,
    ) -> Result<ChamferPlacement, ChamferRefusal> {
        let base = self
            .fillet_placement(source, first_witness, context)
            .map_err(ChamferRefusal::Corner)?;
        let first_tangent = match base.first_endpoint {
            ExtendEndpoint::Start => base.shortened_first.start(),
            ExtendEndpoint::End => base.shortened_first.end(),
        };
        let equal_second_tangent = match base.second_endpoint {
            ExtendEndpoint::Start => base.shortened_second.start(),
            ExtendEndpoint::End => base.shortened_second.end(),
        };
        let (shortened_second, second_tangent) = if let Some(witness) = second_witness {
            let corner = self
                .sketch
                .points
                .iter()
                .find(|point| point.id == base.corner)
                .map(|point| point.at.in_plane())
                .ok_or(ChamferRefusal::Corner(FilletRefusal::UnknownCurve))?;
            let far = match base.second_endpoint {
                ExtendEndpoint::Start => base.shortened_second.end(),
                ExtendEndpoint::End => base.shortened_second.start(),
            };
            let span = [far[0] - corner[0], far[1] - corner[1]];
            let length = span[0].hypot(span[1]);
            if length <= EXTEND_EPSILON {
                return Err(ChamferRefusal::DistanceOutOfRange);
            }
            let unit = [span[0] / length, span[1] / length];
            let from_corner = [witness[0] - corner[0], witness[1] - corner[1]];
            let distance = unit[0].mul_add(from_corner[0], unit[1] * from_corner[1]);
            if distance <= EXTEND_EPSILON || distance >= length - EXTEND_EPSILON {
                return Err(ChamferRefusal::DistanceOutOfRange);
            }
            let tangent = [
                unit[0].mul_add(distance, corner[0]),
                unit[1].mul_add(distance, corner[1]),
            ];
            let shortened = match base.second_endpoint {
                ExtendEndpoint::Start => PlanarCurve::Segment {
                    start: tangent,
                    end: far,
                },
                ExtendEndpoint::End => PlanarCurve::Segment {
                    start: far,
                    end: tangent,
                },
            };
            (shortened, tangent)
        } else {
            (base.shortened_second, equal_second_tangent)
        };
        Ok(ChamferPlacement {
            first: base.first,
            second: base.second,
            shortened_first: base.shortened_first,
            shortened_second,
            connector: PlanarCurve::Segment {
                start: first_tangent,
                end: second_tangent,
            },
            corner: base.corner,
            second_endpoint: base.second_endpoint,
        })
    }

    /// Atomically persist a canonical Chamfer placement.
    pub fn with_corner_chamfered(
        &self,
        source: SketchCurve,
        first_witness: [f64; 2],
        second_witness: Option<[f64; 2]>,
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, ChamferRefusal> {
        let placement = self.chamfer_placement(source, first_witness, second_witness, context)?;
        let mut next = self.clone();
        next.sketch.apply_chamfer(&placement)?;
        Ok(next)
    }

    /// Construct a native parallel/concentric copy through the distance indicated by `witness`.
    /// A line reads signed perpendicular distance; a circular curve reads the witness's radial
    /// distance from its center. The source remains untouched.
    pub fn offset_placement(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<OffsetPlacement, OffsetRefusal> {
        let source_curve = self
            .sketch
            .planar_curve(source, context)
            .ok_or(OffsetRefusal::UnknownCurve)?;
        let offset = match source_curve {
            PlanarCurve::Segment { start, end } => {
                let span = [end[0] - start[0], end[1] - start[1]];
                let length = span[0].hypot(span[1]);
                if length <= EXTEND_EPSILON {
                    return Err(OffsetRefusal::Degenerate);
                }
                let normal = [-span[1] / length, span[0] / length];
                let from_start = [witness[0] - start[0], witness[1] - start[1]];
                let distance = normal[0].mul_add(from_start[0], normal[1] * from_start[1]);
                if distance.abs() <= EXTEND_EPSILON {
                    return Err(OffsetRefusal::ZeroDistance);
                }
                let shift = [normal[0] * distance, normal[1] * distance];
                PlanarCurve::Segment {
                    start: [start[0] + shift[0], start[1] + shift[1]],
                    end: [end[0] + shift[0], end[1] + shift[1]],
                }
            }
            PlanarCurve::Arc {
                center,
                radius,
                start_radians,
                sweep_radians,
            } => {
                let new_radius = (witness[0] - center[0]).hypot(witness[1] - center[1]);
                if new_radius <= EXTEND_EPSILON {
                    return Err(OffsetRefusal::Degenerate);
                }
                if (new_radius - radius).abs() <= EXTEND_EPSILON {
                    return Err(OffsetRefusal::ZeroDistance);
                }
                PlanarCurve::Arc {
                    center,
                    radius: new_radius,
                    start_radians,
                    sweep_radians,
                }
            }
            PlanarCurve::RationalBezier(_) => return Err(OffsetRefusal::Unrepresentable),
        };
        Ok(OffsetPlacement { source, offset })
    }

    /// Atomically append the native curve described by [`Self::offset_placement`].
    pub fn with_curve_offset(
        &self,
        source: SketchCurve,
        witness: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<SketchSolid, OffsetRefusal> {
        let placement = self.offset_placement(source, witness, context)?;
        let mut next = self.clone();
        next.sketch.apply_offset(&placement)?;
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
            .chain(
                self.beziers
                    .iter()
                    .map(|bezier| SketchCurve::Bezier(bezier.id)),
            )
    }

    pub(super) fn planar_curve(
        &self,
        curve: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Option<PlanarCurve> {
        if let SketchCurve::Bezier(id) = curve {
            let bezier = self.beziers.iter().find(|bezier| bezier.id == id)?;
            return Some(PlanarCurve::RationalBezier(
                self.rational_bezier_from(bezier.controls, bezier.weights)?,
            ));
        }
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
            SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => {
                self.replace_curve_with_pieces(placement.source, &placement.pieces)
            }
        }?;
        self.sync_arc_centers();
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
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

    fn replace_curve_with_pieces(
        &mut self,
        source: SketchCurve,
        pieces: &[PlanarCurve],
    ) -> Result<(), BreakRefusal> {
        let (origin, role) = self.curve_origin_role(source)?;
        self.remove_curve(source);
        for piece in pieces {
            let from = self.point_for_break(piece.start(), role)?;
            let to = self.point_for_break(piece.end(), role)?;
            let id = self.alloc_id();
            match *piece {
                PlanarCurve::Segment { .. } => self.segments.push(Segment {
                    id,
                    from,
                    to,
                    origin,
                    role,
                }),
                PlanarCurve::Arc { .. } => self.arcs.push(Arc {
                    id,
                    from,
                    to,
                    bulge: ArcSweep::free(piece_sweep(piece)?),
                    center: ABSENT_CENTER,
                    origin,
                    role,
                }),
                PlanarCurve::RationalBezier(curve) => {
                    let first_handle =
                        SketchPoint::try_from_continuous(curve.control[1][0], curve.control[1][1])
                            .map_err(|_| BreakRefusal::Unrepresentable)?;
                    let second_handle =
                        SketchPoint::try_from_continuous(curve.control[2][0], curve.control[2][1])
                            .map_err(|_| BreakRefusal::Unrepresentable)?;
                    let first_handle = self.add_construction_point(first_handle);
                    let second_handle = self.add_construction_point(second_handle);
                    boxed_push(
                        &mut self.beziers,
                        super::Bezier {
                            id,
                            controls: [from, first_handle, second_handle, to],
                            weights: curve.weights,
                            origin,
                            role,
                        },
                    );
                }
            }
        }
        self.sync_arc_centers();
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
        Ok(())
    }

    fn curve_origin_role(
        &self,
        source: SketchCurve,
    ) -> Result<(EntityId, EntityRole), BreakRefusal> {
        let held = match source {
            SketchCurve::Segment(id) => self
                .segments
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Arc(id) => self
                .arcs
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Circle(id) => self
                .circles
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Bezier(id) => self
                .beziers
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Ellipse(id) => self
                .ellipses
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Conic(id) => self
                .conics
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
            SketchCurve::Spline(id) => self
                .splines
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| (curve.origin, curve.role)),
        };
        held.ok_or(BreakRefusal::UnknownCurve)
    }

    fn remove_curve(&mut self, source: SketchCurve) {
        match source {
            SketchCurve::Segment(id) => self.segments.retain(|curve| curve.id != id),
            SketchCurve::Arc(id) => self.arcs.retain(|curve| curve.id != id),
            SketchCurve::Circle(id) => self.circles.retain(|curve| curve.id != id),
            SketchCurve::Bezier(id) => boxed_retain(&mut self.beziers, |curve| curve.id != id),
            SketchCurve::Ellipse(id) => boxed_retain(&mut self.ellipses, |curve| curve.id != id),
            SketchCurve::Conic(id) => boxed_retain(&mut self.conics, |curve| curve.id != id),
            SketchCurve::Spline(id) => boxed_retain(&mut self.splines, |curve| curve.id != id),
        }
    }

    fn apply_extend(&mut self, placement: &ExtendPlacement) -> Result<(), ExtendRefusal> {
        let endpoint = SketchPoint::try_from_continuous(
            match placement.endpoint {
                ExtendEndpoint::Start => placement.extended.start()[0],
                ExtendEndpoint::End => placement.extended.end()[0],
            },
            match placement.endpoint {
                ExtendEndpoint::Start => placement.extended.start()[1],
                ExtendEndpoint::End => placement.extended.end()[1],
            },
        )
        .map_err(|_| ExtendRefusal::Unrepresentable)?;
        let point_id = match placement.source {
            SketchCurve::Segment(id) => {
                let segment = self
                    .segments
                    .iter()
                    .find(|segment| segment.id == id)
                    .ok_or(ExtendRefusal::UnknownCurve)?;
                match placement.endpoint {
                    ExtendEndpoint::Start => segment.from,
                    ExtendEndpoint::End => segment.to,
                }
            }
            SketchCurve::Arc(id) => {
                let sweep =
                    piece_sweep(&placement.extended).map_err(|_| ExtendRefusal::Unrepresentable)?;
                let arc = self
                    .arcs
                    .iter_mut()
                    .find(|arc| arc.id == id)
                    .ok_or(ExtendRefusal::UnknownCurve)?;
                if !arc.replace_free_sweep(sweep) {
                    return Err(ExtendRefusal::FixedSweep);
                }
                match placement.endpoint {
                    ExtendEndpoint::Start => arc.from,
                    ExtendEndpoint::End => arc.to,
                }
            }
            SketchCurve::Circle(_) | SketchCurve::Ellipse(_) => {
                return Err(ExtendRefusal::ClosedCurve);
            }
            SketchCurve::Bezier(_) | SketchCurve::Conic(_) | SketchCurve::Spline(_) => {
                return Err(ExtendRefusal::Unrepresentable);
            }
        };
        let point = self
            .points
            .iter_mut()
            .find(|point| point.id == point_id)
            .ok_or(ExtendRefusal::UnknownCurve)?;
        point.at = endpoint;
        self.sync_arc_centers();
        Ok(())
    }

    fn non_segment_uses_point(&self, point: EntityId) -> bool {
        self.arcs
            .iter()
            .any(|arc| arc.from == point || arc.to == point || arc.center == point)
            || self.circles.iter().any(|circle| circle.center == point)
            || self
                .beziers
                .iter()
                .any(|bezier| bezier.controls.contains(&point))
            || self.ellipses.iter().any(|ellipse| {
                [ellipse.center, ellipse.major_endpoint, ellipse.width_point].contains(&point)
            })
            || self
                .conics
                .iter()
                .any(|conic| [conic.from, conic.to, conic.vertex].contains(&point))
            || self
                .splines
                .iter()
                .any(|spline| spline.points.contains(&point))
    }

    fn apply_fillet(
        &mut self,
        placement: &FilletPlacement,
        context: parametric::EvaluationContext,
    ) -> Result<(), FilletRefusal> {
        let first_id = match placement.first {
            SketchCurve::Segment(id) => id,
            SketchCurve::Arc(_)
            | SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => {
                return Err(FilletRefusal::UnsupportedCurve);
            }
        };
        let second_id = match placement.second {
            SketchCurve::Segment(id) => id,
            SketchCurve::Arc(_)
            | SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => {
                return Err(FilletRefusal::UnsupportedCurve);
            }
        };
        let first_at = match placement.first_endpoint {
            ExtendEndpoint::Start => placement.shortened_first.start(),
            ExtendEndpoint::End => placement.shortened_first.end(),
        };
        let second_at = match placement.second_endpoint {
            ExtendEndpoint::Start => placement.shortened_second.start(),
            ExtendEndpoint::End => placement.shortened_second.end(),
        };
        let first_at = SketchPoint::try_from_continuous(first_at[0], first_at[1])
            .map_err(|_| FilletRefusal::Unrepresentable)?;
        let second_at = SketchPoint::try_from_continuous(second_at[0], second_at[1])
            .map_err(|_| FilletRefusal::Unrepresentable)?;
        let sweep = piece_sweep(&placement.arc).map_err(|_| FilletRefusal::Unrepresentable)?;
        let role = self
            .segments
            .iter()
            .find(|segment| segment.id == first_id)
            .map(|segment| segment.role)
            .ok_or(FilletRefusal::UnknownCurve)?;

        let corner_index = self
            .point_index(placement.corner)
            .ok_or(FilletRefusal::UnknownCurve)?;
        self.points[corner_index].at = first_at;
        let second_point = self.add_point(second_at);
        if let Some(point) = self
            .points
            .iter_mut()
            .find(|point| point.id == second_point)
        {
            point.role = role;
        }
        let second_segment = self
            .segments
            .iter_mut()
            .find(|segment| segment.id == second_id)
            .ok_or(FilletRefusal::UnknownCurve)?;
        match placement.second_endpoint {
            ExtendEndpoint::Start => second_segment.from = second_point,
            ExtendEndpoint::End => second_segment.to = second_point,
        }
        let arc_id = self
            .connect_arc(placement.corner, second_point, sweep)
            .ok_or(FilletRefusal::Unrepresentable)?;
        if let Some(arc) = self.arcs.iter_mut().find(|arc| arc.id == arc_id) {
            arc.role = role;
        }
        self.sync_arc_centers();
        let arc = SketchCurve::Arc(arc_id);
        for (line, locus) in [(placement.first, first_at), (placement.second, second_at)] {
            let locus = locus.in_plane();
            let branch = self
                .choose_tangent_branch(line, locus, arc, locus, context)
                .map_err(|_| FilletRefusal::Constraint)?;
            self.add_constraint(ConstraintKind::tangent(line, arc, branch), context)
                .map_err(|_| FilletRefusal::Constraint)?;
        }
        Ok(())
    }

    fn apply_chamfer(&mut self, placement: &ChamferPlacement) -> Result<(), ChamferRefusal> {
        let first_id = match placement.first {
            SketchCurve::Segment(id) => id,
            SketchCurve::Arc(_)
            | SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => {
                return Err(ChamferRefusal::Corner(FilletRefusal::UnsupportedCurve));
            }
        };
        let second_id = match placement.second {
            SketchCurve::Segment(id) => id,
            SketchCurve::Arc(_)
            | SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => {
                return Err(ChamferRefusal::Corner(FilletRefusal::UnsupportedCurve));
            }
        };
        let first_endpoint = self
            .segments
            .iter()
            .find(|segment| segment.id == first_id)
            .and_then(|segment| {
                (segment.from == placement.corner)
                    .then_some(ExtendEndpoint::Start)
                    .or_else(|| (segment.to == placement.corner).then_some(ExtendEndpoint::End))
            })
            .ok_or(ChamferRefusal::Corner(FilletRefusal::UnknownCurve))?;
        let first_at = match first_endpoint {
            ExtendEndpoint::Start => placement.shortened_first.start(),
            ExtendEndpoint::End => placement.shortened_first.end(),
        };
        let second_at = match placement.second_endpoint {
            ExtendEndpoint::Start => placement.shortened_second.start(),
            ExtendEndpoint::End => placement.shortened_second.end(),
        };
        let first_at = SketchPoint::try_from_continuous(first_at[0], first_at[1])
            .map_err(|_| ChamferRefusal::Unrepresentable)?;
        let second_at = SketchPoint::try_from_continuous(second_at[0], second_at[1])
            .map_err(|_| ChamferRefusal::Unrepresentable)?;
        let role = self
            .segments
            .iter()
            .find(|segment| segment.id == first_id)
            .map(|segment| segment.role)
            .ok_or(ChamferRefusal::Corner(FilletRefusal::UnknownCurve))?;
        let corner_index = self
            .point_index(placement.corner)
            .ok_or(ChamferRefusal::Corner(FilletRefusal::UnknownCurve))?;
        self.points[corner_index].at = first_at;
        let second_point = self.add_point(second_at);
        if let Some(point) = self
            .points
            .iter_mut()
            .find(|point| point.id == second_point)
        {
            point.role = role;
        }
        let second_segment = self
            .segments
            .iter_mut()
            .find(|segment| segment.id == second_id)
            .ok_or(ChamferRefusal::Corner(FilletRefusal::UnknownCurve))?;
        match placement.second_endpoint {
            ExtendEndpoint::Start => second_segment.from = second_point,
            ExtendEndpoint::End => second_segment.to = second_point,
        }
        let connector = self
            .connect(placement.corner, second_point)
            .ok_or(ChamferRefusal::Unrepresentable)?;
        if let Some(segment) = self
            .segments
            .iter_mut()
            .find(|segment| segment.id == connector)
        {
            segment.role = role;
        }
        Ok(())
    }

    fn apply_offset(&mut self, placement: &OffsetPlacement) -> Result<(), OffsetRefusal> {
        let role = match placement.source {
            SketchCurve::Segment(id) => self
                .segments
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Arc(id) => self
                .arcs
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Circle(id) => self
                .circles
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Bezier(id) => self
                .beziers
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Ellipse(id) => self
                .ellipses
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Conic(id) => self
                .conics
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
            SketchCurve::Spline(id) => self
                .splines
                .iter()
                .find(|curve| curve.id == id)
                .map(|curve| curve.role),
        }
        .ok_or(OffsetRefusal::UnknownCurve)?;
        match placement.offset {
            PlanarCurve::Segment { start, end } => {
                let start = SketchPoint::try_from_continuous(start[0], start[1])
                    .map_err(|_| OffsetRefusal::Unrepresentable)?;
                let end = SketchPoint::try_from_continuous(end[0], end[1])
                    .map_err(|_| OffsetRefusal::Unrepresentable)?;
                let from = self.add_point(start);
                let to = self.add_point(end);
                self.set_point_role(from, role);
                self.set_point_role(to, role);
                let id = self
                    .connect(from, to)
                    .ok_or(OffsetRefusal::Unrepresentable)?;
                self.set_curve_role(SketchCurve::Segment(id), role);
            }
            PlanarCurve::Arc {
                center,
                radius,
                sweep_radians,
                ..
            } if placement.offset.is_closed() => {
                let center = SketchPoint::try_from_continuous(center[0], center[1])
                    .map_err(|_| OffsetRefusal::Unrepresentable)?;
                let id = self
                    .add_circle(center, SketchLength::from_continuous(radius))
                    .ok_or(OffsetRefusal::Unrepresentable)?;
                self.set_curve_role(SketchCurve::Circle(id), role);
                debug_assert!((sweep_radians.abs() - std::f64::consts::TAU).abs() < 1.0e-9);
            }
            PlanarCurve::Arc { .. } => {
                let start = placement.offset.start();
                let end = placement.offset.end();
                let start = SketchPoint::try_from_continuous(start[0], start[1])
                    .map_err(|_| OffsetRefusal::Unrepresentable)?;
                let end = SketchPoint::try_from_continuous(end[0], end[1])
                    .map_err(|_| OffsetRefusal::Unrepresentable)?;
                let from = self.add_point(start);
                let to = self.add_point(end);
                self.set_point_role(from, role);
                self.set_point_role(to, role);
                let sweep =
                    piece_sweep(&placement.offset).map_err(|_| OffsetRefusal::Unrepresentable)?;
                let id = self
                    .connect_arc(from, to, sweep)
                    .ok_or(OffsetRefusal::Unrepresentable)?;
                self.set_curve_role(SketchCurve::Arc(id), role);
                self.sync_arc_centers();
            }
            PlanarCurve::RationalBezier(_) => return Err(OffsetRefusal::Unrepresentable),
        }
        Ok(())
    }

    fn set_curve_role(&mut self, curve: SketchCurve, role: EntityRole) {
        match curve {
            SketchCurve::Segment(id) => {
                if let Some(curve) = self.segments.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Arc(id) => {
                if let Some(curve) = self.arcs.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Circle(id) => {
                if let Some(curve) = self.circles.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Bezier(id) => {
                if let Some(curve) = self.beziers.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Ellipse(id) => {
                if let Some(curve) = self.ellipses.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Conic(id) => {
                if let Some(curve) = self.conics.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
            SketchCurve::Spline(id) => {
                if let Some(curve) = self.splines.iter_mut().find(|curve| curve.id == id) {
                    curve.role = role;
                }
            }
        }
    }

    fn set_point_role(&mut self, point: EntityId, role: EntityRole) {
        if let Some(point) = self.points.iter_mut().find(|held| held.id == point) {
            point.role = role;
        }
    }
}

fn piece_sweep(piece: &PlanarCurve) -> Result<AngleMeasurement, BreakRefusal> {
    let PlanarCurve::Arc { sweep_radians, .. } = *piece else {
        return Err(BreakRefusal::Unrepresentable);
    };
    AngleMeasurement::try_from_degrees_f64(sweep_radians.to_degrees())
        .map_err(|_| BreakRefusal::Unrepresentable)
}

fn distance_to_curve(curve: &PlanarCurve, witness: [f64; 2]) -> f64 {
    let nearest = curve.point_at(curve.nearest_parameter(witness));
    let delta = [nearest[0] - witness[0], nearest[1] - witness[1]];
    delta[0].mul_add(delta[0], delta[1] * delta[1])
}

fn squared_distance(first: [f64; 2], second: [f64; 2]) -> f64 {
    let delta = [first[0] - second[0], first[1] - second[1]];
    delta[0].mul_add(delta[0], delta[1] * delta[1])
}

fn extension_candidate(
    source: &PlanarCurve,
    endpoint: ExtendEndpoint,
    crossing: CurveSupportCrossing,
) -> Option<(f64, PlanarCurve)> {
    match *source {
        PlanarCurve::Segment { .. } => {
            let parameter = crossing.parameter_on_support;
            let travel = match endpoint {
                ExtendEndpoint::Start if parameter < -EXTEND_EPSILON => -parameter,
                ExtendEndpoint::End if parameter > 1.0 + EXTEND_EPSILON => parameter - 1.0,
                ExtendEndpoint::Start | ExtendEndpoint::End => return None,
            } * source.length();
            let extended = match endpoint {
                ExtendEndpoint::Start => source.sub_curve(parameter, 1.0),
                ExtendEndpoint::End => source.sub_curve(0.0, parameter),
            };
            Some((travel, extended))
        }
        PlanarCurve::Arc {
            center,
            start_radians,
            sweep_radians,
            radius,
        } => {
            let magnitude = sweep_radians.abs();
            if magnitude <= EXTEND_EPSILON {
                return None;
            }
            let bearing = (crossing.point[1] - center[1]).atan2(crossing.point[0] - center[0]);
            let mut travelled = if sweep_radians.is_sign_negative() {
                (start_radians - bearing).rem_euclid(std::f64::consts::TAU)
            } else {
                (bearing - start_radians).rem_euclid(std::f64::consts::TAU)
            };
            if travelled >= std::f64::consts::TAU - EXTEND_EPSILON {
                travelled = 0.0;
            }
            if travelled <= magnitude + EXTEND_EPSILON {
                return None;
            }
            let turn_parameter = std::f64::consts::TAU / magnitude;
            let point_parameter = travelled / magnitude;
            let (travel, extended) = match endpoint {
                ExtendEndpoint::Start => (
                    (std::f64::consts::TAU - travelled) * radius,
                    source.sub_curve(point_parameter - turn_parameter, 1.0),
                ),
                ExtendEndpoint::End => (
                    (travelled - magnitude) * radius,
                    source.sub_curve(0.0, point_parameter),
                ),
            };
            Some((travel, extended))
        }
        PlanarCurve::RationalBezier(_) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn line_fillet_geometry(
    first: SketchCurve,
    second: SketchCurve,
    first_curve: PlanarCurve,
    second_curve: PlanarCurve,
    corner: EntityId,
    first_endpoint: ExtendEndpoint,
    second_endpoint: ExtendEndpoint,
    witness: [f64; 2],
) -> Result<FilletPlacement, FilletRefusal> {
    let corner_at = match first_endpoint {
        ExtendEndpoint::Start => first_curve.start(),
        ExtendEndpoint::End => first_curve.end(),
    };
    let first_far = match first_endpoint {
        ExtendEndpoint::Start => first_curve.end(),
        ExtendEndpoint::End => first_curve.start(),
    };
    let second_far = match second_endpoint {
        ExtendEndpoint::Start => second_curve.end(),
        ExtendEndpoint::End => second_curve.start(),
    };
    let first_span = [first_far[0] - corner_at[0], first_far[1] - corner_at[1]];
    let second_span = [second_far[0] - corner_at[0], second_far[1] - corner_at[1]];
    let (first_length, second_length) = (
        first_span[0].hypot(first_span[1]),
        second_span[0].hypot(second_span[1]),
    );
    if first_length <= EXTEND_EPSILON || second_length <= EXTEND_EPSILON {
        return Err(FilletRefusal::RadiusOutOfRange);
    }
    let first_unit = [first_span[0] / first_length, first_span[1] / first_length];
    let second_unit = [
        second_span[0] / second_length,
        second_span[1] / second_length,
    ];
    let (first_tangent, second_tangent, arc) = fillet_rounding(
        corner_at,
        first_unit,
        second_unit,
        first_length,
        second_length,
        witness,
    )?;
    let shortened_first = match first_endpoint {
        ExtendEndpoint::Start => PlanarCurve::Segment {
            start: first_tangent,
            end: first_far,
        },
        ExtendEndpoint::End => PlanarCurve::Segment {
            start: first_far,
            end: first_tangent,
        },
    };
    let shortened_second = match second_endpoint {
        ExtendEndpoint::Start => PlanarCurve::Segment {
            start: second_tangent,
            end: second_far,
        },
        ExtendEndpoint::End => PlanarCurve::Segment {
            start: second_far,
            end: second_tangent,
        },
    };
    Ok(FilletPlacement {
        first,
        second,
        shortened_first,
        shortened_second,
        arc,
        corner,
        first_endpoint,
        second_endpoint,
    })
}

fn fillet_rounding(
    corner: [f64; 2],
    first_unit: [f64; 2],
    second_unit: [f64; 2],
    first_length: f64,
    second_length: f64,
    witness: [f64; 2],
) -> Result<([f64; 2], [f64; 2], PlanarCurve), FilletRefusal> {
    let cosine = first_unit[0]
        .mul_add(second_unit[0], first_unit[1] * second_unit[1])
        .clamp(-1.0, 1.0);
    let half_angle = cosine.acos() * 0.5;
    if half_angle.sin() <= EXTEND_EPSILON || half_angle.cos() <= EXTEND_EPSILON {
        return Err(FilletRefusal::RadiusOutOfRange);
    }
    let from_corner = [witness[0] - corner[0], witness[1] - corner[1]];
    let tangent_distance = first_unit[0].mul_add(from_corner[0], first_unit[1] * from_corner[1]);
    if tangent_distance <= EXTEND_EPSILON
        || tangent_distance >= first_length - EXTEND_EPSILON
        || tangent_distance >= second_length - EXTEND_EPSILON
    {
        return Err(FilletRefusal::RadiusOutOfRange);
    }
    let first_tangent = [
        first_unit[0].mul_add(tangent_distance, corner[0]),
        first_unit[1].mul_add(tangent_distance, corner[1]),
    ];
    let second_tangent = [
        second_unit[0].mul_add(tangent_distance, corner[0]),
        second_unit[1].mul_add(tangent_distance, corner[1]),
    ];
    let bisector = [
        first_unit[0] + second_unit[0],
        first_unit[1] + second_unit[1],
    ];
    let bisector_length = bisector[0].hypot(bisector[1]);
    if bisector_length <= EXTEND_EPSILON {
        return Err(FilletRefusal::RadiusOutOfRange);
    }
    let center_distance = tangent_distance / half_angle.cos();
    let center = [
        (bisector[0] / bisector_length).mul_add(center_distance, corner[0]),
        (bisector[1] / bisector_length).mul_add(center_distance, corner[1]),
    ];
    let first_radius = [first_tangent[0] - center[0], first_tangent[1] - center[1]];
    let second_radius = [second_tangent[0] - center[0], second_tangent[1] - center[1]];
    let sweep_radians = (first_radius[0]
        .mul_add(second_radius[1], -(first_radius[1] * second_radius[0])))
    .atan2(first_radius[0].mul_add(second_radius[0], first_radius[1] * second_radius[1]));
    if sweep_radians.abs() <= EXTEND_EPSILON {
        return Err(FilletRefusal::RadiusOutOfRange);
    }
    Ok((
        first_tangent,
        second_tangent,
        PlanarCurve::Arc {
            center,
            radius: tangent_distance * half_angle.tan(),
            start_radians: first_radius[1].atan2(first_radius[0]),
            sweep_radians,
        },
    ))
}
