//! Atomic transforms over an explicit sketch-entity selection.
//!
//! The shell chooses the gesture and supplies typed stable identities; this module owns closure
//! over curve endpoints, exact point writes, copy identity, and the refusal boundary around
//! standing constraints. Transforming a constrained selection without a multi-point pin solve
//! would either violate an assertion or silently delete it, so Move and Scale refuse that case.

use super::{
    boxed_push, Arc, Bezier, Circle, CircleRadius, Conic, Ellipse, EntityId, Point, ResolvedLength,
    Segment, Sketch, SketchCurve, SketchPoint, SketchSolid, Spline, ABSENT_DERIVED_POINT,
};
use std::collections::{HashMap, HashSet};
use substrate::curve_intersection::PlanarCurve;

/// One typed member of a sketch transform selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchTransformEntity {
    Point(EntityId),
    Curve(SketchCurve),
}

/// Why an atomic selection transform cannot be represented safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchTransformRefusal {
    EmptySelection,
    UnknownEntity,
    ConstrainedSelection,
    FixedRadius,
    InvalidScale,
    Unrepresentable,
}

#[derive(Default)]
struct TransformClosure {
    points: HashSet<EntityId>,
    segments: HashSet<EntityId>,
    arcs: HashSet<EntityId>,
    circles: HashSet<EntityId>,
    beziers: HashSet<EntityId>,
    ellipses: HashSet<EntityId>,
    conics: HashSet<EntityId>,
    splines: HashSet<EntityId>,
}

impl SketchSolid {
    /// Largest selected authored-point radius from `center`, widened by selected whole circles.
    /// The shell uses it as Scale's spatial unit so a second click can name a dimensionless factor.
    pub fn selection_scale_radius(
        &self,
        entities: &[SketchTransformEntity],
        center: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<f64, SketchTransformRefusal> {
        let closure = self.sketch.transform_closure(entities)?;
        if self.sketch.closure_is_constrained(&closure) {
            return Err(SketchTransformRefusal::ConstrainedSelection);
        }
        if self.sketch.circles.iter().any(|circle| {
            closure.circles.contains(&circle.id) && circle.radius.free_value().is_none()
        }) {
            return Err(SketchTransformRefusal::FixedRadius);
        }
        let mut radius = 0.0_f64;
        for point in self
            .sketch
            .points
            .iter()
            .filter(|point| closure.points.contains(&point.id))
        {
            let at = point.at.in_plane();
            radius = radius.max((at[0] - center[0]).hypot(at[1] - center[1]));
        }
        for circle in self
            .sketch
            .circles
            .iter()
            .filter(|circle| closure.circles.contains(&circle.id))
        {
            let at = self
                .sketch
                .points
                .iter()
                .find(|point| point.id == circle.center)
                .map(|point| point.at.in_plane())
                .ok_or(SketchTransformRefusal::UnknownEntity)?;
            radius = radius.max(
                (at[0] - center[0]).hypot(at[1] - center[1]) + circle.resolved_radius(context),
            );
        }
        (radius > f64::EPSILON)
            .then_some(radius)
            .ok_or(SketchTransformRefusal::InvalidScale)
    }

    /// Native selected curves after a translation, for shell preview. No ids are minted.
    pub fn translated_curve_preview(
        &self,
        entities: &[SketchTransformEntity],
        delta: [f64; 2],
        copy: bool,
        context: parametric::EvaluationContext,
    ) -> Result<Vec<PlanarCurve>, SketchTransformRefusal> {
        let closure = self.sketch.transform_closure(entities)?;
        if !copy && self.sketch.closure_is_constrained(&closure) {
            return Err(SketchTransformRefusal::ConstrainedSelection);
        }
        self.transformed_curve_preview(entities, context, |point| {
            [point[0] + delta[0], point[1] + delta[1]]
        })
    }

    /// Native selected curves after a uniform scale, for shell preview. No ids are minted.
    pub fn scaled_curve_preview(
        &self,
        entities: &[SketchTransformEntity],
        center: [f64; 2],
        factor: f64,
        context: parametric::EvaluationContext,
    ) -> Result<Vec<PlanarCurve>, SketchTransformRefusal> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(SketchTransformRefusal::InvalidScale);
        }
        let closure = self.sketch.transform_closure(entities)?;
        if self.sketch.closure_is_constrained(&closure) {
            return Err(SketchTransformRefusal::ConstrainedSelection);
        }
        if self.sketch.circles.iter().any(|circle| {
            closure.circles.contains(&circle.id) && circle.radius.free_value().is_none()
        }) {
            return Err(SketchTransformRefusal::FixedRadius);
        }
        self.transformed_curve_preview(entities, context, |point| {
            [
                (point[0] - center[0]).mul_add(factor, center[0]),
                (point[1] - center[1]).mul_add(factor, center[1]),
            ]
        })
        .map(|curves| {
            curves
                .into_iter()
                .map(|curve| match curve {
                    PlanarCurve::Segment { .. } | PlanarCurve::RationalBezier(_) => curve,
                    PlanarCurve::Arc {
                        center: curve_center,
                        radius,
                        start_radians,
                        sweep_radians,
                    } => PlanarCurve::Arc {
                        center: curve_center,
                        radius: radius * factor,
                        start_radians,
                        sweep_radians,
                    },
                })
                .collect()
        })
    }

    /// Translate selected geometry by `delta`. With `copy`, append fresh unconstrained geometry;
    /// otherwise preserve every selected entity id and refuse any standing constraint it touches.
    pub fn with_entities_translated(
        &self,
        entities: &[SketchTransformEntity],
        delta: [f64; 2],
        copy: bool,
    ) -> Result<SketchSolid, SketchTransformRefusal> {
        if !delta.into_iter().all(f64::is_finite) {
            return Err(SketchTransformRefusal::Unrepresentable);
        }
        let closure = self.sketch.transform_closure(entities)?;
        if !copy && self.sketch.closure_is_constrained(&closure) {
            return Err(SketchTransformRefusal::ConstrainedSelection);
        }
        let mut next = self.clone();
        if copy {
            next.sketch
                .copy_closure(&closure, |point| [point[0] + delta[0], point[1] + delta[1]])?;
        } else {
            next.sketch.transform_points(&closure.points, |point| {
                [point[0] + delta[0], point[1] + delta[1]]
            })?;
        }
        next.sketch.sync_derived_points();
        Ok(next)
    }

    /// Scale selected free geometry about `center`, preserving identities. A fixed circle radius
    /// and any standing constraint own quantities the scale would change, so both refuse.
    pub fn with_entities_scaled(
        &self,
        entities: &[SketchTransformEntity],
        center: [f64; 2],
        factor: f64,
    ) -> Result<SketchSolid, SketchTransformRefusal> {
        if !center.into_iter().all(f64::is_finite) || !factor.is_finite() || factor <= 0.0 {
            return Err(SketchTransformRefusal::InvalidScale);
        }
        let closure = self.sketch.transform_closure(entities)?;
        if self.sketch.closure_is_constrained(&closure) {
            return Err(SketchTransformRefusal::ConstrainedSelection);
        }
        let mut next = self.clone();
        next.sketch.transform_points(&closure.points, |point| {
            [
                (point[0] - center[0]).mul_add(factor, center[0]),
                (point[1] - center[1]).mul_add(factor, center[1]),
            ]
        })?;
        for circle in &mut next.sketch.circles {
            if !closure.circles.contains(&circle.id) {
                continue;
            }
            let Some(radius) = circle.radius.free_value().copied() else {
                return Err(SketchTransformRefusal::FixedRadius);
            };
            let scaled = ResolvedLength::try_from_f64(radius.value() * factor)
                .map_err(|_| SketchTransformRefusal::Unrepresentable)?;
            circle.radius = CircleRadius::free(scaled);
        }
        next.sketch.sync_derived_points();
        Ok(next)
    }

    fn transformed_curve_preview(
        &self,
        entities: &[SketchTransformEntity],
        context: parametric::EvaluationContext,
        transform: impl Fn([f64; 2]) -> [f64; 2],
    ) -> Result<Vec<PlanarCurve>, SketchTransformRefusal> {
        let closure = self.sketch.transform_closure(entities)?;
        let curves = closure
            .segments
            .iter()
            .copied()
            .map(SketchCurve::Segment)
            .chain(closure.arcs.iter().copied().map(SketchCurve::Arc))
            .chain(closure.circles.iter().copied().map(SketchCurve::Circle));
        let curves = curves
            .chain(closure.beziers.iter().copied().map(SketchCurve::Bezier))
            .chain(closure.ellipses.iter().copied().map(SketchCurve::Ellipse))
            .chain(closure.conics.iter().copied().map(SketchCurve::Conic))
            .chain(closure.splines.iter().copied().map(SketchCurve::Spline));
        let mut transformed = Vec::new();
        for source in curves {
            let (pieces, _) = self
                .sketch
                .source_planar_curves(source, context)
                .ok_or(SketchTransformRefusal::UnknownEntity)?;
            transformed.extend(pieces.into_iter().map(|curve| match curve {
                PlanarCurve::Segment { start, end } => PlanarCurve::Segment {
                    start: transform(start),
                    end: transform(end),
                },
                PlanarCurve::Arc {
                    center,
                    radius,
                    start_radians,
                    sweep_radians,
                } => PlanarCurve::Arc {
                    center: transform(center),
                    radius,
                    start_radians,
                    sweep_radians,
                },
                PlanarCurve::RationalBezier(curve) => {
                    PlanarCurve::RationalBezier(substrate::rational_bezier::RationalBezier {
                        control: curve.control.map(&transform),
                        weights: curve.weights,
                    })
                }
            }));
        }
        Ok(transformed)
    }
}

impl Sketch {
    fn transform_closure(
        &self,
        entities: &[SketchTransformEntity],
    ) -> Result<TransformClosure, SketchTransformRefusal> {
        if entities.is_empty() {
            return Err(SketchTransformRefusal::EmptySelection);
        }
        let mut closure = TransformClosure::default();
        for entity in entities {
            match *entity {
                SketchTransformEntity::Point(id) => {
                    if !self.points.iter().any(|point| point.id == id) {
                        return Err(SketchTransformRefusal::UnknownEntity);
                    }
                    closure.points.insert(id);
                }
                SketchTransformEntity::Curve(SketchCurve::Segment(id)) => {
                    let segment = self
                        .segments
                        .iter()
                        .find(|segment| segment.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.segments.insert(id);
                    closure.points.extend([segment.from, segment.to]);
                }
                SketchTransformEntity::Curve(SketchCurve::Arc(id)) => {
                    let arc = self
                        .arcs
                        .iter()
                        .find(|arc| arc.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.arcs.insert(id);
                    closure.points.extend([arc.from, arc.to]);
                }
                SketchTransformEntity::Curve(SketchCurve::Circle(id)) => {
                    let circle = self
                        .circles
                        .iter()
                        .find(|circle| circle.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.circles.insert(id);
                    closure.points.insert(circle.center);
                }
                SketchTransformEntity::Curve(SketchCurve::Bezier(id)) => {
                    let bezier = self
                        .beziers
                        .iter()
                        .find(|bezier| bezier.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.beziers.insert(id);
                    closure.points.extend(bezier.controls);
                }
                SketchTransformEntity::Curve(SketchCurve::Ellipse(id)) => {
                    let ellipse = self
                        .ellipses
                        .iter()
                        .find(|ellipse| ellipse.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.ellipses.insert(id);
                    closure.points.extend([
                        ellipse.center,
                        ellipse.major_endpoint,
                        ellipse.width_point,
                    ]);
                }
                SketchTransformEntity::Curve(SketchCurve::Conic(id)) => {
                    let conic = self
                        .conics
                        .iter()
                        .find(|conic| conic.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.conics.insert(id);
                    closure.points.extend([conic.from, conic.to, conic.control]);
                }
                SketchTransformEntity::Curve(SketchCurve::Spline(id)) => {
                    let spline = self
                        .splines
                        .iter()
                        .find(|spline| spline.id == id)
                        .ok_or(SketchTransformRefusal::UnknownEntity)?;
                    closure.splines.insert(id);
                    closure.points.extend(spline.points.iter().copied());
                }
            }
        }
        Ok(closure)
    }

    fn closure_is_constrained(&self, closure: &TransformClosure) -> bool {
        self.constraints.iter().any(|constraint| {
            constraint
                .kind
                .points()
                .into_iter()
                .any(|id| closure.points.contains(&id))
                || constraint
                    .kind
                    .segments()
                    .into_iter()
                    .any(|id| closure.segments.contains(&id))
                || constraint.kind.curves().into_iter().any(|curve| {
                    closure.segments.contains(&curve.id())
                        || closure.arcs.contains(&curve.id())
                        || closure.circles.contains(&curve.id())
                        || closure.beziers.contains(&curve.id())
                        || closure.ellipses.contains(&curve.id())
                        || closure.conics.contains(&curve.id())
                        || closure.splines.contains(&curve.id())
                })
        })
    }

    fn transform_points(
        &mut self,
        points: &HashSet<EntityId>,
        transform: impl Fn([f64; 2]) -> [f64; 2],
    ) -> Result<(), SketchTransformRefusal> {
        for point in self
            .points
            .iter_mut()
            .filter(|point| points.contains(&point.id))
        {
            let moved = transform(point.at.in_plane());
            point.at = SketchPoint::try_from_continuous(moved[0], moved[1])
                .map_err(|_| SketchTransformRefusal::Unrepresentable)?;
        }
        Ok(())
    }

    fn copy_closure(
        &mut self,
        closure: &TransformClosure,
        transform: impl Fn([f64; 2]) -> [f64; 2],
    ) -> Result<(), SketchTransformRefusal> {
        let source_points: Vec<Point> = self
            .points
            .iter()
            .filter(|point| closure.points.contains(&point.id))
            .copied()
            .collect();
        let mut points = HashMap::new();
        for source in source_points {
            let moved = transform(source.at.in_plane());
            let at = SketchPoint::try_from_continuous(moved[0], moved[1])
                .map_err(|_| SketchTransformRefusal::Unrepresentable)?;
            let id = self.alloc_id();
            self.points.push(Point {
                id,
                at,
                lifetime: source.lifetime,
            });
            points.insert(source.id, id);
        }
        let source_segments: Vec<Segment> = self
            .segments
            .iter()
            .filter(|curve| closure.segments.contains(&curve.id))
            .copied()
            .collect();
        for source in source_segments {
            let id = self.alloc_id();
            self.segments.push(Segment {
                id,
                from: mapped(&points, source.from)?,
                to: mapped(&points, source.to)?,
                origin: id,
                role: source.role,
            });
        }
        let source_arcs: Vec<Arc> = self
            .arcs
            .iter()
            .filter(|curve| closure.arcs.contains(&curve.id))
            .copied()
            .collect();
        for source in source_arcs {
            let id = self.alloc_id();
            self.arcs.push(Arc {
                id,
                from: mapped(&points, source.from)?,
                to: mapped(&points, source.to)?,
                bulge: source.bulge,
                center: ABSENT_DERIVED_POINT,
                origin: id,
                role: source.role,
            });
        }
        let source_circles: Vec<Circle> = self
            .circles
            .iter()
            .filter(|curve| closure.circles.contains(&curve.id))
            .copied()
            .collect();
        for source in source_circles {
            let id = self.alloc_id();
            self.circles.push(Circle {
                id,
                center: mapped(&points, source.center)?,
                radius: source.radius,
                origin: id,
                role: source.role,
            });
        }
        let source_beziers: Vec<Bezier> = self
            .beziers
            .iter()
            .filter(|curve| closure.beziers.contains(&curve.id))
            .copied()
            .collect();
        for source in source_beziers {
            let id = self.alloc_id();
            boxed_push(
                &mut self.beziers,
                Bezier {
                    id,
                    controls: [
                        mapped(&points, source.controls[0])?,
                        mapped(&points, source.controls[1])?,
                        mapped(&points, source.controls[2])?,
                        mapped(&points, source.controls[3])?,
                    ],
                    weights: source.weights,
                    origin: id,
                    role: source.role,
                },
            );
        }
        self.copy_higher_curves(closure, &points)?;
        Ok(())
    }

    fn copy_higher_curves(
        &mut self,
        closure: &TransformClosure,
        points: &HashMap<EntityId, EntityId>,
    ) -> Result<(), SketchTransformRefusal> {
        let source_ellipses: Vec<Ellipse> = self
            .ellipses
            .iter()
            .filter(|curve| closure.ellipses.contains(&curve.id))
            .copied()
            .collect();
        for source in source_ellipses {
            let id = self.alloc_id();
            boxed_push(
                &mut self.ellipses,
                Ellipse {
                    id,
                    center: mapped(points, source.center)?,
                    major_endpoint: mapped(points, source.major_endpoint)?,
                    width_point: mapped(points, source.width_point)?,
                    origin: id,
                    role: source.role,
                },
            );
        }
        let source_conics: Vec<Conic> = self
            .conics
            .iter()
            .filter(|curve| closure.conics.contains(&curve.id))
            .copied()
            .collect();
        for source in source_conics {
            let id = self.alloc_id();
            boxed_push(
                &mut self.conics,
                Conic {
                    id,
                    from: mapped(points, source.from)?,
                    to: mapped(points, source.to)?,
                    control: mapped(points, source.control)?,
                    shoulder: ABSENT_DERIVED_POINT,
                    rho: source.rho,
                    origin: id,
                    role: source.role,
                },
            );
        }
        let source_splines: Vec<Spline> = self
            .splines
            .iter()
            .filter(|curve| closure.splines.contains(&curve.id))
            .cloned()
            .collect();
        for source in source_splines {
            let id = self.alloc_id();
            boxed_push(
                &mut self.splines,
                Spline {
                    id,
                    points: source
                        .points
                        .iter()
                        .map(|point| mapped(points, *point))
                        .collect::<Result<Vec<_>, _>>()?,
                    kind: source.kind,
                    closed: source.closed,
                    origin: id,
                    role: source.role,
                },
            );
        }
        Ok(())
    }
}

fn mapped(
    points: &HashMap<EntityId, EntityId>,
    source: EntityId,
) -> Result<EntityId, SketchTransformRefusal> {
    points
        .get(&source)
        .copied()
        .ok_or(SketchTransformRefusal::UnknownEntity)
}
