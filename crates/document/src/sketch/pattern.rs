//! Associative sketch mirrors and patterns.
//!
//! A pattern is a compact authored RULE, not a pile of copied entities. Its sources remain the
//! only solver-owned geometry; every instance is regenerated from their current resolved curves.
//! This is what makes a source edit propagate, gives the operator zero degrees of freedom, and
//! leaves one durable place for count, spacing, angle, and axis parameters.

use std::collections::BTreeSet;
use std::num::NonZeroU32;

use parametric::units::AngleMeasurement;
use substrate::curve_intersection::PlanarCurve;

use super::{EntityId, EntityRole, Sketch, SketchCurve, SketchLength, SketchSolid};

pub(super) fn pattern_store_is_empty(patterns: &[SketchPattern]) -> bool {
    patterns.is_empty()
}

/// A durable in-plane vector. Each component retains its own authored length expression, so a
/// density retarget treats pattern spacing exactly like a dimension or circle radius.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SketchVector {
    pub axis0: SketchLength,
    pub axis1: SketchLength,
}

impl SketchVector {
    pub fn from_continuous(axis0: f64, axis1: f64) -> Self {
        Self {
            axis0: SketchLength::from_continuous(axis0),
            axis1: SketchLength::from_continuous(axis1),
        }
    }

    pub fn value(self) -> [f64; 2] {
        [self.axis0.value(), self.axis1.value()]
    }

    fn retargeted(self, old_density: u32, new_density: u32) -> Self {
        Self {
            axis0: self.axis0.retargeted(old_density, new_density),
            axis1: self.axis1.retargeted(old_density, new_density),
        }
    }
}

/// The parameter block owned by one associative operator.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SketchPatternKind {
    /// One reflected instance across an authored segment used as an infinite axis.
    Mirror { axis: EntityId },
    /// A one- or two-direction grid. Counts include the source at `[0, 0]`.
    Rectangular {
        counts: [u32; 2],
        steps: Box<[SketchVector; 2]>,
    },
    /// Copies around an authored center point. Count includes the source, and `angle` is the
    /// total distribution angle. A full turn spaces `count` items without duplicating the source.
    Circular {
        center: EntityId,
        count: u32,
        angle: AngleMeasurement,
    },
}

/// One persisted generator. `id` identifies the rule; generated instances use composite,
/// ephemeral provenance and never consume entries from the sketch entity-id space.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SketchPattern {
    pub id: EntityId,
    pub sources: Vec<SketchCurve>,
    pub kind: SketchPatternKind,
}

/// Provenance and geometry for one regenerated curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedPatternCurve {
    pub pattern: EntityId,
    pub instance: [u32; 2],
    pub source: SketchCurve,
    pub role: EntityRole,
    pub geometry: PlanarCurve,
}

/// Why an associative operator could not be added atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchPatternRefusal {
    EmptySelection,
    UnknownSource,
    InvalidAxis,
    InvalidCenter,
    InvalidCount,
    DegenerateStep,
}

impl Sketch {
    /// Persist one mirror rule after validating all references. The axis may be construction
    /// geometry, but cannot also be one of the mirrored sources.
    pub fn add_mirror_pattern(
        &mut self,
        sources: impl IntoIterator<Item = SketchCurve>,
        axis: EntityId,
    ) -> Result<EntityId, SketchPatternRefusal> {
        let sources = self.checked_pattern_sources(sources)?;
        if sources.iter().any(|source| source.id() == axis)
            || !matches!(self.curve_geometry(SketchCurve::Segment(axis), default_context()), Some(parametric::sketch::CurveGeometry::Segment { from, to }) if from != to)
        {
            return Err(SketchPatternRefusal::InvalidAxis);
        }
        Ok(self.push_pattern(sources, SketchPatternKind::Mirror { axis }))
    }

    /// Persist a rectangular array. `[1, 1]` is refused because it would generate nothing;
    /// either direction whose count exceeds one must have a non-zero step.
    pub fn add_rectangular_pattern(
        &mut self,
        sources: impl IntoIterator<Item = SketchCurve>,
        counts: [u32; 2],
        steps: [SketchVector; 2],
    ) -> Result<EntityId, SketchPatternRefusal> {
        let sources = self.checked_pattern_sources(sources)?;
        if counts[0] == 0 || counts[1] == 0 || counts == [1, 1] {
            return Err(SketchPatternRefusal::InvalidCount);
        }
        for (count, step) in counts.into_iter().zip(steps) {
            let [x, y] = step.value();
            if count > 1 && (!x.is_finite() || !y.is_finite() || x.hypot(y) <= f64::EPSILON) {
                return Err(SketchPatternRefusal::DegenerateStep);
            }
        }
        Ok(self.push_pattern(
            sources,
            SketchPatternKind::Rectangular {
                counts,
                steps: Box::new(steps),
            },
        ))
    }

    /// Persist a circular array around an authored point. The total angle must be finite and
    /// non-zero; count includes the source and therefore begins at two.
    pub fn add_circular_pattern(
        &mut self,
        sources: impl IntoIterator<Item = SketchCurve>,
        center: EntityId,
        count: u32,
        angle: AngleMeasurement,
    ) -> Result<EntityId, SketchPatternRefusal> {
        let sources = self.checked_pattern_sources(sources)?;
        if !self.points.iter().any(|point| point.id == center) {
            return Err(SketchPatternRefusal::InvalidCenter);
        }
        if count < 2 || !angle.to_degrees_f64().is_finite() || angle.to_degrees_f64() == 0.0 {
            return Err(SketchPatternRefusal::InvalidCount);
        }
        Ok(self.push_pattern(
            sources,
            SketchPatternKind::Circular {
                center,
                count,
                angle,
            },
        ))
    }

    /// Delete a generator without touching its authored sources.
    pub fn delete_pattern(&mut self, id: EntityId) -> bool {
        let before = self.patterns.len();
        self.patterns = self
            .patterns
            .iter()
            .filter(|pattern| pattern.id != id)
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.patterns.len() != before
    }

    /// Regenerate all pattern instances at the current authored coordinates.
    pub fn derived_pattern_curves(
        &self,
        context: parametric::EvaluationContext,
    ) -> Vec<DerivedPatternCurve> {
        let mut derived = Vec::new();
        for pattern in &self.patterns {
            for &source in &pattern.sources {
                let Some((geometries, role)) = self.source_planar_curves(source, context) else {
                    continue;
                };
                for geometry in geometries {
                    match &pattern.kind {
                        SketchPatternKind::Mirror { axis } => {
                            let Some((axis_from, axis_to)) = self.segment_span(*axis) else {
                                continue;
                            };
                            derived.push(DerivedPatternCurve {
                                pattern: pattern.id,
                                instance: [1, 0],
                                source,
                                role,
                                geometry: map_curve(
                                    geometry,
                                    |point| reflect(point, axis_from, axis_to),
                                    true,
                                ),
                            });
                        }
                        SketchPatternKind::Rectangular { counts, steps } => {
                            let [first, second] = [steps[0].value(), steps[1].value()];
                            for i in 0..counts[0] {
                                for j in 0..counts[1] {
                                    if [i, j] == [0, 0] {
                                        continue;
                                    }
                                    let offset = [
                                        f64::from(i) * first[0] + f64::from(j) * second[0],
                                        f64::from(i) * first[1] + f64::from(j) * second[1],
                                    ];
                                    derived.push(DerivedPatternCurve {
                                        pattern: pattern.id,
                                        instance: [i, j],
                                        source,
                                        role,
                                        geometry: map_curve(
                                            geometry,
                                            |point| add(point, offset),
                                            false,
                                        ),
                                    });
                                }
                            }
                        }
                        SketchPatternKind::Circular {
                            center,
                            count,
                            angle,
                        } => {
                            let Some(center) = self.point_position(*center) else {
                                continue;
                            };
                            let total = angle.to_degrees_f64().to_radians();
                            let full_turn = (total.abs() - std::f64::consts::TAU).abs() <= 1.0e-10;
                            let divisor = if full_turn { *count } else { *count - 1 };
                            for i in 1..*count {
                                let radians = total * f64::from(i) / f64::from(divisor);
                                derived.push(DerivedPatternCurve {
                                    pattern: pattern.id,
                                    instance: [i, 0],
                                    source,
                                    role,
                                    geometry: map_curve(
                                        geometry,
                                        |point| rotate_about(point, center, radians),
                                        false,
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
        derived
    }

    pub(super) fn retarget_patterns(&mut self, old_density: u32, new_density: u32) {
        for pattern in &mut self.patterns {
            if let SketchPatternKind::Rectangular { steps, .. } = &mut pattern.kind {
                **steps = [
                    steps[0].retargeted(old_density, new_density),
                    steps[1].retargeted(old_density, new_density),
                ];
            }
        }
    }

    pub(super) fn drop_dangling_patterns(&mut self) {
        let curves: BTreeSet<EntityId> = self
            .segments
            .iter()
            .map(|curve| curve.id)
            .chain(self.arcs.iter().map(|curve| curve.id))
            .chain(self.circles.iter().map(|curve| curve.id))
            .chain(self.beziers.iter().map(|curve| curve.id))
            .chain(self.ellipses.iter().map(|curve| curve.id))
            .chain(self.conics.iter().map(|curve| curve.id))
            .chain(self.splines.iter().map(|curve| curve.id))
            .collect();
        let segments: BTreeSet<EntityId> = self.segments.iter().map(|curve| curve.id).collect();
        let points: BTreeSet<EntityId> = self.points.iter().map(|point| point.id).collect();
        self.patterns = self
            .patterns
            .iter()
            .filter(|pattern| {
                !pattern.sources.is_empty()
                    && pattern
                        .sources
                        .iter()
                        .all(|source| curves.contains(&source.id()))
                    && match &pattern.kind {
                        SketchPatternKind::Mirror { axis } => segments.contains(axis),
                        SketchPatternKind::Circular { center, .. } => points.contains(center),
                        SketchPatternKind::Rectangular { .. } => true,
                    }
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
    }

    fn checked_pattern_sources(
        &self,
        sources: impl IntoIterator<Item = SketchCurve>,
    ) -> Result<Vec<SketchCurve>, SketchPatternRefusal> {
        let mut seen = BTreeSet::new();
        let sources: Vec<_> = sources
            .into_iter()
            .filter(|source| seen.insert(source.id()))
            .collect();
        if sources.is_empty() {
            return Err(SketchPatternRefusal::EmptySelection);
        }
        if sources.iter().any(|&source| {
            self.source_planar_curves(source, default_context())
                .is_none()
        }) {
            return Err(SketchPatternRefusal::UnknownSource);
        }
        Ok(sources)
    }

    fn push_pattern(&mut self, sources: Vec<SketchCurve>, kind: SketchPatternKind) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        let mut patterns = self.patterns.to_vec();
        patterns.push(SketchPattern { id, sources, kind });
        self.patterns = patterns.into_boxed_slice();
        id
    }

    fn point_position(&self, id: EntityId) -> Option<[f64; 2]> {
        self.points
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
    }

    fn segment_span(&self, id: EntityId) -> Option<([f64; 2], [f64; 2])> {
        let segment = self.segments.iter().find(|segment| segment.id == id)?;
        Some((
            self.point_position(segment.from)?,
            self.point_position(segment.to)?,
        ))
    }

    pub(crate) fn source_planar_curves(
        &self,
        source: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Option<(Vec<PlanarCurve>, EntityRole)> {
        match source {
            SketchCurve::Segment(id) => {
                let segment = self.segments.iter().find(|segment| segment.id == id)?;
                Some((
                    vec![PlanarCurve::Segment {
                        start: self.point_position(segment.from)?,
                        end: self.point_position(segment.to)?,
                    }],
                    segment.role,
                ))
            }
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                let form = self.arc_form(arc)?;
                Some((
                    vec![PlanarCurve::Arc {
                        center: form.center,
                        radius: form.radius,
                        start_radians: (form.from[1] - form.center[1])
                            .atan2(form.from[0] - form.center[0]),
                        sweep_radians: form.sweep_degrees.to_radians(),
                    }],
                    arc.role,
                ))
            }
            SketchCurve::Circle(id) => {
                let circle = self.circles.iter().find(|circle| circle.id == id)?;
                Some((
                    vec![PlanarCurve::circle(
                        self.point_position(circle.center)?,
                        circle.resolved_radius(context),
                    )],
                    circle.role,
                ))
            }
            SketchCurve::Bezier(id) => {
                let bezier = self.beziers.iter().find(|bezier| bezier.id == id)?;
                Some((
                    vec![PlanarCurve::RationalBezier(
                        self.rational_bezier_from(bezier.controls, bezier.weights)?,
                    )],
                    bezier.role,
                ))
            }
            SketchCurve::Ellipse(id) => {
                let ellipse = *self.ellipses.iter().find(|ellipse| ellipse.id == id)?;
                let candidate = self.ellipse_candidate(ellipse)?;
                Some((
                    candidate.quarters.map(PlanarCurve::RationalBezier).to_vec(),
                    ellipse.role,
                ))
            }
            SketchCurve::Conic(id) => {
                let conic = *self.conics.iter().find(|conic| conic.id == id)?;
                Some((
                    vec![PlanarCurve::RationalBezier(
                        self.conic_candidate(conic)?.curve,
                    )],
                    conic.role,
                ))
            }
            SketchCurve::Spline(id) => {
                let spline = self.splines.iter().find(|spline| spline.id == id)?;
                Some((
                    self.spline_candidate(spline)?
                        .pieces
                        .into_iter()
                        .map(PlanarCurve::RationalBezier)
                        .collect(),
                    spline.role,
                ))
            }
        }
    }
}

impl SketchSolid {
    /// Clone this producer and append one associative mirror rule atomically.
    pub fn with_mirror_pattern(
        &self,
        sources: impl IntoIterator<Item = SketchCurve>,
        axis: EntityId,
    ) -> Result<Self, SketchPatternRefusal> {
        let mut next = self.clone();
        next.sketch.add_mirror_pattern(sources, axis)?;
        Ok(next)
    }

    /// Clone this producer and append one associative rectangular array atomically.
    pub fn with_rectangular_pattern(
        &self,
        sources: impl IntoIterator<Item = SketchCurve>,
        counts: [u32; 2],
        steps: [SketchVector; 2],
    ) -> Result<Self, SketchPatternRefusal> {
        let mut next = self.clone();
        next.sketch
            .add_rectangular_pattern(sources, counts, steps)?;
        Ok(next)
    }

    /// Clone this producer and append one associative circular array atomically.
    pub fn with_circular_pattern(
        &self,
        sources: impl IntoIterator<Item = SketchCurve>,
        center: EntityId,
        count: u32,
        angle: AngleMeasurement,
    ) -> Result<Self, SketchPatternRefusal> {
        let mut next = self.clone();
        next.sketch
            .add_circular_pattern(sources, center, count, angle)?;
        Ok(next)
    }
}

fn default_context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(NonZeroU32::MIN)
}

fn map_curve(
    curve: PlanarCurve,
    map: impl Fn([f64; 2]) -> [f64; 2],
    reverses_orientation: bool,
) -> PlanarCurve {
    match curve {
        PlanarCurve::Segment { start, end } => PlanarCurve::Segment {
            start: map(start),
            end: map(end),
        },
        PlanarCurve::Arc {
            center,
            radius,
            start_radians,
            sweep_radians,
        } => {
            let start = [
                center[0] + radius * start_radians.cos(),
                center[1] + radius * start_radians.sin(),
            ];
            let center = map(center);
            let start = map(start);
            PlanarCurve::Arc {
                center,
                radius,
                start_radians: (start[1] - center[1]).atan2(start[0] - center[0]),
                sweep_radians: if reverses_orientation {
                    -sweep_radians
                } else {
                    sweep_radians
                },
            }
        }
        PlanarCurve::RationalBezier(curve) => {
            PlanarCurve::RationalBezier(substrate::rational_bezier::RationalBezier {
                control: curve.control.map(&map),
                weights: curve.weights,
            })
        }
    }
}

fn add(point: [f64; 2], offset: [f64; 2]) -> [f64; 2] {
    [point[0] + offset[0], point[1] + offset[1]]
}

fn reflect(point: [f64; 2], from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
    let axis = [to[0] - from[0], to[1] - from[1]];
    let length_squared = axis[0] * axis[0] + axis[1] * axis[1];
    let relative = [point[0] - from[0], point[1] - from[1]];
    let projection = (relative[0] * axis[0] + relative[1] * axis[1]) / length_squared;
    let foot = [
        from[0] + projection * axis[0],
        from[1] + projection * axis[1],
    ];
    [2.0 * foot[0] - point[0], 2.0 * foot[1] - point[1]]
}

fn rotate_about(point: [f64; 2], center: [f64; 2], radians: f64) -> [f64; 2] {
    let relative = [point[0] - center[0], point[1] - center[1]];
    let (sin, cos) = radians.sin_cos();
    [
        center[0] + relative[0] * cos - relative[1] * sin,
        center[1] + relative[0] * sin + relative[1] * cos,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{PlaneAxis, SketchPoint};

    fn segment_source(sketch: &mut Sketch, from: [i64; 2], to: [i64; 2]) -> SketchCurve {
        let from = sketch.add_point(SketchPoint::new(from[0], from[1]));
        let to = sketch.add_point(SketchPoint::new(to[0], to[1]));
        SketchCurve::Segment(sketch.add_segment(from, to))
    }

    #[test]
    fn mirror_regenerates_after_source_moves_and_adds_no_authored_geometry() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let source = segment_source(&mut sketch, [1, 1], [3, 1]);
        let axis = segment_source(&mut sketch, [0, 0], [0, 4]).id();
        sketch
            .add_mirror_pattern([source], axis)
            .expect("valid mirror rule");

        let derived = sketch.derived_pattern_curves(default_context());
        assert_eq!(derived.len(), 1);
        assert_eq!(derived[0].geometry.start(), [-1.0, 1.0]);
        assert_eq!(sketch.segments().len(), 2);

        let source_from = sketch
            .segments()
            .iter()
            .find(|curve| curve.id == source.id())
            .expect("source segment remains authored")
            .from;
        sketch
            .move_point(source_from, SketchPoint::new(2, 2), default_context())
            .expect("moving an unconstrained source succeeds");
        assert_eq!(
            sketch.derived_pattern_curves(default_context())[0]
                .geometry
                .start(),
            [-2.0, 2.0]
        );
    }

    #[test]
    fn rectangular_and_circular_patterns_skip_the_source_instance() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let source = segment_source(&mut sketch, [1, 0], [2, 0]);
        sketch
            .add_rectangular_pattern(
                [source],
                [3, 2],
                [
                    SketchVector::from_continuous(2.0, 0.0),
                    SketchVector::from_continuous(0.0, 4.0),
                ],
            )
            .expect("valid rectangular rule");
        assert_eq!(sketch.derived_pattern_curves(default_context()).len(), 5);

        let center = sketch.add_point(SketchPoint::new(0, 0));
        let full_turn =
            AngleMeasurement::try_from_degrees_f64(360.0).expect("full turn is representable");
        sketch
            .add_circular_pattern([source], center, 4, full_turn)
            .expect("valid circular rule");
        let all = sketch.derived_pattern_curves(default_context());
        assert_eq!(all.len(), 8);
        let circular = all
            .iter()
            .filter(|curve| curve.pattern == sketch.patterns()[1].id)
            .collect::<Vec<_>>();
        assert!((circular[0].geometry.start()[0]).abs() <= 1.0e-12);
        assert!((circular[0].geometry.start()[1] - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn deleting_a_source_or_reference_cascades_the_rule() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let source = segment_source(&mut sketch, [1, 1], [2, 1]);
        let axis = segment_source(&mut sketch, [0, 0], [0, 2]).id();
        sketch
            .add_mirror_pattern([source], axis)
            .expect("valid mirror rule");
        sketch.delete_segment(axis);
        assert!(sketch.patterns().is_empty());
    }

    #[test]
    fn generated_real_curves_participate_in_face_derivation() {
        let mut sketch = Sketch::rectangle(PlaneAxis::Z, 2, 2);
        let sources = sketch
            .segments()
            .iter()
            .map(|segment| SketchCurve::Segment(segment.id))
            .collect::<Vec<_>>();
        sketch
            .add_rectangular_pattern(
                sources,
                [2, 1],
                [
                    SketchVector::from_continuous(4.0, 0.0),
                    SketchVector::from_continuous(0.0, 0.0),
                ],
            )
            .expect("translated rectangle is a valid pattern");

        assert_eq!(sketch.faces(default_context()).len(), 2);
    }

    #[test]
    fn rectangular_spacing_retargets_with_the_sketch() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let source = segment_source(&mut sketch, [0, 0], [1, 0]);
        sketch
            .add_rectangular_pattern(
                [source],
                [2, 1],
                [
                    SketchVector::from_continuous(3.0, 0.0),
                    SketchVector::from_continuous(0.0, 0.0),
                ],
            )
            .expect("valid one-direction pattern");

        sketch.retarget_density(16, 32);
        let generated = sketch.derived_pattern_curves(default_context());
        assert_eq!(generated[0].geometry.start(), [6.0, 0.0]);
    }
}
