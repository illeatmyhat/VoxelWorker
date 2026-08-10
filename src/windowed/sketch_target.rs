//! Private resolution seam shared by sketch drawing tools.

use document::sketch::{EntityId, SketchCurve, SketchPoint, SketchSolid, SketchTarget};
use parametric::EvaluationContext;

/// Resolve one sketch-plane target. A grabbed vertex is authoritative over snap policy so an
/// off-grid stored point never previews at one position and commits at another.
///
/// A point under the cursor also outranks a curve under it, which is why a hovered curve never
/// reaches an [`SketchTarget::Existing`]: the author pointing at a vertex means that vertex, even
/// when the vertex happens to sit on an edge. Both are coincidences either way — one to a point,
/// one to a curve — but only the point-to-point one says which point. The two answers live in
/// separate arms of the target, so no caller can assert both for one click.
///
/// A pick that does land on a curve lands ON it — the grid decides where along the curve, the
/// curve decides the rest. Every drawing tool gets that, whether or not it goes on to plant a
/// coincidence there, because the affordance the author saw said nothing about which it was.
pub(super) fn resolve_target(
    producer: &SketchSolid,
    grabbed: Option<EntityId>,
    snapped: Option<SketchPoint>,
    hovered: Option<SketchCurve>,
    context: EvaluationContext,
) -> Option<SketchTarget> {
    if let Some(id) = grabbed {
        let at = producer
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)?
            .at;
        return Some(SketchTarget::Existing { id, at });
    }
    // A pick taken on a curve stands ON the curve, whatever the grid would otherwise have said.
    // The highlight already told the author what they are aiming at, and leaving the pick a
    // fraction of a step off the curve would make that a promise only the tools that plant a
    // coincidence go on to keep.
    let snapped = snapped?;
    let at = hovered
        .and_then(|curve| producer.sketch.point_on_curve(curve, snapped, context))
        .unwrap_or(snapped);
    Some(
        producer
            .sketch
            .point_at(at)
            .map_or(SketchTarget::Fresh { at, onto: hovered }, |id| {
                SketchTarget::Existing { id, at }
            }),
    )
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn context() -> EvaluationContext {
        EvaluationContext::new(std::num::NonZeroU32::new(16).unwrap())
    }

    fn empty() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    #[test]
    fn grabbed_off_grid_vertex_wins_over_the_snapped_cursor_target() {
        let (producer, grabbed) =
            empty().with_point_placed(SketchPoint::from_continuous(3.25, 4.75));
        let resolved = resolve_target(
            &producer,
            Some(grabbed),
            Some(SketchPoint::new(3, 5)),
            None,
            context(),
        )
        .unwrap();
        assert_eq!(resolved.existing(), Some(grabbed));
        assert_eq!(resolved.at().in_plane(), [3.25, 4.75]);
    }

    #[test]
    fn snapped_coordinate_reuses_an_existing_identity_and_missing_inputs_refuse() {
        let at = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(at);
        assert_eq!(
            resolve_target(&producer, None, Some(at), None, context()),
            Some(SketchTarget::Existing { id: existing, at })
        );
        assert_eq!(resolve_target(&producer, None, None, None, context()), None);
        assert_eq!(
            resolve_target(&producer, Some(9999), Some(at), None, context()),
            None
        );
    }

    /// **A curve under the cursor travels, and a point under the cursor takes precedence over it.**
    ///
    /// The whole rule rests on which arm answers: a tool that mints a point here can only hold it
    /// to the curve if the curve is still named by the time the tool sees the target. Resolving to
    /// `Existing` when a point answers is what keeps the two coincidences from both being asserted
    /// for one click, and the type is what keeps that from being a rule anyone has to remember.
    #[test]
    fn a_hovered_curve_travels_unless_a_point_answers_first() {
        let on_a_curve = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(on_a_curve);
        let hovered = SketchCurve::Segment(7);

        let empty_plane = SketchPoint::new(20, 20);
        assert_eq!(
            resolve_target(&producer, None, Some(empty_plane), Some(hovered), context()),
            Some(SketchTarget::Fresh {
                at: empty_plane,
                onto: Some(hovered),
            }),
            "nothing else is there, so the curve is what the click landed on"
        );
        assert_eq!(
            resolve_target(&producer, None, Some(on_a_curve), Some(hovered), context()),
            Some(SketchTarget::Existing {
                id: existing,
                at: on_a_curve,
            }),
            "the point outranks the curve it sits on"
        );
        assert_eq!(
            resolve_target(
                &producer,
                Some(existing),
                Some(empty_plane),
                Some(hovered),
                context()
            ),
            Some(SketchTarget::Existing {
                id: existing,
                at: on_a_curve,
            }),
            "and so does a grabbed one, wherever the cursor drifted to"
        );
    }

    /// **A pick taken on a curve stands on the curve, not on the grid step nearest it.**
    ///
    /// Which is what makes the highlight mean the same thing to every tool. A tool that plants a
    /// point would eventually be pulled onto the curve by its coincidence; a tool that only reads
    /// a radius off the pick has nothing to pull it, and would quietly draw a circle sized from a
    /// point beside the curve the author was pointing at.
    #[test]
    fn a_pick_on_a_curve_lands_on_it() {
        let (with_tail, tail) = empty().with_point_placed(SketchPoint::new(0, 0));
        let (with_head, head) = with_tail.with_point_placed(SketchPoint::new(40, 0));
        let (rail, segment) = with_head.with_segment_between_traced(tail, head).unwrap();

        let beside = SketchPoint::new(12, 3);
        let resolved = resolve_target(&rail, None, Some(beside), Some(segment), context()).unwrap();
        assert_eq!(resolved.at().in_plane(), [12.0, 0.0]);
        assert_eq!(resolved.onto(), Some(segment));

        assert_eq!(
            resolve_target(&rail, None, Some(beside), None, context())
                .unwrap()
                .at()
                .in_plane(),
            [12.0, 3.0],
            "with no curve under it the pick keeps the position the grid gave it"
        );
    }

    /// **Every curve kind answers, aggregates included.**
    ///
    /// A spline carries no relation geometry — it has no one center, radius or direction for a
    /// relation to be about — and the drawing turns a Tangent or a point-on-curve on one away for
    /// exactly that reason. None of it is a statement about where the pointer is standing. Landing
    /// a pick asks only for a position, and a spline has one everywhere along it, so the kinds a
    /// relation refuses still snap.
    #[test]
    fn a_pick_lands_on_a_spline_even_though_no_relation_can_hold_it_there() {
        let mut drawing = Sketch::empty(PlaneAxis::Z);
        let spline = drawing
            .add_fit_point_spline(
                &[
                    SketchPoint::new(0, 0),
                    SketchPoint::new(10, 10),
                    SketchPoint::new(20, 0),
                ],
                false,
            )
            .unwrap();
        let producer = SketchSolid::extrude(drawing, 3);
        let curve = SketchCurve::Spline(spline);
        assert!(
            !curve.carries_relation_geometry(),
            "the premise: no relation can be held to this"
        );

        let beside = SketchPoint::new(10, 20);
        let landed = resolve_target(&producer, None, Some(beside), Some(curve), context())
            .unwrap()
            .at();
        assert_ne!(landed.in_plane(), beside.in_plane(), "it moved");

        // Landing twice lands in the same place, which is what says the first landing reached the
        // curve rather than merely stepping toward it.
        let again = producer
            .sketch
            .point_on_curve(curve, landed, context())
            .unwrap();
        let (settled, once) = (again.in_plane(), landed.in_plane());
        assert!(
            (settled[0] - once[0]).hypot(settled[1] - once[1]) < 1.0e-6,
            "{settled:?} vs {once:?}"
        );
    }
}
