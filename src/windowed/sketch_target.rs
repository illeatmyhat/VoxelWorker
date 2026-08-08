//! Private resolution seam shared by sketch drawing tools.

use document::sketch::{EntityId, SketchCurve, SketchPoint, SketchSolid};

/// The canonical position under the cursor and, when it names stored geometry, that point's
/// stable identity. Callers decide whether identity is meaningful for their particular input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ResolvedSketchTarget {
    pub at: SketchPoint,
    pub existing: Option<EntityId>,
    /// The curve the cursor is over, when it is over one and not over a point. A tool that MINTS a
    /// point here holds it to this curve; a tool that only reads a position ignores the field.
    pub on_curve: Option<SketchCurve>,
}

/// Resolve one sketch-plane target. A grabbed vertex is authoritative over snap policy so an
/// off-grid stored point never previews at one position and commits at another.
///
/// A point under the cursor also outranks a curve under it, which is why `on_curve` is dropped
/// whenever `existing` answers: the author pointing at a vertex means that vertex, even when the
/// vertex happens to sit on an edge. Both are coincidences either way — one to a point, one to a
/// curve — but only the point-to-point one says which point.
pub(super) fn resolve_target(
    producer: &SketchSolid,
    grabbed: Option<EntityId>,
    snapped: Option<SketchPoint>,
    hovered: Option<SketchCurve>,
) -> Option<ResolvedSketchTarget> {
    if let Some(id) = grabbed {
        let at = producer
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)?
            .at;
        return Some(ResolvedSketchTarget {
            at,
            existing: Some(id),
            on_curve: None,
        });
    }
    let at = snapped?;
    let existing = producer.sketch.point_at(at);
    Some(ResolvedSketchTarget {
        at,
        existing,
        on_curve: hovered.filter(|_| existing.is_none()),
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn empty() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    #[test]
    fn grabbed_off_grid_vertex_wins_over_the_snapped_cursor_target() {
        let (producer, grabbed) =
            empty().with_point_placed(SketchPoint::from_continuous(3.25, 4.75));
        let resolved =
            resolve_target(&producer, Some(grabbed), Some(SketchPoint::new(3, 5)), None).unwrap();
        assert_eq!(resolved.existing, Some(grabbed));
        assert_eq!(resolved.at.in_plane(), [3.25, 4.75]);
    }

    #[test]
    fn snapped_coordinate_reuses_an_existing_identity_and_missing_inputs_refuse() {
        let at = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(at);
        assert_eq!(
            resolve_target(&producer, None, Some(at), None),
            Some(ResolvedSketchTarget {
                at,
                existing: Some(existing),
                on_curve: None,
            })
        );
        assert_eq!(resolve_target(&producer, None, None, None), None);
        assert_eq!(resolve_target(&producer, Some(9999), Some(at), None), None);
    }

    /// **A curve under the cursor travels, and a point under the cursor takes precedence over it.**
    ///
    /// The whole rule rests on this field: a tool that mints a point here can only hold it to the
    /// curve if the curve is still named by the time the tool sees the target. Dropping it when a
    /// point answers is what keeps the two coincidences from both being asserted for one click.
    #[test]
    fn a_hovered_curve_travels_unless_a_point_answers_first() {
        let on_a_curve = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(on_a_curve);
        let hovered = SketchCurve::Segment(7);

        let empty_plane = SketchPoint::new(20, 20);
        assert_eq!(
            resolve_target(&producer, None, Some(empty_plane), Some(hovered))
                .and_then(|resolved| resolved.on_curve),
            Some(hovered),
            "nothing else is there, so the curve is what the click landed on"
        );
        assert_eq!(
            resolve_target(&producer, None, Some(on_a_curve), Some(hovered)),
            Some(ResolvedSketchTarget {
                at: on_a_curve,
                existing: Some(existing),
                on_curve: None,
            }),
            "the point outranks the curve it sits on"
        );
        assert_eq!(
            resolve_target(&producer, Some(existing), Some(empty_plane), Some(hovered))
                .and_then(|resolved| resolved.on_curve),
            None,
            "and so does a grabbed one, wherever the cursor drifted to"
        );
    }
}
