//! Private resolution seam shared by sketch drawing tools.

use document::sketch::{EntityId, SketchCurve, SketchPoint, SketchSolid, SketchTarget};

/// Resolve one sketch-plane target. A grabbed vertex is authoritative over snap policy so an
/// off-grid stored point never previews at one position and commits at another.
///
/// A point under the cursor also outranks a curve under it, which is why a hovered curve never
/// reaches an [`SketchTarget::Existing`]: the author pointing at a vertex means that vertex, even
/// when the vertex happens to sit on an edge. Both are coincidences either way — one to a point,
/// one to a curve — but only the point-to-point one says which point. The two answers live in
/// separate arms of the target, so no caller can assert both for one click.
pub(super) fn resolve_target(
    producer: &SketchSolid,
    grabbed: Option<EntityId>,
    snapped: Option<SketchPoint>,
    hovered: Option<SketchCurve>,
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
    let at = snapped?;
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

    fn empty() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    #[test]
    fn grabbed_off_grid_vertex_wins_over_the_snapped_cursor_target() {
        let (producer, grabbed) =
            empty().with_point_placed(SketchPoint::from_continuous(3.25, 4.75));
        let resolved =
            resolve_target(&producer, Some(grabbed), Some(SketchPoint::new(3, 5)), None).unwrap();
        assert_eq!(resolved.existing(), Some(grabbed));
        assert_eq!(resolved.at().in_plane(), [3.25, 4.75]);
    }

    #[test]
    fn snapped_coordinate_reuses_an_existing_identity_and_missing_inputs_refuse() {
        let at = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(at);
        assert_eq!(
            resolve_target(&producer, None, Some(at), None),
            Some(SketchTarget::Existing { id: existing, at })
        );
        assert_eq!(resolve_target(&producer, None, None, None), None);
        assert_eq!(resolve_target(&producer, Some(9999), Some(at), None), None);
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
            resolve_target(&producer, None, Some(empty_plane), Some(hovered)),
            Some(SketchTarget::Fresh {
                at: empty_plane,
                onto: Some(hovered),
            }),
            "nothing else is there, so the curve is what the click landed on"
        );
        assert_eq!(
            resolve_target(&producer, None, Some(on_a_curve), Some(hovered)),
            Some(SketchTarget::Existing {
                id: existing,
                at: on_a_curve,
            }),
            "the point outranks the curve it sits on"
        );
        assert_eq!(
            resolve_target(&producer, Some(existing), Some(empty_plane), Some(hovered)),
            Some(SketchTarget::Existing {
                id: existing,
                at: on_a_curve,
            }),
            "and so does a grabbed one, wherever the cursor drifted to"
        );
    }
}
