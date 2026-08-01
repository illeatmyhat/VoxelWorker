//! Private resolution seam shared by sketch drawing tools.

use document::sketch::{EntityId, SketchPoint, SketchSolid};

/// The canonical position under the cursor and, when it names stored geometry, that point's
/// stable identity. Callers decide whether identity is meaningful for their particular input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ResolvedSketchTarget {
    pub at: SketchPoint,
    pub existing: Option<EntityId>,
}

/// Resolve one sketch-plane target. A grabbed vertex is authoritative over snap policy so an
/// off-grid stored point never previews at one position and commits at another.
pub(super) fn resolve_target(
    producer: &SketchSolid,
    grabbed: Option<EntityId>,
    snapped: Option<SketchPoint>,
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
        });
    }
    let at = snapped?;
    Some(ResolvedSketchTarget {
        at,
        existing: producer.sketch.point_at(at),
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
            resolve_target(&producer, Some(grabbed), Some(SketchPoint::new(3, 5))).unwrap();
        assert_eq!(resolved.existing, Some(grabbed));
        assert_eq!(resolved.at.in_plane(), [3.25, 4.75]);
    }

    #[test]
    fn snapped_coordinate_reuses_an_existing_identity_and_missing_inputs_refuse() {
        let at = SketchPoint::new(3, 5);
        let (producer, existing) = empty().with_point_placed(at);
        assert_eq!(
            resolve_target(&producer, None, Some(at)),
            Some(ResolvedSketchTarget {
                at,
                existing: Some(existing),
            })
        );
        assert_eq!(resolve_target(&producer, None, None), None);
        assert_eq!(resolve_target(&producer, Some(9999), Some(at)), None);
    }
}
