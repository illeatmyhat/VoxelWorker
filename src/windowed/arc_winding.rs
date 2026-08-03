//! Which way round the pointer went, for the two center-first arc gestures.
//!
//! A center-first arc's direction is not in its picks. Center, start and a cursor on the far side
//! of the circle describe two arcs equally well, and the author chose between them by the route
//! they swept. Both the Center Point Arc and the center-arc Slot need the same reading, so it lives
//! here once rather than being mirrored into two gesture modules that could drift apart.

use document::sketch::SketchPoint;
use substrate::winding::TurnLatch;

/// How far from the center a reading has to be before its bearing means anything.
///
/// Half a voxel from the center the bearing is dominated by pointer noise, and a hand that pauses
/// there would otherwise scribble a direction the author never chose.
const BEARING_DEAD_ZONE_VOXELS: f64 = 0.5;

fn bearing(center: SketchPoint, at: SketchPoint) -> Option<f64> {
    let center = center.in_plane();
    let at = at.in_plane();
    let offset = [at[0] - center[0], at[1] - center[1]];
    (offset[0].hypot(offset[1]) > BEARING_DEAD_ZONE_VOXELS).then(|| offset[1].atan2(offset[0]))
}

/// Fold one cursor reading into a gesture's direction latch, seeding it if this is the first.
///
/// The seed is the START point's bearing, not the first cursor position: the direction is read from
/// where the arc actually begins, so a pointer that has already travelled before the first frame
/// arrives cannot lose the move it made getting there.
pub(super) fn track(
    winding: &mut Option<TurnLatch>,
    center: SketchPoint,
    start: SketchPoint,
    cursor: SketchPoint,
) {
    let Some(cursor_bearing) = bearing(center, cursor) else {
        return;
    };
    match winding {
        Some(winding) => {
            winding.advance(cursor_bearing);
        }
        None => {
            let mut seeded =
                TurnLatch::starting_at(bearing(center, start).unwrap_or(cursor_bearing));
            seeded.advance(cursor_bearing);
            *winding = Some(seeded);
        }
    }
}

/// The direction a tracked latch describes; counter-clockwise until the pointer says otherwise.
pub(super) fn turn(winding: Option<TurnLatch>) -> parametric::sketch::ArcTurn {
    if winding.is_some_and(|winding| winding.held() < 0.0) {
        parametric::sketch::ArcTurn::Clockwise
    } else {
        parametric::sketch::ArcTurn::CounterClockwise
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn at(axis0: f64, axis1: f64) -> SketchPoint {
        SketchPoint::try_from_continuous(axis0, axis1).unwrap()
    }

    /// The same cursor position, reached two ways, gives two different arcs.
    #[test]
    fn the_route_to_the_cursor_decides_the_turn_not_the_cursor() {
        let center = at(0.0, 0.0);
        let start = at(4.0, 0.0);
        let mut counter_clockwise = None;
        let mut clockwise = None;
        for step in 1..=16 {
            let fraction = f64::from(step) / 16.0;
            let up = fraction * std::f64::consts::PI;
            track(
                &mut counter_clockwise,
                center,
                start,
                at(4.0 * up.cos(), 4.0 * up.sin()),
            );
            track(
                &mut clockwise,
                center,
                start,
                at(4.0 * (-up).cos(), 4.0 * (-up).sin()),
            );
        }
        assert_eq!(
            turn(counter_clockwise),
            parametric::sketch::ArcTurn::CounterClockwise
        );
        assert_eq!(turn(clockwise), parametric::sketch::ArcTurn::Clockwise);
    }

    #[test]
    fn a_cursor_sitting_on_the_center_cannot_scribble_a_direction() {
        let center = at(0.0, 0.0);
        let start = at(4.0, 0.0);
        let mut winding = None;
        track(&mut winding, center, start, at(4.0, -1.0));
        let after_a_real_reading = winding.unwrap().held();
        for _ in 0..32 {
            track(&mut winding, center, start, at(0.1, -0.1));
        }
        assert_eq!(winding.unwrap().held(), after_a_real_reading);
        assert_eq!(turn(winding), parametric::sketch::ArcTurn::Clockwise);
    }

    #[test]
    fn an_untracked_gesture_reads_counter_clockwise() {
        assert_eq!(turn(None), parametric::sketch::ArcTurn::CounterClockwise);
    }
}
