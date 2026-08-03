//! Which way a bearing is going, from a run of readings.
//!
//! A bearing read on its own cannot say which way something has been going. `atan2` wraps, so a
//! pointer that has swung 200 degrees one way and one that has swung 160 degrees the other way
//! report the same angle. Only the SEQUENCE of readings distinguishes them, so the direction has to
//! come from the deltas rather than from the latest value.
//!
//! What comes out is a direction, not a tally. Someone who sweeps three times round has not asked
//! for three turns of anything — they have asked to go that way, and the cost of changing their
//! mind should be the same small movement whether they went round once or ten times. That is why
//! the running total SATURATES at a dead band instead of counting: it only ever holds enough
//! history to resist jitter.

use std::f64::consts::{PI, TAU};

/// How far the running total may build up in either direction, in radians — ten degrees.
///
/// It is the whole memory of the latch. Reversing therefore costs twice this much counter-motion
/// (from one rail across to the other) no matter how far the sweep went first, and anything smaller
/// than it is jitter that cannot flip the answer.
const DIRECTION_DEAD_BAND: f64 = PI / 18.0;

/// Wrap an angle difference into `(-PI, PI]`.
///
/// `sticky` breaks the one ambiguous case. A delta of exactly half a turn could have gone either
/// way, and picking a fixed side would make a jump — a snap landing across the circle — flip the
/// direction the author had established. Carrying the sign already held keeps the direction they
/// were going.
#[must_use]
pub fn wrap_signed(delta: f64, sticky: f64) -> f64 {
    if !delta.is_finite() {
        return 0.0;
    }
    let wrapped = delta.rem_euclid(TAU);
    let wrapped = if wrapped > PI { wrapped - TAU } else { wrapped };
    if (wrapped.abs() - PI).abs() <= f64::EPSILON && sticky < 0.0 {
        return -PI;
    }
    wrapped
}

/// Which way a bearing has been going lately: positive one way, negative the other.
///
/// Not how far — [`held`](Self::held) saturates at [`DIRECTION_DEAD_BAND`]. Ask it for a sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnLatch {
    last_bearing: f64,
    held: f64,
}

impl TurnLatch {
    /// Begin at a bearing, with no direction yet established.
    #[must_use]
    pub const fn starting_at(bearing: f64) -> Self {
        Self {
            last_bearing: bearing,
            held: 0.0,
        }
    }

    /// Take one more reading.
    ///
    /// A repeated bearing contributes nothing, so a still cursor cannot drift. Callers that cannot
    /// trust a reading — a pointer sitting on the center, where the bearing is noise — should skip
    /// the call rather than pass a fabricated one.
    pub fn advance(&mut self, bearing: f64) -> f64 {
        if !bearing.is_finite() {
            return self.held;
        }
        let delta = wrap_signed(bearing - self.last_bearing, self.held);
        self.last_bearing = bearing;
        self.held = (self.held + delta).clamp(-DIRECTION_DEAD_BAND, DIRECTION_DEAD_BAND);
        self.held
    }

    /// The latched direction as a signed value: negative one way, positive the other, zero before
    /// any movement has established one.
    #[must_use]
    pub const fn held(&self) -> f64 {
        self.held
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn a_still_reading_adds_nothing() {
        let mut latch = TurnLatch::starting_at(0.4);
        for _ in 0..100 {
            latch.advance(0.4);
        }
        assert_eq!(latch.held(), 0.0);
    }

    /// The whole point: two paths ending at the same bearing report opposite directions.
    #[test]
    fn the_path_decides_the_direction_not_the_destination() {
        let steps = 40;
        let mut counter_clockwise = TurnLatch::starting_at(0.0);
        let mut clockwise = TurnLatch::starting_at(0.0);
        for step in 1..=steps {
            let fraction = f64::from(step) / f64::from(steps);
            counter_clockwise.advance(fraction * 3.0);
            clockwise.advance(-fraction * (TAU - 3.0));
        }
        assert!(counter_clockwise.held() > 0.0);
        assert!(clockwise.held() < 0.0);
    }

    /// Going round and round is still just going that way. This is the complaint the dead band
    /// exists to answer: nobody should have to give back three revolutions to change their mind.
    #[test]
    fn sweeping_round_repeatedly_costs_no_more_to_reverse_than_a_nudge_does() {
        let mut once = TurnLatch::starting_at(0.0);
        let mut many = TurnLatch::starting_at(0.0);
        once.advance(0.5);
        for step in 1..=1000 {
            many.advance(f64::from(step) * 0.1);
        }
        assert!(once.held() > 0.0 && many.held() > 0.0);
        assert_eq!(once.held(), many.held(), "both are simply going that way");

        // The same counter-motion flips both, however far each of them went.
        let (mut once_bearing, mut many_bearing) = (0.5, 100.0);
        for _ in 0..8 {
            once_bearing -= 0.05;
            many_bearing -= 0.05;
            once.advance(once_bearing);
            many.advance(many_bearing);
        }
        assert!(once.held() < 0.0, "{}", once.held());
        assert!(many.held() < 0.0, "{}", many.held());
    }

    #[test]
    fn jitter_smaller_than_the_dead_band_cannot_flip_an_established_direction() {
        let mut latch = TurnLatch::starting_at(0.0);
        for step in 1..=20 {
            latch.advance(f64::from(step) * 0.1);
        }
        let mut bearing = 2.0;
        for _ in 0..10 {
            bearing -= 0.02;
            latch.advance(bearing);
            bearing += 0.02;
            latch.advance(bearing);
        }
        assert!(latch.held() > 0.0, "{}", latch.held());
    }

    /// A jump exactly across the circle keeps the direction already established.
    #[test]
    fn a_half_turn_jump_follows_the_direction_already_underway() {
        assert_eq!(wrap_signed(PI, -0.5), -PI);
        assert_eq!(wrap_signed(PI, 0.5), PI);
        assert_eq!(wrap_signed(PI, 0.0), PI);
    }

    #[test]
    fn a_non_finite_reading_is_ignored_rather_than_poisoning_the_latch() {
        let mut latch = TurnLatch::starting_at(0.0);
        latch.advance(0.05);
        latch.advance(f64::NAN);
        assert_eq!(latch.held(), 0.05);
        assert!((latch.advance(0.09) - 0.09).abs() < 1.0e-12);
    }
}
