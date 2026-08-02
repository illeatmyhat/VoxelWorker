//! Integrating a bearing into a signed turn.
//!
//! A bearing read on its own cannot say which way something has been going. `atan2` wraps, so a
//! pointer that has swung 200 degrees one way and one that has swung 160 degrees the other way
//! report the same angle. Only the sequence of readings distinguishes them, so direction has to be
//! integrated from the deltas rather than recovered from the latest value.

use std::f64::consts::{PI, TAU};

/// The largest turn an accumulator will report, just under one full revolution.
///
/// Clamping the STORED total, rather than a hidden raw one, is what makes a reversal unwind
/// immediately: an author who has wound past a full turn and comes back does not have to pay back
/// revolutions nobody could see.
const TURN_LIMIT: f64 = TAU - 1.0e-9;

/// Wrap an angle difference into `(-PI, PI]`.
///
/// `sticky` breaks the one ambiguous case. A delta of exactly half a turn could have gone either
/// way, and picking a fixed side would make a jump — a snap landing across the circle — flip the
/// direction the author had established. Carrying the sign already accumulated keeps the winding
/// they were doing.
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

/// How far a bearing has turned in total, sign included, across a run of readings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindingAccumulator {
    last_bearing: f64,
    turned: f64,
}

impl WindingAccumulator {
    /// Begin at a bearing, having turned nothing yet.
    #[must_use]
    pub const fn starting_at(bearing: f64) -> Self {
        Self {
            last_bearing: bearing,
            turned: 0.0,
        }
    }

    /// Take one more reading, and report the total turn including it.
    ///
    /// A repeated bearing contributes nothing, so a still cursor cannot drift. Callers that cannot
    /// trust a reading — a pointer sitting on the center, where the bearing is noise — should skip
    /// the call rather than pass a fabricated one.
    pub fn advance(&mut self, bearing: f64) -> f64 {
        if !bearing.is_finite() {
            return self.turned;
        }
        let delta = wrap_signed(bearing - self.last_bearing, self.turned);
        self.last_bearing = bearing;
        self.turned = (self.turned + delta).clamp(-TURN_LIMIT, TURN_LIMIT);
        self.turned
    }

    /// The total turn so far.
    #[must_use]
    pub const fn turned(&self) -> f64 {
        self.turned
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn a_still_reading_adds_nothing() {
        let mut winding = WindingAccumulator::starting_at(0.4);
        for _ in 0..100 {
            winding.advance(0.4);
        }
        assert_eq!(winding.turned(), 0.0);
    }

    /// The whole point: two paths ending at the same bearing report opposite turns.
    #[test]
    fn the_path_decides_the_direction_not_the_destination() {
        let steps = 40;
        let mut counter_clockwise = WindingAccumulator::starting_at(0.0);
        let mut clockwise = WindingAccumulator::starting_at(0.0);
        for step in 1..=steps {
            let fraction = f64::from(step) / f64::from(steps);
            counter_clockwise.advance(fraction * 3.0);
            clockwise.advance(-fraction * (TAU - 3.0));
        }
        assert!((counter_clockwise.turned() - 3.0).abs() < 1.0e-12);
        assert!((clockwise.turned() + (TAU - 3.0)).abs() < 1.0e-12);
        // Same place on the circle, opposite histories.
        let apart = (counter_clockwise.turned() - clockwise.turned()).rem_euclid(TAU);
        assert!(apart < 1.0e-9 || (TAU - apart) < 1.0e-9);
    }

    #[test]
    fn winding_past_a_half_turn_keeps_going_instead_of_flipping() {
        let mut winding = WindingAccumulator::starting_at(0.0);
        for step in 1..=60 {
            winding.advance(f64::from(step) * 0.1);
        }
        assert!(winding.turned() > PI, "{}", winding.turned());
    }

    #[test]
    fn a_reversal_unwinds_from_where_it_stopped() {
        let mut winding = WindingAccumulator::starting_at(0.0);
        for step in 1..=30 {
            winding.advance(f64::from(step) * 0.1);
        }
        let peak = winding.turned();
        for step in (0..30).rev() {
            winding.advance(f64::from(step) * 0.1);
        }
        assert!(
            winding.turned().abs() < 1.0e-12,
            "{peak} then {}",
            winding.turned()
        );
    }

    #[test]
    fn no_reading_can_wind_past_a_full_turn() {
        let mut winding = WindingAccumulator::starting_at(0.0);
        for step in 1..=1000 {
            winding.advance(f64::from(step) * 0.1);
        }
        assert!(winding.turned() <= TURN_LIMIT);
        assert!(winding.turned() > 0.0);
    }

    /// A jump exactly across the circle keeps the direction already established.
    #[test]
    fn a_half_turn_jump_follows_the_winding_already_underway() {
        assert_eq!(wrap_signed(PI, -0.5), -PI);
        assert_eq!(wrap_signed(PI, 0.5), PI);
        assert_eq!(wrap_signed(PI, 0.0), PI);
    }

    #[test]
    fn a_non_finite_reading_is_ignored_rather_than_poisoning_the_total() {
        let mut winding = WindingAccumulator::starting_at(0.0);
        winding.advance(1.0);
        winding.advance(f64::NAN);
        assert_eq!(winding.turned(), 1.0);
        assert!((winding.advance(1.5) - 1.5).abs() < 1.0e-12);
    }
}
