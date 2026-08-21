//! Where the pointer is, and where the press that is still down began.
//!
//! Two facts that were two loose `Option` fields, and had drifted apart: the release path cleared
//! the POSITION along with the press latches, so a second click that arrived without any pointer
//! motion in between found no position, recorded no press, and did nothing at all. A perfectly
//! still hand can double-click — a mouse reports motion only when it moves — so a whole class of
//! gesture was inert, and nothing could see it because neither field was anybody's to keep.
//!
//! ## The invariant
//!
//! **A release forgets the press. It does not forget where the pointer is.** The pointer is still
//! there; letting go of a button does not move it. Only leaving the window does, and that is
//! [`left`](PointerTrack::left).

/// The pointer's position and the press it is holding, in PHYSICAL pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct PointerTrack {
    /// Where the pointer last was. `None` before the first motion, and once it leaves the window.
    at: Option<(f64, f64)>,
    /// Where the press that is still down began. `None` when no button is held.
    pressed_at: Option<(f64, f64)>,
}

impl PointerTrack {
    /// The pointer moved to `position`.
    pub(super) const fn see(&mut self, position: (f64, f64)) {
        self.at = Some(position);
    }

    /// A button went down, at wherever the pointer is.
    ///
    /// A press with no known position records none, which is honest: the window has not been
    /// told where the pointer is, so there is nothing to compare a later release against.
    pub(super) const fn press(&mut self) {
        self.pressed_at = self.at;
    }

    /// The press is over. The position SURVIVES — see the module's invariant.
    pub(super) const fn end_press(&mut self) {
        self.pressed_at = None;
    }

    /// The pointer left the window, so there is no longer a place it is.
    pub(super) const fn left(&mut self) {
        self.at = None;
        self.pressed_at = None;
    }

    /// Where the pointer is.
    pub(super) const fn at(&self) -> Option<(f64, f64)> {
        self.at
    }

    /// Where the held press began.
    pub(super) const fn pressed_at(&self) -> Option<(f64, f64)> {
        self.pressed_at
    }

    /// Where the press began and where the pointer is now, as `(down_x, down_y, up_x, up_y)`.
    ///
    /// The pair every click-versus-drag test reads: a release compares the two ends against its
    /// own stationary threshold. `None` unless BOTH are known.
    pub(super) const fn press_and_now(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.pressed_at, self.at) {
            (Some((down_x, down_y)), Some((up_x, up_y))) => Some((down_x, down_y, up_x, up_y)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PointerTrack;

    /// A press before the window has ever seen the pointer has no place, and neither does the
    /// release that follows it.
    #[test]
    fn a_press_with_no_known_position_records_none() {
        let mut pointer = PointerTrack::default();
        pointer.press();
        assert_eq!(pointer.pressed_at(), None);
        assert_eq!(pointer.press_and_now(), None);
    }

    /// One ordinary click: move, press, release.
    #[test]
    fn a_click_reports_where_it_went_down_and_where_it_came_up() {
        let mut pointer = PointerTrack::default();
        pointer.see((100.0, 50.0));
        pointer.press();
        pointer.see((104.0, 53.0));
        assert_eq!(pointer.press_and_now(), Some((100.0, 50.0, 104.0, 53.0)));
        pointer.end_press();
        assert_eq!(pointer.pressed_at(), None, "the release forgets the press");
    }

    /// THE regression. A double click from a hand that does not move reports no motion between
    /// the two clicks, so the second one has only what the first left behind.
    ///
    /// The release used to clear the position along with the press latches, which made the second
    /// press positionless and the second release a no-op — every double-click gesture, on a steady
    /// hand, silently did nothing.
    #[test]
    fn a_second_click_in_the_same_place_is_still_a_click() {
        let mut pointer = PointerTrack::default();
        pointer.see((820.0, 415.0));

        pointer.press();
        assert_eq!(pointer.press_and_now(), Some((820.0, 415.0, 820.0, 415.0)));
        pointer.end_press();

        // No `see` here on purpose: a still mouse sends no motion.
        pointer.press();
        assert_eq!(
            pointer.press_and_now(),
            Some((820.0, 415.0, 820.0, 415.0)),
            "the second click of a double click knows where it is"
        );
    }

    /// Leaving the window is the one thing that does erase the position — there is no longer a
    /// place the pointer is, and a stale one would answer hover tests for a cursor that is gone.
    #[test]
    fn leaving_the_window_erases_the_place() {
        let mut pointer = PointerTrack::default();
        pointer.see((10.0, 10.0));
        pointer.press();
        pointer.left();
        assert_eq!(pointer.at(), None);
        assert_eq!(pointer.pressed_at(), None);
    }
}
