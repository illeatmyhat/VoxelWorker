//! Which quantity a dimension states, read from where its annotation was dropped.
//!
//! A segment between two points offers three different lengths, and they are three different
//! claims: how far apart the points are, how far apart they are ACROSS the plane, and how far
//! apart they are UP it. An author who wants the second does not want the first stated instead,
//! and there is no way to ask them without either a second command or a modifier key.
//!
//! Fusion answers with the drawing itself. The segment is the diagonal of an axis-aligned
//! rectangle; extending that rectangle's four sides cuts the plane into nine regions, and the
//! region the text lands in is the question. Drop it above or below and the dimension line is
//! horizontal, so the quantity is horizontal. Drop it left or right and the line is vertical, so
//! the quantity is. Drop it out past a corner — which is where moving PERPENDICULAR to a diagonal
//! takes you — and the line is parallel to the segment, so the quantity is its true length.
//!
//! The rule is not a convention to memorize: in every case the dimension you get is the one whose
//! dimension line could actually be drawn where the cursor is. The author is pointing at the
//! answer.
//!
//! Read in the sketch plane's own coordinates, never on screen. Horizontal means the plane's first
//! axis, the same thing [`Relation::Horizontal`](super::Relation::Horizontal) means, and a rule
//! read after a projection would give a different answer from a different camera.

/// Which of the three lengths between two points a dropped annotation is asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanReading {
    /// The distance between the points — the dimension line runs parallel to the segment.
    Aligned,
    /// How far apart they are along the plane's first axis — a horizontal dimension line.
    AcrossThePlane,
    /// How far apart they are along the plane's second axis — a vertical dimension line.
    UpThePlane,
}

/// Where a coordinate sits against a closed interval: the one-dimensional half of the region grid.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Against {
    Before,
    Within,
    Beyond,
}

impl Against {
    /// `value` against `[low, high]`, which the caller has already put in order.
    fn read(value: f64, low: f64, high: f64) -> Self {
        if value < low {
            Self::Before
        } else if value > high {
            Self::Beyond
        } else {
            Self::Within
        }
    }

    /// Whether the coordinate is inside the segment's extent on this axis, which is what makes a
    /// dimension line perpendicular to it drawable there.
    const fn is_within(self) -> bool {
        matches!(self, Self::Within)
    }
}

/// Which length the author is asking for, having dropped the annotation at `anchor`.
///
/// `from` and `to` are the segment's ends and `anchor` the drop point, all in the sketch plane's
/// own coordinates.
///
/// An axis-aligned segment reads [`Aligned`](SpanReading::Aligned) from everywhere, and that is
/// not a special case being papered over: a vertical segment's horizontal extent is zero, so the
/// horizontal dimension it would otherwise offer is the claim that two points share a coordinate,
/// which is [`Vertical`](super::Relation::Vertical) and not a number the author types. Its aligned
/// length and its vertical length are the same quantity anyway.
///
/// The rectangle's INTERIOR reads aligned too. There is no room in there for a dimension line that
/// clears the geometry, so the author has not yet said anything; aligned is what the gesture
/// started as and what it stays until they move somewhere that means something else.
#[must_use]
pub fn span_reading(from: [f64; 2], to: [f64; 2], anchor: [f64; 2]) -> SpanReading {
    let (left, right) = (from[0].min(to[0]), from[0].max(to[0]));
    let (bottom, top) = (from[1].min(to[1]), from[1].max(to[1]));
    // A degenerate extent offers no dimension across it — see the doc comment. Compared against
    // zero rather than a tolerance: a segment a thousandth of a voxel off vertical still HAS a
    // horizontal extent, and refusing to state it would be the tool overruling the drawing.
    let (has_width, has_height) = (right > left, top > bottom);
    let across = Against::read(anchor[0], left, right);
    let up = Against::read(anchor[1], bottom, top);
    match (across.is_within(), up.is_within()) {
        // Directly above or below: a horizontal line fits there and a vertical one does not.
        (true, false) if has_width => SpanReading::AcrossThePlane,
        // Directly beside: the other way round.
        (false, true) if has_height => SpanReading::UpThePlane,
        // Out past a corner, or inside the rectangle.
        _ => SpanReading::Aligned,
    }
}

#[cfg(test)]
mod tests {
    use super::{span_reading, SpanReading};

    /// The diagonal of the unit-ish rectangle every case below is read against: (0,0) to (10,6).
    const FROM: [f64; 2] = [0.0, 0.0];
    const TO: [f64; 2] = [10.0, 6.0];

    /// **The nine regions, one assertion each.**
    ///
    /// The grid IS the rule, so the test is the grid. Read it as a picture: the middle row is the
    /// segment's own band, the middle column its own span, and the corners are where perpendicular
    /// takes you off a diagonal.
    #[test]
    fn the_region_the_text_lands_in_is_the_question() {
        let cases = [
            // (anchor, expected, what the author did)
            ([5.0, 20.0], SpanReading::AcrossThePlane, "above"),
            ([5.0, -20.0], SpanReading::AcrossThePlane, "below"),
            ([-20.0, 3.0], SpanReading::UpThePlane, "left"),
            ([20.0, 3.0], SpanReading::UpThePlane, "right"),
            ([-20.0, 20.0], SpanReading::Aligned, "up and left"),
            ([20.0, 20.0], SpanReading::Aligned, "up and right"),
            ([-20.0, -20.0], SpanReading::Aligned, "down and left"),
            ([20.0, -20.0], SpanReading::Aligned, "down and right"),
            ([5.0, 3.0], SpanReading::Aligned, "inside the rectangle"),
        ];
        for (anchor, expected, what) in cases {
            assert_eq!(span_reading(FROM, TO, anchor), expected, "dropped {what}");
        }
    }

    /// The rectangle's own edges belong to the band, not to the outside — so a drop level with an
    /// endpoint reads the same as one in the middle of the run rather than flipping on a
    /// hairsbreadth of cursor travel.
    #[test]
    fn an_edge_of_the_rectangle_counts_as_inside_it() {
        assert_eq!(
            span_reading(FROM, TO, [0.0, 20.0]),
            SpanReading::AcrossThePlane,
            "level with the left end, but still above the run"
        );
        assert_eq!(
            span_reading(FROM, TO, [20.0, 6.0]),
            SpanReading::UpThePlane,
            "level with the top end, but still right of the run"
        );
    }

    /// **An axis-aligned segment never offers the dimension it has no extent for**, and still
    /// offers the one it does.
    ///
    /// A vertical segment's width is zero, so no drop asks for it — that claim is
    /// [`Vertical`](crate::sketch::Relation::Vertical), not a number. Its HEIGHT is still a real
    /// question, and reading it as such rather than folding it into the aligned length matters
    /// later: the two are the same number today and different claims the moment the segment turns.
    #[test]
    fn a_straight_segment_never_states_the_extent_it_does_not_have() {
        let (from, to) = ([0.0, 0.0], [0.0, 10.0]);
        for anchor in [[0.0, 20.0], [0.0, -20.0], [9.0, 20.0], [-9.0, -20.0]] {
            assert_eq!(
                span_reading(from, to, anchor),
                SpanReading::Aligned,
                "a vertical segment has no width to state, at {anchor:?}"
            );
        }
        for anchor in [[9.0, 5.0], [-9.0, 5.0]] {
            assert_eq!(
                span_reading(from, to, anchor),
                SpanReading::UpThePlane,
                "beside it, the dimension line is vertical and so is the claim"
            );
        }

        let (from, to) = ([0.0, 0.0], [10.0, 0.0]);
        for anchor in [[20.0, 0.0], [-20.0, 0.0], [20.0, 9.0], [-20.0, -9.0]] {
            assert_eq!(
                span_reading(from, to, anchor),
                SpanReading::Aligned,
                "a horizontal segment has no height to state, at {anchor:?}"
            );
        }
        for anchor in [[5.0, 9.0], [5.0, -9.0]] {
            assert_eq!(
                span_reading(from, to, anchor),
                SpanReading::AcrossThePlane,
                "above it, the dimension line is horizontal and so is the claim"
            );
        }
    }

    /// The reading does not depend on which end was picked first: a segment drawn the other way
    /// round is the same rectangle, and the author is pointing at the same region of it.
    #[test]
    fn the_order_the_ends_were_picked_in_says_nothing() {
        for anchor in [[5.0, 20.0], [20.0, 3.0], [-20.0, -20.0], [5.0, 3.0]] {
            assert_eq!(
                span_reading(FROM, TO, anchor),
                span_reading(TO, FROM, anchor),
                "at {anchor:?}"
            );
        }
    }
}
