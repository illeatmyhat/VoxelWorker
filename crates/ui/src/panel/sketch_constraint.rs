//! Arming a constraint, and collecting the entities it needs (ADR 0035, ADR 0030 §5).
//!
//! A constraint ARMS like every other sketch tool: press the rail cell, then pick the geometry it
//! is about. It is not a verb over whatever happened to be selected first. The two models differ
//! in what the author has to know before pressing anything — selection-first requires them to
//! already know each constraint's arity and to have assembled it; arm-first lets the tool ask, one
//! pick at a time, and say no to a pick that does not fit.
//!
//! Saying no is the part that carries its weight. A constraint's slots are typed: Horizontal wants
//! a line, Fix wants a point, and the two-entity relations will want a specific pair. Offering a
//! point to a slot that wants a line is a **refusal that leaves the gesture running** — the tool
//! stays armed and keeps waiting, because a mis-click is not a decision to abandon the command.
//!
//! Completion disarms. Once the last slot fills there is nothing left to ask, so holding the mode
//! open would make every constraint need an explicit end the author has no reason to expect.
//!
//! A verb is not one-to-one with a constraint. `Horizontal / Vertical` is ONE cell that asserts
//! either of two kinds, picked from the drawing — Fusion's arrangement, and the right one: the
//! author is saying "line this up with an axis", and which axis is already visible in what they
//! drew. The badge then reports the answer rather than the question.
//!
//! Only the kinds whose residuals ship are here. The glyphs still missing from the rail —
//! Concentric, Tangent, Curvature, Symmetry and `Quantize` — are drawn and named on the design
//! sheet but have no residual behind them yet
//! (`crates/document/src/sketch/constraint.rs`), and an armable verb that asserts nothing is worse
//! than a cell that is not there. The first three wait on arcs and circles entering the
//! parameter vector.

use document::sketch::{ConstraintKind, EntityId, Sketch};

use crate::icons::Icon;

/// A sketch entity a constraint can name. Arcs are pickable geometry but no shipped constraint
/// slot accepts one, so offering an arc is a refusal rather than a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchEntity {
    Point(EntityId),
    Segment(EntityId),
}

impl SketchEntity {
    /// The slot kind this entity can fill.
    fn kind(self) -> SlotKind {
        match self {
            SketchEntity::Point(_) => SlotKind::Point,
            SketchEntity::Segment(_) => SlotKind::Segment,
        }
    }
}

/// What a constraint slot accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Point,
    Segment,
}

impl SlotKind {
    /// What the prompt asks for when this slot is the one waiting.
    pub fn wanted(self) -> &'static str {
        match self {
            SlotKind::Point => "a point",
            SlotKind::Segment => "a line",
        }
    }
}

/// A constraint the rail can arm.
///
/// A verb is not one-to-one with a [`ConstraintKind`]: [`HorizontalOrVertical`] asserts either of
/// two, chosen from the drawing. What the author asks for and what gets asserted are different
/// questions, and the rail asks the first.
///
/// [`HorizontalOrVertical`]: ConstraintVerb::HorizontalOrVertical
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintVerb {
    /// The picked segment lies along ONE of the plane's two in-plane axes — whichever it is
    /// already nearer (Fusion's arrangement: one tool, two constraints).
    HorizontalOrVertical,
    /// The picked point stays where it is.
    Fix,
    /// Two picked points occupy one place.
    Coincident,
    /// Two picked segments run the same way.
    Parallel,
    /// Two picked segments meet square.
    Perpendicular,
    /// Two picked segments have the same length.
    Equal,
    /// The picked point sits halfway along the picked segment.
    Midpoint,
    /// Two picked segments lie on one infinite line.
    Collinear,
}

impl ConstraintVerb {
    /// The entities this verb asks for, in the order it asks for them.
    ///
    /// The list IS the arity: a verb is complete when every slot is filled, so adding a
    /// two-entity relation later is an entry here rather than a new branch in the gesture.
    pub fn slots(self) -> &'static [SlotKind] {
        match self {
            ConstraintVerb::HorizontalOrVertical => &[SlotKind::Segment],
            ConstraintVerb::Fix => &[SlotKind::Point],
            ConstraintVerb::Coincident => &[SlotKind::Point, SlotKind::Point],
            ConstraintVerb::Parallel
            | ConstraintVerb::Perpendicular
            | ConstraintVerb::Equal
            | ConstraintVerb::Collinear => &[SlotKind::Segment, SlotKind::Segment],
            // The point first, because it is the thing being placed: the gesture reads "put THIS
            // in the middle of THAT", and a slot order that asked for the carrier first would
            // read as picking a line and then being asked what for.
            ConstraintVerb::Midpoint => &[SlotKind::Point, SlotKind::Segment],
        }
    }

    /// The rail tooltip. It names the verb and then what the FIRST pick is, because that is the
    /// only thing the author has to decide at the moment they read it.
    pub fn tooltip(self) -> &'static str {
        match self {
            ConstraintVerb::HorizontalOrVertical => "Horizontal / Vertical — then pick a line",
            ConstraintVerb::Fix => "Fix — then pick a point",
            ConstraintVerb::Coincident => "Coincident — then pick two points",
            ConstraintVerb::Parallel => "Parallel — then pick two lines",
            ConstraintVerb::Perpendicular => "Perpendicular — then pick two lines",
            ConstraintVerb::Equal => "Equal — then pick two lines",
            ConstraintVerb::Midpoint => "Midpoint — then pick a point and a line",
            ConstraintVerb::Collinear => "Collinear — then pick two lines",
        }
    }

    /// The glyph the rail cell carries.
    ///
    /// For a verb that asserts exactly one kind this is also the badge the drawing ends up with,
    /// so the mark pressed is the mark then seen. [`ConstraintVerb::HorizontalOrVertical`] is the
    /// deliberate exception: it asks one question and asserts one of two answers, and the badge
    /// reports the ANSWER — a level line is marked level, not marked "level or plumb".
    pub fn icon(self) -> Icon {
        match self {
            ConstraintVerb::HorizontalOrVertical => Icon::ConstraintHorizontalVertical,
            ConstraintVerb::Fix => Icon::ConstraintFix,
            ConstraintVerb::Coincident => Icon::ConstraintCoincident,
            ConstraintVerb::Parallel => Icon::ConstraintParallel,
            ConstraintVerb::Perpendicular => Icon::ConstraintPerpendicular,
            ConstraintVerb::Equal => Icon::ConstraintEqual,
            ConstraintVerb::Midpoint => Icon::ConstraintMidpoint,
            ConstraintVerb::Collinear => Icon::ConstraintCollinear,
        }
    }
}

/// The glyph that stands for a constraint already on the drawing (ADR 0035). The badge is the
/// only thing that makes an assertion visible after the solve has moved the geometry and the
/// evidence of the constraint is a line that merely *looks* level.
pub fn constraint_icon(kind: ConstraintKind) -> Icon {
    match kind {
        ConstraintKind::Horizontal { .. } => Icon::ConstraintHorizontal,
        ConstraintKind::Vertical { .. } => Icon::ConstraintVertical,
        ConstraintKind::Fix { .. } => Icon::ConstraintFix,
        ConstraintKind::Distance { .. } => Icon::SketchDimension,
        ConstraintKind::Coincident { .. } => Icon::ConstraintCoincident,
        ConstraintKind::Parallel { .. } => Icon::ConstraintParallel,
        ConstraintKind::Perpendicular { .. } => Icon::ConstraintPerpendicular,
        ConstraintKind::Equal { .. } => Icon::ConstraintEqual,
        ConstraintKind::Midpoint { .. } => Icon::ConstraintMidpoint,
        ConstraintKind::Collinear { .. } => Icon::ConstraintCollinear,
    }
}

/// A constraint mid-gesture: armed, with the entities picked for it so far.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmedConstraint {
    verb: ConstraintVerb,
    /// One entity per filled slot, in slot order.
    picked: Vec<SketchEntity>,
}

/// What [`ArmedConstraint::offer`] did with a pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offer {
    /// Taken, and slots remain — keep picking.
    Taken,
    /// Taken, and that was the last slot. The caller applies [`ArmedConstraint::kind`] and
    /// disarms.
    Complete,
    /// Not taken, and the reason. The gesture stays armed: a mis-click is not an abandonment.
    Refused(&'static str),
}

impl ArmedConstraint {
    /// Arm `verb` with nothing picked.
    pub fn new(verb: ConstraintVerb) -> Self {
        ArmedConstraint {
            verb,
            picked: Vec::new(),
        }
    }

    /// Rebuild a gesture from its parts — the door a dump comes back through (ADR 0024: a
    /// mid-pick repro must re-enter with the same question on screen).
    ///
    /// Picks past the verb's slot count are DROPPED rather than trusted: a dump written by a
    /// build whose slot list was longer would otherwise hand back a gesture that reports itself
    /// complete while `kind` cannot build anything. Truncating degrades it to "still asking",
    /// which every part of the gesture already handles.
    pub fn from_parts(verb: ConstraintVerb, picked: Vec<SketchEntity>) -> Self {
        let mut picked = picked;
        picked.truncate(verb.slots().len());
        ArmedConstraint { verb, picked }
    }

    pub fn verb(&self) -> ConstraintVerb {
        self.verb
    }

    /// The entities taken so far, so the caller can light them on the drawing.
    pub fn picked(&self) -> &[SketchEntity] {
        &self.picked
    }

    /// The slot still waiting, or `None` when every slot is filled.
    ///
    /// The shell hit-tests THROUGH this: a gesture waiting for a line looks for lines and ignores
    /// the vertices sitting on them, rather than resolving the click by the general
    /// most-specific-thing-wins rule and then refusing what it found. A question that already
    /// knows what kind of answer it wants should not be able to pick up the wrong kind.
    pub fn wants(&self) -> Option<SlotKind> {
        self.verb.slots().get(self.picked.len()).copied()
    }

    /// What the status line says while this gesture runs.
    pub fn prompt(&self) -> String {
        match self.wants() {
            Some(slot) => format!("pick {}", slot.wanted()),
            None => "done".to_string(),
        }
    }

    /// Offer `candidate` to the slot that is waiting.
    ///
    /// Four things can refuse it: the wrong kind of entity, an entity already picked for an
    /// earlier slot, geometry the sketch does not hold (a selection that went stale between the
    /// hit-test and here), and a point the drawing DERIVES rather than the author places. All four
    /// leave the gesture exactly as it was.
    ///
    /// The derived case is caught here and not only at `Sketch::add_constraint` because the
    /// document's refusal arrives after the last slot fills — the author would pick two points and
    /// then be told the first one was never eligible. Refusing at the pick keeps the gesture
    /// running and asks again.
    pub fn offer(&mut self, candidate: SketchEntity, sketch: &Sketch) -> Offer {
        let Some(slot) = self.wants() else {
            return Offer::Refused("already complete");
        };
        if candidate.kind() != slot {
            return Offer::Refused(match slot {
                SlotKind::Point => "that is not a point",
                SlotKind::Segment => "that is not a line",
            });
        }
        if self.picked.contains(&candidate) {
            return Offer::Refused("already picked");
        }
        if !holds(sketch, candidate) {
            return Offer::Refused("that geometry is gone");
        }
        if let SketchEntity::Point(id) = candidate {
            if sketch.is_derived_point(id) {
                return Offer::Refused("an arc's centre follows its ends — constrain those");
            }
        }
        self.picked.push(candidate);
        match self.wants() {
            Some(_) => Offer::Taken,
            None => Offer::Complete,
        }
    }

    /// The constraint the filled slots assert, or `None` while any slot is still empty.
    ///
    /// `Fix` reads the point's position out of the drawing here rather than leaving it implicit:
    /// it asserts immovability AT A PLACE, so the place is captured at the moment the author
    /// finishes asking for it. [`ConstraintVerb::HorizontalOrVertical`] reads the drawing for a
    /// different reason: to decide WHICH of its two constraints was meant.
    pub fn kind(&self, sketch: &Sketch) -> Option<ConstraintKind> {
        if self.wants().is_some() {
            return None;
        }
        // The two-slot verbs read their pair the same way, so the pair is pulled out once. A
        // verb's slot list is what guarantees these are the kinds asked for; `offer` enforces it.
        let point_pair = || match (self.picked.first()?, self.picked.get(1)?) {
            (SketchEntity::Point(first), SketchEntity::Point(second)) => Some((*first, *second)),
            _ => None,
        };
        let segment_pair = || match (self.picked.first()?, self.picked.get(1)?) {
            (SketchEntity::Segment(first), SketchEntity::Segment(second)) => {
                Some((*first, *second))
            }
            _ => None,
        };
        match self.verb {
            ConstraintVerb::HorizontalOrVertical => match self.picked.first()? {
                SketchEntity::Segment(segment) => nearer_axis(sketch, *segment),
                SketchEntity::Point(_) => None,
            },
            ConstraintVerb::Fix => match self.picked.first()? {
                SketchEntity::Point(point) => {
                    let at = sketch.points().iter().find(|p| p.id == *point)?.at;
                    Some(ConstraintKind::Fix { point: *point, at })
                }
                SketchEntity::Segment(_) => None,
            },
            ConstraintVerb::Coincident => {
                let (first, second) = point_pair()?;
                Some(ConstraintKind::Coincident { first, second })
            }
            ConstraintVerb::Parallel => {
                let (first, second) = segment_pair()?;
                Some(ConstraintKind::Parallel { first, second })
            }
            ConstraintVerb::Perpendicular => {
                let (first, second) = segment_pair()?;
                Some(ConstraintKind::Perpendicular { first, second })
            }
            ConstraintVerb::Equal => {
                let (first, second) = segment_pair()?;
                Some(ConstraintKind::Equal { first, second })
            }
            ConstraintVerb::Collinear => {
                let (first, second) = segment_pair()?;
                Some(ConstraintKind::Collinear { first, second })
            }
            ConstraintVerb::Midpoint => match (self.picked.first()?, self.picked.get(1)?) {
                (SketchEntity::Point(point), SketchEntity::Segment(segment)) => {
                    Some(ConstraintKind::Midpoint {
                        point: *point,
                        segment: *segment,
                    })
                }
                _ => None,
            },
        }
    }
}

/// Which axis constraint a segment is asking for: the one it is ALREADY nearer.
///
/// The whole point of folding the pair into one tool is that the author does not have to say
/// something the drawing already shows. A line 5° off level wants to be level; asserting plumb on
/// it would swing it 85° and read as the tool misfiring rather than as an instruction obeyed.
///
/// **The tie goes to Horizontal.** At exactly 45° neither answer is more obviously meant, so this
/// is a coin toss resolved once and stated, rather than left to whichever comparison happened to
/// be written. An author who wanted the other one deletes the badge and says so; that is one
/// click, and it is the case the rule is allowed to get wrong.
fn nearer_axis(sketch: &Sketch, segment: EntityId) -> Option<ConstraintKind> {
    let held = sketch.segments().iter().find(|held| held.id == segment)?;
    let at = |id: EntityId| Some(sketch.points().iter().find(|p| p.id == id)?.at.in_plane());
    let (from, to) = (at(held.from)?, at(held.to)?);
    let run = (to[0] - from[0]).abs();
    let rise = (to[1] - from[1]).abs();
    Some(if run >= rise {
        ConstraintKind::Horizontal { segment }
    } else {
        ConstraintKind::Vertical { segment }
    })
}

/// Whether the sketch still holds `entity` in a form a constraint can name — for a segment that
/// includes having two distinct ends, the degenerate case `Sketch::add_constraint` refuses.
fn holds(sketch: &Sketch, entity: SketchEntity) -> bool {
    match entity {
        SketchEntity::Point(id) => sketch.points().iter().any(|point| point.id == id),
        SketchEntity::Segment(id) => sketch
            .segments()
            .iter()
            .any(|segment| segment.id == id && segment.from != segment.to),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, SketchPoint};
    use parametric::units::AngleMeasurement;

    /// Two points joined by one segment.
    fn one_segment() -> (Sketch, EntityId, EntityId, EntityId) {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let to = sketch.add_free_point(SketchPoint::from_continuous(8.0, 3.0));
        let segment = sketch.connect(from, to).expect("two distinct points join");
        (sketch, from, to, segment)
    }

    /// One pick fills the only slot, so the gesture completes on it.
    #[test]
    fn a_one_slot_verb_completes_on_its_first_pick() {
        let (sketch, _, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(armed.prompt(), "pick a line");
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::Horizontal { segment })
        );
    }

    /// The gesture names the kind of thing it is waiting for BEFORE a click resolves, which is
    /// what lets the shell hit-test for that kind alone instead of resolving by the general
    /// vertex-beats-edge rule and refusing what it finds.
    #[test]
    fn a_running_gesture_says_what_kind_of_pick_it_is_waiting_for() {
        let (sketch, _, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(armed.wants(), Some(SlotKind::Segment));
        assert_eq!(SlotKind::Segment.wanted(), "a line");
        armed.offer(SketchEntity::Segment(segment), &sketch);
        assert_eq!(armed.wants(), None, "a filled gesture asks for nothing");

        let fix = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(fix.wants(), Some(SlotKind::Point));
    }

    /// The refusal that motivates the whole gesture: a pick of the wrong kind is turned away and
    /// the tool keeps waiting, so the author simply clicks again.
    #[test]
    fn a_pick_of_the_wrong_kind_is_refused_and_the_gesture_survives() {
        let (sketch, from, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(
            armed.offer(SketchEntity::Point(from), &sketch),
            Offer::Refused("that is not a line")
        );
        assert!(armed.picked().is_empty(), "nothing was taken");
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Complete,
            "still armed, and the next pick lands"
        );
    }

    #[test]
    fn fix_wants_a_point_and_captures_where_it_is() {
        let (sketch, _, to, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(armed.prompt(), "pick a point");
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Refused("that is not a point")
        );
        assert_eq!(
            armed.offer(SketchEntity::Point(to), &sketch),
            Offer::Complete
        );
        let Some(ConstraintKind::Fix { point, at }) = armed.kind(&sketch) else {
            panic!("Fix builds a Fix");
        };
        assert_eq!(point, to);
        assert_eq!(at.in_plane(), [8.0, 3.0]);
    }

    /// Geometry that went away between the hit-test and the offer is refused, not asserted about.
    #[test]
    fn a_stale_pick_is_refused() {
        let (mut sketch, from, _, segment) = one_segment();
        sketch.delete_point_cascade(from);
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Refused("that geometry is gone")
        );
        let mut fixing = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(
            fixing.offer(SketchEntity::Point(from), &sketch),
            Offer::Refused("that geometry is gone")
        );
    }

    /// An incomplete gesture asserts nothing — the caller has no constraint to commit until the
    /// last slot fills.
    #[test]
    fn an_unfilled_gesture_has_no_constraint() {
        let (sketch, _, _, _) = one_segment();
        let armed = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(armed.kind(&sketch), None);
    }

    /// An arc's centre is a point the DRAWING owns — it is re-derived after every edit that moves
    /// the arc — so offering it to a point slot is refused at the pick, while the gesture is still
    /// asking. Waiting for `Sketch::add_constraint` to refuse would make the author fill every
    /// slot before hearing that the first pick was never eligible (owner 2026-07-31).
    #[test]
    fn an_arcs_centre_is_refused_at_the_pick() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let tail = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let head = sketch.add_free_point(SketchPoint::from_continuous(20.0, 0.0));
        sketch
            .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
            .expect("a quarter turn");
        let centre = sketch.arcs()[0].center;
        let loose = sketch.add_free_point(SketchPoint::from_continuous(40.0, 17.0));

        let mut armed = ArmedConstraint::new(ConstraintVerb::Coincident);
        assert_eq!(
            armed.offer(SketchEntity::Point(centre), &sketch),
            Offer::Refused("an arc's centre follows its ends — constrain those")
        );
        assert!(armed.picked().is_empty(), "nothing was taken");
        assert_eq!(
            armed.offer(SketchEntity::Point(loose), &sketch),
            Offer::Taken,
            "still armed, and an ordinary point still lands"
        );
    }

    /// A verb that asserts exactly one kind leaves its own glyph on the drawing, so pressing a
    /// mark and then seeing it is the same mark.
    #[test]
    fn a_single_kind_verb_leaves_its_own_glyph() {
        let (sketch, _, to, _) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(
            armed.offer(SketchEntity::Point(to), &sketch),
            Offer::Complete
        );
        let kind = armed.kind(&sketch).expect("complete");
        assert_eq!(constraint_icon(kind), ConstraintVerb::Fix.icon());
    }

    /// A segment nearer level is asserted level, and one nearer plumb is asserted plumb: the
    /// author says "line this up with an axis" and the drawing supplies which.
    #[test]
    fn the_axis_tool_asserts_whichever_axis_the_line_is_nearer() {
        for (corner, expected) in [
            ([8.0, 3.0], "horizontal"),
            ([3.0, 8.0], "vertical"),
            ([-8.0, 3.0], "horizontal"),
            ([3.0, -8.0], "vertical"),
        ] {
            let mut sketch = Sketch::empty(PlaneAxis::Z);
            let from = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
            let to = sketch.add_free_point(SketchPoint::from_continuous(corner[0], corner[1]));
            let segment = sketch.connect(from, to).expect("two distinct points join");

            let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
            assert_eq!(
                armed.offer(SketchEntity::Segment(segment), &sketch),
                Offer::Complete
            );
            let got = match armed.kind(&sketch) {
                Some(ConstraintKind::Horizontal { .. }) => "horizontal",
                Some(ConstraintKind::Vertical { .. }) => "vertical",
                other => panic!("the axis tool asserts an axis, got {other:?}"),
            };
            assert_eq!(got, expected, "for a segment reaching {corner:?}");
        }
    }

    /// The tie is resolved once and stated: at exactly 45° neither answer is more obviously meant,
    /// and Horizontal wins. This is the case the rule is allowed to get wrong — it is one badge
    /// deletion away from the other answer.
    #[test]
    fn a_line_at_exactly_forty_five_degrees_goes_horizontal() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let to = sketch.add_free_point(SketchPoint::from_continuous(6.0, 6.0));
        let segment = sketch.connect(from, to).expect("two distinct points join");

        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::Horizontal { segment })
        );
    }

    /// The badge reports the ANSWER, not the question: a line asserted plumb carries the plain
    /// Vertical mark, never the two-axis glyph of the cell that was pressed.
    #[test]
    fn the_axis_tool_leaves_the_mark_of_what_it_decided() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let to = sketch.add_free_point(SketchPoint::from_continuous(1.0, 9.0));
        let segment = sketch.connect(from, to).expect("two distinct points join");

        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(
            armed.offer(SketchEntity::Segment(segment), &sketch),
            Offer::Complete
        );
        let kind = armed.kind(&sketch).expect("complete");
        assert_eq!(constraint_icon(kind), Icon::ConstraintVertical);
        assert_ne!(
            constraint_icon(kind),
            ConstraintVerb::HorizontalOrVertical.icon()
        );
    }
}
