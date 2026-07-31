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
//! Only the three kinds whose residuals ship are here. The other eleven glyphs on the constraint
//! shelf are drawn and named but have no residual behind them
//! (`crates/document/src/sketch/constraint.rs`), and an armable verb that asserts nothing is worse
//! than a cell that is not there.

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
    /// What the status line asks for when this slot is the one waiting.
    fn wanted(self) -> &'static str {
        match self {
            SlotKind::Point => "a point",
            SlotKind::Segment => "a line",
        }
    }
}

/// A constraint the rail can arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintVerb {
    /// The picked segment lies along the plane's first in-plane axis.
    Horizontal,
    /// The picked segment lies along the plane's second in-plane axis.
    Vertical,
    /// The picked point stays where it is.
    Fix,
}

impl ConstraintVerb {
    /// The entities this verb asks for, in the order it asks for them.
    ///
    /// The list IS the arity: a verb is complete when every slot is filled, so adding a
    /// two-entity relation later is an entry here rather than a new branch in the gesture.
    pub fn slots(self) -> &'static [SlotKind] {
        match self {
            ConstraintVerb::Horizontal | ConstraintVerb::Vertical => &[SlotKind::Segment],
            ConstraintVerb::Fix => &[SlotKind::Point],
        }
    }

    /// The rail tooltip.
    pub fn tooltip(self) -> &'static str {
        match self {
            ConstraintVerb::Horizontal => "Horizontal — then pick a line",
            ConstraintVerb::Vertical => "Vertical — then pick a line",
            ConstraintVerb::Fix => "Fix — then pick a point",
        }
    }

    /// The glyph the rail cell and the badge on the drawing both carry, so the mark the author
    /// pressed is the mark they then see standing beside the geometry.
    pub fn icon(self) -> Icon {
        match self {
            ConstraintVerb::Horizontal => Icon::ConstraintHorizontal,
            ConstraintVerb::Vertical => Icon::ConstraintVertical,
            ConstraintVerb::Fix => Icon::ConstraintFix,
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
    fn next_slot(&self) -> Option<SlotKind> {
        self.verb.slots().get(self.picked.len()).copied()
    }

    /// What the status line says while this gesture runs.
    pub fn prompt(&self) -> String {
        match self.next_slot() {
            Some(slot) => format!("pick {}", slot.wanted()),
            None => "done".to_string(),
        }
    }

    /// Offer `candidate` to the slot that is waiting.
    ///
    /// Three things can refuse it: the wrong kind of entity, an entity already picked for an
    /// earlier slot, and geometry the sketch does not hold (a selection that went stale between
    /// the hit-test and here). All three leave the gesture exactly as it was.
    pub fn offer(&mut self, candidate: SketchEntity, sketch: &Sketch) -> Offer {
        let Some(slot) = self.next_slot() else {
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
        self.picked.push(candidate);
        match self.next_slot() {
            Some(_) => Offer::Taken,
            None => Offer::Complete,
        }
    }

    /// The constraint the filled slots assert, or `None` while any slot is still empty.
    ///
    /// `Fix` reads the point's position out of the drawing here rather than leaving it implicit:
    /// it asserts immovability AT A PLACE, so the place is captured at the moment the author
    /// finishes asking for it.
    pub fn kind(&self, sketch: &Sketch) -> Option<ConstraintKind> {
        if self.next_slot().is_some() {
            return None;
        }
        match (self.verb, self.picked.first()?) {
            (ConstraintVerb::Horizontal, SketchEntity::Segment(segment)) => {
                Some(ConstraintKind::Horizontal { segment: *segment })
            }
            (ConstraintVerb::Vertical, SketchEntity::Segment(segment)) => {
                Some(ConstraintKind::Vertical { segment: *segment })
            }
            (ConstraintVerb::Fix, SketchEntity::Point(point)) => {
                let at = sketch.points().iter().find(|p| p.id == *point)?.at;
                Some(ConstraintKind::Fix { point: *point, at })
            }
            // `offer` type-checks every slot, so a filled gesture whose entity kinds do not match
            // its verb cannot be built. Refusing here rather than unwrapping keeps that a
            // no-constraint instead of a panic if a future verb's slots and arms disagree.
            _ => None,
        }
    }
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
        let mut armed = ArmedConstraint::new(ConstraintVerb::Horizontal);
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

    /// The refusal that motivates the whole gesture: a pick of the wrong kind is turned away and
    /// the tool keeps waiting, so the author simply clicks again.
    #[test]
    fn a_pick_of_the_wrong_kind_is_refused_and_the_gesture_survives() {
        let (sketch, from, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::Horizontal);
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
        let mut armed = ArmedConstraint::new(ConstraintVerb::Horizontal);
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

    /// Every verb's glyph is the one its shelf entry draws, so pressing a mark and then seeing it
    /// on the drawing is the same mark.
    #[test]
    fn the_badge_glyph_is_the_cell_glyph() {
        let (sketch, _, to, segment) = one_segment();
        for (verb, entity) in [
            (ConstraintVerb::Horizontal, SketchEntity::Segment(segment)),
            (ConstraintVerb::Vertical, SketchEntity::Segment(segment)),
            (ConstraintVerb::Fix, SketchEntity::Point(to)),
        ] {
            let mut armed = ArmedConstraint::new(verb);
            assert_eq!(armed.offer(entity, &sketch), Offer::Complete);
            let kind = armed.kind(&sketch).expect("complete");
            assert_eq!(constraint_icon(kind), verb.icon());
        }
    }
}
