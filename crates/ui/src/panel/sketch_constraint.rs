//! Arming a constraint, and collecting the entities it needs.
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
//! either of two kinds, picked from the drawing: the author is saying "line this up with an
//! axis", and which axis is already visible in what they drew. The badge then reports the answer
//! rather than the question.
//!
//! Only the kinds carrying a residual in `document::sketch::constraint` are here. A verb whose
//! glyph is drawn but whose residual is absent stays off the rail — an armable verb that asserts
//! nothing is worse than a cell that is not there.

use document::sketch::{ConstraintKind, EntityId, Sketch, SketchCurve};

use crate::icons::Icon;

/// A sketch entity a constraint can name: a point, or **the document's own curve identity**.
///
/// The curve arm carries [`SketchCurve`] whole rather than re-spelling its variants. A second
/// enumeration of the curve kinds would be a vocabulary the drawing does not have — every hit,
/// every persisted pick and every selection would need a hand-written translation both ways, and
/// a curve kind added to the document would compile clean here while being silently unnameable.
/// What a relation can actually be ABOUT is a separate question, and
/// [`SketchCurve::carries_relation_geometry`] is where the drawing answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchEntity {
    Point(EntityId),
    Curve(SketchCurve),
}

/// What a constraint slot accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Point,
    Segment,
    Curve,
    CircularCurve,
    /// Either — the verb asserts a different relation depending on which arrives. Coincident is
    /// the only one: a point on a point and a point on a curve are the same claim to the author.
    PointOrCurve,
}

/// The exact kind the current gesture asks the shell to resolve before comparing distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickRequirement {
    Point,
    Segment,
    Curve,
    CircularCurve,
    /// A curve of the same KIND as one already picked. Symmetry is the only verb that narrows this
    /// way, and it carries the curve rather than a kind tag so that adding a curve kind to the
    /// document needs nothing here.
    MatchingCurve(SketchCurve),
    PointOrCurve,
}

impl PickRequirement {
    /// Whether `entity` fills this slot.
    ///
    /// Every curve arm asks [`SketchCurve::carries_relation_geometry`] first. An aggregate is
    /// nameable — it is a real curve with a real identity, and the hit-test resolves it — but no
    /// relation can read one, so taking the pick would arm a gesture that completes and then
    /// cannot be applied. The refusal belongs at the click, where the tool is still waiting.
    fn accepts(self, entity: SketchEntity) -> bool {
        let curve = match entity {
            SketchEntity::Point(_) => return matches!(self, Self::Point | Self::PointOrCurve),
            SketchEntity::Curve(curve) => curve,
        };
        match self {
            Self::Point => false,
            Self::Segment => matches!(curve, SketchCurve::Segment(_)),
            Self::Curve | Self::PointOrCurve => curve.carries_relation_geometry(),
            Self::CircularCurve => curve.is_circular(),
            Self::MatchingCurve(like) => {
                like.same_kind_as(curve) && curve.carries_relation_geometry()
            }
        }
    }

    pub fn wanted(self) -> &'static str {
        match self {
            Self::Point => "a point",
            Self::Segment => "a line",
            Self::Curve => "a curve",
            Self::CircularCurve => "an arc or circle",
            Self::MatchingCurve(like) => another_of_the_same_kind(like),
            Self::PointOrCurve => "a point or a curve",
        }
    }

    /// What the status line says when a click found nothing of the kind this slot wants.
    ///
    /// Written out rather than assembled from [`wanted`](Self::wanted), because the refusal is
    /// held as one `&'static str`. It lives HERE, beside the vocabulary it extends, so the shell
    /// that shows it does not carry a second table of what each slot is called.
    pub fn nothing_under_the_cursor(self) -> &'static str {
        match self {
            Self::Point => "nothing under the cursor — pick a point",
            Self::Segment => "nothing under the cursor — pick a line",
            Self::Curve => "nothing under the cursor — pick a curve",
            Self::CircularCurve => "nothing under the cursor — pick an arc or circle",
            Self::MatchingCurve(SketchCurve::Segment(_)) => {
                "nothing under the cursor — pick another line"
            }
            Self::MatchingCurve(SketchCurve::Arc(_)) => {
                "nothing under the cursor — pick another arc"
            }
            Self::MatchingCurve(SketchCurve::Circle(_)) => {
                "nothing under the cursor — pick another circle"
            }
            Self::MatchingCurve(_) => "nothing under the cursor — pick a curve of the same kind",
            Self::PointOrCurve => "nothing under the cursor — pick a point or a curve",
        }
    }
}

/// How the prompt asks for a second curve like the first one. It names the KIND, because that is
/// the whole of what the second pick has to match.
fn another_of_the_same_kind(like: SketchCurve) -> &'static str {
    match like {
        SketchCurve::Segment(_) => "another line",
        SketchCurve::Arc(_) => "another arc",
        SketchCurve::Circle(_) => "another circle",
        SketchCurve::Bezier(_) => "another curve piece",
        SketchCurve::Ellipse(_) => "another ellipse",
        SketchCurve::Conic(_) => "another conic",
        SketchCurve::Spline(_) => "another spline",
    }
}

impl From<SlotKind> for PickRequirement {
    fn from(slot: SlotKind) -> Self {
        match slot {
            SlotKind::Point => Self::Point,
            SlotKind::Segment => Self::Segment,
            SlotKind::Curve => Self::Curve,
            SlotKind::CircularCurve => Self::CircularCurve,
            SlotKind::PointOrCurve => Self::PointOrCurve,
        }
    }
}

impl SlotKind {
    /// What the prompt asks for when this slot is the one waiting.
    pub fn wanted(self) -> &'static str {
        match self {
            SlotKind::Point => "a point",
            SlotKind::Segment => "a line",
            SlotKind::Curve => "a curve",
            SlotKind::CircularCurve => "an arc or circle",
            SlotKind::PointOrCurve => "a point or a curve",
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
    /// already nearer — one tool, two constraints.
    HorizontalOrVertical,
    /// The picked point stays where it is.
    Fix,
    /// Both coordinates of the picked point stay on the whole-voxel lattice.
    Quantize,
    /// A picked point occupies one place with a second point, or STANDS ON a picked curve.
    ///
    /// One verb, two relations. Fusion spells both as coincident and so does the badge, because
    /// the author is making the same claim either way — put this here. The kinds stay separate
    /// underneath because they pin a different number of coordinates.
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
    /// Two finite curves touch; the click loci choose the durable branch.
    Tangent,
    /// Two arcs or circles share one center.
    Concentric,
    /// Two same-kind curves mirror across a third picked segment axis.
    Symmetry,
    /// A fit-point spline's END runs smoothly out of the picked curve: same direction there, and
    /// the same curvature.
    Curvature,
}

impl ConstraintVerb {
    /// The entities this verb asks for, in the order it asks for them.
    ///
    /// The list IS the arity: a verb is complete when every slot is filled, so adding a
    /// two-entity relation later is an entry here rather than a new branch in the gesture.
    pub fn slots(self) -> &'static [SlotKind] {
        match self {
            ConstraintVerb::HorizontalOrVertical => &[SlotKind::Segment],
            ConstraintVerb::Fix | ConstraintVerb::Quantize => &[SlotKind::Point],
            // The point first: the gesture reads "put THIS on THAT", and the second slot is what
            // decides which of the two relations is asserted.
            ConstraintVerb::Coincident => &[SlotKind::Point, SlotKind::PointOrCurve],
            ConstraintVerb::Parallel
            | ConstraintVerb::Perpendicular
            | ConstraintVerb::Equal
            | ConstraintVerb::Collinear => &[SlotKind::Segment, SlotKind::Segment],
            // The point first, because it is the thing being placed: the gesture reads "put THIS
            // in the middle of THAT", and a slot order that asked for the carrier first would
            // read as picking a line and then being asked what for.
            ConstraintVerb::Midpoint => &[SlotKind::Point, SlotKind::Segment],
            ConstraintVerb::Tangent => &[SlotKind::Curve, SlotKind::Curve],
            ConstraintVerb::Concentric => &[SlotKind::CircularCurve, SlotKind::CircularCurve],
            ConstraintVerb::Symmetry => &[SlotKind::Curve, SlotKind::Curve, SlotKind::Segment],
            // The spline's end first, for Midpoint's reason: the gesture reads "make THIS end run
            // smoothly out of THAT", and asking for the curve first would read as picking a curve
            // and only then being told what for.
            ConstraintVerb::Curvature => &[SlotKind::Point, SlotKind::Curve],
        }
    }

    /// The rail tooltip. It names the verb and then what the FIRST pick is, because that is the
    /// only thing the author has to decide at the moment they read it.
    pub fn tooltip(self) -> &'static str {
        match self {
            ConstraintVerb::HorizontalOrVertical => "Horizontal / Vertical — then pick a line",
            ConstraintVerb::Fix => "Fix — then pick a point",
            ConstraintVerb::Quantize => "Quantize — then pick a point to keep on the voxel lattice",
            ConstraintVerb::Coincident => "Coincident — then pick a point and what to put it on",
            ConstraintVerb::Parallel => "Parallel — then pick two lines",
            ConstraintVerb::Perpendicular => "Perpendicular — then pick two lines",
            ConstraintVerb::Equal => "Equal — then pick two lines",
            ConstraintVerb::Midpoint => "Midpoint — then pick a point and a line",
            ConstraintVerb::Collinear => "Collinear — then pick two lines",
            ConstraintVerb::Tangent => "Tangent — then pick two curves",
            ConstraintVerb::Concentric => "Concentric — then pick two arcs or circles",
            ConstraintVerb::Symmetry => "Symmetry — then pick two matching curves and an axis",
            ConstraintVerb::Curvature => {
                "Curvature — then pick a spline's end and the curve it runs out of"
            }
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
            ConstraintVerb::Quantize => Icon::ConstraintQuantize,
            ConstraintVerb::Coincident => Icon::ConstraintCoincident,
            ConstraintVerb::Parallel => Icon::ConstraintParallel,
            ConstraintVerb::Perpendicular => Icon::ConstraintPerpendicular,
            ConstraintVerb::Equal => Icon::ConstraintEqual,
            ConstraintVerb::Midpoint => Icon::ConstraintMidpoint,
            ConstraintVerb::Collinear => Icon::ConstraintCollinear,
            ConstraintVerb::Tangent => Icon::ConstraintTangent,
            ConstraintVerb::Concentric => Icon::ConstraintConcentric,
            ConstraintVerb::Symmetry => Icon::ConstraintSymmetry,
            ConstraintVerb::Curvature => Icon::ConstraintCurvature,
        }
    }
}

/// The glyph that stands for a constraint already on the drawing. The badge is the
/// only thing that makes an assertion visible after the solve has moved the geometry and the
/// evidence of the constraint is a line that merely *looks* level.
pub fn constraint_icon(kind: ConstraintKind) -> Icon {
    match kind {
        ConstraintKind::Horizontal { .. } => Icon::ConstraintHorizontal,
        ConstraintKind::Vertical { .. } => Icon::ConstraintVertical,
        ConstraintKind::Fix { .. } => Icon::ConstraintFix,
        ConstraintKind::Quantize { .. } => Icon::ConstraintQuantize,
        ConstraintKind::Dimension(_) => Icon::SketchDimension,
        // Point-on-curve wears the coincident mark deliberately: it is the same claim the author
        // makes when they put a point ON something, and Fusion spells both with one glyph. The
        // kinds stay separate underneath because they pin a different number of coordinates.
        ConstraintKind::Coincident { .. } | ConstraintKind::PointOnCurve { .. } => {
            Icon::ConstraintCoincident
        }
        ConstraintKind::Parallel { .. } => Icon::ConstraintParallel,
        ConstraintKind::Perpendicular { .. } => Icon::ConstraintPerpendicular,
        ConstraintKind::Equal { .. } => Icon::ConstraintEqual,
        ConstraintKind::Midpoint { .. } => Icon::ConstraintMidpoint,
        ConstraintKind::Collinear { .. } => Icon::ConstraintCollinear,
        ConstraintKind::Tangent { .. } => Icon::ConstraintTangent,
        ConstraintKind::Concentric { .. } => Icon::ConstraintConcentric,
        ConstraintKind::Symmetry { .. } => Icon::ConstraintSymmetry,
        ConstraintKind::Curvature { .. } => Icon::ConstraintCurvature,
    }
}

/// A constraint mid-gesture: armed, with the entities picked for it so far.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmedConstraint {
    verb: ConstraintVerb,
    /// One entity per filled slot, in slot order.
    picked: Vec<SketchEntity>,
    /// Unsnapped profile click locations, only meaningful for Tangent and never persisted.
    loci: Vec<[f64; 2]>,
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
            loci: Vec::new(),
        }
    }

    /// Rebuild a gesture from its parts — the door a dump comes back through, so a mid-pick
    /// repro re-enters with the same question on screen.
    ///
    /// A full or overfull restored list restarts empty: completed gestures are dispatched and
    /// disarmed rather than persisted, so such a list is malformed session state.
    pub fn from_parts(verb: ConstraintVerb, picked: Vec<SketchEntity>) -> Self {
        // Tangent depends on unsnapped click evidence; restored artifacts intentionally restart it.
        if verb == ConstraintVerb::Tangent {
            return Self::new(verb);
        }
        if picked.len() >= verb.slots().len() {
            return Self::new(verb);
        }
        let mut restored = Self::new(verb);
        for candidate in picked.into_iter().take(verb.slots().len()) {
            if restored.picked.contains(&candidate)
                || restored
                    .wants()
                    .is_none_or(|wanted| !wanted.accepts(candidate))
            {
                return Self::new(verb);
            }
            restored.picked.push(candidate);
        }
        restored
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
    pub fn wants(&self) -> Option<PickRequirement> {
        if self.verb == ConstraintVerb::Symmetry && self.picked.len() == 1 {
            return match self.picked[0] {
                SketchEntity::Curve(curve) => Some(PickRequirement::MatchingCurve(curve)),
                SketchEntity::Point(_) => None,
            };
        }
        self.verb
            .slots()
            .get(self.picked.len())
            .copied()
            .map(Into::into)
    }

    /// Restart a restored gesture whose held entities are dead or no longer fit its dynamic slots.
    pub fn restart_if_invalid(&mut self, sketch: &Sketch) -> bool {
        let restored = Self::from_parts(self.verb, self.picked.clone());
        let invalid = restored.picked.len() != self.picked.len()
            || restored.picked.iter().any(|entity| !holds(sketch, *entity));
        if invalid {
            *self = Self::new(self.verb);
        }
        invalid
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
    /// Three things can refuse it: the wrong kind of entity, an entity already picked for an
    /// earlier slot, and geometry the sketch does not hold (a selection that went stale between
    /// the hit-test and here). All three leave the gesture exactly as it was.
    ///
    /// An arc's CENTER is not among them. It is a point the drawing derives rather than one the
    /// author places, but the residual system reads it as the function of the arc's ends that it
    /// is, so a constraint on it moves the arc and holds like any other.
    pub fn offer(&mut self, candidate: SketchEntity, sketch: &Sketch) -> Offer {
        self.offer_at(candidate, [0.0, 0.0], sketch)
    }

    /// Offer a pick with its continuous profile locus. Non-Tangent verbs ignore the locus.
    pub fn offer_at(&mut self, candidate: SketchEntity, locus: [f64; 2], sketch: &Sketch) -> Offer {
        let Some(slot) = self.wants() else {
            return Offer::Refused("already complete");
        };
        if !slot.accepts(candidate) {
            return Offer::Refused(match slot {
                PickRequirement::Point => "that is not a point",
                PickRequirement::Segment => "that is not a line",
                // An aggregate reaches this arm too, and the message is right for it: no relation
                // reads a spline or an ellipse as a whole shape, so to this slot it is not one.
                PickRequirement::Curve => "that is not a curve a constraint can hold",
                PickRequirement::CircularCurve => "pick an arc or circle — lines have no center",
                PickRequirement::MatchingCurve(like) => match like {
                    SketchCurve::Segment(_) => "pick another line",
                    SketchCurve::Arc(_) => "pick another arc",
                    SketchCurve::Circle(_) => "pick another circle",
                    SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => "pick a curve of the same kind",
                },
                // A point or any relation-bearing curve fills this, so the message is reached only
                // by an aggregate — which is a curve the author drew and a claim they cannot make.
                PickRequirement::PointOrCurve => "that is not a point or a curve",
            });
        }
        if self.picked.contains(&candidate) {
            return Offer::Refused("already picked");
        }
        if !holds(sketch, candidate) {
            return Offer::Refused("that geometry is gone");
        }
        if self.verb == ConstraintVerb::Tangent
            && matches!(
                (&self.picked[..], candidate),
                (
                    [SketchEntity::Curve(SketchCurve::Segment(_))],
                    SketchEntity::Curve(SketchCurve::Segment(_))
                )
            )
        {
            return Offer::Refused("two lines use Parallel — pick a curve");
        }
        self.picked.push(candidate);
        if self.verb == ConstraintVerb::Tangent {
            self.loci.push(locus);
        }
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
        let segment_pair = || match (self.picked.first()?, self.picked.get(1)?) {
            (
                SketchEntity::Curve(SketchCurve::Segment(first)),
                SketchEntity::Curve(SketchCurve::Segment(second)),
            ) => Some((*first, *second)),
            _ => None,
        };
        let circular_pair = || match (self.picked.first()?, self.picked.get(1)?) {
            (SketchEntity::Curve(first), SketchEntity::Curve(second))
                if first.is_circular() && second.is_circular() =>
            {
                Some((*first, *second))
            }
            _ => None,
        };
        match self.verb {
            ConstraintVerb::HorizontalOrVertical => match self.picked.first()? {
                SketchEntity::Curve(SketchCurve::Segment(segment)) => nearer_axis(sketch, *segment),
                SketchEntity::Point(_) | SketchEntity::Curve(_) => None,
            },
            ConstraintVerb::Fix => match self.picked.first()? {
                SketchEntity::Point(point) => {
                    let at = sketch.points().iter().find(|p| p.id == *point)?.at;
                    Some(ConstraintKind::Fix { point: *point, at })
                }
                SketchEntity::Curve(_) => None,
            },
            ConstraintVerb::Quantize => match self.picked.first()? {
                SketchEntity::Point(point) => Some(ConstraintKind::Quantize {
                    point: *point,
                    pitch: document::sketch::SketchLength::retained_voxels(1),
                    phase: document::sketch::SketchLength::retained_voxels(0),
                }),
                SketchEntity::Curve(_) => None,
            },
            ConstraintVerb::Coincident => match (self.picked.first()?, self.picked.get(1)?) {
                (SketchEntity::Point(first), SketchEntity::Point(second)) => {
                    Some(ConstraintKind::Coincident {
                        first: *first,
                        second: *second,
                    })
                }
                (SketchEntity::Point(point), SketchEntity::Curve(curve)) => {
                    Some(ConstraintKind::PointOnCurve {
                        point: *point,
                        curve: *curve,
                    })
                }
                _ => None,
            },
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
                (
                    SketchEntity::Point(point),
                    SketchEntity::Curve(SketchCurve::Segment(segment)),
                ) => Some(ConstraintKind::Midpoint {
                    point: *point,
                    segment: *segment,
                }),
                _ => None,
            },
            ConstraintVerb::Curvature => match (self.picked.first()?, self.picked.get(1)?) {
                (SketchEntity::Point(joint), SketchEntity::Curve(against)) => {
                    Some(ConstraintKind::Curvature {
                        joint: *joint,
                        against: *against,
                    })
                }
                _ => None,
            },
            ConstraintVerb::Tangent | ConstraintVerb::Symmetry => None,
            ConstraintVerb::Concentric => {
                let (first, second) = circular_pair()?;
                Some(ConstraintKind::concentric(first, second))
            }
        }
    }

    /// Complete a Tangent using the session-only loci and explicit scalar evaluation context.
    ///
    /// # Errors
    ///
    /// Returns a user-facing refusal when the gesture is incomplete or its loci cannot choose a
    /// valid Tangent branch from the current curve geometry.
    pub fn kind_at_context(
        &self,
        sketch: &Sketch,
        context: parametric::EvaluationContext,
    ) -> Result<ConstraintKind, &'static str> {
        if !matches!(
            self.verb,
            ConstraintVerb::Tangent | ConstraintVerb::Symmetry
        ) {
            return self.kind(sketch).ok_or("constraint is incomplete");
        }
        if self.verb == ConstraintVerb::Symmetry {
            let (
                Some(SketchEntity::Curve(first)),
                Some(SketchEntity::Curve(second)),
                Some(SketchEntity::Curve(SketchCurve::Segment(axis))),
            ) = (
                self.picked.first().copied(),
                self.picked.get(1).copied(),
                self.picked.get(2).copied(),
            )
            else {
                return Err("pick two matching curves and an axis");
            };
            let branch = sketch
                .choose_symmetry_branch(first, second, axis, context)
                .map_err(|_| "cannot mirror those curves about that axis")?;
            return Ok(ConstraintKind::symmetry(first, second, axis, branch));
        }
        let (
            Some(SketchEntity::Curve(first)),
            Some(SketchEntity::Curve(second)),
            Some(first_locus),
            Some(second_locus),
        ) = (
            self.picked.first().copied(),
            self.picked.get(1).copied(),
            self.loci.first(),
            self.loci.get(1),
        )
        else {
            return Err("pick two curves");
        };
        let branch = sketch
            .choose_tangent_branch(first, *first_locus, second, *second_locus, context)
            .map_err(|_| "cannot choose a tangent branch here")?;
        Ok(ConstraintKind::tangent(first, second, branch))
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
        SketchEntity::Curve(SketchCurve::Segment(id)) => sketch
            .segments()
            .iter()
            .any(|segment| segment.id == id && segment.from != segment.to),
        SketchEntity::Curve(curve) => sketch.holds_curve(curve),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::duration_subsec,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::match_same_arms,
        clippy::panic,
        clippy::semicolon_if_nothing_returned,
        clippy::unwrap_used,
        clippy::while_float
    )]

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

    /// Coincident's second pick may be a CURVE, and then it asserts point-on-curve.
    ///
    /// One verb, two relations, because to the author it is one claim: put this here. Fusion
    /// spells it the same way, and the badge already did — this is the gesture catching up.
    #[test]
    fn coincident_takes_a_curve_for_its_second_pick_and_stands_the_point_on_it() {
        let (mut sketch, from, _, segment) = one_segment();
        let loose = sketch.add_free_point(SketchPoint::from_continuous(40.0, 17.0));

        let mut armed = ArmedConstraint::new(ConstraintVerb::Coincident);
        assert_eq!(armed.wants(), Some(PickRequirement::Point));
        assert_eq!(
            armed.offer(SketchEntity::Point(loose), &sketch),
            Offer::Taken
        );
        // The second slot admits either kind, and says so.
        assert_eq!(armed.wants(), Some(PickRequirement::PointOrCurve));
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::PointOnCurve {
                point: loose,
                curve: SketchCurve::Segment(segment),
            })
        );

        // A point in that same slot still means what it always meant.
        let mut pair = ArmedConstraint::new(ConstraintVerb::Coincident);
        pair.offer(SketchEntity::Point(loose), &sketch);
        pair.offer(SketchEntity::Point(from), &sketch);
        assert_eq!(
            pair.kind(&sketch),
            Some(ConstraintKind::Coincident {
                first: loose,
                second: from,
            })
        );
    }

    /// The FIRST pick is still a point only: "put this line on that point" is not the gesture.
    #[test]
    fn coincident_still_refuses_a_curve_for_its_first_pick() {
        let (sketch, _, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::Coincident);
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
            Offer::Refused("that is not a point")
        );
        assert_eq!(armed.wants(), Some(PickRequirement::Point));
    }

    /// One pick fills the only slot, so the gesture completes on it.
    #[test]
    fn a_one_slot_verb_completes_on_its_first_pick() {
        let (sketch, _, _, segment) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(armed.prompt(), "pick a line");
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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
        assert_eq!(armed.wants(), Some(PickRequirement::Segment));
        assert_eq!(SlotKind::Segment.wanted(), "a line");
        armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch);
        assert_eq!(armed.wants(), None, "a filled gesture asks for nothing");

        let fix = ArmedConstraint::new(ConstraintVerb::Fix);
        assert_eq!(fix.wants(), Some(PickRequirement::Point));
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
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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

    #[test]
    fn quantize_wants_a_point_and_authors_the_voxel_lattice() {
        let (sketch, from, _, _) = one_segment();
        let mut armed = ArmedConstraint::new(ConstraintVerb::Quantize);
        assert_eq!(
            armed.offer(SketchEntity::Point(from), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::Quantize {
                point: from,
                pitch: document::sketch::SketchLength::retained_voxels(1),
                phase: document::sketch::SketchLength::retained_voxels(0),
            })
        );
        assert_eq!(
            constraint_icon(armed.kind(&sketch).unwrap()),
            Icon::ConstraintQuantize
        );
    }

    /// Geometry that went away between the hit-test and the offer is refused, not asserted about.
    #[test]
    fn a_stale_pick_is_refused() {
        let (mut sketch, from, _, segment) = one_segment();
        sketch.delete_point_cascade(from);
        let mut armed = ArmedConstraint::new(ConstraintVerb::HorizontalOrVertical);
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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

    /// An arc's center is a point the DRAWING owns — it is re-derived from the arc's ends and
    /// its sweep — but it fills a point slot like any other, because a constraint naming it is
    /// met by moving the arc.
    #[test]
    fn an_arcs_center_fills_a_point_slot() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let tail = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let head = sketch.add_free_point(SketchPoint::from_continuous(20.0, 0.0));
        sketch
            .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
            .expect("a quarter turn");
        let center = sketch.arcs()[0].center;
        let loose = sketch.add_free_point(SketchPoint::from_continuous(40.0, 17.0));
        assert!(sketch.is_arc_center(center));

        let mut armed = ArmedConstraint::new(ConstraintVerb::Coincident);
        assert_eq!(
            armed.offer(SketchEntity::Point(center), &sketch),
            Offer::Taken
        );
        assert_eq!(
            armed.offer(SketchEntity::Point(loose), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::Coincident {
                first: center,
                second: loose
            })
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
                armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::Horizontal { segment })
        );
    }

    #[test]
    fn tangent_curve_slot_accepts_every_curve_and_refuses_two_lines() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let a = sketch.add_free_point(SketchPoint::new(0, 0));
        let b = sketch.add_free_point(SketchPoint::new(10, 0));
        let line = sketch.connect(a, b).expect("line");
        let c = sketch.add_free_point(SketchPoint::new(0, 8));
        let d = sketch.add_free_point(SketchPoint::new(10, 8));
        let other_line = sketch.connect(c, d).expect("other line");
        let arc = sketch
            .connect_arc(a, b, AngleMeasurement::from_degrees(90))
            .expect("arc");
        let circle = sketch
            .add_circle(
                SketchPoint::new(5, 4),
                document::sketch::SketchLength::new(4),
            )
            .expect("circle");
        let mut tangent = ArmedConstraint::new(ConstraintVerb::Tangent);
        assert_eq!(
            tangent.offer_at(
                SketchEntity::Curve(SketchCurve::Segment(line)),
                [0.0, 0.0],
                &sketch
            ),
            Offer::Taken
        );
        assert_eq!(
            tangent.offer_at(
                SketchEntity::Curve(SketchCurve::Arc(arc)),
                [0.0, 0.0],
                &sketch
            ),
            Offer::Complete
        );
        let mut lines = ArmedConstraint::new(ConstraintVerb::Tangent);
        assert_eq!(
            lines.offer_at(
                SketchEntity::Curve(SketchCurve::Segment(line)),
                [0.0, 0.0],
                &sketch
            ),
            Offer::Taken
        );
        assert_eq!(
            lines.offer_at(
                SketchEntity::Curve(SketchCurve::Segment(other_line)),
                [0.0, 0.0],
                &sketch
            ),
            Offer::Refused("two lines use Parallel — pick a curve")
        );
        let mut circle_pair = ArmedConstraint::new(ConstraintVerb::Tangent);
        assert_eq!(
            circle_pair.offer_at(
                SketchEntity::Curve(SketchCurve::Circle(circle)),
                [5.0, 0.0],
                &sketch
            ),
            Offer::Taken
        );
        assert_eq!(
            circle_pair.offer_at(
                SketchEntity::Curve(SketchCurve::Segment(line)),
                [5.0, 0.0],
                &sketch
            ),
            Offer::Complete
        );
    }

    #[test]
    fn concentric_slots_accept_only_circular_curves_and_complete_canonically() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::new(0, 0));
        let to = sketch.add_free_point(SketchPoint::new(10, 0));
        let segment = sketch.connect(from, to).expect("line");
        let arc = sketch
            .connect_arc(from, to, AngleMeasurement::from_degrees(90))
            .expect("arc");
        let circle = sketch
            .add_circle(
                SketchPoint::new(5, 5),
                document::sketch::SketchLength::new(3),
            )
            .expect("circle");
        let mut armed = ArmedConstraint::new(ConstraintVerb::Concentric);
        assert_eq!(armed.wants(), Some(PickRequirement::CircularCurve));
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
            Offer::Refused("pick an arc or circle — lines have no center")
        );
        assert!(armed.picked().is_empty());
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Circle(circle)), &sketch),
            Offer::Taken
        );
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Arc(arc)), &sketch),
            Offer::Complete
        );
        assert_eq!(
            armed.kind(&sketch),
            Some(ConstraintKind::concentric(
                SketchCurve::Circle(circle),
                SketchCurve::Arc(arc)
            ))
        );
        assert_eq!(
            armed.kind_at_context(
                &sketch,
                parametric::EvaluationContext::new(std::num::NonZeroU32::new(16).expect("density"))
            ),
            armed.kind(&sketch).ok_or("constraint is incomplete")
        );
        assert_eq!(
            ConstraintVerb::Concentric.icon(),
            Icon::ConstraintConcentric
        );
    }

    #[test]
    fn tangent_branch_is_canonical_under_reversed_curve_picks() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let a = sketch.add_free_point(SketchPoint::new(0, 0));
        let b = sketch.add_free_point(SketchPoint::new(10, 0));
        let line = sketch.connect(a, b).expect("line");
        let circle = sketch
            .add_circle(
                SketchPoint::new(5, 4),
                document::sketch::SketchLength::new(4),
            )
            .expect("circle");
        let context =
            parametric::EvaluationContext::new(std::num::NonZeroU32::new(16).expect("density"));
        let complete = |first, first_locus, second, second_locus| {
            let mut armed = ArmedConstraint::new(ConstraintVerb::Tangent);
            assert_eq!(armed.offer_at(first, first_locus, &sketch), Offer::Taken);
            assert_eq!(
                armed.offer_at(second, second_locus, &sketch),
                Offer::Complete
            );
            armed.kind_at_context(&sketch, context).expect("branch")
        };
        let one = complete(
            SketchEntity::Curve(SketchCurve::Segment(line)),
            [5.0, 0.0],
            SketchEntity::Curve(SketchCurve::Circle(circle)),
            [5.0, 0.0],
        );
        let two = complete(
            SketchEntity::Curve(SketchCurve::Circle(circle)),
            [5.0, 0.0],
            SketchEntity::Curve(SketchCurve::Segment(line)),
            [5.0, 0.0],
        );
        assert_eq!(one, two);
        assert!(matches!(
            one,
            ConstraintKind::Tangent {
                branch: document::sketch::TangentBranch::Line(document::sketch::LineSide::Left),
                ..
            }
        ));
    }

    #[test]
    fn symmetry_waits_for_the_first_subjects_exact_curve_kind_then_a_segment_axis() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let axis_from = sketch.add_free_point(SketchPoint::new(0, -10));
        let axis_to = sketch.add_free_point(SketchPoint::new(0, 10));
        let axis = sketch.connect(axis_from, axis_to).expect("axis");
        let a0 = sketch.add_free_point(SketchPoint::new(-4, 0));
        let a1 = sketch.add_free_point(SketchPoint::new(-4, 4));
        let first_segment = sketch.connect(a0, a1).expect("segment");
        let b0 = sketch.add_free_point(SketchPoint::new(4, 0));
        let b1 = sketch.add_free_point(SketchPoint::new(4, 4));
        let second_segment = sketch.connect(b0, b1).expect("segment");
        let arc = sketch
            .connect_arc(a0, a1, AngleMeasurement::from_degrees(90))
            .expect("arc");
        let circle = sketch
            .add_circle(
                SketchPoint::new(8, 0),
                document::sketch::SketchLength::new(2),
            )
            .expect("circle");
        let context =
            parametric::EvaluationContext::new(std::num::NonZeroU32::new(16).expect("density"));
        let mut armed = ArmedConstraint::new(ConstraintVerb::Symmetry);
        assert_eq!(armed.wants(), Some(PickRequirement::Curve));
        assert_eq!(
            armed.offer(
                SketchEntity::Curve(SketchCurve::Segment(first_segment)),
                &sketch
            ),
            Offer::Taken
        );
        // The second slot narrowed to the first pick's own kind, and says so in the prompt.
        let like_the_first = PickRequirement::MatchingCurve(SketchCurve::Segment(first_segment));
        assert_eq!(armed.wants(), Some(like_the_first));
        assert_eq!(like_the_first.wanted(), "another line");
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Arc(arc)), &sketch),
            Offer::Refused("pick another line")
        );
        assert_eq!(
            armed.picked(),
            &[SketchEntity::Curve(SketchCurve::Segment(first_segment))]
        );
        assert_eq!(
            armed.offer(
                SketchEntity::Curve(SketchCurve::Segment(second_segment)),
                &sketch
            ),
            Offer::Taken
        );
        // The third slot is the mirror axis, which is a line whatever the subjects were.
        assert_eq!(armed.wants(), Some(PickRequirement::Segment));
        assert_eq!(
            armed.offer(
                SketchEntity::Curve(SketchCurve::Segment(first_segment)),
                &sketch
            ),
            Offer::Refused("already picked")
        );
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Circle(circle)), &sketch),
            Offer::Refused("that is not a line")
        );
        assert_eq!(
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(axis)), &sketch),
            Offer::Complete
        );
        assert!(matches!(
            armed.kind_at_context(&sketch, context),
            Ok(ConstraintKind::Symmetry {
                axis: held_axis,
                ..
            }) if held_axis == axis
        ));
        assert_eq!(ConstraintVerb::Symmetry.icon(), Icon::ConstraintSymmetry);
    }

    #[test]
    fn symmetry_dynamic_requirement_covers_arcs_circles_and_malformed_restore() {
        let arc = ArmedConstraint::from_parts(
            ConstraintVerb::Symmetry,
            vec![SketchEntity::Curve(SketchCurve::Arc(1))],
        );
        assert_eq!(
            arc.wants(),
            Some(PickRequirement::MatchingCurve(SketchCurve::Arc(1)))
        );
        assert_eq!(arc.wants().expect("waiting").wanted(), "another arc");
        let circle = ArmedConstraint::from_parts(
            ConstraintVerb::Symmetry,
            vec![SketchEntity::Curve(SketchCurve::Circle(2))],
        );
        assert_eq!(
            circle.wants(),
            Some(PickRequirement::MatchingCurve(SketchCurve::Circle(2)))
        );
        let malformed = ArmedConstraint::from_parts(
            ConstraintVerb::Symmetry,
            vec![
                SketchEntity::Curve(SketchCurve::Arc(1)),
                SketchEntity::Curve(SketchCurve::Circle(2)),
            ],
        );
        assert!(malformed.picked().is_empty());
        assert_eq!(malformed.wants(), Some(PickRequirement::Curve));
        let complete = ArmedConstraint::from_parts(
            ConstraintVerb::Symmetry,
            vec![
                SketchEntity::Curve(SketchCurve::Segment(1)),
                SketchEntity::Curve(SketchCurve::Segment(2)),
                SketchEntity::Curve(SketchCurve::Segment(3)),
            ],
        );
        assert!(complete.picked().is_empty());
        let overfull = ArmedConstraint::from_parts(
            ConstraintVerb::Symmetry,
            vec![
                SketchEntity::Curve(SketchCurve::Circle(1)),
                SketchEntity::Curve(SketchCurve::Circle(2)),
                SketchEntity::Curve(SketchCurve::Segment(3)),
                SketchEntity::Point(4),
            ],
        );
        assert!(overfull.picked().is_empty());
    }

    /// An aggregate is a real curve with a real identity, and the slots still turn it away: no
    /// relation can read a spline or an ellipse as a whole shape, so taking the pick would arm a
    /// gesture that completes and then cannot be applied. The refusal lands at the click, where the
    /// tool is still waiting for a curve it can hold.
    #[test]
    fn a_curve_slot_refuses_an_aggregate_the_relations_cannot_read() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::new(0, 0));
        let to = sketch.add_free_point(SketchPoint::new(10, 0));
        let line = sketch.connect(from, to).expect("line");
        let ellipse = sketch
            .add_ellipse(
                SketchPoint::new(0, 20),
                SketchPoint::new(10, 20),
                SketchPoint::new(0, 24),
            )
            .expect("ellipse");
        let aggregate = SketchEntity::Curve(SketchCurve::Ellipse(ellipse));
        assert!(
            holds(&sketch, aggregate),
            "the drawing does hold it — this is not a staleness refusal"
        );

        let mut tangent = ArmedConstraint::new(ConstraintVerb::Tangent);
        assert_eq!(
            tangent.offer_at(aggregate, [0.0, 0.0], &sketch),
            Offer::Refused("that is not a curve a constraint can hold")
        );
        assert!(
            tangent.picked().is_empty(),
            "and the gesture is still armed"
        );

        let mut concentric = ArmedConstraint::new(ConstraintVerb::Concentric);
        assert_eq!(
            concentric.offer(aggregate, &sketch),
            Offer::Refused("pick an arc or circle — lines have no center"),
            "an ellipse has a center and is still not a constant radius about one"
        );

        let mut coincident = ArmedConstraint::new(ConstraintVerb::Coincident);
        coincident.offer(SketchEntity::Point(from), &sketch);
        assert_eq!(
            coincident.offer(aggregate, &sketch),
            Offer::Refused("that is not a point or a curve")
        );
        assert_eq!(
            coincident.offer(SketchEntity::Curve(SketchCurve::Segment(line)), &sketch),
            Offer::Complete,
            "still armed, and a curve it can hold lands"
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
            armed.offer(SketchEntity::Curve(SketchCurve::Segment(segment)), &sketch),
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
