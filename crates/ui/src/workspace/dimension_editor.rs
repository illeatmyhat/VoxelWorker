//! The inline editor bound to a sketch dimension.
//!
//! The widget is a box at a place holding text and knows nothing about dimensions; this is the
//! half that knows. It looks the constraint up, seeds the box from what the drawing currently
//! says, and turns a committed value back into the same
//! [`PanelResponse::restate_sketch_dimension`] the rail used to produce — so the door the shell
//! applies it through did not change when the door the author reaches for did.

use crate::panel::{PanelResponse, PanelState};
use crate::widgets::{MeasurementEdit, MeasurementEntryOutcome};

/// An open editor and the dimension it will restate.
///
/// The pairing lives here rather than on [`MeasurementEdit`] because the widget must stay able to
/// open over a placement gesture, which has no constraint to name yet.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenDimensionEditor {
    /// The constraint whose value is being typed.
    pub constraint: document::sketch::EntityId,
    /// Where the box sits and what it opened holding.
    pub editor: MeasurementEdit,
}

/// The sentence shown when a dimension is typed below its floor.
///
/// A dimension of nothing is not a smaller dimension, it is a different claim — two points in one
/// place is Coincident, and a rim of nothing is not a curve — so every member has a floor of one
/// voxel and says so in its own words.
fn at_least_one_voxel(dimension: &document::sketch::Dimension) -> &'static str {
    use document::sketch::Dimension;
    match dimension {
        Dimension::Span { .. } | Dimension::SpanAlong { .. } => "a span is at least one voxel",
        Dimension::Gap { .. } | Dimension::RimGap { .. } => "a gap is at least one voxel",
        Dimension::Radius { .. } | Dimension::Diameter { .. } => "a rim needs at least one voxel",
        // Never reached: an angle opens no editor, because the grammar has no angle literals.
        Dimension::Angle { .. } => "an angle is measured in degrees",
    }
}

/// Draw the open dimension editor, if one is open, and act on what it did.
///
/// **The editor closes itself when its subject goes.** A restate mints a NEW constraint id, and an
/// undo or a delete can take the old one out from under an open box; either way the lookup below
/// fails and the box goes rather than hanging over the drawing addressing nothing.
pub fn sketch_dimension_editor(
    ctx: &egui::Context,
    state: &PanelState,
    open: &mut Option<OpenDimensionEditor>,
    response: &mut PanelResponse,
) {
    let Some(held) = open.as_ref() else {
        return;
    };
    let Some(dimension) = dimension_of(state, held.constraint) else {
        *open = None;
        return;
    };
    let id_base = egui::Id::new(("sketch_dimension_editor", held.constraint));
    let outcome = held.editor.show(
        ctx,
        id_base,
        state.geometry.voxels_per_block,
        Some((1, at_least_one_voxel(&dimension))),
    );
    match outcome {
        MeasurementEntryOutcome::Committed(commit) => {
            let length =
                document::sketch::SketchLength::retained(commit.measurement, commit.voxels);
            if let Some(restated) = dimension.with_length(length) {
                response.restate_sketch_dimension = Some((held.constraint, restated));
            }
            *open = None;
        }
        MeasurementEntryOutcome::Cancelled => *open = None,
        // A refusal keeps the box open with the rejected text and its complaint — closing here
        // would throw away what the author typed at the moment they can still fix it.
        MeasurementEntryOutcome::Refused | MeasurementEntryOutcome::Idle => {}
    }
}

/// The dimension `constraint` names in the sketch being edited, if it still names one.
fn dimension_of(
    state: &PanelState,
    constraint: document::sketch::EntityId,
) -> Option<document::sketch::Dimension> {
    let sketch = state.sketch_mode?;
    let document::scene::NodeContent::SketchTool { producer, .. } =
        &state.scene.node_by_id(sketch)?.content
    else {
        return None;
    };
    let document::sketch::ConstraintKind::Dimension(dimension) = producer
        .sketch
        .constraints()
        .iter()
        .find(|held| held.id == constraint)
        .map(|held| held.kind)?
    else {
        return None;
    };
    Some(dimension)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{sketch_dimension_editor, OpenDimensionEditor};
    use crate::widgets::MeasurementEdit;

    const DENSITY: u32 = 16;

    /// A panel holding one sketch node with one span dimension, and that dimension's id.
    fn a_panel_showing_a_dimension() -> (crate::panel::PanelState, document::sketch::EntityId) {
        let mut sketch = document::sketch::Sketch::empty(document::sketch::PlaneAxis::Z);
        let tail = sketch.add_free_point(document::sketch::SketchPoint::new(0, 0));
        let head = sketch.add_free_point(document::sketch::SketchPoint::new(32, 0));
        let constraint = sketch
            .add_constraint(
                document::sketch::ConstraintKind::Dimension(document::sketch::Dimension::Span {
                    from: tail,
                    to: head,
                    length: document::sketch::SketchLength::new(32),
                }),
                parametric::EvaluationContext::new(
                    core::num::NonZeroU32::new(DENSITY).expect("a nonzero density"),
                ),
            )
            .expect("a span the drawing already satisfies");

        let mut state = crate::panel::PanelState::default();
        state.geometry.voxels_per_block = DENSITY;
        let node = state.scene.add_node(document::scene::Node::new(
            "Sketch",
            document::scene::NodeContent::SketchTool {
                producer: document::sketch::SketchSolid::extrude(sketch, 8),
                material: voxel_core::core_geom::MaterialChoice::default(),
            },
        ));
        state.sketch_mode = Some(node);
        (state, constraint)
    }

    /// Run one frame of the editor over `state`, feeding `events`, and report what the panel was
    /// told.
    ///
    /// The context is the CALLER's, held across frames. A fresh one per frame would lose egui's
    /// memory between them, and focus — which the box asks for on one frame and receives on the
    /// next — is exactly what lives there.
    fn editor_frame(
        context: &egui::Context,
        state: &crate::panel::PanelState,
        open: &mut Option<OpenDimensionEditor>,
        events: Vec<egui::Event>,
    ) -> crate::panel::PanelResponse {
        let mut response = crate::panel::PanelResponse::default();
        let input = egui::RawInput {
            events,
            ..Default::default()
        };
        let _ = context.run_ui(input, |ctx| {
            sketch_dimension_editor(ctx, state, open, &mut response);
        });
        response
    }

    /// An open editor over `constraint`, seeded the way a double-click on a two-block span seeds
    /// one.
    fn opened_on(constraint: document::sketch::EntityId) -> OpenDimensionEditor {
        OpenDimensionEditor {
            constraint,
            editor: MeasurementEdit::new(
                egui::Rect::from_min_size(egui::pos2(100.0, 60.0), egui::vec2(40.0, 14.0)),
                "2b",
            ),
        }
    }

    /// Select the whole seed and type `text` over it — what an author does to a box that opened
    /// with its contents selected.
    fn replace_the_seed_with(text: &str) -> Vec<egui::Event> {
        vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text(text.to_owned()),
        ]
    }

    /// A bare key press.
    fn pressing(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    /// The whole binding, driven through the widget rather than around it: the box opens, takes
    /// the keyboard on its own, and Enter restates the dimension through the panel's door.
    ///
    /// Two frames, because that is what the real thing does — the box requests focus on the frame
    /// it appears, and egui hands it over for the next one.
    #[test]
    fn typing_a_length_and_pressing_enter_restates_the_dimension() {
        let (state, constraint) = a_panel_showing_a_dimension();
        let context = egui::Context::default();
        let mut open = Some(opened_on(constraint));

        let opening = editor_frame(&context, &state, &mut open, Vec::new());
        assert_eq!(
            opening.restate_sketch_dimension, None,
            "opening writes nothing"
        );
        assert!(open.is_some(), "and the box stays up");

        let mut events = replace_the_seed_with("3b");
        events.push(pressing(egui::Key::Enter));
        let committed = editor_frame(&context, &state, &mut open, events);

        let (restated_id, restated) = committed
            .restate_sketch_dimension
            .expect("Enter commits the typed length");
        assert_eq!(
            restated_id, constraint,
            "and it restates the one that was open"
        );
        assert_eq!(
            restated.length().map(|length| length.value()),
            Some(f64::from(3 * DENSITY)),
            "three blocks at this density"
        );
        assert!(open.is_none(), "and the box closes behind it");
    }

    /// Escape abandons: nothing is written, the box goes, and the shell never sees the key.
    ///
    /// The last of those is the one worth a test. Escape is also the global cancel, and an author
    /// abandoning a number has not asked to abandon the tool they are holding.
    #[test]
    fn escape_closes_the_box_writing_nothing_and_the_shell_never_sees_the_key() {
        let (state, constraint) = a_panel_showing_a_dimension();
        let context = egui::Context::default();
        let mut open = Some(opened_on(constraint));
        let _ = editor_frame(&context, &state, &mut open, Vec::new());

        let mut response = crate::panel::PanelResponse::default();
        let mut events = replace_the_seed_with("9b");
        events.push(pressing(egui::Key::Escape));
        let input = egui::RawInput {
            events,
            ..Default::default()
        };
        let mut reached_the_shell = true;
        let _ = context.run_ui(input, |ctx| {
            sketch_dimension_editor(ctx, &state, &mut open, &mut response);
            // Read AFTER the editor ran, the way the shell reads it: the global shortcut pass runs
            // once the panel is done, and this is the question it asks.
            reached_the_shell =
                ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        });

        assert_eq!(
            response.restate_sketch_dimension, None,
            "an abandoned edit writes nothing"
        );
        assert!(open.is_none(), "and the box goes");
        assert!(
            !reached_the_shell,
            "the editor consumed the key, so the global cancel does not also fire"
        );
    }
}
