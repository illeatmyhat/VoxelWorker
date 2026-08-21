//! Coverage for the measurement commit protocol.
//!
//! This path had NO tests while it existed as two hand-rolled copies, despite being the
//! only route authored spatial values take into the document. The validation matrix is
//! exercised directly; the frame-level rules go through a headless egui `Context`.

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

/// A field with the properties of the OFFSET site: signed, no lower bound.
fn signed_field<'a>(text_seed: i64, density: u32) -> MeasurementField<'a> {
    MeasurementField::new(egui::Id::new("test_field"), "X", text_seed, density)
}

/// A field with the properties of the SIZE site: at least one voxel.
fn bounded_field<'a>(text_seed: i64, density: u32) -> MeasurementField<'a> {
    signed_field(text_seed, density).min_voxels(1, "size must be at least 1 voxel")
}

/// Run one headless frame containing the field, returning what it committed.
///
/// The field is rebuilt by the closure rather than passed in, because egui's `run_ui`
/// takes an `FnMut` and a field is consumed by `show`.
fn run_field_frame(build: impl Fn() -> MeasurementField<'static>) -> Option<MeasurementCommit> {
    let context = egui::Context::default();
    let mut committed = None;
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(400.0, 200.0),
        )),
        ..Default::default()
    };
    let _ = context.run_ui(raw_input, |ui| {
        committed = build().show(ui);
    });
    committed
}

/// Rule 4, at frame level: a field nobody has touched commits nothing. An idle frame
/// must never write to the document — this is what keeps the headless render path
/// (`shot`, the goldens) mutation-free.
#[test]
fn an_idle_frame_commits_nothing() {
    assert_eq!(run_field_frame(|| signed_field(32, 16)), None);
}

/// Drive the field across frames on ONE context, so focus survives between them.
fn field_frame(
    context: &egui::Context,
    events: Vec<egui::Event>,
    committed: &mut Option<MeasurementCommit>,
) {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(400.0, 200.0),
        )),
        events,
        ..Default::default()
    };
    let _ = context.run_ui(raw_input, |ui| {
        if let Some(commit) = signed_field(0, 16).show(ui) {
            *committed = Some(commit);
        }
    });
}

/// Type three blocks into a focused field, finish with `key`, and report what committed.
///
/// The box is focused BY ID rather than by clicking a guessed coordinate — it publishes a stable
/// one precisely so nothing has to know where in a layout it landed.
fn type_then_press(key: egui::Key) -> Option<MeasurementCommit> {
    let context = egui::Context::default();
    let mut committed = None;
    field_frame(&context, Vec::new(), &mut committed);
    context.memory_mut(|memory| {
        memory.request_focus(MeasurementField::box_id(egui::Id::new("test_field")));
    });
    field_frame(&context, Vec::new(), &mut committed);
    // Select the seed before typing, or the new text lands BESIDE it and the field refuses a
    // duplicated unit instead of reading three blocks.
    field_frame(
        &context,
        vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text("3b".to_owned()),
        ],
        &mut committed,
    );
    field_frame(
        &context,
        vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }],
        &mut committed,
    );
    committed
}

/// Escape ABANDONS an edit; Enter commits it.
///
/// The two go together because Escape surrenders focus in egui, which fires the same
/// `lost_focus()` the commit hangs off — so without the guard the abandon arrives as a commit of
/// whatever was half-typed, and the only way to see the guard working is beside the case it must
/// not break. `DragValue` in egui's own tree carries this same pair of lines.
///
/// The Enter half is also this harness's control: if the click did not land in the box, nothing
/// would commit and the Escape assertion would pass for the wrong reason.
#[test]
fn escape_abandons_an_edit_and_enter_commits_it() {
    assert_eq!(
        type_then_press(egui::Key::Enter).map(|commit| commit.voxels),
        Some(48),
        "the control: three blocks typed and entered commits forty-eight voxels"
    );
    assert_eq!(
        type_then_press(egui::Key::Escape),
        None,
        "escape writes nothing, even though it surrendered focus"
    );
}
