//! Coverage for the measurement commit protocol.
//!
//! This path had NO tests while it existed as two hand-rolled copies, despite being the
//! only route authored spatial values take into the document. The validation matrix is
//! exercised directly; the frame-level rules go through a headless egui `Context`.

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

/// An UNSIGNED bound is opt-in, so an unbounded field takes negatives. This is the
/// property the outset editor needs — a negative outset insets — and the property a
/// naive "size-shaped" extraction would have silently dropped.
#[test]
fn a_signed_field_accepts_a_negative_measurement() {
    let field = signed_field(0, 16);
    let commit = field
        .parse_and_validate("-2v")
        .expect("a negative measurement must commit on an unbounded field");
    assert_eq!(commit.voxels, -2);
}

/// The bound rejects with the CALLER's sentence, not a generated one — the message is
/// about the quantity ("size"), which only the call site knows.
#[test]
fn a_bounded_field_rejects_below_the_minimum_with_the_callers_message() {
    let error = bounded_field(4, 16)
        .parse_and_validate("0v")
        .expect_err("zero must be rejected when a minimum of 1 is set");
    assert_eq!(error, "size must be at least 1 voxel");
}

/// The same text that a bounded field rejects is legitimate on an unbounded one, so the
/// bound is genuinely the only difference between the two sites.
#[test]
fn the_bound_is_the_only_difference_between_the_two_sites() {
    assert!(signed_field(4, 16).parse_and_validate("0v").is_ok());
    assert!(bounded_field(4, 16).parse_and_validate("0v").is_err());
}

/// A block term that does not land on a whole voxel names BOTH neighbours, because
/// picking one of them is the user's next action.
#[test]
fn a_non_landing_block_term_names_both_neighbouring_voxels() {
    // At density 3, half a block is 1.5 voxels — between 1 and 2.
    let error = signed_field(0, 3)
        .parse_and_validate("0.5b")
        .expect_err("a fractional voxel count must be rejected");
    assert!(
        error.contains("whole voxel") && error.contains('1') && error.contains('2'),
        "the message must name both neighbours, got: {error}"
    );
}

/// Unparseable text reports the parse error and commits nothing.
#[test]
fn unparseable_text_does_not_commit() {
    assert!(signed_field(0, 16).parse_and_validate("not a measurement").is_err());
}

/// A commit carries BOTH halves: the authored expression to retain, and the canonical
/// voxels the resolve uses. At density 16 one block is 16 voxels, so the two differ and
/// neither can be reconstructed from the other without the density.
#[test]
fn a_commit_carries_both_the_expression_and_the_landed_voxels() {
    let commit = signed_field(0, 16)
        .parse_and_validate("1b")
        .expect("one whole block must commit at density 16");
    assert_eq!(commit.voxels, 16);
    assert_eq!(commit.measurement, units::parse("1b").unwrap());
}

/// Rule 4, at frame level: a field nobody has touched commits nothing. An idle frame
/// must never write to the document — this is what keeps the headless render path
/// (`shot`, the goldens) mutation-free.
#[test]
fn an_idle_frame_commits_nothing() {
    assert_eq!(run_field_frame(|| signed_field(32, 16)), None);
}
