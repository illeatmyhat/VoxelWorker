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

/// A block term that does not land on a whole voxel names BOTH neighbors, because
/// picking one of them is the user's next action.
#[test]
fn a_non_landing_block_term_names_both_neighboring_voxels() {
    // At density 3, half a block is 1.5 voxels — between 1 and 2.
    let error = signed_field(0, 3)
        .parse_and_validate("0.5b")
        .expect_err("a fractional voxel count must be rejected");
    assert!(
        error.contains("whole voxel") && error.contains('1') && error.contains('2'),
        "the message must name both neighbors, got: {error}"
    );
}

/// Unparseable text reports the parse error and commits nothing.
#[test]
fn unparseable_text_does_not_commit() {
    assert!(signed_field(0, 16)
        .parse_and_validate("not a measurement")
        .is_err());
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

/// The field reads ARITHMETIC, not just a literal.
///
/// This is the whole of what the expression grammar buys an author today: a size can be typed
/// as the calculation that produced it instead of as the answer, and the answer still has to
/// land on a whole voxel like any other.
#[test]
fn a_field_accepts_an_expression_and_lands_it_on_whole_voxels() {
    let commit = signed_field(0, 16)
        .parse_and_validate("2 * 3 blocks + 4 voxels")
        .expect("six blocks and four voxels is a whole number of voxels");
    assert_eq!(commit.voxels, 100);

    assert!(
        signed_field(0, 16)
            .parse_and_validate("1 block / 3")
            .is_err(),
        "a third of a block is not a whole voxel at density 16"
    );
    assert!(
        bounded_field(4, 16)
            .parse_and_validate("2 blocks - 2 blocks")
            .is_err(),
        "the lower bound judges the ANSWER, not the literals it was built from"
    );
}

/// A lone literal keeps the split it was authored with; a calculation cannot.
///
/// The retained measurement is what a density re-target re-evaluates, so `3 blocks` must come
/// back as a BLOCK term and not as forty-eight voxels that will not rescale. The compound case
/// pins the known limitation rather than hiding it: it retains the answer, so it retains a
/// voxel count.
#[test]
fn a_lone_literal_retains_its_blocks_and_a_calculation_retains_its_answer() {
    let literal = signed_field(0, 16)
        .parse_and_validate("3 blocks")
        .expect("a plain literal commits");
    assert_eq!(literal.measurement, units::parse("3b").unwrap());
    assert_eq!(
        literal.measurement.to_voxels(32),
        Ok(96),
        "the block term rescales with the density"
    );

    let calculated = signed_field(0, 16)
        .parse_and_validate("1 block * 3")
        .expect("the same quantity, calculated");
    assert_eq!(calculated.voxels, literal.voxels);
    assert_eq!(
        calculated.measurement.to_voxels(32),
        Ok(48),
        "a retained ANSWER is a voxel count and does not rescale"
    );
}

/// A name is refused because the table is empty, and the message says which name.
///
/// Not a stub refusal: an empty symbol table's honest answer to `width` is that it knows no
/// such parameter, and that is the same answer a populated one gives for a typo. The day the
/// document carries parameters, this path starts succeeding without changing.
#[test]
fn a_parameter_name_is_refused_by_name() {
    let refusal = signed_field(0, 16)
        .parse_and_validate("width * 2")
        .expect_err("no parameter is defined");
    assert!(
        refusal.contains("width"),
        "the refusal must name the parameter, got: {refusal}"
    );

    assert!(
        signed_field(0, 16)
            .parse_and_validate("3 blocks * 3 blocks")
            .is_err(),
        "a length times a length is an area, and a field holds lengths"
    );
}
