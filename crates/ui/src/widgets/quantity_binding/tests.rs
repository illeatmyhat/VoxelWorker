//! Coverage for what the text in a quantity box MEANS: the validation matrix, per dimension.
//!
//! The protocol's own rules live beside the protocol; these are the two bindings' answers.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]

use super::*;
use parametric::units;

/// The OFFSET site: signed, no lower bound.
fn signed() -> LengthBinding<'static> {
    LengthBinding::new(16)
}

/// The SIZE site: at least one voxel.
fn bounded() -> LengthBinding<'static> {
    LengthBinding::new(16).floor(1, "size must be at least 1 voxel")
}

/// What a length binding committed, or the sentence it refused with.
fn length_of(binding: &LengthBinding<'_>, text: &str) -> Result<MeasurementCommit, String> {
    binding.read(text).map(|accepted| accepted.value)
}

/// **THE SEED MUST PARSE.** A box the author opened and left alone commits nothing, and the
/// protocol arranges that by comparing text against the seed — so a binding whose formatter emits
/// something its own reader chokes on would turn an untouched box into a refusal the moment focus
/// moved. Asserted here for both dimensions, because it is a property of the PAIR.
#[test]
fn every_binding_reads_the_seed_it_hands_out() {
    for voxels in [-33_i64, -1, 0, 1, 7, 16, 100] {
        let seed = LengthBinding::seed(voxels, 16);
        let commit = length_of(&signed(), &seed)
            .unwrap_or_else(|error| panic!("the length binding must read `{seed}`: {error}"));
        assert_eq!(commit.voxels, voxels, "and read it back unchanged");
    }

    for degrees in [0.0_f64, 1.0, 45.0, 31.24, 90.0, 179.5, 180.0] {
        let angle = parametric::units::AngleMeasurement::try_from_degrees_f64(degrees)
            .expect("a finite angle");
        let seed = AngleBinding::seed(angle);
        let read = AngleBinding
            .read(&seed)
            .unwrap_or_else(|error| panic!("the angle binding must read `{seed}`: {error}"));
        assert_eq!(
            read.value.to_degrees_f64(),
            degrees,
            "and read `{seed}` back unchanged"
        );
        assert_eq!(read.settled_text, seed, "and settle on the same text");
    }
}

/// An UNSIGNED bound is opt-in, so an unbounded field takes negatives. This is the property the
/// outset editor needs — a negative outset insets — and the property a naive "size-shaped"
/// extraction would have silently dropped.
#[test]
fn a_signed_length_accepts_a_negative_measurement() {
    let commit = length_of(&signed(), "-2v")
        .expect("a negative measurement must commit on an unbounded binding");
    assert_eq!(commit.voxels, -2);
}

/// The bound rejects with the CALLER's sentence, not a generated one — the message is about the
/// quantity ("size"), which only the call site knows.
#[test]
fn a_floored_length_rejects_below_the_minimum_with_the_callers_message() {
    let error =
        length_of(&bounded(), "0v").expect_err("zero must be rejected when a minimum of 1 is set");
    assert_eq!(error, "size must be at least 1 voxel");
}

/// The same text that a floored binding rejects is legitimate on an unbounded one, so the bound
/// is genuinely the only difference between the two sites.
#[test]
fn the_bound_is_the_only_difference_between_the_two_sites() {
    assert!(length_of(&signed(), "0v").is_ok());
    assert!(length_of(&bounded(), "0v").is_err());
}

/// A block term that does not land on a whole voxel names BOTH neighbors, because picking one of
/// them is the user's next action.
#[test]
fn a_non_landing_block_term_names_both_neighboring_voxels() {
    // At density 3, half a block is 1.5 voxels — between 1 and 2.
    let error = length_of(&LengthBinding::new(3), "0.5b")
        .expect_err("a fractional voxel count must be rejected");
    assert!(
        error.contains("whole voxel") && error.contains('1') && error.contains('2'),
        "the message must name both neighbors, got: {error}"
    );
}

/// Unparseable text reports the parse error and commits nothing.
#[test]
fn unparseable_text_does_not_commit() {
    assert!(length_of(&signed(), "not a measurement").is_err());
    assert!(AngleBinding.read("not an angle").is_err());
}

/// A commit carries BOTH halves: the authored expression to retain, and the canonical voxels the
/// resolve uses. At density 16 one block is 16 voxels, so the two differ and neither can be
/// reconstructed from the other without the density.
#[test]
fn a_commit_carries_both_the_expression_and_the_landed_voxels() {
    let commit = length_of(&signed(), "1b").expect("one whole block must commit at density 16");
    assert_eq!(commit.voxels, 16);
    assert_eq!(commit.measurement, units::parse("1b").unwrap());
}

/// The binding reads ARITHMETIC, not just a literal.
///
/// This is the whole of what the expression grammar buys an author today: a size can be typed as
/// the calculation that produced it instead of as the answer, and the answer still has to land on
/// a whole voxel like any other.
#[test]
fn a_length_accepts_an_expression_and_lands_it_on_whole_voxels() {
    let commit = length_of(&signed(), "2 * 3 blocks + 4 voxels")
        .expect("six blocks and four voxels is a whole number of voxels");
    assert_eq!(commit.voxels, 100);

    assert!(
        length_of(&signed(), "1 block / 3").is_err(),
        "a third of a block is not a whole voxel at density 16"
    );
    assert!(
        length_of(&bounded(), "2 blocks - 2 blocks").is_err(),
        "the lower bound judges the ANSWER, not the literals it was built from"
    );
}

/// A lone literal keeps the split it was authored with; a calculation cannot.
///
/// The retained measurement is what a density re-target re-evaluates, so `3 blocks` must come
/// back as a BLOCK term and not as forty-eight voxels that will not rescale. The compound case
/// pins the known limitation rather than hiding it: it retains the answer, so it retains a voxel
/// count.
#[test]
fn a_lone_literal_retains_its_blocks_and_a_calculation_retains_its_answer() {
    let literal = length_of(&signed(), "3 blocks").expect("a plain literal commits");
    assert_eq!(literal.measurement, units::parse("3b").unwrap());
    assert_eq!(
        literal.measurement.to_voxels(32),
        Ok(96),
        "the block term rescales with the density"
    );

    let calculated = length_of(&signed(), "1 block * 3").expect("the same quantity, calculated");
    assert_eq!(calculated.voxels, literal.voxels);
    assert_eq!(
        calculated.measurement.to_voxels(32),
        Ok(48),
        "a retained ANSWER is a voxel count and does not rescale"
    );
}

/// A name is refused because the table is empty, and the message says which name.
///
/// Not a stub refusal: an empty symbol table's honest answer to `width` is that it knows no such
/// parameter, and that is the same answer a populated one gives for a typo. The day the document
/// carries parameters, this path starts succeeding without changing.
#[test]
fn a_parameter_name_is_refused_by_name() {
    let refusal = length_of(&signed(), "width * 2").expect_err("no parameter is defined");
    assert!(
        refusal.contains("width"),
        "the refusal must name the parameter, got: {refusal}"
    );

    assert!(
        length_of(&signed(), "3 blocks * 3 blocks").is_err(),
        "a length times a length is an area, and a field holds lengths"
    );
}

/// Each binding turns the OTHER dimension away, and the grammar's own words say which is which.
///
/// The pair is the point. Neither field is the privileged one, and neither has to know the
/// dimension list to say no — the grammar it reads with already does.
#[test]
fn each_binding_refuses_the_other_dimension() {
    let into_a_length = length_of(&signed(), "45 deg").expect_err("a degree is not a length");
    assert!(
        into_a_length.contains("deg") && into_a_length.contains("length"),
        "got: {into_a_length}"
    );

    let into_an_angle = AngleBinding
        .read("3 blocks")
        .expect_err("a length is not an angle");
    assert!(into_an_angle.contains("angle"), "got: {into_an_angle}");
}

/// An angle reads the same arithmetic a length does, and stays exact through it.
#[test]
fn an_angle_accepts_an_expression() {
    let read = AngleBinding
        .read("45 deg / 2")
        .expect("half of forty-five degrees");
    assert_eq!(read.value.to_degrees_f64(), 22.5);
    assert_eq!(read.settled_text, "22.5\u{b0}");
}

/// Past the half turn the drawing cannot tell the claim apart from one inside it, so the refusal
/// names the equivalent rather than folding to it silently.
///
/// Folding would be the worse bug: the author types 200, the drawing settles at 20, and nothing
/// on screen says why.
#[test]
fn an_angle_past_the_half_turn_is_refused_with_its_equivalent() {
    let refusal = AngleBinding
        .read("200 deg")
        .expect_err("two hundred degrees is not statable");
    assert!(
        refusal.contains("20.00\u{b0}"),
        "the refusal must offer the equivalent, got: {refusal}"
    );
    assert!(
        AngleBinding.read("-1 deg").is_err(),
        "and the bound holds on the low side too"
    );

    // The ends themselves are legitimate: at zero the solver row IS Parallel's, and at a half turn
    // it is the same claim from the other end.
    assert!(AngleBinding.read("0 deg").is_ok());
    assert!(AngleBinding.read("180 deg").is_ok());
}

/// A bare number in a length field is BLOCKS, and it is retained as a block TERM so it rescales
/// with the density exactly as `3 blocks` does. The default belongs to the binding, not the
/// grammar: a bare number stays dimensionless in an expression, so `2 * 3 blocks` keeps its scale
/// factor and `3 blocks * 3 blocks` stays an area. Only a dimensionless ANSWER takes the default.
#[test]
fn a_bare_number_is_blocks_in_a_length_field() {
    let bare = length_of(&signed(), "3").expect("a bare number defaults to blocks");
    assert_eq!(bare.voxels, 48);
    assert_eq!(bare.measurement, units::parse("3 blocks").unwrap());
    assert_eq!(
        bare.measurement.to_voxels(32),
        Ok(96),
        "and rescales like a block term"
    );

    let calculated = length_of(&signed(), "2 * 3").expect("a dimensionless answer is blocks too");
    assert_eq!(calculated.voxels, 96);
    assert_eq!(length_of(&signed(), "3.5").unwrap().voxels, 56);
    assert_eq!(length_of(&signed(), "-1").unwrap().voxels, -16);

    assert!(
        length_of(&signed(), "2 blocks + 4").is_err(),
        "the default is for a dimensionless ANSWER, not a term inside a sum"
    );
}

/// A bare number in an angle field is \u{b0}REES, and a degree in a length field is still refused.
#[test]
fn a_bare_number_is_degrees_in_an_angle_field() {
    let read = AngleBinding
        .read("45")
        .expect("a bare number defaults to degrees");
    assert_eq!(read.value.to_degrees_f64(), 45.0);
    assert_eq!(read.settled_text, "45\u{b0}");
    assert_eq!(
        AngleBinding.read("90 / 2").unwrap().value.to_degrees_f64(),
        45.0
    );
    assert!(
        AngleBinding.read("200").is_err(),
        "the half-turn bound still judges the answer"
    );
    assert!(length_of(&signed(), "45 deg").is_err());
    assert!(AngleBinding.read("3 blocks").is_err());
}

/// Radians are an angle unit. The store is exact degrees and pi is not a rational, so a radian
/// literal converts at the literal through the same door a solved angle takes, and seeds back in
/// degrees: `1 rad` settles as `57.3\u{b0}`, which is the canonical rendering, not a loss.
#[test]
fn radians_are_an_angle_and_seed_back_as_degrees() {
    let read = AngleBinding.read("1 rad").expect("a radian is an angle");
    assert_eq!(read.value.to_degrees_f64(), 1.0_f64.to_degrees());
    assert_eq!(read.settled_text, "57.3\u{b0}");
    let half_turn = AngleBinding.read("3.14159265358979 radians").unwrap();
    assert!((half_turn.value.to_degrees_f64() - 180.0).abs() < 1e-9);
    assert_eq!(
        AngleBinding
            .read("0.5rad * 2")
            .unwrap()
            .value
            .to_degrees_f64(),
        1.0_f64.to_degrees()
    );
    let into_a_length = length_of(&signed(), "1 rad").expect_err("a radian is not a length");
    assert!(
        into_a_length.contains("rad") && into_a_length.contains("length"),
        "got: {into_a_length}"
    );
}

/// The default is for text that NAMED NO UNIT, not for an answer that happens to be dimensionless.
/// `3 blocks / 1 block` is a ratio the author built out of units, and minting it as three blocks
/// would be the tool deciding what they meant; it is refused like any other non-length.
#[test]
fn a_cancelled_ratio_does_not_take_the_default() {
    assert!(length_of(&signed(), "3 blocks / 1 block").is_err());
    assert!(AngleBinding.read("90 deg / 45 deg").is_err());
    assert!(
        length_of(&signed(), "(2 + 4) * 1.5").is_ok(),
        "no unit named anywhere: blocks"
    );
}
