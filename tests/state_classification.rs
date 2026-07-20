//! The classification scheme (ADR 0022) as it stands on the REAL state structs, not on a
//! fixture. `crates/snapshot` proves the derive works; this file pins what the derive was
//! pointed at, which is the part that can drift silently.
//!
//! Two things are worth a test rather than a reading. First, that the document set is
//! exactly the scene — the whole force of ADR 0022's decision 1 is that a shared project
//! file carries what the model IS and nothing about where somebody was working, and a
//! field quietly acquiring `#[snapshot(document)]` is how that erodes. Second, that the
//! escape hatches stay countable: `transient` and `derived` are the two ways to make the
//! compiler stop complaining without deciding anything, and ADR 0022 says plainly that
//! whether they stay honest depends on review. A test that names each one by hand is how
//! a new one gets noticed — adding an escape hatch should cost a deliberate edit here.

use snapshot::{Snapshot, StateCategory};
use ui::panel::PanelState;
use voxel_worker::AppConfig;

#[test]
fn the_document_carries_the_scene_and_nothing_else() {
    let document_fields: Vec<&str> = AppConfig::document_fields()
        .iter()
        .map(|field| field.name)
        .collect();
    assert_eq!(
        document_fields,
        ["scene"],
        "a field gained `#[snapshot(document)]`: it will now travel inside every shared \
         project file. If that is intended, update this test deliberately"
    );

    let panel_document_fields: Vec<&str> = PanelState::document_fields()
        .iter()
        .map(|field| field.name)
        .collect();
    assert_eq!(panel_document_fields, ["scene"]);
}

#[test]
fn the_dump_reaches_every_field_that_is_not_an_escape_hatch() {
    // The dump's law (ADR 0022 decision 1): a scene must be completely reproducible from
    // it. Concretely that means every field except the ones explicitly excused.
    for state in [
        AppConfig::CLASSIFIED_FIELDS,
        PanelState::CLASSIFIED_FIELDS,
    ] {
        for field in state {
            let excused = matches!(
                field.category,
                StateCategory::Transient | StateCategory::Derived
            );
            assert_eq!(
                field.category.reaches_dump(),
                !excused,
                "field `{}` classified {:?}",
                field.name,
                field.category
            );
        }
    }
}

/// Every escape hatch in the shipping state structs, named. The assertion is not that
/// these are the right calls — that is review's job — but that the list is short, visible,
/// and cannot grow without somebody editing this test.
#[test]
fn the_escape_hatches_are_exactly_these() {
    let hatches: Vec<(&str, StateCategory)> = AppConfig::CLASSIFIED_FIELDS
        .iter()
        .chain(PanelState::CLASSIFIED_FIELDS)
        .filter(|field| {
            matches!(
                field.category,
                StateCategory::Transient | StateCategory::Derived
            )
        })
        .map(|field| (field.name, field.category))
        .collect();

    assert_eq!(
        hatches,
        [
            // A function of the scene and its density, recomputed at every rebuild.
            ("voxel_cap_warning_millions", StateCategory::Derived),
            // The camera target, rounded to whole blocks, refreshed each frame.
            ("point_add_position_blocks", StateCategory::Derived),
        ],
        "the set of fields excused from both persistence artifacts changed"
    );

    // Nothing is `transient` yet, and that is the healthy state: `derived` makes a claim a
    // reviewer can falsify, `transient` makes one nobody can.
    assert!(!hatches
        .iter()
        .any(|(_, category)| *category == StateCategory::Transient));
}

#[test]
fn the_pan_target_that_started_this_is_classified() {
    // `orbit_target` is the field whose absence from the repro dump is the reason the
    // scheme exists. It is view state: in the dump, out of the document.
    assert_eq!(
        AppConfig::category_of("orbit_target"),
        Some(StateCategory::View)
    );
    assert!(AppConfig::category_of("orbit_target")
        .expect("classified above")
        .reaches_dump());
}
