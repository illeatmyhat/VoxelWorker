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

/// The reachability guarantee holds at one seam and not at the other, and this pins the
/// gap so that it is a tracked fact rather than a discovery waiting to happen.
///
/// `src/artifacts.rs` destructures `AppConfig` exhaustively, so a classified field there
/// cannot fail to reach an artifact. Nothing does the same for `PanelState`:
/// `AppConfig::capture` reads the panel field by field, by hand, exactly the way the
/// capture that lost the pan target did. So `PanelState` classification is presently a
/// *statement of intent* — the compiler checks that every field is decided, and nothing
/// checks that the decision is honoured.
///
/// The four fields below are the live consequence. Each is classified `view`, which by
/// the derive's own error text means it reaches the dump; each is hard-coded to a default
/// in `to_panel_state` and captured by nobody. Two of them (`view_mode`, `stack`) were
/// deliberately excluded from persistence by ADR 0018 Decision 3 and issue #88 — so this
/// is not a bug to fix quietly, it is a contradiction between two decisions that needs an
/// owner ruling: either those fields are not `view`, or the dump is not capturing what it
/// claims to.
#[test]
fn panel_state_classification_is_not_yet_enforced_by_any_capture() {
    // Fields whose category promises the dump, and whose value the dump has no route to.
    let unreached = ["debug_face_orientation", "debug_brick_faces", "view_mode", "stack"];
    for name in unreached {
        assert_eq!(
            PanelState::category_of(name).map(|category| category.reaches_dump()),
            Some(true),
            "`{name}` is expected to be classified as reaching the dump"
        );
        assert!(
            AppConfig::category_of(name).is_none(),
            "`{name}` gained a route into the dump — delete it from this list and from the \
             gap recorded in ADR 0022"
        );
    }
}

/// The other half of the same gap, and the more surprising one.
///
/// ADR 0022's amendment states that "a classified object is saved whole", on the grounds
/// that serialization carries every field inside it. That holds only where the object
/// itself is what gets serialized. `PanelState::geometry` and `PanelState::layer_range`
/// are each classified as ONE view object, and each is carried into `AppConfig` as a
/// hand-picked subset of its fields — the density out of `GeometryParams`, the three
/// sticky preferences out of `LayerRange`. Whatever else those types hold does not travel,
/// and no destructuring anywhere says so.
///
/// The band bounds are a defensible omission (they re-derive against the live grid). The
/// point is that nothing distinguishes a defensible omission from the pan-target kind.
#[test]
fn classified_panel_objects_are_carried_as_subsets_not_whole() {
    for (object, carried) in [
        ("geometry", &["voxels_per_block"][..]),
        (
            "layer_range",
            &["snap_to_blocks", "onion_skin", "onion_depth"][..],
        ),
    ] {
        assert_eq!(
            PanelState::category_of(object),
            Some(StateCategory::View),
            "`{object}` is classified as one view object"
        );
        assert!(
            AppConfig::category_of(object).is_none(),
            "`{object}` now travels whole — this gap has closed, update ADR 0022"
        );
        for field in carried {
            assert!(
                AppConfig::category_of(field).is_some(),
                "`{field}` is the piece of `{object}` that actually reaches the dump"
            );
        }
    }
}
