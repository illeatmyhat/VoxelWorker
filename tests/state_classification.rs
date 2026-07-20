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
//!
//! Since ADR 0024 there is a third, and it is the one with a scar behind it: that every
//! field classified as reaching the dump has an actual route into it. The compiler
//! delivers that on `AppConfig`, whose captures destructure exhaustively, and cannot
//! deliver it on `PanelState`, which is read by hand — so four panel fields spent a
//! release promising the dump and reaching nothing. That gap is now a test rather than an
//! amendment describing a gap.

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

/// The two objects that are classified whole and carried as a hand-picked subset, named
/// so that being a subset stays a deliberate act.
///
/// `PanelState::geometry` and `PanelState::layer_range` are each classified as ONE view
/// object, and each reaches `AppConfig` as some of its fields — the density out of
/// `GeometryParams`, the three sticky preferences out of `LayerRange`. The band bounds are
/// a defensible omission (they re-derive against the live grid); the density mirror's
/// siblings are the inspector's in-progress edit, which is genuinely momentary. The point
/// is not that these calls are wrong, it is that a subset must be *declared* — otherwise
/// nothing distinguishes a defensible omission from the pan-target kind.
const CARRIED_AS_A_SUBSET: &[(&str, &[&str])] = &[
    ("geometry", &["voxels_per_block"]),
    (
        "layer_range",
        &["snap_to_blocks", "onion_skin", "onion_depth"],
    ),
];

/// **The seam that lost four fields for a release.**
///
/// `src/artifacts.rs` destructures `AppConfig` exhaustively, so a classified field there
/// cannot fail to reach an artifact — that is a compile error. Nothing gives `PanelState`
/// the same treatment: `AppConfig::capture` reads the panel field by field, by hand,
/// exactly the way the capture that lost the pan target did. The compiler can only check
/// that a panel field is *decided*; whether the decision is *honoured* is this test's job,
/// and before ADR 0024 nothing did it at all.
///
/// What that cost: `view_mode`, `stack`, `debug_face_orientation` and `debug_brick_faces`
/// were classified `view` — which by the derive's own error text means they reach the dump
/// — and were hard-coded to defaults in `to_panel_state`, captured by nobody. A category
/// promising one thing while the code did another, silently, because no test asked.
///
/// So this asks. Every `PanelState` field whose category reaches the dump must have a
/// route there: a same-named `AppConfig` field, or membership in
/// [`CARRIED_AS_A_SUBSET`] above. Adding a dump-reaching panel field without one now fails
/// here rather than in somebody's unreproducible repro.
#[test]
fn every_dump_reaching_panel_field_has_a_route_into_the_dump() {
    for field in PanelState::CLASSIFIED_FIELDS {
        if !field.category.reaches_dump() {
            continue;
        }
        if let Some((_, carried)) = CARRIED_AS_A_SUBSET
            .iter()
            .find(|(object, _)| *object == field.name)
        {
            for piece in *carried {
                assert!(
                    AppConfig::category_of(piece).is_some(),
                    "`{piece}` is declared as the piece of `{}` that reaches the dump, but \
                     `AppConfig` has no such field",
                    field.name
                );
            }
            continue;
        }
        assert!(
            AppConfig::category_of(field.name).is_some(),
            "`{}` is classified {:?}, which reaches the dump, but `AppConfig` carries no \
             field of that name and it is not declared as carried in a subset. This is the \
             defect ADR 0024 closed, recurring: either route the field, or declare the \
             subset in `CARRIED_AS_A_SUBSET`, or classify it as something that does not \
             claim to persist",
            field.name,
            field.category
        );
    }
}

/// The categories agree across the seam. A field routed under one name must not be view
/// state on one side and settings on the other — the two structs would then disagree about
/// what a shared project file may contain, which is the one boundary that is a routing
/// decision rather than a matter of meaning.
#[test]
fn a_field_carried_across_the_seam_keeps_its_category() {
    for field in PanelState::CLASSIFIED_FIELDS {
        let Some(config_category) = AppConfig::category_of(field.name) else {
            continue;
        };
        assert_eq!(
            config_category, field.category,
            "`{}` is classified {:?} on `PanelState` and {:?} on `AppConfig`",
            field.name, field.category, config_category
        );
    }
}

/// The session category (ADR 0024), pinned the way the document set is: by naming its
/// membership, so that a field joining or leaving it costs a deliberate edit here.
///
/// These four are the ones ADR 0018 decision 3 and issue #88 kept out of persistence
/// entirely, on a reading of "not document state" that the owner has since narrowed back
/// to what it says. They are in the dump and out of the document — the browser's bargain,
/// which is the whole content of the category.
#[test]
fn the_session_is_the_workspace_and_nothing_else() {
    let session_fields: Vec<&str> = AppConfig::CLASSIFIED_FIELDS
        .iter()
        .filter(|field| field.category == StateCategory::Session)
        .map(|field| field.name)
        .collect();
    assert_eq!(
        session_fields,
        [
            "view_mode",
            "stack",
            "debug_face_orientation",
            "debug_brick_faces"
        ],
        "the session set changed: a field joining it now survives relaunch, and a field \
         leaving it stops surviving one"
    );

    // The same four, classified the same way on the struct they are captured from. The
    // preceding test proves they have a route; this proves both ends call it the same
    // thing.
    for name in session_fields {
        assert_eq!(
            PanelState::category_of(name),
            Some(StateCategory::Session),
            "`{name}` is session state on `AppConfig` but not on `PanelState`"
        );
    }
}

/// Session state reaches the dump and never the document. Stated against the category
/// itself rather than against the field list, because this is the property that makes it a
/// sibling of `view` rather than of `document` — a viewer mode inside a shared project file
/// would impose one person's session on everyone who opened it, which is precisely what
/// ADR 0018 decision 3 was protecting and what ADR 0024 leaves untouched.
#[test]
fn session_state_is_dumped_and_never_documented() {
    assert!(StateCategory::Session.reaches_dump());
    assert!(!StateCategory::Session.reaches_document());
}
