//! What the classification scheme promises, tested at both ends: that a classified
//! struct reports every one of its fields with the category that was authored on it,
//! and that an UNclassified field cannot be compiled at all.
//!
//! The second half is the one that matters, and it is why `trybuild` is here. The
//! derive's whole contribution is a diagnostic (ADR 0022's amendment: the macro buys
//! reviewability, not safety), so the diagnostic is pinned as a fixture. A test that
//! only checked "unclassified fields fail" would pass just as happily against a bare
//! `error: unknown attribute`, which is exactly the outcome the fixture exists to stop.

use snapshot::{ClassifiedField, Snapshot, StateCategory};

/// A stand-in for a real state struct, holding one field of each category so the table
/// can be checked whole rather than a category at a time.
#[derive(Snapshot)]
#[allow(dead_code)]
struct ExampleState {
    #[snapshot(document)]
    body: u32,
    #[snapshot(settings)]
    window_size: [u32; 2],
    #[snapshot(view)]
    rollback_cursor: Option<usize>,
    #[snapshot(transient)]
    mouse_is_held: bool,
    #[snapshot(derived)]
    cached_mesh_triangle_count: usize,
}

#[test]
fn every_field_appears_in_declaration_order_with_its_authored_category() {
    assert_eq!(
        ExampleState::CLASSIFIED_FIELDS,
        &[
            ClassifiedField {
                name: "body",
                category: StateCategory::Document
            },
            ClassifiedField {
                name: "window_size",
                category: StateCategory::Settings
            },
            ClassifiedField {
                name: "rollback_cursor",
                category: StateCategory::View
            },
            ClassifiedField {
                name: "mouse_is_held",
                category: StateCategory::Transient
            },
            ClassifiedField {
                name: "cached_mesh_triangle_count",
                category: StateCategory::Derived
            },
        ]
    );
}

#[test]
fn the_document_carries_only_document_state_and_the_dump_is_its_superset() {
    let document: Vec<&str> = ExampleState::document_fields()
        .iter()
        .map(|field| field.name)
        .collect();
    assert_eq!(document, ["body"]);

    let dump: Vec<&str> = ExampleState::dump_fields()
        .iter()
        .map(|field| field.name)
        .collect();
    // ADR 0022 decision 1: the dump is a SUPERSET of the document, not a variant.
    assert_eq!(dump, ["body", "window_size", "rollback_cursor"]);
    assert!(document.iter().all(|name| dump.contains(name)));
}

#[test]
fn the_two_escape_hatches_reach_nothing() {
    for category in [StateCategory::Transient, StateCategory::Derived] {
        assert!(!category.reaches_document(), "{category:?}");
        assert!(!category.reaches_dump(), "{category:?}");
    }
}

#[test]
fn a_missing_field_is_none_rather_than_a_panic() {
    assert_eq!(
        ExampleState::category_of("rollback_cursor"),
        Some(StateCategory::View)
    );
    assert_eq!(ExampleState::category_of("no_such_field"), None);
}

/// The derive keeps its own copy of the category spellings (a proc-macro crate cannot
/// depend on the crate that depends on it), so the two lists are pinned together here.
/// Adding a category to one side and not the other would otherwise be found only by
/// whoever next tried to use it.
#[test]
fn category_vocabulary_matches_the_derive() {
    let spellings: Vec<&str> = [
        StateCategory::Settings,
        StateCategory::Document,
        StateCategory::View,
        StateCategory::Transient,
        StateCategory::Derived,
    ]
    .iter()
    .map(|category| category.as_str())
    .collect();
    assert_eq!(
        spellings,
        ["settings", "document", "view", "transient", "derived"]
    );
    // Each spelling round-trips through the derive: if the derive did not accept one of
    // these words, the `ExampleState` above would not have compiled.
}

/// The compile-FAIL fixtures. `TRYBUILD=overwrite cargo test -p snapshot` regenerates
/// the `.stderr` files after a deliberate change to a diagnostic — and the resulting
/// diff is the review of that change, which is the point.
#[test]
fn unclassified_and_misclassified_fields_do_not_compile() {
    let harness = trybuild::TestCases::new();
    harness.compile_fail("tests/compile_fail/*.rs");
}
