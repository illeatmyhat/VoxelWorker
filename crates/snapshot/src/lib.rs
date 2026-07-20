//! # snapshot — the vocabulary of classified application state
//!
//! State is lost when nothing forces the question to be asked. The camera's orbit
//! target went missing from the F9 repro dump exactly that way: nobody decided a
//! panned view did not matter, the decision simply never came up, because a capture
//! function that reads the fields it happens to know about has no place where the
//! omission shows. ADR 0022 answers that with two mechanisms that are deliberately
//! not one mechanism:
//!
//! * **Completeness** comes from exhaustive destructuring — a capture that binds
//!   every field with no `..` rest pattern stops compiling the day a field is added.
//! * **Classification** comes from this crate — [`StateCategory`] recorded at the
//!   field, where a reader looking at the field will actually find it.
//!
//! The second is why the derive exists at all. It is not the safety mechanism, and
//! pretending otherwise would oversell it. `#[snapshot(transient)]` sitting on the
//! field is *visible in review*; the same decision buried as a `skipped:` line inside
//! a hand-written `classify` function is not, and ADR 0022 is explicit that review is
//! the only thing keeping `transient` honest.
//!
//! ## Classification does not recurse
//!
//! A category is attached to a **whole object** and says nothing about that object's
//! own fields, by design (ADR 0022, amendment 2026-07-20). Classifying
//! `camera: OrbitCamera` as [`StateCategory::View`] finishes the job: serialization
//! already carries every field inside it, so the level-down version of the pan-target
//! bug cannot occur. That bug was never "a field was added and not serialized" — it
//! was that the state was never *reached* by the capture, which is precisely what
//! top-level exhaustive destructuring prevents. Annotating nested fields would be
//! enormous and would buy nothing.
//!
//! ## The two escape hatches, and why only one of them is safe
//!
//! [`StateCategory::Transient`] and [`StateCategory::Derived`] both reach neither
//! artifact, but they are not equally trustworthy. `Transient` asserts something
//! unfalsifiable — "this genuinely does not need to survive" — and marking a field
//! transient will always be the cheapest way to make the compiler stop complaining,
//! so ADR 0022 names it as the scheme's most likely rot. `Derived` carries a
//! *checkable* claim (ADR 0023): reconstructible from classified state alone, such
//! that dropping it changes how long something takes and nothing else. A reviewer can
//! ask "reconstructible from what?" and get either a real answer or a real bug. Prefer
//! `Derived` whenever the claim is actually true.
//!
//! ## What this crate does not cover
//!
//! Only fields of classified structs. Anything in a `static`, a thread-local, or on
//! the GPU is outside every struct the scheme can see. The guarantee is "no field of a
//! classified struct is forgotten", never "nothing is forgotten".

pub use snapshot_derive::Snapshot;

/// Which persistence artifacts a piece of application state reaches (ADR 0022
/// decision 4, extended with [`Derived`](Self::Derived) by ADR 0023).
///
/// The categories are about *destination*, not about data type or lifetime. Two fields
/// of the same Rust type routinely land in different categories — the scene's density
/// is document truth while the inspector slider mirroring it is view state — so the
/// category can only ever be authored, never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateCategory {
    /// A user preference that outlives any one project: window size, the projection
    /// mode, whether the view cube is drawn. Reaches the dump (which must reproduce a
    /// scene *completely*, settings included) but never the document, because a
    /// preference travelling inside a shared file would impose one user's setup on
    /// everyone who opens it.
    Settings,
    /// What the model **is** — the thing a user saves, shares, and reopens. Reaches
    /// the document, and therefore also the dump, the dump being a superset rather
    /// than a variant.
    Document,
    /// Where *you* are working rather than what the model is: the layer scrubber, the
    /// rollback cursor, the camera. Reaches the dump and stays out of the document, so
    /// reopening a project always shows the complete model and a rollback someone else
    /// left set is never what a collaborator sees.
    View,
    /// How the workspace was arranged when you left it: which viewer mode was active,
    /// which panels were folded, which diagnostic overlays were on. Reaches the dump and
    /// stays out of the document, the same destinations [`View`](Self::View) has — the
    /// distinction is meaning, not routing, and it is the same kind of distinction that
    /// already separates [`Settings`](Self::Settings) from `View`.
    ///
    /// The line against `View` is *what the state is about*: view state answers "where
    /// was the camera", session state answers "what was the workspace doing". The line
    /// against `Settings` is **preference versus circumstance** — a setting is something
    /// the user chose and would want honoured in every project, whereas session state is
    /// merely where they happened to leave things. Restoring a session is the browser's
    /// bargain (ADR 0024): your tabs come back, and nobody calls that a preference.
    Session,
    /// Reaches neither artifact, on the grounds that it is genuinely momentary —
    /// whether the mouse is currently held mid-drag, a warning computed fresh each
    /// frame. **The unfalsifiable escape hatch.** It must be justified in review and
    /// never used to silence the compiler; see this module's header.
    Transient,
    /// Reaches neither artifact because it is reconstructible from classified state
    /// alone. The admission test (ADR 0023) is falsifiable and should be applied
    /// literally: **dropping the field must change how long something takes and
    /// nothing else.** A cache that cannot be rebuilt from classified state is not
    /// derived — it is undeclared truth, and the compiler is right to demand a real
    /// classification for it.
    Derived,
}

impl StateCategory {
    /// The spelling accepted inside `#[snapshot(...)]`, and the one used in
    /// diagnostics. Kept as the single source of truth for the vocabulary so the
    /// derive's error text and this enum can never name different category sets.
    pub const fn as_str(self) -> &'static str {
        match self {
            StateCategory::Settings => "settings",
            StateCategory::Document => "document",
            StateCategory::View => "view",
            StateCategory::Session => "session",
            StateCategory::Transient => "transient",
            StateCategory::Derived => "derived",
        }
    }

    /// Whether state in this category is written to the **document** — the shared,
    /// versioned project artifact. Only [`Document`](Self::Document) is.
    pub const fn reaches_document(self) -> bool {
        matches!(self, StateCategory::Document)
    }

    /// Whether state in this category is written to the **dump** — the unversioned
    /// debugging artifact from which a scene must be completely reproducible. Settings,
    /// document, view and session state all are; the two escape hatches are not.
    pub const fn reaches_dump(self) -> bool {
        matches!(
            self,
            StateCategory::Settings
                | StateCategory::Document
                | StateCategory::View
                | StateCategory::Session
        )
    }
}

/// One field's classification, as recorded by `#[derive(Snapshot)]`.
///
/// Deliberately a flat name/category pair rather than anything type-aware: the
/// classification is about where an object goes, and walking into its type would be
/// the recursion ADR 0022's amendment rules out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifiedField {
    /// The field's identifier as written in the struct.
    pub name: &'static str,
    /// The artifacts that field reaches.
    pub category: StateCategory,
}

/// Implemented by `#[derive(Snapshot)]` for a struct whose every field has been
/// classified.
///
/// The trait carries no capture behaviour on purpose. Splitting `AppConfig` into a
/// document, a dump and a settings artifact is a separate and much larger job (ADR
/// 0022's stated largest implementation cost); what this trait provides today is the
/// classification table those artifacts will be built against, plus the fact that
/// producing it at all forced every field to be decided.
pub trait Snapshot {
    /// Every field of the struct, in declaration order, with its category. Exhaustive
    /// by construction — the derive builds it from the field list, so a field cannot be
    /// absent here without also being absent from the struct.
    const CLASSIFIED_FIELDS: &'static [ClassifiedField];

    /// The category of a named field, or `None` if the struct has no such field.
    /// A linear scan: these tables are tens of entries and are read by tests and
    /// tooling, never in a frame.
    fn category_of(field_name: &str) -> Option<StateCategory> {
        Self::CLASSIFIED_FIELDS
            .iter()
            .find(|field| field.name == field_name)
            .map(|field| field.category)
    }

    /// The fields reaching the document, in declaration order.
    fn document_fields() -> Vec<ClassifiedField> {
        Self::CLASSIFIED_FIELDS
            .iter()
            .copied()
            .filter(|field| field.category.reaches_document())
            .collect()
    }

    /// The fields reaching the dump, in declaration order.
    fn dump_fields() -> Vec<ClassifiedField> {
        Self::CLASSIFIED_FIELDS
            .iter()
            .copied()
            .filter(|field| field.category.reaches_dump())
            .collect()
    }
}
