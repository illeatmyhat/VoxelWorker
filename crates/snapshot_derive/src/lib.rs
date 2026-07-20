//! # snapshot_derive — `#[derive(Snapshot)]`
//!
//! The workspace's first proc-macro crate, and it exists for a reason worth stating
//! plainly because it is not the obvious one: **this macro does not make persistence
//! safe.** Hand-written exhaustive destructuring catches a forgotten field just as
//! well, and ADR 0022 says so outright. What the macro buys is that the decision is
//! recorded *at the field*, in review-visible form. `#[snapshot(transient)]` on the
//! line above a field is something a reviewer trips over; the same decision expressed
//! as an omission inside somebody's `capture` function is invisible, and ADR 0022
//! identifies review as the only force keeping `transient` honest.
//!
//! Everything here follows from that. The generated code is a classification table and
//! nothing more — no capture, no serialization, no traversal of field types. The one
//! piece of real machinery is the diagnostic in `unclassified_field_error`, which is
//! the feature's actual user interface: for almost every developer, the only encounter
//! with this crate will be adding a field and reading what the compiler says back.
//!
//! Deliberately absent: any inference. There is no rule by which a field's type could
//! suggest its category — the scene's density is document truth while the inspector
//! slider mirroring it is view state, and both are `u32`. Defaulting to *anything*
//! would recreate the failure the scheme exists to prevent, because the moment an
//! unclassified field silently means something, nothing forces the question to be
//! asked. So the attribute is mandatory, with no default and no struct-level fallback.

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput, Field, Fields};

/// The categories, spelled exactly as they are accepted inside `#[snapshot(...)]`,
/// paired with the variant they map to and the one-line gloss the error message shows.
///
/// This table is duplicated from `snapshot::StateCategory` rather than shared, because
/// a proc-macro crate cannot depend on the crate that depends on it. The duplication is
/// load-bearing-free: a category present here but absent there fails to compile in the
/// generated code, and the `category_vocabulary_matches_snapshot_crate` test in
/// `crates/snapshot` pins the two lists together.
const CATEGORIES: &[(&str, &str, &str)] = &[
    (
        "settings",
        "Settings",
        "a user preference outliving any one project; reaches the dump, not the document",
    ),
    (
        "document",
        "Document",
        "what the model IS; reaches the document, and so also the dump",
    ),
    (
        "view",
        "View",
        "where you are working rather than what the model is; reaches the dump, not the document",
    ),
    (
        "session",
        "Session",
        "how the workspace was left, not what the user prefers; reaches the dump, not the document",
    ),
    (
        "transient",
        "Transient",
        "genuinely momentary; reaches neither artifact — justify it in review",
    ),
    (
        "derived",
        "Derived",
        "reconstructible from classified state alone; reaches neither artifact",
    ),
];

/// Derive the classification table for a struct of application state (ADR 0022).
///
/// Every field must carry exactly one `#[snapshot(<category>)]` attribute; a field
/// without one is a compile error naming the field and listing the categories, and a
/// field with two is a compile error too, since "which of these did you mean" has no
/// safe answer. Only structs with named fields are supported: a tuple struct's fields
/// have no names to record a decision against, and an enum's variants are alternatives
/// rather than state that must all be accounted for.
///
/// Classification does not recurse. The category describes the field's whole object;
/// serialization carries what is inside it (ADR 0022 amendment 2026-07-20).
#[proc_macro_derive(Snapshot, attributes(snapshot))]
pub fn derive_snapshot(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let named_fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "`#[derive(Snapshot)]` needs named fields: a classification is recorded \
                     against a field's name, and a tuple struct has none",
                ))
            }
        },
        Data::Enum(data) => {
            return Err(syn::Error::new(
                data.enum_token.span(),
                "`#[derive(Snapshot)]` applies to structs, not enums: the scheme accounts for \
                 every field of a state struct, whereas an enum's variants are alternatives. \
                 Classify the field that HOLDS this enum instead",
            ))
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span(),
                "`#[derive(Snapshot)]` applies to structs, not unions",
            ))
        }
    };

    // Collect every field's verdict before returning, so a struct with four
    // unclassified fields reports four errors in one build rather than making the
    // developer rediscover the same message four times.
    let mut entries = Vec::new();
    let mut errors: Option<syn::Error> = None;
    for field in named_fields {
        match classify(field) {
            Ok(entry) => entries.push(entry),
            Err(error) => match &mut errors {
                Some(accumulated) => accumulated.combine(error),
                None => errors = Some(error),
            },
        }
    }
    if let Some(errors) = errors {
        return Err(errors);
    }

    let type_name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let rows = entries.iter().map(|(name, variant)| {
        let variant = syn::Ident::new(variant, proc_macro2::Span::call_site());
        quote! {
            ::snapshot::ClassifiedField {
                name: #name,
                category: ::snapshot::StateCategory::#variant,
            }
        }
    });

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::snapshot::Snapshot for #type_name #type_generics #where_clause {
            const CLASSIFIED_FIELDS: &'static [::snapshot::ClassifiedField] = &[ #(#rows),* ];
        }
    })
}

/// One field's `(name, StateCategory variant)` row, or the diagnostic explaining why
/// it has none.
fn classify(field: &Field) -> syn::Result<(String, &'static str)> {
    let field_name = field
        .ident
        .as_ref()
        .expect("named fields were checked by the caller")
        .to_string();

    let mut found: Option<(&'static str, proc_macro2::Span)> = None;
    for attribute in field.attrs.iter().filter(|a| a.path().is_ident("snapshot")) {
        let category = parse_category(attribute)?;
        if let Some((previous, _)) = found {
            return Err(syn::Error::new(
                attribute.span(),
                format!(
                    "field `{field_name}` is classified twice. A field reaches one set of \
                     artifacts, so keep whichever of `{}` / `{}` is true and delete the other",
                    CATEGORIES
                        .iter()
                        .find(|(_, variant, _)| *variant == previous)
                        .map(|(spelling, _, _)| *spelling)
                        .unwrap_or(previous),
                    CATEGORIES
                        .iter()
                        .find(|(_, variant, _)| *variant == category.0)
                        .map(|(spelling, _, _)| *spelling)
                        .unwrap_or(category.0),
                ),
            ));
        }
        found = Some(category);
    }

    match found {
        Some((variant, _)) => Ok((field_name, variant)),
        None => Err(unclassified_field_error(field, &field_name)),
    }
}

/// Read `#[snapshot(<category>)]`, rejecting anything that is not exactly one known
/// category word.
fn parse_category(attribute: &syn::Attribute) -> syn::Result<(&'static str, proc_macro2::Span)> {
    let list = attribute.meta.require_list().map_err(|_| {
        syn::Error::new(
            attribute.span(),
            format!(
                "`#[snapshot]` takes a category in parentheses, one of: {}",
                category_list()
            ),
        )
    })?;
    let word: syn::Ident = list.parse_args().map_err(|_| {
        syn::Error::new(
            list.tokens.span(),
            format!(
                "`#[snapshot(...)]` takes exactly one category, one of: {}",
                category_list()
            ),
        )
    })?;
    let spelling = word.to_string();
    CATEGORIES
        .iter()
        .find(|(name, _, _)| *name == spelling)
        .map(|(_, variant, _)| (*variant, word.span()))
        .ok_or_else(|| {
            syn::Error::new(
                word.span(),
                format!("`{spelling}` is not a state category. Valid: {}", category_list()),
            )
        })
}

fn category_list() -> String {
    CATEGORIES
        .iter()
        .map(|(name, _, _)| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The diagnostic for the case this whole crate exists to produce.
///
/// It is written long on purpose. A developer meets it while adding a field, which is
/// exactly the moment the decision is cheapest to make and the moment they are least
/// likely to go and read an ADR — so the message carries the choice, the consequence of
/// each option, the warning about the escape hatches, and the one rule (no recursion)
/// that stops people from over-applying the scheme. The pointer to ADR 0022 is last,
/// for the reader who wants the argument rather than the answer.
fn unclassified_field_error(field: &Field, field_name: &str) -> syn::Error {
    let mut message = format!(
        "field `{field_name}` is not classified: every field of application state must say \
         which persistence artifacts it reaches, so that nothing is ever left out silently.\n\n\
         Add one of:\n\n"
    );
    for (spelling, _, gloss) in CATEGORIES {
        // Pad to the longest spelling so the glosses line up into a readable column;
        // rustc indents the whole block, and ragged left edges make five options look
        // like five unrelated sentences.
        let declaration = format!("#[snapshot({spelling})]");
        message.push_str(&format!("    {declaration:<22} — {gloss}\n"));
    }
    message.push_str(
        "\n`transient` and `derived` reach neither artifact and are the two ways to get this \
         error to go away without deciding anything, so they are the ones review looks at. \
         `derived` at least makes a checkable claim: dropping the field must change how long \
         something takes and NOTHING else.\n\n\
         The category describes this field's WHOLE object and does not recurse into it — \
         serialization already carries what is inside. See docs/adr/0022.",
    );
    syn::Error::new(field.span(), message)
}
