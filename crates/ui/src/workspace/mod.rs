//! The workspace: the app's information architecture.
//!
//! Five regions:
//!
//! ```text
//!   +-----------------------------------------------------------------+
//!   | TOP BAR   brand · breadcrumbs · viewer modes · readouts       42 |
//!   +------+---------+------------------------------------+-----------+
//!   | RAIL | BROWSER |                                    | INSPECTOR |
//!   |   54 |     222 |            VIEWPORT                |       318 |
//!   |      |         |                                    |           |
//!   +------+---------+------------------------------------+-----------+
//!   | FOLD STRIP   the ordered fold, as cards, with the cursor    166 |
//!   +-----------------------------------------------------------------+
//! ```
//!
//! ## Why these regions, and not others
//!
//! The load-bearing rule is **closed sets pin, open sets browse**. A set that cannot grow —
//! the shapes, the tools — is a permanent rail the user builds muscle memory against. A set
//! that grows with the project — materials, saved parts — is summoned and searched, so it
//! belongs in the drawer rather than on screen forever. Nothing appears in both places; the
//! pinned favorites are a *projection* of the drawer, not a second copy of it.
//!
//! The right column is read top-to-bottom as orientation → verbs → height: the view cube,
//! then the icon rail beneath it, then the layer ladder. Those float over the viewport and
//! are drawn by the shell (`signal_chrome`), not here, because they must render identically
//! on the windowed surface and in the headless capture.
//!
//! ## What this module may and may not do
//!
//! Every document mutation leaves as an [`Intent`](document::intent::Intent) on the returned
//! [`PanelResponse`] — the architecture's third law, *one door for change*. A region NEVER
//! mutates the scene, and never edits "the active node": it takes the target
//! [`NodeId`](document::scene::NodeId) explicitly, because an edit that resolves its own
//! target through the selection silently retargets when the selection moves. The document
//! carries no selection at all, so there is no `active node` for a region to reach for.
//!
//! Viewer state — which mode, what is folded, where the insert cursor sits — is view state
//! and stays on `PanelState`: never serialized, never in undo history.

#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod browser;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod fold_strip;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod inspector;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod rail;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod top_bar;

use crate::palette::BlockPalette;
use crate::panel::{ExportPanelState, PanelResponse, PanelState};
use crate::theme;

/// Top bar height (design points) — brand, breadcrumbs, viewer segment, readouts.
const TOP_BAR_HEIGHT: f32 = 42.0;
/// The pinned rail: shapes and tools, the two sets that cannot grow.
const RAIL_WIDTH: f32 = 54.0;
/// The browser: the scene's parts, definitions and sketches.
const BROWSER_WIDTH: f32 = 222.0;
/// The inspector: the selected node's own dialog.
const INSPECTOR_WIDTH: f32 = 318.0;
/// The fold strip: the ordered fold as a flat row of cards.
const FOLD_STRIP_HEIGHT: f32 = 166.0;

/// Lay the workspace out into the root [`egui::Ui`] and return what the user changed.
///
/// Region order is load-bearing: egui panels claim space in the order they are shown, so the
/// top bar and fold strip take full width, then the side columns divide what is left, and
/// whatever remains is the viewport. The caller reads that leftover with
/// `available_rect_before_wrap` after this returns.
pub fn build_workspace(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    palette: &BlockPalette,
) -> PanelResponse {
    let mut response = PanelResponse::default();

    // Full-width bands first, so the side columns sit between them rather than beside them.
    top_bar::build_top_bar(root_ui, state, &mut response);
    fold_strip::build_fold_strip(root_ui, state, &mut response);

    // Then the columns, outside-in on the left so the rail hugs the window edge.
    rail::build_rail(root_ui, state, &mut response);
    browser::build_browser(root_ui, state, &mut response);
    inspector::build_inspector(root_ui, state, export, palette, &mut response);

    response
}

/// A region's chrome: opaque near-black with a hairline on the edge that faces the viewport.
///
/// Opaque rather than the mock's convenience alpha — the approved screenshots' solid look is
/// the law, because a translucent panel over a textured voxel scene stops reading as an
/// instrument surface.
fn region_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme::BG)
        .inner_margin(egui::Margin::ZERO)
}

/// Paint a 1 px hairline along one edge of `rect`.
fn hairline(painter: &egui::Painter, rect: egui::Rect, edge: Edge, color: egui::Color32) {
    let (a, b) = match edge {
        Edge::Top => (rect.left_top(), rect.right_top()),
        Edge::Bottom => (rect.left_bottom(), rect.right_bottom()),
        Edge::Left => (rect.left_top(), rect.left_bottom()),
        Edge::Right => (rect.right_top(), rect.right_bottom()),
    };
    painter.line_segment([a, b], egui::Stroke::new(1.0_f32, color));
}

/// Which edge of a region a hairline sits on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}
