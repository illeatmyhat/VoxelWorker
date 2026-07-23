//! The workspace: the app's information architecture.
//!
//! This is the layout the six forked UX concepts converged on (the owner's claude.ai/design
//! project "VoxelWorker — Future UX", `synthesis.html`). It replaces the prototype's single
//! right-hand sidebar + bottom palette dock with five regions:
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
//! pinned favourites are a *projection* of the drawer, not a second copy of it.
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
//! target through the selection silently retargets when the selection moves. (That is why
//! `Scene::active_node_mut` was deleted; see its tombstone in `document`'s `scene::graph`.)
//!
//! Viewer state — which mode, what is folded, where the insert cursor sits — is view state
//! and stays on `PanelState`: never serialized, never in undo history.

mod browser;
mod fold_strip;
mod inspector;
mod rail;
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
