//! The Signal status line pinned bottom-left of the viewport.

use egui::{Color32, FontId, Id, LayerId, Order, Pos2, Rect, TextFormat};

use crate::panel::ViewMode;
use crate::theme;

/// Draw `VIEWPORT <MODE> · SEL <node> · <dims> · <density> vx/blk` in faint mono, mode + selection
/// in accent, `·` in border grey. `viewport_rect` is the central 3D rect (egui points); `selection`
/// the active node name or `None` (→ `—`); `dims` the resolved grid extent (voxels); `density`
/// voxels-per-block. Foreground `layer_painter` at an absolute position, so it renders on `shot`.
pub fn status_line(
    ui: &egui::Ui,
    viewport_rect: Rect,
    view_mode: ViewMode,
    selection: Option<&str>,
    dims: [u32; 3],
    density: u32,
) {
    let mono = FontId::monospace(10.0);
    let format_with = |color: Color32| TextFormat {
        font_id: mono.clone(),
        color,
        ..Default::default()
    };
    let faint = format_with(theme::TEXT_FAINT);
    let accent = format_with(theme::ACCENT);
    let dot = format_with(theme::BORDER);

    let mut job = egui::text::LayoutJob::default();
    job.append("VIEWPORT ", 0.0, faint.clone());
    job.append(view_mode.status_label(), 0.0, accent.clone());
    job.append("  ·  ", 0.0, dot.clone());
    job.append("SEL ", 0.0, faint.clone());
    job.append(selection.unwrap_or("—"), 0.0, accent);
    job.append("  ·  ", 0.0, dot.clone());
    job.append(&format!("{}×{}×{}", dims[0], dims[1], dims[2]), 0.0, faint.clone());
    job.append("  ·  ", 0.0, dot);
    job.append(&format!("{density} vx/blk"), 0.0, faint);

    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("signal_status_line")));
    let galley = painter.layout_job(job);
    let pos = Pos2::new(
        viewport_rect.left() + 10.0,
        viewport_rect.bottom() - galley.size().y - 6.0,
    );
    painter.galley(pos, galley, theme::TEXT_FAINT);
}
