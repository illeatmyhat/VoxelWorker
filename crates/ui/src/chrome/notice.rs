//! The floating viewport notice — a refusal, drawn where the author is looking.

use egui::{Color32, FontId, Id, LayerId, Order, Pos2, Rect, Stroke, StrokeKind, TextFormat, Vec2};

use crate::theme;

/// Horizontal / vertical padding inside the box, and its inset from the viewport corner.
const PAD_X: f32 = 9.0;
const PAD_Y: f32 = 6.0;
const INSET: f32 = 10.0;

/// Draw `REFUSED · <why>` as a bordered box in the viewport's bottom-left.
///
/// A refusal is the one thing on screen the author has to act on, so it belongs where they are
/// looking rather than in the top bar beside the passive readouts, where it goes unread. Nothing
/// else stands in that corner, so it is empty until something has to be said.
///
/// `viewport_rect` is the central 3D rect in egui points. Foreground `layer_painter` at an
/// absolute position, so it renders on `shot`.
pub fn viewport_notice(ui: &egui::Ui, viewport_rect: Rect, why: &str) {
    let mono = FontId::monospace(10.0);
    let format_with = |color: Color32| TextFormat {
        font_id: mono.clone(),
        color,
        ..Default::default()
    };

    let mut job = egui::text::LayoutJob::default();
    job.append("REFUSED", 0.0, format_with(theme::WARN));
    job.append("  ·  ", 0.0, format_with(theme::BORDER));
    job.append(why, 0.0, format_with(theme::TEXT_PRIMARY));

    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("signal_viewport_notice"),
    ));
    let galley = painter.layout_job(job);
    let size = galley.size() + Vec2::new(PAD_X * 2.0, PAD_Y * 2.0);
    let corner = Pos2::new(
        viewport_rect.left() + INSET,
        viewport_rect.bottom() - size.y - 6.0,
    );
    let box_rect = Rect::from_min_size(corner, size);
    painter.rect_filled(box_rect, 3.0, theme::BG_FLOAT);
    painter.rect_stroke(
        box_rect,
        3.0,
        Stroke::new(1.0_f32, theme::WARN),
        StrokeKind::Inside,
    );
    painter.galley(
        corner + Vec2::new(PAD_X, PAD_Y),
        galley,
        theme::TEXT_PRIMARY,
    );
}
