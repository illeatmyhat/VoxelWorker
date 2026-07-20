//! Tests for the glyph families as a whole.

use std::sync::mpsc;
use std::time::Duration;

use super::large::LargeIcon;
use super::Icon;

/// Every glyph in BOTH families paints across the whole range of sizes the app and the design
/// reference render at — and, more to the point, terminates.
///
/// The size sweep is the whole value of this test. [`IconPainter::dash_path`] once walked a
/// polyline by advancing a cursor, and the cursor landed exactly on a dash boundary at every
/// step; wherever rounding then put the phase one ULP short of the boundary, the next advance
/// (~1e-7) was smaller than the cursor's own precision, the cursor stopped moving, and the walk
/// span forever. It reproduced at 184 of 532 simulated size/grid combinations — including a
/// plain dashed line at 18 pt, the rail's own natural size — and it hung `design_reference` on
/// a white window with no output at all. A two-size smoke test sailed straight past it.
///
/// The work runs on its own thread behind a watchdog because the failure mode is a HANG, not a
/// wrong pixel: an assertion at the end of a stalled loop is never reached.
#[test]
fn every_glyph_paints_at_every_size_and_terminates() {
    let (painted, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let context = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(400.0, 400.0),
            )),
            ..Default::default()
        };
        let _ = context.run_ui(raw_input, |ui| {
            for size in 8..=140 {
                let rect =
                    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(size as f32));
                for icon in Icon::ALL {
                    icon.draw(ui.painter(), rect, egui::Color32::WHITE);
                }
                for large in LargeIcon::ALL {
                    large.draw(ui.painter(), rect, egui::Color32::WHITE);
                }
            }
        });
        let _ = painted.send(());
    });

    match finished.recv_timeout(Duration::from_secs(60)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("a glyph never finished painting — the dash walk is stalling again")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("a glyph panicked while painting")
        }
    }
}
