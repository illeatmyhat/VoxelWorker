//! The Signal viewport chrome painters (ADR 0018): the icon [`rail`], the [`status`] line, and the
//! sketch [`sketch_overlay`]. Pure egui painting at shell-computed positions — the shell owns the
//! projection, hit-testing and interaction routing; these only draw and report clicks.

mod rail;
mod sketch_overlay;
mod status;

pub use rail::{icon_rail, orbit_type_button_rect, rail_height, rail_rect, rail_top, RailClick};
pub use sketch_overlay::{
    sketch_draw_preview, sketch_exit_control, sketch_insert_marker, sketch_marquee_band,
    sketch_segment_lines, sketch_vertex_handles, SKETCH_HANDLE_GRAB_PAD, SKETCH_HANDLE_HALF,
    SKETCH_INSERT_MARKER_HALF, SKETCH_SEGMENT_GRAB_PAD,
};
pub use status::status_line;
