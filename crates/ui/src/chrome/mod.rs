//! The Signal viewport chrome painters: the icon [`rail`], the floating [`notice`], and the
//! sketch [`sketch_overlay`]. Pure egui painting at shell-computed positions — the shell owns the
//! projection, hit-testing and interaction routing; these only draw and report clicks.
//!
//! The viewport's bottom-left corner belongs to the [`notice`], which is only there when it has
//! something to say. Nothing persistent stands over the drawing: the viewer mode is on the top
//! bar, and the extent and the density are inspector facts.

mod notice;
mod rail;
mod sketch_overlay;

pub use notice::viewport_notice;
pub use rail::{icon_rail, orbit_type_button_rect, rail_height, rail_rect, rail_top, RailClick};
pub use sketch_overlay::{
    sketch_arc_curves, sketch_constraint_badges, sketch_draw_preview, sketch_exit_control,
    sketch_insert_marker, sketch_marquee_band, sketch_segment_lines, sketch_vertex_handles,
    ConstraintBadge, SKETCH_CONSTRAINT_BADGE, SKETCH_CONSTRAINT_BADGE_OFFSET,
    SKETCH_HANDLE_GRAB_PAD, SKETCH_HANDLE_HALF, SKETCH_INSERT_MARKER_HALF, SKETCH_SEGMENT_GRAB_PAD,
};
