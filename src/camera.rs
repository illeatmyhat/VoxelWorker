//! Domain seam for the orbit camera rig (see the viewing/projection chapter of the
//! architecture docs).
//!
//! The viewing and projective geometry — the [`OrbitCamera`] rig and its control
//! math, the projection matrices, the Autodesk ViewCube model, eased snaps, framing
//! fit, and screen→ray unprojection — now lives in the wgpu-free `camera` crate
//! (`crates/camera`), a sibling of `substrate`. This module is the thin re-export
//! that keeps the application's call sites (`main.rs`, `app_core.rs`, `renderer.rs`,
//! `panel.rs`, `settings.rs`, …) importing `crate::camera::*` unchanged. The
//! winit/egui input handling and the domain palette vocabulary stay in the app; only
//! the pure math moved.
//!
//! The leading `::` disambiguates the extern `camera` crate from this module path.

pub use ::camera::*;
