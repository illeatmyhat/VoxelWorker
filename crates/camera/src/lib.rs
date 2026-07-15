//! # camera — viewing and projective geometry (wgpu-free)
//!
//! This crate holds the graphics **mathematics** of the viewer: the orbit camera
//! rig and its control math, the perspective/orthographic projection matrices, the
//! Autodesk ViewCube orientation model with its chrome and eased snaps, framing
//! fit, screen-point → world ray unprojection, and Gribb–Hartmann frustum culling.
//! It is a sibling of `substrate` — a read-first library of well-known concepts
//! under their literature names — but graphics-specific rather than pure CS.
//!
//! ## The graphics-crate boundary law
//!
//! A component belongs here if and only if it is describable entirely in the
//! vocabulary of *viewing and projective geometry* — a spherical orbit, a look-at
//! frame, a projection matrix, a view frustum, an unprojection — parameterised
//! only by plain numbers and `glam` vectors, **never** by wgpu, winit, egui, or any
//! domain type (a scene, a producer, a chunk). The dependency edge is one-way:
//! `substrate ← camera ← the application`. The only non-`glam` dependency is
//! `substrate`, for the shared [`substrate::spatial::Ray`] the unprojection produces; this
//! crate never depends on the sibling `raycast` crate that consumes those rays.
//!
//! The WGSL shaders that draw the scene are maintained *mirrors* of this CPU math,
//! so this crate is the readable specification of those shaders; the app's parity
//! suite is the mechanical link that keeps the two byte-identical.
//!
//! ## Naming rule and citations
//!
//! Each module names the concept it implements and cites the canonical literature
//! in its own module documentation (definition-of-done):
//!
//! * [`orbit`] — the spherical orbit rig and its arcball-family control math
//!   (Shoemake, "ARCBALL", *Graphics Interface* 1992, lineage), pole-continuous up
//!   and roll, and bounding-box framing fit.
//! * [`projection`] — look-at + perspective/orthographic projection and inverse-VP
//!   unprojection (Akenine-Möller, Haines & Hoffman, *Real-Time Rendering*).
//! * [`view_cube`] — the Autodesk ViewCube 26-orientation model, its screen chrome,
//!   and the pure zone→action dispatch.
//! * [`tween`] — eased angle interpolation (`easeInOutQuad`) and angle normalisation.
//! * [`frustum`] — Gribb–Hartmann plane extraction (2001) + Ericson positive-vertex
//!   AABB culling (2005), over substrate's closed continuous [`substrate::spatial::RealAabb`]
//!   (co-located there beside its half-open integer twin [`substrate::spatial::LatticeAabb`]).

pub mod frustum;
pub mod orbit;
pub mod projection;
pub mod tween;
pub mod view_cube;

pub use frustum::{Frustum, RealAabb};
pub use orbit::{HomeView, OrbitCamera, ProjectionMode, POLE_EPSILON};
pub use projection::unproject_screen_point_to_ray;
pub use tween::{ease_in_out_quad, nearest_equivalent_theta, normalize_roll, SnapTween};
pub use view_cube::{
    adjacent_face, chrome_zone_left_click_action, classify_cube_point, ArrowDir, ChromeClickAction,
    CubeChromeZone, CubeFace, CubeRect, RollDir, ViewCubeElement, CUBE_FACES,
};
