//! The per-frame pipeline, split out of the shell root (ADR 0031): the egui pass
//! ([`egui_frame`]) and the GPU viewport pass ([`render`]).

pub mod egui_frame;
pub mod render;
