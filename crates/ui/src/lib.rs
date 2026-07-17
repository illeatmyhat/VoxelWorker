//! The egui control surface: the inspector panel + the block-palette UI state.
//!
//! This crate is the app's **UI layer** — the egui widgets that read the authored
//! document and describe the user's edits. It is the panel slice of the Shell layer in
//! `docs/architecture/README.md` ("window, input, panel UI, camera, persistence"): the
//! right-hand inspector ([`panel`]) that shows the scene graph, the shape/size/density
//! controls, the layer scrubber, and the bottom block-palette dock, plus the UI-facing
//! [`palette`] state (tiles + status + click counter) those widgets read.
//!
//! ## The law: the control surface links egui, never wgpu
//!
//! The panel emits every document mutation as an [`Intent`](document::intent::Intent)
//! rather than touching the scene itself — the architecture's third law, *one door for
//! change*: the shell applies the returned intents through the single intent boundary.
//! The panel never draws the 3D scene and never names a GPU resource — the fourth law,
//! *the CPU owns truth; the GPU owns the frame*: rendering the voxels is `display`'s job,
//! and the palette thumbnails arrive here as already-registered [`egui::TextureId`]s the
//! shell hands in. So this crate links `egui` and the domain crates it reads
//! (`document`, `voxel_core`, `camera`) and NOTHING below or beside it — never wgpu,
//! egui-wgpu, `display`, `evaluation`, `work`, or `interchange`. That no-wgpu law is what
//! earns it its own crate: a widget here cannot accidentally reach into a render pipeline,
//! because the crate cannot name one. The GPU/asset half of the palette (the thumbnail
//! renderer, the texture keep-alives, the scanned block groups) stays in the shell's
//! `PaletteHost`, which drives this state. See
//! `docs/design/per-layer-crates-extraction-map.md` (the ui row) for provenance.

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference is deliberate and stays a navigable link under `--document-private-items`.
// The CI doc gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod palette;
pub mod panel;
pub mod signal_theme;
