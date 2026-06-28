//! Headless orchestrator owning scene + store + camera — the AppCore keystone.
//!
//! ADR 0003 (foundation rework). `AppCore` is the headless half of the app:
//! it owns the scene data layer, the `store`, and the camera, and exposes
//! query/rebuild getters that both binaries drive. `WindowedState` becomes a
//! thin shell (winit/egui/surface + GPU renderers fed from `AppCore` data);
//! `bin/shot` re-points at the same `AppCore` in **A3**, at which point the
//! golden net finally tests the real app instead of a parallel render copy.
//!
//! Currently an empty placeholder created in slice **A2a**. `AppCore::new` +
//! headless query getters land in **A2d**; resolve state + the borrow-sensitive
//! `AppCore::rebuild` in **A2e**; `render` reads all headless data from here in
//! **A2f**.
