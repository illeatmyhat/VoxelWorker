//! Onion-skin ghost parameters (issue #12; ADR 0012 — the volumetric fog subsystem
//! was retired, the ghost pass on the live display paths replaces it).

use super::*;

/// The recentred-Z spans of one onion frame, derived by `AppCore::onion_fog_params`:
/// the onion-band Z range (the ghosted layers OUTSIDE the solid band) and the solid
/// band Z range. Both display paths (brick raymarch + cuboid mesh) select their ghost
/// slabs from these edges — Z-up, layers are Z-slices.
#[derive(Debug, Clone, Copy)]
pub struct OnionFogParams {
    /// Inverse camera view-projection (to unproject screen → world rays).
    pub inverse_view_projection: glam::Mat4,
    /// Inscribed semi-axes (= grid_dimensions / 2); maps world → normalised grid.
    pub semi_axes: [f32; 3],
    /// World-space Z extent of the onion band (the ghosted layers).
    pub onion_z_min: f32,
    pub onion_z_max: f32,
    /// World-space Z extent of the displayed solid band (the opaque voxel pass drew it).
    pub band_z_min: f32,
    pub band_z_max: f32,
}

/// The onion tint hue (cool blue-grey), matching the retired volumetric fog haze so
/// the crisp ghost reads as the same "context around the band" the fog conveyed.
const ONION_FOG_COLOR_HEX: u32 = 0x9c_b4_d8;

/// ADR 0012 (H1) — the **onion ghost tint**: the flat translucent colour both display
/// paths (brick raymarch + cuboid mesh) shade the onion-slab ghost voxels with. The hue
/// matches the retired volumetric fog haze ([`ONION_FOG_COLOR_HEX`]) so the crisp ghost
/// reads as the same "context around the band" the fog conveyed; the alpha is the src-alpha
/// the ghost pass blends with (depth-tested `Less`, depth write ON — see the ghost
/// pipelines in `brick/raymarch.rs` / `mesh/pipeline.rs`). Linear-space RGB, matching the
/// linear shading both cuboid + brick shaders work in.
const ONION_GHOST_ALPHA: f32 = 0.5;

/// The onion ghost tint as linear `[r, g, b, a]` (ADR 0012 H1). Both display paths read
/// this ONE constant so the raymarch ghost and the mesh ghost tint identically — the
/// cross-path golden parity (`brick_golden_matches_dense`) depends on it.
pub fn onion_ghost_tint() -> [f32; 4] {
    let [r, g, b] = srgb_hex_to_linear(ONION_FOG_COLOR_HEX);
    [r, g, b, ONION_GHOST_ALPHA]
}
