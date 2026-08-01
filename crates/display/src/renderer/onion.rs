//! Onion-skin ghost parameters shared by the display paths.

use super::*;

/// The recentered-Z spans of one onion frame.
///
/// The spans identify the ghosted layers outside the solid band and the solid band itself. Both
/// display paths use these edges in the Z-up voxel frame.
#[derive(Debug, Clone, Copy)]
pub struct OnionFogParams {
    /// Inverse camera view-projection (to unproject screen → world rays).
    pub inverse_view_projection: glam::Mat4,
    /// Inscribed semi-axes (= `grid_dimensions` / 2); maps world → normalized grid.
    pub semi_axes: [f32; 3],
    /// World-space Z extent of the onion band (the ghosted layers).
    pub onion_z_min: f32,
    pub onion_z_max: f32,
    /// World-space Z extent of the displayed solid band (the opaque voxel pass drew it).
    pub band_z_min: f32,
    pub band_z_max: f32,
}

/// The onion tint hue shared by both display paths.
const ONION_FOG_COLOR_HEX: u32 = 0x9c_b4_d8;

/// The onion ghost's fixed alpha, used as the flat translucent
/// blend the CUBOID MESH path shades its onion-slab ghost with (depth-tested `Less`,
/// depth write ON — the ghost pipeline in `mesh/pipeline.rs`). The BRICK RAYMARCH
/// path shades its ghost differently since the Beer-Lambert haze spike:
/// `fragment_ghost_haze` (`brick/raymarch.rs`) computes its own opacity from the ray's
/// accumulated in-solid thickness and ignores this constant outright, reading only the
/// tint's RGB below (depth write OFF there — the haze march folds a whole slab into one
/// fragment, so there is no intra-slab overlap for a depth write to disambiguate).
/// Linear-space, matching the linear shading both cuboid + brick shaders work in.
const ONION_GHOST_ALPHA: f32 = 0.5;

/// The onion ghost tint as linear `[r, g, b, a]`.
///
/// Both display paths use this constant, so raymarch haze and mesh ghost share one hue
/// — but NOT the same alpha (the raymarch path discards `a` for its own computed haze
/// opacity, see `ONION_GHOST_ALPHA`'s doc above), so the two paths' onion aesthetics
/// legitimately differ (haze vs crisp). The cross-path golden parity
/// (`brick_golden_matches_dense`) depends only on the shared hue, not on identical tinting.
pub fn onion_ghost_tint() -> [f32; 4] {
    let [r, g, b] = srgb_hex_to_linear(ONION_FOG_COLOR_HEX);
    [r, g, b, ONION_GHOST_ALPHA]
}
