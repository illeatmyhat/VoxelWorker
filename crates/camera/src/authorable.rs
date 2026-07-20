//! **How far away a block stops being worth authoring at.**
//!
//! A placement tool needs a bound: an armed preview whose picked point falls on the ground
//! plane at a grazing angle would otherwise land at the horizon, and a click would drop a node
//! hundreds of blocks from anything. The bound also gives the viewport an honest "you are too
//! far out to place anything" state, which a purely relative rule could never produce.
//!
//! ## The bound is perceptual, and the pixel cancels out
//!
//! The naive form — "a block must cover N pixels" — is not invariant: the same scene on a 4K
//! 27-inch monitor and a 1080p laptop would bound differently, though a block *looks* the same
//! size on both. Pixels measure sharpness, not apparent size.
//!
//! The invariant quantity is the angle a block subtends **at the viewer's eye**, which is the
//! reasoning behind Apple's "retina" threshold. Running it through:
//!
//! 1. A typical desktop display subtends roughly **25°** vertically at the eye (a 27-inch at
//!    600 mm is ~31°, a 14-inch laptop at 500 mm is ~20°).
//! 2. A block needs roughly **0.3°** (~18 arcminutes) to be worth authoring against — far above
//!    the ~1 arcminute acuity limit, and about the angular size of a small UI target.
//! 3. Their ratio, `0.3 / 25`, is [`MIN_BLOCK_SCREEN_FRACTION`] — **and the 25° cancels**.
//!
//! So the implementable rule carries no pixels, no monitor size, and no viewing distance: *a
//! block must span at least 1/80 of the viewport's height*. A higher-resolution display gives
//! the same bound, correctly — the block is sharper, not bigger.
//!
//! The residue of the assumption is that 1/80 was derived for a typical desktop setup, because
//! nothing can measure the actual display or how far away the viewer sits. Someone on a
//! projector three metres back gets a bound tuned for a monitor — but that is true of every
//! UI-sizing decision ever made, so it is the standard assumption rather than a new one.
//!
//! ## One rule covers both projections
//!
//! Write `V` for the world-space distance the viewport's height spans. A block spans
//! `block_size / V` of the screen, so the rule is `V <= block_size / MIN_BLOCK_SCREEN_FRACTION`
//! in either projection.
//!
//! What differs is only how `V` is built, and — because this rig derives the orthographic
//! frustum from `orbit_distance` rather than carrying a separate zoom — **both forms are
//! `2 · orbit_distance · k`**, differing only in `k`. Orthographic zoom *is* the orbit
//! distance here. The two constants are within 1.4% of each other by deliberate tuning (so
//! toggling projection keeps the framing), so the limit is very nearly one number regardless
//! of mode.
//!
//! One genuine difference survives, and callers must respect it: under perspective `V` grows
//! with depth, so a point beyond the target is less authorable than the target itself. Under
//! orthographic `V` is the same everywhere, so depth does not enter.

use crate::orbit::{ORTHO_HALF_HEIGHT_FACTOR, PERSPECTIVE_FOV_Y};
use crate::{OrbitCamera, ProjectionMode};

/// The fraction of the viewport's height a block must span to be worth authoring against.
///
/// `1/80`, derived in the module docs from a ~0.3° block against a ~25° display. Deliberately
/// a bare fraction: it is what survives after the perceptual argument cancels the display's
/// angular size, and it is therefore free of pixels, resolution and monitor dimensions.
pub const MIN_BLOCK_SCREEN_FRACTION: f32 = 1.0 / 80.0;

/// The half-height factor the projection uses at `orbit_distance`, per mode — the `k` in
/// `V = 2 · orbit_distance · k`.
///
/// Perspective takes `tan(fov_y / 2)`; orthographic takes the frustum factor directly, because
/// this rig scales the orthographic half-height by `orbit_distance` instead of carrying an
/// independent zoom. That is what lets one rule serve both projections.
fn half_extent_factor(projection_mode: ProjectionMode) -> f32 {
    match projection_mode {
        ProjectionMode::Perspective => (PERSPECTIVE_FOV_Y * 0.5).tan(),
        ProjectionMode::Orthographic => ORTHO_HALF_HEIGHT_FACTOR,
    }
}

impl OrbitCamera {
    /// The world-space distance the viewport's height spans, at `depth` from the eye.
    ///
    /// Perspective grows with depth; orthographic ignores it entirely, which is the one place
    /// the two projections genuinely part company for this purpose.
    pub fn view_extent_at_depth(&self, depth: f32) -> f32 {
        match self.projection_mode {
            ProjectionMode::Perspective => 2.0 * depth.max(0.0) * half_extent_factor(self.projection_mode),
            ProjectionMode::Orthographic => {
                2.0 * self.orbit_distance.max(0.0) * half_extent_factor(self.projection_mode)
            }
        }
    }

    /// The world-space distance the viewport's height spans **at the target** — the depth the
    /// user is actually working at, and the one the authorability question is asked about.
    pub fn view_extent_at_target(&self) -> f32 {
        self.view_extent_at_depth(self.orbit_distance)
    }

    /// The farthest `orbit_distance` at which a block of `block_size` world units still spans
    /// [`MIN_BLOCK_SCREEN_FRACTION`] of the viewport — i.e. is still worth authoring against.
    ///
    /// Solving `2 · d · k <= block_size / fraction` for `d`. Under perspective this reads as a
    /// distance limit; under orthographic, where `orbit_distance` is the zoom, it reads as a
    /// zoom limit. Same inequality either way.
    pub fn authorable_distance_limit(&self, block_size: f32) -> f32 {
        let factor = half_extent_factor(self.projection_mode);
        if factor <= 0.0 || block_size <= 0.0 {
            return 0.0;
        }
        block_size / (2.0 * factor * MIN_BLOCK_SCREEN_FRACTION)
    }

    /// Whether anything can be authored at all from where the camera currently sits.
    ///
    /// False means the point the camera orbits is itself too far to work at, so nothing nearer
    /// the cursor can be either — the viewport should say **zoom in**, which is a different
    /// message from "point at something" and must not share its affordance.
    pub fn can_author_at_all(&self, block_size: f32) -> bool {
        self.orbit_distance <= self.authorable_distance_limit(block_size)
    }

    /// Whether a point at `depth` from the eye is close enough to author against.
    ///
    /// Under orthographic this is independent of `depth` and answers the same question as
    /// [`can_author_at_all`](Self::can_author_at_all) — correctly, since a block's apparent size
    /// there does not change with distance.
    pub fn depth_is_authorable(&self, depth: f32, block_size: f32) -> bool {
        let limit = block_size / MIN_BLOCK_SCREEN_FRACTION;
        self.view_extent_at_depth(depth) <= limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-voxel block — the document default density, and the unit the limit is quoted in.
    const BLOCK: f32 = 16.0;

    fn camera(projection_mode: ProjectionMode, orbit_distance: f32) -> OrbitCamera {
        OrbitCamera { orbit_distance, projection_mode, ..OrbitCamera::default() }
    }

    /// **The bound is one number in both projections**, because the orthographic half-height
    /// factor (0.42) was tuned to sit within 1.4% of `tan(22.5°)` so toggling projection keeps
    /// the framing. Pinning it here means a future change to either constant that breaks the
    /// agreement fails a test rather than silently making one mode place further than the other.
    #[test]
    fn the_limit_agrees_across_projections() {
        let perspective = camera(ProjectionMode::Perspective, 100.0).authorable_distance_limit(BLOCK);
        let orthographic = camera(ProjectionMode::Orthographic, 100.0).authorable_distance_limit(BLOCK);
        let relative_gap = (perspective - orthographic).abs() / perspective;
        assert!(
            relative_gap < 0.02,
            "the two projections must bound alike: {perspective} vs {orthographic}"
        );
        // ~95 blocks at the default density — sanity, not a golden.
        assert!(
            (1400.0..1700.0).contains(&perspective),
            "limit landed at {perspective}, expected roughly 1500 world units"
        );
    }

    /// At exactly the limit a block spans exactly the minimum fraction. This is the definition
    /// restated as a test, which is the point: it pins the SOLVE, and an algebra slip in
    /// `authorable_distance_limit` would break it while leaving the constants intact.
    #[test]
    fn at_the_limit_a_block_spans_exactly_the_minimum_fraction() {
        for mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            let probe = camera(mode, 1.0);
            let limit = probe.authorable_distance_limit(BLOCK);
            let at_limit = camera(mode, limit);
            let spanned = BLOCK / at_limit.view_extent_at_target();
            assert!(
                (spanned - MIN_BLOCK_SCREEN_FRACTION).abs() < 1e-4,
                "{mode:?}: block spans {spanned}, expected {MIN_BLOCK_SCREEN_FRACTION}"
            );
        }
    }

    /// Zooming out past the limit turns authoring off, and zooming in turns it back on. The
    /// "nothing placeable" viewport state hangs off exactly this.
    #[test]
    fn zooming_out_past_the_limit_disables_authoring() {
        for mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            let limit = camera(mode, 1.0).authorable_distance_limit(BLOCK);
            assert!(camera(mode, limit * 0.5).can_author_at_all(BLOCK), "{mode:?} inside the limit");
            assert!(!camera(mode, limit * 2.0).can_author_at_all(BLOCK), "{mode:?} beyond it");
        }
    }

    /// **The one place the projections genuinely differ.** Perspective's view extent grows with
    /// depth, so a point past the target is less authorable than the target; orthographic's does
    /// not, so depth is irrelevant there. A caller that forgot the distinction would silently
    /// allow placement at any depth under perspective.
    #[test]
    fn depth_matters_under_perspective_and_not_under_orthographic() {
        let limit = camera(ProjectionMode::Perspective, 1.0).authorable_distance_limit(BLOCK);
        let perspective = camera(ProjectionMode::Perspective, limit * 0.5);
        assert!(perspective.depth_is_authorable(limit * 0.5, BLOCK), "at the target");
        assert!(!perspective.depth_is_authorable(limit * 4.0, BLOCK), "far beyond the target");

        let orthographic = camera(ProjectionMode::Orthographic, limit * 0.5);
        assert!(
            orthographic.depth_is_authorable(limit * 4.0, BLOCK),
            "orthographic apparent size does not fall off with depth"
        );
    }

    /// A denser document makes a block bigger in world units, so it stays authorable further
    /// out — the limit scales linearly with block size and carries no hidden density term.
    #[test]
    fn the_limit_scales_with_block_size() {
        let probe = camera(ProjectionMode::Perspective, 1.0);
        let small = probe.authorable_distance_limit(BLOCK);
        let doubled = probe.authorable_distance_limit(BLOCK * 2.0);
        assert!((doubled - small * 2.0).abs() < 1e-2, "{doubled} should be twice {small}");
        assert_eq!(probe.authorable_distance_limit(0.0), 0.0, "a zero-size block is never authorable");
    }
}
