//! **How far away a block stops being worth authoring at.**
//!
//! A placement tool needs a bound: an armed preview whose picked point falls on the ground
//! plane at a grazing angle would otherwise land at the horizon, and a click would drop a node
//! hundreds of blocks from anything. The bound also gives the viewport an honest "you are too
//! far out to place anything" state, which a purely relative rule could never produce.
//!
//! ## The bound is perceptual, not pixel-based
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
//! 3. Their ratio, `0.3 / 25`, is [`MIN_BLOCK_SCREEN_FRACTION`].
//!
//! So the implementable rule carries no pixels: *a block must span at least 1/80 of the
//! viewport's height*. A higher-resolution display gives the same bound, correctly — the block
//! is sharper, not bigger.
//!
//! **Two honest caveats, because an earlier draft of this comment overstated the result.**
//!
//! The 25° does not *cancel* — it is **absorbed into a nominal constant**, exactly as the W3C
//! reference pixel absorbs its own nominal 28-inch viewing distance (`1px` is *defined* as a
//! visual angle of 0.0213°, which is the standardised version of this whole argument). Across a
//! laptop-to-27-inch spread the real figure runs 20–30°, so 1/80 is really 1/65 to 1/100. The
//! rule is display-*independent* only in the sense that it fixes an assumption rather than
//! measuring one.
//!
//! And **0.3° is a chosen floor, not a derived one**. It is ~14 CSS pixels — below every
//! published accessibility minimum (WCAG 2.5.8 is 24 CSS px ≈ 0.51°, i.e. 1/50 of viewport
//! height) and about six times above the ~3 arcminute point where mouse pointing measurably
//! degrades (Hourcade & Bullock-Rest, CHI 2012). It is defensible as "you can still author
//! against this", not as "this is comfortable". If a citable number is ever wanted instead,
//! WCAG's 1/50 is the one to take. See `docs/design/placement-prior-art.md`.
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
/// `1/80`, from a ~0.3° block against a nominal ~25° display (see the module docs, which also
/// record that both figures are chosen rather than derived). A bare fraction because the
/// display's angular size is absorbed into it, which is what frees it of pixels and resolution
/// — the same move the W3C reference pixel makes with its nominal viewing distance.
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

    /// The world length that occupies a fixed `screen_fraction` of the viewport height when
    /// placed at `anchor` — the size a screen-stable manipulator (the transform gizmos) must
    /// use so it holds constant on screen through any zoom. Perspective scales it by the
    /// depth-to-anchor; orthographic ignores depth, its extent already tracking `orbit_distance`.
    pub fn screen_stable_size(&self, anchor: glam::Vec3, screen_fraction: f32) -> f32 {
        let forward = -self.direction();
        let depth = (anchor - self.eye()).dot(forward);
        screen_fraction * self.view_extent_at_depth(depth)
    }

    /// The model matrix that seats a **unit-space** manipulator gizmo at `anchor` and sizes it
    /// to [`screen_stable_size`](Self::screen_stable_size): `translate(anchor) · scale(size)`.
    /// Left-multiply by the view-projection to draw any screen-stable gizmo; a gizmo that also
    /// carries an orientation composes its own rotation onto this.
    pub fn screen_stable_model(&self, anchor: glam::Vec3, screen_fraction: f32) -> glam::Mat4 {
        let size = self.screen_stable_size(anchor, screen_fraction);
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(size),
            glam::Quat::IDENTITY,
            anchor,
        )
    }

    /// A view-projection for a screen-stable gizmo at `anchor` whose near/far bracket the
    /// gizmo itself. Zoomed far out, a screen-stable gizmo's *world* size grows with depth, so
    /// the scene's own tight near/far window (sized to the model) would clip its axes — this
    /// keeps the same eye and frustum but widens the depth range to `anchor ± size`. Meant for a
    /// depth-test-OFF overlay pass, where its private depth range disturbs nothing else.
    pub fn screen_stable_view_projection(
        &self,
        aspect_ratio: f32,
        anchor: glam::Vec3,
        screen_fraction: f32,
    ) -> glam::Mat4 {
        // Axis tips sit one `size` from the anchor; a margin covers the diagonal + slack.
        let radius = self.screen_stable_size(anchor, screen_fraction) * 1.3;
        self.view_projection(aspect_ratio, anchor, radius)
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

    /// The whole point of a screen-stable gizmo: its apparent size (world size ÷ view extent at
    /// its depth) is the requested fraction at *every* zoom, in both projections. A gizmo sized
    /// in world units instead would fail this — it would grow as you zoom in.
    #[test]
    fn a_screen_stable_size_holds_its_fraction_across_zoom() {
        const FRACTION: f32 = 0.15;
        let anchor = glam::Vec3::ZERO; // the default target, so depth == orbit_distance
        for mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            for orbit_distance in [5.0, 50.0, 500.0] {
                let probe = camera(mode, orbit_distance);
                let size = probe.screen_stable_size(anchor, FRACTION);
                let apparent = size / probe.view_extent_at_depth(orbit_distance);
                assert!(
                    (apparent - FRACTION).abs() < 1e-4,
                    "{mode:?} at distance {orbit_distance}: apparent {apparent}, expected {FRACTION}"
                );
            }
        }
    }

    /// Whether a world point survives near/far clipping under `vp` — clip-space z within
    /// `[0, w]` (glam's `*_rh` matrices use the wgpu [0,1] depth range).
    fn within_depth(vp: glam::Mat4, point: glam::Vec3) -> bool {
        let clip = vp * point.extend(1.0);
        clip.w > 0.0 && (0.0..=clip.w).contains(&clip.z)
    }

    /// The regression for the far-zoom cut-off: at a large orbit distance a screen-stable gizmo
    /// is big in world units, so its axis tip falls OUTSIDE the scene's tight near/far window —
    /// [`screen_stable_view_projection`](OrbitCamera::screen_stable_view_projection) widens the
    /// depth range to keep it whole.
    #[test]
    fn the_gizmo_projection_keeps_a_far_zoomed_gizmo_off_the_clip_planes() {
        const FRACTION: f32 = 0.16;
        let anchor = glam::Vec3::ZERO;
        for mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            let probe = camera(mode, 4096.0);
            let size = probe.screen_stable_size(anchor, FRACTION);
            let axis_tip = anchor + glam::Vec3::Z * size;
            // The scene's own matrix, sized to a small (~5-block) model, clips the tip.
            let scene_vp = probe.view_projection(1.6, anchor, 45.0);
            assert!(!within_depth(scene_vp, axis_tip), "{mode:?}: scene matrix should clip the tip");
            // The gizmo's own matrix brackets it.
            let gizmo_vp = probe.screen_stable_view_projection(1.6, anchor, FRACTION);
            assert!(within_depth(gizmo_vp, axis_tip), "{mode:?}: gizmo matrix must keep the tip");
        }
    }
}
