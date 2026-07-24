//! # The persistence artifacts (ADR 0022, extended by ADR 0024)
//!
//! One structure used to serve as config, project and debug repro at once, and the cost
//! of that was concrete: the camera's orbit target went missing from the F9 dump for a
//! release, not because anyone judged a panned view unimportant but because a single
//! capture function has nowhere for an omission to show. This module is the other half
//! of the answer to that. `crates/snapshot` records, at each field, *which* artifacts it
//! reaches; what follows is the code that actually carries it there, written so that the
//! compiler refuses a field nobody routed.
//!
//! ## What each one is for
//!
//! * [`DocumentArtifact`] is what the model **is** — today, the scene and nothing else.
//!   It is the thing that would be shared and reopened, so a preference or a scrubber
//!   position travelling inside it would impose one person's session on everyone.
//! * [`SettingsArtifact`] is preference that outlives any one project: the window size,
//!   the projection, the Home view the user deliberately kept.
//! * [`ViewArtifact`] is where the author was looking from: the camera pose, the layer
//!   band, the density mirror.
//! * [`SessionArtifact`] is how they had the workspace arranged while looking (ADR 0024):
//!   the viewer mode, the folded panels, the diagnostic overlays. It shares the view's
//!   destinations exactly — dump yes, document no — and is a separate type because the
//!   *question* differs, which is the same reason settings and view are separate types
//!   despite also routing identically. Categories here record meaning; only the document
//!   boundary is a routing decision.
//! * [`Dump`] is the **superset** — every setting, every input, every piece of view
//!   state, because its defining property is that a scene is completely reproducible
//!   from it. It is what F9 writes and what `shot --from-config` replays.
//!
//! The dump being a superset rather than a variant is load-bearing here: it means the
//! reachability guarantee needs to be enforced in exactly one place. A field that
//! reaches the dump has reached an artifact; a field the dump's capture does not mention
//! has reached nothing, and that is the case the compiler must reject.
//!
//! ## How the guarantee is delivered
//!
//! Every capture in this module destructures [`AppConfig`] with **no `..` rest
//! pattern**. Adding a field to that struct therefore fails to compile in
//! [`Dump::from_state`] and in [`DocumentArtifact::from_state`] until somebody says
//! where it goes — `error[E0027]: pattern does not mention field`. The derive proves a
//! field is *classified*; only this destructuring proves the classification was
//! *honoured*, which is the distinction ADR 0022's second amendment had to make after
//! the derive landed alone.
//!
//! The document's capture binds the fields it declines with `field_name: _`, rather than
//! reaching for `..`. That is deliberate and is most of this module's review value: a
//! new field forces the question "does this belong in a shared project file?" to be
//! answered in writing, at the one place where answering "no" is the easy default and
//! therefore the dangerous one.
//!
//! ## What is on disk, and why it is one file and not three
//!
//! The dump is the only artifact written today. F9 writes one to the temp directory, and
//! exit writes one to the platform config path — because restoring a session needs the
//! scene *and* the preferences *and* the camera pose, which is the dump's field set
//! exactly. Splitting that into separate files would be a user-facing save/open workflow,
//! and this is a structural split, not a product feature. The document and the settings
//! are real, separately serializable values that a dump is *composed of*, which is what
//! makes the composition checkable; giving each one its own path is a later decision that
//! nothing here forecloses.
//!
//! The on-disk JSON stays flat — the three parts are merged into one object by
//! [`Dump::to_json`] and each read back from the whole of it — so an existing config and
//! an existing repro file both still load. That is not back-compat for its own sake, since
//! the standing rule is that old configs may break; it is that `shot --from-config` reads
//! dumps written by other builds, and a gratuitous shape change would strand every repro
//! file anyone has already saved for no gain at all.

use serde::{Deserialize, Serialize};

use camera::ProjectionMode;
use document::scene::Scene;
use ui::panel::{SignalStackState, SketchTool, ViewMode};
use voxel_core::core_geom::MaterialChoice;

use crate::settings::{AppConfig, PlacementGhostConfig};

/// serde remote-derive shim for the `camera` crate's [`ProjectionMode`], which carries no
/// serde dependency of its own (the graphics-crate boundary law keeps it to glam +
/// substrate). It mirrors the enum's two unit variants so [`SettingsArtifact`] can persist
/// the projection choice. The on-disk representation ("Perspective" / "Orthographic") is
/// unchanged from when this lived on `AppConfig`.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ProjectionMode")]
enum ProjectionModeConfig {
    Perspective,
    Orthographic,
}

/// The same shim for the `ui` crate's [`ViewMode`]. `ui` links egui and the domain crates
/// and no serde (ADR 0016's crate law), so the viewer mode is persisted from out here,
/// exactly as the projection is. Mirrors all three variants; a dump naming a variant this
/// build does not have fails its part's deserialize and falls back to the default, which
/// is the same tolerance every other key gets.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ViewMode")]
enum ViewModeConfig {
    Normal,
    OnionFog,
    ShowBooleans,
}

/// The same shim for the `ui` crate's [`SketchTool`] (ADR 0028, #95) — the armed sketch-mode
/// verb. `ui` carries no serde (the crate law), so the tool is persisted from out here exactly
/// as the viewer mode is. Mirrors all three variants; a dump naming a variant this build lacks
/// falls back to the default (`Select`), the same tolerance every other key gets.
#[derive(Serialize, Deserialize)]
#[serde(remote = "SketchTool")]
enum SketchToolConfig {
    Select,
    AddPoint,
    Delete,
}

/// And for the Signal display stack's fold state. A struct rather than an enum, which
/// changes nothing about why the shim is needed: the type lives in a crate that cannot
/// name serde, and the four flags are what "classified as one object, saved whole" means
/// in practice — none of them is annotated, and all four travel.
#[derive(Serialize, Deserialize)]
#[serde(remote = "SignalStackState")]
struct SignalStackStateConfig {
    folded: bool,
    viewport_open: bool,
    onion_open: bool,
    grids_open: bool,
}

/// What the model **is**: the project a user would save, share and reopen.
///
/// One field today, and that is the point rather than an embarrassment — ADR 0022's first
/// decision is that a shared file carries the model and nothing about where somebody was
/// working, and the smallest honest document is the strongest statement of it. The type
/// exists so that "is this document state?" is a question with a place to be answered,
/// and so that the day a second field earns `#[snapshot(document)]` the addition is a
/// visible edit here rather than a silent widening of what travels between people.
///
/// This is the artifact that will need versioning when it acquires a file of its own: it
/// crosses between people and across releases, which is exactly the property the dump
/// does not have.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DocumentArtifact {
    /// The whole assembly — node tree, reusable definitions, the active selection, and
    /// the document-level density (ADR 0003 §3f(0)). `None` means "no scene was
    /// persisted", which the load path answers with the default seed scene rather than
    /// an empty document.
    #[serde(default)]
    pub scene: Option<Scene>,
}

/// Preference that outlives any one project.
///
/// The membership test that decides this struct is not "does it change rarely" but "would
/// a collaborator want it imposed on them" — which is why the Home view is here while the
/// live orbit pose is view state next door. A Home view is a viewpoint the user chose to
/// *keep*; an orbit angle is merely where they last happened to be looking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingsArtifact {
    /// Perspective or orthographic. Persisted through the remote-derive shim above,
    /// because `camera` carries no serde dependency.
    #[serde(default, with = "ProjectionModeConfig")]
    pub projection_mode: ProjectionMode,
    /// The procedural material the viewport shades with.
    #[serde(default)]
    pub material: MaterialChoice,
    /// Whether the Points' axes draw on top of the model vs occluded (ADR 0031).
    #[serde(default = "default_true")]
    pub axes_on_top: bool,
    /// The applied VS block's **label** only. Re-resolving its texture on load is heavy
    /// (a folder scan plus JSON resolution), so the label is restored best-effort for the
    /// readout and the material reverts to procedural until the user re-applies. An
    /// intentional lazy re-apply, recorded here because it looks like a bug otherwise.
    #[serde(default)]
    pub applied_block_label: Option<String>,
    /// The Home button's saved orbit angle.
    #[serde(default = "default_theta")]
    pub home_theta: f32,
    /// The Home button's saved elevation.
    #[serde(default = "default_phi")]
    pub home_phi: f32,
    /// The Home button's saved distance, honoured only when [`home_explicit`] is set.
    ///
    /// [`home_explicit`]: Self::home_explicit
    #[serde(default = "default_distance")]
    pub home_distance: f32,
    /// Did the user actually press "set home"? When `false`, Home re-frames the model
    /// instead of restoring [`home_distance`](Self::home_distance), so a home nobody set
    /// never zooms in too close.
    #[serde(default)]
    pub home_explicit: bool,
    /// The window size to restore on next launch.
    #[serde(default = "default_window_size")]
    pub window_size: [u32; 2],
}

/// Where *you* are working rather than what the model is.
///
/// Everything here is deliberately excluded from the document and deliberately present in
/// the dump. [`orbit_target`](Self::orbit_target) is the field this whole scheme was built
/// around: a panned view reproduces only if the point being orbited travels with the
/// angles, and for a while it did not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewArtifact {
    /// The inspector slider's density mirror. The document truth is
    /// `scene.voxels_per_block` (ADR 0003 §3f(0)); a mirror of document truth is view
    /// state, which is why it is here and not next door.
    #[serde(default = "default_density")]
    pub voxels_per_block: u32,
    /// Whether the layer scrubber snaps its band to whole blocks.
    #[serde(default = "default_true")]
    pub snap_to_blocks: bool,
    /// Whether the onion-skin ghost passes are drawn.
    #[serde(default)]
    pub onion_skin: bool,
    /// How many layers deep the onion skin reaches.
    #[serde(default = "default_onion_depth")]
    pub onion_depth: u32,
    /// Camera orbit angle.
    #[serde(default = "default_theta")]
    pub orbit_theta: f32,
    /// Camera elevation.
    #[serde(default = "default_phi")]
    pub orbit_phi: f32,
    /// Camera distance from the orbit target.
    #[serde(default = "default_distance")]
    pub orbit_distance: f32,
    /// The world point the camera looks at and orbits. Panning moves it off the origin,
    /// so a dump without it reframes a repro on the origin and quietly misses the very
    /// view the bug was seen at.
    #[serde(default)]
    pub orbit_target: [f32; 3],
}

/// How the workspace was left, which is neither what the model is nor what the user
/// prefers.
///
/// The browser bargain (ADR 0024): close it, open it, and your tabs come back — nobody
/// files that under preferences, and nobody expects it inside a document they share. The
/// membership test that separates this from [`SettingsArtifact`] next door is **chosen
/// versus left**: a Home view is a viewpoint the user pressed a button to keep, whereas a
/// viewer mode is simply the one they were last in.
///
/// Every field here was classified as reaching the dump and reached nothing, for a
/// release — hard-coded to a default in `AppConfig::to_panel_state` and captured by
/// nobody. That is the same shape as the pan-target bug the whole scheme was built
/// around, and it survived because `PanelState` had no exhaustive capture. This struct is
/// the route those four fields were promised; `tests/state_classification.rs` is what
/// stops the promise being broken again.
/// Its `Default` derives, where [`SettingsArtifact`]'s and [`ViewArtifact`]'s are written
/// out — the session's defaults are each type's own (Normal, the expanded stack, both
/// diagnostics off), so there is nothing to state that the field types do not already say.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SessionArtifact {
    /// The viewer's exclusive rendering mode. ADR 0018 decision 3 kept it out of the
    /// document and the implementation read that as out of persistence entirely;
    /// ADR 0024 supersedes that half.
    #[serde(default, with = "ViewModeConfig")]
    pub view_mode: ViewMode,
    /// The floating Signal display stack: folded to edge tabs, and which sections are
    /// open. Saved whole, per ADR 0022's amendment — all four flags, not a subset.
    #[serde(default = "default_signal_stack", with = "SignalStackStateConfig")]
    pub stack: SignalStackState,
    /// The face-orientation debug shading (colour by outward normal, cull off).
    #[serde(default)]
    pub debug_face_orientation: bool,
    /// The brick-raymarch grazing-rim diagnostic. A dump taken while chasing a rendering
    /// fault has to replay with the diagnostic that was revealing it, or it replays a
    /// different picture than the one the bug was seen in.
    #[serde(default)]
    pub debug_brick_faces: bool,
    /// The armed-tool placement ghost (ADR 0022), `None` when nothing is armed. A dump
    /// taken mid-gesture replays the pending drop; `PlacementGhostConfig` derives its own
    /// serde (it lives in a serde-aware crate), so no remote shim is needed here.
    #[serde(default)]
    pub placement_ghost: Option<PlacementGhostConfig>,
    /// The armed-tool placement snap settings (position + orientation, owner ruling
    /// 2026-07-21). Durable across adds and relaunch; `PlacementSnap` derives its own serde.
    #[serde(default)]
    pub placement_snap: ui::panel::PlacementSnap,
    /// The sketch node under edit in sketch mode (ADR 0028), `None` in the normal chrome. A
    /// dump taken mid-edit re-enters the same sketch; `NodeId` derives its own serde (it lives
    /// in the serde-aware `document` crate), so no remote shim is needed. The `serde(default)`
    /// degrades a pre-field dump to `None`.
    #[serde(default)]
    pub sketch_mode: Option<document::scene::NodeId>,
    /// The armed sketch-mode tool (ADR 0028, #95). Persisted through the `SketchToolConfig`
    /// remote shim (the `ui` crate carries no serde); a pre-field dump degrades to the default
    /// `Select`.
    #[serde(default, with = "SketchToolConfig")]
    pub sketch_tool: SketchTool,
}

/// The debugging artifact, and the superset: **a scene must be completely reproducible
/// from it.**
///
/// It needs no versioning, because it is read by the build that wrote it — that asymmetry
/// with the document is the whole reason they are different types. What it does need is
/// completeness, and [`from_state`](Self::from_state) is where that is enforced.
///
/// The nesting is in the type, where it carries meaning, and not in the format: on disk a
/// dump is one flat JSON object, exactly as it was before the split, so every repro file
/// and config already written still loads. [`to_json`](Self::to_json) and
/// [`from_json`](Self::from_json) do that merging by hand rather than with
/// `#[serde(flatten)]`, for a reason worth recording because it is not obvious and cost a
/// full test run to find: flatten buffers the whole object through `serde`'s internal
/// `Content` type, and the scene's id-keyed node arena does not survive that trip
/// (`i128 is not supported`). Composing `serde_json::Value`s instead keeps the parts as
/// three independent, ordinarily-derived serde types and lets the scene deserialize by the
/// normal path.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dump {
    /// What the model is.
    pub document: DocumentArtifact,
    /// The preferences in force when it was written.
    pub settings: SettingsArtifact,
    /// Where the author was working and looking from.
    pub view: ViewArtifact,
    /// How their workspace was arranged while they were.
    pub session: SessionArtifact,
}

impl DocumentArtifact {
    /// Project the document out of the classified state.
    ///
    /// Exhaustive on purpose. Every field the document declines is named and bound to `_`
    /// rather than swept up by `..`, so adding a field to [`AppConfig`] stops this
    /// function compiling until somebody decides whether it belongs in a file that
    /// travels between people. Declining is the right answer for almost everything, and
    /// that is precisely why it should cost a line.
    pub fn from_state(state: &AppConfig) -> Self {
        let AppConfig {
            scene,
            // Declined — view state. A collaborator opening this project must not
            // inherit where somebody else's inspector slider or scrubber was parked.
            voxels_per_block: _,
            snap_to_blocks: _,
            onion_skin: _,
            onion_depth: _,
            // Declined — view state. Nor should they inherit a camera pose.
            orbit_theta: _,
            orbit_phi: _,
            orbit_distance: _,
            orbit_target: _,
            // Declined — session state. Which viewer mode somebody was in, and which
            // panels they had folded, is the most obviously personal thing here: it is
            // not even a preference they chose, merely where they stopped.
            view_mode: _,
            stack: _,
            debug_face_orientation: _,
            debug_brick_faces: _,
            // Declined — session state. An armed drop is where somebody stopped, not part
            // of the model a collaborator would open.
            placement_ghost: _,
            // Declined — session state. Which sketch someone was editing is where they
            // stopped, not part of the shared model.
            sketch_mode: _,
            // Declined — session state. Which sketch tool was armed is where they stopped too.
            sketch_tool: _,
            // Declined — session/settings. One person's snap preference must not ride into a
            // shared document.
            placement_snap: _,
            // Declined — settings. A preference inside a shared file would impose one
            // person's setup on everyone who opened it.
            projection_mode: _,
            material: _,
            axes_on_top: _,
            applied_block_label: _,
            home_theta: _,
            home_phi: _,
            home_distance: _,
            home_explicit: _,
            window_size: _,
        } = state;
        Self {
            scene: scene.clone(),
        }
    }
}

impl Dump {
    /// Capture the dump from the classified state — **the reachability guarantee's one
    /// enforcement point.**
    ///
    /// The destructuring carries no `..`, so a field added to [`AppConfig`] fails the
    /// build here with `error[E0027]: pattern does not mention field`. Because the dump is
    /// a superset of every other artifact, a field that survives this function has
    /// reached persistence and a field that does not has reached nothing at all — which
    /// is exactly the property ADR 0022 decision 4 asks for, and the one a derive alone
    /// could not give.
    pub fn from_state(state: &AppConfig) -> Self {
        let AppConfig {
            scene,
            voxels_per_block,
            projection_mode,
            material,
            axes_on_top,
            applied_block_label,
            snap_to_blocks,
            onion_skin,
            onion_depth,
            orbit_theta,
            orbit_phi,
            orbit_distance,
            orbit_target,
            home_theta,
            home_phi,
            home_distance,
            home_explicit,
            window_size,
            view_mode,
            stack,
            debug_face_orientation,
            debug_brick_faces,
            placement_ghost,
            placement_snap,
            sketch_mode,
            sketch_tool,
        } = state;
        Self {
            document: DocumentArtifact {
                scene: scene.clone(),
            },
            settings: SettingsArtifact {
                projection_mode: *projection_mode,
                material: *material,
                axes_on_top: *axes_on_top,
                applied_block_label: applied_block_label.clone(),
                home_theta: *home_theta,
                home_phi: *home_phi,
                home_distance: *home_distance,
                home_explicit: *home_explicit,
                window_size: *window_size,
            },
            view: ViewArtifact {
                voxels_per_block: *voxels_per_block,
                snap_to_blocks: *snap_to_blocks,
                onion_skin: *onion_skin,
                onion_depth: *onion_depth,
                orbit_theta: *orbit_theta,
                orbit_phi: *orbit_phi,
                orbit_distance: *orbit_distance,
                orbit_target: *orbit_target,
            },
            session: SessionArtifact {
                view_mode: *view_mode,
                stack: *stack,
                debug_face_orientation: *debug_face_orientation,
                debug_brick_faces: *debug_brick_faces,
                placement_ghost: placement_ghost.clone(),
                placement_snap: *placement_snap,
                sketch_mode: *sketch_mode,
                sketch_tool: *sketch_tool,
            },
        }
    }

    /// Rebuild the classified state a dump was captured from.
    ///
    /// The struct literal is the mirror of the destructuring above and gets its
    /// completeness the same way: Rust has no `..` here either without an explicit base
    /// value, so a new [`AppConfig`] field that this cannot fill is a compile error rather
    /// than a silently-defaulted value. Capture and restore are thus forced to stay the
    /// same size, which is what makes "the dump reproduces the scene completely" a
    /// statement the build can check instead of one a reviewer has to believe.
    pub fn into_state(self) -> AppConfig {
        let Dump {
            document,
            settings,
            view,
            session,
        } = self;
        AppConfig {
            scene: document.scene,
            voxels_per_block: view.voxels_per_block,
            projection_mode: settings.projection_mode,
            material: settings.material,
            axes_on_top: settings.axes_on_top,
            applied_block_label: settings.applied_block_label,
            snap_to_blocks: view.snap_to_blocks,
            onion_skin: view.onion_skin,
            onion_depth: view.onion_depth,
            orbit_theta: view.orbit_theta,
            orbit_phi: view.orbit_phi,
            orbit_distance: view.orbit_distance,
            orbit_target: view.orbit_target,
            home_theta: settings.home_theta,
            home_phi: settings.home_phi,
            home_distance: settings.home_distance,
            home_explicit: settings.home_explicit,
            window_size: settings.window_size,
            view_mode: session.view_mode,
            stack: session.stack,
            debug_face_orientation: session.debug_face_orientation,
            debug_brick_faces: session.debug_brick_faces,
            placement_ghost: session.placement_ghost,
            placement_snap: session.placement_snap,
            sketch_mode: session.sketch_mode,
            sketch_tool: session.sketch_tool,
        }
    }

    /// Serialize to the pretty, **flat** JSON both on-disk dumps use.
    ///
    /// The three parts are merged into one object rather than nested. Their key sets are
    /// disjoint by construction — a field belongs to exactly one category — so the merge
    /// is total and order-independent, and a collision would mean a field had been routed
    /// to two artifacts at once, which the destructuring above cannot express.
    pub fn to_json(&self) -> Result<String, String> {
        let mut merged = serde_json::Map::new();
        for part in [
            serde_json::to_value(&self.document),
            serde_json::to_value(&self.settings),
            serde_json::to_value(&self.view),
            serde_json::to_value(&self.session),
        ] {
            let part = part.map_err(|error| error.to_string())?;
            let serde_json::Value::Object(fields) = part else {
                return Err("a dump's parts must each serialize to a JSON object".to_string());
            };
            merged.extend(fields);
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(merged))
            .map_err(|error| error.to_string())
    }

    /// Read a dump back out of that flat object.
    ///
    /// Each part is deserialized from the **whole** object, ignoring the keys that belong
    /// to its siblings — none of them sets `deny_unknown_fields`, and none of them may,
    /// which is also what lets a dump written by a build with extra state still load here.
    /// Every field carries a serde default, so a dump missing a key restores that field's
    /// default rather than failing: a repro from another build is worth more partially
    /// than not at all.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        use serde::Deserialize;
        let value: serde_json::Value = serde_json::from_str(text)?;
        Ok(Self {
            document: DocumentArtifact::deserialize(&value)?,
            settings: SettingsArtifact::deserialize(&value)?,
            view: ViewArtifact::deserialize(&value)?,
            session: SessionArtifact::deserialize(&value)?,
        })
    }
}

impl Default for SettingsArtifact {
    fn default() -> Self {
        Self {
            projection_mode: ProjectionMode::default(),
            material: MaterialChoice::default(),
            axes_on_top: true,
            applied_block_label: None,
            home_theta: default_theta(),
            home_phi: default_phi(),
            home_distance: default_distance(),
            home_explicit: false,
            window_size: default_window_size(),
        }
    }
}

impl Default for ViewArtifact {
    fn default() -> Self {
        Self {
            voxels_per_block: default_density(),
            snap_to_blocks: true,
            onion_skin: false,
            onion_depth: default_onion_depth(),
            orbit_theta: default_theta(),
            orbit_phi: default_phi(),
            orbit_distance: default_distance(),
            orbit_target: [0.0, 0.0, 0.0],
        }
    }
}

// The per-field serde defaults. They live here beside the artifacts that use them rather
// than beside `AppConfig`, because they describe the on-disk format's behaviour for a
// missing key. The camera ones derive from `OrbitCamera::default()` so a persisted default
// can never drift from the live one.

pub(crate) fn default_density() -> u32 {
    16
}
pub(crate) fn default_true() -> bool {
    true
}
pub(crate) fn default_theta() -> f32 {
    camera::OrbitCamera::default().orbit_theta
}
pub(crate) fn default_phi() -> f32 {
    camera::OrbitCamera::default().orbit_phi
}
pub(crate) fn default_distance() -> f32 {
    camera::OrbitCamera::default().orbit_distance
}
pub(crate) fn default_window_size() -> [u32; 2] {
    [1280, 800]
}
pub(crate) fn default_onion_depth() -> u32 {
    2
}
/// The stack's own default (expanded, every section open), not a second copy of it — a
/// persisted default can never drift from the live one, the same reason the camera
/// defaults above delegate to `OrbitCamera::default()`.
pub(crate) fn default_signal_stack() -> SignalStackState {
    SignalStackState::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use snapshot::{Snapshot, StateCategory};

    /// A state value with every field driven off its default, so a capture that dropped a
    /// field would show up as a difference rather than coinciding with the default.
    fn distinctive_state() -> AppConfig {
        AppConfig {
            scene: None,
            voxels_per_block: 24,
            projection_mode: ProjectionMode::Orthographic,
            material: MaterialChoice::Wood,
            axes_on_top: false,
            applied_block_label: Some("Granite".to_string()),
            snap_to_blocks: false,
            onion_skin: true,
            onion_depth: 5,
            orbit_theta: 1.23,
            orbit_phi: 0.95,
            orbit_distance: 42.0,
            orbit_target: [3.0, -7.5, 11.0],
            home_theta: 2.34,
            home_phi: 1.11,
            home_distance: 18.0,
            home_explicit: true,
            window_size: [1600, 900],
            // Every session field off its default too, for the same reason as the rest:
            // a capture that dropped one would otherwise coincide with what a default
            // restore produces, and pass.
            view_mode: ViewMode::ShowBooleans,
            stack: SignalStackState {
                folded: true,
                viewport_open: false,
                onion_open: true,
                grids_open: false,
            },
            debug_face_orientation: true,
            debug_brick_faces: true,
            // Off its default (Some, not None) so a capture that dropped it fails the
            // round-trip rather than coinciding with a default restore.
            placement_ghost: Some(PlacementGhostConfig {
                shape_kind: voxel_core::voxel::ShapeKind::Box,
                size_voxels: [16, 16, 16],
                wall_blocks: 1,
                offset_voxels: [7, -3, 5],
            }),
            // Off its default so a dropped capture fails the round-trip.
            placement_snap: ui::panel::PlacementSnap {
                position: ui::panel::PositionSnap::NoSnap,
                angle: ui::panel::AngleSnap::Deg15,
                pivot: ui::panel::PlacementPivot::VolumetricCenter,
            },
            // Off its default (Some, not None) so a capture that dropped it fails the
            // round-trip rather than coinciding with a default restore (ADR 0028).
            sketch_mode: Some(document::scene::NodeId(9)),
            // Off its default (Delete, not Select) for the same reason (ADR 0028, #95).
            sketch_tool: SketchTool::Delete,
        }
    }

    /// The dump's defining claim, tested rather than asserted in prose: what goes in comes
    /// back out, through the format it is actually stored in. A field that the capture
    /// reached but the restore forgot would survive the type system (both sides compile)
    /// and die here.
    #[test]
    fn a_dump_round_trips_every_field_through_json() {
        let state = distinctive_state();
        let json = Dump::from_state(&state).to_json().expect("serialise");
        let restored = Dump::from_json(&json).expect("deserialise");
        assert_eq!(restored.into_state(), state);
    }

    /// The flattening is a format decision, so it is pinned as one. If the three parts ever
    /// nested themselves into the JSON, every repro file already on disk — and every
    /// `config.json` — would silently load as defaults, which is a failure that shows up as
    /// "my scene vanished" rather than as an error.
    #[test]
    fn the_dump_is_a_flat_json_object() {
        let json = Dump::from_state(&distinctive_state())
            .to_json()
            .expect("serialise");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        let object = value.as_object().expect("a JSON object");
        for key in ["orbit_target", "window_size", "voxels_per_block", "material"] {
            assert!(
                object.contains_key(key),
                "`{key}` must sit at the top level, not inside a nested part: {json}"
            );
        }
        for absent in ["document", "settings", "view"] {
            assert!(
                !object.contains_key(absent),
                "the parts must not appear as nested objects: {json}"
            );
        }
    }

    /// The document declines everything that is not the model. Stated against the
    /// classification table rather than against a hand-written list, so the test and the
    /// annotations cannot disagree: whatever `#[snapshot(document)]` says reaches the
    /// document is what the document is built from.
    #[test]
    fn the_document_carries_exactly_the_document_classified_state() {
        let state = distinctive_state();
        let document = DocumentArtifact::from_state(&state);
        assert_eq!(document.scene, state.scene);

        let classified: Vec<&str> = AppConfig::document_fields()
            .iter()
            .map(|field| field.name)
            .collect();
        assert_eq!(
            classified,
            ["scene"],
            "the document artifact above is built from `scene` alone; a field that gained \
             `#[snapshot(document)]` must be added to it too"
        );
    }

    /// ADR 0028 (#93 acceptance): sketch mode is session/editing state, never document state —
    /// a saved document is byte-identical whether or not a sketch was being edited. The
    /// exhaustive destructure in `DocumentArtifact::from_state` already declines `sketch_mode`;
    /// this pins the property the ADR's acceptance names, from the outside.
    #[test]
    fn sketch_mode_never_reaches_the_document() {
        let mut editing = distinctive_state();
        editing.sketch_mode = Some(document::scene::NodeId(42));
        let mut idle = distinctive_state();
        idle.sketch_mode = None;
        assert_eq!(
            DocumentArtifact::from_state(&editing).scene,
            DocumentArtifact::from_state(&idle).scene,
            "the document must not depend on whether a sketch was being edited"
        );
        assert_eq!(
            AppConfig::category_of("sketch_mode"),
            Some(StateCategory::Session),
            "sketch mode is session state (reaches the dump, not the document)"
        );
    }

    /// Every field the classification promises reaches the dump has a corresponding key in
    /// the dump's JSON. This is the reachability guarantee checked from the outside — the
    /// destructuring makes forgetting a field a compile error, and this makes routing one
    /// into a part that never serializes a test failure.
    #[test]
    fn every_dump_classified_field_appears_in_the_written_json() {
        let json = Dump::from_state(&distinctive_state())
            .to_json()
            .expect("serialise");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        let object = value.as_object().expect("a JSON object");
        for field in AppConfig::CLASSIFIED_FIELDS {
            if !field.category.reaches_dump() {
                continue;
            }
            // `scene` is the one nullable field: `None` still serializes, as `null`.
            assert!(
                object.contains_key(field.name),
                "`{}` is classified {:?} — which reaches the dump — but no such key was \
                 written",
                field.name,
                field.category
            );
        }
    }

    /// A dump written by a build that did not have some field must still load. Not for
    /// back-compat's sake (old configs may break, by standing rule) but because
    /// `shot --from-config` replays files written by other builds, and a missing key
    /// falling back to its default is the difference between a usable repro and a hard
    /// failure.
    #[test]
    fn a_partial_dump_loads_with_per_field_defaults() {
        let dump = Dump::from_json(r#"{"voxels_per_block": 8}"#).expect("a partial dump parses");
        let state = dump.into_state();
        assert_eq!(state.voxels_per_block, 8);
        assert_eq!(state.window_size, default_window_size());
        assert_eq!(state.orbit_target, [0.0, 0.0, 0.0]);
        assert!(state.scene.is_none());
    }

    /// The escape hatches are not silently routed anywhere. Nothing in `AppConfig` is
    /// classified `transient` or `derived` today; if one ever is, it must not appear in the
    /// dump's JSON, and this is where that would be caught.
    #[test]
    fn escape_hatch_state_reaches_no_artifact() {
        let json = Dump::from_state(&distinctive_state())
            .to_json()
            .expect("serialise");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        let object = value.as_object().expect("a JSON object");
        for field in AppConfig::CLASSIFIED_FIELDS {
            if matches!(
                field.category,
                StateCategory::Transient | StateCategory::Derived
            ) {
                assert!(
                    !object.contains_key(field.name),
                    "`{}` reaches neither artifact but was written to the dump",
                    field.name
                );
            }
        }
    }
}
