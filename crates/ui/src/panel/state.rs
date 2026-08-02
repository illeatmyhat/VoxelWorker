//! The panel's mutable state ([`PanelState`], [`LayerRange`]) and the per-frame
//! [`PanelResponse`] / [`ExportPanelState`] carried between the shell and the
//! section builders.

use camera::{OrbitType, ProjectionMode};
use document::intent::{Intent, NodeSpec};
use document::scene::{NodeContent, NodeId, Scene};
use document::voxel::{GeometryParams, SdfShape};
use voxel_core::core_geom::MaterialChoice;

use super::ArmedConstraint;

/// The armed-tool **placement ghost**: the translucent analytic-SDF preview of
/// where a primitive's voxels will land, drawn without recomposing the scene ("render a
/// colored transparent SDF where the voxels will be"). Lives INSIDE [`ArmedTool`] as its
/// [`pending_drop`](ArmedTool::pending_drop) — `Some` while the armed tool is pointed at a
/// valid drop, `None` otherwise — so a ghost cannot exist without the tool that derives it.
///
/// It carries the armed [`SdfShape`] and the ABSOLUTE, corner-anchored voxel offset the
/// node would take — the SAME frame `Intent::PlaceNode { offset_voxels }` uses
/// (`src/app_core/placement.rs`). The render-frame field center the shader needs is
/// DERIVED at draw time from the live resolve's recenter via [`center_world`], keeping the
/// frame law in one place rather than baked into stored state that a later rebuild would
/// stale.
///
/// [`center_world`]: PlacementGhost::center_world
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementGhost {
    /// The armed primitive whose surface the ghost traces.
    pub shape: SdfShape,
    /// The absolute, corner-anchored voxel offset where the node would drop — a node with
    /// `offset_voxels = V` occupies absolute `[V, V + turn_extent(grid))` (the placement
    /// frame, `src/app_core/placement.rs`).
    pub offset_voxels: [i64; 3],
    /// The **sub-voxel** remainder of the corner offset — the continuous fraction a
    /// `NoSnap` drop keeps under the cursor while `offset_voxels` holds the integer floor. The
    /// committed node seats at `offset_voxels + offset_local`, so the ghost MUST carry it too or
    /// it snaps to the integer voxel while the real geometry lands a fraction off (the confusing
    /// off-by-a-few-voxels mismatch in `NoSnap` mode). Zero for Voxel / Block snap.
    pub offset_local: [f32; 3],
    /// The node's **continuous** rotation — the exact tilt the drop would apply, so
    /// the ghost previews the shape the way it will actually land (a tube tilted to a cylinder's
    /// curved radial normal, not merely the nearest of the 24 lattice turns). Identity for a
    /// world-plane or upright drop.
    pub rotation: glam::Quat,
}

impl PlacementGhost {
    /// The field center in the display's render frame — the box center of the placed node, seated
    /// through the **SAME** corner-anchored affine the classifier folds occupancy through
    /// ([`substrate::spatial::LeafPlacement`], the `LeafAffine` alias), so the ghost coincides with
    /// the solid drop BY CONSTRUCTION rather than by a kept-in-sync mirror.
    ///
    /// Seat the continuous corner `offset_voxels + offset_local` (integer floor plus the sub-voxel
    /// `NoSnap` remainder) via `LeafPlacement`, ask it where the producer-local center `full/2`
    /// lands in absolute voxels, then rebase into this rebuild's render frame by subtracting
    /// `recenter`. `full` is the EXACT grid (a half-integer half on odd axes), `recenter` the
    /// FLOORED half — the difference is the half-voxel term a naive "the shape is at the origin"
    /// drops.
    pub fn center_world(&self, recenter_voxels: [i64; 3], voxels_per_block: u32) -> [f32; 3] {
        use substrate::spatial::{LeafPlacement, ProducerLocalVoxelPoint, TrueWorldVoxelPoint};
        let grid = self.shape.grid_dimensions(voxels_per_block);
        let full = glam::Vec3::new(grid[0] as f32, grid[1] as f32, grid[2] as f32);
        // The continuous corner offset in ABSOLUTE voxels: integer floor + sub-voxel remainder.
        let world_offset = glam::Vec3::new(
            self.offset_voxels[0] as f32 + self.offset_local[0],
            self.offset_voxels[1] as f32 + self.offset_local[1],
            self.offset_voxels[2] as f32 + self.offset_local[2],
        );
        let placement = LeafPlacement::new(
            self.rotation,
            full,
            TrueWorldVoxelPoint::from_voxels(world_offset),
        );
        let center_absolute = placement
            .world_of(ProducerLocalVoxelPoint::from_voxels(full * 0.5))
            .voxels();
        let recenter = glam::Vec3::new(
            recenter_voxels[0] as f32,
            recenter_voxels[1] as f32,
            recenter_voxels[2] as f32,
        );
        (center_absolute - recenter).to_array()
    }

    /// The inscribed semi-axes in voxels (`grid/2` per axis, EXACT half) the SDF is
    /// evaluated against. These are the shape's OWN (un-turned) half-extents — the shader
    /// evaluates the field in the shape's local frame after un-turning the sample point
    /// ([`rotation_inverse_columns`](Self::rotation_inverse_columns)), so the semi-axes
    /// never turn (only the sample point does).
    pub fn semi_axes(&self, voxels_per_block: u32) -> [f32; 3] {
        self.shape
            .grid_dimensions(voxels_per_block)
            .map(|axis| axis as f32 / 2.0)
    }

    /// The **inverse** rotation as column-major `f32` columns for the shader's `mat3x3<f32>`
    /// uniform. The ghost stores the forward rotation; the shader maps a world sample
    /// back into the shape's local frame with its inverse, so `rotation_inverse · (world − center)`
    /// lands in the un-turned SDF frame. Each column is padded to a `vec4` (std140 mat3 stride);
    /// the `w` lane is unused.
    pub fn rotation_inverse_columns(&self) -> [[f32; 4]; 3] {
        // glam `Mat3` is column-major and WGSL `m * v = Σ col[j]·v[j]` is too, so column `j`
        // passes straight through. `Mat3::from_quat(rotation.inverse())` is the inverse rotation.
        let inverse = glam::Mat3::from_quat(self.rotation.inverse());
        std::array::from_fn(|column| {
            let col = inverse.col(column);
            [col.x, col.y, col.z, 0.0]
        })
    }

    /// `wall_blocks * density`, in voxels — the Tube wall thickness the SDF needs (ignored
    /// by every other kind).
    pub fn wall_voxels(&self, voxels_per_block: u32) -> f32 {
        (self.shape.wall_blocks * voxels_per_block) as f32
    }
}

/// The **armed tool** and everything it carries: the [`NodeSpec`] a stationary viewport
/// click drops, plus the pending-drop [`PlacementGhost`] the per-frame arm pass derives
/// from it. One field, nested by construction, so a dump cannot carry the ghost (the mirror)
/// without the tool (its authority) and leave frame 1 re-deriving from nothing: the ghost cannot
/// outlive, precede, or travel without its tool.
#[derive(Debug, Clone)]
pub struct ArmedTool {
    /// What a stationary click places — the authority the arm pass re-derives
    /// [`pending_drop`](Self::pending_drop) from every frame.
    pub spec: NodeSpec,
    /// The resolved drop under the current cursor (`Some` over a valid surface), or the
    /// restored drop from a dump/config until the first cursor motion re-resolves it.
    pub pending_drop: Option<PlacementGhost>,
}

/// How a placed node's **position** snaps to the lattice. A **session** setting, durable across
/// adds and relaunch, set from the armed-tool `Add <shape>` dialog. Progressively coarsens the
/// drop point from the raycast hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PositionSnap {
    /// Drop at the raycast surface hit itself, at the finest (voxel) granularity — the freest
    /// placement, the object seated exactly where the cursor points.
    NoSnap,
    /// Snap the drop so the object's grid aligns to **block** boundaries (offset a multiple of
    /// the density) — clean inter-part mating.
    Block,
    /// Snap the drop to the **voxel** lattice (whole-voxel offset). The default.
    #[default]
    Voxel,
}

/// How a placed node's **seated rotation** snaps in angle. A **session** setting like
/// [`PositionSnap`]. The node ALWAYS seats to the surface
/// normal — that part is not a choice — this only picks the angle granularity of that seated
/// rotation: exact (any angle) or quantized to 15° steps. The quantization itself is applied by
/// the placement spine (`place_primitive`), not here; this enum only names the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AngleSnap {
    /// Use the seated rotation exactly, at any angle. The default.
    #[default]
    Continuous,
    /// Quantize the seated rotation's angle to 15° steps, position-dominant.
    Deg15,
}

/// Which authoring **pivot** a placed node seats by — the continuous
/// handle the drop lands at and rotates about. A **session** setting like [`PositionSnap`]. The
/// node ALWAYS seats to the surface normal — that part is not a choice — this only picks which
/// point of the object touches the contact. Centering yields a FRACTIONAL sub-voxel offset that
/// the placement spine (`place_primitive`) carries; it is never rounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PlacementPivot {
    /// Bottom-center: the object's base rests on the contact and its centroid rides half its
    /// local height out along the normal. The default.
    #[default]
    Base,
    /// Volumetric center: the object's centroid sits on the contact, so it straddles the surface
    /// half in / half out.
    VolumetricCenter,
}

/// The armed-tool placement snap settings, read by `place_primitive` and edited by the
/// `Add <shape>` dialog. Grouped so the one seam threads a single value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PlacementSnap {
    /// How the drop point snaps to the lattice.
    pub position: PositionSnap,
    /// How the seated rotation snaps in angle.
    pub angle: AngleSnap,
    /// Which authoring pivot the drop seats by.
    pub pivot: PlacementPivot,
}

/// The viewer's exclusive rendering mode. The viewer is always in exactly one of these three;
/// the mode is **never document state** — it follows the active selection, is not saved with the
/// scene, and never enters undo history, like [`ProjectionMode`] and the other [`PanelState`]
/// display params. Sticky across selection changes; default [`Normal`](Self::Normal).
///
/// It **is** restored across relaunch, as *session* state: out of the document, into the dump.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ViewMode {
    /// The finished look: no ghosts, no band clip, anywhere.
    #[default]
    Normal,
    /// Onion fog: the selected object clips to the layer band with ghost haze outside it. The
    /// scrubber's `lower`/`upper` are object-relative over the selected object's Z extent (the
    /// shell's `AppCore::mesh_clip` derives the region-scoped clip from them); selecting the
    /// root part makes the band scene-wide.
    OnionFog,
    /// Show booleans: every Subtract/Intersect operand in the selected subtree x-rays
    /// over the finished scene. Selecting the root part covers the
    /// whole scene.
    ShowBooleans,
}

impl ViewMode {
    /// The next mode in the Signal icon rail's cycle order: Normal -> Onion fog -> Show
    /// booleans -> Normal. The viewport-mode button steps through this; it is pure display
    /// state (no
    /// rebuild, never serialized, never undone), so cycling it only re-derives the
    /// display overlays at the shell's existing mode-change seam.
    pub fn next(self) -> Self {
        match self {
            ViewMode::Normal => ViewMode::OnionFog,
            ViewMode::OnionFog => ViewMode::ShowBooleans,
            ViewMode::ShowBooleans => ViewMode::Normal,
        }
    }

    /// The UPPERCASE status-line label for this mode (the Signal status line's
    /// `VIEWPORT <MODE>` field): `NORMAL` / `ONION FOG` / `SHOW BOOLEANS`.
    pub fn status_label(self) -> &'static str {
        match self {
            ViewMode::Normal => "NORMAL",
            ViewMode::OnionFog => "ONION FOG",
            ViewMode::ShowBooleans => "SHOW BOOLEANS",
        }
    }
}

/// How the author leaves **sketch mode** — the two arms of the floating `CANCEL | FINISH
/// SKETCH` exit control.
///
/// The mode opens an undo GROUP on enter; these are the two ways it closes. `Finish` collapses
/// the session to one main-history entry, `Cancel` rolls it back to the enter-state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchExit {
    /// Commit the sketch edits — closes the undo group as one main-stack entry.
    Finish,
    /// Discard the sketch edits — rolls the undo group back to the enter-state.
    Cancel,
}

/// What the viewport context menu asked of the **orbit center** — the pivot Shift+MMB turns
/// about.
///
/// The two menu items are the entire set of things that may move it. Every other camera verb
/// (pan, zoom, the view cube, the explicit orbit mode) operates on `camera.target` instead, and
/// keeping the two apart is the point: a pan slides the view across the model while the feature
/// you are inspecting stays the feature you turn around.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrbitCenterRequest {
    /// ARM a placement: the center starts following the cursor, drawn as its own gizmo, and
    /// the next viewport click commits it. Deliberately not "place it at the right-clicked
    /// point" — the user would not see where it landed until the menu had already closed.
    Place,
    /// Send it back to the world origin.
    Reset,
}

/// How the user ended a **running modal command** from the viewport context menu.
///
/// While a modal command is up, the viewport menu is REPLACED by this pair — there is no third
/// choice, because a menu that offered unrelated verbs mid-command would be offering to start a
/// second one. This is the general seam every modal command reports through, not an orbit-mode
/// detail.
///
/// The two are distinct in general: `Accept` keeps what the command produced, `Cancel` discards
/// it. A command with nothing pending to discard — the explicit orbit mode is one; navigating IS
/// the result, and it has already happened — simply ends on either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeCommand {
    /// Keep what the command produced and end it.
    Accept,
    /// Discard what the command produced and end it.
    Cancel,
}

/// The armed **sketch-mode tool** — which direct-manipulation verb a viewport click performs
/// while a sketch is being edited. Delete is an ACTION on the selection (Delete key / context
/// menu), not a tool.
///
/// **Session** state on the same footing as [`PanelState::armed_tool`] and
/// [`PanelState::sketch_mode`]: which tool was armed is how the workspace was left, never
/// document state, and it rides into the dump so a mid-edit repro re-enters the mode with the
/// same tool in hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SketchTool {
    /// Select / move a profile vertex — press a handle and drag it on the plane. The default,
    /// and the only tool that grabs a vertex on press.
    #[default]
    Select,
    /// Add a point: click a profile **segment** to insert a new vertex there, splitting the
    /// edge at the grid-snapped click.
    AddPoint,
    /// Draw connected straight segments. Dragging from a live end after the first curve appends
    /// a tangent arc, then returns to straight continuation. The start closes the chain; clicking
    /// the live end or pressing Enter finishes it open.
    Line,
    /// Draw one segment from its midpoint: click the construction midpoint, then one endpoint.
    /// The reflected endpoint is derived; the midpoint itself is never persisted.
    MidpointLine,
    /// Draw a rectangle: press one corner, drag, release at the opposite corner to
    /// append the closed four-segment loop. A degenerate (zero-span) drag draws nothing.
    Rectangle,
    /// Draw an oriented rectangle from a base edge and perpendicular-width pick.
    Rectangle3Point,
    /// Draw an axis-aligned rectangle from its center and one corner.
    RectangleCenterCorner,
    /// Draw a 3-point arc: click the start, the end, then a point the arc passes THROUGH. The
    /// through-point is consumed — the stored form is the two endpoints plus the solved included
    /// angle.
    ThreePointArc,
    /// Draw a counter-clockwise arc by clicking its center, start point, and end direction. The
    /// final pick is projected onto the start radius.
    ArcCenterEndpoints,
    /// Draw an arc tangent to a line or arc: click an endpoint on the incoming curve, then the
    /// destination endpoint. The durable Tangent relation is authored with the arc.
    ArcTangent,
    /// Draw a circle: click its center, then a point on its perimeter.
    CircleCenterDiameter,
    /// Draw a circle from two opposite diameter endpoints.
    Circle2Point,
    /// Draw the unique circle through three circumference points.
    Circle3Point,
    /// Draw a radius-selected circle tangent to two selected line segments.
    Circle2Tangent,
    /// Draw the circle tangent to three selected line segments.
    Circle3Tangent,
    /// Draw a regular polygon whose vertices lie on the authored center-radius circle.
    PolygonInscribed,
    /// Draw a regular polygon whose edge midpoints lie on the authored center-apothem circle.
    PolygonCircumscribed,
    /// Draw a regular polygon from one edge and a third pick selecting the body side.
    PolygonEdge,
    /// Draw a linear slot from the centers of its semicircular ends, then its width.
    SlotCenterToCenter,
    /// Draw a linear slot from its overall endpoints, then its width.
    SlotOverall,
    /// Draw a linear slot from its midpoint, one cap center, then its width.
    SlotCenterPoint,
    /// Draw a curved slot from a center-first arc and a width pick.
    SlotCenterPointArc,
    /// Draw a curved slot from a three-point center arc and a width pick.
    Slot3PointArc,
}

/// The floating Signal **display stack**'s viewer state.
///
/// The stack is the near-black instrument panel that floats top-right of the 3D viewport
/// (the cube + rail slide left of it). Whether it is folded to edge tabs, and which
/// sections are open, are **never document state** — like [`ViewMode`], they are not saved
/// with the scene and never enter undo history. They are *session* state, so the fold state
/// is restored on relaunch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalStackState {
    /// When `true` the whole stack is collapsed to vertical edge tabs hugging the
    /// viewport's right edge; the `»` header button folds it and a
    /// `«` tab (or any section tab) expands it again.
    pub folded: bool,
    /// The VIEWPORT section (mode readout + camera projection) is expanded.
    pub viewport_open: bool,
    /// The ONION FOG section (layer scrubber + onion depth + widest-run stat) is expanded.
    /// Only mounts in [`ViewMode::OnionFog`]; ignored in other modes.
    pub onion_open: bool,
    /// The GRIDS section (the display master toggles) is expanded.
    pub grids_open: bool,
}

impl Default for SignalStackState {
    fn default() -> Self {
        // Expanded with every section open — the finished-look default the goldens pin.
        Self {
            folded: false,
            viewport_open: true,
            onion_open: true,
            grids_open: true,
        }
    }
}

/// Layer-range scrubber state.
///
/// Z-up: layers run along **Z** (height). `lower`/`upper` are voxel Z-layer indices selected on a
/// track `0..grid_z`; the visible band is layers `[lower, upper]` INCLUSIVE on both
/// ends (so `lower == upper` shows a single layer). Default = the full range.
///
/// When `snap_to_blocks` is on, the handles snap to multiples of
/// `voxels_per_block` (plus the endpoints `0` and `grid_z`); a narrowed
/// single-layer band viewed from the top is the chisel stencil. `onion_skin`
/// ghosts up to `onion_depth` layers on each side of the band (3D screen-door).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    /// Lower handle: the first visible layer index (`0..=grid_z`).
    pub lower: u32,
    /// Upper handle: the last visible layer index (`lower..=grid_z`).
    pub upper: u32,
    /// Snap the handles to block boundaries (multiples of `voxels_per_block`).
    pub snap_to_blocks: bool,
    /// Show ghosted neighbor layers around the band (3D onion skin).
    pub onion_skin: bool,
    /// How many layers on each side of the band to ghost (1..=8).
    pub onion_depth: u32,
}

impl Default for LayerRange {
    fn default() -> Self {
        // Full range over the default cylinder grid_z (1 block × 16 density = 16).
        // The real bounds are clamped/rescaled to the live grid on first rebuild
        // and whenever grid_z changes (see `LayerRange::rescale_to_grid_z`).
        Self {
            lower: 0,
            upper: 16,
            snap_to_blocks: true,
            onion_skin: false,
            onion_depth: 2,
        }
    }
}

impl LayerRange {
    /// Snap a layer index to the nearest block boundary, keeping the endpoints
    /// `0` and `grid_z` exact (they are always valid snap points even when
    /// `grid_z` is not a clean multiple of the density, which it always is here).
    pub fn snap_value(value: u32, voxels_per_block: u32, grid_z: u32) -> u32 {
        let step = voxels_per_block.max(1);
        if value >= grid_z {
            return grid_z;
        }
        let snapped = ((value + step / 2) / step) * step;
        snapped.min(grid_z)
    }

    /// Clamp/rescale the bounds to a (possibly new) `grid_z` (Z-up: layers are
    /// Z-slices). Called on every geometry rebuild: when `grid_z` shrinks the handles
    /// are clamped in; the default full-range state widens to the new top. Re-snaps to
    /// block multiples when snapping is on so the band keeps landing on boundaries.
    pub fn rescale_to_grid_z(&mut self, previous_grid_z: u32, grid_z: u32, voxels_per_block: u32) {
        // A band that spanned the whole previous grid stays "full" on the new one.
        let was_full = self.lower == 0 && self.upper >= previous_grid_z;
        if was_full || previous_grid_z == 0 {
            self.lower = 0;
            self.upper = grid_z;
        } else {
            self.lower = self.lower.min(grid_z);
            self.upper = self.upper.min(grid_z);
        }
        if self.snap_to_blocks {
            self.lower = Self::snap_value(self.lower, voxels_per_block, grid_z);
            self.upper = Self::snap_value(self.upper, voxels_per_block, grid_z);
        }
        if self.lower > self.upper {
            std::mem::swap(&mut self.lower, &mut self.upper);
        }
        self.onion_depth = self.onion_depth.clamp(1, 8);
    }

    /// Whether this band covers the whole grid (so the 3D render is unclipped).
    pub fn is_full_range(&self, grid_z: u32) -> bool {
        self.lower == 0 && self.upper >= grid_z
    }
}

/// Mutable UI state passed to [`build_panel`](super::build_panel).
///
/// Holds the geometry params (rebuild-driving) and the display/camera params
/// (no rebuild). The binaries own one of these and feed it to the panel each
/// frame; [`PanelResponse`] tells them what changed.
///
/// Every field is **classified**: it declares which persistence artifacts it reaches, and a new
/// field that declares nothing does not compile. This struct is where the scheme earns its keep,
/// because it is the one the shell hands to `AppConfig::capture` — the exact seam at which a
/// camera pan target can go quietly missing from a repro.
/// Each category applies to the whole object and does not recurse: `layer_range` is view
/// state entire, and nothing inside [`LayerRange`] is annotated, because serialization
/// already carries what is inside a saved object.
#[derive(Debug, Clone, Default, snapshot::Snapshot)]
pub struct PanelState {
    /// The scene: the flat node list that is the panel's source of truth.
    /// The node list section adds/selects/deletes nodes; the inspector
    /// edits the ACTIVE node. [`geometry`](Self::geometry) / [`material`](Self::material)
    /// are the inspector's working mirror of the active Tool node (synced both
    /// ways) so the renderer/export call sites that read voxel dims + density keep
    /// working unchanged.
    #[snapshot(document)]
    pub scene: Scene,
    /// Rebuild-driving geometry params — the inspector's editing surface, mirrored
    /// onto the active Tool node (and re-read from it when the selection changes).
    ///
    /// Classified **view**, not document: the truth is the node the mirror was synced
    /// from ([`sync_mirror_from_active`](Self::sync_mirror_from_active)). It is not
    /// `derived` either, and the distinction is worth being exact about — a half-typed
    /// size sitting here has not reached any node yet, so dropping it would lose an
    /// edit in progress rather than merely cost a recomputation.
    #[snapshot(view)]
    pub geometry: GeometryParams,
    /// Camera projection (display-only: no rebuild). A preference that outlives any one
    /// project, so it is settings rather than view.
    #[snapshot(settings)]
    pub projection_mode: ProjectionMode,
    /// Material selection (display-only: selects the procedural texture).
    ///
    /// Settings, because this field is the *picker's* current value and persists across
    /// projects; the document's copy of a material lives on the node the pick was applied
    /// to, and travels in the scene.
    #[snapshot(settings)]
    pub material: MaterialChoice,
    /// Whether the Points' axes draw ON TOP of the model (depth off, through it — a nav marker)
    /// vs occluded by it (depth-tested scaffold). ON by default; screen-stable either way. A
    /// display preference that outlives a project, so settings like the view cube.
    #[snapshot(settings)]
    pub axes_on_top: bool,
    /// Whether the voxel cubes render in face-orientation debug mode (color by
    /// outward face normal + a back-facing marker, cull off). Display toggle, OFF
    /// by default; the standard way to verify face winding/culling.
    ///
    /// **Session** state, on the same footing as [`view_mode`](Self::view_mode): it
    /// describes what the workspace was doing, not what the model is and not what the user
    /// prefers. A debug mode a fault was observed under is precisely the sort of thing a dump
    /// must carry.
    #[snapshot(session)]
    pub debug_face_orientation: bool,
    /// Grazing-rim DIAGNOSTIC for the BRICK raymarch (`set_debug_mode`): shade every hit
    /// by its face axis + a per-voxel UV checkerboard, so a wrong first-hit voxel/face
    /// shows as a face-color break and a UV smear. Unlike `debug_face_orientation` (which
    /// drops to the mesh path), this keeps the brick path ENGAGED — it IS the path under
    /// investigation. Display toggle, OFF by default.
    ///
    /// **Session** state. This one makes the argument by itself: the diagnostic exists to be
    /// on while a rendering fault is being chased, so a dump taken during that chase and
    /// replayed without it reproduces the wrong picture.
    #[snapshot(session)]
    pub debug_brick_faces: bool,
    /// When `Some`, the 3D rebuild was skipped because the grid exceeds the
    /// voxel cap; the panel shows a warning. Set by the caller after it decides
    /// whether to rebuild. Value is the would-be voxel count (in millions).
    ///
    /// **Derived**: the value is a function of the scene and its density, both classified, and
    /// recomputed by the caller at every rebuild. Dropping it costs one more count and changes
    /// nothing else.
    #[snapshot(derived)]
    pub voxel_cap_warning_millions: Option<f32>,
    /// When `true`, the last authored edit was REJECTED because it would push a node
    /// past the ±1,000,000-block display coordinate envelope. The panel shows a warning.
    /// Set by the caller on a rejected intent, cleared on the next accepted geometry edit.
    ///
    /// **Derived**, on the same footing as [`voxel_cap_warning_millions`](Self::voxel_cap_warning_millions):
    /// a function of the last edit's outcome, recomputed at every apply, and dropping it
    /// changes nothing else.
    #[snapshot(derived)]
    pub coordinate_limit_warning: bool,
    /// When `Some`, the last constraint the author asked for was REFUSED, and the value says
    /// why. The top bar shows it while a sketch is open; the next constraint that
    /// lands clears it. A fixed string rather than an owned one, because the refusals are a
    /// closed set and nothing about them is per-sketch.
    ///
    /// **Derived**, on the same footing as
    /// [`coordinate_limit_warning`](Self::coordinate_limit_warning): a function of the last
    /// edit's outcome, recomputed at every apply, and dropping it changes nothing else.
    #[snapshot(derived)]
    pub sketch_constraint_refusal: Option<&'static str>,
    /// When `Some`, a loaded VS block (M6) is the active material; the value is
    /// its label, shown under the Material selector. `None` = a procedural
    /// material is active.
    ///
    /// Settings, and deliberately NOT derived: it cannot be recomputed, because the
    /// texture it names is re-resolved lazily and best-effort (see the `settings` module
    /// header) — the label is the only surviving record of the pick.
    #[snapshot(settings)]
    pub applied_block_label: Option<String>,
    /// The viewer's exclusive rendering mode: Normal / Onion fog / Show booleans. No
    /// rebuild, never in undo, sticky across selection changes; defaults to Normal.
    ///
    /// **Session** state, and the field the category was named for. It stays out of the
    /// document, which is not the same claim as "not persisted at all": leaving the app in
    /// Onion fog and finding it in Normal on relaunch is losing work, in the small.
    #[snapshot(session)]
    pub view_mode: ViewMode,
    /// The floating Signal display stack's state: folded-to-edge-tabs and per-section
    /// open/closed.
    ///
    /// **Session** state alongside [`view_mode`](Self::view_mode) — where the furniture
    /// was left, which is not a preference the user would want imposed on a project and
    /// not something the model is. Classified as one object; the four section flags inside
    /// it are not annotated, and do not need to be.
    #[snapshot(session)]
    pub stack: SignalStackState,
    /// Layer-range scrubber state: the visible band along Z (Z-up: layers
    /// are Z-slices) plus the snap/onion controls. Bounds clamped/rescaled on rebuild.
    #[snapshot(view)]
    pub layer_range: LayerRange,
    /// Where **+ Add Point** drops a new Point, in whole world blocks.
    /// The caller refreshes it each frame from the camera target (rounded to blocks)
    /// so a new Point lands where the user is looking; it defaults to the world origin
    /// (`[0, 0, 0]`) when the caller does not set it (e.g. the headless harness).
    ///
    /// **Derived**: the camera target rounded to blocks, and the camera is classified view
    /// state. Dropping it means recomputing the rounding, and nothing else — which is the
    /// admission test, met exactly.
    #[snapshot(derived)]
    pub point_add_position_blocks: [i64; 3],
    /// The **armed tool**: the [`NodeSpec`] a stationary viewport click drops
    /// plus its pending-drop ghost, or `None` when nothing is armed. See [`ArmedTool`].
    ///
    /// **Session** state, on the same footing as [`view_mode`](Self::view_mode): an armed
    /// tool is how the workspace was left, not what the model is and not a preference. A
    /// dump taken mid-gesture and replayed must show the same pending drop, so the tool (WITH
    /// its drop) travels into the dump and never into the shared document. The authority and
    /// its derived ghost are ONE field, so the capture cannot carry one without the other.
    #[snapshot(session)]
    pub armed_tool: Option<ArmedTool>,
    /// The armed-tool placement snap settings: position (no snap / block / voxel) and
    /// orientation (no snap / surface). **Session** state — durable across adds and relaunch,
    /// edited in the `Add <shape>` dialog, read by `place_primitive`.
    #[snapshot(session)]
    pub placement_snap: PlacementSnap,
    /// The sketch node currently being edited in **sketch mode**, or `None` when
    /// the workspace is in its normal chrome. `Some(id)` swaps the rail to the sketch toolset,
    /// withdraws the non-sketch operations, marks the node "editing" in the browser, and shows
    /// the floating `CANCEL | FINISH SKETCH` exit control.
    ///
    /// **Session** state, on the same footing as [`view_mode`](Self::view_mode) and
    /// [`armed_tool`](Self::armed_tool): the mode follows what you are editing, is
    /// never document state (a saved document is byte-identical whether or not a
    /// sketch was being edited), and rides into the dump so a mid-edit repro re-enters the same
    /// sketch. Cleared when the id leaves the scene (a stale node can never trap the mode).
    #[snapshot(session)]
    pub sketch_mode: Option<NodeId>,
    /// The armed sketch-mode tool: which vertex verb a viewport click performs
    /// while [`sketch_mode`](Self::sketch_mode) is `Some`. Ignored (but retained) outside the
    /// mode, exactly like [`placement_snap`](Self::placement_snap) is retained with nothing
    /// armed. Defaults to [`SketchTool::Select`].
    ///
    /// **Session** state alongside [`sketch_mode`](Self::sketch_mode) and
    /// [`armed_tool`](Self::armed_tool): the armed tool is where the author left the
    /// workspace, never document state, and rides into the dump so a mid-edit repro re-enters
    /// with the same tool in hand.
    #[snapshot(session)]
    pub sketch_tool: SketchTool,
    /// Side count used by all regular-polygon creation tools. Values outside `3..=128` are
    /// normalized to the six-sided default at the interaction seam, which keeps older session
    /// artifacts (where this field is absent and therefore zero) deterministic.
    #[snapshot(session)]
    pub sketch_polygon_sides: u16,
    /// The armed **constraint** and the entities picked for it so far.
    ///
    /// Held apart from [`sketch_tool`](Self::sketch_tool) rather than joining its enum, because
    /// the two arm different things and one does not replace the other: a constraint gesture
    /// runs *over* the drawing tools' vocabulary — it hit-tests the same entities Select does —
    /// and folding it in would make every `SketchTool` match arm answer for a mode that draws
    /// nothing. It also ends on its own, at completion, which no drawing tool does.
    ///
    /// **Session** state alongside [`sketch_tool`](Self::sketch_tool): a half-finished gesture
    /// is where the author left the workspace, and a dump taken mid-pick should re-enter with
    /// the same question on screen.
    #[snapshot(session)]
    pub armed_constraint: Option<ArmedConstraint>,
    /// The sketch-mode **position snap**: how a vertex edit quantizes on the sketch plane's
    /// own grid — sub-voxel continuous ([`PositionSnap::NoSnap`], the fraction rides the
    /// vertex), whole-voxel (the default), or block boundaries. Reuses the placement
    /// [`PositionSnap`]; the lattice stands in for a constraint solver.
    ///
    /// **Session** state alongside [`sketch_tool`](Self::sketch_tool) and on the same footing
    /// as [`placement_snap`](Self::placement_snap): an editing preference that is never
    /// document state, durable across edits and relaunch.
    #[snapshot(session)]
    pub sketch_snap: PositionSnap,
    /// The workspace **selection** — every picked target, whatever its kind: scene nodes,
    /// reference Points, sketch vertices and sketch edges. One set for every kind: mode
    /// exclusivity is an admission filter, not a reason for parallel structures. Edits steer it
    /// as an effect, but it is never document truth and undo never restores it as such.
    ///
    /// **Session** state alongside [`sketch_mode`](Self::sketch_mode): what you had picked is
    /// where you left the workspace, never travels in a shared file, and rides the dump so a
    /// repro re-enters with the same thing selected.
    ///
    /// The SKETCH-kind targets are the exception, and deliberately: `SelectionConfig` drops
    /// them on capture. In-mode picks are momentary, cleared on entering and leaving a sketch,
    /// and persisting an `EntityId` is the one way a target could go stale against an edited
    /// profile. Folding the data structure does not oblige folding the persistence policy.
    #[snapshot(session)]
    pub selection: super::Selection,
    /// The **default orbit type**: what an orbit gesture turns as when nothing named a type.
    ///
    /// Say "the default type", never "the type" — a command that *names* a type (the viewport
    /// context menu's Constrained Orbit) overrides it for the orbit-mode session WITHOUT writing
    /// here, so the two differ whenever an override is running. Only the display rail's orbit
    /// split button writes this: picking from a split button re-faces it, which is the same act
    /// as setting the default, whereas invoking a command has never meant "make this the
    /// default".
    ///
    /// **Session** state alongside [`sketch_tool`](Self::sketch_tool): how the author was last
    /// steering the view is where they left the workspace — never Settings (it is a
    /// most-recently-used working state, not a preference), never the document, and it rides the
    /// dump so a repro orbits the way the report did.
    #[snapshot(session)]
    pub default_orbit_type: OrbitType,
    /// Whether the explicit **orbit mode** is running, and what it turns as — see [`OrbitMode`].
    ///
    /// **Session** state alongside [`sketch_mode`](Self::sketch_mode), and for the same reason: a
    /// mode is where you left the workspace, and a dump taken mid-orbit should re-enter turning
    /// the way the report was.
    #[snapshot(session)]
    pub orbit_mode: OrbitMode,
    /// The keyboard-shortcut settings — see [`shortcuts`](crate::shortcuts). The menus read their
    /// right-hand column out of this, and the shell dispatches key presses through it, so this is
    /// the single place a binding is written down.
    ///
    /// **Settings**: a rebound key is preference that outlives any one project, and it is
    /// emphatically not something a collaborator should have imposed on them by opening a file.
    #[snapshot(settings)]
    pub shortcuts: crate::shortcuts::Shortcuts,
}

/// Whether the explicit **orbit mode** is running, and what it turns as.
///
/// While it runs the left button turns the camera about `camera.target` instead of selecting, a
/// targeting reticle marks that target, and a stationary click re-centers the view on the surface
/// it hits. Leaving restores left = select. It is independent of the orbit center — the Shift+MMB
/// pivot — and never writes it.
///
/// It also exists to carry the TYPE OVERRIDE's lifetime, which is why [`Named`](Self::Named) is a
/// variant here rather than a flag on the gesture: a command that names a type runs that type
/// until the mode ends, and an override whose boundaries the user cannot see is one they cannot
/// reason about. "Mode" already means a state with an exit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OrbitMode {
    /// Not in the mode: the left button selects.
    #[default]
    Off,
    /// In the mode, turning as [`default_orbit_type`](PanelState::default_orbit_type) — what the
    /// rail's split-button FACE enters, since a split button's face never names a type.
    UsingDefault,
    /// In the mode, turning as the type a command NAMED — the viewport menu's Constrained Orbit.
    /// Naming a type here does not write the default; invoking a tool has never meant "make this
    /// the default".
    Named(OrbitType),
}

impl OrbitMode {
    /// Whether the mode is running at all — the one question the input router and the reticle ask.
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// What an orbit STARTED NOW turns as, given the default. The single definition of "active
    /// type": the rail's face shows it and the gesture latches it, and those two must not be able
    /// to disagree.
    pub fn active_type(self, default_type: OrbitType) -> OrbitType {
        match self {
            Self::Named(named) => named,
            Self::Off | Self::UsingDefault => default_type,
        }
    }
}

impl PanelState {
    /// What an orbit started now would turn as — the mode's override if one is running, else the
    /// default. See [`OrbitMode::active_type`].
    pub fn active_orbit_type(&self) -> OrbitType {
        self.orbit_mode.active_type(self.default_orbit_type)
    }
}

impl PanelState {
    /// Sensible display defaults for the windowed app: like [`Default`] but with the Point
    /// axes on top (the derived default is off). The view cube is always drawn, so it is no
    /// longer a toggle here.
    pub fn with_view_cube_default() -> Self {
        let mut state = Self {
            axes_on_top: true,
            ..Self::default()
        };
        state.seed_scene_from_geometry();
        state
    }

    /// Seed the scene with a single Tool node from the current geometry/material
    /// mirror (the back-compat path: a default or a config-loaded one-geometry
    /// state becomes a one-Tool-node scene). Does nothing if the scene already has
    /// nodes.
    pub fn seed_scene_from_geometry(&mut self) {
        if self.scene.roots.is_empty() {
            self.scene = Scene::from_geometry(self.geometry.clone(), self.material);
        }
        // Every scene carries exactly one Origin Point. Idempotent, so calling it on an
        // already-seeded scene is a no-op.
        self.scene.ensure_origin_point();
        // Mint a stable NodeId for every node (idempotent).
        self.scene.ensure_node_ids();
        // The workspace arrives with the seed node picked, so the inspector has
        // something to mirror on a fresh launch. Only when nothing is picked yet — a
        // config-restored selection wins.
        if self.selection.is_empty() {
            self.selection
                .set_primary_node(self.scene.roots.first().copied());
        }
    }

    /// The ARMED primitive's kind, or `None` when nothing is armed (or the armed spec is
    /// not a primitive Tool) — what lights the rail's shape cell and shows the
    /// `Add <shape>` dialog. Read straight off [`armed_tool`](Self::armed_tool), never
    /// mirrored.
    pub fn armed_shape(&self) -> Option<voxel_core::voxel::ShapeKind> {
        match &self.armed_tool.as_ref()?.spec {
            NodeSpec::Tool { shape, .. } => Some(shape.kind),
            _ => None,
        }
    }

    /// The armed tool's pending-drop ghost, or `None` when nothing is armed or the cursor
    /// is off a valid drop. Read straight off [`armed_tool`](Self::armed_tool).
    pub fn placement_ghost(&self) -> Option<&PlacementGhost> {
        self.armed_tool.as_ref()?.pending_drop.as_ref()
    }

    /// The primary selected node, resolved against the workspace
    /// [`selection`](Self::selection) rather than any document field. `None` when no node is
    /// picked, or when the picked id has left the scene.
    pub fn selected_node(&self) -> Option<&document::scene::Node> {
        self.selection
            .primary_node_id()
            .and_then(|id| self.scene.node_by_id(id))
    }

    /// Copy the active node's parameters into the inspector mirror
    /// ([`geometry`](Self::geometry) / [`material`](Self::material)) when it is a
    /// Tool, so the inspector edits the active selection. Called whenever the
    /// active node changes (selection or delete). A VoxelBody active node leaves the
    /// mirror untouched (its editor shows name + seed instead).
    pub fn sync_mirror_from_active(&mut self) {
        let selected_id = self.selection.primary_node_id();
        if let Some(node) = selected_id.and_then(|id| self.scene.node_by_id(id)) {
            // A sketch node shares the single `material` field; mirror it so the
            // inspector's Material selector reflects the selected sketch's material
            // (its producer is read straight from the node, not from the geometry
            // mirror, so only the material needs syncing here).
            if let NodeContent::SketchTool { material, .. } = &node.content {
                self.material = *material;
            }
            if let NodeContent::Tool { shape, material } = &node.content {
                self.geometry = GeometryParams {
                    shape: shape.kind,
                    // Size is voxel-granular: carry the canonical
                    // voxels AND the retained authored expression so the inspector
                    // seeds / re-emits the exact size the user typed.
                    size_voxels: shape.size_voxels,
                    size_measurements: if shape.has_retained_size_measurements() {
                        Some(Box::new(shape.size_measurements()))
                    } else {
                        None
                    },
                    // Density is document-level: the slider's
                    // transient mirror value comes from the scene, not the shape.
                    voxels_per_block: self.scene.voxels_per_block,
                    wall_blocks: shape.wall_blocks,
                };
                self.material = *material;
            }
        }
    }
}

/// What changed during a [`build_panel`](super::build_panel) call, so the caller can react.
///
/// The panel never mutates `state.scene` directly: every document mutation this frame is
/// DESCRIBED as an [`Intent`] pushed onto [`intents`](Self::intents), which the loop applies
/// through the shell's `AppCore::apply_intent`, folding the returned `IntentEffect`s into its
/// rebuild / points / selection decisions. The remaining fields are NON-scene side effects
/// (palette / export / folder picker) the panel only flags, plus the
/// [`frame_after_apply`](Self::frame_after_apply) auto-frame hint — a panel UX concern, since a
/// size-slider `SetShape` re-frames and a shape-chip `SetShape` does not even though both are
/// the same intent KIND, so it cannot be derived from the intent alone.
#[derive(Debug, Clone, Default)]
pub struct PanelResponse {
    /// The document mutations the user made this frame, in emission order. The loop applies
    /// each through `AppCore::apply_intent` and merges the effects; the panel itself performs
    /// NONE of them.
    pub intents: Vec<Intent>,
    /// The caller should auto-frame the camera after applying this frame's intents
    /// Set by the panel for every emitted intent EXCEPT a pure shape-chip switch and a
    /// material pick — a shape switch re-resolves at the same size and must NOT move the
    /// camera. A panel-level signal because the
    /// same intent KIND (`SetShape`) auto-frames from a size slider but not from a
    /// shape chip.
    pub frame_after_apply: bool,
    /// A palette tile was clicked this frame → apply a pseudo-random variant of
    /// this tile index as the active loaded material.
    pub clicked_palette_tile: Option<usize>,
    /// The "Connect folder…" button was clicked → open the OS folder picker and
    /// scan the chosen folder via `CustomFolderSource` (M6).
    pub clicked_connect_folder: bool,
    /// A built-in procedural material (Stone/Wood/Plain) was selected this frame →
    /// clear any applied loaded block and revert to the procedural material (M6).
    pub selected_procedural_material: bool,
    /// The "Export .vox" button was clicked this frame → open the OS save dialog
    /// and write the resolved grid as a MagicaVoxel `.vox` file (M8).
    pub clicked_export_vox: bool,
    /// The user picked **Focus** from a node row's right-click context menu this
    /// frame → the loop should frame that node (set the camera target to the node's
    /// world center + fit the distance). A VIEW action, NOT a document `Intent` (it
    /// is not undoable), so it rides on the response rather than `intents`. `None`
    /// when no Focus was requested.
    pub focus_node: Option<NodeId>,
    /// The tool the user armed from "+ Add" this frame → the shell starts the live
    /// placement flow (a translucent ghost follows the cursor, a stationary click drops
    /// the node). A VIEW action, NOT a document `Intent` (arming places nothing until a
    /// click), so it rides on the response rather than `intents`. `None` when nothing
    /// was armed this frame.
    pub arm_tool: Option<NodeSpec>,
    /// The user clicked the rail's ARMED shape cell again this frame → the shell disarms
    /// the placement flow (the same full disarm Escape and a viewport right-click perform).
    /// A VIEW action like [`arm_tool`](Self::arm_tool); the two are emitted mutually
    /// exclusively at the source (the cell reads the armed mirror), and the shell lets an
    /// explicit disarm win if a future second source ever sets both in one frame.
    pub disarm_tool: bool,
    /// The sketch node the user asked to **enter sketch mode** on this frame, via
    /// the inspector's "Edit sketch" button. A VIEW action, NOT a document `Intent` (entering
    /// a mode mutates nothing in the document), so it rides on the response like `focus_node`.
    /// The shell sets [`PanelState::sketch_mode`](PanelState::sketch_mode) to it. `None` when
    /// no enter was requested.
    pub enter_sketch: Option<NodeId>,
    /// The user chose **Delete** from the general viewport context menu this frame → the shell
    /// removes what is picked. WHAT that means is the shell's to decide, not the panel's: inside a
    /// sketch it deletes the picked entities as one edit with points cascading their segments,
    /// and outside one it removes the picked node. Routed as a flag rather than an
    /// `Intent`, because the selection and the sketch commit path both live on the shell — and
    /// because the keyboard binding for the same command reaches the same door
    /// (`ui::shortcuts::ShortcutCommand::DeleteSelection`), which a menu-built intent could not.
    /// `false` when no delete was requested.
    pub delete_selection: bool,
    /// The user chose **Carve hole / Fill region** from the viewport menu this frame → the shell
    /// flips the pick state of the face the menu opened over. A flag rather
    /// than a key, for the same reason `delete_selection` is: WHICH face the right-click landed
    /// in is a screen-space hit-test only the shell can answer, and it already answered it to
    /// decide whether to offer the row at all. `false` when the row was not chosen.
    pub toggle_sketch_face: bool,
    /// The sketch rail's Construction command was pressed. The shell owns the selected entity
    /// ids and commits their role changes through the normal anchor-preserving sketch edit door.
    pub toggle_sketch_construction: bool,
    /// How the user asked to move the **orbit center** this frame from the general viewport
    /// context menu — the deliberate act that is
    /// the ONLY thing allowed to move it, which is what makes a pan leave it alone. A VIEW
    /// action, not an `Intent`: the camera is not the document. `None` when neither menu item
    /// was chosen.
    pub orbit_center_request: Option<OrbitCenterRequest>,
    /// How the user ended a **running modal command** this frame, from the viewport context
    /// menu's OK / Cancel variant. A VIEW action like the rest of this block: the shell owns what
    /// each modal command's accept and cancel actually mean. `None` when no command was running
    /// or the menu was dismissed without a choice.
    pub mode_command: Option<ModeCommand>,
    /// How the user asked to **leave sketch mode** this frame, via the floating
    /// `CANCEL | FINISH SKETCH` control — `Finish` commits, `Cancel` discards. A VIEW action:
    /// the shell clears [`PanelState::sketch_mode`](PanelState::sketch_mode) and closes or
    /// rolls back the undo group. `None` when no exit was requested.
    pub exit_sketch: Option<SketchExit>,
    /// How the user asked the **workspace selection** to change this frame — a
    /// clicked browser/tree/points row, or a deselect. A VIEW action, NOT a document
    /// `Intent`: selecting is not an edit and reverses nothing, so it rides on the response
    /// like [`focus_node`](Self::focus_node) and the shell lands it on
    /// [`PanelState::selection`](PanelState::selection). `None` when the selection was not
    /// touched this frame.
    pub select: Option<super::SelectionRequest>,
}

impl PanelResponse {
    /// Push a mutation the user described this frame. The loop applies it through
    /// `AppCore::apply_intent`; the panel never mutates the scene.
    pub(crate) fn emit(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    /// Push a mutation AND request an auto-frame after this frame's intents apply. Used for
    /// structural edits and size/density edits — everything that re-frames; a shape-chip switch
    /// and a material pick use [`emit`](Self::emit) instead so the camera stays put.
    pub(crate) fn emit_and_frame(&mut self, intent: Intent) {
        self.frame_after_apply = true;
        self.intents.push(intent);
    }
}

/// The export section's live state, passed in by the shell so the panel stays free of
/// file-system concerns (slow-paths item 2 — the `.vox` write runs on a background
/// worker). While an export is in flight the button is disabled and `status_line` carries
/// the progress readout; otherwise `status_line` is the last completion / failure /
/// large-export message (or `None`). The shell formats the line — the panel only shows it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExportPanelState<'a> {
    /// True while an export is running: the button is disabled (the shell serializes
    /// exports, so a second one can never be queued).
    pub in_flight: bool,
    /// The already-formatted line to show under the button, or `None`.
    pub status_line: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::duration_subsec,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::match_same_arms,
        clippy::panic,
        clippy::semicolon_if_nothing_returned,
        clippy::unwrap_used,
        clippy::while_float
    )]

    use super::*;
    use document::voxel::SdfShape;
    use voxel_core::voxel::ShapeKind;

    /// **The ghost center carries the sub-voxel `offset_local`** — a `NoSnap` drop's fractional
    /// remainder — so the translucent preview sits exactly where the committed node lands rather
    /// than snapping to the integer voxel (the off-by-a-few-voxels mismatch a user hit in `NoSnap`
    /// mode). Two ghosts differing ONLY in `offset_local` must have centers that differ by exactly
    /// that fraction, since `center_world` now seats through the same `LeafPlacement` affine the
    /// classifier folds occupancy through.
    #[test]
    fn ghost_center_carries_the_sub_voxel_offset() {
        let shape = SdfShape::from_voxels(ShapeKind::Box, [16, 16, 16], 1);
        let recenter = [3, 4, 5];
        let density = 1;
        let base = PlacementGhost {
            shape,
            offset_voxels: [10, 20, 30],
            offset_local: [0.0, 0.0, 0.0],
            rotation: glam::Quat::IDENTITY,
        };
        let shifted = PlacementGhost {
            offset_local: [0.25, -0.5, 0.75],
            ..base.clone()
        };
        let base_center = base.center_world(recenter, density);
        let shifted_center = shifted.center_world(recenter, density);
        assert_eq!(
            [
                shifted_center[0] - base_center[0],
                shifted_center[1] - base_center[1],
                shifted_center[2] - base_center[2],
            ],
            [0.25, -0.5, 0.75],
            "the ghost center must carry the sub-voxel offset, not snap to the integer voxel"
        );
    }
}
