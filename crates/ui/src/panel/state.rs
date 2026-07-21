//! The panel's mutable state ([`PanelState`], [`LayerRange`]) and the per-frame
//! [`PanelResponse`] / [`ExportPanelState`] carried between the shell and the
//! section builders.

use camera::ProjectionMode;
use document::intent::{Intent, NodeSpec};
use document::scene::{NodeContent, NodeId, Scene};
use document::voxel::{GeometryParams, SdfShape};
use voxel_core::core_geom::MaterialChoice;

/// The armed-tool **placement ghost** (ADR 0022): the translucent analytic-SDF preview of
/// where a primitive's voxels will land, drawn without recomposing the scene ("render a
/// coloured transparent SDF where the voxels will be"). `PanelState::placement_ghost` is
/// `Some` while a tool is armed and pointed at a valid drop, `None` otherwise.
///
/// It carries the armed [`SdfShape`] and the ABSOLUTE, corner-anchored voxel offset the
/// node would take — the SAME frame `Intent::PlaceNode { offset_voxels }` uses
/// (`src/app_core/placement.rs`). The render-frame field centre the shader needs is
/// DERIVED at draw time from the live resolve's recentre via [`center_world`], keeping the
/// frame law (ADR 0008) in one place rather than baked into stored state that a later
/// rebuild would stale.
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
    /// The node's lattice orientation (ADR 0026) — the turn the drop would apply, so the
    /// ghost previews the shape the way it will actually land (a cylinder on its side, not
    /// upright). Identity for a world-plane or `+Z`-face drop.
    pub orientation: substrate::spatial::LatticeOrientation,
}

impl PlacementGhost {
    /// The field centre in the display's render frame:
    /// `center_world = offset_voxels + turn_extent(grid)/2 - recentre` (ADR 0008 + ADR 0026;
    /// the frame note atop `crates/display/src/shaders/placement_ghost.wgsl` derives it). The
    /// node occupies the **turned** grid in the world, so the centre uses the turned extent;
    /// `grid/2` is the EXACT half (a half-integer on odd axes), `recentre` the FLOORED half —
    /// the difference is the half-voxel term a naive "the shape is at the origin" drops.
    pub fn center_world(&self, recentre_voxels: [i64; 3], voxels_per_block: u32) -> [f32; 3] {
        let grid = self.orientation.turn_extent(self.shape.grid_dimensions(voxels_per_block));
        std::array::from_fn(|axis| {
            (self.offset_voxels[axis] - recentre_voxels[axis]) as f32 + grid[axis] as f32 / 2.0
        })
    }

    /// The inscribed semi-axes in voxels (`grid/2` per axis, EXACT half) the SDF is
    /// evaluated against. These are the shape's OWN (un-turned) half-extents — the shader
    /// evaluates the field in the shape's local frame after un-turning the sample point
    /// ([`orientation_inverse_columns`](Self::orientation_inverse_columns)), so the semi-axes
    /// never turn (only the sample point does).
    pub fn semi_axes(&self, voxels_per_block: u32) -> [f32; 3] {
        self.shape
            .grid_dimensions(voxels_per_block)
            .map(|axis| axis as f32 / 2.0)
    }

    /// The **inverse** orientation as column-major `f32` columns for the shader's
    /// `mat3x3<f32>` uniform (ADR 0026). The ghost stores the forward turn; the shader maps a
    /// world sample back into the shape's local frame with its inverse, so
    /// `orientation_inverse · (world − centre)` lands in the un-turned SDF frame. Each column
    /// is padded to a `vec4` (std140 mat3 stride); the `w` lane is unused.
    pub fn orientation_inverse_columns(&self) -> [[f32; 4]; 3] {
        // Row-major integer matrix of the inverse turn; column `j` of the matrix becomes the
        // shader's column `j` (WGSL `m * v = Σ col[j]·v[j]`).
        let inverse = self.orientation.inverse().to_matrix();
        std::array::from_fn(|column| {
            [
                inverse[0][column] as f32,
                inverse[1][column] as f32,
                inverse[2][column] as f32,
                0.0,
            ]
        })
    }

    /// `wall_blocks * density`, in voxels — the Tube wall thickness the SDF needs (ignored
    /// by every other kind).
    pub fn wall_voxels(&self, voxels_per_block: u32) -> f32 {
        (self.shape.wall_blocks * voxels_per_block) as f32
    }
}

/// The viewer's exclusive rendering mode (ADR 0018 Decision 3). The viewer is always in
/// exactly one of these three; the mode is **never document state** — it follows the
/// active selection, is not saved with the scene, and never enters undo history (the
/// [`PanelState`] display-param precedent, like [`ProjectionMode`]). Sticky across
/// selection changes; default [`Normal`](Self::Normal).
///
/// It **is** restored across relaunch, as *session* state (ADR 0024): out of the document,
/// into the dump. ADR 0018 Decision 3 said "not saved with the scene" and the code read
/// that as "not saved at all", which is the narrower claim it never made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ViewMode {
    /// The finished look: no ghosts, no band clip, anywhere (ADR 0018 Decision 4).
    #[default]
    Normal,
    /// Onion fog: the selected object clips to the layer band with ghost haze outside it
    /// (ADR 0018 Decision 5). The scrubber's `lower`/`upper` are object-relative over the
    /// selected object's Z extent (the shell's `AppCore::mesh_clip` derives the region-scoped
    /// clip from them); selecting the root part recovers the pre-Decision-5 scene-wide
    /// meaning.
    OnionFog,
    /// Show booleans: every Subtract/Intersect operand in the selected subtree x-rays
    /// over the finished scene (ADR 0018 Decision 6). Selecting the root part covers the
    /// whole scene.
    ShowBooleans,
}

impl ViewMode {
    /// The next mode in the Signal icon rail's cycle order (ADR 0018 Decision 8 /
    /// `docs/design/viewport-chrome-signal.md`): Normal -> Onion fog -> Show booleans ->
    /// Normal. The viewport-mode button steps through this; it is pure display state (no
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

/// The floating Signal **display stack**'s viewer state (issue #88; ADR 0018 Decision 8,
/// `docs/design/viewport-chrome-signal.md` §Chrome layout — display panel bullet).
///
/// The stack is the near-black instrument panel that floats top-right of the 3D viewport
/// (the cube + rail slide left of it). Whether it is folded to edge tabs, and which
/// sections are open, are **never document state** — like [`ViewMode`], they are not saved
/// with the scene and never enter undo history. They follow the *session*, and since
/// ADR 0024 that is a category with a route rather than a figure of speech: the fold state
/// is restored on relaunch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalStackState {
    /// When `true` the whole stack is collapsed to vertical edge tabs hugging the
    /// viewport's right edge (Blender N-panel style); the `»` header button folds it and a
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

/// Layer-range scrubber state (issue #12).
///
/// The layer-range scrubber subsumes the old 2D mid-vertical slice map. Z-up: layers
/// run along **Z** (height). `lower`/`upper` are voxel Z-layer indices selected on a
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
    /// Show ghosted neighbour layers around the band (3D onion skin).
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
/// Every field is **classified** (ADR 0022): it declares which persistence artifacts it
/// reaches, and a new field that declares nothing does not compile. This struct is where
/// the scheme earns its keep, because it is the one the shell hands to `AppConfig::capture`
/// — the exact seam at which the camera's pan target once went quietly missing from a repro.
/// Each category applies to the whole object and does not recurse: `layer_range` is view
/// state entire, and nothing inside [`LayerRange`] is annotated, because serialization
/// already carries what is inside a saved object.
#[derive(Debug, Clone, Default, snapshot::Snapshot)]
pub struct PanelState {
    /// The scene (ADR 0001): the flat node list that is now the panel's source of
    /// truth. The node list section adds/selects/deletes nodes; the inspector
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
    /// Material selection (display-only: selects the M4 procedural texture).
    ///
    /// Settings, because this field is the *picker's* current value and persists across
    /// projects; the document's copy of a material lives on the node the pick was applied
    /// to, and travels in the scene.
    #[snapshot(settings)]
    pub material: MaterialChoice,
    /// Whether the corner view cube is drawn (M5 Display toggle, ON by default).
    #[snapshot(settings)]
    pub show_view_cube: bool,
    /// Whether the voxel cubes render in face-orientation debug mode (colour by
    /// outward face normal + a back-facing marker, cull off). Display toggle, OFF
    /// by default; the standard way to verify face winding/culling.
    ///
    /// **Session** state, on the same footing as [`view_mode`](Self::view_mode): it
    /// describes what the workspace was doing, not what the model is and not what the
    /// user prefers. The note that used to sit here — "a debug mode a fault was observed
    /// under is precisely the sort of thing a dump must carry" — was right, and ADR 0024
    /// is where it stopped being an observation and became a route.
    #[snapshot(session)]
    pub debug_face_orientation: bool,
    /// Grazing-rim DIAGNOSTIC for the BRICK raymarch (`set_debug_mode`): shade every hit
    /// by its face axis + a per-voxel UV checkerboard, so a wrong first-hit voxel/face
    /// shows as a face-colour break and a UV smear. Unlike `debug_face_orientation` (which
    /// drops to the mesh path), this keeps the brick path ENGAGED — it IS the path under
    /// investigation. Display toggle, OFF by default.
    ///
    /// **Session** state (ADR 0024). This one makes the argument by itself: the diagnostic
    /// exists to be on while a rendering fault is being chased, so an F9 dump taken during
    /// that chase and replayed without it reproduces the wrong picture — the pan-target
    /// bug wearing a different hat.
    #[snapshot(session)]
    pub debug_brick_faces: bool,
    /// When `Some`, the 3D rebuild was skipped because the grid exceeds the
    /// voxel cap; the panel shows a warning. Set by the caller after it decides
    /// whether to rebuild. Value is the would-be voxel count (in millions).
    ///
    /// **Derived**, and it passes ADR 0023's admission test literally: the value is a
    /// function of the scene and its density, both classified, recomputed by the caller at
    /// every rebuild. Dropping it costs one more count and changes nothing else.
    #[snapshot(derived)]
    pub voxel_cap_warning_millions: Option<f32>,
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
    /// **Session** state, and the field the category was named for (ADR 0024, superseding
    /// ADR 0018 Decision 3). It stays out of the document exactly as Decision 3 required —
    /// what changed is that "not document state" was being read as "not persisted at all",
    /// and those are different claims. Leaving the app in Onion fog and finding it in
    /// Normal on relaunch is losing work, in the small.
    #[snapshot(session)]
    pub view_mode: ViewMode,
    /// The floating Signal display stack's state (issue #88): folded-to-edge-tabs and
    /// per-section open/closed.
    ///
    /// **Session** state alongside [`view_mode`](Self::view_mode) — where the furniture
    /// was left, which is not a preference the user would want imposed on a project and
    /// not something the model is. Classified as one object; the four section flags inside
    /// it are not annotated, and do not need to be.
    #[snapshot(session)]
    pub stack: SignalStackState,
    /// Layer-range scrubber state (issue #12): the visible band along Z (Z-up: layers
    /// are Z-slices) plus the snap/onion controls. Bounds clamped/rescaled on rebuild.
    #[snapshot(view)]
    pub layer_range: LayerRange,
    /// Where **+ Add Point** drops a new Point (issue #29 S5), in whole world blocks.
    /// The caller refreshes it each frame from the camera target (rounded to blocks)
    /// so a new Point lands where the user is looking; it defaults to the world origin
    /// (`[0, 0, 0]`) when the caller does not set it (e.g. the headless harness).
    ///
    /// **Derived**: the camera target rounded to blocks, and the camera is classified view
    /// state. Dropping it means recomputing the rounding, and nothing else — which is the
    /// admission test, met exactly.
    #[snapshot(derived)]
    pub point_add_position_blocks: [i64; 3],
    /// The armed-tool placement ghost (ADR 0022): the translucent SDF preview of where a
    /// primitive would drop, or `None` when no tool is armed. The renderer draws it when
    /// `Some` and nothing when `None`.
    ///
    /// **Session** state, on the same footing as [`view_mode`](Self::view_mode): an armed
    /// tool is how the workspace was left, not what the model is and not a preference. A
    /// dump taken mid-gesture and replayed must show the same pending drop — so the ghost
    /// travels into the dump (and never the shared document), the ADR 0024 route the
    /// viewer mode blazed. This phase populates it from config; the live cursor/click
    /// arming is a later slice.
    #[snapshot(session)]
    pub placement_ghost: Option<PlacementGhost>,
}

impl PanelState {
    /// Sensible defaults for the windowed app: like [`Default`] but with the view
    /// cube enabled (prototype `showCube: true`).
    pub fn with_view_cube_default() -> Self {
        let mut state = Self {
            show_view_cube: true,
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
        // issue #29 (grid rework S1): every scene carries exactly one Origin Point.
        // Idempotent, so calling it on an already-seeded scene is a no-op.
        self.scene.ensure_origin_point();
        // ADR 0003 Phase B: mint a stable NodeId for every node (idempotent).
        self.scene.ensure_node_ids();
    }

    /// Copy the active node's parameters into the inspector mirror
    /// ([`geometry`](Self::geometry) / [`material`](Self::material)) when it is a
    /// Tool, so the inspector edits the active selection. Called whenever the
    /// active node changes (selection or delete). A VoxelBody active node leaves the
    /// mirror untouched (its editor shows name + seed instead).
    pub fn sync_mirror_from_active(&mut self) {
        if let Some(node) = self.scene.active_node() {
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
                    // Size is voxel-granular (ADR 0003 §3f(0)): carry the canonical
                    // voxels AND the retained authored expression so the inspector
                    // seeds / re-emits the exact size the user typed.
                    size_voxels: shape.size_voxels,
                    size_measurements: if shape.has_retained_size_measurements() {
                        Some(Box::new(shape.size_measurements()))
                    } else {
                        None
                    },
                    // Density is document-level (ADR 0003 §3f(0)): the slider's
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
/// **ADR 0003 Phase C, slice C4a.** The panel no longer mutates `state.scene`
/// directly; instead every document mutation this frame is DESCRIBED as an
/// [`Intent`] pushed onto [`intents`](Self::intents), which the loop applies through
/// the shell's `AppCore::apply_intent` and folds the returned `IntentEffect`s into its
/// rebuild / points / selection
/// decisions. The remaining fields are NON-scene side effects (palette / export /
/// folder picker) the panel still only flags, plus the
/// [`frame_after_apply`](Self::frame_after_apply) auto-frame hint (which is a panel
/// UX concern — a size-slider `SetShape` re-frames, a shape-chip `SetShape` does
/// not, even though both are the same intent KIND — so it cannot be derived from the
/// intent alone and stays on the response).
#[derive(Debug, Clone, Default)]
pub struct PanelResponse {
    /// The document mutations the user made this frame, in emission order (ADR 0003
    /// Phase C C4a). The loop applies each through `AppCore::apply_intent` and merges
    /// the effects; the panel itself performs NONE of them.
    pub intents: Vec<Intent>,
    /// The caller should auto-frame the camera after applying this frame's intents
    /// (the typed successor of the old `size_or_density_changed || scene_changed`
    /// auto-frame trigger). Set by the panel for every emitted intent EXCEPT a pure
    /// shape-chip switch and a material pick (guard #1: a shape switch re-resolves at
    /// the same size and must NOT move the camera). A panel-level signal because the
    /// same intent KIND (`SetShape`) auto-frames from a size slider but not from a
    /// shape chip.
    pub frame_after_apply: bool,
    /// A palette tile was clicked this frame → apply a pseudo-random variant of
    /// this tile index as the active loaded material (M6).
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
    /// world centre + fit the distance). A VIEW action, NOT a document `Intent` (it
    /// is not undoable), so it rides on the response rather than `intents`. `None`
    /// when no Focus was requested.
    pub focus_node: Option<NodeId>,
    /// The tool the user armed from "+ Add" this frame → the shell starts the live
    /// placement flow (a translucent ghost follows the cursor, a stationary click drops
    /// the node). A VIEW action, NOT a document `Intent` (arming places nothing until a
    /// click), so it rides on the response rather than `intents`. `None` when nothing
    /// was armed this frame.
    pub armed_tool: Option<NodeSpec>,
}

impl PanelResponse {
    /// Push a mutation the user described this frame (ADR 0003 Phase C C4a). The loop
    /// applies it through `AppCore::apply_intent`; the panel never mutates the scene.
    pub(crate) fn emit(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    /// Push a mutation AND request an auto-frame after this frame's intents apply (the
    /// old `scene_changed` / `size_or_density_changed` behaviour). Used for structural
    /// edits and size/density edits — everything that re-frames; a shape-chip switch
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
    /// True while an export is running: the button is disabled (the shell serialises
    /// exports, so a second one can never be queued).
    pub in_flight: bool,
    /// The already-formatted line to show under the button, or `None`.
    pub status_line: Option<&'a str>,
}
