//! The dump round-trip gate (ADR 0022/0024): serialize → deserialize → apply → capture
//! must be the identity on [`AppConfig`], not just for one hand-picked fixture but across
//! the state space — every enum variant, the numeric boundaries, and seeded fills.
//!
//! Two properties, because the loop has two layers with different failure modes:
//!
//! * **Format identity** — `Dump::from_state → to_json → from_json → into_state`. No code
//!   in this layer may change a value, however extreme, so it runs on the full space
//!   (`u32::MAX`, `f32::MAX`, `i64::MIN`, …). Catches serde rot: a renamed key, a shim
//!   variant collapsing, a swapped field pair, float precision loss.
//! * **Apply/capture identity** — the format loop continued through
//!   `to_panel_state`/`apply_camera`/`home_view` and back out via `AppConfig::capture`.
//!   This is the seam where fields historically died (ADR 0024: four session fields
//!   classified as reaching the dump and hard-coded to defaults on restore). It runs on
//!   the space of states the application can actually be in, because the load path
//!   deliberately normalizes outside it — each such seam is pinned by its own test below
//!   rather than silently excused:
//!   - `onion_depth` clamps to `1..=8` on load;
//!   - an implicit home (`home_explicit == false`) collapses to `HomeView::default()`;
//!   - an absent or empty `scene` loads the default seed scene (then origin-point /
//!     node-id / sketch repair, all idempotent on a normalized scene);
//!   - a `sketch_mode` id that no longer resolves to a sketch node clears to `None`;
//!   - `Shortcuts::bind` steals a chord held elsewhere and the override map dedupes and
//!     re-sorts rows, so only distinct non-builtin chords in command order are identity.
//!
//! Deterministic by decision (the fable consult, 2026-07-28): per-variant sweeps plus a
//! fixed-seed LCG, no proptest — a failure names its case and replays exactly. Non-finite
//! floats are out of scope: `serde_json` cannot represent them and `to_json` errors, and
//! no live path produces them.

use camera::{HomeView, OrbitCamera, OrbitType, ProjectionMode};
use document::scene::{NodeId, PointId, Scene};
use ui::panel::{
    AngleSnap, ArmedConstraint, ConstraintVerb, OrbitMode, PlacementPivot, PlacementSnap,
    PositionSnap, SignalStackState, SketchEntity, SketchTool, ViewMode,
};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use voxel_worker::artifacts::Dump;
use voxel_worker::settings::{
    AppConfig, ArmedToolConfig, SelectionConfig, SelectionTargetConfig, ShortcutBindingConfig,
    ShortcutCommandConfig, ShortcutsConfig,
};

/// A fixed-constant LCG (Knuth's MMIX multiplier). Deterministic and seed-addressable so a
/// red case reproduces from its printed label alone.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn index(&mut self, len: usize) -> usize {
        ((self.next() >> 33) as usize) % len
    }

    fn flag(&mut self) -> bool {
        self.next() & (1 << 40) != 0
    }

    fn pick<T: Copy>(&mut self, options: &[T]) -> T {
        options[self.index(options.len())]
    }
}

/// The f32 boundary set. Finite only (see module docs); `f32::MAX`'s shortest decimal
/// narrows back exactly through serde_json's f64 path, which is part of what is pinned.
const F32_BOUNDARIES: &[f32] = &[
    0.0,
    1.0,
    -1.0,
    0.25,
    -3.75,
    std::f32::consts::TAU,
    1.0e-6,
    -1.0e6,
    f32::MIN_POSITIVE,
    f32::MAX,
];

const U32_BOUNDARIES: &[u32] = &[0, 1, 2, 63, 64, 65, 4096, u32::MAX];

const I64_BOUNDARIES: &[i64] = &[0, 1, -1, 7, -3, i64::MAX, i64::MIN];

fn vec3(rng: &mut Lcg, values: &[f32]) -> [f32; 3] {
    [rng.pick(values), rng.pick(values), rng.pick(values)]
}

fn some_chord(key: egui::Key) -> Option<egui::KeyboardShortcut> {
    Some(egui::KeyboardShortcut::new(
        egui::Modifiers::CTRL | egui::Modifiers::ALT,
        key,
    ))
}

/// Every override command in declaration order, paired with a chord no built-in holds on
/// any platform (`Ctrl+Alt+F<n>`), so a full override set survives `bind` untouched.
const OVERRIDE_COMMANDS: [(ShortcutCommandConfig, egui::Key); 9] = [
    (ShortcutCommandConfig::AcceptCommand, egui::Key::F1),
    (ShortcutCommandConfig::CancelCommand, egui::Key::F2),
    (ShortcutCommandConfig::DeleteSelection, egui::Key::F3),
    (ShortcutCommandConfig::Undo, egui::Key::F4),
    (ShortcutCommandConfig::Redo, egui::Key::F5),
    (ShortcutCommandConfig::PlaceOrbitCenter, egui::Key::F6),
    (ShortcutCommandConfig::ResetOrbitCenter, egui::Key::F7),
    (ShortcutCommandConfig::EnterConstrainedOrbit, egui::Key::F8),
    (ShortcutCommandConfig::ExportRepro, egui::Key::F9),
];

fn shortcuts_from(rng: &mut Lcg) -> ShortcutsConfig {
    let mut overrides = Vec::new();
    for (command, key) in OVERRIDE_COMMANDS {
        if rng.flag() {
            overrides.push(ShortcutBindingConfig {
                command,
                shortcut: if rng.flag() { some_chord(key) } else { None },
            });
        }
    }
    ShortcutsConfig { overrides }
}

/// Distinct targets of both kinds in pick order — [`Selection`] is a set, so a duplicate
/// target would collapse and break identity by design.
const SELECTION_POOL: [SelectionTargetConfig; 4] = [
    SelectionTargetConfig::Node(NodeId(3)),
    SelectionTargetConfig::ReferencePoint(PointId(2)),
    SelectionTargetConfig::Node(NodeId(11)),
    SelectionTargetConfig::ReferencePoint(PointId(5)),
];

fn selection_from(rng: &mut Lcg) -> SelectionConfig {
    let count = rng.index(SELECTION_POOL.len() + 1);
    SelectionConfig {
        targets: SELECTION_POOL[..count].to_vec(),
    }
}

fn armed_tool_from(rng: &mut Lcg, offsets: &[i64]) -> Option<ArmedToolConfig> {
    if !rng.flag() {
        return None;
    }
    Some(ArmedToolConfig {
        shape_kind: rng.pick(&[
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Sphere,
            ShapeKind::Torus,
            ShapeKind::Box,
        ]),
        // Sizes the placement path accepts as-is: `SdfShape::from_voxels` clamps, so a
        // size outside the legal range is a normalization, not an identity case.
        size_voxels: [rng.pick(&[1, 4, 16, 80]); 3],
        wall_blocks: rng.pick(&[1, 2, 3]),
        pending_offset_voxels: rng
            .flag()
            .then(|| [rng.pick(offsets), rng.pick(offsets), rng.pick(offsets)]),
    })
}

/// A state drawn from the FULL space, exhaustive struct literal — a new [`AppConfig`]
/// field fails this build until it gains a generator arm, mirroring the capture law.
fn arbitrary_config(rng: &mut Lcg, scene: Option<Scene>) -> AppConfig {
    AppConfig {
        scene,
        voxels_per_block: rng.pick(U32_BOUNDARIES),
        projection_mode: rng.pick(&[ProjectionMode::Perspective, ProjectionMode::Orthographic]),
        material: rng.pick(&[
            MaterialChoice::Stone,
            MaterialChoice::Wood,
            MaterialChoice::Plain,
        ]),
        axes_on_top: rng.flag(),
        applied_block_label: rng.flag().then(|| "Granite ▤ 花崗岩".to_string()),
        snap_to_blocks: rng.flag(),
        onion_skin: rng.flag(),
        onion_depth: rng.pick(U32_BOUNDARIES),
        orbit_theta: rng.pick(F32_BOUNDARIES),
        orbit_phi: rng.pick(F32_BOUNDARIES),
        orbit_distance: rng.pick(F32_BOUNDARIES),
        orbit_target: vec3(rng, F32_BOUNDARIES),
        orbit_center: vec3(rng, F32_BOUNDARIES),
        home_theta: rng.pick(F32_BOUNDARIES),
        home_phi: rng.pick(F32_BOUNDARIES),
        home_distance: rng.pick(F32_BOUNDARIES),
        home_explicit: rng.flag(),
        window_size: [rng.pick(U32_BOUNDARIES), rng.pick(U32_BOUNDARIES)],
        view_mode: rng.pick(&[ViewMode::Normal, ViewMode::OnionFog, ViewMode::ShowBooleans]),
        stack: SignalStackState {
            folded: rng.flag(),
            viewport_open: rng.flag(),
            onion_open: rng.flag(),
            grids_open: rng.flag(),
        },
        debug_face_orientation: rng.flag(),
        debug_brick_faces: rng.flag(),
        armed_tool: armed_tool_from(rng, I64_BOUNDARIES),
        placement_snap: PlacementSnap {
            position: rng.pick(&[
                PositionSnap::NoSnap,
                PositionSnap::Block,
                PositionSnap::Voxel,
            ]),
            angle: rng.pick(&[AngleSnap::Continuous, AngleSnap::Deg15]),
            pivot: rng.pick(&[PlacementPivot::Base, PlacementPivot::VolumetricCenter]),
        },
        sketch_mode: rng.flag().then_some(NodeId(9)),
        sketch_tool: rng.pick(&[
            SketchTool::Select,
            SketchTool::AddPoint,
            SketchTool::Polyline,
            SketchTool::Rectangle,
        ]),
        armed_constraint: rng.flag().then(|| {
            ArmedConstraint::from_parts(
                rng.pick(&[
                    ConstraintVerb::Horizontal,
                    ConstraintVerb::Vertical,
                    ConstraintVerb::Fix,
                ]),
                Vec::new(),
            )
        }),
        sketch_snap: rng.pick(&[
            PositionSnap::NoSnap,
            PositionSnap::Block,
            PositionSnap::Voxel,
        ]),
        default_orbit_type: rng.pick(&[OrbitType::Constrained, OrbitType::Free]),
        orbit_mode: rng.pick(&[
            OrbitMode::Off,
            OrbitMode::UsingDefault,
            OrbitMode::Named(OrbitType::Constrained),
            OrbitMode::Named(OrbitType::Free),
        ]),
        selection: selection_from(rng),
        shortcuts: shortcuts_from(rng),
    }
}

/// A state the application can actually be in — the module docs' normalization seams
/// respected, so apply → capture owes an exact identity on it.
fn applyable_config(rng: &mut Lcg, seed_scene: &Scene) -> AppConfig {
    let home_explicit = rng.flag();
    let home = if home_explicit {
        [rng.pick(F32_BOUNDARIES), rng.pick(F32_BOUNDARIES), 18.5]
    } else {
        let default = HomeView::default();
        [default.theta, default.phi, default.distance]
    };
    AppConfig {
        scene: Some(seed_scene.clone()),
        voxels_per_block: rng.pick(&[1, 2, 16, 24, 64]),
        onion_depth: rng.pick(&[1, 2, 5, 8]),
        home_theta: home[0],
        home_phi: home[1],
        home_distance: home[2],
        home_explicit,
        sketch_mode: None,
        armed_tool: armed_tool_from(rng, &[0, 7, -3, 5, 200, -41]),
        window_size: [
            rng.pick(&[1, 640, 1280, 3840]),
            rng.pick(&[1, 480, 800, 2160]),
        ],
        ..arbitrary_config(rng, None)
    }
}

/// The normalized default seed scene, exactly as loading a scene-less config produces it.
fn seed_scene() -> Scene {
    AppConfig::default().to_panel_state().scene
}

fn format_round_trip(label: &str, config: &AppConfig) {
    let json = Dump::from_state(config)
        .to_json()
        .unwrap_or_else(|error| panic!("{label}: serialize failed: {error}"));
    let restored = Dump::from_json(&json)
        .unwrap_or_else(|error| panic!("{label}: deserialize failed: {error}"))
        .into_state();
    assert_eq!(
        &restored, config,
        "{label}: the JSON trip changed the state (dump written and reread by the SAME build)"
    );
}

/// The whole loop `shot --from-config` and a relaunch actually run: through the flat JSON,
/// into the live panel + camera, and back out through the one capture function.
fn apply_capture_round_trip(label: &str, config: &AppConfig) {
    let json = Dump::from_state(config)
        .to_json()
        .unwrap_or_else(|error| panic!("{label}: serialize failed: {error}"));
    let loaded = Dump::from_json(&json)
        .unwrap_or_else(|error| panic!("{label}: deserialize failed: {error}"))
        .into_state();
    let panel = loaded.to_panel_state();
    let mut camera = OrbitCamera::default();
    loaded.apply_camera(&mut camera);
    let recaptured = AppConfig::capture(&panel, &camera, loaded.home_view(), loaded.window_size);
    assert_eq!(
        &recaptured, config,
        "{label}: serialize → deserialize → apply → capture changed the state"
    );
}

/// Every variant of every persisted enum, pinned one at a time on an LCG-filled applyable
/// base, through the FULL loop. The base varies per case so a coincidence with a default
/// cannot hide a collapsed variant.
#[test]
fn every_enum_variant_survives_the_full_loop() {
    let scene = seed_scene();
    let mut cases: Vec<(String, AppConfig)> = Vec::new();
    let mut case_seed: u64 = 0;
    let mut base = || {
        case_seed += 1;
        applyable_config(&mut Lcg::new(0xE1 + case_seed), &scene)
    };

    for projection in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
        let mut config = base();
        config.projection_mode = projection;
        cases.push((format!("projection_mode={projection:?}"), config));
    }
    for material in [
        MaterialChoice::Stone,
        MaterialChoice::Wood,
        MaterialChoice::Plain,
    ] {
        let mut config = base();
        config.material = material;
        cases.push((format!("material={material:?}"), config));
    }
    for view_mode in [ViewMode::Normal, ViewMode::OnionFog, ViewMode::ShowBooleans] {
        let mut config = base();
        config.view_mode = view_mode;
        cases.push((format!("view_mode={view_mode:?}"), config));
    }
    for sketch_tool in [
        SketchTool::Select,
        SketchTool::AddPoint,
        SketchTool::Polyline,
        SketchTool::Rectangle,
    ] {
        let mut config = base();
        config.sketch_tool = sketch_tool;
        cases.push((format!("sketch_tool={sketch_tool:?}"), config));
    }
    // ADR 0035 Decision 15: each verb, and a gesture holding a pick — the picks are the half of
    // an armed constraint that a shim mirroring only the verb would silently drop.
    for verb in [
        ConstraintVerb::Horizontal,
        ConstraintVerb::Vertical,
        ConstraintVerb::Fix,
    ] {
        let mut config = base();
        config.armed_constraint = Some(ArmedConstraint::from_parts(verb, Vec::new()));
        cases.push((format!("armed_constraint={verb:?}"), config));
    }
    for (label, picked) in [
        ("point", vec![SketchEntity::Point(4)]),
        ("segment", vec![SketchEntity::Segment(6)]),
    ] {
        let mut config = base();
        let verb = match label {
            "point" => ConstraintVerb::Fix,
            _ => ConstraintVerb::Horizontal,
        };
        config.armed_constraint = Some(ArmedConstraint::from_parts(verb, picked));
        cases.push((format!("armed_constraint_pick={label}"), config));
    }
    for orbit_type in [OrbitType::Constrained, OrbitType::Free] {
        let mut config = base();
        config.default_orbit_type = orbit_type;
        cases.push((format!("default_orbit_type={orbit_type:?}"), config));
    }
    for orbit_mode in [
        OrbitMode::Off,
        OrbitMode::UsingDefault,
        OrbitMode::Named(OrbitType::Constrained),
        OrbitMode::Named(OrbitType::Free),
    ] {
        let mut config = base();
        config.orbit_mode = orbit_mode;
        cases.push((format!("orbit_mode={orbit_mode:?}"), config));
    }
    for position in [
        PositionSnap::NoSnap,
        PositionSnap::Block,
        PositionSnap::Voxel,
    ] {
        let mut config = base();
        config.placement_snap.position = position;
        cases.push((format!("placement_snap.position={position:?}"), config));
    }
    for sketch_snap in [
        PositionSnap::NoSnap,
        PositionSnap::Block,
        PositionSnap::Voxel,
    ] {
        let mut config = base();
        config.sketch_snap = sketch_snap;
        cases.push((format!("sketch_snap={sketch_snap:?}"), config));
    }
    for angle in [AngleSnap::Continuous, AngleSnap::Deg15] {
        let mut config = base();
        config.placement_snap.angle = angle;
        cases.push((format!("placement_snap.angle={angle:?}"), config));
    }
    for pivot in [PlacementPivot::Base, PlacementPivot::VolumetricCenter] {
        let mut config = base();
        config.placement_snap.pivot = pivot;
        cases.push((format!("placement_snap.pivot={pivot:?}"), config));
    }
    for shape_kind in [
        ShapeKind::Cylinder,
        ShapeKind::Tube,
        ShapeKind::Sphere,
        ShapeKind::Torus,
        ShapeKind::Box,
    ] {
        let mut config = base();
        config.armed_tool = Some(ArmedToolConfig {
            shape_kind,
            size_voxels: [16, 8, 16],
            wall_blocks: 2,
            pending_offset_voxels: Some([7, -3, 5]),
        });
        cases.push((format!("armed_tool.shape_kind={shape_kind:?}"), config));
    }
    {
        let mut config = base();
        config.armed_tool = None;
        cases.push(("armed_tool=None".to_string(), config));
    }
    {
        let mut config = base();
        config.armed_tool = Some(ArmedToolConfig {
            shape_kind: ShapeKind::Sphere,
            size_voxels: [4, 4, 4],
            wall_blocks: 1,
            pending_offset_voxels: None,
        });
        cases.push(("armed_tool.pending=None".to_string(), config));
    }
    for (command, key) in OVERRIDE_COMMANDS {
        for shortcut in [some_chord(key), None] {
            let mut config = base();
            config.shortcuts = ShortcutsConfig {
                overrides: vec![ShortcutBindingConfig { command, shortcut }],
            };
            let bound = if shortcut.is_some() {
                "bound"
            } else {
                "unbound"
            };
            cases.push((format!("shortcuts.{command:?}={bound}"), config));
        }
    }
    {
        let mut config = base();
        config.shortcuts = ShortcutsConfig {
            overrides: OVERRIDE_COMMANDS
                .iter()
                .map(|(command, key)| ShortcutBindingConfig {
                    command: *command,
                    shortcut: some_chord(*key),
                })
                .collect(),
        };
        cases.push(("shortcuts.all_nine_overridden".to_string(), config));
    }
    for count in 0..=4 {
        let mut config = base();
        let mut rng = Lcg::new(0x5E1 + count as u64);
        loop {
            config.selection = selection_from(&mut rng);
            if config.selection.targets.len() == count {
                break;
            }
        }
        cases.push((format!("selection.targets.len={count}"), config));
    }

    for (label, config) in &cases {
        apply_capture_round_trip(label, config);
    }
}

/// The numeric boundary values on the fields the apply path accepts unclamped, plus both
/// poles of every persisted bool, through the FULL loop.
#[test]
fn numeric_boundaries_and_bool_poles_survive_the_full_loop() {
    let scene = seed_scene();
    let mut cases: Vec<(String, AppConfig)> = Vec::new();
    let mut case_seed: u64 = 0;
    let mut base = || {
        case_seed += 1;
        applyable_config(&mut Lcg::new(0xB0 + case_seed), &scene)
    };

    for density in [1u32, 2, 63, 64] {
        let mut config = base();
        config.voxels_per_block = density;
        cases.push((format!("voxels_per_block={density}"), config));
    }
    for depth in [1u32, 8] {
        let mut config = base();
        config.onion_depth = depth;
        cases.push((format!("onion_depth={depth}"), config));
    }
    for &value in F32_BOUNDARIES {
        let mut config = base();
        config.orbit_theta = value;
        config.orbit_phi = -value;
        config.orbit_distance = value;
        config.orbit_target = [value, 0.0, -value];
        config.orbit_center = [-value, value, 0.0];
        cases.push((format!("camera_f32={value:e}"), config));
    }
    for size in [[1u32, 1], [3840, 2160]] {
        let mut config = base();
        config.window_size = size;
        cases.push((format!("window_size={size:?}"), config));
    }
    for label in [
        None,
        Some(String::new()),
        Some("Granite ▤ 花崗岩".to_string()),
    ] {
        let mut config = base();
        config.applied_block_label = label.clone();
        cases.push((format!("applied_block_label={label:?}"), config));
    }
    for pole in [false, true] {
        let stack = SignalStackState {
            folded: pole,
            viewport_open: !pole,
            onion_open: pole,
            grids_open: !pole,
        };
        let mut config = base();
        config.axes_on_top = pole;
        config.snap_to_blocks = !pole;
        config.onion_skin = pole;
        config.debug_face_orientation = !pole;
        config.debug_brick_faces = pole;
        config.stack = stack;
        cases.push((format!("bool_pole={pole}"), config));
    }
    {
        let mut config = base();
        config.home_explicit = true;
        config.home_theta = 2.34;
        config.home_phi = 1.11;
        config.home_distance = 18.0;
        cases.push(("home_explicit=true".to_string(), config));
    }

    for (label, config) in &cases {
        apply_capture_round_trip(label, config);
    }
}

/// Seeded whole-state fills through the FULL loop — the cross-field interaction sweep the
/// one-factor cases above cannot give.
#[test]
fn seeded_fills_survive_the_full_loop() {
    let scene = seed_scene();
    for seed in 0..32 {
        let config = applyable_config(&mut Lcg::new(seed), &scene);
        apply_capture_round_trip(&format!("fill_seed={seed}"), &config);
    }
}

/// Seeded fills from the FULL value space through the FORMAT loop alone — no code between
/// `from_state` and `into_state` is entitled to change any value, however extreme.
#[test]
fn extreme_fills_survive_the_format_loop() {
    for seed in 0..64 {
        let mut rng = Lcg::new(0xF0F0 + seed);
        let scene = rng.flag().then(seed_scene);
        let config = arbitrary_config(&mut rng, scene);
        format_round_trip(&format!("format_seed={seed}"), &config);
    }
}

// --- The normalization seams, pinned as deliberate rather than silently excused. ---

/// `onion_depth` clamps to `1..=8` on load: the scrubber's range, applied at the one door.
#[test]
fn onion_depth_outside_the_scrubber_range_clamps_on_load() {
    let scene = seed_scene();
    for (stored, loaded) in [(0u32, 1u32), (99, 8)] {
        let mut config = applyable_config(&mut Lcg::new(0xD0 + stored as u64), &scene);
        config.onion_depth = stored;
        let panel = config.to_panel_state();
        assert_eq!(
            panel.layer_range.onion_depth, loaded,
            "onion_depth {stored} must clamp to {loaded}"
        );
    }
}

/// An implicit home ignores whatever angles a prior session persisted, so a changed code
/// default reaches existing configs (#13). The stored values are deliberately lost.
#[test]
fn an_implicit_home_collapses_to_the_current_default_on_capture() {
    let scene = seed_scene();
    let mut config = applyable_config(&mut Lcg::new(0xA5), &scene);
    config.home_explicit = false;
    config.home_theta = 9.9;
    config.home_phi = 8.8;
    config.home_distance = 777.0;
    let home = config.home_view();
    let default = HomeView::default();
    assert!(!home.explicitly_set);
    assert_eq!(
        (home.theta, home.phi, home.distance),
        (default.theta, default.phi, default.distance),
        "an implicit home must track the code default, not the stored angles"
    );
}

/// A scene-less config and an empty persisted scene both load the SAME default seed scene
/// — loading never yields an empty document.
#[test]
fn an_absent_or_empty_scene_loads_the_default_seed() {
    for scene in [None, Some(Scene::default())] {
        let label = if scene.is_some() { "empty" } else { "absent" };
        let config = AppConfig {
            scene,
            ..AppConfig::default()
        };
        let panel = config.to_panel_state();
        assert!(
            !panel.scene.roots.is_empty(),
            "an {label} scene must load the seed, not an empty document"
        );
        assert_eq!(
            panel.scene,
            seed_scene(),
            "an {label} scene must load the SAME seed a fresh config gets"
        );
    }
}

/// A restored `sketch_mode` pointing at an id that is not a live sketch node clears to
/// `None`, so a stale id cannot trap the mode (ADR 0028).
#[test]
fn a_stale_sketch_mode_clears_on_load() {
    let scene = seed_scene();
    let mut config = applyable_config(&mut Lcg::new(0x51), &scene);
    config.sketch_mode = Some(NodeId(u64::MAX));
    let panel = config.to_panel_state();
    assert_eq!(panel.sketch_mode, None);
}

/// A stored override whose chord a built-in already holds is a conflict the load resolves:
/// `bind` steals the chord and records an explicit unbind for the previous holder, so the
/// recapture carries MORE rows than were stored. Deliberate — two commands must never race
/// for one press — and therefore outside the identity, pinned here instead.
#[test]
fn a_stored_chord_conflict_resolves_to_a_steal_plus_unbind_on_load() {
    // Enter is AcceptCommand's built-in on every platform; give it to ExportRepro.
    let enter = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Enter);
    let stored = ShortcutsConfig {
        overrides: vec![ShortcutBindingConfig {
            command: ShortcutCommandConfig::ExportRepro,
            shortcut: Some(enter),
        }],
    };
    let recaptured = ShortcutsConfig::from_shortcuts(&stored.to_shortcuts());
    assert_eq!(
        recaptured.overrides,
        vec![
            ShortcutBindingConfig {
                command: ShortcutCommandConfig::AcceptCommand,
                shortcut: None,
            },
            ShortcutBindingConfig {
                command: ShortcutCommandConfig::ExportRepro,
                shortcut: Some(enter),
            },
        ],
        "the steal must unbind the built-in holder explicitly, in command order"
    );
}
