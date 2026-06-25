//! The scene (assembly) model — ADR 0001, sequence step 1.
//!
//! Today the app has exactly one producer, smuggled in through
//! [`GeometryParams`](crate::panel::GeometryParams) (the SDF shape) plus a
//! `debug_clouds: bool` selector. ADR 0001 replaces that single-producer
//! assumption with a **Scene**: an assembly graph of **nodes**, each wrapping a
//! producer plus a placement. This module introduces that model and routes ALL
//! voxel resolution through it.
//!
//! **Step 1 scope (this file):** the data model exists in full (so later steps
//! are data changes, not rewrites), but only the two leaves that exist today are
//! actually resolved:
//!
//!   * [`NodeContent::Tool`] — a *parametric* producer ([`SdfShape`]) that carries
//!     the Tool's single [`MaterialChoice`].
//!   * [`NodeContent::Part`] — a *static* voxel body; today the only variant is
//!     [`Part::DebugClouds`].
//!
//! [`NodeContent::Group`] and [`NodeContent::Instance`] (recursion + reuse) exist
//! as types but are intentionally not resolved yet — see the `// step 4` markers
//! in [`Scene::resolve_region`].
//!
//! ## Identical-behaviour guarantee
//!
//! The producer trait ([`VoxelProducer`]) does **not** change: producers still
//! emit content centred at the origin. The Scene's new job is **compositing** —
//! walk the node tree, resolve each visible leaf into its own local grid, and
//! **stamp** it (under the node's transform) into the output grid. For a one-node
//! scene whose region is the node's full extent with a zero offset, the stamp is
//! the identity, so the resulting [`VoxelGrid`] is bit-for-bit what
//! `SdfShape::resolve` / `DebugCloudField::resolve` produce today (same
//! dimensions, same occupied set). See `tool_scene_matches_bare_producer` below.

use crate::debug_clouds::DebugCloudField;
use crate::panel::{GeometryParams, MaterialChoice};
use crate::voxel::{SdfShape, VoxelGrid, VoxelProducer};

/// The working volume the scene resolves into, expressed in **whole blocks**
/// (ADR 0001 "Scale": the canvas is the user-set stock / build volume). Step 1
/// always resolves the whole extent as a single region, so this equals the lone
/// node's block extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionBlocks {
    /// Size of the region in whole blocks (X, Y, Z).
    pub size_blocks: [u32; 3],
}

impl RegionBlocks {
    /// A region of the given whole-block size.
    pub fn new(size_blocks: [u32; 3]) -> Self {
        Self { size_blocks }
    }
}

/// A reusable identifier for a [`Tool`-or-`Part`](NodeContent) definition that an
/// [`NodeContent::Instance`] points at (ADR 0001: reuse by reference). Step 1
/// never constructs an Instance, so this is a forward-declared type only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

/// How a node combines with the nodes resolved before it. v1 only ever
/// constructs [`CombineOp::Union`]; the enum exists so subtract / intersect /
/// override become a data change on the node rather than a re-architecture
/// (ADR 0001 decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CombineOp {
    /// Additive: the output occupied set is the OR of the contributing nodes; on
    /// overlap the later node wins the material.
    #[default]
    Union,
    // future: Subtract, Intersect, Override, …
}

/// A node's LOCAL placement. v1 exposes integer block translation only, but the
/// type targets a full affine (translation + rotation + scale) so rotation /
/// scale (with voxel resampling) slot in later without a rewrite (ADR 0001
/// decision 3). In step 1 the offset is always `[0, 0, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NodeTransform {
    /// Translation in whole blocks (X, Y, Z).
    pub offset_blocks: [i32; 3],
    // future: rotation, scale → a general affine.
}

impl NodeTransform {
    /// The identity transform (zero offset) — the only transform step 1 uses.
    pub fn identity() -> Self {
        Self::default()
    }
}

/// A *static* voxel body with no meaningful generation parameters — dropped in
/// as-is (ADR 0001). v1 has one variant; future variants are saved chiseled
/// blocks and imported `.vox` bodies, each carrying baked per-voxel materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// The debug cloud field (several distinct billowy fBm blobs) — "a part with
    /// one trivial knob" (the seed).
    DebugClouds {
        /// Seed for the deterministic placement + noise permutation.
        seed: u32,
    },
    // future: SavedBody(VoxelBlob), ImportedVox(...).
}

/// What a node *is*: a leaf producer (Tool or Part) or an interior assembly
/// (Group or Instance).
///
/// Step 1 resolves only the two leaf kinds; `Group` / `Instance` are present as
/// types but unimplemented in [`Scene::resolve_region`] (recursion + instancing
/// arrive in step 4).
#[derive(Debug, Clone)]
pub enum NodeContent {
    /// A parametric producer (an [`SdfShape`]) plus the single material the Tool
    /// assigns to every voxel it emits. Step 1 keeps the existing
    /// [`MaterialChoice`]; a richer material table is a later step.
    Tool {
        /// The parametric primitive to resolve.
        shape: SdfShape,
        /// The single material this Tool stamps onto its voxels.
        material: MaterialChoice,
    },
    /// A static voxel body, dropped in as-is.
    Part(Part),
    /// An owned, one-off sub-assembly. **Not resolved in step 1** (step 4).
    Group(Vec<Node>),
    /// A reuse-by-reference of a definition. **Not resolved in step 1** (step 4).
    Instance(DefId),
}

/// One placed node in the assembly graph: a producer (or sub-assembly) plus its
/// local placement and combine operation.
#[derive(Debug, Clone)]
pub struct Node {
    /// Human-readable name (for the future node-list UI).
    pub name: String,
    /// LOCAL transform; composes with ancestors' (`world = parent ∘ local`).
    /// Step 1 only ever uses the identity (zero offset).
    pub transform: NodeTransform,
    /// How this node combines with earlier ones. v1: always [`CombineOp::Union`].
    pub operation: CombineOp,
    /// Whether the node contributes to resolution (a hidden node stamps nothing).
    pub visible: bool,
    /// What the node is.
    pub content: NodeContent,
}

impl Node {
    /// A visible, identity-placed, union node wrapping `content`.
    pub fn new(name: impl Into<String>, content: NodeContent) -> Self {
        Self {
            name: name.into(),
            transform: NodeTransform::identity(),
            operation: CombineOp::Union,
            visible: true,
            content,
        }
    }
}

/// A reusable sub-assembly (e.g. "house") placed by [`NodeContent::Instance`]
/// (ADR 0001). Step 1 never constructs or resolves one; it exists so the model is
/// complete. The top-level assembly is also an `AssemblyDef` (its `root`).
#[derive(Debug, Clone)]
pub struct AssemblyDef {
    /// The definition's identifier (referenced by an `Instance`).
    pub id: DefId,
    /// Human-readable name.
    pub name: String,
    /// The nodes that make up this assembly.
    pub children: Vec<Node>,
}

/// The scene (assembly): a list of placed nodes resolved into the shared
/// [`VoxelGrid`] truth. ADR 0001's full model carries reusable `definitions` too;
/// step 2 adds the flat node list plus the `active` selection that drives the
/// inspector. `definitions` (recursion + instancing) stay deferred to step 4.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    /// The top-level assembly's nodes, resolved in order (later nodes win on
    /// overlap under [`CombineOp::Union`]).
    pub nodes: Vec<Node>,
    /// Index into [`nodes`](Self::nodes) of the active/selected node — the one the
    /// inspector edits. `None` when the scene is empty. Step 2 keeps this valid
    /// (clamped) across add/delete.
    pub active: Option<usize>,
}

impl Scene {
    /// A scene with a single node — the shape every one-node call site builds. The
    /// lone node is the active selection.
    pub fn single_node(node: Node) -> Self {
        Self {
            nodes: vec![node],
            active: Some(0),
        }
    }

    /// The active node, if any.
    pub fn active_node(&self) -> Option<&Node> {
        self.active.and_then(|index| self.nodes.get(index))
    }

    /// The active node mutably, if any (the inspector edits through this).
    pub fn active_node_mut(&mut self) -> Option<&mut Node> {
        match self.active {
            Some(index) => self.nodes.get_mut(index),
            None => None,
        }
    }

    /// Append `node` and make it the active selection. Returns the new index.
    pub fn add_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        let index = self.nodes.len() - 1;
        self.active = Some(index);
        index
    }

    /// Remove the node at `index`, keeping the `active` selection valid: the
    /// active index shifts down when a node before it is removed, clamps to the new
    /// last node when the removed node was the active (or last) one, and becomes
    /// `None` when the scene empties. Out-of-range indices are ignored.
    pub fn remove_node(&mut self, index: usize) {
        if index >= self.nodes.len() {
            return;
        }
        self.nodes.remove(index);
        if self.nodes.is_empty() {
            self.active = None;
            return;
        }
        let last = self.nodes.len() - 1;
        self.active = Some(match self.active {
            Some(active) if active > index => active - 1,
            Some(active) => active.min(last),
            None => last,
        });
    }

    /// Build the one-node Tool scene that reproduces today's single-shape
    /// behaviour from the panel's [`GeometryParams`] plus the active
    /// [`MaterialChoice`]. The node is a [`NodeContent::Tool`] wrapping the SDF
    /// shape, carrying `material` as its single material.
    ///
    /// Step 2 removed the `debug_clouds: bool` selector — "Clouds" is now an
    /// Add-a-Part action in the node list ([`Part::DebugClouds`]), not a mode of
    /// the geometry. So this constructor only ever builds a Tool; the back-compat
    /// config load (a single persisted geometry) routes through here.
    pub fn from_geometry(geometry: GeometryParams, material: MaterialChoice) -> Self {
        Self::single_node(Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_geometry(geometry),
                material,
            },
        ))
    }

    /// The whole-block extent of the scene: the per-axis size of the bounding box
    /// that encompasses every placed leaf node (ADR 0001 step 3). Each leaf
    /// occupies `offset_blocks ± size/2`; the composite extent is the union of
    /// those boxes (`max_corner - min_corner` per axis). With every node at a zero
    /// offset this reduces to the per-axis MAX of the node sizes (the step-2
    /// behaviour). A Part-only node (the cloud field, which has no intrinsic size)
    /// contributes no box and adopts whatever extent the Tools establish.
    ///
    /// Returns a zero-sized region when no leaf has an intrinsic size.
    pub fn full_extent_blocks(&self, voxels_per_block: u32) -> RegionBlocks {
        match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => RegionBlocks::new([
                (max_corner[0] - min_corner[0]) as u32,
                (max_corner[1] - min_corner[1]) as u32,
                (max_corner[2] - min_corner[2]) as u32,
            ]),
            None => RegionBlocks::new([0, 0, 0]),
        }
    }

    /// The composite bounding box of all placed leaf nodes, in **whole-block**
    /// coordinates: `(min_corner, max_corner)` where each leaf with intrinsic
    /// `size_blocks` placed at `offset_blocks` spans
    /// `[offset - size/2, offset - size/2 + size]`. `None` when no leaf has an
    /// intrinsic size (a Part-only scene). Drives both [`full_extent_blocks`] (the
    /// size) and the recentre in [`resolve_region`] (centring the composite so its
    /// world positions sit symmetrically about the origin — what the renderer and
    /// camera assume).
    ///
    /// Block extents are split into a low/high half (`floor(size/2)` below the
    /// centre, the remainder above) so an odd block size keeps the same parity the
    /// voxel-space resolution uses, and the returned box is exact in blocks.
    fn placed_extent_blocks(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        let mut min_corner = [i32::MAX; 3];
        let mut max_corner = [i32::MIN; 3];
        let mut any = false;
        for node in &self.nodes {
            let Some(size_blocks) = leaf_size_blocks(&node.content, voxels_per_block) else {
                continue;
            };
            any = true;
            let offset = node.transform.offset_blocks;
            for axis in 0..3 {
                let half_low = (size_blocks[axis] / 2) as i32;
                let low = offset[axis] - half_low;
                let high = low + size_blocks[axis] as i32;
                min_corner[axis] = min_corner[axis].min(low);
                max_corner[axis] = max_corner[axis].max(high);
            }
        }
        any.then_some((min_corner, max_corner))
    }

    /// Resolve `region` into a fresh [`VoxelGrid`] by a union tree-walk: each
    /// visible leaf producer is resolved into its own local grid and **stamped**
    /// into the output under the node's transform.
    ///
    /// `voxels_per_block` is the application density (ADR 0001 "Density": a global
    /// setting, default 16, that the scene reads at resolve time).
    ///
    /// `lod` is the level-of-detail seam required by ADR 0001 ("Deferred: LOD").
    /// It is **always `0`** (full resolution) for now; the parameter exists from
    /// day one so a future LOD level (which would downsample a chunk before
    /// meshing) is a possible change rather than a signature break. Step 1
    /// asserts it is `0`.
    ///
    /// **Identical-behaviour guarantee:** for a one-node scene whose `region`
    /// equals the node's full extent with a zero offset, the stamp is the
    /// identity, so the result equals what the bare producer emits today.
    pub fn resolve_region(
        &self,
        region: RegionBlocks,
        voxels_per_block: u32,
        lod: u32,
    ) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "step 1 only resolves full resolution (lod 0)");

        let region_dimensions = [
            region.size_blocks[0] * voxels_per_block,
            region.size_blocks[1] * voxels_per_block,
            region.size_blocks[2] * voxels_per_block,
        ];
        let mut output = VoxelGrid::new(region_dimensions);

        // Recentre the composite so its world positions sit symmetrically about
        // the origin (what the renderer + camera auto-frame assume). Each producer
        // emits voxels centred on ITS OWN grid; a node's placed centre in the
        // composite's voxel space is `offset_voxels`, and the whole composite's
        // centre is `((min + max) / 2) * voxels_per_block`. Subtracting that centre
        // from every node's translation lands the composite centred in `output`.
        // With a single zero-offset node the composite centre is the node's own
        // centre, so the shift is zero — the step-2 identity is preserved.
        let recentre_voxels = match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => [
                ((min_corner[0] + max_corner[0]) * voxels_per_block as i32) / 2,
                ((min_corner[1] + max_corner[1]) * voxels_per_block as i32) / 2,
                ((min_corner[2] + max_corner[2]) * voxels_per_block as i32) / 2,
            ],
            None => [0, 0, 0],
        };

        for node in &self.nodes {
            if !node.visible {
                continue;
            }
            // The translation applied to this node's producer voxels: its block
            // offset (× density) minus the composite recentre, both in voxels.
            let offset = node.transform.offset_blocks;
            let translation_voxels = [
                offset[0] * voxels_per_block as i32 - recentre_voxels[0],
                offset[1] * voxels_per_block as i32 - recentre_voxels[1],
                offset[2] * voxels_per_block as i32 - recentre_voxels[2],
            ];
            match &node.content {
                NodeContent::Tool { shape, material } => {
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        material_id_for(*material),
                        shape,
                    );
                }
                NodeContent::Part(Part::DebugClouds { seed }) => {
                    let producer = DebugCloudField {
                        // The cloud field sizes itself from the region (today's
                        // behaviour resolved it at the shape's grid dimensions).
                        dimensions: region_dimensions,
                        voxels_per_block,
                        seed: *seed,
                    };
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        // A Part brings its own per-voxel materials; today the
                        // cloud field emits material 0, so the stamp keeps that.
                        None,
                        &producer,
                    );
                }
                // step 4: recurse into owned sub-assemblies, composing transforms
                // down (`world = parent ∘ local`).
                NodeContent::Group(_children) => {}
                // step 4: resolve the referenced definition under this transform.
                NodeContent::Instance(_def_id) => {}
            }
        }

        output
    }
}

/// The whole-block extent of a leaf node's producer, or `None` for a non-leaf /
/// not-yet-implemented content kind.
fn leaf_size_blocks(content: &NodeContent, voxels_per_block: u32) -> Option<[u32; 3]> {
    let density = voxels_per_block.max(1);
    match content {
        NodeContent::Tool { shape, .. } => Some(shape.size_blocks),
        // The cloud field has no intrinsic size; today it adopts the shape's grid
        // dimensions, so a step-1 Part-only scene has no extent of its own. The
        // call sites that resolve a Part always pass the region explicitly, so
        // this path is unused by them; report whole blocks for completeness.
        NodeContent::Part(Part::DebugClouds { .. }) => {
            // A Part stamped at the app density occupies `dimensions / density`
            // blocks; with no stored body in step 1 it has no size. Returning
            // `None` keeps `full_extent_blocks` deferring to the next leaf.
            let _ = density;
            None
        }
        NodeContent::Group(_) | NodeContent::Instance(_) => None,
    }
}

/// Map a Tool's [`MaterialChoice`] to the `material_id` it stamps (ADR 0001 step 3
/// "Materials"). A Tool is single-material by nature: every voxel it emits takes
/// this one id, so distinct nodes render in distinct materials. Stone = 0,
/// Wood = 1, Plain = 2 (see [`MaterialChoice::material_id`]).
fn material_id_for(material: MaterialChoice) -> Option<u16> {
    Some(material.material_id())
}

/// Resolve `producer` into its own local grid (centred at the origin, as the
/// trait guarantees) and **stamp** it into `output`, translated by
/// `translation_voxels` (the node's placement minus the composite recentre, in
/// voxels).
///
/// When `translation_voxels` is zero and no material override applies, the stamp
/// is the identity: the producer's occupied set is moved into `output` unchanged
/// (the one-node, zero-offset path — guarantees a bit-for-bit match with the bare
/// producer). When `material_override` is `Some(id)`, every stamped voxel takes
/// that id (a Tool's single material); when `None`, each voxel keeps the material
/// the producer emitted (a Part's own per-voxel materials).
fn stamp_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i32; 3],
    material_override: Option<u16>,
    producer: &dyn VoxelProducer,
) {
    // The producer sizes its own grid (`SdfShape::resolve` overwrites
    // `dimensions` to its own `size_blocks × density`, centred at the origin), so
    // the local grid need only seed the dimensions; the cloud field, which has no
    // intrinsic size, fills the region it is handed.
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local);

    let zero_offset = translation_voxels == [0, 0, 0];

    if zero_offset && material_override.is_none() {
        // Fast path / exact identity: no translation and no material rewrite, so
        // the local occupied set IS the output.
        if output.occupied.is_empty() {
            output.occupied = local.occupied;
            return;
        }
        output.occupied.extend(local.occupied);
        return;
    }

    // General stamp: translate each voxel into the composite (the producer's
    // origin-centred position plus the node's recentred placement) and, for a
    // Tool, overwrite its material id.
    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        if !zero_offset {
            voxel.world_position[0] += translation_voxels[0] as f32;
            voxel.world_position[1] += translation_voxels[1] as f32;
            voxel.world_position[2] += translation_voxels[2] as f32;
        }
        if let Some(id) = material_override {
            voxel.material_id = id;
        }
        output.occupied.push(voxel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ShapeKind;

    /// The identical-behaviour guarantee (ADR 0001 step 1): a one-node Tool scene
    /// resolved over the node's full extent yields the SAME occupied count as
    /// calling `SdfShape::resolve` directly — and the same grid dimensions.
    #[test]
    fn tool_scene_matches_bare_producer() {
        let geometry = GeometryParams {
            shape: ShapeKind::Sphere,
            size_blocks: [6, 6, 6],
            voxels_per_block: 16,
            wall_blocks: 1,
        };

        // Bare producer (today's path).
        let shape = SdfShape::from_geometry(geometry);
        let mut bare = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut bare);

        // Through the scene.
        let scene = Scene::from_geometry(geometry, MaterialChoice::Stone);
        let region = scene.full_extent_blocks(geometry.voxels_per_block);
        let resolved = scene.resolve_region(region, geometry.voxels_per_block, 0);

        assert_eq!(
            resolved.dimensions, bare.dimensions,
            "scene grid dimensions must match the bare producer"
        );
        assert_eq!(
            resolved.occupied_count(),
            bare.occupied_count(),
            "scene occupied count must match the bare producer"
        );
    }

    /// The same guarantee for a Part (the debug cloud field): a one-node Part
    /// scene matches `DebugCloudField::resolve` at the same dimensions. Step 2
    /// builds the Part node directly (the `debug_clouds` selector is gone).
    #[test]
    fn part_scene_matches_bare_cloud_field() {
        let size_blocks = [4u32, 4, 4];
        let voxels_per_block = 16u32;
        let dimensions = [
            size_blocks[0] * voxels_per_block,
            size_blocks[1] * voxels_per_block,
            size_blocks[2] * voxels_per_block,
        ];
        let bare_field = DebugCloudField {
            dimensions,
            voxels_per_block,
            seed: 0,
        };
        let mut bare = VoxelGrid::new(dimensions);
        bare_field.resolve(&mut bare);

        let scene =
            Scene::single_node(Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })));
        let region = RegionBlocks::new(size_blocks);
        let resolved = scene.resolve_region(region, voxels_per_block, 0);

        assert_eq!(resolved.dimensions, bare.dimensions);
        assert_eq!(resolved.occupied_count(), bare.occupied_count());
    }

    /// ADR 0001 step 2: several leaf nodes composite into one region under union.
    /// A 2-node scene (a sphere Tool + a box Tool, both centred at origin) yields
    /// the SET-UNION of their occupied voxels: the union count is at least each
    /// node alone, and exactly equals the union of the two single-node sets.
    #[test]
    fn two_node_scene_resolves_to_union() {
        let voxels_per_block = 12u32;
        let region = RegionBlocks::new([6, 6, 6]);

        let sphere = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Sphere,
                    size_blocks: [6, 6, 6],
                    voxels_per_block,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        // A full-extent box: its corners poke outside the inscribed sphere, so the
        // union is strictly larger than the sphere alone (a real composite).
        let cube = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [6, 6, 6],
                    voxels_per_block,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Wood,
            },
        );

        // Each node resolved alone.
        let sphere_only = Scene::single_node(sphere.clone())
            .resolve_region(region, voxels_per_block, 0);
        let cube_only =
            Scene::single_node(cube.clone()).resolve_region(region, voxels_per_block, 0);

        // Both nodes composited.
        let scene = Scene {
            nodes: vec![sphere, cube],
            active: Some(0),
        };
        let union = scene.resolve_region(region, voxels_per_block, 0);

        // The expected set-union of the two single-node occupied sets, keyed by
        // integer voxel position (the producers emit voxel-centre world positions).
        use std::collections::HashSet;
        let key = |grid: &VoxelGrid| -> HashSet<[i64; 3]> {
            grid.occupied
                .iter()
                .map(|voxel| {
                    [
                        voxel.world_position[0].round() as i64,
                        voxel.world_position[1].round() as i64,
                        voxel.world_position[2].round() as i64,
                    ]
                })
                .collect()
        };
        let sphere_set = key(&sphere_only);
        let cube_set = key(&cube_only);
        let union_set = key(&union);
        let expected: HashSet<[i64; 3]> = sphere_set.union(&cube_set).copied().collect();

        // Union is at least as occupied as either node alone …
        assert!(union_set.len() >= sphere_set.len());
        assert!(union_set.len() >= cube_set.len());
        // … and equals the set-union exactly (the box pokes outside the sphere, so
        // the union is strictly larger than the sphere alone — a real composite).
        assert_eq!(union_set, expected);
        assert!(union_set.len() > sphere_set.len());
    }

    /// ADR 0001 step 3 (per-voxel material): a Tool with `MaterialChoice::Wood`
    /// stamps voxels whose `material_id` equals the Wood id (1) — every voxel it
    /// emits carries the Tool's single material, so distinct nodes are distinct.
    #[test]
    fn wood_tool_stamps_wood_material_id() {
        let voxels_per_block = 8u32;
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block,
            wall_blocks: 1,
        };
        let scene = Scene::single_node(Node::new(
            "Wood box",
            NodeContent::Tool { shape, material: MaterialChoice::Wood },
        ));
        let grid = scene.resolve_region(RegionBlocks::new([2, 2, 2]), voxels_per_block, 0);
        let wood_id = MaterialChoice::Wood.material_id();
        assert!(grid.occupied_count() > 0, "the box must emit voxels");
        assert!(
            grid.occupied.iter().all(|voxel| voxel.material_id == wood_id),
            "every voxel a Wood Tool stamps must carry the Wood material id"
        );
    }

    /// ADR 0001 step 3 (per-voxel material): a 2-Tool scene (Stone + Wood, placed
    /// disjointly) yields BOTH material ids present — proving the per-voxel id
    /// travels through compositing so the two nodes render in distinct materials.
    #[test]
    fn two_material_scene_has_both_material_ids() {
        let voxels_per_block = 8u32;
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut stone = Node::new("Stone", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        stone.transform.offset_blocks = [0, 0, 0];
        let mut wood = Node::new("Wood", NodeContent::Tool { shape: base, material: MaterialChoice::Wood });
        wood.transform.offset_blocks = [5, 0, 0];
        let scene = Scene {
            nodes: vec![stone, wood],
            active: Some(0),
        };
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        assert_ne!(stone_id, wood_id, "Stone and Wood must map to distinct ids");
        assert!(
            grid.occupied.iter().any(|voxel| voxel.material_id == stone_id),
            "the Stone node's voxels must carry the Stone id"
        );
        assert!(
            grid.occupied.iter().any(|voxel| voxel.material_id == wood_id),
            "the Wood node's voxels must carry the Wood id"
        );
    }

    /// A hidden node contributes nothing.
    #[test]
    fn hidden_node_stamps_nothing() {
        let mut node = Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [2, 2, 2],
                    voxels_per_block: 8,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        node.visible = false;
        let scene = Scene::single_node(node);
        let resolved = scene.resolve_region(RegionBlocks::new([2, 2, 2]), 8, 0);
        assert_eq!(resolved.occupied_count(), 0);
    }

    /// A box Tool sized to fill a single block (so the whole block of voxels is
    /// occupied), at the given block offset along X, in a wide region. Returns the
    /// set of occupied voxel positions keyed to integer coordinates.
    fn boxed_block_positions(offset_x: i32, voxels_per_block: u32) -> std::collections::HashSet<[i64; 3]> {
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform.offset_blocks = [offset_x, 0, 0];
        // A region wide enough to hold the offset box without clipping.
        let region = RegionBlocks::new([8, 1, 1]);
        let grid = Scene::single_node(node).resolve_region(region, voxels_per_block, 0);
        grid.occupied
            .iter()
            .map(|voxel| {
                [
                    voxel.world_position[0].round() as i64,
                    voxel.world_position[1].round() as i64,
                    voxel.world_position[2].round() as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 3 (a): a node with `offset_blocks = [N, 0, 0]` places its
    /// voxels shifted by exactly `N × voxels_per_block` in X versus offset 0.
    ///
    /// A two-node scene (a 1-block box at offset 0 and an identical box at offset
    /// N, far enough apart to be disjoint) shares ONE composite recentre, so the
    /// only difference between the two boxes' positions is the N-block placement.
    /// The occupied set splits into two equal clusters whose X-spans are exactly
    /// `N × voxels_per_block` apart; shifting one cluster by that amount reproduces
    /// the other.
    #[test]
    fn offset_node_shifts_voxels_by_blocks_times_density() {
        let voxels_per_block = 8u32;
        let n = 5i32; // 5 blocks apart: a 1-block box leaves a 4-block gap (disjoint).
        let region = RegionBlocks::new([8, 1, 1]);
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut at_zero = Node::new("A", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_zero.transform.offset_blocks = [0, 0, 0];
        let mut at_n = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_n.transform.offset_blocks = [n, 0, 0];

        let scene = Scene {
            nodes: vec![at_zero, at_n],
            active: Some(0),
        };
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        // Key each voxel by its EXACT world position (the producers emit voxel-
        // centre positions; the placement is an exact integer-voxel translation, so
        // float comparison is safe and exact — no rounding). The boxes are disjoint
        // in X (5 blocks apart, 1 block wide), so the occupied set splits cleanly at
        // the gap between box A's X-run and box B's X-run.
        let shift = (n * voxels_per_block as i32) as f32; // N blocks → N×density voxels.
        let key = |position: [f32; 3]| -> [i64; 3] {
            // Bit-exact key: positions are k+0.5 half-integers, so ×2 is an exact
            // integer and avoids any float-equality fragility in the HashSet.
            [
                (position[0] * 2.0) as i64,
                (position[1] * 2.0) as i64,
                (position[2] * 2.0) as i64,
            ]
        };

        // The composite centre lies between the two boxes; split there.
        let mut xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let split_x = (xs.first().unwrap() + xs.last().unwrap()) / 2.0;

        let cluster_low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] < split_x)
            .map(|v| key(v.world_position))
            .collect();
        let cluster_high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] >= split_x)
            .map(|v| key(v.world_position))
            .collect();

        assert!(!cluster_low.is_empty() && !cluster_high.is_empty(), "both boxes present");
        assert_eq!(cluster_low.len(), cluster_high.len(), "both boxes fill one block");
        // Shifting the low box by exactly N×density in X reproduces the high box.
        let shifted: std::collections::HashSet<[i64; 3]> = cluster_low
            .iter()
            .map(|c| [c[0] + (shift * 2.0) as i64, c[1], c[2]])
            .collect();
        assert_eq!(shifted, cluster_high, "offset N blocks shifts voxels by exactly N×density");
    }

    /// ADR 0001 step 3 (b): two nodes at non-overlapping offsets give an occupied
    /// count equal to the SUM of each alone (a disjoint union — the placement
    /// genuinely separates them in space, no longer overlapping at the origin).
    #[test]
    fn disjoint_offsets_give_summed_occupancy() {
        let voxels_per_block = 8u32;
        // Two 1-block boxes 5 blocks apart in X — far enough that their voxel sets
        // never touch (each is 1 block = 8 voxels wide, gap is 4 empty blocks).
        let a_alone = boxed_block_positions(0, voxels_per_block).len();
        let b_alone = boxed_block_positions(5, voxels_per_block).len();
        assert!(a_alone > 0 && b_alone > 0);

        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut a = Node::new("A", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        a.transform.offset_blocks = [0, 0, 0];
        let mut b = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        b.transform.offset_blocks = [5, 0, 0];

        let scene = Scene {
            nodes: vec![a, b],
            active: Some(0),
        };
        // Region spans the full composite (offset 0..5, each 1 block) → 6 blocks X.
        let region = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(region.size_blocks, [6, 1, 1], "composite extent encompasses both offsets");
        let grid = scene.resolve_region(region, voxels_per_block, 0);
        assert_eq!(
            grid.occupied_count(),
            a_alone + b_alone,
            "disjoint placement → occupied count is the sum (no overlap)"
        );
    }

    /// ADR 0001 step 3 (c): `full_extent_blocks` grows to encompass an offset node.
    /// A single 2-block box pushed +4 blocks in X spans blocks `[3, 5]` in X (centre
    /// 4, ±1), so the composite X extent is 6 blocks (`0..6` once recentred), while
    /// Y/Z stay at the box's 2 blocks. (A zero-offset single node would be just the
    /// box's own 2×2×2.)
    #[test]
    fn full_extent_encompasses_offset_node() {
        let voxels_per_block = 4u32;
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new("Box", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        node.transform.offset_blocks = [4, 0, 0];
        let scene = Scene::single_node(node);

        // The box centred at block 4 with half-size 1 spans X blocks [3, 5] → its
        // own size (2) is unchanged but its placement means the bounding box from
        // the origin is wider. `full_extent_blocks` returns the box SIZE of the
        // composite: for a single node that is just the node's own size in every
        // axis (the offset moves it but doesn't enlarge a single box). To prove the
        // extent ACCOUNTS for the offset, compare against a two-node scene where the
        // offset opens a real gap.
        let single = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(single.size_blocks, [2, 2, 2], "a lone offset box keeps its own size");

        // Add a second box at the origin: now the composite must span from the
        // origin box (blocks [-1, 1]) to the offset box (blocks [3, 5]) → X width 6.
        let mut origin_box =
            Node::new("Origin", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        origin_box.transform.offset_blocks = [0, 0, 0];
        let mut offset_box =
            Node::new("Offset", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        offset_box.transform.offset_blocks = [4, 0, 0];
        let two = Scene {
            nodes: vec![origin_box, offset_box],
            active: Some(0),
        };
        let extent = two.full_extent_blocks(voxels_per_block);
        assert_eq!(
            extent.size_blocks,
            [6, 2, 2],
            "the offset node widens the composite extent in X from 2 to 6 blocks"
        );
    }
}
