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

    /// The whole-block extent of the scene: the per-axis MAX over every leaf
    /// node's block extent. Step 2 composites several nodes into one region; since
    /// transforms are step 3 every node is centred at the origin, so the region
    /// that contains them all is the axis-wise maximum of their sizes. A
    /// Part-only node (the cloud field, which has no intrinsic size) contributes
    /// nothing and adopts whatever extent the Tools establish.
    pub fn full_extent_blocks(&self, voxels_per_block: u32) -> RegionBlocks {
        let mut extent = [0u32; 3];
        for node in &self.nodes {
            if let Some(size_blocks) = leaf_size_blocks(&node.content, voxels_per_block) {
                for axis in 0..3 {
                    extent[axis] = extent[axis].max(size_blocks[axis]);
                }
            }
        }
        RegionBlocks::new(extent)
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

        for node in &self.nodes {
            if !node.visible {
                continue;
            }
            match &node.content {
                NodeContent::Tool { shape, material } => {
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        node.transform.offset_blocks,
                        voxels_per_block,
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
                        node.transform.offset_blocks,
                        voxels_per_block,
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

/// Map a Tool's [`MaterialChoice`] to the `material_id` it stamps. Step 1 keeps
/// the existing single procedural material set; today every producer emits
/// `material_id == 0`, so to preserve identical behaviour the Tool also stamps
/// `0`. (The per-voxel material wiring — distinct ids per choice — is ADR 0001
/// step 3.) Returning `Some(0)` documents the seam while matching today exactly.
fn material_id_for(_material: MaterialChoice) -> Option<u16> {
    // step 3: return Some(material.material_id()) so the Tool stamps its own id.
    Some(0)
}

/// Resolve `producer` into its own local grid (centred at the origin, as the
/// trait guarantees) and **stamp** it into `output` under `offset_blocks`.
///
/// In step 1 the offset is always `[0, 0, 0]` and the producer's grid equals the
/// region, so the stamp is the identity: the producer's occupied set is moved
/// into `output` unchanged. When `material_override` is `Some(id)`, every stamped
/// voxel takes that id (a Tool's single material); when `None`, each voxel keeps
/// the material the producer emitted (a Part's own per-voxel materials).
fn stamp_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    offset_blocks: [i32; 3],
    voxels_per_block: u32,
    material_override: Option<u16>,
    producer: &dyn VoxelProducer,
) {
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local);

    let offset_voxels = [
        offset_blocks[0] * voxels_per_block as i32,
        offset_blocks[1] * voxels_per_block as i32,
        offset_blocks[2] * voxels_per_block as i32,
    ];
    let zero_offset = offset_voxels == [0, 0, 0];

    if zero_offset && material_override.is_none() {
        // Fast path / exact identity: no translation and no material rewrite, so
        // the local occupied set IS the output. This is the step-1 cloud case and
        // guarantees a bit-for-bit match with the bare producer.
        if output.occupied.is_empty() {
            output.occupied = local.occupied;
            return;
        }
        output.occupied.extend(local.occupied);
        return;
    }

    // General stamp: translate each voxel by the block offset (in voxel units)
    // and, for a Tool, overwrite its material id. Step 1 only ever hits the
    // zero-offset branch, but the translation is written so step 3 needs no
    // change here.
    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        if !zero_offset {
            voxel.world_position[0] += offset_voxels[0] as f32;
            voxel.world_position[1] += offset_voxels[1] as f32;
            voxel.world_position[2] += offset_voxels[2] as f32;
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
}
