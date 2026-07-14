# Graphics-math crates extraction map (2026-07-13)

**Provenance:** owner direction — graphics-related crates "in the same vein as substrate,
focusing on implementations of well-known concepts and structures in an accessible way." A
very-thorough survey of the display-side code produced this inventory. Dated analysis input;
the decision record is `docs/adr/0015`; the crate test remains ADR 0014's (a dependency law
worth compile-enforcing) plus the graphics-specific law: **crates hold wgpu-free graphics
mathematics; wgpu plumbing stays app**. The binding discipline: the WGSL shaders are mirrors
of these crates' CPU implementations — **the crate is the readable specification of the
shader, and gpu_parity is the law that holds them together.**

**Status (2026-07-14): EXECUTED IN FULL — G0–G2 all landed** (`b5cb208` G0 substrate additions →
`b50c03f` G1 crates/camera + the AABB co-location ruling → `1dcce29` G2 crates/raycast). Every
parity/golden suite passed unmodified throughout; the raymarch WGSL now carries the GPU-mirror
header naming `crates/raycast` as its readable specification. Baselines after G2: app lib 423/6,
substrate 87, camera 54, raycast 9. Next per ADR 0015's consequences: the ADR 0013
material-atlas epic builds its per-voxel sampling inside `raycast`.

**Survey verdicts that shaped this map:** `camera.rs` (1922 lines) already imports ONLY glam —
zero winit/egui/wgpu/domain types; it is a pure-math island with 48 movable tests. The CPU
reference march's DDA kernel is provably separable (`cpu_march_exact_occupancy` already takes
an injected `Fn([i64;3])->bool` occupancy closure). A `shading` crate is NOT warranted (CPU
side ≈ 4 small functions; all substantive shading is WGSL-only) — rejected, see restraint list.

## Crate 1 — `crates/camera` (viewing + projective geometry)

Moves essentially all of `src/camera.rs` + `src/frustum.rs` (both already glam-only):

| Component | Today | Literature identity |
|---|---|---|
| `OrbitCamera` (spherical parameterization, Z-up) + orbit/pan/zoom control math | camera.rs ~673–928 | arcball-family orbit control (Shoemake 1992 lineage); pole-clamped drag, sin(φ) azimuth damping, cursor-locked view-plane pan |
| `view_projection` (perspective + near-behind-eye-tolerant orthographic, near/far from bounding-sphere enclosure) | camera.rs ~950–1005 | look-at + perspective/orthographic projection (Real-Time Rendering); the ortho near<0 tolerance explained in the definition |
| Pole-continuous up vector + roll | camera.rs ~808–866 | singular-frame handling via smoothstep blend; Gram–Schmidt screen-up |
| `ViewCube` 26-orientation model, snap angles, great-circle rotation table, chrome hit zones, `SnapTween`/easing | camera.rs ~44–658 | the Autodesk ViewCube model; easeInOutQuad; nearest-angle-mod-2π |
| `HomeView` / focus-and-fit framing | camera.rs ~456–739 | bounding-box framing fit; consumes a RECENTRED render-frame AABB (the voxel→render adapter stays domain, ADR 0008 lens) |
| `Frustum` (plane extraction + AABB culling) + f32 `Aabb` | frustum.rs (160 lines) | **Gribb–Hartmann 2001** plane extraction; **positive-vertex culling test** (Ericson 2005). The f32 render-space twin of substrate's integer `Aabb` — the two docs cross-reference |

Also gains: the generic **screen-point → world `Ray` unprojection** (inverse-VP; today ad-hoc in
main.rs and brick_raymarch.rs), returning substrate's `Ray`.

Tests: 48 camera tests + 4 frustum tests move (all pure math). Domain seams: thin re-exports;
winit/egui input handling and `core_geom.rs` (domain palette vocabulary) stay app.

## Crate 2 — `crates/raycast` (ray–volume traversal)

The pure kernel of the CPU reference march (`brick_raymarch.rs`), generic over injected
occupancy closures — the seam `cpu_march_exact_occupancy` already proves:

| Component | Today | Literature identity |
|---|---|---|
| Ray–AABB slab entry test | brick_raymarch.rs ~2014 | **slab method** (Kay–Kajiya 1986; Ericson 2005; Williams et al. 2005) |
| `VoxelDda` — the stepping loop, used at block AND voxel scale | ~2026–2044, ~2156–2240 | **Amanatides & Woo 1987** (t_max/t_delta seed, min-axis advance, x→y→z ties) |
| Hierarchical empty-space skip (coarsest-empty-level jump to cell exit) | ~2051–2106 | **hierarchical DDA** (Crassin et al. 2009 GigaVoxels; Museth 2013); level occupancy via an injected closure (domain wires substrate's `sorted_cell_keys_contain` + its empty-level policy) |
| Entry-face normal + band-clip (Z-slab intersect) | ~2156–2220 | axis-aligned entry-face normal = −sign(dir); slab intersection |
| View-cube element picking (slab test vs [-0.7,0.7]³ + face mapping) | main.rs ~1016–1090 | same slab primitive; currently trapped in the winit `State` — small detangle |

Domain adapters STAY in `brick_raymarch.rs`: `BrickMarchFrame` (the ADR 0008 carried-frame
value block — plain PODs, feeds the kernel), record binary search + atlas byte fetch (they
already lean on substrate's `lattice_key`/`cube_packing`; the extraction DEDUPES
`cpu_pack_key_split` against substrate rather than moving it), the `>127` occupancy threshold,
and the WGSL shader (documented as this crate's GPU mirror). gpu_parity + golden pin the whole
move; `cpu_march_exact_occupancy` moves as the kernel's own oracle.

## Substrate additions (not graphics crates — pure CS/math)

- **`Ray { origin, direction }`** — no Rust ray type exists today (three ad-hoc tuple sites).
  Lives in substrate beside the boxes so `camera` (produces) and `raycast` (consumes) stay
  independent of each other.
- **AABB co-location (owner ruling 2026-07-14, landed with G1):** the two box types unify by
  CO-LOCATION with distinct names, NOT generics/traits — `LatticeAabb` (integer, half-open,
  touching boxes are DISJOINT: edit-broadphase semantics; the former `substrate::Aabb`) and
  `RealAabb` (f32, closed, ±inf-sentinel empty, touching boxes INTERSECT: culling must be
  conservative; the former `frustum::Aabb`), documented side-by-side. A policy-parameterized
  generic was rejected as ceremony that hurts the accessibility goal; a `BoundingVolume` trait
  waits for a second BVH consumer (e.g. an f32 render-culling BVH). `Frustum::intersects_aabb`
  and `Ray`'s slab test both consume `&RealAabb`.
- **`ShelfBinPack`** — texture_atlas.rs's gutter-padded shelf rect packer + half-texel-inset
  UV layout + replicated-edge blit (~132–283): textbook shelf/next-fit rectangle packing, the
  2D sibling of `CubeTilePacking`. Its 5 tests move. `from_procedural_materials` stays as the
  domain adapter.
- **sRGB↔linear codec** — renderer.rs ~124–142, the standard piecewise EOTF (IEC 61966-2-1),
  reused ~10 sites. Textbook transfer function, no graphics-concept crate needed.

## Deliberately NOT crates / NOT moved (restraint list)

- **`shading` crate — rejected** (survey): CPU-side shading ≈ 4 small functions (procedural
  Stone/Wood/Plain texel gen, average-color reduction, one ghost-tint constant); all
  substantive shading (lighting, grid-overlay AA via screen-space derivatives, per-voxel
  tiling) is WGSL-only. No dependency law to enforce. Revisit only if CPU-side shading math
  actually accumulates.
- **wgpu plumbing** (renderer.rs pipelines/bind groups/passes, gpu.rs, upload paths), the
  WGSL files (they are the crates' documented GPU mirrors), orchestrator/routing, shot.rs.
- **`core_geom.rs`** — palette vocabulary, domain.
- The chrome-glyph supersampled rasterizer (renderer.rs ~1180) — view-cube UI chrome, stays
  with the UI.

## Dependency picture

```
substrate ← camera  ─┐
substrate ← raycast ─┴─ voxel_worker (wgpu, WGSL mirrors, orchestration)
```
Graphics crates: glam + substrate only; never wgpu; never each other (Ray in substrate is the
shared vocabulary). Benches: none initially — the CPU march is a parity oracle, not a hot
path (per-frame traversal runs on the GPU); ADR 0014's hot-components-only policy applies.

## Slice order

G0 substrate additions (Ray, ShelfBinPack, sRGB codec) → G1 `crates/camera` (camera.rs +
frustum.rs wholesale + unprojection) → G2 `crates/raycast` (DDA kernel generic over closures,
picker slab math, lattice-key dedupe). Then the ADR 0013 material-atlas epic builds its
per-voxel sampling inside `raycast` (the sequencing motive). Literature anchors are inline
above; each module doc cites them per the ADR 0014 definition-of-done.
