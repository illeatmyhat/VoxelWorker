# ARCHITECTURE — Chisel Bench

Reference implementation: `chisel-bench-reference.html` (three.js). All formulas below are
transcribed from its `<script>`; when in doubt, that file wins.

## 1. Pipeline overview

```
   user params (egui)
        │  X,Y,Z (blocks), density (vx/block), shape, wall, isolevel, material
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ CPU: build voxel set                                          │
   │   Nx,Ny,Nz = X*d, Y*d, Z*d                                    │
   │   for j in 0..Ny, k in 0..Nz, i in 0..Nx:                     │
   │     p = (i+0.5-Nx/2, j+0.5-Ny/2, k+0.5-Nz/2)                  │
   │     if sdf(p) <= isolevel:                                     │
   │        push instance{ pos:p, iLocal:(i%d, j%d, k%d) }         │
   └─────────────────────────────────────────────────────────────┘
        │ instance buffer (pos + iLocal)         │ same loop, mid-Y only
        ▼                                          ▼
   ┌───────────────────────────┐         ┌──────────────────────────┐
   │ GPU: instanced unit cubes │         │ 2D slice map (CPU pixels) │
   │  vtx: per-voxel tex slice │         │  egui texture, block lines│
   │  frag: texture + grid     │         └──────────────────────────┘
   │  overlay (from world pos) │
   └───────────────────────────┘
        ▼
   ┌───────────────────────────────────────────────┐
   │ Composite: main scene (persp|ortho camera)     │
   │  + origin gizmo (depth-test off)               │
   │  + view-cube corner viewport                   │
   │  + egui panel & palette dock                   │
   └───────────────────────────────────────────────┘
```

## 2. SDFs (CPU inside/outside test only)

`hyp(...)` = hypot. Work in **voxel space**; voxel-space dims = block dims × density:
`Rv` etc. aren't used directly — instead the shape is inscribed in the box with semi-axes
`AX=Nx/2, AY=Ny/2, AZ=Nz/2`. `WV = wall*density`.

```
sdBox(p,bx,by,bz):
    q = abs(p) - (bx,by,bz)
    return length(max(q,0)) + min(max(q.x,max(q.y,q.z)),0)

sdEllipsoid(p,ax,ay,az):              # inscribed sphere/ellipsoid (IQ approximation)
    k0 = length(p/(ax,ay,az))
    if k0==0: return -min(ax,ay,az)
    k1 = length(p/(ax*ax,ay*ay,az*az))
    return k0*(k0-1)/k1

sdCylE(p,ax,az,ay):                   # elliptical cylinder, axis = Y, half-height ay
    dr = (length((p.x/ax, p.z/az)) - 1) * min(ax,az)
    dy = abs(p.y) - ay
    return min(max(dr,dy),0) + length((max(dr,0), max(dy,0)))

sdf(p) by shape:
    Cylinder : sdCylE(p, AX, AZ, AY)
    Tube     : max( sdCylE(p,AX,AZ,AY), -sdCylE(p, max(AX-WV,.01), max(AZ-WV,.01), AY+1) )
    Sphere   : sdEllipsoid(p, AX, AY, AZ)
    Torus    : t = AY; R = max(min(AX,AZ) - t, 0); return hyp(hyp(p.x,p.z)-R, p.y) - t
    Box      : sdBox(p, AX, AY, AZ)
```

Voxel exists when `sdf(p) <= isolevel`. The **isolevel** slider (≈ -2..2) nudges the boundary;
it's the key control for tuning rim run-lengths on circles. The ellipsoid/cylinder forms are
approximate distance fields — fine for thresholding; exact only for the analytic primitives.

## 3. The two shader bugs (DO NOT regress)

### Bug 1 — texture repeated per cube
Wrong: map the texture 0..1 onto every cube → every voxel shows the whole block texture, so a
chiseled shape looks like a stack of whole blocks. Right: the texture belongs to the **block**,
so each voxel face shows only its `1/density` slice. Vertex shader, per face, picks the two
in-plane axes from the face normal and offsets the UV by the block-local coord:

```
// vertex, after standard uv:
an = abs(normal)
if   an.x > 0.5: voff = (iLocal.z, iLocal.y)
elif an.y > 0.5: voff = (iLocal.x, iLocal.z)
else           : voff = (iLocal.x, iLocal.y)
vUv = (uv + voff) / density          // 1/density slice within the block
```

### Bug 2 — grid overlay off-by-one on vertical faces
Wrong: detect block boundaries from face UVs (`is this voxel first/last in its block`). Cube
faces flip UV direction per face (BoxGeometry orients UVs so textures stay upright), so "UV≈0"
points at the high-coordinate edge on some faces → bold line lands one voxel off on the sides.

Right: compute the grid from the fragment's **absolute voxel position**, orientation-independent.
Pass `vVoxAbs = worldPosOfFragment + (Nx/2,Ny/2,Nz/2)` (so voxel boundaries are at integers).
In the fragment:

```
inpl = step(abs(normal), 0.5)                 // 1 on the two in-plane axes, 0 on the normal axis
di   = abs(A - floor(A + 0.5))                 // distance to nearest voxel boundary (per axis)
db   = abs(A/density - floor(A/density + 0.5)) * density   // distance to nearest block boundary
voxelLine = max over in-plane axes of (1 - smoothstep(voxHW, voxHW+aa, di))
blockLine = max over in-plane axes of (1 - smoothstep(blkHW, blkHW+aa, db))   // bolder: blkHW>voxHW
color = mix(texColor, lineColor, max(voxelLine*voxA, blockLine*blkA))         // block line wins/darker
```

In wgpu you'd pass the fragment's world position as a varying from the vertex stage (compute it
from the instance transform), plus `density` and the half-extents in a uniform. WGSL has
`step`, `floor`, `abs`, `smoothstep`, `mix` — direct ports.

## 4. Camera rig

Orbit around `target` (origin) with spherical `theta` (azimuth), `phi` (polar from +Y), `dist`:

```
dir = (sin(phi)cos(theta), cos(phi), sin(phi)sin(theta))
camera.pos = target + dir*dist ; up = +Y ; look at target
```

- **Perspective**: fov 45°.
- **Orthographic**: half-height `vh = dist*0.42` (≈ matches the 45° fov at the target so toggling
  keeps framing); `top=vh, bottom=-vh, left=-vh*aspect, right=vh*aspect`. Recompute when `dist`
  changes (wheel zoom) so ortho zoom works.
- **Wheel**: `dist *= 1 ± 0.08`.
- **Auto-frame** on size/density change: `dist = max(Nx,Ny,Nz) * 1.9`.

### View cube
Small cube in its own scene drawn in a corner viewport (top-left, ~128px). Its camera copies the
main camera's *direction* (`cubeCam.pos = dir*4; look at 0`) so the cube mirrors the current view.
Click a face → raycast (NDC computed from the pointer's position within the cube's screen rect) →
`materialIndex` → snap target angles, tween `theta/phi` (easeInOutQuad, ~380ms; instant if
`prefers-reduced-motion`).

Face → (theta, phi) snap table (materialIndex order = +X,-X,+Y,-Y,+Z,-Z):
```
RIGHT +X : (0,        π/2)
LEFT  -X : (π,        π/2)
TOP   +Y : (-π/2,     ~0)
BOTTOM-Y : (-π/2,     ~π)
FRONT +Z : (π/2,      π/2)
BACK  -Z : (-π/2,     π/2)
```
Pick the nearest equivalent `theta` (add/sub 2π) before tweening to avoid long spins.

In wgpu you don't have separate "scenes"; render the cube as a second draw with its own
view/proj matrices into a scissored corner viewport after clearing depth there. Or, simpler in
egui, draw the cube as a custom egui widget. Either is fine.

## 5. Origin gizmo
Group at origin, sized to the box (`L = max(Nx,Ny,Nz)*0.62`). Three arrows: X red `0xd9603f`,
Y green `0x6fcf5f`, Z blue `0x5a8cff`. Three small square line-loops in the XY/YZ/ZX planes
(side `0.28*L`) marking the right angles. **Render with depth-test disabled** (and high render
order) so it shows through a solid model. Toggle-able; off by default.

## 6. Render layering (per frame)
```
clear (color+depth)
draw main instanced voxels (active camera)
draw origin gizmo (depth-test off)        [if enabled]
draw view-cube into corner viewport (clear depth in scissor first)   [if enabled]
draw egui (panel + palette dock)
```

## 7. Frame/perf notes
- Rebuild the instance buffer only when a shape/size/density/isolevel param changes (dirty flag).
- Cap voxel count (prototype caps ~450k instances; pauses 3D >~6M voxels, keeps the 2D slice
  live). A 5×1×5 @16 is tiny; a sphere at high density/size is where you hit the cap.
- The 2D slice recompute is cheap (one mid-Y layer); always update it.

## 8. Visual identity (carry over if you want consistency)
Dark warm workshop background; parchment text `#e9e1d1`; patina/verdigris accent `#5fb8a4`
(echoes VS copper tools oxidizing); copper secondary `#c08457`; mono for numeric readouts.
Grid line colors: voxel `#17120b`, block `#080605` (darker, bolder). Not load-bearing — match
or rethink as you like.
