# DATA — units, VS paths, chiselable blocks

## Units model (the one that keeps biting; internalize it)

| concept   | unit            | example         | notes |
|-----------|-----------------|-----------------|-------|
| X, Y, Z   | whole **blocks**| 5 × 1 × 5       | the bounding box the shape fills |
| density   | **voxels/block**| 16 (VS default) | chisel fineness ONLY |
| Nx,Ny,Nz  | voxels          | 80 × 16 × 80    | = X*d, Y*d, Z*d |
| wall      | whole blocks    | 1               | Tube only |
| isolevel  | distance        | 0.0             | boundary nudge for rim tuning |

The density bug, stated once more so it's unmissable: **density changes only how finely each
block is subdivided.** A 5-block disc is 5 blocks wide at density 16 and still 5 blocks wide at
density 32 — just smoother. The texture is one tile per block at every density. If raising
density makes the object bigger or the texture finer, the dimensions are wrongly stored in
voxels. Store sizes in blocks; multiply by density only to get the sampling grid.

## VS install locations (where the textures live)

Two separate folders — you want the **program/assets** one, not the data one.

- **Windows:** `%APPDATA%\Vintagestory` → i.e. `C:\Users\<user>\AppData\Roaming\Vintagestory`.
  Assets at `…\Vintagestory\assets`. (Custom install: `<installdir>\assets`.)
- **Linux:** `~/.config/Vintagestory` or `~/.local/share/vintagestory`; Flatpak under
  `~/.var/app/at.vintagestory.VintageStory/...` (assets at
  `…/current/active/files/extra/vintagestory/assets`); Arch AUR `/opt/vintagestory/`.
- **macOS:** inside the app bundle (Show Package Contents).

Do **not** confuse `Vintagestory\assets` (textures, JSON — what we want) with
`VintagestoryData` (saves/mods/logs — not what we want). Native app reads either freely; the
native folder picker (`rfd`) has none of the browser AppData blocklist problem that killed the
web version.

Asset tree of interest:
```
assets/
  survival/
    textures/block/**.png      ← block textures (what we scan)
    blocktypes/**.json          ← block definitions (per-face textures, drawtype, material)
  game/  creative/              ← also exist; survival has the bulk
```

## Scanning for chiselable blocks

Chiselability can't be read from a texture; keep a curated allow-list and match by lowercase
substring on the texture's path. Auto-discover variants by stripping trailing digits from the
filename (`granite1.png, granite2.png → group "granite", 2 variants`).

```
ALLOW (substring match): all rock types + these:
  granite, andesite, basalt, chalk, chert, claystone, conglomerate, diorite, dolomite,
  dolostone, sandstone, limestone, shale, slate, peridotite, phyllite, suevite, kimberlite,
  bauxite, halite, graphite, obsidian, marble, quartzite, gneiss, breccia, greenschist,
  plank, ashlar, polished, stonebrick, claybrick, cobblestone, drystone
  + always accept anything under a ".../rock/..." path segment

EXCLUDE (substring): "/ore", "gravel", "overlay", "/soil", "crack", "_n.png", "-n.png", "glow"
  (note: exclude by these tokens — they are NOT substrings of any allow name, so "sandstone"
   survives even though "sand"-like tokens are risky; do not exclude bare "sand")

GROUP KEY = path without ".png", trailing digits stripped.
LABEL     = last path segment of the key, '-'/'_' → space, Title Case.
```

This list is a best-guess at the vanilla set and folder layout. **First real scan is the
test** — if it finds too few or pulls junk, adjust ALLOW/EXCLUDE. Keep them as plain arrays/consts.

## Per-face textures (the native-only upgrade)

The web version puts one texture on all six faces. VS blocks can texture faces differently
(top vs sides) and the block's `drawtype`/`textures` live in `assets/survival/blocktypes/**.json`.
To do it properly:
1. For a chosen block, find its blocktype JSON (match by `code` / texture references).
2. Read `textures` (and `texturesbytype`) → map face → texture path. Single `base` = all faces;
   named keys (e.g. top/side) → per-face.
3. Resolve `game:` domain prefix → `assets/survival/textures/<path>.png`.
4. Upload up to 6 textures (or a small atlas) and index per face in the shader.
This is optional polish for v1 but is the main reason the port exists, so design the texture
binding to allow >1 texture per material from the start.

## `.vox` export (later)

Exporting the voxel set as MagicaVoxel `.vox` lets the result drop into the
**Automatic Chiselling REBORN** VS mod, which ingests `.vox`/`.obj` and auto-carves the shape
in-world. Nice end-to-end payoff once the core works. `.vox` is a simple chunked binary format;
many Rust crates exist, or it's ~100 lines to write directly.

## Procedural fallback materials (for milestone 4 before real textures)

- **Stone:** 32×32, base grey ~rgb(132,126,118) with ±20 per-pixel noise, a few darker speckles.
- **Wood:** 32×32, brownish base with horizontal sine grain + noise.
- **Plain:** flat warm grey `#b6a079`.
All nearest-filtered, clamp-to-edge (the per-voxel slice stays within [0,1] so no wrap needed).
