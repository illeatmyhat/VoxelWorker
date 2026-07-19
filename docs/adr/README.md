# Architecture Decision Records

This directory is **append-only decision history**: each record captures a decision at
the moment it was made — the context, the alternatives weighed, the evidence, and the
ruling. Records are never rewritten to match later reality; when reality moves, the
record's **Status** line is amended (Superseded / Retired / Amended-by) and a new
record carries the new decision.

The **current shape of the system** is not described here. It lives in
[`docs/architecture/`](../architecture/README.md), which is edited freely and kept
timeless. The division of labour:

| Place | Role | Editing rule |
| --- | --- | --- |
| `CONTEXT.md` (repo root) | Terms and their meanings | Prune freely; terms only |
| `docs/adr/` | Decisions and their reasoning | Append-only; amend Status lines only |
| `docs/architecture/` | The living shape of the system | Edit in place; no history, no roadmap |
| `docs/design/` | Dated analysis inputs (sweeps, maps) | Snapshots; supersede by newer files |

When writing a new ADR, describe the **delta** against `docs/architecture/` rather
than restating it, and update the architecture set in the same change that ships the
decision.

Read an ADR for *why* and *what was rejected* — its Status line tells you whether the
*what* still stands.

## Retired root documents

The early ADRs quote four root-level design docs that **no longer exist**:
`ARCHITECTURE.md`, `DATA.md`, `REPRESENTATION.md` and `HANDOFF.md` (deleted 2026-07-19,
along with the `PROGRESS.md` milestone log). They described the project as a
single-shape parametric tool ported from a three.js prototype, and had drifted into
being actively misleading — they still claimed the renderer does no raymarching, that
`isolevel` was a UI slider, and that the 450k-instance / 6M-voxel caps were live. All
three are false.

Those quotations are **left in place**: these records are append-only, and a quote with
attribution is still readable as provenance. Nothing was lost — every substantive claim
had already been absorbed, usually in more depth:

| Retired doc | Where its content lives now |
| --- | --- |
| `REPRESENTATION.md` — "the voxel grid is the one consumed truth" | `0006-authoring-truth-and-gpu-boundary.md` (quoted verbatim); the sparse-override layer in `0003` §3g |
| `ARCHITECTURE.md` §3 — the two shader-bug regression guards | `0002-engine-streaming-meshing.md` (per-voxel texture slice; position-based grid overlay), and the README's "Regression guards" |
| `ARCHITECTURE.md` §4/§5/§8 — camera rig, gizmo, palette | `0015` (camera crate), `0018` + `docs/design/viewport-chrome-signal.md`, `docs/design/colour-vocabulary.md` |
| `ARCHITECTURE.md` §7 — the instance/voxel caps | `0002` retires them explicitly; `0009`/`0010` dissolve the need |
| `DATA.md` — units model, VS install paths, chiselable block list | `docs/architecture/01-document.md` (units); the `assets` crate is now the source of truth for paths and the block list |
| `HANDOFF.md` — tech-choice rationale, build order | `docs/DEV_NOTES.md` (pinned versions); the build order is complete and historical |
