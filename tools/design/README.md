# Design-sheet tooling

The sketch icon set (ADR 0035) is authored twice: as SVG on the sheets in
`docs/design/sketch-marks/`, where each mark's geometry is argued for and shown beside the
variants it beat, and as `Mark` data in `crates/ui/src/icons/`, where the prose lives. Neither
is compiled from the other — a glyph file is a hand transposition of the sheet's 36-unit canvas
onto the 18-unit glyph grid.

That transposition is dozens of retyped coordinates per shelf, and a slipped digit produces an
icon that is wrong in a way nobody catches by looking: a tangency that is nearly a tangency, a
node a third of a unit off its own vertex. These two scripts are what make that a test failure.

| | |
|---|---|
| `check-marks.mjs` | Proves the SHEET is sound: tangencies, arc sweeps, inradii, canvas containment, that a rejected variant cannot resolve, that the accent and the construction ink are spent where the rules say. Loads the sheet's `<script>` in a sandbox — no rendering, no eyeballing. |
| `reference.mjs` | Exports the sheet's RESOLVED geometry to `crates/ui/src/icons/design_reference.rs`, which `glyphs_match_the_design_sheet` diffs every sketch glyph against. |

```sh
node tools/design/check-marks.mjs     # the sheet is sound
node tools/design/reference.mjs       # regenerate the reference
cargo test -p ui --lib icons::        # the glyphs match it
```

Run all three after touching a sheet or a glyph. `design_reference.rs` is generated — a diff in
it during review means the drawing moved, which is exactly the signal it exists to give.

Only marks listed in `reference.mjs`'s `IDS` table are exported; a sketch mark that is drawn but
not yet authored in Rust simply is not there yet. Add its row when the glyph file lands.

The sheets are also published to the `VoxelWorker — Viewport Chrome` Claude Design project. The
copies here are the source; publish from `docs/design/sketch-marks/`, never the other way round.
