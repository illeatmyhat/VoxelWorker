# Design-sheet tooling

The sketch icon set is authored twice: as SVG on the design sheets, where each mark's geometry
is argued for and shown beside the variants it beat, and as `Mark` data in
`crates/ui/src/icons/`, where the prose lives. Neither is compiled from the other — a glyph file
is a hand transposition of the sheet's 36-unit canvas onto the 18-unit glyph grid.

That transposition is dozens of retyped coordinates per shelf, and a slipped digit produces an
icon that is wrong in a way nobody catches by looking: a tangency that is nearly a tangency, a
node a third of a unit off its own vertex. These two scripts are what make that a test failure.

**The sheets live in the Claude Design project, not in this repo.** Both scripts take the path
to a downloaded copy, as an argument or through `SKETCH_MARKS_DIR`.

| | |
|---|---|
| `check-marks.mjs` | Proves the SHEET is sound: tangencies, arc sweeps, inradii, canvas containment, that a rejected variant cannot resolve, that the accent and the construction ink are spent where the rules say. Loads the sheet's `<script>` in a sandbox — no rendering, no eyeballing. |
| `reference.mjs` | Exports the sheet's RESOLVED geometry to `crates/ui/src/icons/design_reference.rs`, which `glyphs_match_the_design_sheet` diffs every sketch glyph against. |

```sh
export SKETCH_MARKS_DIR=/path/to/downloaded/sheets
node tools/design/check-marks.mjs "$SKETCH_MARKS_DIR/tool-marks.html"
node tools/design/reference.mjs "$SKETCH_MARKS_DIR"
cargo test -p ui --lib icons::        # the glyphs match it
```

`design_reference.rs` is generated and committed, so the glyph parity test runs on every build
whether or not the sheets are present locally — only regeneration needs them. A diff in that
file during review means the drawing moved, which is exactly the signal it exists to give.

Only marks listed in `reference.mjs`'s `IDS` table are exported; a mark that is drawn but not
yet authored in Rust simply is not there. Add its row when the glyph file lands.
