# Sketch tool suite — remaining work

The sketch-tool work shipped in five slices (`crates/parametric`, the arrangement, the solver
core, the glyph set, the wiring). What the shipped parts *are* lives in
`docs/architecture/01-document.md`; the rulings behind the solver's behavior live in
`sketch-constraint-solve.md`. This file holds only the part of the record that has no code
behind it yet.

## Open

**The expression text parser.** `parametric::expression` ships the AST, the evaluator and the
symbol table, and `parametric::units::parse` reads a single measurement literal. Nothing reads
`2*width + 3mm` — a string with an operator and a symbol in it. This belongs to the parameters
panel, which is what would give a user somewhere to type it.

**Kani harnesses for `curve_intersection` and `deepest_interior_point`.** Both are plausible
bounded targets under the substrate's machine-checking rule; the float solver beside them is not, which is why it was
never on this list. `substrate::geom2d::deepest_interior_point` and the curve-intersection
module carry tests and no proof.

**The inspector counts the wrong thing.** A circle sketch reads "Custom profile (1 points)"
because the readout counts document points and a circle carries one, its center. Cosmetic, and
the plural is wrong too.

## Closed since the record

- **The glyph set and the dimension gizmos.** All 59 sheet-resolved glyphs are transposed and
  gated by `glyphs_match_the_design_sheet`; the constraint marks and the angle / radius / span
  gizmos are in `crates/ui`.
- **Degrees of freedom on screen.** The top bar reads `DOF`, and `0` reads "fully constrained".
- **Point Quantize and its integer tier.** The parametric relation freezes the nearest
  `phase + n*pitch` lattice point per pass; the exact pass re-evaluates from the preferred answer.
  The document persists exact pitch/phase lengths and the rail authors a one-voxel point lattice.
- **The solver has callers.** Constraint entities exist and the two-pass settle runs; see
  `sketch-constraint-solve.md` for what each pass was measured to do.
