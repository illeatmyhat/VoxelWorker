//! `intersect` — two dashed operands with their agreement filled.
//!
//! The overlap is FILLED rather than stroked, and that is a deliberate exception to the rail
//! set's no-fills habit. Intersect is the one combine op whose subject is a *region* — what
//! survives — and an outlined 3-unit overlap is about two pixels of hairline at 15 pt, which
//! disappears next to `subtract`'s solid operand. A filled patch survives the rail, and it
//! cannot be confused with subtract because subtract fills nothing.
//!
//! The operands are dashed for the same reason they are in `subtract`: a dashed body is an
//! operand the fold reads, not material that ends up in the result.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The two operands, overlapping generously so their agreement is the mark's subject
    // rather than a detail caught between two frames.
    g.dashed_rect((2.5, 2.5), (12.0, 12.0));
    g.dashed_rect((6.0, 6.0), (15.5, 15.5));
    // Their agreement — the only thing that survives the fold.
    g.fill(&[(6.0, 6.0), (12.0, 6.0), (12.0, 12.0), (6.0, 12.0)]);
}
