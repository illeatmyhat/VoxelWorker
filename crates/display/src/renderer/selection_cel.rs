//! Selection tint (ADR 0032 — viewport selection feedback, reworked 2026-07-29).
//!
//! A selected node shows as a screen-space WASH + 1px OUTLINE built from a depth map
//! of its derived body (`mesh::SelectionOutlineRenderer` + `selection_outline.wgsl`),
//! replacing the rejected cel re-draw. Still depth-coincident with the composed model
//! — feedback shows the surface the model actually shows, never an x-ray — and applies
//! in ALL view modes (owner-resolved 2026-07-26): showing what is selected is a
//! property of having selected it, not a way of displaying the document.

use super::*;

/// The Signal accent (`ui::theme::ACCENT` — "ACTIVE · SELECTED · LIVE"), the ONE accent
/// the chrome speaks; the viewport's selection mark speaks it too.
const SELECTION_CEL_COLOR_HEX: u32 = 0x9c_b4_d8;

/// The wash's src alpha. Deliberately below the operand ghost's quiet 0.32: the wash is
/// on in EVERY mode, so its resting weight must leave the material readable; the
/// emphasis lives in the near-opaque outline ring, not here.
const SELECTION_CEL_ALPHA: f32 = 0.28;

/// The selection tint as linear `[r, g, b, a]` — the ONE color the outline+wash
/// composite speaks (the outline reuses the rgb at its own alpha).
pub fn selection_cel_tint() -> [f32; 4] {
    let [r, g, b] = srgb_hex_to_linear(SELECTION_CEL_COLOR_HEX);
    [r, g, b, SELECTION_CEL_ALPHA]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cel rests QUIETER than the operand ghost's quiet pass — it is on in every
    /// mode, so it must not compete with the mode treatments it composes under.
    #[test]
    fn cel_rests_quieter_than_the_operand_ghost() {
        let cel = selection_cel_tint();
        let ghost = operand_ghost_quiet_tint(OperandGhostStyle::Subtract);
        assert!(cel[3] < ghost[3]);
    }

    /// Accent hue: cool blue-leaning (b ≥ g ≥ r) — the Signal accent, not a boolean
    /// operation hue (red / amber are the operand ghost's vocabulary).
    #[test]
    fn cel_speaks_the_accent_not_an_operation_hue() {
        let cel = selection_cel_tint();
        assert!(cel[2] >= cel[1] && cel[1] >= cel[0]);
    }
}
