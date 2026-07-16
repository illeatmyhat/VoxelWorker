//! Selected-operand ghost styles (issue #78; the tint-constant voice of the ADR 0012
//! onion ghost in `onion.rs`).
//!
//! When the active selection is a node, the shell renders that node's OWN body as an
//! operation-coded x-ray ghost over the composed scene ("a Subtract cutter is invisible
//! by success"). Each ghost body draws TWICE with the same mesh (the two-pass depth
//! split, `mesh/selected_operand.rs`): the depth-pass fragments (the directly visible
//! operand surface) in the QUIET translucent tint, and the depth-FAIL fragments (the
//! operand surface occluded by scene geometry) in the LOUDER tint — so an entirely
//! internal cutter renders wholly loud, deliberately more obvious than Fusion's
//! invisible internal voids.

use super::*;

/// The operation role a selected-operand ghost body renders as (issue #78). Display's
/// OWN vocabulary — the app_core derivation maps the document's `CombineOp` onto it, so
/// the display layer renders a style without reading documents (ADR 0016 layering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandGhostStyle {
    /// The operand carves — translucent red.
    Subtract,
    /// The operand masks — translucent amber.
    Intersect,
}

/// Subtract ghost hue (a clear "this body REMOVES" red). sRGB hex, converted to the
/// linear space both cuboid + brick shaders work in (matching [`super::onion`]).
const SUBTRACT_OPERAND_COLOR_HEX: u32 = 0xd8_3c_34;

/// Intersect ghost hue (amber — "this body MASKS").
const INTERSECT_OPERAND_COLOR_HEX: u32 = 0xe0_9a_28;

/// Src alphas for the two depth-split passes: `(quiet, loud)`. The QUIET pass shades the
/// directly visible operand surface; the LOUD pass shades the occluded remainder at
/// noticeably higher opacity (the x-ray). The boolean masks (Subtract / Intersect) are
/// the ghost's whole reason to exist, so they read strongly.
const MASK_OPERAND_ALPHAS: (f32, f32) = (0.32, 0.62);

/// The style's hue + `(quiet, loud)` alphas.
fn operand_ghost_palette(style: OperandGhostStyle) -> (u32, (f32, f32)) {
    match style {
        OperandGhostStyle::Subtract => (SUBTRACT_OPERAND_COLOR_HEX, MASK_OPERAND_ALPHAS),
        OperandGhostStyle::Intersect => (INTERSECT_OPERAND_COLOR_HEX, MASK_OPERAND_ALPHAS),
    }
}

/// The QUIET (directly visible operand surface) tint as linear `[r, g, b, a]`.
pub fn operand_ghost_quiet_tint(style: OperandGhostStyle) -> [f32; 4] {
    let (hex, (quiet, _)) = operand_ghost_palette(style);
    let [r, g, b] = srgb_hex_to_linear(hex);
    [r, g, b, quiet]
}

/// The LOUD (occluded-by-scene-geometry) tint as linear `[r, g, b, a]` — the same hue as
/// the quiet pass at noticeably higher opacity, so buried voxels x-ray through.
pub fn operand_ghost_loud_tint(style: OperandGhostStyle) -> [f32; 4] {
    let (hex, (_, loud)) = operand_ghost_palette(style);
    let [r, g, b] = srgb_hex_to_linear(hex);
    [r, g, b, loud]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loud pass is MORE opaque than the quiet pass for every style — the
    /// owner-decided x-ray split (occluded = louder) — and CLEARLY so (a buried cutter
    /// must be blatant), with the hue unchanged between passes.
    #[test]
    fn loud_pass_is_more_opaque_than_quiet_for_every_style() {
        for style in [OperandGhostStyle::Subtract, OperandGhostStyle::Intersect] {
            let quiet = operand_ghost_quiet_tint(style);
            let loud = operand_ghost_loud_tint(style);
            assert!(
                loud[3] > quiet[3],
                "{style:?}: loud alpha {} must exceed quiet alpha {}",
                loud[3],
                quiet[3]
            );
            // Same hue in both passes (only the opacity splits them).
            assert_eq!(&loud[..3], &quiet[..3]);
            assert!(
                loud[3] >= quiet[3] + 0.1,
                "{style:?}: a buried mask must read NOTICEABLY louder"
            );
        }
    }

    /// Subtract is red-dominant, Intersect amber (red ≥ green > blue) — the
    /// operation-coded hues the issue names.
    #[test]
    fn mask_styles_carry_their_operation_hues() {
        let red = operand_ghost_quiet_tint(OperandGhostStyle::Subtract);
        assert!(red[0] > red[1] && red[0] > red[2]);
        let amber = operand_ghost_quiet_tint(OperandGhostStyle::Intersect);
        assert!(amber[0] >= amber[1] && amber[1] > amber[2]);
    }
}
