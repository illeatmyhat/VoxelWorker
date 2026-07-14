//! The sRGB ↔ linear transfer function (electro-optical transfer function).
//!
//! sRGB stores colour with a non-linear encoding so 8-bit codes are spread
//! perceptually rather than uniformly in light energy; any correct *arithmetic* on
//! colour (blending, filtering, lighting) must happen in **linear** light, so an
//! encoded value has to be decoded first. This module is the standard decode: the
//! piecewise sRGB electro-optical transfer function (EOTF) of **IEC 61966-2-1:1999**.
//!
//! The curve is a short linear toe near black joined to a `2.4`-power segment:
//!
//! ```text
//!   linear = value / 12.92                         if value <= 0.04045
//!   linear = ((value + 0.055) / 1.055) ^ 2.4       otherwise
//! ```
//!
//! where `value` is the encoded component in `[0, 1]`. The `0.04045` breakpoint,
//! the `12.92` toe slope, and the `0.055` offset with the `2.4` exponent are the
//! IEC 61966-2-1 constants; the toe keeps the curve's slope finite at the origin
//! (a pure power curve has an infinite-slope, hard-to-invert foot). Only the decode
//! (sRGB → linear) is needed here — the inverse OETF would add the matching
//! `1/2.4`-power encode if a linear→sRGB path ever wants it.

/// Decode one 8-bit sRGB component to linear light in `[0, 1]`, via the piecewise
/// IEC 61966-2-1 EOTF (linear toe below `0.04045`, `2.4`-power above). This is the
/// same decode a GPU applies when sampling an sRGB-format texture, so colours
/// computed through it mix in the same space as textured surfaces.
pub fn srgb_component_to_linear(byte: u8) -> f32 {
    let value = byte as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode a packed `0xRRGGBB` sRGB colour to a linear `[r, g, b]`, each channel
/// through [`srgb_component_to_linear`].
pub fn srgb_hex_to_linear(hex: u32) -> [f32; 3] {
    [
        srgb_component_to_linear(((hex >> 16) & 0xff) as u8),
        srgb_component_to_linear(((hex >> 8) & 0xff) as u8),
        srgb_component_to_linear((hex & 0xff) as u8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The endpoints of the encoding map to the endpoints of linear light.
    #[test]
    fn black_and_white_are_exact_endpoints() {
        assert_eq!(srgb_component_to_linear(0), 0.0);
        assert_eq!(srgb_component_to_linear(255), 1.0);
    }

    /// Around the `0.04045` breakpoint the two pieces agree (the curve is
    /// continuous there): byte `10` (`≈0.0392`) is on the linear toe, byte `11`
    /// (`≈0.0431`) is on the power segment, and each equals its own branch.
    #[test]
    fn piecewise_branches_match_their_formulas() {
        // Byte 10 → 0.039216 ≤ 0.04045, linear toe: value / 12.92.
        let toe = 10u8;
        let toe_value = toe as f32 / 255.0;
        assert_eq!(srgb_component_to_linear(toe), toe_value / 12.92);

        // Byte 11 → 0.043137 > 0.04045, power segment.
        let power = 11u8;
        let power_value = power as f32 / 255.0;
        assert_eq!(
            srgb_component_to_linear(power),
            ((power_value + 0.055) / 1.055).powf(2.4)
        );
    }

    /// A known mid value: sRGB `188` (`≈0.737` encoded) decodes to ≈`0.5` linear —
    /// linear mid-grey encodes to just under byte 188, so the exact decode is
    /// ≈`0.503`, checked against a curve-shape tolerance rather than a bit value.
    #[test]
    fn mid_grey_decodes_near_half() {
        let mid = srgb_component_to_linear(188);
        assert!((mid - 0.5).abs() < 5e-3, "expected ≈0.5, got {mid}");
    }

    /// The hex splitter routes each byte to the matching channel and decodes it.
    #[test]
    fn hex_splits_channels_and_decodes() {
        let rgb = srgb_hex_to_linear(0x00_80_ff);
        assert_eq!(rgb[0], srgb_component_to_linear(0x00));
        assert_eq!(rgb[1], srgb_component_to_linear(0x80));
        assert_eq!(rgb[2], srgb_component_to_linear(0xff));
        assert_eq!(rgb[0], 0.0);
        assert_eq!(rgb[2], 1.0);
    }
}
